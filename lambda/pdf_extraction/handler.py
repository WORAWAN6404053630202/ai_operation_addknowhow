# lambda/pdf_extraction/handler.py
"""AWS Lambda handler v2: PDF extraction ONLY (Typhoon OCR + Claude vision), no
validation/queueing/Sheet logic here — per the original architecture diagram,
Lambda's job stops at "extract and save the result to S3"; everything downstream
(validation, field-drafting, review queue) runs on the EC2 app, reusing the
already-tested code in code/service/pdf_extraction_validation.py etc., not
duplicated here.

v1 of this file used AWS Textract — ruled out, no Thai support at all (see
project history: avg_confidence=60.58, fully garbled output on a real test doc).

STANDALONE + zip-deployable (not a container image) — do not import from code/
(the FastAPI app). Renders PDF pages to PNG via PyMuPDF (pure Python wheel, NO
system Poppler dependency), then hands the PNG directly to typhoon_ocr — its
own source confirms passing a non-.pdf path skips its internal Poppler-based
render entirely (see typhoon_ocr.ocr_utils.prepare_ocr_messages: the `is_image`
branch uses PIL.Image.open() directly). This is what keeps this Lambda a plain
zip upload instead of needing a container image with Poppler installed at the
OS level.

Requires these Lambda environment variables (set in the console, NOT read from
this repo's conf.py/env.properties — this function has no access to those):
  TYPHOON_OCR_API_KEY, OPENROUTER_API_KEY, SQS_QUEUE_URL
Optional:
  OPENROUTER_BASE_URL     (default: https://openrouter.ai/api/v1)
  OPENROUTER_MODEL_VISION (default: google/gemini-2.5-flash-lite) — per-page image OCR
  OUTPUT_S3_PREFIX        (default: restbiz/extracted/)
  HANDOFF_S3_PREFIX       (default: restbiz/pending_large/)
  MAX_PAGES_FOR_LAMBDA    (default: 35)

SQS DELIVERY (2026-08-24): originally relied on S3's own bucket notification
configuration (prefix/suffix rule -> SQS) to tell the EC2 consumer a result
was ready. Live-tested and reproduced twice: Lambda wrote the output to S3
successfully both times, but the S3->SQS notification never fired — no
message ever arrived (queue confirmed at 0 messages via GetQueueAttributes
each time), despite the notification config and the queue's access policy
both checking out correct via the console and API. Rather than keep
debugging an opaque AWS-side gap, this Lambda now sends the SQS message
itself directly (_notify_sqs below) right after each S3 write — the same
Records[]-shaped body a real S3 event notification would have sent, so
sqs_consumer.py's process_sqs_message() needed zero changes. This removes
the dependency on that notification path entirely; the bucket's
extracted/ and pending_large/ -> SQS notification rules are no longer
needed and were removed (the restbiz/ -> Lambda trigger notification is
unrelated and stays)."""

"""LARGE-DOCUMENT HANDLING (was "KNOWN LIMITATION: not solved here" before
2026-08-22): a real test (66-page PDF) confirmed Lambda's 15-minute hard cap
gets hit around ~55-60 pages even with 4-way page concurrency, and that cap
cannot be raised — it's a platform limit, not a config knob. Rather than
building the fan-out/Step-Functions machinery to split one document across
many Lambda invocations, large documents instead hand off to the EC2 instance
(already running 24/7 for the SQS consumer, so this adds zero marginal
infrastructure cost, and a long-lived process has no 15-minute ceiling at
all): this Lambda does a CHEAP screen — OCR + relevance-check just the first
2 pages — and either (a) writes a "not worth the spend" skip marker if that
partial content is clearly out of scope, saving the cost of the remaining
~30+ pages entirely, or (b) writes a small handoff marker (not the extraction
itself) to HANDOFF_S3_PREFIX for service/pdf_large_extraction.py on the EC2
side to pick up and fully process with no time pressure. Small/typical
documents (the common case) are completely unaffected — same one Lambda
invocation, same output shape, same downstream SQS path as before."""

import base64
import json
import logging
import os
import random
import re
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import boto3
import pymupdf
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from typhoon_ocr import ocr_document

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
sqs = boto3.client("sqs")

TYPHOON_OCR_API_KEY = os.environ.get("TYPHOON_OCR_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Model for the per-page vision OCR call (_run_vision_verify below).
# DECISION 2026-08/09: backtested against pdf_large_extraction.py's same
# check on the EC2 side — Gemini 2.5 Flash Lite ($0.10/$0.40 per M vs
# Sonnet's $3/$15, ~30x cheaper) matched or beat Sonnet's agreement rate
# with Typhoon on every page tested.
OPENROUTER_MODEL_VISION = os.environ.get("OPENROUTER_MODEL_VISION", "google/gemini-2.5-flash-lite")
OUTPUT_S3_PREFIX = os.environ.get("OUTPUT_S3_PREFIX", "restbiz/extracted/")
HANDOFF_S3_PREFIX = os.environ.get("HANDOFF_S3_PREFIX", "restbiz/pending_large/")
# Same default as conf.PDF_STATUS_S3_PREFIX on the EC2 side (service/
# pdf_status_tracker.py) — both sides write the identical JSON shape to the
# same prefix so router/admin.py needs no knowledge of which side wrote it.
STATUS_S3_PREFIX = os.environ.get("STATUS_S3_PREFIX", "restbiz/status/")
MAX_PAGES_FOR_LAMBDA = int(os.environ.get("MAX_PAGES_FOR_LAMBDA", "35"))
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "")

_MAX_WORKERS = 4  # concurrent pages — keep modest, each page = 2 API calls already

# Duplicated from code/utils/rate_limiter.py (NOT imported — this Lambda has
# no access to code/, see module docstring) — see that module for the full
# reasoning: Typhoon's own docs (checked 2026-09) say typhoon-ocr is
# rate-limited to 2 requests/sec, 20 requests/min, and _MAX_WORKERS=4
# page-level concurrency above can otherwise burst past that on a single
# document alone. Per-invocation only — does NOT protect against multiple
# concurrent Lambda invocations processing different documents at once.
class _MinIntervalRateLimiter:
    def __init__(self, min_interval_seconds: float):
        self._min_interval = min_interval_seconds
        self._lock = threading.Lock()
        self._last_call_at: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_call_at is not None:
                remaining = self._min_interval - (now - self._last_call_at)
                if remaining > 0:
                    time.sleep(remaining)
                    now = time.monotonic()
            self._last_call_at = now


# 1.8 (not 2.0) leaves a small margin for clock/network jitter — same value
# as pdf_large_extraction.py's EC2-side copy.
_typhoon_rate_limiter = _MinIntervalRateLimiter(min_interval_seconds=1 / 1.8)

_TYPHOON_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_TYPHOON_MAX_RETRIES = 4  # bounded, never infinite — same lesson as the SQS retry-loop incident this pipeline already had once


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    header = getattr(response, "headers", {}).get("Retry-After")
    if header is None:
        return None
    try:
        return float(header)
    except (TypeError, ValueError):
        return None

_CLAUDE_PROMPT = (
    "ถอดข้อความภาษาไทย (และอังกฤษถ้ามี) ทั้งหมดในภาพเอกสารนี้ให้ครบถ้วนและถูกต้องที่สุด "
    "รักษาโครงสร้างเดิมไว้เป็น markdown (หัวข้อ, ตาราง, รายการเลข/bullet) "
    "กฎสำคัญ: ห้ามเดาหรือแต่งเติมข้อความที่อ่านไม่ชัดเจนเด็ดขาด "
    "ถ้าตัวอักษร/ตัวเลขจุดไหนอ่านไม่ออกจริงๆ ให้ใส่ [อ่านไม่ชัด] แทนตรงจุดนั้น "
    "แทนที่จะเดาคำที่ดูสมเหตุสมผล"
)

# Duplicated from code/utils/llm_cost_logging.py (NOT imported — this Lambda
# has no access to code/, see module docstring). Kept intentionally minimal
# (this Lambda has exactly one LLM call site, _run_vision_verify below) —
# see the EC2-side module for the full pricing-source explanation and the
# claude-haiku-4-5 stale-value note. Re-verify against openrouter.ai before
# trusting this table if it's ever copied elsewhere.
_MODEL_PRICING = {
    "anthropic/claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "anthropic/claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "qwen/qwen3.7-flash": {"input": 0.03, "output": 0.13},
    "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}


class _CostAccumulator:
    """Duplicated from code/utils/llm_cost_logging.py's CostAccumulator (NOT
    imported — this Lambda has no access to code/) — thread-safe running
    total for one document's extraction phase (all page's vision-verify
    calls), included in the JSON this Lambda writes to S3 so sqs_consumer.py
    on the EC2 side can add it to its own build-phase total for one
    document-level cost shown in the admin UI. See that module for why."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_cost_usd = 0.0

    def add(self, cost_usd: float) -> None:
        with self._lock:
            self.total_cost_usd += cost_usd


def _log_llm_cost(label: str, model: str, response, elapsed_seconds: float | None = None, accumulator: "_CostAccumulator | None" = None) -> float:
    """Never raises — a logging failure must not break real OCR work.
    elapsed_seconds added 2026-09 alongside code/utils/llm_cost_logging.py's
    same addition — see that module for why (Typhoon's undocumented-until-
    checked 2 RPS / 20 RPM rate limit; a slow/retried call is the first
    visible symptom, and this pipeline had no timing data anywhere before).
    Returns the computed cost (0.0 on any failure/unknown-pricing path) and
    adds it to `accumulator` if given — added alongside _CostAccumulator."""
    elapsed_str = "" if elapsed_seconds is None else f" elapsed_s={elapsed_seconds:.2f}"
    try:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    except Exception as e:
        logger.warning(f"[CostLog] {label}: could not read token usage from the response, logging nothing{elapsed_str}: {e}")
        if accumulator is not None:
            accumulator.add(0.0)
        return 0.0

    total_tokens = prompt_tokens + completion_tokens
    pricing = _MODEL_PRICING.get(model)
    if pricing is None:
        logger.warning(
            f"[CostLog] {label} | model={model} prompt_tokens={prompt_tokens} "
            f"completion_tokens={completion_tokens} total_tokens={total_tokens} "
            f"cost_usd=unknown (no pricing entry for this model in _MODEL_PRICING){elapsed_str}"
        )
        if accumulator is not None:
            accumulator.add(0.0)
        return 0.0

    cost_usd = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000
    logger.info(
        f"[CostLog] {label} | model={model} prompt_tokens={prompt_tokens} "
        f"completion_tokens={completion_tokens} total_tokens={total_tokens} "
        f"cost_usd={cost_usd:.6f}{elapsed_str}"
    )
    if accumulator is not None:
        accumulator.add(cost_usd)
    return cost_usd


def _log_call_duration(label: str, elapsed_seconds: float) -> None:
    """For external calls with no token/cost data (Typhoon's ocr_document()
    returns a plain string, not a response object with .usage) — mirrors
    code/utils/llm_cost_logging.py's log_call_duration."""
    logger.info(f"[CostLog] {label} | elapsed_s={elapsed_seconds:.2f} (no token/cost data for this call)")


def _write_status(bucket, filename, *, stage, pages_done=None, pages_total=None, attempt=1, error=None) -> None:
    """Progress/error visibility for the ≤MAX_PAGES_FOR_LAMBDA path — the
    same in-flight-status mechanism as the EC2 large-doc path (see
    conf.PDF_STATUS_S3_PREFIX / service/pdf_status_tracker.py on the EC2
    side), duplicated inline here rather than imported since this Lambda is
    a standalone zip with no access to code/ (see module docstring). Never
    raises — a status-write failure must not break real OCR work."""
    payload = {
        "filename": filename, "stage": stage, "pages_done": pages_done,
        "pages_total": pages_total, "attempt": attempt,
        "error": error[:500] if error else None, "updated_at": time.time(),
    }
    try:
        s3.put_object(
            Bucket=bucket, Key=f"{STATUS_S3_PREFIX}{filename}.json",
            Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as e:
        logger.warning(f"{filename}: failed to write status: {e}")


def _next_attempt_number(bucket, filename) -> int:
    """1 for a fresh document, or (prior attempt + 1) if a status object
    from an earlier failed invocation is still there."""
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"{STATUS_S3_PREFIX}{filename}.json")
        prior = json.loads(obj["Body"].read().decode("utf-8"))
        return prior.get("attempt", 0) + 1
    except Exception:
        return 1


def _notify_sqs(bucket: str, key: str) -> None:
    """Sends the same Records[]-shaped body a real S3 event notification would
    have sent — sqs_consumer.py's process_sqs_message() parses this exact
    shape and doesn't know or care whether S3 or this function sent it. Logs
    and re-raises on failure rather than swallowing it: if this fails, the
    S3 object this Lambda just wrote would otherwise sit there forever with
    nothing to ever pick it up, which is worse than a visible Lambda error
    (visible in CloudWatch, and S3 will retry the whole invocation)."""
    if not SQS_QUEUE_URL:
        raise RuntimeError("SQS_QUEUE_URL is not set — cannot notify the consumer that s3://{}/{} is ready".format(bucket, key))
    message_body = json.dumps({"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]}, ensure_ascii=False)
    sqs.send_message(QueueUrl=SQS_QUEUE_URL, MessageBody=message_body)
    logger.info(f"Notified SQS for s3://{bucket}/{key}")


def _render_page_to_png_bytes(pdf_bytes: bytes, page_num: int, dpi: int = 200) -> bytes:
    """page_num is 1-indexed to match the rest of this codebase's convention."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def _run_typhoon(png_bytes: bytes, page_num: int) -> str:
    """Throttled (_typhoon_rate_limiter) + retried with backoff on transient
    failures — mirrors pdf_large_extraction.py's EC2-side copy, added
    2026-09. Never retries forever: gives up after _TYPHOON_MAX_RETRIES and
    re-raises, so Lambda's own async-invoke retry policy still applies as
    the final safety net."""
    # ocr_document takes a file path, not bytes — /tmp is Lambda's writable scratch space.
    tmp_path = f"/tmp/page_{page_num}.png"
    with open(tmp_path, "wb") as f:
        f.write(png_bytes)
    try:
        attempt = 0
        while True:
            attempt += 1
            _typhoon_rate_limiter.wait()
            _call_start = time.time()
            try:
                result = ocr_document(
                    tmp_path,
                    model="typhoon-ocr",
                    figure_language="Thai",
                    task_type="v1.5",
                    api_key=TYPHOON_OCR_API_KEY,
                )
                _log_call_duration(f"LambdaExtraction/Typhoon[page {page_num}]", time.time() - _call_start)
                return result
            except _TYPHOON_RETRYABLE_ERRORS as e:
                _log_call_duration(f"LambdaExtraction/Typhoon[page {page_num}] attempt {attempt} FAILED", time.time() - _call_start)
                if attempt > _TYPHOON_MAX_RETRIES:
                    logger.error(f"Page {page_num}: Typhoon OCR failed after {attempt} attempts, giving up: {e}")
                    raise
                delay = _retry_after_seconds(e)
                if delay is None:
                    delay = min(2 ** (attempt - 1), 30) + random.uniform(0, 1)
                logger.warning(f"Page {page_num}: Typhoon OCR attempt {attempt} failed ({type(e).__name__}), retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
    finally:
        os.remove(tmp_path)


def _run_vision_verify(png_bytes: bytes, cost_accumulator: "_CostAccumulator | None" = None) -> str:
    """Second-opinion OCR pass on top of Typhoon's — model is
    OPENROUTER_MODEL_VISION, not necessarily Claude despite the old function
    name this replaces (renamed 2026-09 to match pdf_large_extraction.py's
    same rename on the EC2 side)."""
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    _call_start = time.time()
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL_VISION,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _CLAUDE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=4000,
    )
    _log_llm_cost("VisionVerify", OPENROUTER_MODEL_VISION, resp, time.time() - _call_start, accumulator=cost_accumulator)
    return resp.choices[0].message.content or "(no text returned)"


# Duplicated from service/pdf_extraction_validation.py's _extract_salient_tokens
# (NOT imported — this Lambda has no access to code/, see module docstring)
# — kept byte-for-byte identical on purpose, since the H2 backtest result
# (100% recall skipping vision-verify on pages with none of these 4 token
# types, only 32.7% of pages needed it) that justified this optimization on
# the EC2 side was measured against exactly this logic; drifting the two
# copies apart would invalidate that result for this side.
_FORM_CODE_RE = re.compile(r"[ก-ฮ]{1,3}\.?\s?\d{2}-\d{2}")
_BAHT_AMOUNT_RE = re.compile(r"[\d,]+(?:\.\d{1,2})?\s*บาท")
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
_LICENSE_NUMBER_RE = re.compile(r"\b\d{2,4}-?[A-Zก-ฮ]?-?\d{6,8}\b")


def _clean_number(raw: str) -> float:
    return float(raw.replace(",", "").replace("บาท", "").strip())


def _normalize_date(raw: str) -> str:
    day, month, year = raw.split("/")
    return f"{int(day):02d}/{int(month):02d}/{year}"


def _extract_salient_tokens(markdown: str) -> dict[str, set]:
    return {
        "baht_amounts": {_BAHT_AMOUNT_RE.sub(lambda m: str(_clean_number(m.group())), a) for a in _BAHT_AMOUNT_RE.findall(markdown)},
        "dates": {_normalize_date(d) for d in _DATE_RE.findall(markdown)},
        "form_codes": {m.group().replace(" ", "") for m in _FORM_CODE_RE.finditer(markdown)},
        "license_numbers": set(_LICENSE_NUMBER_RE.findall(markdown)),
    }


_MIN_PAGE_TEXT_LAYER_CHARS = 40  # below this, treat as scanned/image-only — not enough embedded text to trust

# Character-validity check for a PDF's embedded text layer — the established
# root cause of a corrupted text layer (per PyMuPDF/Docling issue trackers
# and mojibake-detection literature, e.g. the `ftfy` library's heuristic
# docs) is a CID-keyed font with no ToUnicode map, or a straight wrong-
# encoding decode. Both failure modes overwhelmingly produce characters
# OUTSIDE what a legitimate Thai document should contain: Unicode Private
# Use Area (U+E000–U+F8FF, the single most-cited signature of the missing-
# ToUnicode-map case) or the U+FFFD replacement character (the signature of
# a wrong-encoding decode). Scoped to Thai + ASCII rather than reusing
# `ftfy`-style pattern lists on purpose — those are tuned for Western
# encoding confusion (UTF-8 decoded as Windows-1252 etc.), not the CID-font
# failure mode that's the more common cause for Thai-language PDFs
# specifically, and this needs no new dependency for what's meant to stay a
# minimal, zip-deployable Lambda.
_EXPECTED_TEXT_RANGES = (
    (0x0009, 0x000A),  # tab, newline
    (0x0020, 0x007E),  # ASCII printable: Latin letters, digits, punctuation
    (0x0E00, 0x0E7F),  # Thai
    (0x2018, 0x201F),  # curly quotes
    (0x2013, 0x2014),  # en/em dash
)
_MAX_BAD_CHAR_RATIO = 0.05  # >5% characters outside the expected ranges = likely corrupted


def _text_looks_valid(text: str) -> bool:
    """See _EXPECTED_TEXT_RANGES above for the reasoning. Ignores whitespace
    (not a signal either way) and never raises."""
    stripped = "".join(text.split())
    if not stripped:
        return False
    bad = sum(1 for ch in stripped if not any(lo <= ord(ch) <= hi for lo, hi in _EXPECTED_TEXT_RANGES))
    return (bad / len(stripped)) <= _MAX_BAD_CHAR_RATIO


def _extract_page_native_text(pdf_bytes: bytes, page_num: int) -> str | None:
    """Zero-cost alternative to OCR: many digital-native (non-scanned) PDFs
    already carry an extractable text layer via pymupdf's page.get_text() —
    added 2026-09, this codebase had never used it anywhere in the actual
    extraction path before. Returns None (caller falls back to OCR) for a
    scanned/image-only page with no meaningful embedded text, a text layer
    that fails the character-validity check above (likely a corrupted/
    mismapped font), or if extraction itself errors — never raises."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            text = doc[page_num - 1].get_text()
        finally:
            doc.close()
    except Exception as e:
        logger.warning(f"Page {page_num}: native text-layer extraction failed, falling back to OCR: {e}")
        return None
    if len(text.strip()) < _MIN_PAGE_TEXT_LAYER_CHARS:
        return None
    if not _text_looks_valid(text):
        logger.info(f"Page {page_num}: native text layer present but failed the character-validity check (likely a corrupted/mismapped font) — falling back to OCR")
        return None
    return text


def _extract_one_page(pdf_bytes: bytes, page_num: int, cost_accumulator: "_CostAccumulator | None" = None) -> dict:
    native_text = _extract_page_native_text(pdf_bytes, page_num)
    if native_text is not None:
        logger.info(f"Page {page_num}: used native PDF text layer ({len(native_text)} chars) — skipped OCR entirely, $0 for this page")
        # Same value in both fields: this IS the source text, not a second
        # extractor's independent read of it — validate_extraction()'s
        # compare_extractions() finding "no disagreement" here is correct
        # (nothing to disagree about), not a fake cross-check.
        return {"page_num": page_num, "typhoon_markdown": native_text, "claude_markdown": native_text}

    try:
        png_bytes = _render_page_to_png_bytes(pdf_bytes, page_num)
        typhoon_markdown = _run_typhoon(png_bytes, page_num)

        # Same H2 optimization as pdf_large_extraction.py on the EC2 side
        # (2026-08/09 backtest, 49 pages / 13 documents: 100% recall skipping
        # vision-verify on pages with none of these 4 salient-token types,
        # while only calling it on 32.7% of pages) — compare_extractions() in
        # pdf_extraction_validation.py can only ever flag a disagreement in one
        # of these categories, so a page with none of them can't produce a flag
        # regardless of what the vision model would say.
        salient = _extract_salient_tokens(typhoon_markdown)
        if any(salient[k] for k in salient):
            claude_markdown = _run_vision_verify(png_bytes, cost_accumulator=cost_accumulator)
        else:
            claude_markdown = ""
            logger.info(f"Page {page_num}: no salient tokens in Typhoon output, skipping vision-verify")

        logger.info(f"Page {page_num}: extracted via OCR (Typhoon {len(typhoon_markdown)} chars, Claude {len(claude_markdown)} chars)")
        return {"page_num": page_num, "typhoon_markdown": typhoon_markdown, "claude_markdown": claude_markdown}
    except Exception as e:
        # Added 2026-09 (mirrors pdf_large_extraction.py's EC2-side copy):
        # without this, one page's unrecoverable failure propagates out of
        # the ThreadPoolExecutor future in lambda_handler() and kills the
        # ENTIRE document — discarding every other page already successfully
        # (and expensively) extracted in the same invocation. A placeholder
        # record for just this one page costs nothing and keeps the rest of
        # the document's work intact; a reviewer sees the error text in the
        # review queue and knows exactly which page to re-check manually.
        logger.error(f"Page {page_num}: extraction failed, saving placeholder instead of losing the whole document: {e}")
        error_markdown = f"[สกัดข้อมูลหน้านี้ไม่สำเร็จ — เกิดข้อผิดพลาดระหว่างประมวลผล: {e}]"
        return {"page_num": page_num, "typhoon_markdown": error_markdown, "claude_markdown": ""}


def lambda_handler(event, context):
    if not TYPHOON_OCR_API_KEY or not OPENROUTER_API_KEY:
        raise RuntimeError("TYPHOON_OCR_API_KEY and OPENROUTER_API_KEY must be set as Lambda environment variables.")

    results = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        filename = key.rsplit("/", 1)[-1]

        logger.info(f"Processing s3://{bucket}/{key}")
        # Captured here, not at Lambda cold-start — this is "extraction
        # actually begins" for THIS record, which is what
        # ReviewItem.processing_duration_seconds on the EC2 side measures
        # against extraction_completed_at for a real wall-clock number.
        record_started_at = time.time()
        obj = s3.get_object(Bucket=bucket, Key=key)
        pdf_bytes = obj["Body"].read()

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(doc)
        doc.close()
        logger.info(f"{filename}: {num_pages} page(s)")

        if num_pages > MAX_PAGES_FOR_LAMBDA:
            # Removed 2026-09: used to run a cheap relevance pre-screen here
            # (OCR a few sample pages, ask an LLM "worth processing?") before
            # committing to full OCR on EC2 — dropped along with
            # check_relevance() elsewhere in this pipeline (see
            # sqs_consumer.py) for the same reason: every document entering
            # this pipeline is already curated by whoever uploaded it, so
            # there's no real "is this in scope" decision left to make, and
            # the screen's only real failure mode (wrongly skipping a
            # genuinely relevant large document, losing it entirely with no
            # OCR ever run) was strictly worse than the cost it was trying
            # to save — cost exposure it was guarding against is now mostly
            # covered anyway by _extract_page_native_text()'s free text-layer
            # skip and _extract_one_page()'s salient-token check on the EC2
            # side, both of which apply regardless of topic relevance.
            handoff = {
                "filename": filename,
                "s3_raw_pdf_path": key,
                "bucket": bucket,
                "page_count": num_pages,
            }
            handoff_key = f"{HANDOFF_S3_PREFIX}{filename}.json"
            s3.put_object(
                Bucket=bucket, Key=handoff_key,
                Body=json.dumps(handoff, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json",
            )
            logger.info(f"{filename}: {num_pages} pages exceeds MAX_PAGES_FOR_LAMBDA={MAX_PAGES_FOR_LAMBDA} — handed off to EC2 via {handoff_key}, not processed in Lambda")
            _notify_sqs(bucket, handoff_key)
            results.append({"input_key": key, "handoff_key": handoff_key, "page_count": num_pages, "handed_off": True})
            continue

        # Progress visibility (2026-08-27): this is the common case (most
        # uploads are well under MAX_PAGES_FOR_LAMBDA) and it shares
        # OPENROUTER_API_KEY with _run_vision_verify() below, so it can fail the
        # exact same silent way an out-of-credits account broke the EC2
        # large-doc path — previously invisible until the item either landed
        # in the review queue or someone went digging through CloudWatch.
        attempt = _next_attempt_number(bucket, filename)
        _write_status(bucket, filename, stage="processing", pages_done=0, pages_total=num_pages, attempt=attempt)
        _PROGRESS_WRITE_EVERY_N_PAGES = 5  # small enough to matter for a ≤35-page doc

        extraction_cost_accumulator = _CostAccumulator()
        try:
            pages = [None] * num_pages
            pages_done = 0
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
                futures = {pool.submit(_extract_one_page, pdf_bytes, p, extraction_cost_accumulator): p for p in range(1, num_pages + 1)}
                for future in as_completed(futures):
                    page_num = futures[future]
                    pages[page_num - 1] = future.result()
                    pages_done += 1
                    if pages_done == num_pages or pages_done % _PROGRESS_WRITE_EVERY_N_PAGES == 0:
                        _write_status(bucket, filename, stage="processing", pages_done=pages_done, pages_total=num_pages, attempt=attempt)

            # extraction_cost_usd/extraction_started_at: read by sqs_consumer.py
            # on the EC2 side to seed its own build-phase CostAccumulator and
            # ReviewItem.extraction_started_at — see model/pdf_review_item.py.
            output = {
                "filename": filename, "s3_raw_pdf_path": key, "pages": pages,
                "extraction_cost_usd": extraction_cost_accumulator.total_cost_usd,
                "extraction_started_at": record_started_at,
            }
            output_key = f"{OUTPUT_S3_PREFIX}{filename}.json"
            s3.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=json.dumps(output, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json",
            )
            logger.info(f"Wrote extraction result to s3://{bucket}/{output_key}")
            _notify_sqs(bucket, output_key)
        except Exception as e:
            # Status only — Lambda's own async-invoke retry policy is
            # unchanged by this, this just makes a failed attempt visible
            # instead of only existing in CloudWatch logs.
            _write_status(bucket, filename, stage="failed", pages_total=num_pages, attempt=attempt, error=str(e))
            raise
        results.append({"input_key": key, "output_key": output_key, "page_count": num_pages})

    return {"statusCode": 200, "body": json.dumps(results, ensure_ascii=False)}

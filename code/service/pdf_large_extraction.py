# code/service/pdf_large_extraction.py
"""EC2-side full-document OCR extraction for PDFs too large for Lambda's hard
15-minute execution cap (feature/pdf-ingestion) — the counterpart to
lambda/pdf_extraction/handler.py's MAX_PAGES_FOR_LAMBDA handoff.

Real test evidence (2026-08): a 66-page PDF hit Lambda's 15-minute ceiling
around page ~55-60 even with 4-way page concurrency — and that ceiling is a
hard AWS platform limit, not something more memory/timeout config can raise.
Rather than building fan-out/Step-Functions machinery to split one document
across many Lambda invocations, oversized documents hand off to this module
instead: it runs as part of the EC2 consumer process, which is already
running 24/7 for other reasons, so a long extraction here has no hard time
limit and adds zero marginal infrastructure cost or complexity beyond what's
already deployed.

Deliberately duplicates (does not import) lambda/pdf_extraction/handler.py's
OCR functions — that file is a standalone zip-deployable Lambda with no
access to code/ at all, so sharing a module isn't an option; keeping the
extraction logic here as a close, clearly-labeled mirror is the pragmatic
choice over maintaining a shared package two different deployment targets
would both need to vendor anyway."""

from __future__ import annotations

import base64
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import boto3
import pymupdf
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from typhoon_ocr import ocr_document

import conf
from service.pdf_extraction_validation import _extract_salient_tokens
from utils.llm_cost_logging import log_call_duration, log_llm_cost
from utils.logger import get_logger
from utils.pdf_text_layer import extract_page_native_text
from utils.rate_limiter import MinIntervalRateLimiter

logger = get_logger(__name__)

_MAX_WORKERS = 4  # same concurrency cap as the Lambda extractor, for the same reason

# Typhoon's own docs (https://docs.opentyphoon.ai/en/rate-limits/, checked
# 2026-09): typhoon-ocr is rate-limited to 2 requests/sec, 20 requests/min —
# _MAX_WORKERS=4 page-level concurrency above can otherwise burst past that
# on a single document alone. 1.8 (not 2.0) leaves a small margin for clock/
# network jitter between our sleep and Typhoon's own window boundary.
_typhoon_rate_limiter = MinIntervalRateLimiter(min_interval_seconds=1 / 1.8)

# Retryable = transient (rate limit, connection blip, Typhoon-side 5xx) —
# anything else (auth error, bad request, etc.) propagates immediately since
# retrying it would just fail the same way every time.
_TYPHOON_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_TYPHOON_MAX_RETRIES = 4  # additional attempts after the first — bounded, never infinite (same lesson as the SQS retry-loop incident, see sqs_consumer.py's _MAX_HANDOFF_ATTEMPTS)


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """Reads the Retry-After header off an OpenAI SDK HTTP error, if present
    — Typhoon's API is OpenAI-SDK-compatible (see typhoon_ocr's own source),
    so a 429 response may carry a server-specified wait time more accurate
    than a guessed backoff. Falls back to None (caller uses exponential
    backoff instead) for any error shape that doesn't have it."""
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


def _render_page_to_png_bytes(pdf_bytes: bytes, page_num: int, dpi: int = 200) -> bytes:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def _run_typhoon(png_bytes: bytes, page_num: int) -> str:
    """Throttled to stay under Typhoon's 2 RPS ceiling (_typhoon_rate_limiter
    above) and retried with backoff on transient failures — added 2026-09
    after confirming this pipeline's own page concurrency could exceed that
    ceiling with zero retry protection before this. Never retries forever:
    gives up after _TYPHOON_MAX_RETRIES and re-raises, so the existing
    higher-level bounded-retry protection (sqs_consumer.py's
    _MAX_HANDOFF_ATTEMPTS on this EC2 path) still applies as the final
    safety net."""
    tmp_path = f"{tempfile.gettempdir()}/restbiz_large_page_{page_num}.png"
    with open(tmp_path, "wb") as f:
        f.write(png_bytes)
    try:
        attempt = 0
        while True:
            attempt += 1
            _typhoon_rate_limiter.wait()
            _call_start = time.monotonic()
            try:
                result = ocr_document(
                    tmp_path,
                    model="typhoon-ocr",
                    figure_language="Thai",
                    task_type="v1.5",
                    api_key=conf.TYPHOON_OCR_API_KEY,
                )
                log_call_duration(logger, f"LargeExtraction/Typhoon[page {page_num}]", time.monotonic() - _call_start)
                return result
            except _TYPHOON_RETRYABLE_ERRORS as e:
                log_call_duration(logger, f"LargeExtraction/Typhoon[page {page_num}] attempt {attempt} FAILED", time.monotonic() - _call_start)
                if attempt > _TYPHOON_MAX_RETRIES:
                    logger.error(f"[LargeExtraction] Page {page_num}: Typhoon OCR failed after {attempt} attempts, giving up: {e}")
                    raise
                delay = _retry_after_seconds(e)
                if delay is None:
                    delay = min(2 ** (attempt - 1), 30) + random.uniform(0, 1)  # exponential backoff + jitter, capped at ~30s
                logger.warning(f"[LargeExtraction] Page {page_num}: Typhoon OCR attempt {attempt} failed ({type(e).__name__}), retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
    finally:
        os.remove(tmp_path)


def _run_vision_verify(png_bytes: bytes) -> str:
    """Second-opinion OCR pass on top of Typhoon's — model is
    conf.OPENROUTER_MODEL_PDF_VISION (see conf.py for why this is its own
    constant, not conf.OPENROUTER_MODEL), not necessarily Claude despite the
    old function name this replaces."""
    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    _call_start = time.monotonic()
    resp = client.chat.completions.create(
        model=conf.OPENROUTER_MODEL_PDF_VISION,
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
    log_llm_cost(logger, "LargeExtraction/VisionVerify", conf.OPENROUTER_MODEL_PDF_VISION, resp, time.monotonic() - _call_start)
    return resp.choices[0].message.content or "(no text returned)"


def _extract_one_page(pdf_bytes: bytes, page_num: int) -> dict:
    # Added 2026-09 (same optimization shipped to lambda/pdf_extraction/
    # handler.py's normal path first) — many digital-native pages need no
    # OCR at all. Matters even more here than on the Lambda side: this is
    # the large-document path, where skipping both OCR calls on a page that
    # doesn't need them saves the most per page, and every document over
    # MAX_PAGES_FOR_LAMBDA now lands here unconditionally (the Lambda
    # pre-screen that used to filter some of them out was removed 2026-09).
    native_text = extract_page_native_text(pdf_bytes, page_num)
    if native_text is not None:
        logger.info(f"[LargeExtraction] Page {page_num}: used native PDF text layer ({len(native_text)} chars) — skipped OCR entirely, $0 for this page")
        return {"page_num": page_num, "typhoon_markdown": native_text, "claude_markdown": native_text}

    try:
        png_bytes = _render_page_to_png_bytes(pdf_bytes, page_num)
        typhoon_markdown = _run_typhoon(png_bytes, page_num)

        # Added 2026-08-31: backtested against 13 already-processed documents
        # (49 pages) — compare_extractions() in pdf_extraction_validation.py
        # only ever flags a disagreement in one of 4 salient-token categories
        # (baht amounts, dates, form codes, license numbers), so a page where
        # Typhoon found none of those can't produce a flag regardless of what
        # Claude would have said. Skipping Claude on those pages caught 14/14
        # (100%) of real disagreements in the backtest while only calling Claude
        # on 32.7% of pages — roughly a 67% cut in the most expensive step of
        # this pipeline (Sonnet Vision OCR, up to 4000 output tokens/page) with
        # no observed recall loss. Re-run the backtest periodically as more
        # documents accumulate to confirm this still holds.
        salient = _extract_salient_tokens(typhoon_markdown)
        if any(salient[k] for k in salient):
            claude_markdown = _run_vision_verify(png_bytes)
        else:
            claude_markdown = ""
            logger.info(f"[LargeExtraction] Page {page_num}: no salient tokens in Typhoon output, skipping Claude cross-check")

        logger.info(f"[LargeExtraction] Page {page_num}: extracted (Typhoon {len(typhoon_markdown)} chars, Claude {len(claude_markdown)} chars)")
        return {"page_num": page_num, "typhoon_markdown": typhoon_markdown, "claude_markdown": claude_markdown}
    except Exception as e:
        # Added 2026-09: without this, one page's unrecoverable failure (e.g.
        # Typhoon retries exhausted, vision-verify erroring, a corrupted page
        # pymupdf can't render) propagates out of the ThreadPoolExecutor
        # future in extract_full_document() and kills the ENTIRE document —
        # discarding every other page already successfully (and expensively)
        # extracted in the same run. A placeholder record for just this one
        # page costs nothing and keeps the rest of the document's work
        # intact; a reviewer sees the error text in the review queue and
        # knows exactly which page to re-check manually.
        logger.error(f"[LargeExtraction] Page {page_num}: extraction failed, saving placeholder instead of losing the whole document: {e}")
        error_markdown = f"[สกัดข้อมูลหน้านี้ไม่สำเร็จ — เกิดข้อผิดพลาดระหว่างประมวลผล: {e}]"
        return {"page_num": page_num, "typhoon_markdown": error_markdown, "claude_markdown": ""}


def extract_full_document(
    bucket: str, raw_pdf_key: str, on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
    """Downloads the original PDF from S3 and OCRs every page with no time
    pressure (unlike the Lambda path, this can safely take as long as it
    needs to). Returns the same page-record shape process_extraction_result()
    already expects, so downstream validation/drafting/candidate-matching
    code needs zero changes to handle large-document results.

    on_progress(pages_done, pages_total), if given, is called after every
    page completes — from this function's own thread (the as_completed loop
    below), never from a worker thread, so the callback doesn't need to be
    thread-safe itself. Whether/how often that actually turns into an S3
    write is the caller's call (see sqs_consumer.py) — this function just
    reports every single page, no throttling logic here."""
    s3 = boto3.client("s3", region_name=conf.AWS_REGION or None)
    logger.info(f"[LargeExtraction] Downloading s3://{bucket}/{raw_pdf_key}")
    obj = s3.get_object(Bucket=bucket, Key=raw_pdf_key)
    pdf_bytes = obj["Body"].read()

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    num_pages = len(doc)
    doc.close()
    logger.info(f"[LargeExtraction] {raw_pdf_key}: {num_pages} page(s), starting full extraction (no time limit on EC2)")

    pages: list[dict | None] = [None] * num_pages
    pages_done = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_extract_one_page, pdf_bytes, p): p for p in range(1, num_pages + 1)}
        for future in as_completed(futures):
            page_num = futures[future]
            pages[page_num - 1] = future.result()
            pages_done += 1
            if on_progress is not None:
                on_progress(pages_done, num_pages)

    logger.info(f"[LargeExtraction] {raw_pdf_key}: all {num_pages} page(s) extracted successfully")
    return pages

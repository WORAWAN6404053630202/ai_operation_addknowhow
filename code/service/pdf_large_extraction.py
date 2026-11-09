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
import difflib
import os
import random
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import boto3
import pymupdf
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from typhoon_ocr import ocr_document

import conf
from service.pdf_extraction_validation import _extract_salient_tokens
from utils.llm_call import get_shared_pdf_http_client
from utils.llm_cost_logging import CostAccumulator, log_call_duration, log_llm_cost
from utils.logger import get_logger, hardlog
from utils.pdf_cost_guard import get_cost_guard
from utils.pdf_text_layer import extract_page_native_text, extract_page_supplementary_text, extract_raw_page_text_best_effort, find_embedded_image_regions, render_region_to_png_bytes
from utils.pdf_rate_limiter import GlobalTyphoonRateLimiter

logger = get_logger(__name__)

_MAX_WORKERS = 4  # same concurrency cap as the Lambda extractor, for the same reason

# 2026-10 fix (QA-reproduced: real multi-page PDFs landing with ~224 bytes of
# total extracted text) — mirrors the identical fix in lambda/pdf_extraction/
# handler.py's _extract_one_page. Typhoon's OCR SDK can return an empty string
# with NO exception; forcing vision-verify on truly-empty output (not just on
# salient-token presence) closes the gap where that looked like a genuinely
# blank page instead of a failed OCR call. Deliberately checks for EMPTY
# (stripped length 0), not merely "short" — a length-threshold version of
# this check (first attempt, since reverted) false-triggered on legitimate
# short-but-correct OCR text (caught by tests/test_pdf_fault_isolation.py's
# existing fixture text, 19 chars of genuinely complete content). Recall-only
# change — can only make vision-verify fire more often, never less.

# Typhoon's own docs (https://docs.opentyphoon.ai/en/rate-limits/, checked
# 2026-09): typhoon-ocr is rate-limited to 2 requests/sec AND 20 requests/min.
# GlobalTyphoonRateLimiter (utils/pdf_rate_limiter.py) coordinates against
# BOTH ceilings via a shared DynamoDB table with every concurrent Lambda
# invocation AND this EC2 consumer — not just this process's own
# _MAX_WORKERS=4 page-level concurrency, which a purely local limiter can't
# protect against once more than one process is extracting at the same time.
_typhoon_rate_limiter = GlobalTyphoonRateLimiter(
    table_name=conf.PDF_RATE_LIMIT_DYNAMODB_TABLE,
    region_name=conf.PDF_RATE_LIMIT_DYNAMODB_REGION,
    logger=logger,
)

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
            hardlog(f"_run_typhoon page {page_num} attempt {attempt}: waiting on rate limiter")
            _typhoon_rate_limiter.wait()
            hardlog(f"_run_typhoon page {page_num} attempt {attempt}: calling ocr_document() now")
            _call_start = time.monotonic()
            try:
                result = ocr_document(
                    tmp_path,
                    model="typhoon-ocr",
                    figure_language="Thai",
                    task_type="v1.5",
                    api_key=conf.TYPHOON_OCR_API_KEY,
                )
                hardlog(f"_run_typhoon page {page_num} attempt {attempt}: ocr_document() returned ({len(result or '')} chars)")
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


def _run_vision_verify(png_bytes: bytes, cost_accumulator: Optional[CostAccumulator] = None, cost_guard=None, anchor_text: str = "") -> str:
    """Second-opinion OCR pass on top of Typhoon's — model is
    conf.OPENROUTER_MODEL_PDF_VISION (see conf.py for why this is its own
    constant, not conf.OPENROUTER_MODEL), not necessarily Claude despite the
    old function name this replaces.

    2026-11 (document-anchoring add-on): `anchor_text`, if given, is
    whatever (possibly partial/rejected) native text the page already had
    — see extract_raw_page_text_best_effort()'s docstring for why this
    isn't available from extract_page_native_text() itself. Appended to the
    prompt as explicitly-untrusted reference context, not an instruction to
    defer to it — it may be exactly the corrupted/incomplete text that
    caused OCR to run in the first place. Only Typhoon's prompt
    (ocr_document(), a third-party library call with no such hook) can't
    receive this; _run_vision_verify builds its own prompt, so it can."""
    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL, http_client=get_shared_pdf_http_client())
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    prompt_text = _CLAUDE_PROMPT
    if anchor_text.strip():
        prompt_text += (
            "\n\nข้อมูลอ้างอิงจากเลเยอร์ข้อความเดิมของหน้านี้ (อาจไม่สมบูรณ์หรือมีจุดผิดพลาด "
            "ใช้เป็นแนวทางเทียบเคียงเท่านั้น ห้ามเชื่อถือทั้งหมดหากขัดแย้งกับภาพจริง):\n" + anchor_text.strip()
        )
    hardlog(f"_run_vision_verify: calling {conf.OPENROUTER_MODEL_PDF_VISION} now")
    _call_start = time.monotonic()
    resp = client.chat.completions.create(
        model=conf.OPENROUTER_MODEL_PDF_VISION,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=4000,
    )
    hardlog("_run_vision_verify: call returned")
    log_llm_cost(logger, "LargeExtraction/VisionVerify", conf.OPENROUTER_MODEL_PDF_VISION, resp, time.monotonic() - _call_start, accumulator=cost_accumulator, cost_guard=cost_guard)
    return resp.choices[0].message.content or "(no text returned)"


# 2026-11 (PDF pipeline overhaul, gap #5 — the second OCR pass
# [_run_vision_verify] was gated by _extract_salient_tokens matching one of
# only 4 content categories [baht amounts, dates, form codes, license
# numbers]; a prose-heavy page with none of those — a marketing/business
# chapter, a procedural narrative — could never trigger verification no
# matter how garbled Typhoon's output was. ~67% of OCR'd pages took this
# unverified path per this codebase's own 2026-08 backtest. These three
# signals are computed from TYPHOON'S OWN OUTPUT (output-derived), not from
# content category, so they fire equally on prose or numbers — closing
# exactly that gap. The salient-token trigger is KEPT (not retired; still a
# real, cheap, zero-false-positive signal for "definitely verify this"),
# these are ADDITIONAL triggers layered on top.
_REPETITION_COMPRESSION_RATIO_THRESHOLD = 0.20  # live-tested this session via zlib: normal Thai/English prose compresses to ~0.43-0.78, a realistic "starts normal then loops" degenerate sample compressed to ~0.05 — 0.20 sits comfortably between
_REPETITION_MIN_CHARS = 30  # below this, zlib's fixed per-call header overhead makes the ratio meaningless (short legitimate strings naturally score >1.0, which never trips a <threshold check anyway, but skip explicitly for clarity)


def _is_likely_degenerate_repetition(text: str) -> bool:
    """Flags text that compresses abnormally well — LightOnOCR's technique
    (arXiv:2601.14251) for detecting VLM OCR's known repetition-loop
    failure mode (Nougat, ICLR 2024, documents this as architectural to
    this model class, not a rare edge case). Never raises; a zlib failure
    (should not happen on any str) fails toward False (don't flag)."""
    stripped = (text or "").strip()
    if len(stripped) < _REPETITION_MIN_CHARS:
        return False
    try:
        data = stripped.encode("utf-8")
        compressed = zlib.compress(data, level=9)
        ratio = len(compressed) / len(data)
        return ratio < _REPETITION_COMPRESSION_RATIO_THRESHOLD
    except Exception:
        return False


_BLANK_PAGE_PNG_BYTES_FLOOR = 15000  # rough baseline for a near-blank rendered page at this pipeline's dpi (150-200) — live-tested this session; needs gold-set validation (see plan) before being treated as precisely calibrated
_MIN_CHARS_FOR_DENSE_PNG = 30  # if the rendered page is clearly non-blank (large PNG) but OCR returned under this many characters, likely truncation/omission, not a genuinely sparse page


def _is_length_implausible(png_bytes: bytes, ocr_text: str) -> bool:
    """Coarse proxy for 'this page looks visually dense but OCR returned
    almost nothing' — uses PNG-compressed byte size as a stand-in for
    visual complexity (PNG's filters compress blank/uniform regions far
    more efficiently than busy ones, confirmed empirically this session: a
    text-dense synthetic page rendered ~14x larger than a near-blank one at
    the same DPI) rather than per-pixel ink-density sampling, which would
    need iterating every pixel. Deliberately conservative — only flags the
    EXTREME case (substantial image, near-empty text), not partial
    truncation, to avoid false-positiving cost onto genuinely sparse pages.
    Never raises."""
    try:
        if len(png_bytes) <= _BLANK_PAGE_PNG_BYTES_FLOOR:
            return False
        return len((ocr_text or "").strip()) < _MIN_CHARS_FOR_DENSE_PNG
    except Exception:
        return False


_RANDOM_AUDIT_SAMPLE_RATE = 0.07  # ~7%, matching Consensus Entropy's (arXiv:2504.11101) empirically-found escalation rate for when verification actually helps


def _should_random_audit_sample() -> bool:
    """Unconditional, content-independent — purely to measure the TRUE
    unverified-population error rate over time (feeds Phase 6's
    reconciliation data), not to catch any specific page. A small
    unconditional sample is the only way to learn whether the other
    triggers above are actually catching what needs catching, since by
    definition every page that passes them is never checked."""
    return random.random() < _RANDOM_AUDIT_SAMPLE_RATE


def _texts_substantially_agree(text_a: str, text_b: str) -> bool:
    """Lightweight reconciliation between Typhoon's and vision-verify's
    output when both exist — difflib's SequenceMatcher ratio (stdlib, no
    new dependency) rather than a full Consensus-Entropy-style multi-model
    embedding comparison (arXiv:2504.11101's full technique, out of scope
    for a first cut — this is the cheap, directly-applicable piece of it).
    0.70 similarity threshold is a starting point, not gold-set-validated."""
    a, b = (text_a or "").strip(), (text_b or "").strip()
    if not a and not b:
        return True
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.70


def _extract_one_page(pdf_bytes: bytes, page_num: int, cost_accumulator: Optional[CostAccumulator] = None, cost_guard=None) -> dict:
    # Added 2026-09 (same optimization shipped to lambda/pdf_extraction/
    # handler.py's normal path first) — many digital-native pages need no
    # OCR at all. Matters even more here than on the Lambda side: this is
    # the large-document path, where skipping both OCR calls on a page that
    # doesn't need them saves the most per page, and every document over
    # MAX_PAGES_FOR_LAMBDA now lands here unconditionally (the Lambda
    # pre-screen that used to filter some of them out was removed 2026-09).
    # 2026-11 (PDF pipeline overhaul, gap #2): form-field/annotation values
    # are invisible to BOTH page.get_text() and OCR alike (a filled-in value
    # isn't rendered as part of the page's normal content stream the way
    # visible text is) — unconditional and additive regardless of which
    # branch below handles the base text, so computed once here up front.
    supplementary_text = extract_page_supplementary_text(pdf_bytes, page_num)

    native_text = extract_page_native_text(pdf_bytes, page_num, cost_accumulator=cost_accumulator, cost_guard=cost_guard)
    if native_text is not None:
        if supplementary_text:
            native_text = f"{native_text}\n\n[ข้อมูลจากฟอร์ม/คำอธิบายประกอบ]\n{supplementary_text}"
        # 2026-11 (PDF pipeline overhaul, gap #1): accepting native text used
        # to mean zero OCR for the WHOLE page, silently dropping a scanned
        # table/stamp/image embedded alongside otherwise-good text — this is
        # ADDITIVE (appended, never replaces native_text) and independent of
        # the vision-check veto gate above (which still covers vector-drawn
        # diagrams, which get_image_info() can't see).
        image_regions = find_embedded_image_regions(pdf_bytes, page_num)
        for region_bbox in image_regions:
            try:
                region_png = render_region_to_png_bytes(pdf_bytes, page_num, region_bbox)
                region_text = _run_typhoon(region_png, page_num)
            except Exception as e:
                logger.warning(f"[LargeExtraction] Page {page_num}: region-OCR failed for bbox {region_bbox} (non-fatal, native text unaffected): {e}")
                continue
            if (region_text or "").strip():
                native_text = f"{native_text}\n\n[รูปภาพ/ตารางที่แนบในหน้านี้]\n{region_text}"
        logger.info(f"[LargeExtraction] Page {page_num}: used native PDF text layer ({len(native_text)} chars) — skipped full-page OCR, $0-+region-OCR-only for this page")
        return {"page_num": page_num, "typhoon_markdown": native_text, "claude_markdown": native_text, "typhoon_was_empty": False}

    try:
        png_bytes = _render_page_to_png_bytes(pdf_bytes, page_num)
        typhoon_markdown = _run_typhoon(png_bytes, page_num)

        # Added 2026-08-31, EXTENDED 2026-11 (PDF pipeline overhaul, gap #5).
        # Original backtest (13 documents/49 pages): compare_extractions()
        # only ever flags a disagreement in one of 4 salient-token categories
        # (baht amounts, dates, form codes, license numbers), so a page where
        # Typhoon found none of those couldn't produce a flag regardless of
        # what Claude would have said — caught 14/14 (100%) of real
        # disagreements in that backtest while only calling Claude on 32.7%
        # of pages. BUT that content-category gate structurally can't fire on
        # a prose-heavy page with none of those 4 categories, no matter how
        # garbled — confirmed this session to be ~67% of OCR'd pages,
        # entirely unverified. The three new signals below are computed from
        # Typhoon's own OUTPUT instead (degenerate repetition, implausibly
        # short given the image's visual complexity, a small unconditional
        # random audit sample) — see their docstrings above this function.
        # The salient-token trigger is KEPT, not replaced; these are
        # additional, independent triggers layered on top of it.
        salient = _extract_salient_tokens(typhoon_markdown)
        _typhoon_is_empty = not (typhoon_markdown or "").strip()
        _is_degenerate = _is_likely_degenerate_repetition(typhoon_markdown)
        _length_implausible = _is_length_implausible(png_bytes, typhoon_markdown)
        _is_audit_sample = _should_random_audit_sample()
        _verify_reason = (
            "empty" if _typhoon_is_empty else
            "salient_token" if any(salient[k] for k in salient) else
            "degenerate_repetition" if _is_degenerate else
            "length_implausible" if _length_implausible else
            "random_audit" if _is_audit_sample else
            None
        )
        if _verify_reason is not None:
            logger.info(f"[LargeExtraction] Page {page_num}: running vision-verify (reason={_verify_reason})")
            # 2026-11 (document-anchoring add-on): whatever native text this
            # page had — even if extract_page_native_text() rejected it —
            # goes along as grounding context; see extract_raw_page_text_
            # best_effort()'s and _run_vision_verify()'s own docstrings.
            anchor_text = extract_raw_page_text_best_effort(pdf_bytes, page_num)
            claude_markdown = _run_vision_verify(png_bytes, cost_accumulator=cost_accumulator, cost_guard=cost_guard, anchor_text=anchor_text)
        else:
            claude_markdown = ""
            logger.info(f"[LargeExtraction] Page {page_num}: no salient tokens and output looks clean — skipping Claude cross-check")

        if not (typhoon_markdown or "").strip() and not (claude_markdown or "").strip():
            logger.warning(f"[LargeExtraction] Page {page_num}: BOTH Typhoon and vision-verify returned empty text — this page is likely silently missing from the ingested document, worth a manual re-check")
        elif _typhoon_is_empty and (claude_markdown or "").strip():
            # 2026-11 (hardcoded-logic audit): previously the recovered claude_markdown
            # was used ONLY as compare_extractions()'s comparison input, never merged
            # back as the actual page content — every downstream consumer reads
            # typhoon_markdown only (sqs_consumer.py's page_markdowns, feeding
            # identify_knowhow_topics/draft_fields_from_pages/the embedded document),
            # so a page Typhoon failed on but Claude successfully recovered silently
            # contributed nothing. Promoting it here fixes every consumer at once.
            logger.info(f"[LargeExtraction] Page {page_num}: Typhoon was empty, recovered via vision-verify ({len(claude_markdown)} chars) — using Claude's text as the page content")
            typhoon_markdown = claude_markdown
        elif (claude_markdown or "").strip() and not _texts_substantially_agree(typhoon_markdown, claude_markdown):
            # 2026-11 (gap #5, reconciliation half — FIXED 2026-11-10 after a
            # real-corpus validation run caught this live): BOTH models
            # produced non-empty output but they substantially disagree —
            # one of them is likely wrong (hallucination, omission, or a
            # degenerate loop that still emitted some text).
            #
            # Originally this failed toward the LONGER of the two as a
            # "conservative" tie-breaker, reasoning that the failure modes
            # this check targets (truncation/omission) are length-reducing.
            # That reasoning breaks exactly when the trigger itself was
            # degenerate_repetition: a repetition-loop failure INFLATES
            # length (confirmed live — Typhoon looped on a form's ID-number
            # blank-line placeholder and returned 32,771 characters of
            # repeated "- ", which "prefer longer" then picked over
            # vision-verify's correct short answer, corrupting that page).
            # Now: if exactly one side is itself a degenerate repetition,
            # the OTHER side wins regardless of length. Only when neither
            # (or both) are degenerate does length-based tie-breaking apply.
            typhoon_is_degenerate = _is_likely_degenerate_repetition(typhoon_markdown)
            claude_is_degenerate = _is_likely_degenerate_repetition(claude_markdown)
            if typhoon_is_degenerate and not claude_is_degenerate:
                winner = claude_markdown
                winner_reason = "typhoon output itself was a degenerate repetition"
            elif claude_is_degenerate and not typhoon_is_degenerate:
                winner = typhoon_markdown
                winner_reason = "vision-verify output itself was a degenerate repetition"
            elif typhoon_is_degenerate and claude_is_degenerate:
                # Both degenerate (rare, pathological) — prefer the SHORTER
                # one here, the opposite of the default below: a repetition
                # loop inflates length, so shorter is the safer guess when
                # both sides show the same failure mode.
                winner = claude_markdown if len(claude_markdown) < len(typhoon_markdown) else typhoon_markdown
                winner_reason = "both outputs looked degenerate — preferred the shorter one"
            else:
                winner = claude_markdown if len(claude_markdown) > len(typhoon_markdown) else typhoon_markdown
                winner_reason = "conservative tie-breaker (neither output looked degenerate)"
            logger.warning(
                f"[LargeExtraction] Page {page_num}: Typhoon and vision-verify substantially disagree "
                f"(reason={_verify_reason}) — using the {len(winner)}-char output ({winner_reason})"
            )
            typhoon_markdown = winner

        if supplementary_text:
            typhoon_markdown = f"{typhoon_markdown}\n\n[ข้อมูลจากฟอร์ม/คำอธิบายประกอบ]\n{supplementary_text}"

        logger.info(f"[LargeExtraction] Page {page_num}: extracted (Typhoon {len(typhoon_markdown)} chars, Claude {len(claude_markdown)} chars)")
        return {
            "page_num": page_num, "typhoon_markdown": typhoon_markdown, "claude_markdown": claude_markdown,
            "typhoon_was_empty": _typhoon_is_empty,
        }
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
        return {"page_num": page_num, "typhoon_markdown": error_markdown, "claude_markdown": "", "typhoon_was_empty": False}


# 2026-11 (PDF pipeline overhaul, gap #4 — tables spanning 2 pages get
# split with no linking step: the continuation's repeated header lands as a
# data row in the middle of the table, or the continuation is simply lost
# as a disconnected second table). First-cut heuristic, text-level (not
# structural) because pages are extracted independently and in PARALLEL
# (see the ThreadPoolExecutor below) — no Table objects survive past a
# single page's own extraction, only the markdown text
# extract_page_text_with_tables() already produced. Deliberately fails
# TOWARD leaving two separate, visible tables rather than risking a wrong
# merge — a split-but-intact table is recoverable by a reader; a wrongly
# fused one looks like one coherent table and isn't. Needs validation
# against real multi-page tables from the gold-set corpus before being
# trusted as reliable (see tests/fixtures/pdf_gold_set/README.md) — this
# is explicitly flagged in the plan as the least-tested piece of gap #4.
_MARKDOWN_TABLE_LINE_RE_PREFIX = "|"


def _extract_trailing_markdown_table(text: str) -> tuple[str, str] | None:
    """If `text` ends with a markdown table block (every trailing non-blank
    line starts and ends with '|'), returns (text_before_table, table_block).
    Else None. At least a header + separator row (2 lines) required."""
    lines = text.rstrip().split("\n")
    table_lines: list[str] = []
    i = len(lines) - 1
    while i >= 0 and lines[i].strip().startswith(_MARKDOWN_TABLE_LINE_RE_PREFIX) and lines[i].strip().endswith("|"):
        table_lines.insert(0, lines[i])
        i -= 1
    if len(table_lines) < 2:
        return None
    return "\n".join(lines[: i + 1]), "\n".join(table_lines)


def _extract_leading_markdown_table(text: str) -> tuple[str, str] | None:
    """Mirror of _extract_trailing_markdown_table for the START of a page's
    text instead of the end."""
    lines = text.lstrip().split("\n")
    table_lines: list[str] = []
    i = 0
    while i < len(lines) and lines[i].strip().startswith(_MARKDOWN_TABLE_LINE_RE_PREFIX) and lines[i].strip().endswith("|"):
        table_lines.append(lines[i])
        i += 1
    if len(table_lines) < 2:
        return None
    return "\n".join(table_lines), "\n".join(lines[i:])


def link_cross_page_tables(pages: list[dict]) -> list[dict]:
    """Mutates and returns `pages` in place: for each adjacent pair where
    page N's text ends with a markdown table and page N+1's begins with one
    whose HEADER ROW matches exactly (normalized whitespace), merges them —
    keeps page N's header+separator, appends page N+1's data rows (skipping
    its own duplicate header+separator), and removes the now-redundant
    table from page N+1's leading text. A col_count/header mismatch means
    no merge — both tables are left exactly as extracted. Never raises;
    any internal error leaves `pages` unmodified from that point on rather
    than risking a partial/corrupted merge."""
    try:
        anchor = 0
        while anchor < len(pages):
            page_a = pages[anchor]
            if page_a is None:
                anchor += 1
                continue
            next_idx = anchor + 1
            extended = False
            while next_idx < len(pages):
                page_b = pages[next_idx]
                if page_b is None:
                    break
                text_a = page_a.get("typhoon_markdown") or ""
                text_b = page_b.get("typhoon_markdown") or ""
                trailing = _extract_trailing_markdown_table(text_a)
                leading = _extract_leading_markdown_table(text_b)
                if not trailing or not leading:
                    break
                before_a, table_a = trailing
                table_b, after_b = leading
                table_a_lines = table_a.split("\n")
                table_b_lines = table_b.split("\n")
                if len(table_a_lines) < 2 or len(table_b_lines) < 2:
                    break
                header_a = " ".join(table_a_lines[0].split())
                header_b = " ".join(table_b_lines[0].split())
                if header_a != header_b:
                    break
                data_rows_b = table_b_lines[2:]
                merged_table = table_a + ("\n" + "\n".join(data_rows_b) if data_rows_b else "")
                new_text_a = f"{before_a}\n\n{merged_table}" if before_a.strip() else merged_table
                page_a["typhoon_markdown"] = new_text_a
                page_a["claude_markdown"] = new_text_a
                page_b["typhoon_markdown"] = after_b
                page_b["claude_markdown"] = after_b
                logger.info(f"[LargeExtraction] Linked a table spanning page {page_a.get('page_num')} -> page {page_b.get('page_num')} (matching header: {header_a[:60]!r})")
                extended = True
                next_idx += 1
            # Advance past every page just absorbed into page_a's table —
            # this is the actual fix for a 3+-page chain: without it, the
            # outer loop would next inspect page_b (now possibly emptied of
            # its own trailing table, since its content WAS the table that
            # just got merged away), lose the "this table is still growing"
            # context, and silently fail to continue the chain into
            # whatever page comes after page_b.
            anchor = next_idx if extended else anchor + 1
        return pages
    except Exception as e:
        logger.warning(f"[LargeExtraction] Cross-page table linking failed (non-fatal, tables left unmerged/split): {e}")
        return pages


def extract_full_document(
    bucket: str, raw_pdf_key: str, on_progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[list[dict], float, float]:
    """Downloads the original PDF from S3 and OCRs every page with no time
    pressure (unlike the Lambda path, this can safely take as long as it
    needs to). Returns (pages, extraction_cost_usd, extraction_started_at) —
    the pages list is the same page-record shape process_extraction_result()
    already expects, so downstream validation/drafting/candidate-matching
    code needs zero changes to handle large-document results; the cost/
    started_at pair (added 2026-09) lets sqs_consumer.py seed its own
    build-phase CostAccumulator and set ReviewItem.extraction_started_at for
    a real wall-clock number in the admin UI — see model/pdf_review_item.py.

    on_progress(pages_done, pages_total), if given, is called after every
    page completes — from this function's own thread (the as_completed loop
    below), never from a worker thread, so the callback doesn't need to be
    thread-safe itself. Whether/how often that actually turns into an S3
    write is the caller's call (see sqs_consumer.py) — this function just
    reports every single page, no throttling logic here."""
    # Cost tracking (2026-10) — see utils/pdf_cost_guard.py. No longer a
    # blocking ceiling (REMOVED 2026-10: a calendar-day auto-block had no way
    # to distinguish a genuine large bulk upload from an actual runaway
    # malfunction, and made the first look worse than the second — see
    # sqs_consumer.py's _MAX_HANDOFF_ATTEMPTS and lambda/pdf_extraction/
    # handler.py's _MAX_ATTEMPTS for the actual fix: a bounded per-document
    # retry cap, which targets the real failure mode directly instead of
    # capping legitimate throughput). cost_guard here still records every
    # call's cost for the admin dashboard's running total — display only.
    cost_guard = get_cost_guard()

    started_at = time.time()
    s3 = boto3.client("s3", region_name=conf.AWS_REGION or None)
    logger.info(f"[LargeExtraction] Downloading s3://{bucket}/{raw_pdf_key}")
    obj = s3.get_object(Bucket=bucket, Key=raw_pdf_key)
    pdf_bytes = obj["Body"].read()

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    num_pages = len(doc)
    doc.close()
    logger.info(f"[LargeExtraction] {raw_pdf_key}: {num_pages} page(s), starting full extraction (no time limit on EC2)")
    hardlog(f"extract_full_document {raw_pdf_key}: {num_pages} page(s), submitting to {_MAX_WORKERS}-worker pool")

    cost_accumulator = CostAccumulator()
    pages: list[dict | None] = [None] * num_pages
    pages_done = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_extract_one_page, pdf_bytes, p, cost_accumulator, cost_guard): p for p in range(1, num_pages + 1)}
        for future in as_completed(futures):
            page_num = futures[future]
            pages[page_num - 1] = future.result()
            pages_done += 1
            hardlog(f"extract_full_document {raw_pdf_key}: page {page_num} done ({pages_done}/{num_pages})")
            if on_progress is not None:
                on_progress(pages_done, num_pages)

    logger.info(f"[LargeExtraction] {raw_pdf_key}: all {num_pages} page(s) extracted successfully")
    hardlog(f"extract_full_document {raw_pdf_key}: all {num_pages} page(s) extracted successfully")
    # 2026-11 (PDF pipeline overhaul, gap #4): must run AFTER every page is
    # extracted (pages ran in parallel above, so no reliable N->N+1 order
    # existed until now) and BEFORE this function returns, since every
    # downstream consumer reads the final `pages` list.
    pages = link_cross_page_tables(pages)
    return pages, cost_accumulator.total_cost_usd, started_at

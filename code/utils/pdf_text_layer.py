# code/utils/pdf_text_layer.py
"""Zero-cost alternative to OCR for a PDF page (feature/pdf-ingestion) — many
digital-native (non-scanned) PDFs already carry an extractable text layer via
pymupdf's page.get_text(), which costs nothing to read and is more accurate
than OCR when it's genuinely there. Added 2026-09 to lambda/pdf_extraction/
handler.py first (the normal ≤MAX_PAGES_FOR_LAMBDA path); pulled out into this
shared module so service/pdf_large_extraction.py (the EC2 large-document path)
can use the exact same logic via import instead of a third hand-copy — unlike
the Lambda handler, this side of the pipeline has no deployment-boundary
reason to duplicate code within code/ itself (see pdf_large_extraction.py's
own module docstring for why ONLY the Lambda side needs to stay a literal
duplicate: it's a standalone zip with no access to code/ at all).

Character-validity check background: the established root cause of a
corrupted PDF text layer (per PyMuPDF/Docling issue trackers and mojibake-
detection literature, e.g. the `ftfy` library's heuristic docs) is a
CID-keyed font with no ToUnicode map, or a straight wrong-encoding decode.
Both failure modes overwhelmingly produce characters OUTSIDE what a
legitimate Thai document should contain: Unicode Private Use Area
(U+E000-U+F8FF, the single most-cited signature of the missing-ToUnicode-map
case) or the U+FFFD replacement character (the signature of a wrong-encoding
decode). Scoped to Thai + ASCII rather than reusing `ftfy`-style pattern
lists on purpose — those are tuned for Western encoding confusion (UTF-8
decoded as Windows-1252 etc.), not the CID-font failure mode that's the more
common cause for Thai-language PDFs specifically.

Image-detection history (2026-09): the first version of this module forced
OCR fallback whenever page.get_image_info() found an embedded raster image
covering >=5% of the page area. Replaced after live review surfaced two real
blind spots in that approach, calibrated on only one document: (1) it is
blind to vector-drawn diagrams (page.get_drawings() content — lines/fills
from a PowerPoint/Illustrator export), since get_image_info() only sees
embedded raster XObjects; (2) a fixed area threshold is a proxy for "is this
visually meaningful", and the actual answer depends on what's drawn, not how
many square points it occupies (a QR code or signature stamp can be small AND
critical). Replaced with a two-stage check that asks the actual question
instead of approximating it via area: a cheap local pre-filter (Approach A,
_text_coverage_ratio) that only ever SKIPS the paid check when a page is
confidently text-only, plus a real vision-model check (Approach B,
_page_needs_ocr_for_visual_content) that looks at the rendered page directly.
Real-measured cost: $0.0001355/page via google/gemini-2.5-flash-lite (see
conf.OPENROUTER_MODEL_PDF_IMAGE_CHECK) — a 634-page document costs ≈$0.086
(≈3 THB) if every page needed the check. An audit-sampling safety net
(re-verifying a random sample of already-ingested pages after the fact) was
considered and explicitly declined for this round: targeted per-topic
re-ingestion of just the bad page was judged riskier than it's worth without
whole-file reprocessing, and whole-file reprocessing for a single bad page
was judged not worth the cost/time — so correctness here rests entirely on
Approach A+B being right the first time, not on a later safety net.

Two further gaps closed 2026-09 (raised directly by user review, not found
proactively):

1. text_looks_valid() above averages the bad-character ratio across the
   WHOLE page (5% threshold) — this can pass even when every corrupted
   character concentrates right next to the one number that actually
   matters (a fee amount, a date, a form/license code) while 95%+ of the
   page's prose is fine. See _salient_regions_look_valid() below: rather
   than re-run service/pdf_extraction_validation.py's exact salient-token
   regexes (_BAHT_AMOUNT_RE etc.) — which would silently fail to match the
   very corruption they're meant to catch, since a replacement/PUA
   character isn't `\d` and breaks the pattern instead of triggering it —
   this scans for any out-of-range character sitting near a digit run
   instead, which survives exactly that failure mode.
2. _VISION_IMAGE_CHECK_PROMPT below used to ask only about diagrams/charts/
   photos. A status stamp or watermark ("DRAFT", "APPROVED", official seal)
   is not a diagram, so the vision model was never actually asked about it
   even though it was already looking at the rendered page — the mechanism
   existed but the question didn't. Widened the prompt to ask about both."""

from __future__ import annotations

import base64
import re
import time
from typing import Optional

import pymupdf
from openai import OpenAI

import conf
from utils.llm_cost_logging import CostAccumulator, log_llm_cost
from utils.llm_json_parse import parse_leading_json
from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_PAGE_TEXT_LAYER_CHARS = 40  # below this, treat as scanned/image-only — not enough embedded text to trust

_EXPECTED_TEXT_RANGES = (
    (0x0009, 0x000A),  # tab, newline
    (0x0020, 0x007E),  # ASCII printable: Latin letters, digits, punctuation
    (0x0E00, 0x0E7F),  # Thai
    (0x2018, 0x201F),  # curly quotes
    (0x2013, 0x2014),  # en/em dash
)
_MAX_BAD_CHAR_RATIO = 0.05  # >5% characters outside the expected ranges = likely corrupted

# Approach A — a page whose visible text blocks already cover most of the
# page area is confidently text-only; skip the paid vision check (Approach B)
# for it. This threshold is loose and one-directional on purpose: it only
# ever short-circuits to "skip the check", never to "force OCR" — a page
# that DOESN'T clear this bar still goes through the real vision check below
# rather than being assumed to have an image. Getting this threshold
# slightly wrong therefore costs at most a few extra vision-check calls
# ($0.0001355 each), never a missed diagram.
_TEXT_COVERAGE_SKIP_THRESHOLD = 0.70

_VISION_IMAGE_CHECK_PROMPT = (
    "Does this page contain EITHER of the following that would be lost or "
    "misunderstood if only the text were extracted?\n"
    "1. A diagram, chart, flowchart, photo, or other meaningful image.\n"
    "2. A stamp, watermark, or seal indicating the document's status "
    "(e.g. \"DRAFT\", \"APPROVED\", \"COPY\", a red official seal) — this "
    "changes how the whole page should be interpreted, not just one detail.\n"
    "Ignore decorative logos, letterheads, and plain page borders. "
    'Answer with only JSON: {"has_meaningful_visual_content": true or false}'
)


def text_looks_valid(text: str) -> bool:
    """See _EXPECTED_TEXT_RANGES above for the reasoning. Ignores whitespace
    (not a signal either way) and never raises."""
    stripped = "".join(text.split())
    if not stripped:
        return False
    bad = sum(1 for ch in stripped if not any(lo <= ord(ch) <= hi for lo, hi in _EXPECTED_TEXT_RANGES))
    return (bad / len(stripped)) <= _MAX_BAD_CHAR_RATIO


# How close (in characters) an out-of-range character must be to a digit to
# count as "corrupting a high-value token" in _salient_regions_look_valid()
# below. All four salient-token shapes this pipeline already cares about
# (baht amounts, dates, form codes, license numbers — see
# service/pdf_extraction_validation.py's _BAHT_AMOUNT_RE etc.) are digit
# runs, so proximity-to-digit is a document-type-agnostic proxy for
# proximity-to-high-value-content — no per-document-family keyword list
# needed, matching that module's own stated preference for checks that need
# no per-document-type configuration.
_SALIENT_CORRUPTION_PROXIMITY_CHARS = 3

# Excludes ordinary page-number self-references ("หน้า 6", "หน้า 12") from
# counting as "near a digit" in _salient_regions_look_valid() below. Found
# live against the real ELICENSE document: EVERY page's footer carries a
# decorative icon glyph (a single PUA codepoint, U+F09E, confirmed via a
# full-document scan — no other codepoint ever triggered) immediately after
# "หน้า <N>" — 43 of 66 pages got flagged before this exclusion, and ALL 43
# were this exact same icon in this exact same context, zero real
# corruption among them. "หน้า" is a generic Thai word used only for page
# references (never for a fee amount, date, form code, or license number),
# so excluding digits inside this specific pattern carries near-zero risk
# of masking a real corrupted salient token.
_PAGE_NUMBER_REF_RE = re.compile(r"หน้า\s*\d+")


def _salient_regions_look_valid(text: str) -> bool:
    """text_looks_valid() above catches WIDESPREAD corruption (>5% of the
    page); this catches corruption LOCALIZED right next to a number — the
    exact case that check's page-wide average can miss. Deliberately does
    NOT reuse service/pdf_extraction_validation.py's salient-token regexes
    directly: those regexes require an unbroken run of `\\d` characters to
    match at all, so a replacement/PUA character sitting INSIDE a fee amount
    or license number (e.g. "3,06� บาท") breaks the very pattern meant
    to detect it — the regex would simply fail to match, going silent
    instead of firing. Scanning for any out-of-range character within
    _SALIENT_CORRUPTION_PROXIMITY_CHARS of a digit survives that failure
    mode, at the cost of also flagging unrelated corruption that happens to
    sit near an ordinary page number or list index (see
    _PAGE_NUMBER_REF_RE above for the one such case found live and
    specifically excluded) — an intentional fail-toward-OCR tradeoff for any
    remaining case, same direction as _page_needs_ocr_for_visual_content()'s
    fail-safe below, since the cost of an unnecessary OCR pass is far lower
    than the cost of a corrupted fee amount or legal reference silently
    reaching a user. Never raises."""
    page_number_ref_positions = {i for m in _PAGE_NUMBER_REF_RE.finditer(text) for i in range(*m.span())}
    for i, ch in enumerate(text):
        if ch.isspace() or any(lo <= ord(ch) <= hi for lo, hi in _EXPECTED_TEXT_RANGES):
            continue
        nearby_range = range(max(0, i - _SALIENT_CORRUPTION_PROXIMITY_CHARS), min(len(text), i + 1 + _SALIENT_CORRUPTION_PROXIMITY_CHARS))
        if any(text[j].isdigit() and j not in page_number_ref_positions for j in nearby_range if j != i):
            return False
    return True


def _text_coverage_ratio(page) -> float:
    """Approach A (see module docstring): sums the bounding-box area of every
    text block pymupdf finds against the total page area — cheap, local, no
    API call. Never raises; an unexpected failure returns 0.0 (never
    confidently text-only), which only means the real vision check runs, not
    that anything gets misclassified."""
    try:
        page_area = page.rect.width * page.rect.height
        if page_area <= 0:
            return 0.0
        text_area = 0.0
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:  # 0 = text block, 1 = image block — a full-page
                continue                # image block must never count as "text coverage"
            bbox = block.get("bbox")
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox
            text_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
        return min(1.0, text_area / page_area)
    except Exception:
        return 0.0


def _page_needs_ocr_for_visual_content(page, cost_accumulator: Optional[CostAccumulator] = None) -> bool:
    """Approach B (see module docstring): renders the page as a low-res
    thumbnail and asks a real vision model whether it carries a diagram/
    chart/meaningful image OR a document-status stamp/watermark (see
    _VISION_IMAGE_CHECK_PROMPT — widened 2026-09 to ask about both; the
    mechanism of looking at the rendered page already covered stamps, but
    the question asked of it didn't). Fails SAFE toward True (assume
    meaningful visual content is present -> caller falls back to OCR) on
    any error — an unnecessary OCR pass costs a little money; a silently
    dropped diagram or missed "DRAFT" stamp costs correctness, and this
    pipeline has already been burned by that exact failure mode once (see
    the replaced area-ratio approach above)."""
    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(0.5, 0.5))
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
        _call_start = time.monotonic()
        resp = client.chat.completions.create(
            model=conf.OPENROUTER_MODEL_PDF_IMAGE_CHECK,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_IMAGE_CHECK_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            max_tokens=30,
        )
        log_llm_cost(
            logger,
            "PdfTextLayer/ImageCheck",
            conf.OPENROUTER_MODEL_PDF_IMAGE_CHECK,
            resp,
            time.monotonic() - _call_start,
            accumulator=cost_accumulator,
        )
        result = parse_leading_json(resp.choices[0].message.content or "")
        return bool(result.get("has_meaningful_visual_content", True))
    except Exception as e:
        logger.warning(f"[PdfTextLayer] Vision image-check failed, defaulting to OCR fallback (safe direction): {e}")
        return True


def extract_page_native_text(pdf_bytes: bytes, page_num: int, cost_accumulator: Optional[CostAccumulator] = None) -> str | None:
    """Returns None (caller falls back to OCR) for a scanned/image-only page
    with no meaningful embedded text, a text layer that fails the
    page-wide or salient-region character-validity check above (likely a
    corrupted/mismapped font), a page whose rendered content includes a
    meaningful diagram/chart/image/status-stamp per the Approach A+B check
    above (OCR is vision-capable and will describe it; plain text
    extraction cannot), or if extraction itself errors — never raises."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            page = doc[page_num - 1]
            text = page.get_text()
            if len(text.strip()) < _MIN_PAGE_TEXT_LAYER_CHARS:
                return None
            if not text_looks_valid(text):
                logger.info(f"[PdfTextLayer] Page {page_num}: native text layer present but failed the character-validity check (likely a corrupted/mismapped font) — falling back to OCR")
                return None
            if not _salient_regions_look_valid(text):
                logger.info(f"[PdfTextLayer] Page {page_num}: corrupted character found right next to a number (fee amount/date/form/license code) — falling back to OCR even though the page-wide validity check passed")
                return None
            if _text_coverage_ratio(page) >= _TEXT_COVERAGE_SKIP_THRESHOLD:
                return text
            if _page_needs_ocr_for_visual_content(page, cost_accumulator=cost_accumulator):
                logger.info(f"[PdfTextLayer] Page {page_num}: vision check found meaningful visual content — falling back to OCR so it gets described, not silently dropped")
                return None
            return text
        finally:
            doc.close()
    except Exception as e:
        logger.warning(f"[PdfTextLayer] Page {page_num}: native text-layer extraction failed, falling back to OCR: {e}")
        return None

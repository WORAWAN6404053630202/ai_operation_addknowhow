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


# 2026-11 (PDF pipeline overhaul, gap #3): text_looks_valid()/
# _salient_regions_look_valid() above are kept defined (still used by
# existing callers/tests that want a standalone Unicode-range check) but
# RETIRED from classify_page_text_blocks()'s main decision below, replaced
# by a composite score with two real weaknesses the old checks couldn't
# see: (1) IN-SCRIPT Thai mojibake (wrong tone-mark order, a bad legacy
# scanner's OCR layer) stays entirely within the Thai Unicode block and is
# invisible to a character-RANGE check no matter how corrupted the actual
# words are; (2) a single corrupted character in an otherwise-clean 400+
# char PAGE is nowhere near text_looks_valid()'s 5% page-wide average, but
# can be the entirety of one 15-char table cell — moving the unit of
# analysis from "whole page" to "one text block" (the same block.bbox
# structure page.get_text("dict")["blocks"] already exposes, also walked by
# _text_coverage_ratio() above) catches exactly that case.
#
# Live-tested this session against pythainlp 5.3.4's word_tokenize(engine=
# "newmm"): a character-shuffled Thai string tokenized to mean_token_length
# ~3 chars vs ~5.5 for the clean original — known-word-RATE alone was a much
# weaker signal (newmm's greedy longest-match "launders" garbled input into
# short-but-real dictionary fragments wherever a local match exists, hiding
# corruption in token COUNT while it still shows up clearly in token
# LENGTH) — so mean_token_length is weighted higher (0.7) than
# char-weighted known-word rate (0.3) in the composite below.
_THAI_KNOWN_WORDS = None


def _thai_known_words():
    """Lazy-loaded, module-level singleton — pythainlp.corpus.thai_words()
    is a frozenset (~62k entries) that only needs loading once per process,
    same pattern as this codebase's other singletons (e.g. utils/reranker.py)."""
    global _THAI_KNOWN_WORDS
    if _THAI_KNOWN_WORDS is None:
        from pythainlp.corpus import thai_words
        _THAI_KNOWN_WORDS = thai_words()
    return _THAI_KNOWN_WORDS


_MEAN_TOKEN_LENGTH_NORM = 6.0  # empirical: ~6 chars/token is typical for a clean Thai content word this session's live test


_MIN_THAI_SCRIPT_RATIO = 0.20  # below this, text is predominantly non-Thai; the linguistic check below doesn't apply


def thai_linguistic_score(text: str) -> float:
    """Returns 1.0 for text with no tokenizable content, OR text that isn't
    meaningfully Thai in the first place (pure-English/numeric block —
    nothing Thai to score, must not be penalized), down to ~0.0 for
    severely scrambled Thai. Deliberately scoped to SEVERE corruption (full
    mojibake/scrambling) — mild single-character OCR noise tokenizes around
    the bad character into still-valid shorter words and will NOT reliably
    separate from clean text on this signal alone (confirmed empirically
    this session); that failure mode belongs to Phase 5's output-derived
    uncertainty signals instead, not this check. Never raises — a tokenizer
    failure must fail toward TRUSTING the text (native-text rejection
    should come from a real signal, not a dependency hiccup).

    Bug found and fixed this session: pythainlp's word_tokenize does NOT
    leave pure-English text untokenized — it happily splits "This is real
    page body text" into short English word-chunks, which then score as if
    they were Thai (near-zero Thai-dictionary matches, short mean token
    length) — a legitimate pure-English page scored 0.44 before this fix,
    well below any reasonable validity threshold. Guarding on the
    THAI-SCRIPT character ratio first (not "did tokenization produce
    anything") is the actual fix — a block must contain a meaningful amount
    of Thai script to even be in this check's jurisdiction."""
    thai_chars = sum(1 for ch in text if 0x0E00 <= ord(ch) <= 0x0E7F)
    non_space_chars = sum(1 for ch in text if not ch.isspace())
    if non_space_chars == 0 or (thai_chars / non_space_chars) < _MIN_THAI_SCRIPT_RATIO:
        return 1.0
    try:
        from pythainlp.tokenize import word_tokenize
        toks = [t for t in word_tokenize(text, engine="newmm", keep_whitespace=False) if t.strip()]
    except Exception as e:
        logger.warning(f"[PdfTextLayer] Thai tokenization failed, treating block as trusted (fail-safe direction): {e}")
        return 1.0
    if not toks:
        return 1.0
    total_chars = sum(len(t) for t in toks)
    if total_chars == 0:
        return 1.0
    mean_len = total_chars / len(toks)
    known_words = _thai_known_words()
    char_weighted_known = sum(len(t) for t in toks if t in known_words) / total_chars
    length_signal = min(1.0, mean_len / _MEAN_TOKEN_LENGTH_NORM)
    return 0.7 * length_signal + 0.3 * char_weighted_known


def _block_unicode_bad_ratio(text: str) -> float:
    """Same character-range check as text_looks_valid(), but scoped to ONE
    block's text and reusing _PAGE_NUMBER_REF_RE's exclusion (a decorative
    footer-icon glyph sitting right after "หน้า <N>" — the real, live
    false-positive this exclusion was built for, see _PAGE_NUMBER_REF_RE's
    own comment above — applies equally at block granularity)."""
    stripped_for_check = text
    page_number_ref_positions = {i for m in _PAGE_NUMBER_REF_RE.finditer(stripped_for_check) for i in range(*m.span())}
    bad = 0
    total = 0
    for i, ch in enumerate(stripped_for_check):
        if ch.isspace():
            continue
        total += 1
        if any(lo <= ord(ch) <= hi for lo, hi in _EXPECTED_TEXT_RANGES):
            continue
        if i in page_number_ref_positions:
            continue
        bad += 1
    return bad / total if total else 0.0


# 2026-11: live-tested this session against 5 independently random-shuffled
# full scrambles of a real clean Thai sentence — linguistic_component spread
# 0.50-0.73 (mean ~0.61) vs the CLEAN original's 0.97. Critically, the
# unicode-range component on EVERY scrambled sample was still ~1.0 (shuffled
# Thai characters are still valid Thai Unicode codepoints — scrambling the
# ORDER, not the characters, is exactly the mojibake failure mode this gap
# targets) — a 0.5/0.5 WEIGHTED AVERAGE of the two components therefore
# floors around 0.75-0.77 even for maximally-scrambled text, nowhere near
# low enough to separate from clean. Independent gates (both components
# must individually clear their own threshold) avoid this dilution entirely
# — each threshold tuned to that component's own natural range instead of
# being flattened into one shared scale.
_UNICODE_COMPONENT_THRESHOLD = 0.95  # matches the original _MAX_BAD_CHAR_RATIO=0.05 exactly, preserving that signal's prior behavior
_LINGUISTIC_COMPONENT_THRESHOLD = 0.80  # comfortably between this session's max-scrambled sample (0.73) and min-clean sample (0.93)


def text_block_composite_score(text: str) -> tuple[float, float]:
    """Returns (unicode_component, linguistic_component) — NOT a single
    blended number (see the threshold-design comment above for why
    averaging them dilutes the linguistic signal to uselessness against the
    specific corruption type this gap targets). 1.0 = confidently clean on
    that axis, 0.0 = confidently corrupted."""
    unicode_component = 1.0 - _block_unicode_bad_ratio(text)
    linguistic_component = thai_linguistic_score(text)
    return unicode_component, linguistic_component


def _block_is_valid(text: str) -> bool:
    unicode_component, linguistic_component = text_block_composite_score(text)
    return unicode_component >= _UNICODE_COMPONENT_THRESHOLD and linguistic_component >= _LINGUISTIC_COMPONENT_THRESHOLD


# Blocks shorter than this (e.g. a lone page-number footer, "Page 6") are
# too short for either score component to be statistically reliable — a
# single decorative glyph in a 5-char block swings the unicode-ratio
# component by 20%, which the old page-WIDE 5% average correctly never
# triggered on, and thai_linguistic_score's token-length signal is noisy on
# a 1-2-token block (empirically confirmed this session: "หน้า 6" alone
# scored 0.53, BELOW even some fully-scrambled long-sentence samples, purely
# because so few tokens exist to average over). Trusted implicitly rather
# than risking a false-positive reject on exactly the kind of tiny
# boilerplate block this codebase has already been burned by once (the
# ELICENSE footer-icon false-positive _PAGE_NUMBER_REF_RE itself was built
# to fix).
_MIN_BLOCK_CHARS_FOR_VALIDATION = 15


def classify_page_text_blocks(page) -> list[dict]:
    """Walks page.get_text("dict")["blocks"] (same structure
    _text_coverage_ratio() above already walks) and scores every TEXT block
    (type 0 — image blocks, type 1, are out of scope here) against
    text_block_composite_score(). Returns a list of
    {"bbox", "text", "score", "valid"} dicts; blocks under
    _MIN_BLOCK_CHARS_FOR_VALIDATION chars are skipped entirely (trusted
    implicitly, never appear in the returned list) rather than scored and
    risking a false-positive reject on boilerplate. Never raises — an
    unexpected failure returns [] (no blocks flagged), the same fail-
    direction as this module's other detection helpers (an empty result
    here means extract_page_native_text() below falls through to its other
    checks rather than wrongly rejecting a page over an internal error)."""
    try:
        results = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            block_text = "".join(
                span.get("text", "") for line in block.get("lines", []) for span in line.get("spans", [])
            )
            if len(block_text.strip()) < _MIN_BLOCK_CHARS_FOR_VALIDATION:
                continue
            unicode_component, linguistic_component = text_block_composite_score(block_text)
            results.append({
                "bbox": block.get("bbox"), "text": block_text,
                "unicode_score": unicode_component, "linguistic_score": linguistic_component,
                "valid": unicode_component >= _UNICODE_COMPONENT_THRESHOLD and linguistic_component >= _LINGUISTIC_COMPONENT_THRESHOLD,
            })
        return results
    except Exception as e:
        logger.warning(f"[PdfTextLayer] Per-block text validation failed (non-fatal, treated as no blocks flagged): {e}")
        return []


# 2026-11 (PDF pipeline overhaul, gap #4 — tables read as flowing text lose
# row/column structure; a fee table's values separate from their headers).
# page.find_tables() (PyMuPDF >=1.23, confirmed available this session:
# 1.28.2) is used instead of a new dependency (Camelot/TableFormer) — it
# already ships both ruled-line detection (default strategy="lines") and a
# pdfplumber-equivalent borderless/text-alignment detector
# (strategy="text"), so a first cut needs no new library.
#
# Confirmed this session by reading find_tables()'s own source
# (pymupdf/table.py): by DEFAULT it sets use_layout=True, which first runs
# MuPDF's own geometry-based table-REGION detector and gates results to
# regions that detector recognizes — if that layout pass finds zero table
# regions on a page, find_tables() returns an EMPTY TableFinder immediately,
# without even trying the line/text strategies underneath. A real
# ruled-line or aligned-text table can therefore still return nothing from
# the plain default call. The fallback chain below exists specifically for
# that: try the default first (cheapest, most reliable when it works), then
# strategy="text" (same api, borderless detection), then finally
# use_layout=False + strategy="text" (forces full geometric detection,
# bypassing the layout gate entirely) as a last resort.
_TABLE_STRATEGY_ATTEMPTS = (
    ("default", {}),
    ("text", {"strategy": "text"}),
    ("text_no_layout", {"use_layout": False, "strategy": "text"}),
)


def find_page_tables(page) -> list:
    """Returns a list of pymupdf Table objects (each with .bbox, .extract(),
    .to_markdown()) detected on the page, trying the strategy fallback
    chain above in order and stopping at the first one that finds anything.
    Needs validation against REAL gold-set documents before the fallback
    order is trusted as final (see tests/fixtures/pdf_gold_set/README.md) —
    logs which strategy actually produced a result so real-world fallback
    frequency becomes visible once that happens. Never raises; returns []
    on any failure (caller's existing flowing-text extraction is
    unaffected — this is an enhancement layered on top, not a replacement
    for the base text extraction path)."""
    try:
        for strategy_name, kwargs in _TABLE_STRATEGY_ATTEMPTS:
            tf = page.find_tables(**kwargs)
            if tf.tables:
                logger.info(f"[PdfTextLayer] find_page_tables: {len(tf.tables)} table(s) found via strategy={strategy_name}")
                return tf.tables
        return []
    except Exception as e:
        logger.warning(f"[PdfTextLayer] Table detection failed (non-fatal, page falls back to flowing text only): {e}")
        return []


_TABLE_BLOCK_OVERLAP_THRESHOLD = 0.5  # a text block counts as "covered by this table" once >=50% of its own area sits inside the table's bbox


def _block_overlaps_table(block_bbox, table_bbox) -> bool:
    """Area-overlap ratio relative to the BLOCK's own area (not the
    table's) — a block mostly inside a table counts as covered even if the
    table itself is much larger and covers many other blocks too."""
    if not block_bbox or not table_bbox:
        return False
    bx0, by0, bx1, by1 = block_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    ix0, iy0 = max(bx0, tx0), max(by0, ty0)
    ix1, iy1 = min(bx1, tx1), min(by1, ty1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter_area = (ix1 - ix0) * (iy1 - iy0)
    block_area = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return (inter_area / block_area) >= _TABLE_BLOCK_OVERLAP_THRESHOLD


# 2026-11 (PDF pipeline overhaul, lower-priority multi-column fix). Ported
# from PyMuPDF's own documented community recipe
# (github.com/pymupdf/PyMuPDF-Utilities, text-extraction/multi_column.py) —
# not shipped as an importable module in this pymupdf version, so the
# algorithm is reproduced here verbatim (renamed fitz.* -> pymupdf.* to match
# this codebase's own import convention; no other logic changes). Plain
# page.get_text() with no column awareness interleaves lines ACROSS columns
# in native PDF reading order (left column's line 1, right column's line 1,
# left's line 2, ...) whenever the underlying content stream was written
# column-by-row rather than column-by-column — silently scrambling reading
# order on any genuinely 2+ column layout (common in government gazette-
# style regulatory PDFs). Verified empirically this session against a
# synthetic 2-column fixture: returns exactly 1 box (spanning the full
# width) for a single-column page — zero behavior change there — and 2
# separate boxes, left-column-first, for a genuine 2-column page.
def column_boxes(page, footer_margin: int = 50, header_margin: int = 50, no_image_text: bool = True) -> list:
    """Determine bboxes which wrap a column. Returns a list of pymupdf.IRect,
    sorted ascending by top then left coordinate — the caller extracts each
    box's text via page.get_text(clip=box, sort=True) and joins them in that
    order. See module-level comment above for provenance."""
    paths = page.get_drawings()
    bboxes = []
    path_rects = []
    img_bboxes = []
    vert_bboxes = []

    clip = +page.rect
    clip.y1 -= footer_margin
    clip.y0 += header_margin

    def can_extend(temp, bb, bboxlist):
        for b in bboxlist:
            if not intersects_bboxes(temp, vert_bboxes) and (
                b is None or b == bb or (temp & b).is_empty
            ):
                continue
            return False
        return True

    def in_bbox(bb, bboxes):
        for i, bbox in enumerate(bboxes):
            if bb in bbox:
                return i + 1
        return 0

    def intersects_bboxes(bb, bboxes):
        for bbox in bboxes:
            if not (bb & bbox).is_empty:
                return True
        return False

    def extend_right(bboxes, width, path_bboxes, vert_bboxes, img_bboxes):
        for i, bb in enumerate(bboxes):
            if in_bbox(bb, path_bboxes):
                continue
            if in_bbox(bb, img_bboxes):
                continue
            temp = +bb
            temp.x1 = width
            if intersects_bboxes(temp, path_bboxes + vert_bboxes + img_bboxes):
                continue
            check = can_extend(temp, bb, bboxes)
            if check:
                bboxes[i] = temp
        return [b for b in bboxes if b is not None]

    def clean_nblocks(nblocks):
        blen = len(nblocks)
        if blen < 2:
            return nblocks
        start = blen - 1
        for i in range(start, -1, -1):
            bb1 = nblocks[i]
            bb0 = nblocks[i - 1]
            if bb0 == bb1:
                del nblocks[i]

        y1 = nblocks[0].y1
        i0 = 0
        i1 = -1
        for i in range(1, len(nblocks)):
            b1 = nblocks[i]
            if abs(b1.y1 - y1) > 10:
                if i1 > i0:
                    nblocks[i0 : i1 + 1] = sorted(nblocks[i0 : i1 + 1], key=lambda b: b.x0)
                y1 = b1.y1
                i0 = i
            i1 = i
        if i1 > i0:
            nblocks[i0 : i1 + 1] = sorted(nblocks[i0 : i1 + 1], key=lambda b: b.x0)
        return nblocks

    for p in paths:
        path_rects.append(p["rect"].irect)
    path_bboxes = path_rects
    path_bboxes.sort(key=lambda b: (b.y0, b.x0))

    for item in page.get_images():
        img_bboxes.extend(page.get_image_rects(item[0]))

    blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT, clip=clip)["blocks"]

    for b in blocks:
        bbox = pymupdf.IRect(b["bbox"])
        if no_image_text and in_bbox(bbox, img_bboxes):
            continue
        line0 = b["lines"][0]
        if line0["dir"] != (1, 0):
            vert_bboxes.append(bbox)
            continue
        srect = pymupdf.EMPTY_IRECT()
        for line in b["lines"]:
            lbbox = pymupdf.IRect(line["bbox"])
            text = "".join([s["text"].strip() for s in line["spans"]])
            if len(text) > 1:
                srect |= lbbox
        bbox = +srect
        if not bbox.is_empty:
            bboxes.append(bbox)

    bboxes.sort(key=lambda k: (in_bbox(k, path_bboxes), k.y0, k.x0))
    bboxes = extend_right(bboxes, int(page.rect.width), path_bboxes, vert_bboxes, img_bboxes)

    if bboxes == []:
        return []

    nblocks = [bboxes[0]]
    bboxes = bboxes[1:]

    for i, bb in enumerate(bboxes):
        check = False
        for j in range(len(nblocks)):
            nbb = nblocks[j]
            if bb is None or nbb.x1 < bb.x0 or bb.x1 < nbb.x0:
                continue
            if in_bbox(nbb, path_bboxes) != in_bbox(bb, path_bboxes):
                continue
            temp = bb | nbb
            check = can_extend(temp, nbb, nblocks)
            if check:
                break
        if not check:
            nblocks.append(bb)
            j = len(nblocks) - 1
            temp = nblocks[j]
        check = can_extend(temp, bb, bboxes)
        if not check:
            nblocks.append(bb)
        else:
            nblocks[j] = temp
        bboxes[i] = None

    nblocks = clean_nblocks(nblocks)
    return nblocks


def extract_page_text_reading_order(page) -> str:
    """Wraps column_boxes() above: for a genuinely single-column page (0 or
    1 box detected), behaves identically to plain page.get_text() — the
    overwhelming majority of pages, zero added cost/risk. For 2+ detected
    columns, extracts each column's text separately (page.get_text(clip=box,
    sort=True)) and joins them in column_boxes()'s already-reading-order-
    sorted sequence, instead of letting plain get_text() interleave lines
    across columns. Never raises; any failure (including a pathological
    page that confuses the column-detection heuristic) falls back to plain
    page.get_text() — the pre-existing behavior, not a regression."""
    try:
        boxes = column_boxes(page)
        if len(boxes) <= 1:
            return page.get_text()
        return "\n\n".join(page.get_text(clip=box, sort=True) for box in boxes)
    except Exception as e:
        logger.warning(f"[PdfTextLayer] Multi-column reading-order extraction failed, falling back to plain text extraction: {e}")
        return page.get_text()


def extract_page_text_with_tables(page) -> str:
    """Reconstructs the page's text block-by-block (same
    page.get_text("dict")["blocks"] structure _text_coverage_ratio()/
    classify_page_text_blocks() above already walk), SPLICING each detected
    table's .to_markdown() output in place of the flowing text that
    page.get_text() would otherwise have produced for that same region —
    never both, which would duplicate every cell's content once as prose
    and once as a table row. A table is emitted exactly once, at the
    position of the first text block it covers; every other block the same
    table covers is skipped (already represented by that one markdown
    block). Falls back to plain page.get_text() when no tables are
    detected — the common case, so this adds no per-page cost for the
    large majority of pages that don't have one. Never raises; a failure in
    table reconstruction falls back to plain page.get_text() rather than
    losing the page's text entirely.

    2026-11 note: the no-table branch uses extract_page_text_reading_order()
    (multi-column-aware), not plain page.get_text() — see that function's
    own docstring. Deliberately NOT combined with the table-splicing branch
    below: reconciling column boundaries against table bboxes at the same
    time is real added complexity for a rare combination (a multi-column
    page that ALSO has a detected table), not worth the correctness risk
    for this lower-priority fix — a page with both still gets correct
    table splicing, just without multi-column reordering of its
    surrounding flowing text."""
    try:
        tables = find_page_tables(page)
        if not tables:
            return extract_page_text_reading_order(page)

        blocks = page.get_text("dict").get("blocks", [])
        emitted_table_indices: set[int] = set()
        parts: list[str] = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox")
            covering_table_idx = next(
                (i for i, t in enumerate(tables) if _block_overlaps_table(bbox, t.bbox)), None,
            )
            if covering_table_idx is not None:
                if covering_table_idx not in emitted_table_indices:
                    parts.append(tables[covering_table_idx].to_markdown(clean=False, fill_empty=True))
                    emitted_table_indices.add(covering_table_idx)
                continue
            block_text = "".join(
                span.get("text", "") for line in block.get("lines", []) for span in line.get("spans", [])
            )
            if block_text:
                parts.append(block_text)
        return "\n".join(parts)
    except Exception as e:
        logger.warning(f"[PdfTextLayer] Table-aware text reconstruction failed, falling back to plain text extraction: {e}")
        return page.get_text()


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


def _page_needs_ocr_for_visual_content(page, cost_accumulator: Optional[CostAccumulator] = None, cost_guard=None) -> bool:
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
            cost_guard=cost_guard,
        )
        result = parse_leading_json(resp.choices[0].message.content or "")
        return bool(result.get("has_meaningful_visual_content", True))
    except Exception as e:
        logger.warning(f"[PdfTextLayer] Vision image-check failed, defaulting to OCR fallback (safe direction): {e}")
        return True


# 2026-11 (PDF pipeline overhaul, gap #1 — accepting native text used to
# mean skipping OCR for the WHOLE page, silently dropping any embedded
# scanned table/stamp/image on an otherwise-good text page). Matches
# Docling's bitmap_area_threshold default (arXiv:2408.09869) — a bitmap
# placed occurrence covering >=5% of the page area gets OCR'd ADDITIVELY
# (appended to native text), never as a replacement for it. Deliberately
# kept separate from extract_page_native_text()'s accept/reject decision:
# this module has no OCR capability of its own (no import cycle with
# service/pdf_large_extraction.py, which owns _run_typhoon) — detection and
# region-cropping live here as pure functions; the caller (_extract_one_page)
# does the actual OCR call and appends the result.
_BITMAP_AREA_THRESHOLD = 0.05


def find_embedded_image_regions(pdf_bytes: bytes, page_num: int) -> list[tuple[float, float, float, float]]:
    """Returns the bboxes of every embedded raster image PLACED on this page
    (page.get_image_info(xrefs=True) — one entry per placed occurrence, not
    per unique XObject, so a reused logo stamped 3 times yields 3 bboxes)
    whose area covers >=_BITMAP_AREA_THRESHOLD of the page. Blind to
    vector-drawn diagrams by construction (get_image_info only sees raster
    XObjects) — that gap is covered by the existing
    _page_needs_ocr_for_visual_content vision-check above, kept as a
    complementary mechanism rather than retired. Never raises; an
    unexpected failure returns [] (no regions found), same fail-direction
    as the rest of this module's detection helpers."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            page = doc[page_num - 1]
            page_area = page.rect.width * page.rect.height
            if page_area <= 0:
                return []
            regions: list[tuple[float, float, float, float]] = []
            for info in page.get_image_info(xrefs=True):
                x0, y0, x1, y1 = info["bbox"]
                area_ratio = max(0.0, (x1 - x0) * (y1 - y0)) / page_area
                if area_ratio >= _BITMAP_AREA_THRESHOLD:
                    regions.append((x0, y0, x1, y1))
            return regions
        finally:
            doc.close()
    except Exception as e:
        logger.warning(f"[PdfTextLayer] Page {page_num}: embedded-image region detection failed (non-fatal, treated as no regions found): {e}")
        return []


def render_region_to_png_bytes(pdf_bytes: bytes, page_num: int, bbox: tuple[float, float, float, float], dpi: int = 200) -> bytes:
    """Crops and renders just the given bbox (page.get_pixmap(clip=...)) —
    not the whole page — so an OCR call on a detected embedded-image region
    is cheap (small crop) and naturally scoped to that region's content,
    unlike the full-page render _render_page_to_png_bytes in
    service/pdf_large_extraction.py produces."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap(clip=pymupdf.Rect(bbox), dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


_SUPPLEMENTARY_ANNOT_TYPES = (pymupdf.PDF_ANNOT_FREE_TEXT, pymupdf.PDF_ANNOT_STAMP)


def extract_page_supplementary_text(pdf_bytes: bytes, page_num: int) -> str:
    """2026-11 (PDF pipeline overhaul, gap #2): page.get_text() excludes
    form-field/widget values and most annotation types BY DESIGN — PyMuPDF's
    own FAQ states annotations/widgets are "not part of the page". On a
    filled-in government form, this means every value the applicant actually
    entered is silently absent from extraction while the blank template text
    passes every validity check untouched — confirmed empirically this
    session: get_text() on a synthetic page with a text widget returned only
    the surrounding body text, never the widget's field_value.

    Deliberately UNCONDITIONAL and purely additive — unlike
    extract_page_native_text()'s branch decision (native text vs OCR), this
    has nothing to do with which extraction path ran; a filled form field's
    value isn't recoverable by OCR either if the page otherwise went native,
    nor by native text if the page otherwise went OCR, so this must run on
    every page regardless of which branch handled the base text. Caller
    appends the (possibly empty) result to whatever text came from either
    branch. Never raises — an extraction failure here must never block the
    page's base text from being used."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            page = doc[page_num - 1]
            lines: list[str] = []

            for widget in page.widgets() or []:
                value = (widget.field_value or "").strip() if isinstance(widget.field_value, str) else widget.field_value
                if not value:
                    continue
                name = (widget.field_name or "").strip()
                lines.append(f"{name}: {value}" if name else str(value))

            for annot in page.annots() or []:
                if annot.type[0] not in _SUPPLEMENTARY_ANNOT_TYPES:
                    continue
                annot_text = (annot.get_text() or "").strip()
                if annot_text:
                    lines.append(annot_text)

            return "\n".join(lines)
        finally:
            doc.close()
    except Exception as e:
        logger.warning(f"[PdfTextLayer] Page {page_num}: widget/annotation extraction failed (non-fatal, base text unaffected): {e}")
        return ""


_ANCHOR_TEXT_MAX_CHARS = 800  # "short reference-context" per olmOCR's document-anchoring technique — a hint, not a full transcript


def extract_raw_page_text_best_effort(pdf_bytes: bytes, page_num: int) -> str:
    """2026-11 (PDF pipeline overhaul, document-anchoring add-on). Returns
    page.get_text() with NO validation at all — deliberately distinct from
    extract_page_native_text() above, which discards its own raw text the
    moment it decides to reject a page (returns None, not the text that
    failed validation). This exists purely so a caller that's about to run
    OCR anyway can still hand the vision model whatever partial/possibly-
    corrupted native text existed as grounding context (olmOCR's measured
    hallucination-reduction technique: a model given *some* reference text,
    even an imperfect one, hallucinates less than one given a bare image).
    Truncated to _ANCHOR_TEXT_MAX_CHARS — a short hint, not a replacement
    transcript. Never raises; any failure just means no anchor text, which
    is exactly the pre-existing behavior without this function."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            text = doc[page_num - 1].get_text() or ""
            return text.strip()[:_ANCHOR_TEXT_MAX_CHARS]
        finally:
            doc.close()
    except Exception:
        return ""


def extract_page_native_text(pdf_bytes: bytes, page_num: int, cost_accumulator: Optional[CostAccumulator] = None, cost_guard=None) -> str | None:
    """Returns None (caller falls back to OCR) for a scanned/image-only page
    with no meaningful embedded text, a corrupted character sitting right
    next to a high-value digit per _salient_regions_look_valid (positional
    signal, kept active — see 2026-11 note below), a text layer where ANY
    individual block fails the per-block composite validity check below
    (ratio-based signal: in-script Thai mojibake or corruption concentrated
    in one block — see classify_page_text_blocks()'s docstring for why this
    replaced the old page-wide character-range check 2026-11), a page whose
    rendered content includes a meaningful diagram/chart/image/status-stamp
    per the Approach A+B check above (OCR is vision-capable and will
    describe it; plain text extraction cannot), or if extraction itself
    errors — never raises.

    2026-11 note: the per-block composite check does NOT subsume
    _salient_regions_look_valid — confirmed live this session when a test
    fixture (one corrupted character next to a digit, in an otherwise-clean
    ~75-char English block) passed both of the new block-level ratio gates
    comfortably (1 bad char in 75 is a small ratio regardless of whether
    the denominator is "page" or "block") while still representing exactly
    the kind of high-value corruption the digit-proximity check exists to
    catch. The two checks target different corruption SHAPES (positional
    vs. concentration-ratio) and both stay active, independently.

    2026-11 note on per-block scope: rejects the WHOLE page if any one block fails,
    rather than OCR'ing and splicing back just the bad block — block-level
    splice-repair (Marker's "surgical block-level repair") was considered
    and deliberately deferred: this module has no OCR capability of its own
    (see find_embedded_image_regions's docstring on the import-cycle
    constraint), and splicing OCR'd text back into a page reconstructed
    from get_text()'s block ordering carries real reading-order-correctness
    risk this codebase's own fixtures already warn about (single-line
    insert_text's documented truncation quirks). Rejecting the whole page
    on any bad block still fully fixes the stated gap (page-wide averaging
    hiding corruption localized to one block) — it just costs a full-page
    OCR call instead of a cheaper region-only one when only one block out
    of many is actually bad."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            page = doc[page_num - 1]
            text = page.get_text()
            if len(text.strip()) < _MIN_PAGE_TEXT_LAYER_CHARS:
                return None
            # 2026-11: _salient_regions_look_valid kept ACTIVE here (not
            # retired) — found live this session that it catches a
            # genuinely different, complementary failure mode the new
            # per-block check below does NOT subsume: a single corrupted
            # character sitting immediately next to a high-value digit
            # (a fee amount, a form code) is a POSITIONAL signal, not a
            # ratio one — one bad char in a normal-sized block still scores
            # comfortably above both of the new check's ratio-based
            # thresholds, since ratio-based checks are only sensitive to
            # CONCENTRATION of corruption, not WHERE it sits. The two
            # checks target different corruption shapes and both stay live.
            if not _salient_regions_look_valid(text):
                logger.info(f"[PdfTextLayer] Page {page_num}: corrupted character found right next to a number (fee amount/date/form/license code) — falling back to OCR")
                return None
            blocks = classify_page_text_blocks(page)
            invalid_blocks = [b for b in blocks if not b["valid"]]
            if invalid_blocks:
                worst = min(invalid_blocks, key=lambda b: b["linguistic_score"])
                logger.info(f"[PdfTextLayer] Page {page_num}: text block failed composite validity check (unicode={worst['unicode_score']:.2f}, linguistic={worst['linguistic_score']:.2f}, likely corrupted/mismapped font or scrambled Thai) — falling back to OCR")
                return None
            # 2026-11 (PDF pipeline overhaul, gap #4): validation above runs
            # on the plain flowing-text `text` (unchanged) — only the
            # accepted RETURN value gets table-structure splicing, so
            # tables are still represented as flowing text for validity
            # checks (which is fine; the checks care about character/
            # linguistic validity, not layout) but preserved as real
            # row/column markdown in what actually reaches the document.
            if _text_coverage_ratio(page) >= _TEXT_COVERAGE_SKIP_THRESHOLD:
                return extract_page_text_with_tables(page)
            if _page_needs_ocr_for_visual_content(page, cost_accumulator=cost_accumulator, cost_guard=cost_guard):
                logger.info(f"[PdfTextLayer] Page {page_num}: vision check found meaningful visual content — falling back to OCR so it gets described, not silently dropped")
                return None
            return extract_page_text_with_tables(page)
        finally:
            doc.close()
    except Exception as e:
        logger.warning(f"[PdfTextLayer] Page {page_num}: native text-layer extraction failed, falling back to OCR: {e}")
        return None

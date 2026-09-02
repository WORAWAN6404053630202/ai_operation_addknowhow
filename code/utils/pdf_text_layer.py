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
common cause for Thai-language PDFs specifically."""

from __future__ import annotations

import pymupdf

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


def text_looks_valid(text: str) -> bool:
    """See _EXPECTED_TEXT_RANGES above for the reasoning. Ignores whitespace
    (not a signal either way) and never raises."""
    stripped = "".join(text.split())
    if not stripped:
        return False
    bad = sum(1 for ch in stripped if not any(lo <= ord(ch) <= hi for lo, hi in _EXPECTED_TEXT_RANGES))
    return (bad / len(stripped)) <= _MAX_BAD_CHAR_RATIO


def extract_page_native_text(pdf_bytes: bytes, page_num: int) -> str | None:
    """Returns None (caller falls back to OCR) for a scanned/image-only page
    with no meaningful embedded text, a text layer that fails the
    character-validity check above (likely a corrupted/mismapped font), or
    if extraction itself errors — never raises."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            text = doc[page_num - 1].get_text()
        finally:
            doc.close()
    except Exception as e:
        logger.warning(f"[PdfTextLayer] Page {page_num}: native text-layer extraction failed, falling back to OCR: {e}")
        return None
    if len(text.strip()) < _MIN_PAGE_TEXT_LAYER_CHARS:
        return None
    if not text_looks_valid(text):
        logger.info(f"[PdfTextLayer] Page {page_num}: native text layer present but failed the character-validity check (likely a corrupted/mismapped font) — falling back to OCR")
        return None
    return text

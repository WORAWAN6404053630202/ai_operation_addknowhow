"""Regression tests for native-text-layer image detection in
utils/pdf_text_layer.py (feature/pdf-ingestion, 2026-09).

History: extract_page_native_text() used to decide purely from text length/
validity, with zero awareness that the same page might carry a large
embedded diagram/flowchart — get_text() has no concept of images. A first
fix forced OCR fallback whenever page.get_image_info() found a raster image
covering >=5% of the page area (_page_has_meaningful_image, now removed).
That approach was itself replaced after review surfaced two blind spots: it
was blind to vector-drawn diagrams (page.get_drawings() content is never
raster), and a fixed area threshold is only a proxy for "is this visually
meaningful" rather than actually checking.

Current approach (A+B): a cheap local "text coverage ratio" pre-filter
(_text_coverage_ratio) that only ever SKIPS the paid check for a page
confidently covered in text, plus a real vision-model check
(_page_needs_ocr_for_visual_content, google/gemini-2.5-flash-lite via
OpenRouter) for everything else — the vision call is mocked in these tests,
since hitting the real API isn't appropriate for a unit test. Approach C
(background audit-sampling) was considered and explicitly declined by the
user for this round; not covered here.

Two further gaps closed 2026-09 (see TestSalientRegionValidity and the
"stamp/watermark" case in TestVisionImageCheck below): (1) corruption
localized right next to a number can pass the page-wide 5% validity check,
covered by _salient_regions_look_valid; (2) the vision prompt used to ask
only about diagrams, never about document-status stamps/watermarks even
though it was already looking at the rendered page.
"""
import io
from unittest.mock import MagicMock, patch

import pymupdf
import pytest

from utils.pdf_text_layer import (
    _page_needs_ocr_for_visual_content,
    _salient_regions_look_valid,
    _text_coverage_ratio,
    extract_page_native_text,
)

# Plain ASCII on purpose: pymupdf's default built-in font (used by
# insert_text with no explicit fontfile) can't render Thai glyphs, so real
# Thai text here would round-trip through get_text() as placeholder dots and
# fail text_looks_valid() for a reason unrelated to what this test file
# actually checks (the image-detection logic, which is language-agnostic).
# text_looks_valid() itself already has its own Thai-specific coverage
# elsewhere.
_LONG_TEXT = "This is real page body text content here. " * 5  # comfortably over the 40-char floor


def _make_pdf_page(*, text: str, add_image: bool = False) -> bytes:
    """Builds a single-page PDF with the given text and, optionally, a small
    decorative image placed in a corner — small enough to never push
    _text_coverage_ratio() over the skip threshold on its own, since these
    tests care about the vision-check path, not the coverage-ratio math
    directly (that's covered by test_text_coverage_ratio_* below).

    Uses insert_text at a single point (no wrapping rect) — matches every
    other test in this file. NOTE: this silently drops anything past the
    first line at this font size (confirmed empirically — a 240-char string
    on a 500pt-wide line at fontsize 10 came back as 129 chars), so `text`
    must fit on one line if a test needs its FULL content preserved and
    checked (see TestSalientRegionValidity's short dedicated fixtures below,
    as opposed to _LONG_TEXT-based tests elsewhere that only ever check a
    short prefix substring and don't care about truncation)."""
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((50, 50), text, fontsize=10)
    if add_image:
        px = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 4, 4), False)
        px.set_rect(px.irect, (200, 200, 200))
        png_bytes = px.tobytes("png")
        page.insert_image(pymupdf.Rect(500, 700, 540, 740), stream=png_bytes)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _mock_vision_response(has_content: bool):
    resp = MagicMock()
    resp.choices[0].message.content = f'{{"has_meaningful_visual_content": {"true" if has_content else "false"}}}'
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 10
    return resp


@pytest.mark.unit
class TestTextCoverageRatio:

    def test_dense_text_page_has_high_coverage(self):
        doc = pymupdf.open(stream=_make_pdf_page(text=_LONG_TEXT * 20), filetype="pdf")
        try:
            assert _text_coverage_ratio(doc[0]) > 0.0
        finally:
            doc.close()

    def test_blank_page_has_zero_coverage(self):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        try:
            assert _text_coverage_ratio(page) == 0.0
        finally:
            doc.close()

    def test_full_page_image_does_not_count_as_text_coverage(self):
        """Regression test for a real bug found via live-testing against
        ELICENSE_คู่มือปก-ใบอนุญาตขาย.pdf's page 1: page.get_text('dict')
        returns BOTH text blocks (type 0) AND image blocks (type 1) — summing
        every block's bbox regardless of type let a full-page image score
        100% 'text coverage', which would wrongly trigger Approach A's
        skip-the-vision-check fast path on a page that is pure image."""
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        px = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 4, 4), False)
        px.set_rect(px.irect, (200, 200, 200))
        png_bytes = px.tobytes("png")
        page.insert_image(pymupdf.Rect(0, 0, 600, 800), stream=png_bytes)  # covers 100% of the page
        try:
            assert _text_coverage_ratio(page) == 0.0
        finally:
            doc.close()


@pytest.mark.unit
class TestVisionImageCheck:

    def test_vision_check_true_forces_ocr_fallback(self):
        pdf_bytes = _make_pdf_page(text=_LONG_TEXT)
        with patch("utils.pdf_text_layer.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_vision_response(has_content=True)
            with patch("utils.pdf_text_layer._text_coverage_ratio", return_value=0.1):
                result = extract_page_native_text(pdf_bytes, 1)
        assert result is None
        mock_client.chat.completions.create.assert_called_once()

    def test_vision_check_false_keeps_native_path(self):
        pdf_bytes = _make_pdf_page(text=_LONG_TEXT)
        with patch("utils.pdf_text_layer.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_vision_response(has_content=False)
            with patch("utils.pdf_text_layer._text_coverage_ratio", return_value=0.1):
                result = extract_page_native_text(pdf_bytes, 1)
        assert result is not None
        assert "real page body text" in result

    def test_vision_check_exception_fails_safe_to_ocr(self):
        """The dangerous direction is silently trusting an unchecked page —
        any error talking to the vision model must force OCR, not skip it."""
        doc = pymupdf.open(stream=_make_pdf_page(text=_LONG_TEXT), filetype="pdf")
        try:
            with patch("utils.pdf_text_layer.OpenAI", side_effect=RuntimeError("network down")):
                assert _page_needs_ocr_for_visual_content(doc[0]) is True
        finally:
            doc.close()

    def test_high_text_coverage_skips_vision_check_entirely(self):
        """Approach A: a page confidently covered in text never even calls
        the paid vision check — this is the whole point of the pre-filter."""
        pdf_bytes = _make_pdf_page(text=_LONG_TEXT)
        with patch("utils.pdf_text_layer.OpenAI") as mock_openai_cls:
            with patch("utils.pdf_text_layer._text_coverage_ratio", return_value=0.95):
                result = extract_page_native_text(pdf_bytes, 1)
        assert result is not None
        mock_openai_cls.assert_not_called()

    def test_full_page_image_with_caption_text_still_calls_vision_check(self):
        """End-to-end regression test for the same bug as
        test_full_page_image_does_not_count_as_text_coverage above: a page
        with a full-page image PLUS enough caption text to clear the
        MIN_PAGE_TEXT_LAYER_CHARS floor must still reach the real vision
        check, not get fast-skipped by Approach A just because the (buggy)
        coverage math once counted the image itself as 'text'."""
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 50), _LONG_TEXT, fontsize=10)
        px = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 4, 4), False)
        px.set_rect(px.irect, (200, 200, 200))
        png_bytes = px.tobytes("png")
        page.insert_image(pymupdf.Rect(0, 0, 600, 800), stream=png_bytes)  # covers 100% of the page
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        pdf_bytes = buf.getvalue()

        with patch("utils.pdf_text_layer.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_vision_response(has_content=True)
            result = extract_page_native_text(pdf_bytes, 1)
        assert result is None
        mock_client.chat.completions.create.assert_called_once()

    def test_prompt_asks_about_status_stamps_not_just_diagrams(self):
        """A status stamp/watermark ('DRAFT', 'APPROVED', an official seal) is
        not a diagram — the vision model was never actually asked about it
        even though it was already looking at the rendered page. This checks
        the prompt itself asks the right question, not just that a True
        response forces OCR (that part is identical to the diagram case and
        already covered above)."""
        from utils.pdf_text_layer import _VISION_IMAGE_CHECK_PROMPT

        assert "stamp" in _VISION_IMAGE_CHECK_PROMPT.lower()
        assert "watermark" in _VISION_IMAGE_CHECK_PROMPT.lower()

    def test_stamp_detection_true_forces_ocr_even_with_no_embedded_image(self):
        """A watermark/stamp is often vector-drawn text or a semi-transparent
        overlay, not necessarily an image block at all — so this deliberately
        builds a page with NO inserted image, only text, to prove the OCR
        fallback doesn't depend on an image block existing; it depends
        entirely on what the vision model reports back."""
        pdf_bytes = _make_pdf_page(text=_LONG_TEXT)
        with patch("utils.pdf_text_layer.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_vision_response(has_content=True)
            with patch("utils.pdf_text_layer._text_coverage_ratio", return_value=0.1):
                result = extract_page_native_text(pdf_bytes, 1)
        assert result is None


@pytest.mark.unit
class TestSalientRegionValidity:
    """_salient_regions_look_valid() catches corruption LOCALIZED right next
    to a number (fee amount/date/form/license code) — the exact case
    text_looks_valid()'s page-wide 5% average can miss, since one corrupted
    character in an otherwise-clean 400+ character page is nowhere near 5%
    of the page, but can sit right on top of the one number that matters."""

    def test_clean_text_with_ordinary_numbers_is_valid(self):
        assert _salient_regions_look_valid(_LONG_TEXT + " see page 5, section 12") is True

    def test_corruption_immediately_after_a_digit_is_invalid(self):
        """The real failure mode: a fee amount like '3,060 บาท' with one
        digit replaced by a PUA character — 'text_looks_valid()' would still
        pass this (1 bad char in 400+), but the amount itself is now wrong."""
        text = _LONG_TEXT + " ค่าธรรมเนียม 3,06 บาท ต้องชำระภายในกำหนด"
        assert _salient_regions_look_valid(text) is False

    def test_corruption_immediately_before_a_digit_is_invalid(self):
        text = _LONG_TEXT + " เลขที่ใบอนุญาต 67L0006170 ออกให้เมื่อวันที่"
        assert _salient_regions_look_valid(text) is False

    def test_corruption_far_from_any_digit_is_valid(self):
        """Same single bad character, but nowhere near a number — this is
        exactly the kind of corruption text_looks_valid()'s page-wide ratio
        is meant to catch (and would, at a high enough concentration); a
        single isolated instance far from any digit is not the failure mode
        this specific check exists for."""
        text = _LONG_TEXT + " some ordinary prose with a stray glyph  in the middle of a sentence, no numbers nearby at all here"
        assert _salient_regions_look_valid(text) is True

    def test_extract_page_native_text_falls_back_when_salient_region_corrupted(self):
        """End-to-end: this must be caught before the vision check even
        runs — it's a text-layer-corruption problem, not a visual-content
        problem, so no API call should happen.

        Uses a short ASCII-plus-one-PUA-char fixture (not _LONG_TEXT + a
        Thai suffix) empirically confirmed to survive _make_pdf_page's
        single-line insert_text round-trip intact: the PUA character
        (unencodable in the default built-in font) comes back as a
        literal '·' placeholder, itself correctly out-of-range per
        _EXPECTED_TEXT_RANGES, sitting right after the digit '6' — exactly
        the failure mode this check exists for. 75 chars total, ~1.3% bad-
        char ratio, comfortably under text_looks_valid()'s 5% page-wide
        threshold while still failing this more targeted check."""
        garbled = "Fee due: 3,06 baht by the deadline for this filing period, please pay soon"
        pdf_bytes = _make_pdf_page(text=garbled)
        with patch("utils.pdf_text_layer.OpenAI") as mock_openai_cls:
            result = extract_page_native_text(pdf_bytes, 1)
        assert result is None
        mock_openai_cls.assert_not_called()

@pytest.mark.unit
class TestPageNumberReferenceExclusion:
    """Real false positive found live against ELICENSE_คู่มือปก-ใบอนุญาตขาย.pdf
    (2026-09): 43 of 66 pages got flagged by _salient_regions_look_valid,
    but ALL 43 turned out to be the exact same footer icon glyph (a single
    PUA codepoint, U+F09E) sitting immediately after 'หน้า <N>' (Thai for
    'page N') — zero real corruption among them. Excluded via
    _PAGE_NUMBER_REF_RE rather than accepting the cost of routing ~65% of
    that real document to full OCR for no actual benefit."""

    def test_corruption_right_after_a_page_number_reference_is_excluded(self):
        text = _LONG_TEXT + " หน้า 6 " + "\ue000" + " เข้าสู่ระบบบริการธุรกิจ"
        assert _salient_regions_look_valid(text) is True

    def test_corruption_right_after_a_page_number_reference_no_space(self):
        text = _LONG_TEXT + " หน้า 12" + "\ue000" + " รายละเอียดต่อไปนี้"
        assert _salient_regions_look_valid(text) is True

    def test_corruption_near_an_unrelated_digit_still_flagged(self):
        """The exclusion must stay narrow: a corrupted character near a
        digit that is NOT part of a 'หน้า <N>' reference (e.g. a fee amount
        elsewhere on the same page) must still be caught — this guards
        against the exclusion becoming so broad it defeats the whole check."""
        text = _LONG_TEXT + " หน้า 6 ปกติ ค่าธรรมเนียม 3,06" + "\ue000" + " บาท"
        assert _salient_regions_look_valid(text) is False

    def test_extract_page_native_text_keeps_native_path_for_page_number_footer(self):
        """End-to-end: a page whose only 'corruption' is this footer icon
        must NOT be forced to OCR — it should proceed to the coverage/vision
        check exactly like any other clean page."""
        garbled = "Page 6 continues with the rest of the filing here"
        pdf_bytes = _make_pdf_page(text=garbled)
        with patch("utils.pdf_text_layer.OpenAI") as mock_openai_cls:
            with patch("utils.pdf_text_layer._text_coverage_ratio", return_value=0.95):
                result = extract_page_native_text(pdf_bytes, 1)
        assert result is not None
        mock_openai_cls.assert_not_called()


@pytest.mark.unit
class TestExtractPageNativeTextPreExistingBehavior:
    """Pre-existing behavior (predates the A+B rewrite) must still hold."""

    def test_short_text_falls_back_to_ocr_without_calling_vision(self):
        pdf_bytes = _make_pdf_page(text="short")
        with patch("utils.pdf_text_layer.OpenAI") as mock_openai_cls:
            result = extract_page_native_text(pdf_bytes, 1)
        assert result is None
        mock_openai_cls.assert_not_called()

    def test_invalid_characters_fall_back_to_ocr_without_calling_vision(self):
        garbled = "" * 10
        pdf_bytes = _make_pdf_page(text=garbled)
        with patch("utils.pdf_text_layer.OpenAI") as mock_openai_cls:
            result = extract_page_native_text(pdf_bytes, 1)
        assert result is None
        mock_openai_cls.assert_not_called()

"""Tests for the additive region-OCR mechanism added across
utils/pdf_text_layer.py (find_embedded_image_regions, render_region_to_png_bytes)
and service/pdf_large_extraction.py's _extract_one_page (2026-11,
PDF-extraction-pipeline overhaul, gap #1).

Background: accepting a page's native text used to mean skipping OCR for
the WHOLE page — silently dropping any scanned table/stamp/image embedded
alongside otherwise-good text, since page.get_text() has no concept of
images. Docling's pattern (arXiv:2408.09869, bitmap_area_threshold=0.05)
is additive: detect embedded raster regions independently of the text
layer's accept/reject decision, OCR just the region, append — never
replace. See /Users/w.worawan/.claude/plans/distributed-noodling-ladybug.md."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from pdf_fixtures import make_page_with_embedded_image  # noqa: E402
from utils.pdf_text_layer import find_embedded_image_regions, render_region_to_png_bytes  # noqa: E402

_LONG_TEXT = "This is real page body text content here. " * 10  # well over the 40-char floor and 70% coverage-skip


@pytest.mark.unit
class TestFindEmbeddedImageRegions:
    def test_image_at_or_above_threshold_is_detected(self):
        # 200x200 on a 600x800 page = 40000/480000 = 0.0833, above the 0.05 threshold
        pdf_bytes = make_page_with_embedded_image(text=_LONG_TEXT, image_bbox=(250, 250, 450, 450))
        regions = find_embedded_image_regions(pdf_bytes, 1)
        assert len(regions) == 1
        x0, y0, x1, y1 = regions[0]
        assert (x0, y0, x1, y1) == pytest.approx((250, 250, 450, 450))

    def test_image_below_threshold_is_not_detected(self):
        # 40x40 on a 600x800 page = 1600/480000 = 0.0033, well under threshold
        pdf_bytes = make_page_with_embedded_image(text=_LONG_TEXT, image_bbox=(500, 700, 540, 740))
        regions = find_embedded_image_regions(pdf_bytes, 1)
        assert regions == []

    def test_page_with_no_images_returns_empty_list(self):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 50), _LONG_TEXT, fontsize=10)
        import io
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        regions = find_embedded_image_regions(buf.getvalue(), 1)
        assert regions == []

    def test_malformed_bytes_fail_safe_to_empty_list(self):
        assert find_embedded_image_regions(b"not a pdf", 1) == []


@pytest.mark.unit
class TestRenderRegionToPngBytes:
    def test_crops_to_the_given_bbox_not_the_whole_page(self):
        pdf_bytes = make_page_with_embedded_image(text=_LONG_TEXT, image_bbox=(250, 250, 450, 450))
        full_page_pix_size = len(render_region_to_png_bytes(pdf_bytes, 1, (0, 0, 600, 800), dpi=72))
        region_pix_size = len(render_region_to_png_bytes(pdf_bytes, 1, (250, 250, 450, 450), dpi=72))
        # A real crop produces meaningfully less image data than the full page at the same dpi.
        assert region_pix_size < full_page_pix_size

    def test_returns_valid_png_bytes(self):
        pdf_bytes = make_page_with_embedded_image(text=_LONG_TEXT, image_bbox=(250, 250, 450, 450))
        png_bytes = render_region_to_png_bytes(pdf_bytes, 1, (250, 250, 450, 450))
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
class TestExtractOnePageAdditiveRegionOcr:
    """End-to-end: _extract_one_page must APPEND region-OCR text to accepted
    native text, never replace it, and must call the region-cropped OCR
    (not the full-page one) for this path."""

    def test_region_ocr_text_is_appended_to_native_text(self):
        from service.pdf_large_extraction import _extract_one_page

        pdf_bytes = make_page_with_embedded_image(text=_LONG_TEXT, image_bbox=(250, 250, 450, 450))
        with patch("utils.pdf_text_layer._text_coverage_ratio", return_value=0.95), \
             patch("service.pdf_large_extraction._run_typhoon", return_value="ตารางค่าธรรมเนียมในรูปภาพนี้") as mock_typhoon:
            result = _extract_one_page(pdf_bytes, 1)
        assert "This is real page body text" in result["typhoon_markdown"]
        assert "ตารางค่าธรรมเนียมในรูปภาพนี้" in result["typhoon_markdown"]
        mock_typhoon.assert_called_once()

    def test_no_region_ocr_call_when_image_below_threshold(self):
        from service.pdf_large_extraction import _extract_one_page

        pdf_bytes = make_page_with_embedded_image(text=_LONG_TEXT, image_bbox=(500, 700, 540, 740))
        with patch("utils.pdf_text_layer._text_coverage_ratio", return_value=0.95), \
             patch("service.pdf_large_extraction._run_typhoon") as mock_typhoon:
            result = _extract_one_page(pdf_bytes, 1)
        assert "This is real page body text" in result["typhoon_markdown"]
        mock_typhoon.assert_not_called()

    def test_region_ocr_failure_does_not_lose_native_text(self):
        from service.pdf_large_extraction import _extract_one_page

        pdf_bytes = make_page_with_embedded_image(text=_LONG_TEXT, image_bbox=(250, 250, 450, 450))
        with patch("utils.pdf_text_layer._text_coverage_ratio", return_value=0.95), \
             patch("service.pdf_large_extraction._run_typhoon", side_effect=RuntimeError("OCR provider down")):
            result = _extract_one_page(pdf_bytes, 1)
        assert "This is real page body text" in result["typhoon_markdown"]

    def test_empty_region_ocr_result_appends_nothing(self):
        from service.pdf_large_extraction import _extract_one_page

        pdf_bytes = make_page_with_embedded_image(text=_LONG_TEXT, image_bbox=(250, 250, 450, 450))
        with patch("utils.pdf_text_layer._text_coverage_ratio", return_value=0.95), \
             patch("service.pdf_large_extraction._run_typhoon", return_value=""):
            result = _extract_one_page(pdf_bytes, 1)
        assert "[รูปภาพ/ตารางที่แนบในหน้านี้]" not in result["typhoon_markdown"]

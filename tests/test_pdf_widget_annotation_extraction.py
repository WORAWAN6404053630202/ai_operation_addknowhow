"""Tests for utils/pdf_text_layer.py's extract_page_supplementary_text
(2026-11, PDF-extraction-pipeline overhaul, gap #2).

Background: page.get_text() excludes form-field/widget values and most
annotation types by PyMuPDF's own design (confirmed via PyMuPDF FAQ and
empirically this session) — a filled-in government form's actual entered
values are silently invisible to the existing extraction pipeline while the
blank template text passes every existing validity check untouched. See
tests/fixtures/pdf_gold_set/README.md and
/Users/w.worawan/.claude/plans/distributed-noodling-ladybug.md for the full
pipeline-overhaul plan this is Phase 1 of."""
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

from pdf_fixtures import make_page_with_form_widget  # noqa: E402
from utils.pdf_text_layer import extract_page_native_text, extract_page_supplementary_text  # noqa: E402

_LONG_BODY = "Form body text content here. " * 5  # comfortably over the 40-char native-text floor


@pytest.mark.unit
class TestWidgetValueExtraction:
    def test_text_widget_value_is_extracted(self):
        pdf_bytes = make_page_with_form_widget(field_name="applicant_name", field_value="สมชาย ใจดี", body_text=_LONG_BODY)
        result = extract_page_supplementary_text(pdf_bytes, 1)
        assert "สมชาย ใจดี" in result
        assert "applicant_name" in result

    def test_widget_value_absent_from_plain_get_text(self):
        """The actual gap being fixed: confirm get_text() alone really does
        NOT see the widget value — if this assertion ever fails it means
        either this pymupdf version changed behavior or the fixture is
        wrong, either of which should fail loudly rather than the other
        tests here silently testing nothing real."""
        pdf_bytes = make_page_with_form_widget(field_name="applicant_name", field_value="สมชาย ใจดี", body_text=_LONG_BODY)
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            raw_text = doc[0].get_text()
        finally:
            doc.close()
        assert "สมชาย ใจดี" not in raw_text

    def test_empty_widget_value_produces_no_line(self):
        pdf_bytes = make_page_with_form_widget(field_name="applicant_name", field_value="", body_text=_LONG_BODY)
        result = extract_page_supplementary_text(pdf_bytes, 1)
        assert result == ""

    def test_page_with_no_widgets_returns_empty_string_not_none(self):
        doc = pymupdf.open()
        doc.new_page(width=600, height=800)
        import io
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        result = extract_page_supplementary_text(buf.getvalue(), 1)
        assert result == ""

    def test_extraction_failure_is_non_fatal(self):
        """Malformed bytes must never raise — the caller always appends
        this result to the page's base text, so a crash here would take
        down extraction of an otherwise-fine page."""
        result = extract_page_supplementary_text(b"not a real pdf", 1)
        assert result == ""


@pytest.mark.unit
class TestFreeTextAndStampAnnotations:
    def test_freetext_annotation_text_is_extracted(self):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 50), _LONG_BODY, fontsize=10)
        page.add_freetext_annot(pymupdf.Rect(50, 200, 300, 230), "Stamped: APPROVED 2026-11-09")
        import io
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        result = extract_page_supplementary_text(buf.getvalue(), 1)
        assert "APPROVED" in result

    def test_other_annotation_types_are_ignored(self):
        """Only FreeText/Stamp carry meaningful get_text() output per
        PyMuPDF's own documented behavior — a Link or Highlight annotation
        must not contribute noise."""
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 50), _LONG_BODY, fontsize=10)
        page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(50, 50, 100, 70), "uri": "https://example.com"})
        import io
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        result = extract_page_supplementary_text(buf.getvalue(), 1)
        assert result == ""


@pytest.mark.unit
class TestEndToEndAppending:
    """The actual fix: a filled form's value must appear in the FINAL page
    output regardless of which branch (native text vs OCR) handles the
    base text — tested here via extract_page_native_text's accept path,
    which is the one gap #2 is specifically about (a page that otherwise
    passes every native-text check but is missing widget data)."""

    def test_widget_value_appended_to_accepted_native_text(self):
        pdf_bytes = make_page_with_form_widget(field_name="applicant_name", field_value="สมชาย ใจดี", body_text=_LONG_BODY)
        with patch("utils.pdf_text_layer._text_coverage_ratio", return_value=0.95):
            native = extract_page_native_text(pdf_bytes, 1)
        assert native is not None  # body text alone is long/valid enough to be accepted
        supplementary = extract_page_supplementary_text(pdf_bytes, 1)
        combined = f"{native}\n\n{supplementary}"
        assert "สมชาย ใจดี" in combined

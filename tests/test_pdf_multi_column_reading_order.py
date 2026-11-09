"""Tests for the 2026-11 (PDF pipeline overhaul, lower-priority) multi-
column reading-order fix: plain page.get_text() has no column awareness
and can interleave lines ACROSS columns on a genuinely multi-column page
(common in government gazette-style regulatory PDFs). Ported from
PyMuPDF's own documented community recipe (pymupdf/PyMuPDF-Utilities,
text-extraction/multi_column.py) — see utils/pdf_text_layer.py's
column_boxes()/extract_page_text_reading_order() docstrings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


def _single_column_page():
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((50, 50), "Single column line one.")
    page.insert_text((50, 70), "Single column line two.")
    return doc, page


def _two_column_page():
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    for i in range(10):
        page.insert_text((50, 50 + i * 20), f"Left col line {i}")
    for i in range(10):
        page.insert_text((350, 50 + i * 20), f"Right col line {i}")
    return doc, page


def _blank_page():
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    return doc, page


@pytest.mark.unit
class TestColumnBoxesEC2:
    def test_single_column_page_returns_one_box(self):
        from utils.pdf_text_layer import column_boxes

        doc, page = _single_column_page()
        try:
            boxes = column_boxes(page)
            assert len(boxes) == 1
        finally:
            doc.close()

    def test_two_column_page_returns_two_boxes_left_first(self):
        from utils.pdf_text_layer import column_boxes

        doc, page = _two_column_page()
        try:
            boxes = column_boxes(page)
            assert len(boxes) == 2
            assert boxes[0].x0 < boxes[1].x0
        finally:
            doc.close()

    def test_blank_page_returns_no_boxes(self):
        from utils.pdf_text_layer import column_boxes

        doc, page = _blank_page()
        try:
            assert column_boxes(page) == []
        finally:
            doc.close()


@pytest.mark.unit
class TestExtractPageTextReadingOrderEC2:
    def test_single_column_matches_plain_get_text(self):
        from utils.pdf_text_layer import extract_page_text_reading_order

        doc, page = _single_column_page()
        try:
            assert extract_page_text_reading_order(page) == page.get_text()
        finally:
            doc.close()

    def test_two_column_page_does_not_interleave_columns(self):
        """The actual regression this fix targets: each column's lines must
        stay grouped together, left column fully before right column —
        not interleaved row-by-row across both columns."""
        from utils.pdf_text_layer import extract_page_text_reading_order

        doc, page = _two_column_page()
        try:
            result = extract_page_text_reading_order(page)
            left_last_idx = result.index("Left col line 9")
            right_first_idx = result.index("Right col line 0")
            assert left_last_idx < right_first_idx, (
                "expected all 'Left col' lines before any 'Right col' line, "
                f"got interleaved output:\n{result}"
            )
        finally:
            doc.close()

    def test_blank_page_never_raises(self):
        from utils.pdf_text_layer import extract_page_text_reading_order

        doc, page = _blank_page()
        try:
            assert extract_page_text_reading_order(page) == ""
        finally:
            doc.close()

    def test_column_boxes_failure_falls_back_to_plain_get_text(self, monkeypatch):
        import utils.pdf_text_layer as mod

        doc, page = _two_column_page()
        try:
            monkeypatch.setattr(mod, "column_boxes", lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
            result = mod.extract_page_text_reading_order(page)
            assert result == page.get_text()
        finally:
            doc.close()


@pytest.mark.unit
class TestExtractPageTextWithTablesUsesReadingOrderEC2:
    def test_no_table_page_uses_multi_column_aware_extraction(self):
        """extract_page_text_with_tables' no-table branch must route through
        extract_page_text_reading_order, not plain page.get_text()
        directly — confirmed via source inspection since both produce
        identical output on a single-column page (can't distinguish by
        output alone there)."""
        import inspect

        from utils.pdf_text_layer import extract_page_text_with_tables

        src = inspect.getsource(extract_page_text_with_tables)
        assert "extract_page_text_reading_order(page)" in src


@pytest.mark.unit
class TestColumnBoxesLambda:
    def _import_handler(self):
        import os

        os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-2")
        lambda_dir = Path(__file__).resolve().parent.parent / "lambda" / "pdf_extraction"
        if str(lambda_dir) not in sys.path:
            sys.path.insert(0, str(lambda_dir))
        import handler
        return handler

    def test_single_column_page_returns_one_box(self):
        handler = self._import_handler()
        doc, page = _single_column_page()
        try:
            assert len(handler._column_boxes(page)) == 1
        finally:
            doc.close()

    def test_two_column_page_returns_two_boxes_left_first(self):
        handler = self._import_handler()
        doc, page = _two_column_page()
        try:
            boxes = handler._column_boxes(page)
            assert len(boxes) == 2
            assert boxes[0].x0 < boxes[1].x0
        finally:
            doc.close()

    def test_two_column_page_does_not_interleave_columns(self):
        handler = self._import_handler()
        doc, page = _two_column_page()
        try:
            result = handler._extract_page_text_reading_order(page)
            assert result.index("Left col line 9") < result.index("Right col line 0")
        finally:
            doc.close()

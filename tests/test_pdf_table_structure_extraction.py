"""Tests for table-structure extraction added to utils/pdf_text_layer.py
(find_page_tables, extract_page_text_with_tables) — 2026-11,
PDF-extraction-pipeline overhaul, gap #4.

Background: plain page.get_text() reads a table as flowing text — row/
column structure is lost, a fee amount separates from the license-type
header that gives it meaning. Confirmed this session: PyMuPDF 1.28.2 ships
page.find_tables() natively (no new dependency needed), with both a
ruled-line detector (default) and a pdfplumber-equivalent borderless
detector (strategy="text") already built in."""
from __future__ import annotations

import io
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

from pdf_fixtures import make_page_with_bordered_table  # noqa: E402
from utils.pdf_text_layer import (  # noqa: E402
    _block_overlaps_table,
    extract_page_native_text,
    extract_page_text_with_tables,
    find_page_tables,
)


@pytest.mark.unit
class TestFindPageTables:
    def test_bordered_table_is_detected(self):
        rows = [["Name", "Fee"], ["Alice", "100"], ["Bob", "200"]]
        pdf_bytes = make_page_with_bordered_table(rows=rows)
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            tables = find_page_tables(doc[0])
        finally:
            doc.close()
        assert len(tables) == 1
        assert tables[0].extract() == rows

    def test_page_with_no_table_returns_empty_list(self):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 50), "Ordinary prose with no table here at all.", fontsize=10)
        try:
            tables = find_page_tables(page)
        finally:
            doc.close()
        assert tables == []

    def test_malformed_page_fails_safe_to_empty_list(self):
        class _BrokenPage:
            def find_tables(self, **_kwargs):
                raise RuntimeError("boom")

        assert find_page_tables(_BrokenPage()) == []


@pytest.mark.unit
class TestBlockOverlapsTable:
    def test_block_fully_inside_table_overlaps(self):
        assert _block_overlaps_table((60, 60, 100, 80), (50, 50, 200, 200)) is True

    def test_block_fully_outside_table_does_not_overlap(self):
        assert _block_overlaps_table((300, 300, 350, 320), (50, 50, 200, 200)) is False

    def test_block_mostly_inside_table_overlaps(self):
        # block width=50 (160->210), table right edge at x=200: overlap width
        # = 200-160 = 40, ratio = 40/50 = 0.8 (80% of the block's own area)
        assert _block_overlaps_table((160, 60, 210, 80), (50, 50, 200, 200)) is True

    def test_block_mostly_outside_table_does_not_overlap(self):
        assert _block_overlaps_table((190, 60, 400, 80), (50, 50, 200, 200)) is False

    def test_missing_bbox_fails_safe_to_false(self):
        assert _block_overlaps_table(None, (50, 50, 200, 200)) is False
        assert _block_overlaps_table((60, 60, 100, 80), None) is False


@pytest.mark.unit
class TestExtractPageTextWithTables:
    def test_table_is_spliced_as_markdown_not_flowing_text(self):
        rows = [["Name", "Fee"], ["Alice", "100"], ["Bob", "200"]]
        pdf_bytes = make_page_with_bordered_table(rows=rows)
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            text = extract_page_text_with_tables(doc[0])
        finally:
            doc.close()
        assert "|Name|Fee|" in text
        assert "|Alice|100|" in text

    def test_cell_content_appears_exactly_once_not_duplicated(self):
        """The actual bug this mechanism exists to avoid: splicing (not
        appending) means a cell's text must not ALSO appear as leftover
        flowing text alongside the markdown table."""
        rows = [["Name", "Fee"], ["Alice", "100"], ["Bob", "200"]]
        pdf_bytes = make_page_with_bordered_table(rows=rows)
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            text = extract_page_text_with_tables(doc[0])
        finally:
            doc.close()
        assert text.count("Alice") == 1
        assert text.count("100") == 1

    def test_page_with_no_table_falls_back_to_plain_get_text(self):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 50), "Ordinary prose with no table here at all.", fontsize=10)
        try:
            text = extract_page_text_with_tables(page)
            plain = page.get_text()
        finally:
            doc.close()
        assert text == plain

    def test_body_text_outside_the_table_is_preserved(self):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 50), "Document intro paragraph here, well above the table.", fontsize=10)
        rows = [["Name", "Fee"], ["Alice", "100"]]
        n_cols = 2
        ox, oy = 50, 150
        for r in range(len(rows) + 1):
            y = oy + r * 24
            page.draw_line((ox, y), (ox + n_cols * 80, y))
        for c in range(n_cols + 1):
            x = ox + c * 80
            page.draw_line((x, oy), (x, oy + len(rows) * 24))
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                page.insert_text((ox + c * 80 + 4, oy + r * 24 + 24 - 8), cell, fontsize=9)
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()

        doc2 = pymupdf.open(stream=buf.getvalue(), filetype="pdf")
        try:
            text = extract_page_text_with_tables(doc2[0])
        finally:
            doc2.close()
        assert "Document intro paragraph" in text
        assert "|Name|Fee|" in text


@pytest.mark.unit
class TestExtractPageNativeTextTableIntegration:
    def test_accepted_page_with_table_returns_table_structured_text(self):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 50), "This document lists registration fees for members in the table below here.", fontsize=10)
        rows = [["Name", "Fee"], ["Alice", "100"], ["Bob", "200"]]
        ox, oy = 50, 150
        for r in range(len(rows) + 1):
            y = oy + r * 24
            page.draw_line((ox, y), (ox + 2 * 120, y))
        for c in range(3):
            x = ox + c * 120
            page.draw_line((x, oy), (x, oy + len(rows) * 24))
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                page.insert_text((ox + c * 120 + 4, oy + r * 24 + 24 - 8), cell, fontsize=9)
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        pdf_bytes = buf.getvalue()

        with patch("utils.pdf_text_layer._text_coverage_ratio", return_value=0.95):
            result = extract_page_native_text(pdf_bytes, 1)
        assert result is not None
        assert "registration fees" in result
        assert "|Name|Fee|" in result
        assert result.count("Alice") == 1

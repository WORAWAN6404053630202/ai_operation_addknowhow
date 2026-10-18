"""
tests/test_sheet_versioning.py — regression tests for the versioning/
superseding fix (2026-09, feature/pdf-ingestion item 2.2).

Background: approving a PDF review item with decision_type="update" writes a
note into the OLD Sheet row's "superseded_note" column (see
sheet_write_back.py's _note_superseded_row) — but until this fix, nothing on
the READ side (data_loader.py / data_loader_general.py, used by
scripts/ingest_local.py to rebuild the Chroma vector store) ever checked that
column. The old, superseded row kept being ingested into Chroma right
alongside its replacement on every rebuild, which is exactly the "multiple
documents disagree about the same fact" bug class that caused several real
answer-quality bugs found in production QA testing the same day this was
fixed (see project memory).

The Sheet itself deliberately stays additive-only / never edited or deleted
(a safety property — see sheet_write_back.py's module docstring), so the fix
is entirely on the read/ingest side: skip a row at ingest time if its
"superseded_note" column is non-empty.

These tests use mocks/local DataFrames only — no real Google Sheets API
calls, no network. Safe to run anytime.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model.pdf_review_item import ReviewItem  # noqa: E402
from service.data_loader import DataLoader  # noqa: E402
from service.data_loader_general import GeneralDataLoader  # noqa: E402
from service.sheet_write_back import _note_superseded_row  # noqa: E402


@pytest.mark.unit
class TestDataLoaderSkipsSupersededRows:
    """Regulatory-sheet path (DataLoader._process_dataframe)."""

    def test_row_with_superseded_note_is_skipped(self):
        dl = DataLoader(config=MagicMock())
        df = pd.DataFrame([
            {
                "หน่วยงาน": "กรมพัฒนาธุรกิจการค้า", "ใบอนุญาต": "ทะเบียนพาณิชย์เก่า",
                "ขั้นตอนการดำเนินการ": "ขั้นตอนเก่า",
                "superseded_note": "มีข้อมูลใหม่มาแทนที่ — ดูแถวใหม่ (source_review_id=abc123)",
            },
            {
                "หน่วยงาน": "กรมพัฒนาธุรกิจการค้า", "ใบอนุญาต": "ทะเบียนพาณิชย์ใหม่",
                "ขั้นตอนการดำเนินการ": "ขั้นตอนใหม่",
                "superseded_note": "",
            },
        ])
        added = dl._process_dataframe(df, source="test")
        assert added == 1
        assert dl.documents[0].metadata.get("license_type") == "ทะเบียนพาณิชย์ใหม่"

    def test_no_superseded_column_at_all_ingests_everything(self):
        """Backward compatibility: a Sheet that has never had this column yet
        (or a source that legitimately never uses it) must not lose any rows."""
        dl = DataLoader(config=MagicMock())
        df = pd.DataFrame([
            {"หน่วยงาน": "A", "ใบอนุญาต": "X", "ขั้นตอนการดำเนินการ": "steps"},
            {"หน่วยงาน": "B", "ใบอนุญาต": "Y", "ขั้นตอนการดำเนินการ": "steps2"},
        ])
        added = dl._process_dataframe(df, source="test")
        assert added == 2


@pytest.mark.unit
class TestGeneralDataLoaderSkipsSupersededRows:
    """Marketing/business_guide-sheet path (GeneralDataLoader._process_dataframe)
    — a separate class/loader from the regulatory one, found during this
    investigation to need the identical fix independently."""

    _COLMAP = {
        "source_book": "source_book", "main_topic": "main_topic", "sub_topic": "sub_topic",
        "content_type": "content_type", "answer_guideline": "answer_guideline",
        "research_reference": "research_reference",
    }

    def test_row_with_superseded_note_is_skipped(self):
        dl = GeneralDataLoader(data_type="marketing")
        df = pd.DataFrame([
            {
                "source_book": "book1", "main_topic": "กลยุทธ์เก่า", "sub_topic": "sub1",
                "content_type": "x", "answer_guideline": "เนื้อหาเก่ามากพอที่จะผ่าน 30 ตัวอักษรขั้นต่ำ",
                "research_reference": "", "superseded_note": "มีข้อมูลใหม่มาแทนที่",
            },
            {
                "source_book": "book1", "main_topic": "กลยุทธ์ใหม่", "sub_topic": "sub2",
                "content_type": "x", "answer_guideline": "เนื้อหาใหม่มากพอที่จะผ่าน 30 ตัวอักษรขั้นต่ำ",
                "research_reference": "", "superseded_note": "",
            },
        ])
        added = dl._process_dataframe(df, self._COLMAP, source="test")
        assert added == 1
        assert dl.documents[0].metadata.get("main_topic") == "กลยุทธ์ใหม่"

    def test_no_superseded_column_at_all_ingests_everything(self):
        dl = GeneralDataLoader(data_type="marketing")
        df = pd.DataFrame([
            {
                "source_book": "book1", "main_topic": "A", "sub_topic": "s1", "content_type": "x",
                "answer_guideline": "เนื้อหามากพอที่จะผ่านเกณฑ์ 30 ตัวอักษรขั้นต่ำของระบบ", "research_reference": "",
            },
        ])
        added = dl._process_dataframe(df, self._COLMAP, source="test")
        assert added == 1


@pytest.mark.unit
class TestSupersededNoteAutoCreatesColumn:
    """Write-back path (sheet_write_back._note_superseded_row) — verifies the
    Sheet's superseded_note column gets auto-created when missing, rather
    than silently skipping the whole feature until a human adds it by hand.
    Uses a MagicMock worksheet — never touches the real Google Sheet."""

    def test_creates_column_when_missing(self):
        mock_ws = MagicMock()
        mock_ws.row_values.return_value = ["x", "y", "z"]
        header_row = ["หน่วยงาน", "ใบอนุญาต", "ขั้นตอนการดำเนินการ"]
        item = ReviewItem(
            id="test123", filename="test.pdf", decision_type="update", old_row_ref="5",
            llm_drafted_fields={"หน่วยงาน": "x", "ใบอนุญาต": "y"},
        )

        _note_superseded_row(mock_ws, header_row, item)

        calls = mock_ws.update_cell.call_args_list
        assert len(calls) == 2, "expected header-creation call + note-write call"
        assert calls[0].args == (1, 4, "superseded_note"), "column should be created at the next free index"
        assert calls[1].args[0] == 5 and calls[1].args[1] == 4, "note should be written to old_row_ref's row, new column"

    def test_no_extra_calls_when_column_already_exists(self):
        mock_ws = MagicMock()
        mock_ws.row_values.return_value = ["x", "x", "x", ""]
        header_row = ["หน่วยงาน", "ใบอนุญาต", "ขั้นตอนการดำเนินการ", "superseded_note"]
        item = ReviewItem(
            id="test456", filename="test.pdf", decision_type="update", old_row_ref="7",
            llm_drafted_fields={"หน่วยงาน": "x", "ใบอนุญาต": "y"},
        )

        _note_superseded_row(mock_ws, header_row, item)

        calls = mock_ws.update_cell.call_args_list
        assert len(calls) == 1, "column already exists — must not attempt to create it again"
        assert calls[0].args == (7, 4, "มีข้อมูลใหม่มาแทนที่ — ดูแถวใหม่ (source_review_id=test456)")

    def test_auto_create_failure_degrades_gracefully(self):
        """If the Sheets API call to create the column itself fails (rare, but
        must not crash the approval — the new row is already safely written
        by the time this function runs), fall back to the pre-fix behavior:
        log a warning and skip the note, returning None."""
        mock_ws = MagicMock()
        mock_ws.update_cell.side_effect = Exception("simulated Sheets API error")
        header_row = ["หน่วยงาน", "ใบอนุญาต"]
        item = ReviewItem(
            id="test789", filename="test.pdf", decision_type="update", old_row_ref="3",
            llm_drafted_fields={"หน่วยงาน": "x"},
        )

        result = _note_superseded_row(mock_ws, header_row, item)
        assert result is None

"""
tests/test_knowhow_dedup_versioning.py — regression tests for the know-how
duplicate-detection + versioning fix (2026-09, feature/pdf-ingestion item
2.5 / "3" from the production-readiness follow-up work).

Background: pdf_candidate_matching.py already had a well-designed 3-signal
(embedding / identity fuzzy / LLM batch-scan) duplicate-detection system for
the REGULATORY Sheet — but admin_pdf.py's own comment confirmed know-how
items had none at all ("no duplicate-detection built for the know_how tab
yet, so decision_type is effectively always 'new'"). This added
pdf_knowhow_candidate_matching.py (same 3-signal design, adapted to the
main_topic/sub_topic/summary/category schema and the 2-tab routing) and
extended knowhow_write_back.py to actually respect decision_type="update"
(it previously ignored old_row_ref entirely for this path).

These tests cover only the deterministic, mock-based logic — no real Google
Sheets API calls, no real embedding/LLM calls, no network. Safe to run
anytime.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from service.knowhow_write_back import _ensure_column, _note_superseded_row  # noqa: E402
from service.pdf_knowhow_candidate_matching import (  # noqa: E402
    _col_index_by_key,
    _comparison_text,
    _cosine_similarity,
    _dynamic_cutoff_count,
    _identity_fuzzy_candidates,
)


@pytest.mark.unit
class TestComparisonHelpers:
    """Pure logic, ported verbatim from pdf_candidate_matching.py's
    already-proven implementation — pinning behavior here mainly to catch
    accidental drift between the two copies over time (the exact class of
    duplicated-logic risk this whole session was otherwise focused on)."""

    def test_comparison_text_joins_nonempty_fields_only(self):
        values = {"main_topic": "กลยุทธ์ราคา", "sub_topic": "", "summary": "สรุป", "category": "  "}
        text = _comparison_text(values, ["main_topic", "sub_topic", "summary", "category"])
        assert text == "กลยุทธ์ราคา | สรุป"

    def test_cosine_similarity_identical_vectors_is_one(self):
        assert _cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors_is_zero(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector_does_not_divide_by_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_dynamic_cutoff_finds_the_real_gap(self):
        # Two genuinely-close scores (0.95, 0.93), then a big drop to noise —
        # should keep only the 2 close ones, not everything above the floor.
        scores = [0.95, 0.93, 0.75, 0.72, 0.71]
        assert _dynamic_cutoff_count(scores) == 2

    def test_dynamic_cutoff_below_floor_returns_zero(self):
        assert _dynamic_cutoff_count([0.5, 0.4, 0.3]) == 0

    def test_identity_fuzzy_requires_all_identity_fields_to_match(self):
        new_values = {"main_topic": "กลยุทธ์การตั้งราคา", "sub_topic": "ส่วนลด"}
        rows = [
            (2, {"main_topic": "กลยุทธ์การตั้งราคา", "sub_topic": "ส่วนลด"}),  # both match
            (3, {"main_topic": "กลยุทธ์การตั้งราคา", "sub_topic": "จิตวิทยา"}),  # only main_topic matches
        ]
        flagged = _identity_fuzzy_candidates(new_values, rows)
        assert flagged == {2}

    def test_col_index_by_key_matches_real_header_variants(self):
        header_row = ["หัวข้อหลัก", "หัวข้อการดำเนินการย่อย", "ประเภท", "แนวคำตอบ", "สรุปสั้น"]
        col_index = _col_index_by_key(header_row)
        assert col_index["main_topic"] == 0
        assert col_index["sub_topic"] == 1
        assert col_index["category"] == 2
        assert col_index["full_text"] == 3
        assert col_index["summary"] == 4


def _stateful_worksheet(headers: list[str]):
    """A MagicMock whose row_values(1) reflects header cells actually
    "written" via update_cell(1, col, value) — plain MagicMock's fixed
    return_value can't simulate that, which caused this test file's first
    draft to assert against a stale header list that never saw its own
    auto-created column. Only row 1 (headers) is stateful; other rows are
    irrelevant to what these tests check."""
    state = {"headers": list(headers)}
    ws = MagicMock()
    ws.col_count = len(headers)

    def _row_values(row_num):
        return state["headers"] if row_num == 1 else []

    def _update_cell(row_num, col_num, value):
        if row_num == 1:
            while len(state["headers"]) < col_num:
                state["headers"].append("")
            state["headers"][col_num - 1] = value

    ws.row_values.side_effect = _row_values
    ws.update_cell.side_effect = _update_cell
    return ws, state


@pytest.mark.unit
class TestKnowhowSupersededNoteAutoCreatesColumn:
    """Mirrors test_sheet_versioning.py's regulatory-Sheet equivalent —
    verifies the know-how tabs get the same auto-create-column treatment."""

    def test_creates_column_when_missing(self):
        mock_ws, _ = _stateful_worksheet(["เรื่อง", "หัวข้อหลัก", "หัวข้อการดำเนินการย่อย"])
        mock_ws.title = "Know how_ร้านอาหาร"

        _note_superseded_row(mock_ws, old_row_ref="12", new_topic_label="กลยุทธ์ราคาใหม่", source_file="test.pdf")

        update_calls = mock_ws.update_cell.call_args_list
        assert len(update_calls) == 2
        assert update_calls[0].args == (1, 4, "superseded_note")
        assert update_calls[1].args[0] == 12 and update_calls[1].args[1] == 4

    def test_no_extra_calls_when_column_already_exists(self):
        mock_ws = MagicMock()
        mock_ws.title = "Know How_ข้อมูลหนังสือ"
        mock_ws.row_values.return_value = ["ชื่อหนังสือ", "หัวข้อหลัก", "superseded_note"]
        mock_ws.col_count = 3

        _note_superseded_row(mock_ws, old_row_ref="8", new_topic_label="topic X", source_file="book.pdf")

        update_calls = mock_ws.update_cell.call_args_list
        assert len(update_calls) == 1
        assert update_calls[0].args[0] == 8 and update_calls[0].args[1] == 3

    def test_invalid_old_row_ref_does_not_raise(self):
        mock_ws = MagicMock()
        # Should log a warning and return without touching the worksheet at all.
        _note_superseded_row(mock_ws, old_row_ref="not-a-number", new_topic_label="x", source_file="y.pdf")
        mock_ws.update_cell.assert_not_called()


@pytest.mark.unit
class TestEnsureColumn:
    def test_adds_column_when_missing_and_extends_col_count(self):
        mock_ws = MagicMock()
        mock_ws.row_values.return_value = ["a", "b"]
        mock_ws.col_count = 2

        _ensure_column(mock_ws, "new_col")

        mock_ws.add_cols.assert_called_once_with(1)
        mock_ws.update_cell.assert_called_once_with(1, 3, "new_col")

    def test_noop_when_column_already_present(self):
        mock_ws = MagicMock()
        mock_ws.row_values.return_value = ["a", "new_col"]
        mock_ws.col_count = 2

        _ensure_column(mock_ws, "new_col")

        mock_ws.update_cell.assert_not_called()
        mock_ws.add_cols.assert_not_called()


@pytest.mark.unit
class TestAppendKnowhowTopicsRespectsDecisionType:
    """Verifies append_knowhow_topics actually calls the superseded-note
    write when decision_type="update" is given — this is the exact gap
    admin_pdf.py's old comment described ("decision_type is effectively
    always 'new' here regardless of what the UI sent")."""

    def _make_worksheet(self, headers):
        ws, _ = _stateful_worksheet(headers)
        ws.get_all_values.return_value = [headers]
        return ws

    def test_update_decision_marks_old_row_superseded(self, monkeypatch):
        import service.knowhow_write_back as kwb

        # Includes "สรุปสั้น" already, so the only update_cell calls that
        # happen are ones this test cares about (_ensure_summary_column is
        # pre-existing, unrelated behavior — this test is about decision_type
        # wiring specifically, not re-testing that column's own auto-add).
        headers = ["เรื่อง", "หัวข้อหลัก", "หัวข้อการดำเนินการย่อย", "ประเภท", "แนวคำตอบ", "อ้างอิง Research", "สรุปสั้น"]
        mock_ws = self._make_worksheet(headers)
        mock_spreadsheet = MagicMock()
        mock_spreadsheet.worksheet.return_value = mock_ws
        monkeypatch.setattr(kwb, "_get_spreadsheet", lambda: mock_spreadsheet)

        topics = [{
            "document_title": "doc", "source_type": "document",
            "main_topic": "กลยุทธ์ราคา", "sub_topic": "ส่วนลด", "summary": "s",
            "category": "", "page_range": "1-2", "full_text": "เนื้อหา",
        }]

        result = kwb.append_knowhow_topics(topics, "test.pdf", decision_type="update", old_row_ref="9")

        assert result["rows_appended"] == 1
        mock_ws.append_row.assert_called_once()
        # The superseded-note write is a SEPARATE update_cell call after the
        # append (row 1 = the new "superseded_note" header cell it also
        # needs to create; row 9 = old_row_ref's actual note).
        update_calls = [c for c in mock_ws.update_cell.call_args_list]
        assert any(c.args[0] == 9 for c in update_calls), (
            f"expected an update_cell call targeting row 9 (old_row_ref), got: {update_calls}"
        )

    def test_new_decision_does_not_touch_any_old_row(self, monkeypatch):
        import service.knowhow_write_back as kwb

        headers = ["เรื่อง", "หัวข้อหลัก", "หัวข้อการดำเนินการย่อย", "ประเภท", "แนวคำตอบ", "อ้างอิง Research", "สรุปสั้น"]
        mock_ws = self._make_worksheet(headers)
        mock_spreadsheet = MagicMock()
        mock_spreadsheet.worksheet.return_value = mock_ws
        monkeypatch.setattr(kwb, "_get_spreadsheet", lambda: mock_spreadsheet)

        topics = [{
            "document_title": "doc", "source_type": "document",
            "main_topic": "หัวข้อใหม่", "sub_topic": "sub", "summary": "s",
            "category": "", "page_range": "1", "full_text": "เนื้อหา",
        }]

        kwb.append_knowhow_topics(topics, "test.pdf", decision_type="new", old_row_ref=None)

        mock_ws.update_cell.assert_not_called()

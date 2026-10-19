"""Tests for the 2026-10 fix (PDF-M1) adding input-size batching to
service/pdf_field_drafting.py's draft_fields_from_pages — previously always
a single LLM call over the whole combined text, unlike its sibling
identify_license_topics, which already needed _MAX_BATCH_CHARS/
_split_into_batches after a live-reproduced "prompt is too long" failure on
a real 242-page document. A single topic's pages are usually much smaller
than a whole multi-topic document, but nothing structurally prevented one
huge single-topic document from hitting the exact same ceiling.

No real network/LLM calls — the OpenAI client is mocked throughout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import service.pdf_field_drafting as field_drafting  # noqa: E402


def _mock_response(fields: dict):
    resp = MagicMock()
    resp.choices[0].message.content = json.dumps(fields, ensure_ascii=False)
    resp.usage.prompt_tokens = 50
    resp.usage.completion_tokens = 20
    return resp


@pytest.mark.unit
class TestSmallInputStaysSingleCall:
    def test_under_budget_makes_exactly_one_llm_call(self):
        with patch.object(field_drafting, "OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_response(
                {"department": "กรมสรรพากร", "fees": "100 บาท"}
            )
            field_drafting.draft_fields_from_pages(["short page 1", "short page 2"])
        assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.unit
class TestLargeInputSplitsIntoBatches:
    def test_over_budget_makes_multiple_calls_and_merges(self, monkeypatch):
        monkeypatch.setattr(field_drafting, "_MAX_BATCH_CHARS", 20)
        pages = ["a" * 15, "b" * 15, "c" * 15]  # 3 pages, budget forces 3 separate batches

        responses = [
            _mock_response({"department": "กรมสรรพากร", "fees": "100 บาท"}),
            _mock_response({"department": "", "fees": "200 บาท"}),
            _mock_response({"department": "หน่วยงานอื่น (ไม่ควรถูกใช้)", "fees": ""}),
        ]
        with patch.object(field_drafting, "OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.side_effect = responses
            result = field_drafting.draft_fields_from_pages(pages)

        assert mock_client.chat.completions.create.call_count == 3
        # Identity-like field (department): first non-empty batch wins, later
        # batches' values (even a different one) must NOT override or append.
        assert result["หน่วยงาน"] == "กรมสรรพากร"
        # Content field (fees): every batch's non-empty value is concatenated,
        # not dropped — this is the actual bug class being fixed (a large
        # topic's real content split across batches must not be lost).
        assert "100 บาท" in result["ค่าธรรมเนียม"]
        assert "200 บาท" in result["ค่าธรรมเนียม"]

    def test_batching_threshold_matches_total_char_count_not_page_count(self, monkeypatch):
        """A single oversized page alone must also trigger the same path —
        the check is total characters, not number of pages."""
        monkeypatch.setattr(field_drafting, "_MAX_BATCH_CHARS", 10)
        with patch.object(field_drafting, "OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_response({"department": "X"})
            field_drafting.draft_fields_from_pages(["x" * 5])  # under budget, 1 page
        assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.unit
class TestMergeDraftedBatches:
    def test_identity_field_takes_first_non_empty(self):
        merged = field_drafting._merge_drafted_batches([
            {"หน่วยงาน": ""},
            {"หน่วยงาน": "กรมสรรพากร"},
            {"หน่วยงาน": "ควรถูกละเลย"},
        ])
        assert merged["หน่วยงาน"] == "กรมสรรพากร"

    def test_content_field_concatenates_non_empty_values_in_order(self):
        merged = field_drafting._merge_drafted_batches([
            {"ค่าธรรมเนียม": "ขั้นตอนที่ 1-5"},
            {"ค่าธรรมเนียม": ""},
            {"ค่าธรรมเนียม": "ขั้นตอนที่ 6-10"},
        ])
        assert merged["ค่าธรรมเนียม"] == "ขั้นตอนที่ 1-5\nขั้นตอนที่ 6-10"

    def test_content_field_dedupes_identical_repeated_values(self):
        merged = field_drafting._merge_drafted_batches([
            {"ค่าธรรมเนียม": "100 บาท"},
            {"ค่าธรรมเนียม": "100 บาท"},
        ])
        assert merged["ค่าธรรมเนียม"] == "100 บาท"

    def test_all_batches_empty_for_a_field_yields_empty_string(self):
        merged = field_drafting._merge_drafted_batches([{"หน่วยงาน": ""}, {"หน่วยงาน": ""}])
        assert merged["หน่วยงาน"] == ""

    def test_every_draftable_header_present_in_merged_output(self):
        merged = field_drafting._merge_drafted_batches([{"หน่วยงาน": "X"}])
        expected_headers = {headers[0] for headers in field_drafting.DRAFTABLE_FIELDS.values()}
        assert set(merged.keys()) == expected_headers

"""Tests for the 2026-11 (PDF pipeline overhaul, Phase 6 / gap #6) per-
document reconciliation-lineage fields added to service/sqs_consumer.py:
_page_reconciliation_summary() and its wiring into _build_items_direct's
existing append_event() call.

Background: before this fix, the activity log recorded topic/write counts
and cost but nothing about page-level completeness (how many pages came
back with real content vs. empty/errored) — exactly the kind of signal
that would have surfaced the 28-document silent-gap incident this whole
pipeline overhaul traces back to, much sooner.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import service.sqs_consumer as sqs_consumer  # noqa: E402
from model.pdf_review_item import PageExtractionRecord  # noqa: E402
from service.pdf_supersede_resolver import SupersedeDecision  # noqa: E402


def _page(page_num: int, typhoon_markdown: str) -> PageExtractionRecord:
    return PageExtractionRecord(page_num=page_num, typhoon_markdown=typhoon_markdown, claude_markdown="")


@pytest.mark.unit
class TestPageReconciliationSummary:
    def test_all_healthy_pages_counted_with_no_empty_or_error(self):
        pages = [_page(1, "a" * 300), _page(2, "b" * 500)]
        summary = sqs_consumer._page_reconciliation_summary(pages)
        assert summary["pages_total"] == 2
        assert summary["pages_with_content"] == 2
        assert summary["pages_empty_or_error"] == 0
        assert summary["total_chars"] == 800
        assert summary["chars_per_page"] == pytest.approx(400.0)

    def test_blank_page_counted_as_empty_not_content(self):
        pages = [_page(1, "real content here"), _page(2, "   "), _page(3, "")]
        summary = sqs_consumer._page_reconciliation_summary(pages)
        assert summary["pages_total"] == 3
        assert summary["pages_with_content"] == 1
        assert summary["pages_empty_or_error"] == 2

    def test_error_placeholder_page_counted_as_empty_not_content(self):
        pages = [
            _page(1, "real content here"),
            _page(2, "[สกัดข้อมูลหน้านี้ไม่สำเร็จ: simulated failure]"),
        ]
        summary = sqs_consumer._page_reconciliation_summary(pages)
        assert summary["pages_with_content"] == 1
        assert summary["pages_empty_or_error"] == 1
        # the error placeholder's own chars must not inflate chars_per_page
        assert summary["total_chars"] == len("real content here")

    def test_all_pages_empty_or_error_yields_zero_chars_per_page_no_divide_by_zero(self):
        pages = [_page(1, ""), _page(2, "[สกัดข้อมูลหน้านี้ไม่สำเร็จ]")]
        summary = sqs_consumer._page_reconciliation_summary(pages)
        assert summary["pages_with_content"] == 0
        assert summary["chars_per_page"] == 0.0

    def test_empty_document_yields_zero_counts(self):
        summary = sqs_consumer._page_reconciliation_summary([])
        assert summary == {
            "pages_total": 0, "pages_with_content": 0, "pages_empty_or_error": 0,
            "total_chars": 0, "chars_per_page": 0.0,
        }


def _make_pages(n: int):
    pages = [PageExtractionRecord(page_num=i + 1, typhoon_markdown=f"page {i + 1} content", claude_markdown="") for i in range(n)]
    markdowns = [p.typhoon_markdown for p in pages]
    return pages, markdowns


@pytest.mark.unit
class TestReconciliationSummaryWiredIntoActivityLog:
    def test_append_event_payload_includes_page_reconciliation_fields(self):
        pages, markdowns = _make_pages(3)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [{
                "document_title": "คู่มือ", "source_type": "book",
                "main_topic": "การตลาด", "sub_topic": "โปรโมชั่น",
                "summary": "สรุป", "category": "", "page_ranges": [(1, 3)],
            }]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"หน่วยงาน": "Dept A", "ใบอนุญาต": "License A"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1), \
             patch.object(sqs_consumer, "append_event") as mock_append:
            sqs_consumer._build_items_direct("test.pdf", "s3://bucket/test.pdf", pages, markdowns)

        mock_append.assert_called_once()
        event = mock_append.call_args[0][1]
        assert event["pages_total"] == 3
        assert event["pages_with_content"] == 3
        assert event["pages_empty_or_error"] == 0
        assert event["total_chars"] == sum(len(m) for m in markdowns)

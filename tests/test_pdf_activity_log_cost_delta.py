"""Test for the 2026-10 fix (PDF2-M3) to service/sqs_consumer.py's
_build_knowhow_items_direct/_build_license_items_direct — the activity-log
cost_usd field used to log the SHARED cost_accumulator's full running
total, not this builder's own share. A document that routes through BOTH
builders (uncovered-pages routing, or the mixed-shape secondary pass)
shares one CostAccumulator across both calls — the second builder's
append_event call used to report the FULL cumulative cost (including
whatever the first builder already spent), double-counting that spend in
the dashboard's activity log. Each builder now snapshots the accumulator
at its own entry and logs only the delta since then.
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
from utils.llm_cost_logging import CostAccumulator  # noqa: E402


def _make_pages(n: int):
    pages = [PageExtractionRecord(page_num=i + 1, typhoon_markdown=f"page {i + 1}", claude_markdown="") for i in range(n)]
    markdowns = [p.typhoon_markdown for p in pages]
    return pages, markdowns


@pytest.mark.unit
class TestActivityLogCostIsPerBuilderDeltaNotCumulativeTotal:
    def test_license_builder_logs_only_its_own_delta(self):
        pages, markdowns = _make_pages(2)
        shared_accumulator = CostAccumulator()
        shared_accumulator.total_cost_usd = 0.32  # simulates prior spend from a sibling builder call

        def fake_identify(page_markdowns, cost_accumulator=None):
            if cost_accumulator is not None:
                cost_accumulator.total_cost_usd += 0.08  # this builder's OWN spend
            return [{"department": "Dept A", "license_type": "License A", "page_ranges": [(1, 2)]}]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"แนวคำตอบ": "x"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "identify_license_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_regulatory_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1), \
             patch.object(sqs_consumer, "append_event") as mock_append:
            sqs_consumer._build_license_items_direct(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
                cost_accumulator=shared_accumulator,
            )

        mock_append.assert_called_once()
        event = mock_append.call_args[0][1]
        assert event["cost_usd"] == pytest.approx(0.08), (
            f"expected only this builder's own 0.08 delta, got {event['cost_usd']!r} "
            f"(0.40 would mean the prior 0.32 from a sibling builder leaked in)"
        )

    def test_knowhow_builder_logs_only_its_own_delta(self):
        pages, markdowns = _make_pages(2)
        shared_accumulator = CostAccumulator()
        shared_accumulator.total_cost_usd = 0.32

        def fake_identify_knowhow(page_markdowns, cost_accumulator=None):
            if cost_accumulator is not None:
                cost_accumulator.total_cost_usd += 0.05
            return [{
                "document_title": "คู่มือ", "source_type": "book",
                "main_topic": "การตลาด", "sub_topic": "โปรโมชั่น",
                "summary": "สรุป", "category": "", "page_ranges": [(1, 2)],
            }]

        def fake_resolve(topic_dict, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify_knowhow), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_knowhow_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1), \
             patch.object(sqs_consumer, "append_event") as mock_append:
            sqs_consumer._build_knowhow_items_direct(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "know_how"},
                cost_accumulator=shared_accumulator,
            )

        mock_append.assert_called_once()
        event = mock_append.call_args[0][1]
        assert event["cost_usd"] == pytest.approx(0.05)

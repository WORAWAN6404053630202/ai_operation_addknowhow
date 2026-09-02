"""
Integration-style test for the cost-accumulation ordering fix in
service/sqs_consumer.py's _build_license_items() — added 2026-09 alongside
the admin UI's per-document total_cost_usd display.

Background: the first draft of this stamped each topic_item.total_cost_usd
INSIDE the per-topic loop, right after that topic's own drafting/matching
calls. That's a real bug — the shared CostAccumulator keeps growing as LATER
topics get processed, so an EARLIER topic's stamped total would be missing
cost incurred processing topics after it. Fixed by stamping (and re-saving)
every item's total_cost_usd ONCE, after the whole loop (and any uncovered-
page know-how routing) finishes. This test proves multiple topics from one
document all end up showing the SAME final total, not staggered partial ones.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from model.pdf_review_item import PageExtractionRecord
from utils.llm_cost_logging import CostAccumulator


@pytest.mark.unit
class TestLicenseItemsCostAccumulation:
    def _make_pages(self, n: int) -> tuple[list[PageExtractionRecord], list[str]]:
        pages = [PageExtractionRecord(page_num=i + 1, typhoon_markdown=f"page {i + 1}", claude_markdown="") for i in range(n)]
        markdowns = [p.typhoon_markdown for p in pages]
        return pages, markdowns

    def test_every_topic_item_gets_the_same_final_total_cost(self, tmp_path):
        import service.sqs_consumer as sqs_consumer
        from model.pdf_review_queue_manager import PdfReviewQueueManager

        # Isolated on-disk queue for this test — no shared state with real data.
        test_queue_manager = PdfReviewQueueManager(persist_dir=str(tmp_path))

        pages, markdowns = self._make_pages(4)

        # Two topics: pages 1-2 and pages 3-4. Each downstream call "spends"
        # a fixed, known amount by adding directly to whatever accumulator
        # it's given — mirrors what log_llm_cost(..., accumulator=...) does
        # for a real LLM call, without needing a fake OpenAI response object.
        def fake_identify_license_topics(page_markdowns, cost_accumulator=None):
            if cost_accumulator is not None:
                cost_accumulator.add(0.01)  # cost of the topic-split call itself
            return [
                {"department": "Dept A", "license_type": "License A", "page_ranges": [(1, 2)]},
                {"department": "Dept B", "license_type": "License B", "page_ranges": [(3, 4)]},
            ]

        def fake_draft_fields_from_pages(topic_markdowns, cost_accumulator=None):
            if cost_accumulator is not None:
                cost_accumulator.add(0.02)
            return {"หน่วยงาน": "x"}

        def fake_find_candidate_matches(item, cost_accumulator=None):
            if cost_accumulator is not None:
                cost_accumulator.add(0.005)
            return []

        def fake_check_category_fit(item, cost_accumulator=None):
            if cost_accumulator is not None:
                cost_accumulator.add(0.001)
            return None

        with patch.object(sqs_consumer, "_queue_manager", test_queue_manager), \
             patch.object(sqs_consumer, "identify_license_topics", side_effect=fake_identify_license_topics), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft_fields_from_pages), \
             patch.object(sqs_consumer, "find_candidate_matches", side_effect=fake_find_candidate_matches), \
             patch.object(sqs_consumer, "check_category_fit", side_effect=fake_check_category_fit):
            acc = CostAccumulator()
            acc.add(0.03)  # simulated extraction-phase cost, seeded like sqs_consumer.py does
            items = sqs_consumer._build_license_items(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
                cost_accumulator=acc, extraction_started_at=1000.0,
            )

        # 0.03 (extraction) + 0.01 (topic split) + 2 * (0.02 + 0.005 + 0.001) (per-topic draft/match/category)
        expected_total = 0.03 + 0.01 + 2 * (0.02 + 0.005 + 0.001)
        assert acc.total_cost_usd == pytest.approx(expected_total)

        assert len(items) == 2
        for item in items:
            assert item.total_cost_usd == pytest.approx(expected_total), (
                f"item for {item.filename} showed {item.total_cost_usd}, expected the FINAL "
                f"total {expected_total} — an earlier item showing a smaller value would mean "
                f"the staggered-stamping bug regressed."
            )
            assert item.extraction_started_at == 1000.0

        # Re-read from disk (not just the in-memory objects returned above) to
        # prove the fix's re-save actually persisted the corrected value —
        # the in-memory objects alone wouldn't catch a "computed but never
        # saved" regression.
        for item in items:
            reloaded = test_queue_manager.load(item.id)
            assert reloaded is not None
            assert reloaded.total_cost_usd == pytest.approx(expected_total)

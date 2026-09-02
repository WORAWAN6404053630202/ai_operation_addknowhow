"""
Tests for model/pdf_review_item.py's cost/timing fields (added 2026-09 for
the admin UI's per-document cost/wall-clock-time display) — extraction_
started_at, total_cost_usd, and the processing_duration_seconds property
derived from extraction_started_at/extraction_completed_at.
"""
import pytest

from model.pdf_review_item import ReviewItem


@pytest.mark.unit
class TestReviewItemCostTimingFields:
    def test_defaults_are_none(self):
        item = ReviewItem(filename="test.pdf")
        assert item.extraction_started_at is None
        assert item.total_cost_usd is None
        assert item.processing_duration_seconds is None

    def test_processing_duration_computed_from_timestamps(self):
        item = ReviewItem(
            filename="test.pdf",
            extraction_started_at=1000.0,
            extraction_completed_at=1006.37,
        )
        assert item.processing_duration_seconds == pytest.approx(6.37)

    def test_processing_duration_none_when_started_at_missing(self):
        # e.g. an item processed before this field existed
        item = ReviewItem(filename="test.pdf", extraction_completed_at=1006.37)
        assert item.processing_duration_seconds is None

    def test_processing_duration_none_when_completed_at_missing(self):
        item = ReviewItem(filename="test.pdf", extraction_started_at=1000.0)
        assert item.processing_duration_seconds is None

    def test_total_cost_usd_settable_and_round_trips_via_model_dump(self):
        item = ReviewItem(filename="test.pdf", total_cost_usd=0.003241, extraction_started_at=1000.0)
        dumped = item.model_dump()
        assert dumped["total_cost_usd"] == pytest.approx(0.003241)
        assert dumped["extraction_started_at"] == 1000.0

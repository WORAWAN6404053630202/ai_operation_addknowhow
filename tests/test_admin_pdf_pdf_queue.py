"""Tests for the 2026-10 fix to router/admin_pdf.py's GET /admin/api/pdf-queue
— adds a `finished` list so a file that completes processing successfully
doesn't just vanish from the one dashboard the team actually watches (see
utils/pdf_ingestion_log.py's classify_event_outcome for the shared
new/duplicate/no_content/failed classification this reuses).

Mocks PdfReviewQueueManager, pdf_status_tracker.list_in_flight, and
read_recent_events — no real AWS calls, follows
tests/test_admin_pdf_decision_gate.py's pattern of calling the FastAPI
route function directly as a coroutine.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import router.admin_pdf as admin_pdf  # noqa: E402


def _event(filename: str, timestamp: float, **overrides) -> dict:
    event = {
        "filename": filename, "timestamp": timestamp,
        "new_count": 0, "update_count": 0, "duplicate_count": 0, "error": None,
    }
    event.update(overrides)
    return event


async def _list_pdf_queue(events=None, in_flight=None, items=None):
    mock_mgr = MagicMock()
    mock_mgr.list_all.return_value = items or []
    original_mgr = admin_pdf._pdf_queue_manager
    admin_pdf._pdf_queue_manager = mock_mgr
    try:
        with patch.object(admin_pdf, "read_recent_events", return_value=events or []), \
             patch.object(admin_pdf.pdf_status_tracker, "list_in_flight", return_value=in_flight or []), \
             patch.object(admin_pdf._conf, "PDF_INGESTION_S3_BUCKET", "test-bucket"):
            resp = await admin_pdf.pdf_queue_list()
    finally:
        admin_pdf._pdf_queue_manager = original_mgr
    import json
    return json.loads(resp.body.decode("utf-8"))


@pytest.mark.unit
@pytest.mark.asyncio
class TestFinishedListInPdfQueue:
    async def test_live_event_is_classified_and_included(self):
        data = await _list_pdf_queue(events=[_event("a.pdf", 100, new_count=2)])
        assert len(data["finished"]) == 1
        assert data["finished"][0]["outcome"] == "live"
        assert data["finished"][0]["filename"] == "a.pdf"

    async def test_duplicate_event_is_classified_and_included(self):
        data = await _list_pdf_queue(events=[_event("a.pdf", 100, duplicate_count=1)])
        assert data["finished"][0]["outcome"] == "duplicate"

    async def test_no_content_event_is_classified_and_included(self):
        data = await _list_pdf_queue(events=[_event("a.pdf", 100)])
        assert data["finished"][0]["outcome"] == "no_content"

    async def test_failed_events_are_excluded_from_finished(self):
        """Already shown via in_flight's failed-retry pill (while retrying)
        or via `items` (once given up) — showing it a third time here would
        be confusing, not helpful."""
        data = await _list_pdf_queue(events=[_event("a.pdf", 100, error="boom")])
        assert data["finished"] == []

    async def test_dedupes_by_filename_keeping_the_newest(self):
        """A reprocessed/re-uploaded file produces a second event — events
        arrive newest-first, so the first occurrence per filename wins."""
        events = [
            _event("a.pdf", 200, new_count=1),  # newest
            _event("a.pdf", 100, duplicate_count=1),  # older, should be dropped
        ]
        data = await _list_pdf_queue(events=events)
        assert len(data["finished"]) == 1
        assert data["finished"][0]["outcome"] == "live"
        assert data["finished"][0]["timestamp"] == 200

    async def test_filename_currently_in_flight_is_excluded_from_finished(self):
        """Race: an ingestion-log event can exist microseconds before its
        in-flight status object is cleared (same call stack in
        sqs_consumer.py) — in-flight must win in the merged view, not show
        the same file twice with two different states."""
        data = await _list_pdf_queue(
            events=[_event("a.pdf", 100, new_count=1)],
            in_flight=[{"filename": "a.pdf", "stage": "processing", "attempt": 1, "updated_at": 99}],
        )
        assert data["finished"] == []
        assert len(data["in_flight"]) == 1

    async def test_no_bucket_configured_is_a_safe_empty_list(self):
        mock_mgr = MagicMock()
        mock_mgr.list_all.return_value = []
        original_mgr = admin_pdf._pdf_queue_manager
        admin_pdf._pdf_queue_manager = mock_mgr
        try:
            with patch.object(admin_pdf._conf, "PDF_INGESTION_S3_BUCKET", ""):
                resp = await admin_pdf.pdf_queue_list()
        finally:
            admin_pdf._pdf_queue_manager = original_mgr
        import json
        data = json.loads(resp.body.decode("utf-8"))
        assert data["finished"] == []

    async def test_read_recent_events_failure_does_not_raise(self):
        mock_mgr = MagicMock()
        mock_mgr.list_all.return_value = []
        original_mgr = admin_pdf._pdf_queue_manager
        admin_pdf._pdf_queue_manager = mock_mgr
        try:
            with patch.object(admin_pdf, "read_recent_events", side_effect=RuntimeError("s3 down")), \
                 patch.object(admin_pdf.pdf_status_tracker, "list_in_flight", return_value=[]), \
                 patch.object(admin_pdf._conf, "PDF_INGESTION_S3_BUCKET", "test-bucket"):
                resp = await admin_pdf.pdf_queue_list()  # must not raise
        finally:
            admin_pdf._pdf_queue_manager = original_mgr
        import json
        data = json.loads(resp.body.decode("utf-8"))
        assert data["finished"] == []

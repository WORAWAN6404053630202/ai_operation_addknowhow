"""Tests for utils/pdf_ingestion_log.py — the S3-backed activity log
feeding the admin dashboard (2026-10).

2026-10 fix (PDF2-N5): this used to be one growing JSONL object per UTC
day, with append_event() doing a full get-existing-object -> append-line
-> put_object round trip — a heavy-upload day made every single append
rewrite an ever-growing object, and read_recent_events() had no cap at
all. Switched to one small object per event (no read-before-write needed
— append_event is now a single PUT), enumerated via list_objects_v2 and
capped by a new `limit` parameter. No real AWS — the S3 client is a
MagicMock throughout.
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

import utils.pdf_ingestion_log as mod  # noqa: E402


@pytest.mark.unit
class TestAppendEvent:
    def test_writes_a_single_object_with_no_read_first(self):
        mock_client = MagicMock()
        with patch.object(mod, "_client", return_value=mock_client):
            mod.append_event("test-bucket", {"filename": "a.pdf", "timestamp": 100})

        mock_client.get_object.assert_not_called()  # no read-before-write anymore
        mock_client.put_object.assert_called_once()
        _, kwargs = mock_client.put_object.call_args
        written = json.loads(kwargs["Body"].decode("utf-8"))
        assert written["filename"] == "a.pdf"

    def test_each_event_gets_its_own_key_under_the_day_prefix(self):
        mock_client = MagicMock()
        with patch.object(mod, "_client", return_value=mock_client):
            mod.append_event("test-bucket", {"filename": "a.pdf", "timestamp": 100})
            mod.append_event("test-bucket", {"filename": "b.pdf", "timestamp": 200})

        keys = [c.kwargs["Key"] for c in mock_client.put_object.call_args_list]
        assert len(set(keys)) == 2  # distinct keys, not overwriting each other
        assert all(k.startswith(mod.conf.PDF_INGESTION_LOG_S3_PREFIX) for k in keys)

    def test_event_without_timestamp_gets_one_added(self):
        mock_client = MagicMock()
        with patch.object(mod, "_client", return_value=mock_client):
            mod.append_event("test-bucket", {"filename": "a.pdf"})
        _, kwargs = mock_client.put_object.call_args
        written = json.loads(kwargs["Body"].decode("utf-8"))
        assert "timestamp" in written

    def test_put_error_does_not_raise(self):
        mock_client = MagicMock()
        mock_client.put_object.side_effect = RuntimeError("throttled")
        with patch.object(mod, "_client", return_value=mock_client):
            mod.append_event("test-bucket", {"filename": "a.pdf"})  # must not raise


def _paginator_returning(keys: list[str]):
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [{"Key": k} for k in keys]}] if keys else [{}]
    return paginator


@pytest.mark.unit
class TestReadRecentEvents:
    def test_missing_days_are_skipped_not_errors(self):
        mock_client = MagicMock()
        mock_client.get_paginator.return_value = _paginator_returning([])
        with patch.object(mod, "_client", return_value=mock_client):
            events = mod.read_recent_events("test-bucket", days=3)
        assert events == []

    def test_events_returned_newest_first_across_days(self):
        old_key = mod._event_key("2026-10-10", 1.0)
        new_key = mod._event_key("2026-10-11", 2.0)
        bodies = {
            old_key: json.dumps({"filename": "old.pdf", "timestamp": 1}),
            new_key: json.dumps({"filename": "new.pdf", "timestamp": 2}),
        }
        keys_by_prefix = {
            mod._day_prefix("2026-10-10"): [old_key],
            mod._day_prefix("2026-10-11"): [new_key],
        }

        def fake_get_object(Bucket, Key):
            body = MagicMock()
            body.read.return_value = bodies[Key].encode("utf-8")
            return {"Body": body}

        def fake_get_paginator(name):
            paginator = MagicMock()

            def fake_paginate(Bucket, Prefix):
                return [{"Contents": [{"Key": k} for k in keys_by_prefix.get(Prefix, [])]}]
            paginator.paginate.side_effect = fake_paginate
            return paginator

        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = fake_get_paginator
        mock_client.get_object.side_effect = fake_get_object
        with patch("utils.pdf_ingestion_log.datetime") as mock_dt, \
             patch.object(mod, "_client", return_value=mock_client):
            from datetime import datetime as real_datetime, timezone as real_timezone
            mock_dt.now.return_value = real_datetime(2026, 10, 11, tzinfo=real_timezone.utc)
            events = mod.read_recent_events("test-bucket", days=3)

        assert [e["filename"] for e in events] == ["new.pdf", "old.pdf"]

    def test_limit_caps_the_number_of_events_returned(self):
        keys = [mod._event_key("2026-10-11", float(i)) for i in range(10)]
        bodies = {k: json.dumps({"filename": f"{i}.pdf", "timestamp": i}) for i, k in enumerate(keys)}

        def fake_get_object(Bucket, Key):
            body = MagicMock()
            body.read.return_value = bodies[Key].encode("utf-8")
            return {"Body": body}

        mock_client = MagicMock()
        mock_client.get_paginator.return_value = _paginator_returning(keys)
        mock_client.get_object.side_effect = fake_get_object
        with patch.object(mod, "_client", return_value=mock_client):
            events = mod.read_recent_events("test-bucket", days=1, limit=3)

        assert len(events) == 3

    def test_unbounded_when_limit_is_none(self):
        keys = [mod._event_key("2026-10-11", float(i)) for i in range(10)]
        bodies = {k: json.dumps({"filename": f"{i}.pdf", "timestamp": i}) for i, k in enumerate(keys)}

        def fake_get_object(Bucket, Key):
            body = MagicMock()
            body.read.return_value = bodies[Key].encode("utf-8")
            return {"Body": body}

        mock_client = MagicMock()
        mock_client.get_paginator.return_value = _paginator_returning(keys)
        mock_client.get_object.side_effect = fake_get_object
        with patch.object(mod, "_client", return_value=mock_client):
            events = mod.read_recent_events("test-bucket", days=1, limit=None)

        assert len(events) == 10

    def test_malformed_event_is_skipped_not_fatal(self):
        bad_key = mod._event_key("2026-10-11", 1.0)
        good_key = mod._event_key("2026-10-11", 2.0)

        def fake_get_object(Bucket, Key):
            body = MagicMock()
            if Key == bad_key:
                body.read.return_value = b"not valid json{{"
            else:
                body.read.return_value = json.dumps({"filename": "ok.pdf", "timestamp": 1}).encode("utf-8")
            return {"Body": body}

        mock_client = MagicMock()
        mock_client.get_paginator.return_value = _paginator_returning([bad_key, good_key])
        mock_client.get_object.side_effect = fake_get_object
        with patch.object(mod, "_client", return_value=mock_client):
            events = mod.read_recent_events("test-bucket", days=1)

        assert len(events) == 1
        assert events[0]["filename"] == "ok.pdf"

    def test_listing_error_does_not_raise(self):
        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = RuntimeError("network down")
        with patch.object(mod, "_client", return_value=mock_client):
            events = mod.read_recent_events("test-bucket", days=1)  # must not raise
        assert events == []


@pytest.mark.unit
class TestClassifyEventOutcome:
    """2026-10 (admin dashboard gap): append_event's payload has no explicit
    status field — classify_event_outcome is the one shared place that
    turns new_count/update_count/duplicate_count/error into a human-facing
    outcome, used by both router/admin_pdf.py's /admin/api/pdf-queue
    (finished list) and /admin/api/pdf-status (activity table)."""

    def test_error_set_is_failed_regardless_of_counts(self):
        event = {"error": "Chroma write failed", "new_count": 3, "update_count": 1}
        assert mod.classify_event_outcome(event) == "failed"

    def test_new_or_update_count_is_live(self):
        assert mod.classify_event_outcome({"new_count": 1, "update_count": 0, "duplicate_count": 0}) == "live"
        assert mod.classify_event_outcome({"new_count": 0, "update_count": 2, "duplicate_count": 0}) == "live"

    def test_only_duplicate_count_is_duplicate(self):
        event = {"new_count": 0, "update_count": 0, "duplicate_count": 4}
        assert mod.classify_event_outcome(event) == "duplicate"

    def test_all_zero_counts_is_no_content(self):
        event = {"new_count": 0, "update_count": 0, "duplicate_count": 0}
        assert mod.classify_event_outcome(event) == "no_content"

    def test_missing_count_fields_default_to_zero_not_a_crash(self):
        assert mod.classify_event_outcome({}) == "no_content"

    def test_empty_string_error_is_not_treated_as_set(self):
        """An empty string is falsy — write_error defaults to None/"" on the
        success path (sqs_consumer.py), must not be misread as a failure."""
        event = {"error": "", "new_count": 1, "update_count": 0, "duplicate_count": 0}
        assert mod.classify_event_outcome(event) == "live"

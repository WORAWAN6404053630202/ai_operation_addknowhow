"""Tests for utils/pdf_ingestion_log.py — the S3-backed JSONL activity log
feeding the admin dashboard (2026-10). Covers: append_event's get-append-put
round trip (no native S3 append), the 404/NoSuchKey "start empty" path,
read_recent_events' day-rollover/newest-first ordering, and the
never-raise/best-effort contract. No real AWS — the S3 client is a
MagicMock throughout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import utils.pdf_ingestion_log as mod  # noqa: E402


def _no_such_key_error():
    return ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")


def _body(text: str):
    body = MagicMock()
    body.read.return_value = text.encode("utf-8")
    return {"Body": body}


@pytest.mark.unit
class TestAppendEvent:
    def test_first_event_of_the_day_creates_a_fresh_file(self):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = _no_such_key_error()
        with patch.object(mod, "_client", return_value=mock_client):
            mod.append_event("test-bucket", {"filename": "a.pdf", "timestamp": 100})

        mock_client.put_object.assert_called_once()
        _, kwargs = mock_client.put_object.call_args
        lines = kwargs["Body"].decode("utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["filename"] == "a.pdf"

    def test_second_event_appends_to_existing_file(self):
        mock_client = MagicMock()
        mock_client.get_object.return_value = _body(json.dumps({"filename": "a.pdf", "timestamp": 100}) + "\n")
        with patch.object(mod, "_client", return_value=mock_client):
            mod.append_event("test-bucket", {"filename": "b.pdf", "timestamp": 200})

        _, kwargs = mock_client.put_object.call_args
        lines = kwargs["Body"].decode("utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["filename"] == "a.pdf"
        assert json.loads(lines[1])["filename"] == "b.pdf"

    def test_event_without_timestamp_gets_one_added(self):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = _no_such_key_error()
        with patch.object(mod, "_client", return_value=mock_client):
            mod.append_event("test-bucket", {"filename": "a.pdf"})
        _, kwargs = mock_client.put_object.call_args
        written = json.loads(kwargs["Body"].decode("utf-8").strip())
        assert "timestamp" in written

    def test_s3_error_does_not_raise(self):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = RuntimeError("network down")
        with patch.object(mod, "_client", return_value=mock_client):
            mod.append_event("test-bucket", {"filename": "a.pdf"})  # must not raise

    def test_put_error_does_not_raise(self):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = _no_such_key_error()
        mock_client.put_object.side_effect = RuntimeError("throttled")
        with patch.object(mod, "_client", return_value=mock_client):
            mod.append_event("test-bucket", {"filename": "a.pdf"})  # must not raise


@pytest.mark.unit
class TestReadRecentEvents:
    def test_missing_days_are_skipped_not_errors(self):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = _no_such_key_error()
        with patch.object(mod, "_client", return_value=mock_client):
            events = mod.read_recent_events("test-bucket", days=3)
        assert events == []

    def test_events_returned_newest_first_across_days(self):
        day_bodies = {
            mod._day_key("2026-10-10"): json.dumps({"filename": "old.pdf", "timestamp": 1}) + "\n",
            mod._day_key("2026-10-11"): json.dumps({"filename": "new.pdf", "timestamp": 2}) + "\n",
        }

        def fake_get_object(Bucket, Key):
            if Key in day_bodies:
                return _body(day_bodies[Key])
            raise _no_such_key_error()

        mock_client = MagicMock()
        mock_client.get_object.side_effect = fake_get_object
        with patch("utils.pdf_ingestion_log.datetime") as mock_dt, patch.object(mod, "_client", return_value=mock_client):
            from datetime import date, datetime as real_datetime, timezone as real_timezone
            mock_dt.now.return_value = real_datetime(2026, 10, 11, tzinfo=real_timezone.utc)
            events = mod.read_recent_events("test-bucket", days=3)

        assert [e["filename"] for e in events] == ["new.pdf", "old.pdf"]

    def test_malformed_line_is_skipped_not_fatal(self):
        mock_client = MagicMock()

        def fake_get_object(Bucket, Key):
            if Key == mod._day_key(__import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%d")):
                return _body("not valid json{{\n" + json.dumps({"filename": "ok.pdf", "timestamp": 1}) + "\n")
            raise _no_such_key_error()

        mock_client.get_object.side_effect = fake_get_object
        with patch.object(mod, "_client", return_value=mock_client):
            events = mod.read_recent_events("test-bucket", days=1)
        assert len(events) == 1
        assert events[0]["filename"] == "ok.pdf"

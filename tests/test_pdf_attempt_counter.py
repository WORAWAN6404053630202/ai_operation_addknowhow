"""Tests for utils/pdf_attempt_counter.py — the atomic cross-process
retry-attempt counter (2026-10, Infra-C4) that replaced
service/pdf_status_tracker.py's next_attempt_number() as the input to the
bounded give-up checks (_MAX_HANDOFF_ATTEMPTS / _MAX_ATTEMPTS).

The one property worth a dedicated test suite (distinct from
test_pdf_cost_guard.py's near-identical shape): this counter is
deliberately FAIL-CLOSED, not fail-open — a DynamoDB error must return
FAIL_CLOSED_ATTEMPT (comfortably above any real cap), never 1, since
silently resetting to 1 on an infra hiccup is the exact bug this module
exists to fix. No real AWS — the DynamoDB client is a MagicMock throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils.pdf_attempt_counter import FAIL_CLOSED_ATTEMPT, PdfAttemptCounter  # noqa: E402


def _counter_with_mock_client():
    counter = PdfAttemptCounter(table_name="test-attempt-table", region_name="ap-southeast-7")
    mock_client = MagicMock()
    counter._client = mock_client
    return counter, mock_client


@pytest.mark.unit
class TestIncrementAndGet:
    def test_calls_update_item_with_stage_qualified_key(self):
        counter, mock_client = _counter_with_mock_client()
        mock_client.update_item.return_value = {"Attributes": {"attempt_count": {"N": "1"}}}
        counter.increment_and_get("doc.pdf", stage="lambda_extract")
        _, kwargs = mock_client.update_item.call_args
        assert kwargs["TableName"] == "test-attempt-table"
        assert kwargs["Key"]["attempt_key"]["S"] == "doc.pdf#lambda_extract"

    def test_returns_parsed_attempt_count(self):
        counter, mock_client = _counter_with_mock_client()
        mock_client.update_item.return_value = {"Attributes": {"attempt_count": {"N": "3"}}}
        assert counter.increment_and_get("doc.pdf", stage="ec2_build") == 3

    def test_different_stages_use_different_keys_no_cross_stage_leak(self):
        """The actual bug being fixed when a document goes through both
        lambda_extract and ec2_build back to back — each stage's give-up
        check must count independently."""
        counter, mock_client = _counter_with_mock_client()
        mock_client.update_item.return_value = {"Attributes": {"attempt_count": {"N": "1"}}}
        counter.increment_and_get("doc.pdf", stage="lambda_extract")
        counter.increment_and_get("doc.pdf", stage="ec2_build")
        keys_used = [c.kwargs["Key"]["attempt_key"]["S"] for c in mock_client.update_item.call_args_list]
        assert keys_used == ["doc.pdf#lambda_extract", "doc.pdf#ec2_build"]

    def test_dynamodb_error_fails_closed_not_open(self):
        """The core safety property: on any DynamoDB error this must return
        FAIL_CLOSED_ATTEMPT, NEVER 1 — returning 1 here would silently reset
        the counter and defeat the whole retry cap, exactly the bug that
        motivated moving off the old S3-based next_attempt_number()."""
        counter, mock_client = _counter_with_mock_client()
        mock_client.update_item.side_effect = RuntimeError("throttled")
        result = counter.increment_and_get("doc.pdf", stage="ec2_handoff")
        assert result == FAIL_CLOSED_ATTEMPT
        assert result > 3  # comfortably above every real _MAX_ATTEMPTS/_MAX_HANDOFF_ATTEMPTS value

    def test_missing_attributes_in_response_also_fails_closed(self):
        counter, mock_client = _counter_with_mock_client()
        mock_client.update_item.return_value = {}  # malformed/unexpected response shape
        assert counter.increment_and_get("doc.pdf", stage="ec2_build") == FAIL_CLOSED_ATTEMPT

    def test_increment_and_get_never_raises(self):
        counter, mock_client = _counter_with_mock_client()
        mock_client.update_item.side_effect = Exception("anything")
        counter.increment_and_get("doc.pdf", stage="ec2_build")  # must not raise


@pytest.mark.unit
class TestClear:
    def test_calls_delete_item_with_stage_qualified_key(self):
        counter, mock_client = _counter_with_mock_client()
        counter.clear("doc.pdf", stage="ec2_build")
        mock_client.delete_item.assert_called_once()
        _, kwargs = mock_client.delete_item.call_args
        assert kwargs["Key"]["attempt_key"]["S"] == "doc.pdf#ec2_build"

    def test_dynamodb_error_on_clear_does_not_raise(self):
        counter, mock_client = _counter_with_mock_client()
        mock_client.delete_item.side_effect = RuntimeError("table missing")
        counter.clear("doc.pdf", stage="ec2_build")  # must not raise


@pytest.mark.unit
class TestIncrementThenClearRoundTrip:
    def test_clear_then_increment_restarts_at_one(self):
        """Simulates a real table: delete_item actually removes the row, so
        the next increment_and_get starts fresh — the exact behavior a
        legitimate re-upload of a previously-given-up-on filename should see."""
        counter, mock_client = _counter_with_mock_client()
        state = {"count": 0}

        def fake_update_item(TableName, Key, UpdateExpression, ExpressionAttributeValues, ReturnValues):
            state["count"] += 1
            return {"Attributes": {"attempt_count": {"N": str(state["count"])}}}

        def fake_delete_item(TableName, Key):
            state["count"] = 0

        mock_client.update_item.side_effect = fake_update_item
        mock_client.delete_item.side_effect = fake_delete_item

        assert counter.increment_and_get("doc.pdf", stage="ec2_build") == 1
        assert counter.increment_and_get("doc.pdf", stage="ec2_build") == 2
        counter.clear("doc.pdf", stage="ec2_build")
        assert counter.increment_and_get("doc.pdf", stage="ec2_build") == 1


@pytest.mark.unit
class TestGetAttemptCounterSingleton:
    def test_returns_same_instance(self):
        import utils.pdf_attempt_counter as mod
        mod._COUNTER = None  # reset singleton for this test
        c1 = mod.get_attempt_counter()
        c2 = mod.get_attempt_counter()
        assert c1 is c2
        mod._COUNTER = None  # don't leak state to other tests

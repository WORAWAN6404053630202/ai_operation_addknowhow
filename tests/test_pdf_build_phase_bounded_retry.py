"""Tests for the 2026-10 fix to service/sqs_consumer.py's process_extraction_
result() normal (non-handoff) path: before this, a build-phase failure
(classify/split/draft/resolve) had ZERO attempt tracking and just re-raised
unconditionally forever, relying entirely on SQS's own redelivery/DLQ
config to eventually stop — unlike the handoff path's already-proven
_MAX_HANDOFF_ATTEMPTS. This mirrors that same bounded-give-up pattern onto
the normal path: re-raise (so SQS retries) below the attempt cap, but past
it, save a manual-review placeholder and return normally instead of raising
again, so the message isn't redelivered forever.

No real AWS/LLM calls — the S3 client inside process_extraction_result and
pdf_status_tracker's attempt-tracking are both mocked.
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

import service.sqs_consumer as sqs_consumer  # noqa: E402


def _s3_object_body(payload: dict):
    """Mimics boto3's get_object()["Body"].read() shape."""
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode("utf-8")
    return {"Body": body}


@pytest.mark.unit
class TestNormalPathBoundedGiveUp:
    def _call_with_attempt(self, attempt: int, build_error: Exception):
        """Drives process_extraction_result for the "normal" (non-handoff)
        branch with a given simulated attempt number, returning (raised, result)."""
        payload = {"filename": "test.pdf", "s3_raw_pdf_path": "s3://bucket/test.pdf", "pages": []}
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.return_value = _s3_object_body(payload)

        with patch.object(sqs_consumer, "boto3") as mock_boto3, \
             patch.object(sqs_consumer, "_assert_dev_environment"), \
             patch.object(sqs_consumer.pdf_status_tracker, "next_attempt_number", return_value=attempt), \
             patch.object(sqs_consumer.pdf_status_tracker, "write_status"), \
             patch.object(sqs_consumer.pdf_status_tracker, "clear"), \
             patch.object(sqs_consumer, "_build_and_save_review_item", side_effect=build_error), \
             patch.object(sqs_consumer, "_queue_manager") as mock_queue_manager:
            mock_boto3.client.return_value = mock_s3_client
            try:
                result = sqs_consumer.process_extraction_result("test-bucket", "restbiz/extracted/test.pdf.json")
                return False, result, mock_queue_manager
            except RuntimeError:
                return True, None, mock_queue_manager

    def test_below_cap_reraises_so_sqs_retries(self):
        raised, result, mock_queue_manager = self._call_with_attempt(1, RuntimeError("transient build failure"))
        assert raised is True
        mock_queue_manager.save.assert_not_called()

    def test_still_below_cap_at_second_to_last_attempt_reraises(self):
        raised, result, mock_queue_manager = self._call_with_attempt(
            sqs_consumer._MAX_HANDOFF_ATTEMPTS - 1, RuntimeError("still failing")
        )
        assert raised is True
        mock_queue_manager.save.assert_not_called()

    def test_at_cap_gives_up_and_returns_instead_of_raising(self):
        raised, result, mock_queue_manager = self._call_with_attempt(
            sqs_consumer._MAX_HANDOFF_ATTEMPTS, RuntimeError("persistently broken")
        )
        assert raised is False
        assert len(result) == 1
        assert "หยุดลองใหม่" in result[0].relevance_check["reasoning"]
        mock_queue_manager.save.assert_called_once()

    def test_past_cap_also_gives_up(self):
        raised, result, mock_queue_manager = self._call_with_attempt(
            sqs_consumer._MAX_HANDOFF_ATTEMPTS + 2, RuntimeError("way past the cap")
        )
        assert raised is False
        assert len(result) == 1

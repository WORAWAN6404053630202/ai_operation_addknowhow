"""Tests for the 2026-10 fix (data-integrity audit) adding truncation
detection + retry to service/pdf_field_drafting.py's _draft_fields_single_batch.

Background: this call had max_tokens=3000 with NO retry/escalation at all,
unlike its sibling identify_license_topics/identify_knowhow_topics in the
same pipeline — both bumped (1500→4000→6000, 2500→8000) after live-
reproduced truncation on verbose Thai JSON output, and both retry. On
truncation this used to just log an error and silently return every one of
the 13 drafted fields blank, permanently — no retry at any larger budget.

Also checks response.choices[0].finish_reason == "length" explicitly
(OpenAI/OpenRouter's own documented, authoritative truncation signal) in
addition to the pre-existing JSONDecodeError catch — research on this is
consistent that finish_reason is the primary signal; a parse-error-only
check can in principle miss a response that happens to still parse as
valid-but-incomplete JSON at the exact truncation boundary.

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


def _mock_response(fields: dict | None, finish_reason: str = "stop", raw_content: str | None = None):
    resp = MagicMock()
    resp.choices[0].message.content = raw_content if raw_content is not None else json.dumps(fields or {}, ensure_ascii=False)
    resp.choices[0].finish_reason = finish_reason
    resp.usage.prompt_tokens = 50
    resp.usage.completion_tokens = 20
    return resp


@pytest.mark.unit
class TestTruncationDetectionAndRetry:
    def test_clean_response_makes_exactly_one_call(self):
        with patch.object(field_drafting, "OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_response(
                {"department": "กรมสรรพากร"}, finish_reason="stop",
            )
            result = field_drafting._draft_fields_single_batch("เนื้อหา")
        assert mock_client.chat.completions.create.call_count == 1
        assert result["หน่วยงาน"] == "กรมสรรพากร"

    def test_finish_reason_length_triggers_a_retry_at_larger_max_tokens(self):
        """The core fix: finish_reason=="length" alone (even if the partial
        content happened to still be parseable) must trigger a retry — not
        just a JSONDecodeError."""
        with patch.object(field_drafting, "OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.side_effect = [
                _mock_response({"department": "ตัดตอน"}, finish_reason="length"),
                _mock_response({"department": "กรมสรรพากร", "fees": "100 บาท"}, finish_reason="stop"),
            ]
            result = field_drafting._draft_fields_single_batch("เนื้อหา")

        assert mock_client.chat.completions.create.call_count == 2
        second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
        assert second_call_kwargs["max_tokens"] > mock_client.chat.completions.create.call_args_list[0].kwargs["max_tokens"]
        # Must use the RETRY's result, not the truncated first attempt.
        assert result["หน่วยงาน"] == "กรมสรรพากร"
        assert result["ค่าธรรมเนียม"] == "100 บาท"

    def test_json_parse_failure_still_triggers_a_retry(self):
        """Pre-existing behavior (JSONDecodeError) must still work — this
        fix adds finish_reason checking ALONGSIDE it, not instead of it."""
        with patch.object(field_drafting, "OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.side_effect = [
                _mock_response(None, finish_reason="length", raw_content='{"department": "ถูกตัดกลาง'),
                _mock_response({"department": "กรมสรรพากร"}, finish_reason="stop"),
            ]
            result = field_drafting._draft_fields_single_batch("เนื้อหา")
        assert mock_client.chat.completions.create.call_count == 2
        assert result["หน่วยงาน"] == "กรมสรรพากร"

    def test_retry_also_truncated_returns_its_result_without_looping_forever(self):
        with patch.object(field_drafting, "OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.side_effect = [
                _mock_response({"department": "ตัดตอน1"}, finish_reason="length"),
                _mock_response({"department": "ตัดตอน2"}, finish_reason="length"),
            ]
            result = field_drafting._draft_fields_single_batch("เนื้อหา")
        # Exactly _DRAFT_FIELDS_MAX_TOKENS_ATTEMPTS calls, no more — bounded,
        # not an infinite retry loop.
        assert mock_client.chat.completions.create.call_count == len(field_drafting._DRAFT_FIELDS_MAX_TOKENS_ATTEMPTS)
        assert result["หน่วยงาน"] == "ตัดตอน2"

    def test_retry_uses_a_larger_max_tokens_than_the_first_attempt(self):
        attempts = field_drafting._DRAFT_FIELDS_MAX_TOKENS_ATTEMPTS
        assert len(attempts) >= 2
        assert attempts[1] > attempts[0]

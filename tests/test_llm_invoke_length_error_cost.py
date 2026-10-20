"""Test for the 2026-10 fix to utils/llm_call.py: llm_invoke/llm_invoke_async
used to re-raise LengthFinishReasonError (model hit max_tokens before
finishing) BEFORE ever computing/recording cost — meaning a real, billed API
call's cost never reached state.total_cost. Caught live: a Practical/json
call that truncated twice in a row (first attempt, then a widened-budget
retry) was billed ~$0.60 total by OpenRouter but the chat API's reported
total_cost stayed at $0.0036 (only the earlier, successful Supervisor calls),
completely blind to both failed calls.

openai.LengthFinishReasonError carries the real ChatCompletion (with .usage)
on its own `.completion` attribute — this is what the fix extracts cost from.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import utils.llm_call as llm_call_mod  # noqa: E402
from model.conversation_state import ConversationState  # noqa: E402


class _FakeUsage:
    """Mimics openai's CompletionUsage pydantic model (model_dump() shape)."""

    def __init__(self, prompt_tokens, completion_tokens, cached_tokens=0, cache_write_tokens=0):
        self._data = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens_details": {
                "cached_tokens": cached_tokens,
                "cache_write_tokens": cache_write_tokens,
            },
        }

    def model_dump(self):
        return self._data


class FakeLengthFinishReasonError(Exception):
    """Named to match _is_length_error's string check, carries .completion
    like the real openai.LengthFinishReasonError does."""

    def __init__(self, usage):
        super().__init__("Could not parse response content as the length limit was reached")
        self.completion = SimpleNamespace(usage=usage)


@pytest.mark.unit
class TestRecordLengthErrorCost:
    def test_records_nonzero_cost_into_state(self):
        state = ConversationState(session_id="test")
        assert state.total_cost == 0.0
        exc = FakeLengthFinishReasonError(_FakeUsage(prompt_tokens=64104, completion_tokens=6000))

        llm_call_mod._record_length_error_cost(
            exc, state, "anthropic/claude-haiku-4-5", "Test/json", llm_call_mod._ROOT_LOG
        )

        assert state.total_cost > 0.0
        assert state.total_prompt_tokens == 64104
        assert state.total_completion_tokens == 6000

    def test_cache_tokens_are_extracted(self):
        state = ConversationState(session_id="test")
        exc = FakeLengthFinishReasonError(
            _FakeUsage(prompt_tokens=64104, completion_tokens=6000, cache_write_tokens=20038)
        )

        llm_call_mod._record_length_error_cost(
            exc, state, "anthropic/claude-haiku-4-5", "Test/json", llm_call_mod._ROOT_LOG
        )

        assert state.total_cache_write_tokens == 20038

    def test_missing_completion_attr_is_swallowed_not_raised(self):
        state = ConversationState(session_id="test")
        exc = Exception("some other error, no .completion attribute")

        llm_call_mod._record_length_error_cost(
            exc, state, "anthropic/claude-haiku-4-5", "Test/json", llm_call_mod._ROOT_LOG
        )  # must not raise

        assert state.total_cost == 0.0

    def test_zero_usage_does_not_record_cost(self):
        state = ConversationState(session_id="test")
        exc = FakeLengthFinishReasonError(_FakeUsage(prompt_tokens=0, completion_tokens=0))

        llm_call_mod._record_length_error_cost(
            exc, state, "anthropic/claude-haiku-4-5", "Test/json", llm_call_mod._ROOT_LOG
        )

        assert state.total_cost == 0.0

    def test_none_state_does_not_raise(self):
        exc = FakeLengthFinishReasonError(_FakeUsage(prompt_tokens=1000, completion_tokens=500))
        llm_call_mod._record_length_error_cost(
            exc, None, "anthropic/claude-haiku-4-5", "Test/json", llm_call_mod._ROOT_LOG
        )  # must not raise


@pytest.mark.unit
class TestLlmInvokeRecordsLengthErrorCost:
    def test_llm_invoke_records_cost_before_reraising(self):
        state = ConversationState(session_id="test")
        fake_llm = MagicMock()
        fake_llm.model = "anthropic/claude-haiku-4-5"
        fake_llm.invoke.side_effect = FakeLengthFinishReasonError(
            _FakeUsage(prompt_tokens=64104, completion_tokens=6000)
        )

        with patch.object(llm_call_mod, "_is_length_error", return_value=True):
            with pytest.raises(FakeLengthFinishReasonError):
                llm_call_mod.llm_invoke(fake_llm, [], logger=llm_call_mod._ROOT_LOG, label="Test/json", state=state)

        assert state.total_cost > 0.0

    @pytest.mark.asyncio
    async def test_llm_invoke_async_records_cost_before_reraising(self):
        state = ConversationState(session_id="test")
        fake_llm = MagicMock()
        fake_llm.model = "anthropic/claude-haiku-4-5"

        async def _raise(*a, **kw):
            raise FakeLengthFinishReasonError(_FakeUsage(prompt_tokens=64104, completion_tokens=6000))
        fake_llm.ainvoke = _raise

        with patch.object(llm_call_mod, "_is_length_error", return_value=True):
            with pytest.raises(FakeLengthFinishReasonError):
                await llm_call_mod.llm_invoke_async(
                    fake_llm, [], logger=llm_call_mod._ROOT_LOG, label="Test/json", state=state
                )

        assert state.total_cost > 0.0

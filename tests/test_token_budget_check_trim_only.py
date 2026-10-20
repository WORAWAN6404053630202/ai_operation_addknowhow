"""Tests for the 2026-10 fix (core2-M7) to utils/llm_call.py's
run_pending_token_budget_check — removed the summarize branch entirely.

Background: run_pending_token_budget_check used to try
auto_summarize_if_needed() first (a real paid Haiku call that held the
session lock across it) and fall back to trim_messages only if that
produced nothing. But every consumer of state.messages across all 3
personas either filters to the last N messages by a fixed slice or
filters on role in {"user","assistant"}, and summarize_old_messages
always placed its summary message BEFORE the kept-recent slice — the
summary could never land inside what any persona's prompt-building code
actually read. The whole summarize path paid for an LLM call and
persisted a result nothing downstream ever consulted. Removed; only
trim_messages remains (same message-count bound, no LLM call).
code/utils/conversation_summarizer.py (now fully orphaned) was deleted,
and ConversationState.summarize_old_messages (also orphaned) was removed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model.conversation_state import ConversationState  # noqa: E402
from utils.llm_call import run_pending_token_budget_check  # noqa: E402


def _state_with_messages(n: int) -> ConversationState:
    st = ConversationState(session_id="t1", persona_id="practical", context={})
    for i in range(n):
        st.messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"})
    return st


@pytest.mark.unit
class TestRunPendingTokenBudgetCheckOnlyTrims:
    def test_trims_down_to_4_messages_with_no_llm_call(self):
        state = _state_with_messages(12)
        run_pending_token_budget_check(state, label="test")
        assert len(state.messages) == 4
        # Must be the LAST 4, not a summary + fewer recents.
        assert [m["content"] for m in state.messages] == ["msg 8", "msg 9", "msg 10", "msg 11"]

    def test_no_system_summary_message_is_ever_inserted(self):
        state = _state_with_messages(12)
        run_pending_token_budget_check(state, label="test")
        assert not any(m.get("role") == "system" for m in state.messages)

    def test_short_session_is_left_untouched(self):
        state = _state_with_messages(2)
        run_pending_token_budget_check(state, label="test")
        assert len(state.messages) == 2

    def test_failure_in_trim_is_swallowed_not_raised(self):
        state = MagicMock()
        state.trim_messages.side_effect = RuntimeError("boom")
        run_pending_token_budget_check(state, label="test")  # must not raise


@pytest.mark.unit
class TestSummarizerFullyRemoved:
    def test_conversation_summarizer_module_no_longer_exists(self):
        with pytest.raises(ModuleNotFoundError):
            import utils.conversation_summarizer  # noqa: F401

    def test_conversation_state_has_no_summarize_method(self):
        assert not hasattr(ConversationState, "summarize_old_messages")

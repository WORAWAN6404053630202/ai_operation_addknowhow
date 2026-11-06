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

2026-11 update: run_pending_token_budget_check itself (and the
total_tokens-gated trigger around it in llm_call.py/persona_practical.py/
route_v1.py) was removed in a follow-up fix — state.total_tokens is a
lifetime-cumulative counter that never decreases, so the old threshold-
based trigger tripped once and then fired on every subsequent turn for
the rest of a session's life, not "every 4-5 rounds" as intended.
Trimming is now consolidated into exactly one unconditional call per
turn, PersonaSupervisor.handle()'s own end-of-turn
state.trim_messages(keep_last=8) — see TestSupervisorHandleTrimsHistory
below, which replaces the old TestRunPendingTokenBudgetCheckOnlyTrims
(that class tested the now-deleted function directly).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model.conversation_state import ConversationState  # noqa: E402
from model.persona_supervisor import PersonaSupervisor  # noqa: E402


def _state_with_messages(n: int) -> ConversationState:
    st = ConversationState(session_id="t1", persona_id="practical", context={})
    for i in range(n):
        st.messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"})
        st.display_messages.append(st.messages[-1])
    return st


@pytest.mark.unit
class TestSupervisorHandleTrimsHistory:
    """handle() is now the ONE place a turn's history gets trimmed — no
    threshold, runs every turn unconditionally, relying on trim_messages'
    own cheap early-return when there's nothing to trim."""

    @pytest.fixture
    def supervisor(self):
        return PersonaSupervisor(retriever=MagicMock())

    def test_trims_down_to_8_messages_with_no_llm_call(self, supervisor):
        state = _state_with_messages(12)
        with patch.object(supervisor, "_handle_inner", return_value=(state, "reply")), \
             patch.object(supervisor, "_maybe_offer_feedback", side_effect=lambda st, ui, r: r):
            result_state, _ = supervisor.handle(state, "คำถามทดสอบ")

        assert len(result_state.messages) == 8
        # Must be the LAST 8, not a summary + fewer recents.
        assert [m["content"] for m in result_state.messages] == [f"msg {i}" for i in range(4, 12)]

    def test_no_system_summary_message_is_ever_inserted(self, supervisor):
        state = _state_with_messages(12)
        with patch.object(supervisor, "_handle_inner", return_value=(state, "reply")), \
             patch.object(supervisor, "_maybe_offer_feedback", side_effect=lambda st, ui, r: r):
            result_state, _ = supervisor.handle(state, "คำถามทดสอบ")

        assert not any(m.get("role") == "system" for m in result_state.messages)

    def test_short_session_is_left_untouched(self, supervisor):
        state = _state_with_messages(2)
        with patch.object(supervisor, "_handle_inner", return_value=(state, "reply")), \
             patch.object(supervisor, "_maybe_offer_feedback", side_effect=lambda st, ui, r: r):
            result_state, _ = supervisor.handle(state, "คำถามทดสอบ")

        assert len(result_state.messages) == 2

    def test_failure_in_trim_is_swallowed_not_raised(self, supervisor):
        state = MagicMock()
        state.context = {}
        with patch.object(supervisor, "_handle_inner", return_value=(state, "reply")), \
             patch.object(supervisor, "_maybe_offer_feedback", side_effect=lambda st, ui, r: r):
            state.trim_messages.side_effect = RuntimeError("boom")
            supervisor.handle(state, "คำถามทดสอบ")  # must not raise


@pytest.mark.unit
class TestSummarizerFullyRemoved:
    def test_conversation_summarizer_module_no_longer_exists(self):
        with pytest.raises(ModuleNotFoundError):
            import utils.conversation_summarizer  # noqa: F401

    def test_conversation_state_has_no_summarize_method(self):
        assert not hasattr(ConversationState, "summarize_old_messages")

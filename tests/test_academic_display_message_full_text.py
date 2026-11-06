"""Test for the 2026-11 fix to persona_academic.py's history-append path.

Background: _HISTORY_MAX_CHARS/_summarize_for_history used to truncate any
answer over 800 chars to a "...[…]"-suffixed chunk before calling
_append_assistant — and _append_assistant writes that SAME string into both
state.messages AND state.display_messages (via ConversationState.
add_assistant_message_once, which appends one shared dict/content to both
lists). Since state.display_messages is what /session/load and the admin
UI read, a user reloading a long-answer session saw a truncated version of
an answer they actually received in full (the full text was only ever
returned to the caller, never stored). The in-code comment claiming "only
history is compressed... full answer returned unchanged" was simply wrong
about display_messages.

Fix: removed _HISTORY_MAX_CHARS/_summarize_for_history entirely; all 5
finalize call sites now call self._append_assistant(state, ans) with the
full answer, matching what's already returned to the caller.

AcademicPersonaService(retriever=MagicMock()) constructs with no network
access — same pattern as test_academic_groundedness_safety_net.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from model.conversation_state import ConversationState
from model.persona_academic import AcademicPersonaService


@pytest.fixture
def academic():
    return AcademicPersonaService(retriever=MagicMock())


@pytest.mark.unit
class TestAcademicAppendAssistantStoresFullText:
    def test_long_answer_is_not_truncated_in_messages_or_display_messages(self, academic):
        state = ConversationState(session_id="t1", persona_id="academic", context={})
        long_answer = "ก" * 2000  # well over the old 800-char ceiling

        academic._append_assistant(state, long_answer)

        assert state.messages[-1]["content"] == long_answer
        assert state.display_messages[-1]["content"] == long_answer
        assert "[…]" not in state.messages[-1]["content"]
        assert "[…]" not in state.display_messages[-1]["content"]

    def test_short_answer_unaffected(self, academic):
        state = ConversationState(session_id="t1", persona_id="academic", context={})
        short_answer = "คำตอบสั้นๆ"

        academic._append_assistant(state, short_answer)

        assert state.messages[-1]["content"] == short_answer
        assert state.display_messages[-1]["content"] == short_answer

    def test_summarize_for_history_no_longer_exists(self, academic):
        assert not hasattr(academic, "_summarize_for_history")
        assert not hasattr(academic, "_HISTORY_MAX_CHARS")

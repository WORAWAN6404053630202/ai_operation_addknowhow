"""Test for the 2026-11 fix to model/state_manager.py's _trim_state_for_save.

Background: ConversationState.display_messages (the untrimmed record for
/session/load and the admin UI) was only ever capped at 200 items inside
ConversationState.trim_messages — but that only runs on turns that go
through PersonaSupervisor.handle(). A cache-hit turn (route_v1.py's
cache-HIT branches in chat()/_stream_reply() call add_user_message_once/
add_assistant_message_once directly and save) never reaches handle(), so
a session consisting mostly of cache hits could grow display_messages —
and the persisted session file — unboundedly. _trim_state_for_save already
runs on every save() call regardless of path, so the same 200-item
ceiling is added there too.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from model.conversation_state import ConversationState
from model.state_manager import StateManager


@pytest.mark.unit
class TestTrimStateForSaveCapsDisplayMessages:
    def test_caps_display_messages_to_200_on_save(self, tmp_path: Path):
        sm = StateManager(persist_dir=str(tmp_path))
        state = ConversationState(session_id="s_cache_heavy", persona_id="practical", context={})
        for i in range(250):
            state.display_messages.append({"role": "user", "content": f"msg {i}"})

        sm.save("s_cache_heavy", state)

        assert len(state.display_messages) == 200
        assert state.display_messages[0]["content"] == "msg 50"
        assert state.display_messages[-1]["content"] == "msg 249"

    def test_short_display_messages_untouched(self, tmp_path: Path):
        sm = StateManager(persist_dir=str(tmp_path))
        state = ConversationState(session_id="s_short", persona_id="practical", context={})
        state.display_messages.append({"role": "user", "content": "hi"})

        sm.save("s_short", state)

        assert len(state.display_messages) == 1

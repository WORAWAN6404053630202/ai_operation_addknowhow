"""Tests for utils/session_auth.py (2026-10) — closes a real gap found in a
full-project audit: GET /api/v1/sessions / POST /session/load / POST
/session/delete had no access control at all. static/app.js already sent a
stable per-browser id as X-Client-Id on every request; model/state_manager.py
already had owner-filtering logic keyed on context["client_key"]; neither
half was ever connected to the other. These two functions are that
connection. No real FastAPI/service dependency — http_request/state are
duck-typed (see each function's docstring), so these tests use plain stand-in
objects, not a real Request or ConversationState.
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


@pytest.mark.unit
class TestClientKeyFromRequest:
    def test_reads_x_client_id_header(self):
        req = _request_with_header("abc-123")
        assert client_key_from_request(req) == "abc-123"

    def test_strips_whitespace(self):
        req = _request_with_header("  abc-123  ")
        assert client_key_from_request(req) == "abc-123"

    def test_missing_header_returns_empty_string(self):
        req = _request_with_header(None)
        assert client_key_from_request(req) == ""

    def test_empty_header_returns_empty_string(self):
        req = _request_with_header("")
        assert client_key_from_request(req) == ""


@pytest.mark.unit
class TestOwnsSession:
    def test_legacy_session_with_no_stored_key_is_owned_by_anyone(self):
        """The core transition-safety guarantee: a session saved before this
        fix shipped (context has no "client_key" at all) must not suddenly
        become inaccessible to the browser that was already using it."""
        state = _state_with_context({})
        assert owns_session(state, "some-caller-id") is True
        assert owns_session(state, "") is True

    def test_legacy_session_context_none_is_owned_by_anyone(self):
        state = _state_with_context(None)
        assert owns_session(state, "some-caller-id") is True

    def test_matching_client_key_owns_the_session(self):
        state = _state_with_context({"client_key": "browser-A"})
        assert owns_session(state, "browser-A") is True

    def test_mismatched_client_key_does_not_own_the_session(self):
        """The actual bug this closes: before this fix, ANY caller who knew
        (or could enumerate) a session_id could load or delete it —
        regardless of who created it."""
        state = _state_with_context({"client_key": "browser-A"})
        assert owns_session(state, "browser-B") is False

    def test_caller_with_no_client_key_does_not_own_a_keyed_session(self):
        state = _state_with_context({"client_key": "browser-A"})
        assert owns_session(state, "") is False

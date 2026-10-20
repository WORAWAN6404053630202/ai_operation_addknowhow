"""Tests for the 2026-10 fix (core2-M6) to router/route_v1.py's
TOKEN_BUDGET_PER_SESSION check in both /chat and /chat/stream.

Background: conf.py's own comment has always said "Requests that would
exceed this limit are rejected with HTTP 429", but both real call sites
only logged a WARNING and let the turn through regardless — an operator
who set this after the project's own $78 cost-runaway incident, believing
it capped runaway per-session spend, got no actual protection. The
sibling TOKEN_RATE_LIMIT_PER_WINDOW check right next to each already
enforced correctly, making the asymmetry easy to miss. Now enforced as
documented: /chat raises HTTP 429, /chat/stream emits an SSE error event
(a generator can't raise HTTPException directly).

This module (router/route_v1.py) does real service initialization at
import time (~5-10s, no network calls) — same as other route_v1 tests in
this suite.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import conf  # noqa: E402
import router.route_v1 as route_v1  # noqa: E402
from model.conversation_state import ConversationState  # noqa: E402


def _state_with_tokens(session_id: str, total_tokens: int) -> ConversationState:
    st = ConversationState(session_id=session_id, persona_id="practical", context={})
    st.total_prompt_tokens = total_tokens
    st.total_completion_tokens = 0
    return st


def _http_request(ip: str = "1.2.3.4"):
    return SimpleNamespace(client=SimpleNamespace(host=ip), headers=SimpleNamespace(get=lambda k, default=None: default))


@pytest.mark.unit
@pytest.mark.asyncio
class TestChatEnforcesTokenBudget:
    async def test_over_budget_session_is_rejected_with_429(self):
        session_id = "s_budget_over"
        state = _state_with_tokens(session_id, total_tokens=5000)
        request = route_v1.ChatRequest(message="คำถามทดสอบ", session_id=session_id)

        with patch.object(route_v1, "_load_state_async", AsyncMock(return_value=state)), \
             patch.object(conf, "TOKEN_BUDGET_PER_SESSION", 4000), \
             patch.object(conf, "TOKEN_RATE_LIMIT_PER_WINDOW", 0):
            with pytest.raises(HTTPException) as exc_info:
                await route_v1.chat(request, _http_request())

        assert exc_info.value.status_code == 429

    async def test_under_budget_session_is_not_rejected_by_this_check(self):
        """Control case: a session under budget must not be blocked by
        this specific check (it may still fail later for unrelated reasons
        in this minimal test, e.g. no real supervisor call mocked — only
        asserting THIS check doesn't fire)."""
        session_id = "s_budget_under"
        state = _state_with_tokens(session_id, total_tokens=100)
        request = route_v1.ChatRequest(message="คำถามทดสอบ", session_id=session_id)

        with patch.object(route_v1, "_load_state_async", AsyncMock(return_value=state)), \
             patch.object(conf, "TOKEN_BUDGET_PER_SESSION", 4000), \
             patch.object(conf, "TOKEN_RATE_LIMIT_PER_WINDOW", 0):
            try:
                await route_v1.chat(request, _http_request())
            except HTTPException as e:
                assert e.status_code != 429 or "token budget" not in str(e.detail).lower()

    async def test_budget_disabled_when_zero_never_rejects(self):
        session_id = "s_budget_disabled"
        state = _state_with_tokens(session_id, total_tokens=999_999_999)
        request = route_v1.ChatRequest(message="คำถามทดสอบ", session_id=session_id)

        with patch.object(route_v1, "_load_state_async", AsyncMock(return_value=state)), \
             patch.object(conf, "TOKEN_BUDGET_PER_SESSION", 0), \
             patch.object(conf, "TOKEN_RATE_LIMIT_PER_WINDOW", 0):
            try:
                await route_v1.chat(request, _http_request())
            except HTTPException as e:
                assert e.status_code != 429 or "token budget" not in str(e.detail).lower()


@pytest.mark.unit
@pytest.mark.asyncio
class TestStreamEnforcesTokenBudget:
    async def test_over_budget_session_emits_sse_error_event(self):
        session_id = "s_stream_budget_over"
        state = _state_with_tokens(session_id, total_tokens=5000)

        with patch.object(route_v1, "_load_state_async", AsyncMock(return_value=state)), \
             patch.object(conf, "TOKEN_BUDGET_PER_SESSION", 4000), \
             patch.object(conf, "TOKEN_RATE_LIMIT_PER_WINDOW", 0):
            events = []
            async for chunk in route_v1._stream_reply(session_id, "คำถามทดสอบ", client_id="1.2.3.4", client_key=""):
                events.append(chunk)

        assert events, "expected at least one SSE event"
        payload = json.loads(events[0].removeprefix("data: ").strip())
        assert payload["type"] == "error"
        assert "token budget" in payload["message"].lower()

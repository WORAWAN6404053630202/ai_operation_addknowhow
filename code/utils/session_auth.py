# code/utils/session_auth.py (2026-10)
"""Session-ownership helpers for router/route_v1.py's /sessions,
/session/load, /session/delete — pulled out into their own module
(rather than defined inline in route_v1.py) specifically so they're
cheap to import for testing: route_v1.py does real service
initialization (embedding model, Chroma connection) at module import
time, which makes importing it in a unit test slow and environment-
dependent. These two functions have no such dependency.

Background: static/app.js's getClientId()/getHeaders() have generated a
random per-browser id (localStorage, survives reloads) and sent it as
X-Client-Id on every request the whole time this project has existed.
model/state_manager.py's list_sessions() already had owner-filtering logic
keyed on context["client_key"], also unused. The two halves existed
independently and were never connected — nothing ever read X-Client-Id,
nothing ever wrote client_key — which is how /sessions ended up truly
global: every visitor saw (and could load/delete) the 20 most recently
active sessions from ALL visitors, not just their own."""

from __future__ import annotations

from typing import Any


def client_key_from_request(http_request: Any) -> str:
    """http_request is a fastapi.Request — typed Any here to keep this
    module importable without a FastAPI dependency at test time; any object
    exposing `.headers.get(name)` (case-insensitive, as Starlette's
    Headers does) works."""
    return (http_request.headers.get("x-client-id") or "").strip()


def owns_session(state: Any, client_key: str) -> bool:
    """state is a model.conversation_state.ConversationState (or any object
    with a `.context` dict-like attribute).

    A session created before this fix shipped has no context["client_key"]
    at all (the field never existed) — treating that as "anyone may access
    it" rather than "no one may" is a deliberate transition choice: it
    keeps already-in-flight conversations working exactly as they did
    before this fix (no worse than today), and those sessions age out on
    their own via the existing SESSION_RETENTION_DAYS cleanup within a
    week, after which every session has a real owner. A session WITH a
    stored client_key must match the caller's exactly."""
    owner_key = str((getattr(state, "context", None) or {}).get("client_key") or "").strip()
    if not owner_key:
        return True
    return owner_key == client_key

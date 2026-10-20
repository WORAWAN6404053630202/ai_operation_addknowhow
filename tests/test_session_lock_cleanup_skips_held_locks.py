"""Test for the 2026-10 fix (core2-M10) to router/route_v1.py's
_cleanup_old_sessions — the hourly sweep that removes in-memory session
locks for sessions no longer on disk.

Root cause: a brand-new session's .json file doesn't exist until its
first save, so a request mid-flight on one (load -> handle -> save) isn't
yet in state_manager.list_session_ids()'s result — the old code popped
that session's lock anyway, with no check that it was currently held. A
concurrent retry for the same session_id then got a FRESH asyncio.Lock,
acquired it immediately, and ran load->handle->save in parallel with the
original request — whichever save won, the other turn's messages/slots/
cost were silently lost. Fixed by skipping any lock where `.locked()` is
true; it gets swept next run once it's genuinely idle.

This module (router/route_v1.py) does real service initialization at
import time (~5-10s, no network calls) — same as
test_stream_cancellation_saves_in_background.py.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import router.route_v1 as route_v1  # noqa: E402


@pytest.mark.unit
class TestSessionLockCleanupSkipsHeldLocks:
    def setup_method(self):
        route_v1._session_locks.clear()
        route_v1._last_cleanup_ts = 0.0

    def teardown_method(self):
        route_v1._session_locks.clear()
        route_v1._last_cleanup_ts = 0.0

    def test_held_lock_for_a_not_yet_saved_session_is_not_popped(self):
        """The actual bug scenario: a brand-new session's lock is held
        (mid request) but its .json file doesn't exist yet, so it's
        correctly absent from list_session_ids() — that must NOT be
        treated as "stale"."""
        held_lock = asyncio.Lock()
        asyncio.run(held_lock.acquire())
        route_v1._session_locks["s_new_inflight"] = held_lock

        with patch.object(route_v1.state_manager, "purge_older_than_days"), \
             patch.object(route_v1.state_manager, "list_session_ids", return_value=[]):
            route_v1._cleanup_old_sessions()

        assert "s_new_inflight" in route_v1._session_locks, (
            "a currently-held lock must survive the sweep even if its session_id "
            "isn't in list_session_ids() yet — popping it lets a concurrent retry "
            "acquire a fresh lock and race the in-flight request"
        )

    def test_unheld_lock_for_a_gone_session_is_still_popped(self):
        """Control case: the cleanup must still do its actual job — an
        idle lock for a session that's genuinely gone gets removed."""
        idle_lock = asyncio.Lock()
        route_v1._session_locks["s_old_gone"] = idle_lock

        with patch.object(route_v1.state_manager, "purge_older_than_days"), \
             patch.object(route_v1.state_manager, "list_session_ids", return_value=[]):
            route_v1._cleanup_old_sessions()

        assert "s_old_gone" not in route_v1._session_locks

    def test_unheld_lock_for_an_active_session_is_kept(self):
        idle_lock = asyncio.Lock()
        route_v1._session_locks["s_active"] = idle_lock

        with patch.object(route_v1.state_manager, "purge_older_than_days"), \
             patch.object(route_v1.state_manager, "list_session_ids", return_value=["s_active"]):
            route_v1._cleanup_old_sessions()

        assert "s_active" in route_v1._session_locks

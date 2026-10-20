"""Tests for the 2026-10 fix (core2-M11) to router/route_v1.py's
_stream_reply — a client disconnect mid-stream (tab close, refresh,
navigate) used to silently discard a turn that was already fully paid for.

Root cause: Starlette/FastAPI handle a disconnected streaming client by
cancelling the task driving the response's async generator — which throws
CancelledError into _stream_reply at whatever `await` it's currently
suspended on (here, `await asyncio.sleep(0.15)` inside the progress-polling
loop while supervisor.handle() runs on a background thread). CancelledError
is a BaseException, so the old `except Exception:` there never caught it —
the lines that would await the in-flight future's result and save state
were simply never reached. The executor thread backing that future keeps
running regardless (every LLM call it already dispatched still bills), so
the finished answer and updated state were both silently dropped.

Fix: the `finally` block (which runs for every exception type, not just
`Exception`) now checks whether a future is still in flight and, if so,
spawns a background task (_finish_cancelled_stream_turn_in_background) that
awaits it and saves the result once it actually finishes — nothing left to
stream it to, but the turn isn't lost and double-billed on a re-ask.

This module (router/route_v1.py) does real service initialization
(embedding model, local Chroma connection) at import time — slow (~5-10s)
but deterministic and side-effect-free for this environment; no network/
LLM calls happen during the import itself.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import conf  # noqa: E402
import router.route_v1 as route_v1  # noqa: E402
from model.conversation_state import ConversationState  # noqa: E402


def _fake_cache():
    cache = MagicMock()
    cache.get.return_value = None  # force the LLM path, never a cache hit
    cache.get_stats.return_value = {}
    return cache


def _slow_supervisor_call(state, message, progress_cb):
    """Stands in for _run_supervisor_with_progress — simulates an LLM call
    slow enough that the generator is reliably still polling (and
    therefore cancellable mid-flight) when the test cancels it."""
    time.sleep(0.3)
    state.total_cost = 0.05
    return state, "คำตอบที่คำนวณเสร็จแล้ว"


@pytest.mark.unit
@pytest.mark.asyncio
class TestStreamCancellationSavesInBackground:
    async def test_cancelling_mid_turn_still_saves_the_finished_answer(self):
        session_id = "s_cancel_test"
        state = ConversationState(session_id=session_id, persona_id="practical", context={})
        saved_states = []

        async def _fake_save_state_async(sid, st):
            saved_states.append((sid, st))

        with patch.object(route_v1, "_load_state_async", AsyncMock(return_value=state)), \
             patch.object(route_v1, "_save_state_async", side_effect=_fake_save_state_async), \
             patch.object(route_v1, "get_cache", return_value=_fake_cache()), \
             patch.object(conf, "TOKEN_BUDGET_PER_SESSION", 0), \
             patch.object(conf, "TOKEN_RATE_LIMIT_PER_WINDOW", 0), \
             patch.object(route_v1, "_run_supervisor_with_progress", side_effect=_slow_supervisor_call):

            gen = route_v1._stream_reply(session_id, "คำถามทดสอบ", client_id="1.2.3.4", client_key="")

            before_tasks = set(route_v1._background_tasks)
            task = asyncio.ensure_future(gen.__anext__())
            # Let the generator run past state-load/cache-check and into the
            # progress-polling loop's `await asyncio.sleep(0.15)` — the real
            # suspension point a client disconnect would land on.
            await asyncio.sleep(0.05)
            assert not task.done(), "generator should still be polling, not finished yet"

            # Simulate Starlette cancelling the task driving the response
            # iterator on client disconnect.
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # The lock-held session lock must have been handed off to a
            # background task, not released synchronously with no rescue.
            new_tasks = set(route_v1._background_tasks) - before_tasks
            assert len(new_tasks) == 1, "expected exactly one background rescue task to be spawned"
            rescue_task = new_tasks.pop()

            # Drive the rescue task to completion (awaits the still-running
            # 0.3s fake LLM call, then saves).
            await rescue_task

        assert saved_states, "the finished turn must still be saved even though the client disconnected"
        saved_session_id, saved_state = saved_states[-1]
        assert saved_session_id == session_id
        assert saved_state.total_cost == 0.05

    async def test_cancelling_before_any_llm_call_starts_does_not_spawn_a_rescue(self):
        """Control case: if cancellation happens before a future even
        exists (e.g. during state load), there is nothing to rescue — must
        not spawn a background task or raise."""
        session_id = "s_cancel_early_test"
        state = ConversationState(session_id=session_id, persona_id="practical", context={})

        async def _slow_load(sid):
            await asyncio.sleep(0.3)
            return state

        with patch.object(route_v1, "_load_state_async", side_effect=_slow_load), \
             patch.object(route_v1, "get_cache", return_value=_fake_cache()), \
             patch.object(conf, "TOKEN_BUDGET_PER_SESSION", 0), \
             patch.object(conf, "TOKEN_RATE_LIMIT_PER_WINDOW", 0):

            gen = route_v1._stream_reply(session_id, "คำถามทดสอบ", client_id="1.2.3.4", client_key="")
            before_tasks = set(route_v1._background_tasks)
            task = asyncio.ensure_future(gen.__anext__())
            await asyncio.sleep(0.05)
            assert not task.done()

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            new_tasks = set(route_v1._background_tasks) - before_tasks
            assert new_tasks == set()

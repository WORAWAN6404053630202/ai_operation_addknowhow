"""Tests for the 2026-10 fix (core2-M4) to persona_supervisor.py's
classifier pre-warm pool.

Background: 6 intent classifiers (_thanks_llm_check, _greeting_llm_check,
_smalltalk_llm_check, _switch_without_target_llm_check, _legal_q_llm_check,
_prewarm_style) are submitted to a shared ThreadPoolExecutor and waited on
for up to 1.5s so later sections of _handle_inner can read their cached
result instead of calling the LLM again serially. Two problems with the
old code: (1) each classifier wrote directly into state.context's cache
dict as its own last step, regardless of which thread called it — a
classifier still running past the 1.5s wait kept going on its executor
thread and could write into state.context at the same moment
route_v1.py's state_manager.save() -> state.model_dump() was iterating it,
raising "dictionary changed size during iteration"; (2) a classifier that
didn't finish within 1.5s left no cache entry, so the synchronous
fallback a few lines later called the SAME classifier a second time —
dispatching a real, fully duplicate LLM call for input already in flight.

Fix: every classifier now accepts _skip_cache_write (default False, so
every call site OTHER than the pre-warm dispatcher is unaffected) — when
True, it returns its result instead of writing to state.context itself.
Only the dispatch block (always running on the main thread) writes the
cache, and only for futures confirmed done() within the wait() window —
a future that times out keeps running but its eventual result is
discarded rather than ever touching state.context off-thread.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model.conversation_state import ConversationState  # noqa: E402
from model.persona_supervisor import PersonaSupervisor  # noqa: E402


def _supervisor():
    return PersonaSupervisor.__new__(PersonaSupervisor)


def _state():
    # NOTE: context must be non-empty — several classifiers below read it via
    # `(state.context or {}).setdefault(...)`, and an empty dict `{}` is
    # falsy in Python, so `state.context or {}` would silently operate on a
    # throwaway temporary dict instead of the real one. Pre-existing
    # behavior unrelated to core2-M4 (not something this fix touches) —
    # worked around here rather than exercised, since a real session's
    # context is realistically never actually empty by the time a
    # classifier runs (e.g. /greeting already stamped client_key).
    return ConversationState(session_id="t1", persona_id="practical", context={"_test_marker": True})


@pytest.mark.unit
class TestSkipCacheWriteDefaultsPreserveExistingBehavior:
    """Every direct (non-prewarm) call site omits _skip_cache_write, so its
    default (False) must reproduce the original always-write-to-cache
    behavior exactly."""

    def test_thanks_check_writes_cache_by_default(self):
        sup = _supervisor()
        state = _state()
        sup._llm_thanks_call = lambda *a, **kw: {"is_thanks": True, "confidence": 0.9}
        result = sup._thanks_llm_check("ขอบคุณครับ", state)
        assert result is True
        assert state.context["_thanks_llm_cache"]["ขอบคุณครับ"] is True

    def test_prewarm_style_writes_cache_by_default(self):
        sup = _supervisor()
        state = _state()
        with patch.object(sup, "_infer_user_style_request_hybrid", return_value={"wants_long": True}):
            sup._prewarm_style("ขอแบบละเอียด", "", state)
        assert state.context["_style_pw_cache"][("ขอแบบละเอียด"[:80], "")] == {"wants_long": True}


@pytest.mark.unit
class TestSkipCacheWriteTrueNeverTouchesStateContext:
    """The actual fix: _skip_cache_write=True must return the result
    WITHOUT writing anything into state.context — proving a classifier
    called this way (from the pre-warm pool) can never mutate shared state
    off the main thread."""

    def test_thanks_check_returns_result_without_writing_cache(self):
        sup = _supervisor()
        state = _state()
        sup._llm_thanks_call = lambda *a, **kw: {"is_thanks": True, "confidence": 0.9}
        result = sup._thanks_llm_check("ขอบคุณครับ", state, True)
        assert result is True
        assert "_thanks_llm_cache" not in state.context

    def test_legal_q_check_returns_result_without_writing_cache(self):
        sup = _supervisor()
        state = _state()
        sup._llm_legal_q_call = lambda *a, **kw: {"is_legal": True, "confidence": 0.8}
        result = sup._legal_q_llm_check("ขอข้อมูลค่าใช้จ่าย", state, True)
        assert result is True
        assert "_legal_q_llm_cache" not in state.context

    def test_prewarm_style_returns_result_without_writing_cache(self):
        sup = _supervisor()
        state = _state()
        with patch.object(sup, "_infer_user_style_request_hybrid", return_value={"wants_long": True}):
            result = sup._prewarm_style("ขอแบบละเอียด", "", state, True)
        assert result == {"wants_long": True}
        assert "_style_pw_cache" not in state.context


@pytest.mark.unit
class TestDispatchOnlyCommitsFuturesThatFinishedInTime:
    """Simulates the real pre-warm dispatch pattern (submit with
    _skip_cache_write=True, wait with a short timeout, commit only
    futures that are done()) directly, without driving the full
    ~14,000-line _handle_inner. A slow classifier's eventual result must
    never reach state.context even after it finishes."""

    def test_slow_classifier_never_mutates_state_even_after_it_finishes(self):
        import concurrent.futures as cf

        sup = _supervisor()
        state = _state()
        pool = cf.ThreadPoolExecutor(max_workers=2)
        slow_call_started = threading.Event()

        def _slow_llm_call(*a, **kw):
            slow_call_started.set()
            time.sleep(0.3)
            return {"is_thanks": True, "confidence": 0.9}

        sup._llm_thanks_call = _slow_llm_call
        fut = pool.submit(sup._thanks_llm_check, "ขอบคุณครับ", state, True)
        try:
            assert slow_call_started.wait(timeout=1.0)
            cf.wait([fut], timeout=0.05)  # much shorter than the 0.3s call — times out
            assert not fut.done()

            # Dispatch-block logic: only commit if done() within the window.
            if fut.done():
                state.context.setdefault("_thanks_llm_cache", {})["ขอบคุณครับ"] = fut.result()

            assert "_thanks_llm_cache" not in state.context, (
                "a classifier that didn't finish within the wait window must not "
                "have its result committed to state.context"
            )

            # Let the background call actually finish, same as production where
            # the thread keeps running after the turn has already returned.
            result = fut.result(timeout=2.0)
            assert result is True
            # Confirms the fix: even now that it's done, nothing auto-writes to
            # state.context — only an explicit post-wait commit step would, and
            # the real dispatch block never re-checks a future after its window.
            assert "_thanks_llm_cache" not in state.context
        finally:
            pool.shutdown(wait=True)

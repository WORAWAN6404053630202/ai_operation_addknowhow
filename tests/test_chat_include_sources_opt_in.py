"""
Tests for the Round 3 Phase 7 (.claude/plans/mutable-seeking-wave.md)
opt-in `include_sources` field on /chat: lets an evaluation caller (the
Chroma-vs-ES LangSmith groundedness comparison) get back the exact
retrieved-doc payload the model saw, without changing response shape/
payload for any existing caller that doesn't ask for it.

Two tiers:
1. A real behavioral test of `ChatRequest` itself (cheap, pure Pydantic —
   no service init needed).
2. Source-inspection regression pins for the route_v1.py/persona_practical.py
   wiring (pop-before-save, conditional-only inclusion, the stash site) —
   same established pattern this codebase already uses elsewhere for large/
   stateful code that's awkward to drive end-to-end in a unit test (see
   tests/test_es_backend_awareness_fixes.py).
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


@pytest.mark.unit
class TestChatRequestIncludeSourcesField:
    def test_defaults_to_false(self):
        import router.route_v1 as route_v1
        req = route_v1.ChatRequest(message="test")
        assert req.include_sources is False

    def test_can_be_set_true(self):
        import router.route_v1 as route_v1
        req = route_v1.ChatRequest(message="test", include_sources=True)
        assert req.include_sources is True


@pytest.mark.unit
class TestRetrievedSourcesWiringIsBackendAware:
    def test_chat_pops_last_retrieved_docs_before_saving_state(self):
        """chat() has an EARLIER, unrelated _save_state_async call on the
        cache-HIT path (never has _last_retrieved_docs set, since
        supervisor.handle() is skipped entirely on a cache hit) — the real
        invariant is about the save call that immediately FOLLOWS the pop,
        on the cache-MISS path, not "any save anywhere in the function"."""
        import router.route_v1 as route_v1
        source = inspect.getsource(route_v1.chat)
        pop_idx = source.find('.pop("_last_retrieved_docs"')
        assert pop_idx != -1, "expected a pop() of _last_retrieved_docs in chat()"
        next_save_idx = source.find("await _save_state_async(session_id, state)", pop_idx)
        assert next_save_idx != -1, "expected a _save_state_async call after the pop"
        # Nothing besides whitespace/comments between the pop and its paired save.
        between = source[pop_idx:next_save_idx]
        assert between.count("\n") <= 3, f"pop and its paired save are no longer adjacent: {between!r}"

    def test_retrieved_sources_only_included_when_opted_in(self):
        import router.route_v1 as route_v1
        source = inspect.getsource(route_v1.chat)
        assert "request.include_sources" in source
        assert "retrieved_sources" in source

    def test_persona_practical_stashes_docs_json_onto_state_context(self):
        import model.persona_practical as prac_mod
        source = inspect.getsource(prac_mod)
        assert 'state.context["_last_retrieved_docs"] = docs_json' in source

"""
Regression test for a real incident (2026-09): the internal retrieve→
re-retrieve recursion in PracticalPersonaService.handle() had no hard cap.
When retrieval genuinely returns 0 docs every time (e.g. a broken vectorstore
connection), the LLM keeps returning action="retrieve" with a freshly-worded
query and handle() recurses via "__auto_post_retrieve__" with no limit.
Live-reproduced: one single question recursed for 2h22m and accumulated
5.4M+ tokens before being killed manually.

Root cause: a counter (_auto_post_retrieve_guard) was already being
incremented on every recursive call but was never read/checked anywhere —
a dead tripwire. The fix reads it right after incrementing and aborts with
a graceful fallback once it exceeds MAX_AUTO_RETRIEVE_LOOPS.
"""
import pytest

from model.persona_practical import PracticalPersonaService


@pytest.fixture
def practical() -> PracticalPersonaService:
    return PracticalPersonaService(retriever=None)


@pytest.mark.unit
class TestAutoRetrieveLoopCap:

    def test_cap_aborts_without_calling_llm(self, practical, fresh_state, monkeypatch):
        """Once the internal-recursion counter exceeds the cap, handle() must
        return the graceful fallback immediately — WITHOUT reaching the LLM
        at all. Monkeypatching _call_llm_json to raise proves the early-exit
        guard fires before any LLM call, not just that the final answer
        happens to look like the fallback."""
        def _boom(*a, **kw):
            raise AssertionError("LLM should not be called once the auto-retrieve cap is exceeded")
        monkeypatch.setattr(practical, "_call_llm_json", _boom)

        fresh_state.context["_auto_post_retrieve_guard"] = 6  # already above default cap (5)
        _, reply = practical.handle(fresh_state, "__auto_post_retrieve__", _internal=True)

        assert "ไม่พบข้อมูลที่ตรงกับคำถามนี้" in reply
        assert fresh_state.context.get("_info_gap_detected") is True
        # Counter must reset — a later, unrelated question must not inherit a tripped cap.
        assert fresh_state.context.get("_auto_post_retrieve_guard") == 0

    def test_below_cap_proceeds_past_guard(self, practical, fresh_state, monkeypatch):
        """Below the cap, handle() must NOT take the early-exit path — it
        should proceed far enough to actually call the LLM."""
        called = {"hit": False}

        def _fake_llm_json(prompt, *a, **kw):
            called["hit"] = True
            return {
                "input_type": "new_question", "analysis": "test",
                "action": "answer",
                "execution": {"answer": "ok", "context_update": {}},
            }
        monkeypatch.setattr(practical, "_call_llm_json", _fake_llm_json)

        fresh_state.context["_auto_post_retrieve_guard"] = 1  # well below cap
        practical.handle(fresh_state, "__auto_post_retrieve__", _internal=True)

        assert called["hit"] is True

    def test_fresh_external_call_resets_counter(self, practical, fresh_state, monkeypatch):
        """A genuinely new user turn (_internal=False) must clear any leftover
        counter from a previous turn's internal recursion — otherwise a
        single earlier runaway could poison every later unrelated question in
        the same session."""
        def _fake_llm_json(prompt, *a, **kw):
            # By the time this is reached, the guard-key reset (for _internal=False)
            # must already have happened — assert it here rather than after handle()
            # returns, since later code could re-populate the same key for other reasons.
            assert "_auto_post_retrieve_guard" not in fresh_state.context
            return {
                "input_type": "new_question", "analysis": "test",
                "action": "answer",
                "execution": {"answer": "ok", "context_update": {}},
            }
        monkeypatch.setattr(practical, "_call_llm_json", _fake_llm_json)

        fresh_state.context["_auto_post_retrieve_guard"] = 4  # leftover from a prior turn
        practical.handle(fresh_state, "สวัสดีครับ ขอถามเรื่องใบอนุญาต", _internal=False)

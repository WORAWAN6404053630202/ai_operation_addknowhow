"""
Tests for _retry_if_entity_coverage_incomplete (2026-10) — the deterministic
safety net that retries once when an answer completely omits one entity's
section despite _entity_type_explicitly_skipped and both entity types being
genuinely present in state.current_docs.

Constructs PracticalPersonaService via __new__ (bypassing __init__, which does
real Chroma startup queries) and monkeypatches _call_llm_json_with_truncation_retry
directly — no live API calls, deterministic and free to run.
"""
import pytest

from model.persona_practical import PracticalPersonaService
from model.conversation_state import ConversationState


def _new_practical():
    return PracticalPersonaService.__new__(PracticalPersonaService)


def _state_with_both_entities():
    state = ConversationState(session_id="t", persona_id="practical", context={
        "_entity_type_explicitly_skipped": True,
    })
    state.current_docs = [
        {"content": "...", "metadata": {"entity_type_normalized": "บุคคลธรรมดา"}},
        {"content": "...", "metadata": {"entity_type_normalized": "นิติบุคคล"}},
    ]
    return state


@pytest.mark.unit
class TestEntityCoverageRetry:
    def test_both_entities_present_no_retry(self):
        svc = _new_practical()
        svc._call_llm_json_with_truncation_retry = lambda *a, **k: pytest.fail("should not retry")
        state = _state_with_both_entities()
        ans = "บุคคลธรรมดา ต้องใช้ A, B. นิติบุคคล ต้องใช้ C, D."
        result = svc._retry_if_entity_coverage_incomplete(ans, "prompt", state)
        assert result == ans

    def test_missing_entity_triggers_retry_and_uses_improved_answer(self):
        svc = _new_practical()
        calls = []

        def _fake_retry(prompt, state=None):
            calls.append(prompt)
            return {
                "action": "answer",
                "execution": {"answer": "บุคคลธรรมดา ต้องใช้ A, B. นิติบุคคล ต้องใช้ C, D."},
            }

        svc._call_llm_json_with_truncation_retry = _fake_retry
        state = _state_with_both_entities()
        bad_ans = "บุคคลธรรมดา ต้องใช้ A, B."  # นิติบุคคล never mentioned
        result = svc._retry_if_entity_coverage_incomplete(bad_ans, "prompt", state)

        assert len(calls) == 1, "retry must fire exactly once when an entity is missing"
        assert "นิติบุคคล" in calls[0], "corrective prompt must name the missing entity"
        assert "นิติบุคคล" in result, "final answer must come from the improved retry"
        assert "บุคคลธรรมดา" in result

    def test_retry_still_incomplete_keeps_improvement_only_if_strictly_better(self):
        svc = _new_practical()

        def _fake_retry(prompt, state=None):
            # Retry still omits the entity — no improvement at all.
            return {"action": "answer", "execution": {"answer": "บุคคลธรรมดา ต้องใช้ A, B."}}

        svc._call_llm_json_with_truncation_retry = _fake_retry
        state = _state_with_both_entities()
        bad_ans = "บุคคลธรรมดา ต้องใช้ A, B."
        result = svc._retry_if_entity_coverage_incomplete(bad_ans, "prompt", state)

        # No improvement over the original -> original answer is kept unchanged.
        assert result == bad_ans

    def test_retry_call_failure_keeps_original_answer(self):
        svc = _new_practical()

        def _raise(*a, **k):
            raise RuntimeError("boom")

        svc._call_llm_json_with_truncation_retry = _raise
        state = _state_with_both_entities()
        bad_ans = "บุคคลธรรมดา ต้องใช้ A, B."
        result = svc._retry_if_entity_coverage_incomplete(bad_ans, "prompt", state)
        assert result == bad_ans

    def test_single_entity_in_docs_never_triggers_retry(self):
        svc = _new_practical()
        svc._call_llm_json_with_truncation_retry = lambda *a, **k: pytest.fail("should not retry")
        state = ConversationState(session_id="t", persona_id="practical", context={
            "_entity_type_explicitly_skipped": True,
        })
        state.current_docs = [
            {"content": "...", "metadata": {"entity_type_normalized": "บุคคลธรรมดา"}},
        ]
        ans = "บุคคลธรรมดา ต้องใช้ A, B."
        result = svc._retry_if_entity_coverage_incomplete(ans, "prompt", state)
        assert result == ans

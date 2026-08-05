"""
Tests for the confidence-tiered entity_type auto-fill (2026-07-31, Part 2 of the
regex-tightening fix — see entity_false_positive_regex_tightening.md).

Background: Part 1 removed ambiguous colloquial terms (bare "คนเดียว", "ทำเอง",
"เปิดเอง", etc.) from the direct-write regex, since a false positive there was a
silent, PERMANENT wrong slot value (this project's "no slot re-ask" rule). Part 2
gives those colloquial phrasings a supervised second chance: they now route
through the existing LLM fallback (_entity_type_llm_fallback), and the result is
tagged "inferred" (vs "explicit" for a direct regex/fuzzy match) so the answer
prompt can add a one-line implicit confirmation ("เข้าใจว่าร้านเป็น...นะครับ")
instead of silently locking in the assumption.

These tests exercise the actual instance methods (not just regex patterns) by
constructing personas via `__new__` (bypassing `__init__`, which does real Chroma
startup queries against a retriever) and monkeypatching the one LLM-calling method
each test touches. No live API/retriever calls — deterministic and free to run.
"""
import pytest

from model.persona_supervisor import PersonaSupervisor
from model.persona_practical import PracticalPersonaService
from model.persona_academic import AcademicPersonaService
from model.conversation_state import ConversationState


class _FakePractical:
    """Stand-in for PersonaSupervisor._practical — only _retrieve_docs is ever
    called by _apply_slot_change_if_detected in the paths these tests exercise."""

    def _retrieve_docs(self, *args, **kwargs):
        return []


def _new_supervisor(entity_llm_result=None, location_llm_result=None, area_llm_result=None):
    sup = PersonaSupervisor.__new__(PersonaSupervisor)
    sup._practical = _FakePractical()
    sup._entity_type_llm_fallback = lambda q, st, last_query="": entity_llm_result
    sup._location_llm_fallback = lambda q, st: location_llm_result
    sup._area_size_llm_fallback = lambda q, st: area_llm_result
    return sup


@pytest.mark.unit
class TestPrefillEntityTypeSourceTagging:
    """_prefill_slots_from_message — the INITIAL fill (slot not yet known)."""

    def _state_with_queue(self):
        return ConversationState(
            session_id="t", persona_id="practical",
            context={"topic_slot_queue": [{"key": "entity_type", "options": ["บุคคลธรรมดา", "นิติบุคคล"]}]},
        )

    def test_explicit_term_tagged_explicit_no_confirmation_flag(self):
        sup = _new_supervisor()
        state = self._state_with_queue()
        sup._prefill_slots_from_message(state, "เป็นบุคคลธรรมดาครับ")
        assert state.get_collected_slots().get("entity_type") == "บุคคลธรรมดา"
        assert state.context.get("entity_type_source") == "explicit"
        assert not state.context.get("_entity_type_just_inferred")

    def test_colloquial_term_via_llm_fallback_tagged_inferred(self):
        sup = _new_supervisor(entity_llm_result="บุคคลธรรมดา")
        state = self._state_with_queue()
        sup._prefill_slots_from_message(state, "ทำเองเปิดร้านคนเดียวค่ะ ไม่มีหุ้นส่วน")
        assert state.get_collected_slots().get("entity_type") == "บุคคลธรรมดา"
        assert state.context.get("entity_type_source") == "inferred"
        assert state.context.get("_entity_type_just_inferred") is True

    def test_llm_fallback_returns_none_slot_stays_unset(self):
        # The live-reproduced false positive: "ทำเอง" appears but there's no real
        # entity-type signal — the LLM fallback (correctly) returns None and the
        # slot must NOT be silently filled.
        sup = _new_supervisor(entity_llm_result=None)
        state = self._state_with_queue()
        sup._prefill_slots_from_message(state, "ค่าธรรมเนียมทำเองกับจ้างตัวแทนยื่นต่างกันไหมครับ")
        assert "entity_type" not in state.get_collected_slots()
        assert "entity_type_source" not in (state.context or {})
        assert not (state.context or {}).get("_entity_type_just_inferred")


@pytest.mark.unit
class TestSlotChangeEntityTypeSourceTagging:
    """_apply_slot_change_if_detected — SWITCHING an already-known entity_type."""

    def _state_with_existing_entity(self, value="นิติบุคคล"):
        return ConversationState(
            session_id="t", persona_id="practical",
            context={"collected_slots": {"entity_type": value, "entity_type_normalized": value}},
        )

    def test_explicit_switch_tagged_explicit(self):
        sup = _new_supervisor()
        state = self._state_with_existing_entity("นิติบุคคล")
        sup._apply_slot_change_if_detected(state, "เปลี่ยนเป็นบุคคลธรรมดาค่ะ")
        assert state.get_collected_slots().get("entity_type") == "บุคคลธรรมดา"
        assert state.context.get("entity_type_source") == "explicit"
        assert not state.context.get("_entity_type_just_inferred")

    def test_colloquial_switch_via_llm_fallback_tagged_inferred(self):
        sup = _new_supervisor(entity_llm_result="บุคคลธรรมดา")
        state = self._state_with_existing_entity("นิติบุคคล")
        sup._apply_slot_change_if_detected(state, "ทำเองเปิดร้านคนเดียวค่ะ")
        assert state.get_collected_slots().get("entity_type") == "บุคคลธรรมดา"
        assert state.context.get("entity_type_source") == "inferred"
        assert state.context.get("_entity_type_just_inferred") is True


@pytest.mark.unit
class TestMaybeBuildSlotQueueEntityTypeSourceTagging:
    """
    _maybe_build_slot_queue_from_docs's "always persist entity_type if explicitly
    stated" early block (a 3rd, separate write site from the two above — fires
    before slot-queue building even starts). Uses _infer_entity_type_from_query,
    which is pure Tier-1 regex with no LLM fallback, so anything it returns is
    by construction "explicit" — never "inferred".
    """

    def test_explicit_term_in_topic_establishing_message_tagged_explicit(self):
        sup = PersonaSupervisor.__new__(PersonaSupervisor)
        sup._practical = _FakePractical()
        state = ConversationState(session_id="t", persona_id="practical", context={})
        state.current_docs = [{"metadata": {"license_type": "ใบอนุญาตจำหน่ายสุรา"}}]
        sup._maybe_build_slot_queue_from_docs(state, "เป็นบุคคลธรรมดาครับ")
        assert state.get_collected_slots().get("entity_type") == "บุคคลธรรมดา"
        assert state.context.get("entity_type_source") == "explicit"


@pytest.mark.unit
class TestOneShotFlagLifecycle:
    """
    The confirmation flag must fire exactly once — cleared at the point a reply
    is actually committed (_append_assistant), not at prompt-build time, so it
    survives a handle() recursion/retry within the same turn but never leaks
    into the NEXT turn.
    """

    def test_practical_append_assistant_clears_flag(self):
        pp = PracticalPersonaService.__new__(PracticalPersonaService)
        state = ConversationState(
            session_id="t", persona_id="practical",
            context={"_entity_type_just_inferred": True},
        )
        pp._append_assistant(state, "เข้าใจว่าร้านเป็นบุคคลธรรมดานะครับ — คำตอบ...")
        assert not state.context.get("_entity_type_just_inferred")

    def test_academic_append_assistant_clears_flag(self):
        ac = AcademicPersonaService.__new__(AcademicPersonaService)
        state = ConversationState(
            session_id="t", persona_id="academic",
            context={"_entity_type_just_inferred": True},
        )
        ac._append_assistant(state, "เข้าใจว่าร้านเป็นบุคคลธรรมดานะครับ — คำตอบ...")
        assert not state.context.get("_entity_type_just_inferred")

    def test_empty_content_does_not_clear_flag(self):
        # Guards against silently discarding the one-shot opportunity on a no-op
        # append (empty content is a genuine no-op, not a delivered reply).
        pp = PracticalPersonaService.__new__(PracticalPersonaService)
        state = ConversationState(
            session_id="t", persona_id="practical",
            context={"_entity_type_just_inferred": True},
        )
        pp._append_assistant(state, "")
        assert state.context.get("_entity_type_just_inferred") is True


@pytest.mark.unit
class TestPromptInjectionWiring:
    """
    Structural checks (source inspection) that the implicit-confirmation
    instruction is actually wired into both personas' answer-prompt builders —
    behavioral verification (does the LLM actually open with the confirmation
    line) requires a live retriever/LLM call and is covered by live testing,
    not this unit suite.
    """

    def test_practical_entity_filter_hint_reads_the_flag(self):
        import inspect
        src = inspect.getsource(PracticalPersonaService.handle)
        assert '"_entity_type_just_inferred"' in src
        assert "ENTITY TYPE WAS INFERRED" in src

    def test_academic_build_final_prompt_reads_the_flag(self):
        import inspect
        src = inspect.getsource(AcademicPersonaService._build_final_prompt)
        assert '"_entity_type_just_inferred"' in src
        assert "ENTITY TYPE WAS INFERRED" in src

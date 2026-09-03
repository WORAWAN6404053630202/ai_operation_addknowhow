"""
Unit tests for the multi-entity_type comparison support added 2026-09
(persona_supervisor.py's _apply_slot_change_if_detected).

Background: QA review found that "compare bank A vs bank B" questions
already got a correct both/and answer (via _multi_dept_mentioned +
persona_practical.py's established-topic guarantee + prompts_practical.py's
EXCEPTION B), but the same class of question for entity types (e.g. "ห้าง
หุ้นส่วนจำกัด กับ บุคคลธรรมดา เอกสารต่างกันไหม") had no equivalent — the
existing single-value entity detection silently picked one side, and the
"Entity/registration filter (CRITICAL)" prompt rule then suppressed the
other. This mirrors the department pattern: _multi_entity_mentioned is set
in state.context when the query names both บุคคลธรรมดา and นิติบุคคล in the
same turn, consumed by persona_practical.py's guaranteed-fetch block and a
new prompt exception.

PersonaSupervisor is constructed with retriever=MagicMock() (no network —
ChatOpenAI's constructor doesn't validate credentials). _location_llm_fallback
is stubbed to avoid a background-thread network call unrelated to what's
under test here (old_area is empty on a fresh state so the area fallback
never fires; only location does).
"""
from unittest.mock import MagicMock

import pytest

from model.persona_supervisor import PersonaSupervisor
from model.conversation_state import ConversationState


@pytest.fixture
def supervisor():
    sup = PersonaSupervisor(retriever=MagicMock())
    sup._location_llm_fallback = MagicMock(return_value=None)
    return sup


def _fresh_state(context=None) -> ConversationState:
    return ConversationState(session_id="test", persona_id="practical", context=context or {})


@pytest.mark.unit
class TestMultiEntityMentionedFlag:

    def test_genuine_comparison_query_sets_both_entity_types(self, supervisor):
        state = _fresh_state()
        supervisor._apply_slot_change_if_detected(
            state, "ห้างหุ้นส่วนจำกัด กับ บุคคลธรรมดา เอกสารต่างกันไหมคะ"
        )
        assert state.context.get("_multi_entity_mentioned") == ["บุคคลธรรมดา", "นิติบุคคล"]

    def test_alternate_phrasing_of_comparison_also_sets_flag(self, supervisor):
        state = _fresh_state()
        supervisor._apply_slot_change_if_detected(
            state, "เทียบระหว่างนิติบุคคลกับบุคคลธรรมดาให้หน่อยค่ะ"
        )
        assert state.context.get("_multi_entity_mentioned") == ["บุคคลธรรมดา", "นิติบุคคล"]

    def test_single_entity_mention_does_not_set_flag(self, supervisor):
        state = _fresh_state()
        supervisor._apply_slot_change_if_detected(state, "เป็นนิติบุคคลค่ะ ต้องใช้เอกสารอะไรบ้าง")
        assert state.context.get("_multi_entity_mentioned") is None

    def test_other_single_entity_mention_does_not_set_flag(self, supervisor):
        state = _fresh_state()
        supervisor._apply_slot_change_if_detected(state, "เป็นบุคคลธรรมดาค่ะ ค่าธรรมเนียมเท่าไหร่")
        assert state.context.get("_multi_entity_mentioned") is None

    def test_no_entity_mention_does_not_set_flag(self, supervisor):
        state = _fresh_state()
        supervisor._apply_slot_change_if_detected(state, "ขอสอบถามเรื่องใบอนุญาตขายสุราหน่อยค่ะ")
        assert state.context.get("_multi_entity_mentioned") is None

    def test_stale_flag_from_prior_turn_is_cleared_on_single_entity_followup(self, supervisor):
        # A previous turn set the flag; this turn only re-mentions one side —
        # must not leak the stale both/and state into a single-entity answer.
        state = _fresh_state(context={"_multi_entity_mentioned": ["บุคคลธรรมดา", "นิติบุคคล"]})
        supervisor._apply_slot_change_if_detected(state, "เป็นนิติบุคคลค่ะ ต้องใช้เอกสารอะไรบ้าง")
        assert state.context.get("_multi_entity_mentioned") is None

    def test_new_entity_still_set_to_a_single_value_for_collected_slots(self, supervisor):
        # Even in the both/and case, entity detection must still resolve to
        # a concrete new_entity value for collected_slots bookkeeping — the
        # multi-entity guarantee in persona_practical.py is what surfaces the
        # OTHER side, not a null/ambiguous entity_type slot. Needs a
        # pre-existing entity_type slot so entity_changed can go True (on a
        # totally fresh state, _apply_slot_change_if_detected treats this as
        # first-mention rather than a "change" and skips the save — that's
        # covered separately, this test targets the comparison-vs-switch path).
        state = _fresh_state(context={"collected_slots": {"entity_type": "บุคคลธรรมดา"}})
        supervisor._apply_slot_change_if_detected(
            state, "ห้างหุ้นส่วนจำกัด กับ บุคคลธรรมดา เอกสารต่างกันไหมคะ"
        )
        assert state.get_collected_slot("entity_type") in ("นิติบุคคล", "บุคคลธรรมดา")
        assert state.context.get("_multi_entity_mentioned") == ["บุคคลธรรมดา", "นิติบุคคล"]


@pytest.mark.unit
class TestEntityRegexPreconditions:
    """
    The multi-entity fix depends on ENTITY_NATURAL_RE and ENTITY_NITI_RE both
    matching for genuine comparison phrasing. Pinned directly (no
    instantiation needed — these are class attributes) so a future edit to
    either pattern that breaks simultaneous matching fails fast here instead
    of only showing up as a silent behavior regression.
    """

    @pytest.mark.parametrize("query", [
        "ห้างหุ้นส่วนจำกัด กับ บุคคลธรรมดา เอกสารต่างกันไหมคะ",
        "เทียบระหว่างนิติบุคคลกับบุคคลธรรมดาให้หน่อยค่ะ",
        "บุคคลธรรมดาและนิติบุคคล ต่างกันยังไง",
    ])
    def test_both_entity_patterns_match_comparison_queries(self, query):
        assert PersonaSupervisor._ENTITY_NATURAL_RE.search(query)
        assert PersonaSupervisor._ENTITY_NITI_RE.search(query)

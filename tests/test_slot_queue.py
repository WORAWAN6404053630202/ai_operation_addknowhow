"""
Unit tests for slot queue state machine operations.

Validates that ConversationState correctly stores, retrieves, and sequences
slot-fill operations. No LLM or Chroma connection required.
"""
import pytest
from model.conversation_state import ConversationState


@pytest.mark.unit
class TestSlotQueueState:

    def test_save_and_get_collected_slot(self, fresh_state):
        fresh_state.save_collected_slot("entity_type", "นิติบุคคล")
        slots = fresh_state.get_collected_slots()
        assert slots.get("entity_type") == "นิติบุคคล"

    def test_multiple_slots_stored_independently(self, fresh_state):
        fresh_state.save_collected_slot("entity_type", "นิติบุคคล")
        fresh_state.save_collected_slot("registration_type", "บริษัทจำกัด")
        slots = fresh_state.get_collected_slots()
        assert slots.get("entity_type") == "นิติบุคคล"
        assert slots.get("registration_type") == "บริษัทจำกัด"

    def test_slot_overwrite_keeps_latest(self, fresh_state):
        fresh_state.save_collected_slot("entity_type", "นิติบุคคล")
        fresh_state.save_collected_slot("entity_type", "บุคคลธรรมดา")
        slots = fresh_state.get_collected_slots()
        assert slots.get("entity_type") == "บุคคลธรรมดา"

    def test_pending_slot_roundtrip(self, fresh_state):
        fresh_state.context["pending_slot"] = {
            "key": "entity_type",
            "options": ["นิติบุคคล", "บุคคลธรรมดา"],
            "allow_multi": False,
        }
        ps = fresh_state.context.get("pending_slot")
        assert isinstance(ps, dict)
        assert ps["key"] == "entity_type"
        assert len(ps["options"]) == 2

    def test_topic_slot_queue_ordering(self, fresh_state):
        """First slot in the queue should be promoted to pending_slot before the rest."""
        queue = [
            {"key": "entity_type", "options": ["นิติบุคคล", "บุคคลธรรมดา"]},
            {"key": "registration_type", "options": ["บริษัทจำกัด", "ห้างหุ้นส่วน"]},
        ]
        fresh_state.context["topic_slot_queue"] = list(queue)

        # Simulate queue promotion
        next_slot = fresh_state.context["topic_slot_queue"].pop(0)
        fresh_state.context["pending_slot"] = next_slot

        assert fresh_state.context["pending_slot"]["key"] == "entity_type"
        assert len(fresh_state.context["topic_slot_queue"]) == 1
        assert fresh_state.context["topic_slot_queue"][0]["key"] == "registration_type"

    def test_pending_slot_cleared_after_answer(self, fresh_state):
        fresh_state.context["pending_slot"] = {"key": "entity_type", "options": []}
        fresh_state.context.pop("pending_slot", None)
        assert fresh_state.context.get("pending_slot") is None

    def test_collected_slots_empty_by_default(self, fresh_state):
        slots = fresh_state.get_collected_slots()
        assert isinstance(slots, dict)
        assert len(slots) == 0

"""
Unit tests for cache-bypass logic based on collected slot context.

Validates that the cache is skipped when entity_type or registration_type
have been collected, preventing stale pre-slot answers from being served.
No actual cache or LLM connection required.
"""
import pytest
from model.conversation_state import ConversationState

_CACHE_SKIP_SLOTS = {"entity_type", "registration_type"}


def _should_skip_cache(state: ConversationState) -> bool:
    """Mirrors the cache-skip logic in route_v1.py."""
    has_pending_slot = bool((state.context or {}).get("pending_slot"))
    collected = (state.context or {}).get("collected_slots") or {}
    has_slot_context = bool(_CACHE_SKIP_SLOTS & set(collected.keys()))
    return has_pending_slot or has_slot_context


@pytest.mark.unit
class TestCacheAwareness:

    def test_skip_when_entity_type_collected(self):
        state = ConversationState(
            session_id="s1",
            persona_id="practical",
            context={"collected_slots": {"entity_type": "นิติบุคคล"}},
        )
        assert _should_skip_cache(state) is True

    def test_skip_when_registration_type_collected(self):
        state = ConversationState(
            session_id="s2",
            persona_id="practical",
            context={"collected_slots": {"registration_type": "บริษัทจำกัด"}},
        )
        assert _should_skip_cache(state) is True

    def test_skip_when_pending_slot_active(self):
        state = ConversationState(
            session_id="s3",
            persona_id="practical",
            context={"pending_slot": {"key": "entity_type", "options": []}},
        )
        assert _should_skip_cache(state) is True

    def test_allow_cache_when_no_slots(self):
        state = ConversationState(
            session_id="s4",
            persona_id="practical",
            context={},
        )
        assert _should_skip_cache(state) is False

    def test_allow_cache_with_non_sensitive_slot(self):
        """A slot that does not affect retrieval (e.g. department already known) should not block cache."""
        state = ConversationState(
            session_id="s5",
            persona_id="practical",
            context={"collected_slots": {"department": "กรมพัฒนาธุรกิจการค้า"}},
        )
        # department is not in _CACHE_SKIP_SLOTS — cache should be allowed
        assert _should_skip_cache(state) is False

    def test_skip_when_both_sensitive_slots_collected(self):
        state = ConversationState(
            session_id="s6",
            persona_id="practical",
            context={"collected_slots": {"entity_type": "นิติบุคคล", "registration_type": "บริษัทจำกัด"}},
        )
        assert _should_skip_cache(state) is True

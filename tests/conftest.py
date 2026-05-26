"""
Shared pytest fixtures and path setup for the test suite.
"""
import sys
import os
import pytest

# Make the code/ directory importable without installing as a package.
_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))


from model.conversation_state import ConversationState


@pytest.fixture
def fresh_state() -> ConversationState:
    """A ConversationState with no prior history, slots, or context."""
    return ConversationState(
        session_id="test_session",
        persona_id="practical",
        context={},
    )


@pytest.fixture
def state_with_entity_type() -> ConversationState:
    """A ConversationState where entity_type has already been collected."""
    return ConversationState(
        session_id="test_session",
        persona_id="practical",
        context={"collected_slots": {"entity_type": "นิติบุคคล"}},
    )

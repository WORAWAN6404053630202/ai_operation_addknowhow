"""
Unit tests for MAX_ROUNDS session limit enforcement.

Validates that the round counter increments correctly and that the guard
logic triggers at the configured threshold. No LLM connection required.
"""
import pytest
from model.conversation_state import ConversationState


def _should_block(state: ConversationState, max_rounds: int = 7) -> bool:
    """Mirrors the MAX_ROUNDS guard logic in persona_supervisor._handle_inner."""
    cur_round = int(getattr(state, "round", 0) or 0)
    return max_rounds > 0 and cur_round >= max_rounds


@pytest.mark.unit
class TestMaxRounds:

    def test_round_starts_at_zero(self, fresh_state):
        assert fresh_state.round == 0

    def test_increment_round(self, fresh_state):
        fresh_state.increment_round()
        assert fresh_state.round == 1

    def test_increment_round_multiple_times(self, fresh_state):
        for _ in range(5):
            fresh_state.increment_round()
        assert fresh_state.round == 5

    def test_reset_round(self, fresh_state):
        fresh_state.round = 5
        fresh_state.reset_round()
        assert fresh_state.round == 0

    def test_block_at_max_rounds(self, fresh_state):
        fresh_state.round = 7
        assert _should_block(fresh_state, max_rounds=7) is True

    def test_block_above_max_rounds(self, fresh_state):
        fresh_state.round = 10
        assert _should_block(fresh_state, max_rounds=7) is True

    def test_no_block_below_max_rounds(self, fresh_state):
        fresh_state.round = 6
        assert _should_block(fresh_state, max_rounds=7) is False

    def test_no_block_at_zero(self, fresh_state):
        fresh_state.round = 0
        assert _should_block(fresh_state, max_rounds=7) is False

    def test_disabled_when_max_rounds_zero(self, fresh_state):
        """Setting MAX_ROUNDS=0 should disable the limit entirely."""
        fresh_state.round = 999
        assert _should_block(fresh_state, max_rounds=0) is False

    def test_custom_max_rounds(self, fresh_state):
        fresh_state.round = 3
        assert _should_block(fresh_state, max_rounds=3) is True
        assert _should_block(fresh_state, max_rounds=4) is False

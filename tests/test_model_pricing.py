"""
Tests for utils/model_pricing.py — the consolidated pricing table shared by
utils/llm_call.py (chat bot), utils/llm_cost_logging.py (PDF pipeline), and
router/admin.py (dashboard), added 2026-09 after the 3 previously-separate
copies were found to have drifted out of sync with each other and with
reality.

cache_multipliers() specifically locks in two corrections verified live
against openrouter.ai on 2026-09-05, both of which had zero prior test
coverage:
  1. Anthropic cache-WRITE multiplier is 2.0x, not 1.25x — because this
     codebase's only cache_control call site (build_cached_system_message in
     llm_call.py) always requests ttl="1h", and the 1h write rate is 2.0x
     input price (confirmed: claude-sonnet-4-5's $6.00/M 1h-write rate vs its
     $3.00/M input rate). The old code used the 5m-TTL rate (1.25x)
     unconditionally, which doesn't apply to any call this codebase makes.
  2. Non-Anthropic cache-READ multiplier (used for gpt-5.1/Academic) is 0.1x,
     not 0.5x — confirmed against gpt-5.1's OpenRouter page ($0.125/M read vs
     $1.25/M input) and cross-checked against independent 2026 pricing
     writeups (both show a 90% cache-read discount, not 50%).
"""
import pytest

from utils.model_pricing import MODEL_PRICING, cache_multipliers, get_pricing


@pytest.mark.unit
class TestCacheMultipliers:
    def test_anthropic_cache_write_is_2x_matching_the_1h_ttl_this_codebase_always_uses(self):
        read_mult, write_mult = cache_multipliers("anthropic/claude-sonnet-4-5")
        assert read_mult == pytest.approx(0.1)
        assert write_mult == pytest.approx(2.0)

    def test_anthropic_cache_write_2x_applies_to_haiku_too(self):
        _, write_mult = cache_multipliers("anthropic/claude-haiku-4-5")
        assert write_mult == pytest.approx(2.0)

    def test_non_anthropic_cache_read_is_0_1x_not_0_5x(self):
        read_mult, write_mult = cache_multipliers("openai/gpt-5.1")
        assert read_mult == pytest.approx(0.1)
        assert write_mult == pytest.approx(1.0)  # moot placeholder — no write ever billed

    def test_unknown_model_falls_back_to_non_anthropic_multipliers(self):
        read_mult, write_mult = cache_multipliers("some/unknown-model")
        assert read_mult == pytest.approx(0.1)
        assert write_mult == pytest.approx(1.0)


@pytest.mark.unit
class TestModelPricingTable:
    @pytest.mark.parametrize(
        "model,expected_input,expected_output",
        [
            ("anthropic/claude-sonnet-4-5", 3.00, 15.00),
            ("anthropic/claude-haiku-4-5", 1.00, 5.00),
            ("openai/gpt-5.1", 1.25, 10.00),
            ("qwen/qwen3.7-flash", 0.03, 0.13),
            ("google/gemini-2.5-flash-lite", 0.10, 0.40),
            ("baai/bge-m3", 0.01, 0.0),
        ],
    )
    def test_live_verified_rates_2026_09_05(self, model, expected_input, expected_output):
        """Every rate here was independently re-checked against openrouter.ai
        on 2026-09-05 (not carried over from the old, since-found-stale
        tables) — see this module's docstring and utils/model_pricing.py's."""
        pricing = get_pricing(model)
        assert pricing is not None
        assert pricing["input"] == pytest.approx(expected_input)
        assert pricing["output"] == pytest.approx(expected_output)

    def test_unknown_model_returns_none(self):
        assert get_pricing("some/unknown-model") is None

    def test_pdf_pipeline_and_chat_bot_share_the_exact_same_table(self):
        """Guards against the old failure mode: 2 separate copies of this table
        (utils/llm_call.py's + utils/llm_cost_logging.py's) drifting apart —
        both now import MODEL_PRICING directly from this module, so this is
        really just confirming the import wiring, not a coincidence of equal
        values."""
        from utils.llm_call import PRICING_USD_PER_MILLION_TOKENS
        from utils.llm_cost_logging import PDF_PIPELINE_MODEL_PRICING

        assert PRICING_USD_PER_MILLION_TOKENS is MODEL_PRICING
        assert PDF_PIPELINE_MODEL_PRICING is MODEL_PRICING


@pytest.mark.unit
class TestEstimateCostWithCorrectedMultipliers:
    def test_anthropic_cache_write_billed_at_2x_input_price(self):
        from utils.llm_call import estimate_cost

        model = "anthropic/claude-sonnet-4-5"
        # 1000 fresh input tokens, all 1000 of which are a cache WRITE, no completion.
        cost = estimate_cost(model, prompt_tokens=1000, completion_tokens=0, cache_write_tokens=1000)
        expected = 1000 * 3.00 * 2.0 / 1_000_000  # input price * write multiplier
        assert cost == pytest.approx(expected)

    def test_gpt_5_1_cache_read_billed_at_0_1x_input_price(self):
        from utils.llm_call import estimate_cost

        model = "openai/gpt-5.1"
        cost = estimate_cost(model, prompt_tokens=1000, completion_tokens=0, cache_read_tokens=1000)
        expected = 1000 * 1.25 * 0.1 / 1_000_000
        assert cost == pytest.approx(expected)

"""
Tests for utils/llm_cost_logging.py — the shared token/cost logger every
service/pdf_*.py call site now calls right after each chat-completion
response (see the module's own docstring for why this exists: the PDF
pipeline calls the OpenAI SDK directly and never had its own cost data
before this).
"""
import logging
import threading

import pytest

from utils.llm_cost_logging import CostAccumulator, PDF_PIPELINE_MODEL_PRICING, log_call_duration, log_llm_cost


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, usage):
        self.usage = usage


@pytest.mark.unit
class TestLogLlmCost:
    def test_known_model_cost_matches_manual_calculation(self, caplog):
        response = _FakeResponse(_FakeUsage(prompt_tokens=1000, completion_tokens=500))
        model = "anthropic/claude-sonnet-4-5"
        pricing = PDF_PIPELINE_MODEL_PRICING[model]
        expected = (1000 * pricing["input"] + 500 * pricing["output"]) / 1_000_000

        with caplog.at_level(logging.INFO):
            cost = log_llm_cost(logging.getLogger("test"), "TestCall", model, response)

        assert cost == pytest.approx(expected)
        assert cost > 0

    def test_unknown_model_degrades_gracefully(self):
        response = _FakeResponse(_FakeUsage(prompt_tokens=100, completion_tokens=50))
        cost = log_llm_cost(logging.getLogger("test"), "TestCall", "some/unpriced-model", response)
        assert cost == 0.0

    def test_response_with_no_usage_attribute_degrades_gracefully(self):
        class _NoUsageResponse:
            pass

        cost = log_llm_cost(logging.getLogger("test"), "TestCall", "anthropic/claude-sonnet-4-5", _NoUsageResponse())
        assert cost == 0.0

    def test_every_pdf_pipeline_model_priced_has_positive_rates(self):
        for model, pricing in PDF_PIPELINE_MODEL_PRICING.items():
            assert pricing["input"] > 0, f"{model} input price should be positive"
            assert pricing["output"] > 0, f"{model} output price should be positive"

    def test_elapsed_seconds_appears_in_log_line_when_given(self, caplog):
        response = _FakeResponse(_FakeUsage(prompt_tokens=100, completion_tokens=50))
        with caplog.at_level(logging.INFO):
            log_llm_cost(logging.getLogger("test"), "TestCall", "anthropic/claude-sonnet-4-5", response, elapsed_seconds=1.234)
        assert "elapsed_s=1.23" in caplog.text

    def test_elapsed_seconds_omitted_when_not_given(self, caplog):
        response = _FakeResponse(_FakeUsage(prompt_tokens=100, completion_tokens=50))
        with caplog.at_level(logging.INFO):
            log_llm_cost(logging.getLogger("test"), "TestCall", "anthropic/claude-sonnet-4-5", response)
        assert "elapsed_s=" not in caplog.text

    def test_elapsed_seconds_appears_even_on_unknown_model(self, caplog):
        response = _FakeResponse(_FakeUsage(prompt_tokens=100, completion_tokens=50))
        with caplog.at_level(logging.INFO):
            log_llm_cost(logging.getLogger("test"), "TestCall", "some/unpriced-model", response, elapsed_seconds=0.5)
        assert "elapsed_s=0.50" in caplog.text

    def test_elapsed_seconds_appears_even_when_usage_missing(self, caplog):
        # A response missing .usage entirely doesn't actually raise: getattr()
        # degrades usage->None->0 tokens cleanly, so this logs at INFO with
        # zero cost, not the WARNING path (that's only for pricing-missing or
        # a genuinely malformed usage object — see the other two tests above).
        class _NoUsageResponse:
            pass

        with caplog.at_level(logging.INFO):
            log_llm_cost(logging.getLogger("test"), "TestCall", "anthropic/claude-sonnet-4-5", _NoUsageResponse(), elapsed_seconds=0.75)
        assert "elapsed_s=0.75" in caplog.text

    def test_elapsed_seconds_appears_on_genuine_usage_read_failure(self, caplog):
        # getattr(usage, "prompt_tokens", 0) raising is the actual failure
        # path this function's try/except guards against.
        class _BrokenUsage:
            @property
            def prompt_tokens(self):
                raise RuntimeError("simulated malformed usage object")

        response = _FakeResponse(_BrokenUsage())
        with caplog.at_level(logging.WARNING):
            cost = log_llm_cost(logging.getLogger("test"), "TestCall", "anthropic/claude-sonnet-4-5", response, elapsed_seconds=0.75)
        assert cost == 0.0
        assert "elapsed_s=0.75" in caplog.text


@pytest.mark.unit
class TestLogCallDuration:
    def test_logs_elapsed_seconds(self, caplog):
        with caplog.at_level(logging.INFO):
            log_call_duration(logging.getLogger("test"), "TestNonLlmCall", 2.5)
        assert "elapsed_s=2.50" in caplog.text
        assert "TestNonLlmCall" in caplog.text

    def test_never_raises_on_odd_input(self):
        # elapsed_seconds is always a real float from time.monotonic() differences
        # in practice, but this must not be a crash-prone code path regardless.
        log_call_duration(logging.getLogger("test"), "TestNonLlmCall", 0.0)


@pytest.mark.unit
class TestCostAccumulator:
    """Backs the admin UI's per-document total_cost_usd display — see
    model/pdf_review_item.py and sqs_consumer.py's threading of one
    CostAccumulator instance through every stage of one document's pipeline."""

    def test_starts_at_zero(self):
        assert CostAccumulator().total_cost_usd == 0.0

    def test_add_accumulates(self):
        acc = CostAccumulator()
        acc.add(0.001)
        acc.add(0.002)
        acc.add(0.0)
        assert acc.total_cost_usd == pytest.approx(0.003)

    def test_thread_safe_under_concurrent_adds(self):
        # Mirrors real usage: multiple pages' vision-verify calls add to the
        # same accumulator concurrently via ThreadPoolExecutor.
        acc = CostAccumulator()
        n_threads, adds_per_thread, amount = 20, 100, 0.0001

        def worker():
            for _ in range(adds_per_thread):
                acc.add(amount)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = n_threads * adds_per_thread * amount
        assert acc.total_cost_usd == pytest.approx(expected)

    def test_log_llm_cost_adds_computed_cost_to_accumulator(self):
        acc = CostAccumulator()
        response = _FakeResponse(_FakeUsage(prompt_tokens=1000, completion_tokens=500))
        model = "anthropic/claude-sonnet-4-5"
        pricing = PDF_PIPELINE_MODEL_PRICING[model]
        expected = (1000 * pricing["input"] + 500 * pricing["output"]) / 1_000_000

        cost = log_llm_cost(logging.getLogger("test"), "TestCall", model, response, accumulator=acc)

        assert cost == pytest.approx(expected)
        assert acc.total_cost_usd == pytest.approx(expected)

    def test_log_llm_cost_adds_zero_on_unknown_model_without_crashing(self):
        acc = CostAccumulator()
        response = _FakeResponse(_FakeUsage(prompt_tokens=100, completion_tokens=50))
        log_llm_cost(logging.getLogger("test"), "TestCall", "some/unpriced-model", response, accumulator=acc)
        assert acc.total_cost_usd == 0.0

    def test_log_llm_cost_accumulates_across_multiple_calls(self):
        # Simulates one document's pipeline: several call sites sharing one accumulator.
        acc = CostAccumulator()
        model = "anthropic/claude-haiku-4-5"
        pricing = PDF_PIPELINE_MODEL_PRICING[model]
        for _ in range(3):
            response = _FakeResponse(_FakeUsage(prompt_tokens=200, completion_tokens=100))
            log_llm_cost(logging.getLogger("test"), "TestCall", model, response, accumulator=acc)

        expected_per_call = (200 * pricing["input"] + 100 * pricing["output"]) / 1_000_000
        assert acc.total_cost_usd == pytest.approx(expected_per_call * 3)

    def test_log_llm_cost_without_accumulator_still_works(self):
        # accumulator is optional — every existing call site (before this
        # feature) omits it and must keep working unchanged.
        response = _FakeResponse(_FakeUsage(prompt_tokens=100, completion_tokens=50))
        cost = log_llm_cost(logging.getLogger("test"), "TestCall", "anthropic/claude-sonnet-4-5", response)
        assert cost > 0

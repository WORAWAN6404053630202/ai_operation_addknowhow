"""
Tests for utils/rate_limiter.py (MinIntervalRateLimiter) and the Typhoon
OCR retry/throttle logic built on top of it in service/pdf_large_extraction.py
— added 2026-09 after confirming Typhoon's typhoon-ocr endpoint is
rate-limited to 2 requests/sec with zero retry protection anywhere in this
pipeline before this.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import patch

import httpx
import openai
import pytest

from utils.rate_limiter import MinIntervalRateLimiter


def _fake_rate_limit_error(retry_after: str | None = None) -> openai.RateLimitError:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    response = httpx.Response(
        status_code=429, headers=headers,
        request=httpx.Request("POST", "https://api.opentyphoon.ai/v1/chat/completions"),
    )
    return openai.RateLimitError("rate limited", response=response, body=None)


@pytest.mark.unit
class TestMinIntervalRateLimiter:
    def test_first_call_does_not_wait(self):
        limiter = MinIntervalRateLimiter(min_interval_seconds=1.0)
        start = time.monotonic()
        limiter.wait()
        assert time.monotonic() - start < 0.05

    def test_second_call_within_interval_is_delayed(self):
        limiter = MinIntervalRateLimiter(min_interval_seconds=0.2)
        limiter.wait()
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.18  # allow small scheduling slack

    def test_call_after_interval_has_elapsed_is_not_delayed(self):
        limiter = MinIntervalRateLimiter(min_interval_seconds=0.05)
        limiter.wait()
        time.sleep(0.1)
        start = time.monotonic()
        limiter.wait()
        assert time.monotonic() - start < 0.05

    def test_concurrent_callers_are_serialized_to_the_interval(self):
        # 5 threads all calling .wait() at once on a 0.05s limiter should
        # take at least ~0.2s total (4 gaps of 0.05s after the first call
        # goes through immediately) — proves the lock actually spaces out
        # concurrent callers, not just sequential ones.
        limiter = MinIntervalRateLimiter(min_interval_seconds=0.05)
        call_times: list[float] = []
        lock = threading.Lock()

        def _call():
            limiter.wait()
            with lock:
                call_times.append(time.monotonic())

        start = time.monotonic()
        threads = [threading.Thread(target=_call) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(call_times) == 5
        assert max(call_times) - start >= 0.18


@pytest.mark.unit
class TestTyphoonRetryLogic:
    """service/pdf_large_extraction.py's _run_typhoon: retry with backoff on
    transient errors, bounded by _TYPHOON_MAX_RETRIES, throttled by the
    module-level _typhoon_rate_limiter."""

    def _import_module(self):
        import service.pdf_large_extraction as mod
        return mod

    def test_succeeds_immediately_when_no_error(self):
        mod = self._import_module()

        with patch.object(mod, "ocr_document", return_value="extracted text") as mock_ocr, \
             patch.object(mod, "_typhoon_rate_limiter") as mock_limiter:
            result = mod._run_typhoon(b"fake-png-bytes", page_num=1)

        assert result == "extracted text"
        assert mock_ocr.call_count == 1
        assert mock_limiter.wait.call_count == 1

    def test_retries_on_rate_limit_then_succeeds(self):
        mod = self._import_module()
        call_count = {"n": 0}

        def _side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise _fake_rate_limit_error(retry_after="0.01")
            return "extracted after retries"

        with patch.object(mod, "ocr_document", side_effect=_side_effect), \
             patch.object(mod, "_typhoon_rate_limiter") as mock_limiter, \
             patch.object(mod.time, "sleep") as mock_sleep:
            result = mod._run_typhoon(b"fake-png-bytes", page_num=2)

        assert result == "extracted after retries"
        assert call_count["n"] == 3
        assert mock_limiter.wait.call_count == 3
        assert mock_sleep.call_count == 2  # slept before retry #2 and #3, not after final success

    def test_gives_up_after_max_retries_and_reraises(self):
        mod = self._import_module()

        def _always_fails(*args, **kwargs):
            raise _fake_rate_limit_error()

        with patch.object(mod, "ocr_document", side_effect=_always_fails), \
             patch.object(mod, "_typhoon_rate_limiter"), \
             patch.object(mod.time, "sleep"):
            with pytest.raises(openai.RateLimitError):
                mod._run_typhoon(b"fake-png-bytes", page_num=3)

    def test_non_retryable_error_propagates_immediately(self):
        mod = self._import_module()

        with patch.object(mod, "ocr_document", side_effect=ValueError("not a retryable error")), \
             patch.object(mod, "_typhoon_rate_limiter") as mock_limiter, \
             patch.object(mod.time, "sleep") as mock_sleep:
            with pytest.raises(ValueError):
                mod._run_typhoon(b"fake-png-bytes", page_num=4)

        assert mock_limiter.wait.call_count == 1  # only tried once, no retry
        assert mock_sleep.call_count == 0

    def test_retry_after_header_is_honored_over_backoff(self):
        mod = self._import_module()
        call_count = {"n": 0}

        def _side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _fake_rate_limit_error(retry_after="7")
            return "ok"

        with patch.object(mod, "ocr_document", side_effect=_side_effect), \
             patch.object(mod, "_typhoon_rate_limiter"), \
             patch.object(mod.time, "sleep") as mock_sleep:
            mod._run_typhoon(b"fake-png-bytes", page_num=5)

        mock_sleep.assert_called_once_with(7.0)

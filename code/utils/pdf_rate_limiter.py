# code/utils/pdf_rate_limiter.py (feature/pdf-ingestion addition, 2026-09)
#
# Split out of utils/rate_limiter.py (2026-09, PDF/bot service separation) —
# MinIntervalRateLimiter below is UNRELATED to that file's RateLimiter (that
# one throttles the main chat bot's per-session API requests; this one
# throttles this process's own outbound calls to Typhoon OCR, which is
# rate-limited to 2 requests/second, 20 requests/minute —
# https://docs.opentyphoon.ai/en/rate-limits/, checked 2026-09). The two
# classes always shared nothing and never interacted — this file just gives
# that fact a physical boundary now that the bot and PDF pipeline are
# separate deployable services.
#
# Deliberately per-process, not global/distributed: this fully protects the
# concurrency this module itself controls (one document's page-level worker
# threads, within one EC2 consumer process or one Lambda invocation), but does
# NOT protect against multiple documents being processed by separate
# concurrent processes at once (e.g. two Lambda invocations running in
# parallel for two different uploads) — each gets its own independent
# tracker. A distributed limiter (Redis/DynamoDB-backed token bucket) would be
# needed to close that gap; deliberately out of scope here — this is the
# cheap, immediate fix for the concurrency this codebase actually controls,
# not a claim that it eliminates every way the shared external limit could be
# exceeded.

import threading
import time


class MinIntervalRateLimiter:
    """Blocks callers as needed so consecutive calls to .wait() are spaced
    at least min_interval_seconds apart, thread-safe across any number of
    concurrent callers within this process."""

    def __init__(self, min_interval_seconds: float):
        self._min_interval = min_interval_seconds
        self._lock = threading.Lock()
        self._last_call_at = None  # type: float | None

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_call_at is not None:
                remaining = self._min_interval - (now - self._last_call_at)
                if remaining > 0:
                    time.sleep(remaining)
                    now = time.monotonic()
            self._last_call_at = now

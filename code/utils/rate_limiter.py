"""
Rate limiter to prevent API abuse and excessive costs.

Implements a sliding window algorithm that tracks both request count and token
usage per identifier (session). Two complementary limits are enforced:

  1. Request rate   — max N requests per window (original behaviour).
  2. Token rate     — max T tokens consumed per window (cost-aware extension).

Token tracking requires callers to report actual usage via record_token_usage()
after each LLM call. When TOKEN_RATE_LIMIT_PER_WINDOW=0 the token check is
disabled entirely, preserving backward compatibility.
"""

import time
from collections import defaultdict, deque
from typing import Dict, Tuple
import threading

_GLOBAL_LIMITER_LOCK = threading.Lock()


class RateLimiter:
    """
    Sliding window rate limiter with optional per-window token budget.

    Example:
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        allowed, info = limiter.is_allowed("session_123")
        if not allowed:
            raise HTTPException(429, "Rate limit exceeded")

        # After LLM call completes:
        limiter.record_token_usage("session_123", tokens_used=4200)
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests:   Maximum requests allowed per window (default: 10).
            window_seconds: Sliding window duration in seconds (default: 60).
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds

        # Request timestamps per identifier: {id: deque([ts, ...])}
        self._requests: Dict[str, deque] = defaultdict(deque)

        # Token usage per identifier: {id: deque([(ts, tokens), ...])}
        self._window_tokens: Dict[str, deque] = defaultdict(deque)

        self._lock = threading.RLock()

        # Metrics
        self.total_requests = 0
        self.blocked_requests = 0

    def is_allowed(self, identifier: str) -> Tuple[bool, Dict]:
        """
        Check if request is allowed for given identifier.

        Args:
            identifier: Unique identifier (session_id, IP, user_id, etc.)

        Returns:
            (allowed: bool, info: dict)

        Example:
            allowed, info = limiter.is_allowed("session_123")
            # info = {"remaining": 7, "reset_in": 42, "limit": 10}
        """
        current_time = time.time()

        with self._lock:
            self.total_requests += 1

            # Get request timestamps for this identifier
            timestamps = self._requests[identifier]

            # Remove expired timestamps (outside window)
            cutoff_time = current_time - self.window_seconds
            while timestamps and timestamps[0] < cutoff_time:
                timestamps.popleft()

            # Check if under limit
            if len(timestamps) < self.max_requests:
                timestamps.append(current_time)

                # Calculate reset time (when oldest request expires)
                reset_in = 0
                if timestamps:
                    oldest = timestamps[0]
                    reset_in = int(self.window_seconds - (current_time - oldest))

                return True, {
                    "allowed": True,
                    "remaining": self.max_requests - len(timestamps),
                    "reset_in": max(0, reset_in),
                    "limit": self.max_requests,
                    "window": self.window_seconds
                }
            else:
                self.blocked_requests += 1

                # Calculate when oldest request will expire
                oldest = timestamps[0]
                reset_in = int(self.window_seconds - (current_time - oldest) + 1)

                return False, {
                    "allowed": False,
                    "remaining": 0,
                    "reset_in": max(0, reset_in),
                    "limit": self.max_requests,
                    "window": self.window_seconds,
                    "retry_after": max(1, reset_in)
                }

    def record_token_usage(self, identifier: str, tokens: int) -> None:
        """
        Record LLM token usage for the current window.

        Must be called by the request handler after each LLM call completes.
        Only the tokens consumed in THIS call should be passed (delta, not cumulative).

        Args:
            identifier: Session or IP identifier matching is_allowed().
            tokens:     Number of tokens consumed by this LLM call.
        """
        if tokens <= 0:
            return
        with self._lock:
            self._window_tokens[identifier].append((time.time(), int(tokens)))

    def get_window_tokens(self, identifier: str) -> int:
        """
        Return the total tokens consumed by this identifier within the current window.
        Expired entries are pruned on each call.
        """
        current_time = time.time()
        cutoff = current_time - self.window_seconds
        with self._lock:
            q = self._window_tokens[identifier]
            while q and q[0][0] < cutoff:
                q.popleft()
            return sum(t for _, t in q)

    def is_token_rate_allowed(self, identifier: str, max_tokens_per_window: int) -> Tuple[bool, int]:
        """
        Check whether the identifier is within its per-window token budget.

        Args:
            identifier:            Session identifier.
            max_tokens_per_window: Token ceiling for the sliding window.
                                   Pass 0 to disable the check (always allowed).

        Returns:
            (allowed: bool, tokens_used_in_window: int)
        """
        if max_tokens_per_window <= 0:
            return True, 0
        used = self.get_window_tokens(identifier)
        return used < max_tokens_per_window, used

    def reset(self, identifier: str) -> None:
        """Reset request count and token usage for a specific identifier."""
        with self._lock:
            self._requests.pop(identifier, None)
            self._window_tokens.pop(identifier, None)

    def clear_all(self) -> None:
        """Clear all rate limit data including token tracking."""
        with self._lock:
            self._requests.clear()
            self._window_tokens.clear()
            self.total_requests = 0
            self.blocked_requests = 0

    def get_stats(self) -> Dict:
        """
        Return rate limiter statistics.

        Returns a dict with request metrics and the count of identifiers that
        have consumed tokens in the current window.
        """
        with self._lock:
            block_rate = 0.0
            if self.total_requests > 0:
                block_rate = self.blocked_requests / self.total_requests

            return {
                "total_requests": self.total_requests,
                "blocked_requests": self.blocked_requests,
                "block_rate": round(block_rate, 3),
                "active_identifiers": len(self._requests),
                "active_token_identifiers": len(self._window_tokens),
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
            }


# Global rate limiter instance
_global_limiter: RateLimiter = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance (singleton)."""
    global _global_limiter
    if _global_limiter is None:
        with _GLOBAL_LIMITER_LOCK:
            if _global_limiter is None:
                _global_limiter = RateLimiter(max_requests=10, window_seconds=60)
    return _global_limiter


# ---------------------------------------------------------------------------
# code/utils/rate_limiter.py (feature/pdf-ingestion addition, 2026-09)
#
# MinIntervalRateLimiter below is UNRELATED to RateLimiter above (that one
# throttles the main chat bot's per-session API requests; this one throttles
# this process's own outbound calls to Typhoon OCR, which is rate-limited to
# 2 requests/second, 20 requests/minute — https://docs.opentyphoon.ai/en/
# rate-limits/, checked 2026-09). Kept in the same file as
# service/pdf_large_extraction.py's existing import path rather than moved,
# but the two classes share nothing and never interact.
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
# ---------------------------------------------------------------------------


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

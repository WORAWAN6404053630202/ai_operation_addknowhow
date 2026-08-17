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
from collections import deque, OrderedDict
from typing import Dict, Tuple
import threading

_GLOBAL_LIMITER_LOCK = threading.Lock()


class RateLimiter:
    """
    Sliding window rate limiter with optional per-window token budget.

    Identifiers should be something a single caller cannot cheaply rotate
    (e.g. client IP) — a client-supplied value like a self-issued session_id
    lets an attacker reset their own limit on every request.

    Example:
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        allowed, info = limiter.is_allowed("1.2.3.4")
        if not allowed:
            raise HTTPException(429, "Rate limit exceeded")

        # After LLM call completes:
        limiter.record_token_usage("1.2.3.4", tokens_used=4200)
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60, max_identifiers: int = 5000):
        """
        Initialize rate limiter.

        Args:
            max_requests:   Maximum requests allowed per window (default: 10).
            window_seconds: Sliding window duration in seconds (default: 60).
            max_identifiers: Hard cap on distinct identifiers tracked at once.
                LRU-evicts the least-recently-seen identifier past this —
                without a cap, an identifier that rotates freely (or just a
                lot of distinct real callers) grows these dicts forever.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_identifiers = max_identifiers

        # Request timestamps per identifier: {id: deque([ts, ...])}
        self._requests: "OrderedDict[str, deque]" = OrderedDict()

        # Token usage per identifier: {id: deque([(ts, tokens), ...])}
        self._window_tokens: "OrderedDict[str, deque]" = OrderedDict()

        self._lock = threading.RLock()

        # Metrics
        self.total_requests = 0
        self.blocked_requests = 0
        self._checks_since_cleanup = 0

    def _get_bucket(self, store: "OrderedDict[str, deque]", identifier: str) -> deque:
        """Fetch (creating if needed) an identifier's deque and mark it
        most-recently-used, evicting the oldest identifier if this pushes
        `store` past max_identifiers."""
        bucket = store.get(identifier)
        if bucket is None:
            bucket = deque()
            store[identifier] = bucket
            if len(store) > self.max_identifiers:
                store.popitem(last=False)
        else:
            store.move_to_end(identifier)
        return bucket

    def _cleanup_expired(self, current_time: float) -> None:
        """Drop identifiers whose window has fully elapsed, so one-shot or
        rotated identifiers don't sit in memory until LRU eviction catches up."""
        cutoff = current_time - self.window_seconds
        for ident in [i for i, dq in self._requests.items() if not dq or dq[-1] < cutoff]:
            del self._requests[ident]
        for ident in [i for i, dq in self._window_tokens.items() if not dq or dq[-1][0] < cutoff]:
            del self._window_tokens[ident]

    def is_allowed(self, identifier: str) -> Tuple[bool, Dict]:
        """
        Check if request is allowed for given identifier.

        Args:
            identifier: Unique identifier — use client IP, not a client-supplied value.

        Returns:
            (allowed: bool, info: dict)

        Example:
            allowed, info = limiter.is_allowed("1.2.3.4")
            # info = {"remaining": 7, "reset_in": 42, "limit": 10}
        """
        current_time = time.time()

        with self._lock:
            self.total_requests += 1

            self._checks_since_cleanup += 1
            if self._checks_since_cleanup >= 500:
                self._checks_since_cleanup = 0
                self._cleanup_expired(current_time)

            # Get request timestamps for this identifier
            timestamps = self._get_bucket(self._requests, identifier)

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
            bucket = self._get_bucket(self._window_tokens, identifier)
            bucket.append((time.time(), int(tokens)))

    def get_window_tokens(self, identifier: str) -> int:
        """
        Return the total tokens consumed by this identifier within the current window.
        Expired entries are pruned on each call.
        """
        current_time = time.time()
        cutoff = current_time - self.window_seconds
        with self._lock:
            q = self._get_bucket(self._window_tokens, identifier)
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

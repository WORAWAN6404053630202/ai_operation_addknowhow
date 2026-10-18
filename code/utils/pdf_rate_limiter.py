# code/utils/pdf_rate_limiter.py (feature/pdf-ingestion addition, 2026-09)
#
# Split out of utils/rate_limiter.py (2026-09, PDF/bot service separation) —
# MinIntervalRateLimiter below is UNRELATED to that file's RateLimiter (that
# one throttles the main chat bot's per-session API requests; this one
# throttles this process's own outbound calls to Typhoon OCR, which is
# rate-limited to 2 requests/second AND 20 requests/minute —
# https://docs.opentyphoon.ai/en/rate-limits/, checked 2026-09).
#
# History: MinIntervalRateLimiter below only ever enforced the 2 req/sec half
# of Typhoon's limit (spacing consecutive calls >= 1/1.8s apart) — it never
# tracked a rolling per-minute count at all. Sustained at 1.8 req/s, a single
# process would send ~108 calls/minute, over 5x the 20/min ceiling — a real
# gap that existed independently of concurrency, for any single document with
# enough scanned/image pages to sustain OCR calls for more than ~11 seconds
# (20 calls at 1.8/s), well within MAX_PAGES_FOR_LAMBDA=35's normal range.
#
# GlobalTyphoonRateLimiter (added 2026-09) fixes both gaps at once: (1) tracks
# BOTH the 2/sec and 20/min ceilings, not just the first; (2) coordinates
# across every concurrent process (Lambda invocations each get an isolated
# execution environment — no shared memory — plus the EC2 consumer) via a
# shared DynamoDB table, using atomic conditional transact-writes so
# concurrent callers never both believe they've claimed the same slot. Falls
# back to the plain in-process MinIntervalRateLimiter (fail-open, not
# fail-blocked) if DynamoDB itself is unreachable/misconfigured/exhausted —
# this coordination layer is a correctness improvement, not something that
# should be able to wedge the whole pipeline if AWS has a bad moment.

import random
import threading
import time
from datetime import datetime, timezone
from typing import Optional


class MinIntervalRateLimiter:
    """Blocks callers as needed so consecutive calls to .wait() are spaced
    at least min_interval_seconds apart, thread-safe across any number of
    concurrent callers within this process. Used directly by anything that
    only needs single-process throttling, and as GlobalTyphoonRateLimiter's
    own fail-open fallback below."""

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


class GlobalTyphoonRateLimiter:
    """Cross-process rate limiter for Typhoon OCR, backed by a shared
    DynamoDB table so every concurrent Lambda invocation and the EC2
    consumer all coordinate against the SAME real budget (2 req/sec, 20
    req/min) instead of each independently pacing itself with no idea how
    many others are calling at the same time.

    Table schema (single table, shared by all callers):
      - Partition key `limiter_id` (String) — always "typhoon_ocr" here;
        named generically in case another rate-limited external API needs
        the same table later.
      - Sort key `bucket` (String) — either the literal "_last_call" (one
        fixed item tracking the most recent call's timestamp, for the
        2/sec check) or a UTC-minute string like "2026-09-14T10:32" (one
        item per minute, tracking that minute's call count, for the 20/min
        check — a NEW item per minute means a fresh count with no explicit
        reset logic needed).
      - Attributes: `last_call_at` (Number, epoch seconds) on the
        "_last_call" item; `call_count` (Number) and `expires_at` (Number,
        epoch seconds — enable DynamoDB TTL on this attribute so old
        minute-buckets clean up automatically) on minute-bucket items.

    Each .wait() call atomically checks AND claims both the RPS slot and
    the RPM slot together via a single DynamoDB transact_write_items call
    — if EITHER condition fails, neither attribute is updated (no "wasted"
    partial claim), and the caller retries after a short jittered sleep.
    Bounded by max_wait_seconds (default 90s, a bit over one full RPM
    window) — if still not claimed by then, falls back to the local
    MinIntervalRateLimiter and proceeds, logging a warning. Any DynamoDB
    error OTHER than a condition failure (table missing, no permissions,
    network issue) triggers the same fallback immediately, not after
    exhausting the retry budget — a broken table should degrade gracefully,
    not turn into a 90-second stall on every single call."""

    def __init__(
        self,
        table_name: str,
        region_name: Optional[str],
        limiter_id: str = "typhoon_ocr",
        rps_min_interval_seconds: float = 1 / 1.8,
        rpm_limit: int = 20,
        max_wait_seconds: float = 90.0,
        logger=None,
    ):
        self._table_name = table_name
        self._limiter_id = limiter_id
        self._rps_min_interval = rps_min_interval_seconds
        self._rpm_limit = rpm_limit
        self._max_wait_seconds = max_wait_seconds
        self._logger = logger
        self._fallback = MinIntervalRateLimiter(rps_min_interval_seconds)
        self._client = None
        self._region_name = region_name

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("dynamodb", region_name=self._region_name or None)
        return self._client

    def _log(self, level: str, msg: str) -> None:
        if self._logger is not None:
            getattr(self._logger, level)(msg)

    def wait(self) -> None:
        deadline = time.monotonic() + self._max_wait_seconds
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            now = time.time()
            minute_bucket = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M")
            try:
                client = self._get_client()
                client.transact_write_items(
                    TransactItems=[
                        {
                            "Update": {
                                "TableName": self._table_name,
                                "Key": {"limiter_id": {"S": self._limiter_id}, "bucket": {"S": "_last_call"}},
                                "UpdateExpression": "SET last_call_at = :now",
                                "ConditionExpression": "attribute_not_exists(last_call_at) OR last_call_at <= :threshold",
                                "ExpressionAttributeValues": {
                                    ":now": {"N": str(now)},
                                    ":threshold": {"N": str(now - self._rps_min_interval)},
                                },
                            }
                        },
                        {
                            "Update": {
                                "TableName": self._table_name,
                                "Key": {"limiter_id": {"S": self._limiter_id}, "bucket": {"S": minute_bucket}},
                                "UpdateExpression": "SET call_count = if_not_exists(call_count, :zero) + :one, expires_at = :ttl",
                                "ConditionExpression": "attribute_not_exists(call_count) OR call_count < :rpm_limit",
                                "ExpressionAttributeValues": {
                                    ":zero": {"N": "0"},
                                    ":one": {"N": "1"},
                                    ":rpm_limit": {"N": str(self._rpm_limit)},
                                    ":ttl": {"N": str(int(now) + 120)},
                                },
                            }
                        },
                    ]
                )
                return  # both conditions satisfied and claimed atomically — safe to call Typhoon now
            except Exception as e:
                error_response = getattr(e, "response", None) or {}
                error_code = error_response.get("Error", {}).get("Code")
                if error_code == "TransactionCanceledException":
                    time.sleep(0.1 + random.random() * 0.3)
                    continue
                self._log("warning", f"[GlobalTyphoonRateLimiter] DynamoDB coordination failed ({type(e).__name__}: {e}) — falling back to local-only throttling for this call")
                self._fallback.wait()
                return
        self._log("warning", f"[GlobalTyphoonRateLimiter] Could not claim a global rate-limit slot within {self._max_wait_seconds}s ({attempt} attempts) — falling back to local-only throttling for this call")
        self._fallback.wait()

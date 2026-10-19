# code/utils/pdf_attempt_counter.py (feature/pdf-ingestion addition, 2026-10)
#
# Cross-process, atomic retry-attempt counter for the PDF pipeline's bounded
# give-up logic (service/sqs_consumer.py's _MAX_HANDOFF_ATTEMPTS, lambda/
# pdf_extraction/handler.py's _MAX_ATTEMPTS).
#
# History: both of those give-up checks previously read their attempt number
# from service/pdf_status_tracker.py's next_attempt_number(), which infers
# "prior attempt + 1" by reading back a best-effort S3 status object —
# best-effort BY DESIGN for that module's real job (dashboard visibility
# must never block real OCR/extraction work on a status-write hiccup), but
# that same fail-open property silently defeats a retry CAP: if the S3
# write after a failed attempt itself fails (throttling, a network blip, a
# permissions edge case), the next retry's read sees no object and
# next_attempt_number() returns 1 again — the counter resets instead of
# incrementing, and _MAX_ATTEMPTS/_MAX_HANDOFF_ATTEMPTS never trips. That is
# exactly the failure shape (an unbounded retry loop) this whole mechanism
# exists to prevent — see pdf_status_tracker.py's own docstring for the
# 13+ hour/634-page incident that motivated bounding this in the first
# place. Status DISPLAY (stage/pages_done/error, still via
# pdf_status_tracker.py) stays best-effort/lossy on purpose; this module
# covers only the number that feeds the give-up decision.
#
# Design: a single atomic DynamoDB update_item per attempt (unconditional
# increment, same "if_not_exists(...) + :one" shape as
# utils/pdf_cost_guard.py's running total) — there is no read-then-write
# race to lose, unlike the S3 read-infer approach. The key is
# stage-qualified (f"{filename}#{stage}"), not just the bare filename,
# specifically so a document's Lambda-side OCR-retry count and its EC2-side
# build-retry count (two independent give-up checks, same filename, back to
# back) can never bleed into each other the way they could when both stages
# shared one plain-filename S3 status key.
#
# Fail-CLOSED, not fail-open: unlike the rate limiter/cost guard/draft cache
# (which fail open because nothing depends on their output for correctness
# anymore), this counter directly feeds the give-up decision — failing open
# here ("couldn't reach DynamoDB, so just say attempt 1") reproduces the
# EXACT bug this module exists to fix. A DynamoDB error instead returns a
# sentinel comfortably above any real _MAX_ATTEMPTS/_MAX_HANDOFF_ATTEMPTS
# value, so the caller's existing "if attempt >= cap: give up" check trips
# and the document is flagged for manual review — wasteful if the outage
# was transient, but never an unbounded-cost loop, which is the one
# direction this project has already paid real money to learn matters more
# (see the 2026-08 $78-80 SQS retry-loop incident).

from __future__ import annotations

import time
from typing import Optional

# Comfortably above every real cap (_MAX_ATTEMPTS = _MAX_HANDOFF_ATTEMPTS = 3
# as of 2026-10) so a DynamoDB outage always trips "give up," never "retry
# forever" — see module docstring for why fail-closed is the deliberate
# choice here, unlike every other DynamoDB-backed helper in this pipeline.
FAIL_CLOSED_ATTEMPT = 999


class PdfAttemptCounter:
    """increment_and_get() never raises — any DynamoDB error returns
    FAIL_CLOSED_ATTEMPT instead (see module docstring for why that's the
    safe direction here, unlike this pipeline's other DynamoDB helpers)."""

    def __init__(self, table_name: str, region_name: Optional[str], logger=None):
        self._table_name = table_name
        self._region_name = region_name
        self._logger = logger
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("dynamodb", region_name=self._region_name or None)
        return self._client

    def _log(self, level: str, msg: str) -> None:
        if self._logger is not None:
            getattr(self._logger, level)(msg)

    def increment_and_get(self, filename: str, stage: str) -> int:
        """Atomically increments and returns the attempt count for this
        (filename, stage) pair — 1 on the first call, 2 on the next, etc.
        `stage` keeps independent pipeline stages (e.g. "lambda_extract" vs
        "ec2_build") from sharing a counter for the same filename."""
        key = f"{filename}#{stage}"
        expires_at = int(time.time()) + 14 * 24 * 60 * 60  # 14 days — tied to active in-flight retries, not history
        try:
            client = self._get_client()
            resp = client.update_item(
                TableName=self._table_name,
                Key={"attempt_key": {"S": key}},
                UpdateExpression="SET attempt_count = if_not_exists(attempt_count, :zero) + :one, expires_at = :ttl",
                ExpressionAttributeValues={
                    ":zero": {"N": "0"},
                    ":one": {"N": "1"},
                    ":ttl": {"N": str(expires_at)},
                },
                ReturnValues="UPDATED_NEW",
            )
            raw = resp.get("Attributes", {}).get("attempt_count", {}).get("N")
            return int(raw) if raw is not None else FAIL_CLOSED_ATTEMPT
        except Exception as e:
            self._log(
                "warning",
                f"[AttemptCounter] increment_and_get({key!r}) failed — DynamoDB unreachable, "
                f"failing CLOSED (returning {FAIL_CLOSED_ATTEMPT}, not 1) so the give-up cap "
                f"trips instead of silently resetting: {type(e).__name__}: {e}",
            )
            return FAIL_CLOSED_ATTEMPT

    def clear(self, filename: str, stage: str) -> None:
        """Deletes the counter for this (filename, stage) pair — call once
        that stage finishes successfully, so a future legitimate re-upload
        of the same filename starts back at attempt 1 instead of inheriting
        a stale elevated count. Best-effort: a failure here just leaves a
        harmless leftover row that the TTL cleans up eventually, never a
        cost risk (unlike a failure in increment_and_get)."""
        key = f"{filename}#{stage}"
        try:
            client = self._get_client()
            client.delete_item(TableName=self._table_name, Key={"attempt_key": {"S": key}})
        except Exception as e:
            self._log("warning", f"[AttemptCounter] clear({key!r}) failed (harmless, TTL will clean it up): {type(e).__name__}: {e}")


_COUNTER: Optional[PdfAttemptCounter] = None


def get_attempt_counter() -> PdfAttemptCounter:
    """Singleton accessor, lazily configured from conf.py — mirrors
    utils/pdf_cost_guard.py's get_cost_guard() pattern."""
    global _COUNTER
    if _COUNTER is None:
        import conf
        from utils.logger import get_logger
        _COUNTER = PdfAttemptCounter(
            table_name=str(getattr(conf, "PDF_ATTEMPT_COUNTER_DYNAMODB_TABLE", "restbiz-pdf-attempt-counter")),
            region_name=getattr(conf, "PDF_ATTEMPT_COUNTER_DYNAMODB_REGION", None),
            logger=get_logger(__name__),
        )
    return _COUNTER

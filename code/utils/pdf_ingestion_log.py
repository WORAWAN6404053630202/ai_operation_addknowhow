# code/utils/pdf_ingestion_log.py (feature/pdf-ingestion addition, 2026-10)
#
# Lightweight S3-backed activity log for the admin dashboard — one JSONL
# file per UTC day, one line appended per document the direct-ingest
# pipeline finishes processing (success or failure). Added alongside the
# daily cost ceiling (utils/pdf_cost_guard.py) specifically so the new
# admin dashboard (router/admin_pdf.py's /admin/api/pdf-status, app_pdf.py's
# /admin/dashboard) has something real to show about the live pipeline —
# before this, the only populated admin surface was the dead review-queue,
# which the direct-ingest path never writes to.
#
# Same low-level S3 access style as service/pdf_status_tracker.py (same
# boto3.client("s3", region_name=conf.AWS_REGION or None) construction, same
# bucket conf.PDF_INGESTION_S3_BUCKET, same best-effort/never-raise
# contract — a write failure here must never block real ingestion work,
# this is display-only).
#
# 2026-10 fix (PDF2-N5): this used to be one JSONL object per UTC day, with
# append_event() doing a full get-existing-object -> append-line ->
# put_object round trip — a heavy-upload day made every single append
# rewrite an ever-growing object (O(day's total size) per event, not O(1)),
# and read_recent_events() had no cap at all, returning every event across
# all `days` uncapped to the dashboard. Switched to one small object per
# event (keyed by day-prefix + a sortable per-event timestamp, so no
# read-before-write is needed at all — append_event is now a single PUT),
# enumerated via list_objects_v2 and capped by a new `limit` parameter.

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import boto3

import conf
from utils.logger import get_logger

logger = get_logger(__name__)


def _client():
    return boto3.client("s3", region_name=conf.AWS_REGION or None)


def _day_prefix(date_str: str) -> str:
    return f"{conf.PDF_INGESTION_LOG_S3_PREFIX}{date_str}/"


def _event_key(date_str: str, timestamp: float) -> str:
    # Zero-padded nanosecond timestamp sorts lexicographically in the same
    # order as chronologically — list_objects_v2 returns keys sorted, so
    # this gives newest/oldest ordering for free, no per-object metadata
    # read needed. A short random suffix guards the (practically
    # vanishing, but not impossible) case of two events landing on the
    # exact same nanosecond.
    return f"{_day_prefix(date_str)}{int(timestamp * 1_000_000_000):020d}_{uuid.uuid4().hex[:8]}.json"


def append_event(bucket: str, event: dict[str, Any]) -> None:
    """Best-effort, never raises. `event` should already contain whatever
    the caller wants displayed (see sqs_consumer.py's call site for the
    shape in use: timestamp, filename, new_count, update_count,
    duplicate_count, cost_usd, error) — this function only adds `timestamp`
    if the caller didn't set one.

    One S3 object per event (not one growing object per day) — a single
    PUT, no read-before-write, so this is O(1) per event regardless of how
    much activity already happened that day."""
    event = dict(event)
    ts = event.setdefault("timestamp", time.time())
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = _event_key(date_str, ts)
    try:
        client = _client()
        client.put_object(
            Bucket=bucket, Key=key,
            Body=json.dumps(event, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as e:
        logger.warning(f"[PdfIngestionLog] append_event failed for {date_str}, event NOT recorded (dashboard display only, pipeline unaffected): {type(e).__name__}: {e}")


def read_recent_events(bucket: str, days: int = 7, limit: Optional[int] = 500) -> list[dict]:
    """Returns events from the last `days` UTC days, newest-first, capped
    at `limit` (default 500 — generous for a human skimming a dashboard,
    but no longer unbounded). Missing days (no activity, or before this
    feature existed) are silently skipped — not an error. Pass
    limit=None for the old unbounded behavior."""
    client = _client()
    keys: list[str] = []
    today = datetime.now(timezone.utc).date()
    for offset in range(days):
        date_str = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        prefix = _day_prefix(date_str)
        try:
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                keys.extend(obj["Key"] for obj in page.get("Contents", []))
        except Exception as e:
            logger.warning(f"[PdfIngestionLog] Failed to list events under {prefix}: {e}")
            continue
    # Keys sort lexicographically = chronologically (zero-padded nanosecond
    # timestamp prefix) — newest last per day-prefix; reverse for newest-first.
    keys.sort(reverse=True)
    if limit is not None:
        keys = keys[:limit]

    events: list[dict] = []
    for key in keys:
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
            events.append(json.loads(body))
        except Exception as e:
            logger.warning(f"[PdfIngestionLog] Failed to read event {key}: {e}")
    return events


def classify_event_outcome(event: dict[str, Any]) -> str:
    """2026-10 (admin dashboard gap): append_event's own payload has no
    explicit status field — every caller that wants to SHOW a human
    whether a file's content is now live had to re-derive that from
    new_count/update_count/duplicate_count/error independently, which is
    exactly how router/admin_pdf.py's dashboard ended up reading a field
    ("shape") that didn't exist on the event anymore after schema
    unification. One shared, single source of truth instead.

    Returns one of:
      "failed"     — error is set. Checked first, regardless of counts —
                     a write failure after a drafted count is still a failure.
      "live"       — new_count + update_count > 0: genuinely new/changed
                     content, now queryable.
      "duplicate"  — only duplicate_count > 0: content recognized as fully
                     redundant with what's already live — a correct,
                     intentional no-op, not a problem.
      "no_content" — every count is 0: topic-splitting found nothing to
                     extract from this file at all. Worth a glance, not
                     quite a failure (no error was raised)."""
    if event.get("error"):
        return "failed"
    new_ct = event.get("new_count", 0) or 0
    upd_ct = event.get("update_count", 0) or 0
    dup_ct = event.get("duplicate_count", 0) or 0
    if new_ct + upd_ct > 0:
        return "live"
    if dup_ct > 0:
        return "duplicate"
    return "no_content"

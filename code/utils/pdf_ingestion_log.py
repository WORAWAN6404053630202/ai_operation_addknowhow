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
# this is display-only). Different from that module in one structural way:
# S3 has no native append, so append_event() does a full
# get-existing-object -> append-line -> put_object round trip rather than a
# single put. Safe without a concurrency guard because the SQS consumer
# processes one message at a time (confirmed during the 2026-10 pure-PDF
# audit — no evidence of parallel consumer workers); if that ever changes,
# this would need a lock or a different storage shape.

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

import conf
from utils.logger import get_logger

logger = get_logger(__name__)


def _client():
    return boto3.client("s3", region_name=conf.AWS_REGION or None)


def _day_key(date_str: str) -> str:
    return f"{conf.PDF_INGESTION_LOG_S3_PREFIX}{date_str}.jsonl"


def append_event(bucket: str, event: dict[str, Any]) -> None:
    """Best-effort, never raises. `event` should already contain whatever
    the caller wants displayed (see sqs_consumer.py's call site for the
    shape in use: timestamp, filename, new_count, update_count,
    duplicate_count, cost_usd, error) — this function only adds `timestamp`
    if the caller didn't set one."""
    event = dict(event)
    event.setdefault("timestamp", time.time())
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = _day_key(date_str)
    try:
        client = _client()
        try:
            existing = client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                existing = ""
            else:
                raise
        new_body = existing + json.dumps(event, ensure_ascii=False) + "\n"
        client.put_object(
            Bucket=bucket, Key=key,
            Body=new_body.encode("utf-8"),
            ContentType="application/x-ndjson",
        )
    except Exception as e:
        logger.warning(f"[PdfIngestionLog] append_event failed for {date_str}, event NOT recorded (dashboard display only, pipeline unaffected): {type(e).__name__}: {e}")


def read_recent_events(bucket: str, days: int = 7) -> list[dict]:
    """Returns events from the last `days` UTC days, newest-first. Missing
    day-files (no activity that day, or before this feature existed) are
    silently skipped — not an error."""
    client = _client()
    events: list[dict] = []
    today = datetime.now(timezone.utc).date()
    for offset in range(days):
        date_str = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        key = _day_key(date_str)
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                continue
            logger.warning(f"[PdfIngestionLog] Failed to read {key}: {e}")
            continue
        except Exception as e:
            logger.warning(f"[PdfIngestionLog] Failed to read {key}: {e}")
            continue
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"[PdfIngestionLog] Skipping malformed line in {key}")
    events.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    return events

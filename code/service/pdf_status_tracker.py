# code/service/pdf_status_tracker.py
"""Lightweight S3-backed progress/error visibility for the EC2 large-document
OCR path (service/pdf_large_extraction.py, feature/pdf-ingestion).

Why this exists: a large document that hits a transient failure (e.g. an
LLM provider running out of credits) restarts from page 1 on every SQS retry
(see sqs_consumer.py's process_extraction_result) and previously left ZERO
trace anywhere a human could see until it either succeeded or someone went
digging through journalctl by hand — the exact multi-hour debugging session
that motivated this file (2026-08-27: a 634-page document silently retried
for 13+ hours against an out-of-credits OpenRouter account with nothing
showing in the admin UI).

Scope: deliberately EC2-only — does not touch lambda/pdf_extraction/handler.py
at all. Lambda's own work (skip decision / small-doc extraction / handoff
marker write) already completes in seconds to a couple minutes, so the
visibility gap there isn't worth the separate Lambda zip-redeploy this would
require. The gap that actually hurt was the long-running EC2 handoff path
this module covers.

One status object per in-flight raw PDF, keyed by filename. "Object exists"
is itself the signal for "still in flight" — router/admin.py doesn't need to
cross-reference the review queue to know what to show as in-progress."""

from __future__ import annotations

import json
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

import conf
from utils.logger import get_logger

logger = get_logger(__name__)


def _client():
    return boto3.client("s3", region_name=conf.AWS_REGION or None)


def _status_key(filename: str) -> str:
    return f"{conf.PDF_STATUS_S3_PREFIX}{filename}.json"


def read_status(bucket: str, filename: str) -> Optional[dict]:
    try:
        obj = _client().get_object(Bucket=bucket, Key=_status_key(filename))
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        logger.warning(f"[PdfStatusTracker] Failed to read status for {filename}: {e}")
        return None


def write_status(
    bucket: str, filename: str, *, stage: str,
    pages_done: Optional[int] = None, pages_total: Optional[int] = None,
    attempt: int = 1, error: Optional[str] = None,
) -> None:
    """stage is "processing" or "failed". Best-effort: a write failure here
    must never block the real extraction work, just logs and moves on."""
    payload = {
        "filename": filename,
        "stage": stage,
        "pages_done": pages_done,
        "pages_total": pages_total,
        "attempt": attempt,
        "error": error[:500] if error else None,
        "updated_at": time.time(),
    }
    try:
        _client().put_object(
            Bucket=bucket, Key=_status_key(filename),
            Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
    except ClientError as e:
        logger.warning(f"[PdfStatusTracker] Failed to write status for {filename}: {e}")


def next_attempt_number(bucket: str, filename: str) -> int:
    """1 for a fresh document, or (prior attempt + 1) if a status object from
    an earlier failed attempt is still sitting there — so the admin UI can
    show 'attempt 23' instead of it always resetting to 1."""
    prior = read_status(bucket, filename)
    return (prior.get("attempt", 0) + 1) if prior else 1


def clear(bucket: str, filename: str) -> None:
    """Called once a real ReviewItem is saved — removes the status object so
    the admin UI stops showing this document as in-flight."""
    try:
        _client().delete_object(Bucket=bucket, Key=_status_key(filename))
    except ClientError as e:
        logger.warning(f"[PdfStatusTracker] Failed to clear status for {filename}: {e}")


def list_in_flight(bucket: str) -> list[dict]:
    """Every currently in-flight status object — the admin API's source for
    the 'processing / failed, retrying' rows that have no ReviewItem yet."""
    s3 = _client()
    results: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=conf.PDF_STATUS_S3_PREFIX):
        for obj in page.get("Contents", []):
            try:
                body = s3.get_object(Bucket=bucket, Key=obj["Key"])
                results.append(json.loads(body["Body"].read().decode("utf-8")))
            except Exception as e:
                logger.warning(f"[PdfStatusTracker] Failed to read {obj['Key']}: {e}")
    return results

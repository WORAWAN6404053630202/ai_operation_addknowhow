# code/utils/pdf_cost_guard.py (feature/pdf-ingestion addition, 2026-10)
#
# Cross-process daily $ spend TRACKER for the PDF ingestion pipeline, feeding
# the admin dashboard's running-total display (router/admin_pdf.py's
# /api/pdf-status, app_pdf.py's /admin/dashboard). OBSERVATIONAL ONLY.
#
# History: this briefly ALSO gated processing as a calendar-day auto-block
# (raise once today's total crossed conf.DAILY_BUDGET_USD, relying on SQS
# redelivery to retry the next UTC day once the total reset) — removed the
# same week it shipped. Real-world review of the design surfaced the actual
# flaw: a calendar-day block can't distinguish a genuine large bulk upload
# (legitimate — many real documents, each making real progress) from an
# actual runaway malfunction (the same document retrying over and over,
# never completing) — it punished both identically, and for the first case
# that just means an arbitrary multi-day wait for real backlog work with no
# benefit. The actual fix for "something is stuck looping" is a BOUNDED
# PER-DOCUMENT RETRY CAP, which targets that exact failure mode directly
# without touching legitimate throughput at all — see
# service/sqs_consumer.py's _MAX_HANDOFF_ATTEMPTS and lambda/pdf_extraction/
# handler.py's _MAX_ATTEMPTS. This module's job narrowed to just: track and
# display total spend so a human glancing at the dashboard notices if a day
# looks unusually expensive — never block anything on its own.
#
# DynamoDB-backed, cross-process (mirrors utils/pdf_rate_limiter.py's
# GlobalTyphoonRateLimiter and utils/pdf_draft_cache.py's DraftCache — same
# "table must be created manually first" expectation) but simpler than
# either: a single atomic counter per UTC day, so plain update_item (one
# condition to satisfy: none — it's an unconditional increment) rather than
# the rate limiter's two-condition transact_write_items.
#
# Table schema (create manually, see conf.PDF_COST_GUARD_DYNAMODB_TABLE):
#   - Partition key `date_bucket` (String) — UTC "YYYY-MM-DD".
#   - Attribute `running_total_usd` (Number) — cumulative cost_usd recorded
#     so far today, across BOTH the Lambda (OCR/vision) and the EC2 consumer
#     (drafting/resolver) call sites, so the dashboard shows ONE real total.
#   - Attribute `expires_at` (Number, epoch seconds, DynamoDB TTL attribute)
#     — ~35 days out, so old day-rows clean up automatically. Must enable
#     TTL on this attribute in the table's console settings after creating it.
#
# Fail-open: if DynamoDB is unreachable, get_today_total() returns 0.0 —
# the dashboard just shows $0.00 for that gap rather than erroring, same
# philosophy as the rate limiter/draft cache. Since nothing is gated on this
# anymore, there's no longer a "something stops working" direction to worry
# about either way.

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional


def _today_bucket() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class DailyCostGuard:
    """record_cost()/get_today_total() are both fail-open and NEVER raise —
    any DynamoDB error (table missing, throttled, network down, no
    permissions) must degrade to "dashboard shows $0.00 for now", never
    block or fail the PDF pipeline itself (nothing in the pipeline depends
    on this class's output anymore — see module docstring)."""

    def __init__(self, table_name: str, region_name: Optional[str], ceiling_usd: float, logger=None):
        self._table_name = table_name
        self._region_name = region_name
        self._ceiling_usd = ceiling_usd
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

    def record_cost(self, cost_usd: float) -> None:
        if cost_usd <= 0:
            return
        bucket = _today_bucket()
        expires_at = int(time.time()) + 35 * 24 * 60 * 60
        try:
            client = self._get_client()
            client.update_item(
                TableName=self._table_name,
                Key={"date_bucket": {"S": bucket}},
                UpdateExpression=(
                    "SET running_total_usd = if_not_exists(running_total_usd, :zero) + :amount, "
                    "expires_at = :ttl"
                ),
                ExpressionAttributeValues={
                    ":zero": {"N": "0"},
                    ":amount": {"N": str(cost_usd)},
                    ":ttl": {"N": str(expires_at)},
                },
            )
        except Exception as e:
            self._log(
                "warning",
                f"[CostGuard] record_cost({cost_usd:.6f}) for {bucket} failed, spend NOT tracked "
                f"for this call (pipeline unaffected): {type(e).__name__}: {e}",
            )

    def get_today_total(self) -> float:
        bucket = _today_bucket()
        try:
            client = self._get_client()
            resp = client.get_item(TableName=self._table_name, Key={"date_bucket": {"S": bucket}})
            item = resp.get("Item")
            if not item:
                return 0.0
            raw = item.get("running_total_usd", {}).get("N")
            return float(raw) if raw is not None else 0.0
        except Exception as e:
            self._log(
                "warning",
                f"[CostGuard] get_today_total() for {bucket} failed, dashboard will show "
                f"$0.00 until DynamoDB is reachable again: {type(e).__name__}: {e}",
            )
            return 0.0

    @property
    def ceiling_usd(self) -> float:
        """Dashboard reference amount (conf.DAILY_BUDGET_USD) — purely for
        the color-coded display, not an enforced limit. See module
        docstring for why this class no longer gates anything."""
        return self._ceiling_usd


_GUARD: Optional[DailyCostGuard] = None


def get_cost_guard() -> DailyCostGuard:
    """Singleton accessor, lazily configured from conf.py — mirrors
    utils/pdf_draft_cache.py's get_draft_cache() pattern."""
    global _GUARD
    if _GUARD is None:
        import conf
        from utils.logger import get_logger
        _GUARD = DailyCostGuard(
            table_name=str(getattr(conf, "PDF_COST_GUARD_DYNAMODB_TABLE", "restbiz-pdf-cost-guard")),
            region_name=getattr(conf, "PDF_COST_GUARD_DYNAMODB_REGION", None),
            ceiling_usd=float(getattr(conf, "DAILY_BUDGET_USD", 25.0)),
            logger=get_logger(__name__),
        )
    return _GUARD

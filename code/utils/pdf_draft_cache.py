# code/utils/pdf_draft_cache.py (feature/pdf-ingestion addition, 2026-10)
#
# Content-hash-keyed cache for the PDF pipeline's topic-splitting and
# field-drafting LLM calls (service/pdf_field_drafting.py,
# service/pdf_knowhow_drafting.py). Same input text always drafts to the
# same correct output (these are pure functions of their input text, by
# design) — so hashing the input and checking a shared cache before calling
# the LLM lets an exact re-upload skip every drafting call entirely, AND
# lets an updated document skip drafting for whichever individual topics'
# text is byte-identical to one already seen, with no page-diffing logic
# needed. See service/pdf_supersede_resolver.py's module docstring for how
# this interacts with (and is independent of) the separate new/update/
# duplicate decision against LIVE data — this cache only ever avoids
# REDRAFTING cost, never decides what's already live.
#
# DynamoDB-backed, cross-process (mirrors utils/pdf_rate_limiter.py's
# GlobalTyphoonRateLimiter — same fail-open philosophy, same "table must be
# created manually first" expectation) but much simpler: a cache only needs
# plain get_item/put_item, not GlobalTyphoonRateLimiter's atomic
# transact_write_items (there's no "claim" to coordinate, just a lookup).
#
# Table schema (create manually, see conf.PDF_DRAFT_CACHE_DYNAMODB_TABLE):
#   - Partition key `cache_key` (String) — see content_hash()'s callers for
#     the exact key shape, e.g. "topics:v1:<sha256>" or "draft:v1:<sha256>".
#     The version segment comes from a constant declared NEXT TO the prompt
#     it governs (TOPIC_SPLIT_PROMPT_VERSION / DRAFT_FIELDS_PROMPT_VERSION
#     in pdf_field_drafting.py, KNOWHOW_TOPIC_SPLIT_PROMPT_VERSION in
#     pdf_knowhow_drafting.py) specifically so a future prompt edit and its
#     cache-invalidation live in the same diff — not a separate constant
#     elsewhere that's easy to forget to bump.
#   - Attribute `result_json` (String) — the cached result, JSON-serialized.
#   - Attribute `created_at` (Number, epoch seconds) — informational only,
#     no TTL is set (cache persists indefinitely; invalidation is via the
#     version segment in the key, not expiry).

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Optional


# 2026-10 fix (data-integrity audit): the cache key's version segment used
# to be ONLY each prompt's own PROMPT_VERSION constant (DRAFT_FIELDS_
# PROMPT_VERSION etc. in pdf_field_drafting.py/pdf_knowhow_drafting.py) —
# covering the LLM call's input/instructions, but not any of the
# deterministic code that runs on the result AFTER the LLM call returns
# (merge_topic_chunks, _resolve_page_overlaps, _merge_drafted_batches). A
# future fix to any of THAT code — the same class of fix this whole audit
# has been making all session — would never invalidate an already-cached
# result: a previously-cached document re-uploaded after such a fix would
# silently keep serving the pre-fix, still-buggy drafted output forever,
# logged only as an unremarkable cache hit. Research backing this: build
# systems' content-addressable action caches (e.g. Bazel's Action Cache)
# key on a hash of BOTH the inputs AND the action's own command/logic
# definition — "if even one byte of either changes, the cache key changes"
# — not inputs alone. LOGIC_VERSION is the "action definition" half of
# that key for this cache: bump it whenever ANY downstream processing of a
# cached topic-split/draft result changes, even when the LLM prompt itself
# is untouched. Declared here (not per-prompt) since it covers logic
# shared across all 3 cache key sites in sqs_consumer.py, not any one
# prompt specifically.
#
# v2 (2026-10, PDF ingestion schema unification): the structured_license/
# know_how classification step and its two parallel builder pipelines were
# removed — every document now goes through one topic-split + field-draft
# pipeline (see sqs_consumer.py's _build_items_direct, pdf_direct_ingest.
# build_merged_documents, pdf_supersede_resolver.resolve_topic). Any
# pre-v2 cached result reflects the old two-path logic and must not be
# served as-is under the new pipeline's assumptions.
LOGIC_VERSION = "v2"


def content_hash(text: str) -> str:
    """SHA-256 of the RAW text, no normalization. Deliberately conservative
    (v1): guarantees two meaningfully-different inputs never collide into a
    false cache hit, at the cost of a slightly lower hit-rate if OCR
    produces tiny non-deterministic whitespace differences between runs of
    an otherwise-identical page. A normalization pass is a safe future
    addition if the hit-rate turns out too low in practice — not done now,
    to keep this first version as low-risk as possible."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class DraftCache:
    """get()/put() are both fail-open and NEVER raise — any DynamoDB error
    (table missing, throttled, network down, no permissions) must degrade
    to "cache not available" (get → None, i.e. a miss; put → silently
    skipped), never block or fail the PDF pipeline. The pipeline already
    works correctly with zero caching; this is purely a cost/consistency
    optimization layered on top, and must never become a new failure mode."""

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

    def get(self, cache_key: str) -> Optional[dict]:
        try:
            client = self._get_client()
            resp = client.get_item(TableName=self._table_name, Key={"cache_key": {"S": cache_key}})
            item = resp.get("Item")
            if not item:
                return None
            raw = item.get("result_json", {}).get("S")
            if not raw:
                return None
            return json.loads(raw)
        except Exception as e:
            self._log("warning", f"[DraftCache] get({cache_key!r}) failed, treating as miss: {type(e).__name__}: {e}")
            return None

    def put(self, cache_key: str, result: Any) -> None:
        try:
            client = self._get_client()
            client.put_item(
                TableName=self._table_name,
                Item={
                    "cache_key": {"S": cache_key},
                    "result_json": {"S": json.dumps(result, ensure_ascii=False)},
                    "created_at": {"N": str(int(time.time()))},
                },
            )
        except Exception as e:
            self._log("warning", f"[DraftCache] put({cache_key!r}) failed, result NOT cached (pipeline unaffected): {type(e).__name__}: {e}")


_CACHE: Optional[DraftCache] = None


def get_draft_cache():
    """Singleton accessor, lazily configured from conf.py — mirrors
    service/local_vector_store.py's get_vs_manager() pattern."""
    global _CACHE
    if _CACHE is None:
        import conf
        from utils.logger import get_logger
        _CACHE = DraftCache(
            table_name=str(getattr(conf, "PDF_DRAFT_CACHE_DYNAMODB_TABLE", "restbiz-pdf-draft-cache")),
            region_name=getattr(conf, "PDF_DRAFT_CACHE_DYNAMODB_REGION", None),
            logger=get_logger(__name__),
        )
    return _CACHE

# code/service/pdf_supersede_resolver.py
"""Automated (no-human-review) content supersede resolver for the PDF
direct-ingest pipeline (feature/pdf-ingestion, 2026-10) — decides, for each
newly-drafted PDF topic, whether it is NEW (nothing comparable exists live),
a DUPLICATE (an existing live doc already covers this, content materially
unchanged — skip adding), or an UPDATE (an existing live doc covers this
topic but content differs — write the new one, then retire the old).

Replaces the ROLE of pdf_candidate_matching.py / pdf_knowhow_candidate_matching.py
(same 3-signal design: embedding similarity + identity-fuzzy match + LLM
judgment) for the LIVE direct-ingest path — those two modules are left
completely unchanged (they compare against the Google Sheet, advisory-only,
feeding the now-dead review-queue UI). This module targets the LIVE CHROMA
COLLECTION instead, and the LLM's judgment becomes a FINAL AUTOMATED
DECISION rather than advice for a human, because this pipeline has no human
review step at all (explicit product decision, 2026-10: PDFs become the
sole data source for everything, Sheets become a pure catalog with zero
pipeline involvement).

This module is the ONLY thing that makes "no human review, ever" safe: a
wrong automated decision is a cheap, fully-reversible archive move
(service.local_vector_store.archive_and_remove / restore_from_archive),
never a permanent data loss — see that module's docstring.

Default-to-NEW on any ambiguity or failure, always: a redundant doc costs
almost nothing to carry; a wrongly-retired live doc costs real answer
quality until someone notices and restores it from the archive via
code/scripts/pdf_archive_inspect.py."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from openai import OpenAI

import conf
from service.local_vector_store import get_vs_manager
from service.pdf_candidate_matching import _field_diff
from utils.llm_cost_logging import CostAccumulator, log_llm_cost
from utils.pdf_cost_guard import get_cost_guard
from utils.logger import get_logger
from utils.page_ranges import fuzzy_ratio
from utils.prompt_safety import INJECTION_GUARD

logger = get_logger(__name__)

# Narrower, categorical-vocabulary fields used for the cheap identity-fuzzy
# safety net — mirrors department+license_type's role in
# pdf_candidate_matching.py and main_topic+sub_topic's role in
# pdf_knowhow_candidate_matching.py.
_REGULATORY_IDENTITY_KEYS = ("department", "license_type")
_REGULATORY_COMPARISON_KEYS = ("department", "license_type", "operation_topic", "answer_guideline")
_KNOWHOW_IDENTITY_KEYS = ("main_topic", "sub_topic")
_KNOWHOW_COMPARISON_KEYS = ("main_topic", "sub_topic", "summary", "category")

# Know-how's deterministic catch-all topic (see sqs_consumer.py's
# _build_knowhow_items_direct) — its main_topic/sub_topic are fixed generic
# strings reused across every document that has any uncovered pages, so
# identity-matching it would false-positive-match every other document's
# own catch-all bucket. Always treated as NEW, never matched.
KNOWHOW_CATCHALL_MAIN_TOPIC = "เนื้อหาที่เหลือ (ไม่ถูกจัดหมวดหมู่โดยอัตโนมัติ)"


class SupersedeDecision:
    """decision: "new" | "update" | "duplicate".
    matched_doc_ids: live Chroma doc ids to retire — only non-empty for "update"."""

    __slots__ = ("decision", "matched_doc_ids", "reasoning")

    def __init__(self, decision: str, matched_doc_ids: Optional[list[str]] = None, reasoning: str = ""):
        self.decision = decision
        self.matched_doc_ids = matched_doc_ids or []
        self.reasoning = reasoning

    def __repr__(self) -> str:
        return f"SupersedeDecision({self.decision!r}, matched_doc_ids={self.matched_doc_ids!r})"


def _comparison_text(values: dict, keys: tuple) -> str:
    parts = [str(values.get(k) or "").strip() for k in keys if str(values.get(k) or "").strip()]
    return " | ".join(parts)


def _embedding_candidates(comparison_text: str, data_type_filter: dict) -> dict[str, dict]:
    """Returns {doc_id: {"score", "metadata", "content"}} for live docs
    whose embedding similarity to comparison_text clears
    conf.PDF_SUPERSEDE_EMBEDDING_FLOOR, capped at
    PDF_SUPERSEDE_EMBEDDING_MAX_CANDIDATES. Queries Chroma's own HNSW index
    directly (O(log n)) — the one thing the Sheet-only predecessor module
    couldn't do, since a Sheet has no vector index to query."""
    if not comparison_text.strip():
        return {}
    floor = float(getattr(conf, "PDF_SUPERSEDE_EMBEDDING_FLOOR", 0.70))
    min_gap = float(getattr(conf, "PDF_SUPERSEDE_EMBEDDING_MIN_GAP", 0.05))
    max_candidates = int(getattr(conf, "PDF_SUPERSEDE_EMBEDDING_MAX_CANDIDATES", 10))
    hits = get_vs_manager().query_live_with_ids(
        comparison_text, k=max_candidates * 2, filter=data_type_filter,
    )
    # 2026-10 bug fix: this previously kept EVERY hit above the floor (up to
    # 2x max_candidates, since that's only used to size the Chroma query k=)
    # — the "_MAX_CANDIDATES" name was aspirational, not enforced. Now
    # actually capped via the same dynamic-gap-cutoff logic already proven
    # in pdf_candidate_matching.py's _dynamic_cutoff_count (a real gap in
    # the sorted score distribution, not a blind top-N cut).
    hits_sorted = sorted(hits, key=lambda h: h["score"], reverse=True)
    keep_count = _dynamic_cutoff_count([h["score"] for h in hits_sorted], floor, min_gap, max_candidates)
    return {h["id"]: h for h in hits_sorted[:keep_count]}


def _dynamic_cutoff_count(sorted_scores: list[float], floor: float, min_gap: float, max_candidates: int) -> int:
    """How many of the descending-sorted scores to keep — the largest gap
    between consecutive scores above the floor, not a fixed top-K. Ported
    from pdf_candidate_matching.py's proven implementation (same reasoning:
    a document with 7 genuinely-similar candidates should show 7, one with
    none should show none)."""
    above_floor = [s for s in sorted_scores if s >= floor]
    if len(above_floor) <= 1:
        return len(above_floor)
    gaps = [(above_floor[i] - above_floor[i + 1], i) for i in range(len(above_floor) - 1)]
    best_gap, best_idx = max(gaps)
    if best_gap >= min_gap:
        return min(best_idx + 1, max_candidates)
    return min(len(above_floor), max_candidates)


def _identity_fuzzy_candidates(
    new_values: dict, data_type_filter: dict, identity_keys: tuple,
) -> dict[str, dict]:
    """Independent of embeddings entirely — metadata-only Chroma read (cheap,
    no document bodies/embeddings needed), fuzzy-matching the categorical
    identity fields. Catches same-topic-different-wording cases where the
    free-text embedding signal under-scores (Thai regulatory/know-how text
    reuses a lot of boilerplate phrasing)."""
    threshold = float(getattr(conf, "PDF_SUPERSEDE_IDENTITY_FUZZY_THRESHOLD", 0.82))
    vectorstore = get_vs_manager().vectorstore
    if vectorstore is None:
        get_vs_manager().connect_to_existing(fail_if_empty=False)
        vectorstore = get_vs_manager().vectorstore
    if vectorstore is None:
        return {}
    raw = vectorstore._collection.get(where=data_type_filter, include=["metadatas", "documents"])
    ids = raw.get("ids") or []
    metadatas = raw.get("metadatas") or []
    documents = raw.get("documents") or []
    out: dict[str, dict] = {}
    for doc_id, meta, content in zip(ids, metadatas, documents):
        meta = meta or {}
        matched_fields = 0
        for key in identity_keys:
            new_v = str(new_values.get(key) or "").strip()
            old_v = str(meta.get(key) or "").strip()
            if new_v and fuzzy_ratio(new_v, old_v) >= threshold:
                matched_fields += 1
        if matched_fields == len(identity_keys):
            out[doc_id] = {"score": None, "metadata": meta, "content": content or ""}
    return out


_LLM_JUDGE_PROMPT = """คุณกำลังช่วยตัดสินใจอัตโนมัติว่าเอกสาร/หัวข้อใหม่จากไฟล์ PDF ที่อัปโหลดเข้ามา เป็นเรื่องเดียวกันกับข้อมูลที่มีอยู่แล้วในระบบหรือไม่ และถ้าเป็นเรื่องเดียวกัน เนื้อหาเปลี่ยนไปจากเดิมหรือไม่ — การตัดสินใจนี้จะถูกนำไปใช้จริงทันทีโดยไม่มีคนตรวจสอบ จึงต้องระมัดระวัง: ถ้าไม่แน่ใจจริงๆ ว่าเป็นเรื่องเดียวกัน ให้ตอบว่าไม่ใช่เรื่องเดียวกัน (is_same_topic=false) ดีกว่าตัดสินผิด

""" + INJECTION_GUARD + """

หัวข้อใหม่:
{new_summary}

รายการข้อมูลเดิมที่อาจเกี่ยวข้อง:
{candidates_block}

สำหรับแต่ละ candidate_id ตอบว่า:
- is_same_topic: true ถ้าเป็นเรื่อง/ใบอนุญาต/หัวข้อเดียวกันจริง (ไม่ใช่แค่หน่วยงาน/หมวดเดียวกัน) false ถ้าไม่ใช่
- ถ้า is_same_topic=true ให้ระบุ content_status: "duplicate" (เนื้อหาเหมือนเดิม ไม่มีอะไรเปลี่ยน) หรือ "update" (เนื้อหาเปลี่ยนไปจากเดิม เช่น ค่าธรรมเนียม ขั้นตอน เงื่อนไข ที่ต่างจากเดิม)

ตอบเป็น JSON เท่านั้น:
{{"results": [{{"candidate_id": "<id>", "is_same_topic": true/false, "content_status": "duplicate"|"update"|null, "reasoning": "<เหตุผลสั้นๆ>"}}]}}
"""


def _llm_judge(
    new_summary: str, candidates: list[dict[str, Any]], model: str,
    cost_accumulator: Optional[CostAccumulator] = None,
) -> dict[str, dict]:
    """Returns {candidate_id: {"is_same_topic", "content_status", "reasoning"}}.
    Never raises to the caller — any failure/malformed response returns {}
    (empty), which the caller must treat as "no confirmed match" (safe
    default: NEW), never inferred as a confirmed duplicate/update."""
    if not candidates:
        return {}
    candidates_block = "\n".join(
        f"[{c['candidate_id']}] {c['preview']}" for c in candidates
    )
    prompt = _LLM_JUDGE_PROMPT.format(new_summary=new_summary[:1500], candidates_block=candidates_block)
    try:
        client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
        _call_start = time.monotonic()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            response_format={"type": "json_object"},
            extra_body={"reasoning": {"enabled": False}},
        )
        log_llm_cost(
            logger, "SupersedeResolver/LLMJudge", model, resp,
            time.monotonic() - _call_start, accumulator=cost_accumulator, cost_guard=get_cost_guard(),
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        results = parsed.get("results", [])
        out: dict[str, dict] = {}
        for r in results:
            if not isinstance(r, dict) or "candidate_id" not in r:
                continue
            out[str(r["candidate_id"])] = {
                "is_same_topic": bool(r.get("is_same_topic")),
                "content_status": r.get("content_status"),
                "reasoning": str(r.get("reasoning", "")),
            }
        return out
    except Exception as e:
        logger.error(f"[SupersedeResolver] LLM judgment failed, defaulting every candidate to no-match (safe → new): {e}")
        return {}


def _resolve(
    new_values: dict[str, str],
    identity_keys: tuple,
    comparison_keys: tuple,
    data_type_filter: dict,
    source_file: str,
    cost_accumulator: Optional[CostAccumulator] = None,
) -> SupersedeDecision:
    comparison_text = _comparison_text(new_values, comparison_keys)
    if not comparison_text.strip():
        return SupersedeDecision("new", reasoning="no comparable text drafted")

    try:
        pool: dict[str, dict] = {}
        pool.update(_embedding_candidates(comparison_text, data_type_filter))
        # Merge identity-fuzzy hits WITHOUT clobbering an embedding hit's
        # real score (2026-10 bug fix — plain dict.update() applied here
        # second used to silently overwrite a doc's real embedding score
        # with None whenever both signals matched the same doc; cosmetic
        # only, never affected the decision, but fixed alongside the
        # truncation bug below since it's the same block).
        for doc_id, info in _identity_fuzzy_candidates(new_values, data_type_filter, identity_keys).items():
            pool.setdefault(doc_id, info)
    except Exception as e:
        logger.error(f"[SupersedeResolver] Candidate pool fetch failed for {source_file!r}, defaulting to new: {e}")
        return SupersedeDecision("new", reasoning=f"candidate fetch failed: {e}")

    if not pool:
        return SupersedeDecision("new", reasoning="no live candidates found")

    # 2026-10 bug fix: this used to silently slice the pool to the first
    # `batch_size` candidates (dict insertion order — embedding hits always
    # first) and send ONLY those to the LLM judge. Any remaining candidates,
    # including identity-fuzzy matches specifically designed to catch what
    # embeddings under-score, were silently dropped — never judged, never
    # logged as dropped — so the resolver could default to "new" even when
    # a genuine duplicate/update existed in the pool. Fixed by batching
    # through the WHOLE pool (every candidate gets judged, across as many
    # LLM calls as needed), mirroring pdf_candidate_matching.py's proven
    # _llm_scan_candidates pattern.
    batch_size = int(getattr(conf, "PDF_SUPERSEDE_LLM_BATCH_SIZE", 15))
    pool_items = list(pool.items())
    if len(pool_items) > batch_size:
        logger.info(
            f"[SupersedeResolver] {source_file!r}: {len(pool_items)} candidate(s) found, "
            f"judging in {-(-len(pool_items) // batch_size)} batch(es) of up to {batch_size}"
        )
    model = str(getattr(conf, "OPENROUTER_MODEL_PDF_SUPERSEDE", conf.OPENROUTER_MODEL_PDF_MATCHING))
    judgments: dict[str, dict] = {}
    for i in range(0, len(pool_items), batch_size):
        batch = pool_items[i:i + batch_size]
        candidates_for_llm = []
        for doc_id, info in batch:
            meta = info.get("metadata") or {}
            existing_values = {k: meta.get(k) for k in comparison_keys}
            diff = _field_diff(new_values, existing_values)
            preview = _comparison_text(existing_values, comparison_keys)[:300]
            candidates_for_llm.append({
                "candidate_id": doc_id,
                "preview": f"{preview} | diff={diff}" if diff else preview,
            })
        judgments.update(_llm_judge(comparison_text, candidates_for_llm, model, cost_accumulator=cost_accumulator))

    update_ids = [doc_id for doc_id, j in judgments.items() if j["is_same_topic"] and j.get("content_status") == "update"]
    duplicate_ids = [doc_id for doc_id, j in judgments.items() if j["is_same_topic"] and j.get("content_status") == "duplicate"]

    if duplicate_ids and not update_ids:
        return SupersedeDecision("duplicate", matched_doc_ids=duplicate_ids, reasoning="LLM confirmed duplicate")
    if update_ids:
        return SupersedeDecision("update", matched_doc_ids=update_ids, reasoning="LLM confirmed content changed")
    return SupersedeDecision("new", reasoning="no candidate confirmed as same-topic by LLM")


def resolve_regulatory_topic(
    merged_topic: dict[str, Any], source_file: str, cost_accumulator: Optional[CostAccumulator] = None,
) -> SupersedeDecision:
    """merged_topic: output of pdf_direct_ingest.merge_drafted_fields() —
    department/license_type/operation_topic/answer_guideline/etc. keyed by
    internal snake_case names. Scoped to data_type="regulatory" candidates
    only."""
    return _resolve(
        new_values=merged_topic,
        identity_keys=_REGULATORY_IDENTITY_KEYS,
        comparison_keys=_REGULATORY_COMPARISON_KEYS,
        data_type_filter={"data_type": "regulatory"},
        source_file=source_file,
        cost_accumulator=cost_accumulator,
    )


def resolve_knowhow_topic(
    topic_dict: dict[str, Any], source_file: str, cost_accumulator: Optional[CostAccumulator] = None,
) -> SupersedeDecision:
    """topic_dict: a knowhow_topics entry (main_topic/sub_topic/summary/
    category/full_text/...) from sqs_consumer.py's _build_knowhow_items_direct.
    The deterministic catch-all topic is always NEW (see
    KNOWHOW_CATCHALL_MAIN_TOPIC's docstring above) — never matched, since its
    main_topic/sub_topic are generic placeholder strings shared across every
    document. Scoped to business_guide/marketing candidates only (the shapes
    both Sheet-sourced and PDF-sourced know-how content use)."""
    if str(topic_dict.get("main_topic") or "") == KNOWHOW_CATCHALL_MAIN_TOPIC:
        return SupersedeDecision("new", reasoning="catch-all topic, never matched")
    return _resolve(
        new_values=topic_dict,
        identity_keys=_KNOWHOW_IDENTITY_KEYS,
        comparison_keys=_KNOWHOW_COMPARISON_KEYS,
        data_type_filter={"data_type": {"$in": ["business_guide", "marketing"]}},
        source_file=source_file,
        cost_accumulator=cost_accumulator,
    )


def expand_knowhow_matches_to_full_chunks(matched_doc_ids: list[str]) -> list[str]:
    """A know-how UPDATE match's matched_doc_ids come from whichever single
    chunk(s) the embedding/identity signal happened to surface — a long
    topic can be split across several chunk_index/chunk_total sub-documents
    (pdf_direct_ingest.build_knowhow_documents), and retiring only the
    surfaced chunk(s) would leave its siblings live, orphaned, and
    confusing (same main_topic/sub_topic, no replacement reference).
    Fetches each matched doc's own main_topic/sub_topic, then expands to
    every sibling chunk sharing that exact pair. Deduped, order not
    significant (caller passes this straight to archive_and_remove)."""
    if not matched_doc_ids:
        return []
    vectorstore = get_vs_manager().vectorstore
    if vectorstore is None:
        get_vs_manager().connect_to_existing(fail_if_empty=False)
        vectorstore = get_vs_manager().vectorstore
    if vectorstore is None:
        return list(matched_doc_ids)
    raw = vectorstore._collection.get(ids=matched_doc_ids, include=["metadatas"])
    seen_pairs: set[tuple[str, str]] = set()
    for meta in raw.get("metadatas") or []:
        meta = meta or {}
        pair = (str(meta.get("main_topic") or ""), str(meta.get("sub_topic") or ""))
        if pair[0] or pair[1]:
            seen_pairs.add(pair)

    all_ids: set[str] = set(matched_doc_ids)
    for main_topic, sub_topic in seen_pairs:
        all_ids.update(collect_full_sibling_chunk_ids(main_topic, sub_topic))
    return list(all_ids)


def collect_full_sibling_chunk_ids(main_topic: str, sub_topic: str) -> list[str]:
    """A know-how UPDATE match must retire EVERY chunk of the old topic, not
    just whichever single chunk the embedding/identity signal happened to
    surface (a long topic can be split across several chunk_index/
    chunk_total sub-documents — see pdf_direct_ingest.build_knowhow_documents).
    Exact metadata lookup, not a similarity search."""
    vectorstore = get_vs_manager().vectorstore
    if vectorstore is None:
        get_vs_manager().connect_to_existing(fail_if_empty=False)
        vectorstore = get_vs_manager().vectorstore
    if vectorstore is None:
        return []
    where = {"$and": [{"main_topic": main_topic}, {"sub_topic": sub_topic}]}
    raw = vectorstore._collection.get(where=where, include=[])
    return raw.get("ids") or []

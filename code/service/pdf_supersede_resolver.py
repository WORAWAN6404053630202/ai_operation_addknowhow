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
from service.pdf_candidate_matching import _field_diff, _PROCEDURAL_FIELD_KEYS
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

# Fields an actual content UPDATE would change, fed to _field_diff below —
# deliberately separate from *_IDENTITY_KEYS/*_COMPARISON_KEYS above, which
# are about FINDING the right candidate, not describing what changed about
# it. 2026-10 fix (PDF-M1): _field_diff's own default (_PROCEDURAL_FIELD_KEYS,
# imported from pdf_candidate_matching.py) only matches the regulatory shape
# (registration_type/fees/operation_duration/...) — passed unconditionally
# for know-how topics too, which share ZERO keys with that list, silently
# producing an empty diff for every know-how candidate regardless of whether
# summary/category actually changed.
_REGULATORY_DIFF_KEYS = _PROCEDURAL_FIELD_KEYS
_KNOWHOW_DIFF_KEYS = ["summary", "category"]

# 2026-10 (PDF ingestion schema unification): every topic now goes through
# ONE merged pipeline (sqs_consumer._build_items_direct →
# pdf_direct_ingest.build_merged_documents) instead of being pre-classified
# into a "structured_license" or "know_how" shape — main_topic/sub_topic is
# ALWAYS populated now (every topic gets one), while department/license_type
# are populated only when the content genuinely is a government/legal
# matter. main_topic+sub_topic is therefore the primary identity for every
# candidate match; department+license_type is checked as an ADDITIONAL,
# stronger confirming signal — _identity_fuzzy_candidates already behaves
# correctly as a no-op for this second check when new_values lacks
# department/license_type (requires a truthy new_v before counting a
# match), so no new "only when both sides have it" branching is needed —
# it falls out of the existing per-field match logic for free.
_MERGED_IDENTITY_KEY_TUPLES = [_KNOWHOW_IDENTITY_KEYS, _REGULATORY_IDENTITY_KEYS]
_MERGED_COMPARISON_KEYS = ("main_topic", "sub_topic", "department", "license_type", "operation_topic", "answer_guideline")
_MERGED_DIFF_KEYS = _REGULATORY_DIFF_KEYS + _KNOWHOW_DIFF_KEYS

# 2026-10 fix (PDF2-C2): pdf_direct_ingest.build_knowhow_documents never
# writes "summary"/"category" metadata keys — it writes the same logical
# fields under "answer_guideline"/"content_type" instead (the real field
# names retrieval/prompt-building code elsewhere depends on). Maps the
# logical names used in _KNOWHOW_COMPARISON_KEYS/_KNOWHOW_DIFF_KEYS above
# to the actual Chroma metadata keys when reading an EXISTING live doc back
# (see _resolve's metadata_key_aliases param) — new_values (freshly drafted,
# not yet written) already uses the logical names directly.
_KNOWHOW_METADATA_KEY_ALIASES = {"summary": "answer_guideline", "category": "content_type"}

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
    # 2026-10 fix (data-integrity audit): widened from a fixed 2x to a
    # configurable multiplier (default 5x) — see
    # conf.PDF_SUPERSEDE_EMBEDDING_QUERY_OVERSAMPLE's comment. A corpus with
    # more than 2x max_candidates genuinely-similar live docs for one topic
    # used to never even have the rest fetched from Chroma at all.
    oversample = int(getattr(conf, "PDF_SUPERSEDE_EMBEDDING_QUERY_OVERSAMPLE", 5))
    hits = get_vs_manager().query_live_with_ids(
        comparison_text, k=max_candidates * oversample, filter=data_type_filter,
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
    # 2026-10 fix (PDF2-M4): this used to also fetch include=["documents"]
    # — this function's own docstring already said "metadata-only... no
    # document bodies needed", and `content` below was never actually read
    # anywhere downstream (confirmed: no caller of this function's return
    # value ever accesses the "content" key). Called once per topic from
    # _resolve, a 40-topic book against a 1,500-doc corpus pulled ~40
    # full-partition fetches of real document bodies — ~600MB of churn and
    # real latency on the single-threaded consumer. Dropping "documents"
    # removes the actual cost driver; the per-topic repetition of this now-
    # cheap metadata-only fetch is left as a known, much smaller residual
    # (hoisting it out of the per-topic loop would need threading a
    # pre-fetched partition through 2 more call sites across both
    # sqs_consumer.py builders).
    raw = vectorstore._collection.get(where=data_type_filter, include=["metadatas"])
    ids = raw.get("ids") or []
    metadatas = raw.get("metadatas") or []
    out: dict[str, dict] = {}
    for doc_id, meta in zip(ids, metadatas):
        meta = meta or {}
        matched_fields = 0
        for key in identity_keys:
            new_v = str(new_values.get(key) or "").strip()
            old_v = str(meta.get(key) or "").strip()
            if new_v and fuzzy_ratio(new_v, old_v) >= threshold:
                matched_fields += 1
        if matched_fields == len(identity_keys):
            # "content" kept as an empty placeholder (not fetched above, see
            # this function's docstring) only to match _embedding_candidates'
            # output shape for the pool.setdefault() merge in _resolve.
            out[doc_id] = {"score": None, "metadata": meta, "content": ""}
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


_COMPLETENESS_CHECK_PROMPT = """คุณกำลังช่วยตรวจสอบว่าการแทนที่เอกสารเก่าด้วยเอกสารใหม่ (ระบบตัดสินแล้วว่าเป็นเรื่องเดียวกันและเนื้อหามีการเปลี่ยนแปลง) จะทำให้ข้อมูลสำคัญที่มีอยู่ในเอกสารเก่าหายไปหรือไม่ — การตัดสินใจนี้จะถูกนำไปใช้จริงทันทีโดยไม่มีคนตรวจสอบ

วิธีตรวจ: แยกเนื้อหาของเอกสารเก่าแต่ละ field ด้านล่างออกเป็น "ข้อเท็จจริงย่อย" (claim) ทีละข้อที่แยกจากกันได้ แล้วเช็คว่าแต่ละข้อเท็จจริงยังปรากฏอยู่ในเนื้อหาเอกสารใหม่ทั้งหมดหรือไม่ (ไม่ต้องคำต่อคำ ความหมายเดียวกันถือว่า "ยังมี") ระบุเฉพาะข้อเท็จจริงที่หายไปจริงๆ ไม่ใช่แค่เปลี่ยนคำพูด/จัดรูปแบบใหม่

""" + INJECTION_GUARD + """

เอกสารเก่า (field: เนื้อหา):
{old_fields_block}

เนื้อหาเอกสารใหม่ทั้งหมด (จะมาแทนที่เอกสารเก่า):
{new_content}

ตอบเป็น JSON เท่านั้น ระบุเฉพาะข้อเท็จจริงที่หายไปจริง (ไม่ใช่แค่เปลี่ยนคำพูด):
{{"missing_claims": [{{"field": "<ชื่อ field ของเอกสารเก่า>", "claim": "<ข้อเท็จจริงที่หายไป>"}}]}}
ถ้าไม่มีข้อเท็จจริงไหนหายไปเลย ตอบ {{"missing_claims": []}}
"""


def _new_content_blob(new_values: dict[str, str]) -> str:
    """Every non-empty, user-facing value in the new topic — not just
    diff_keys/comparison_keys — since a genuine replacement might carry an
    old field's information inside a DIFFERENT field's prose instead of the
    same-named one. Excludes internal bookkeeping keys (leading underscore,
    e.g. regulatory's "_raw_text") since those aren't what retrieval/prompt-
    building actually surfaces to a user — checking against what's really
    retrievable, not against the raw source text, is the point."""
    parts = [
        str(v).strip() for k, v in new_values.items()
        if not k.startswith("_") and str(v or "").strip()
    ]
    return "\n".join(parts)


def _check_content_completeness(
    old_values_by_id: dict[str, dict], new_values: dict[str, str], diff_keys: list[str], model: str,
    cost_accumulator: Optional[CostAccumulator] = None,
) -> dict[str, list[dict]]:
    """Second-stage check, layered on top of _resolve's deterministic blank-
    field guard — catches the case that guard CAN'T (old field non-empty,
    new field also non-empty, but the new field's content is narrower/
    missing specific detail the old one had). Claim-level decomposition,
    not a single holistic "does this cover it" judgment — research on LLM-
    as-judge evaluation documents "omission blindness" (judges are reliable
    at verifying what IS present, much less reliable at noticing what's
    ABSENT when asked to assess coverage as one holistic question); RAGAS's
    Context Recall metric and NLI-based factual-consistency checkers
    (SummaC/FactCC) converge on the same mitigation — decompose into atomic
    claims, verify each independently, rather than asking once "is this
    complete?".

    Returns {doc_id: [{"field", "claim"}, ...]} — only ids with 1+ missing
    claims. Only called for candidates that already passed the blank-field
    guard (i.e. a non-empty vs non-empty comparison is actually meaningful
    here) — deliberately NOT run for every candidate, keeping this
    additional LLM cost scoped to the one case the cheaper deterministic
    check structurally cannot see. Never raises — a failed call means "no
    claims flagged" (nothing EXTRA caught by this layer), never "block
    everything"; the blank-field guard already ran and stands regardless."""
    if not old_values_by_id:
        return {}
    new_content = _new_content_blob(new_values)
    if not new_content.strip():
        return {}
    out: dict[str, list[dict]] = {}
    for doc_id, existing_values in old_values_by_id.items():
        old_fields_block = "\n".join(
            f"- {key}: {existing_values[key]}"
            for key in diff_keys
            if str(existing_values.get(key) or "").strip()
        )
        if not old_fields_block.strip():
            continue
        prompt = _COMPLETENESS_CHECK_PROMPT.format(
            old_fields_block=old_fields_block, new_content=new_content[:4000],
        )
        try:
            client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
            _call_start = time.monotonic()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                response_format={"type": "json_object"},
                extra_body={"reasoning": {"enabled": False}},
            )
            log_llm_cost(
                logger, "SupersedeResolver/CompletenessCheck", model, resp,
                time.monotonic() - _call_start, accumulator=cost_accumulator, cost_guard=get_cost_guard(),
            )
            raw = (resp.choices[0].message.content or "{}").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            missing = [
                m for m in parsed.get("missing_claims", [])
                if isinstance(m, dict) and m.get("claim")
            ]
            if missing:
                out[doc_id] = missing
        except Exception as e:
            logger.error(
                f"[SupersedeResolver] Completeness check failed for {doc_id!r}, no extra claims flagged "
                f"(blank-field guard result, if any, still stands): {e}"
            )
    return out


def _resolve(
    new_values: dict[str, str],
    identity_key_tuples: list[tuple],
    comparison_keys: tuple,
    diff_keys: list[str],
    data_type_filter: dict,
    source_file: str,
    cost_accumulator: Optional[CostAccumulator] = None,
    metadata_key_aliases: Optional[dict[str, str]] = None,
) -> SupersedeDecision:
    # 2026-10 fix (PDF2-C2): comparison_keys/diff_keys are logical field
    # names used throughout this module (and sent to the LLM judge as the
    # new/candidate "shape"), but a stored live doc's actual Chroma metadata
    # key can differ — e.g. know-how topics are drafted with "summary"/
    # "category", but pdf_direct_ingest.build_knowhow_documents writes them
    # under "answer_guideline"/"content_type" (the real field names the
    # retrieval/prompt-building code elsewhere already depends on, so the
    # writer isn't changed). Without this mapping, meta.get("summary") and
    # meta.get("category") were always None for every existing know-how
    # doc — every re-upload's diff showed both as spuriously "added",
    # biasing the LLM judge toward "update" instead of "duplicate" even on
    # a byte-identical re-upload.
    metadata_key_aliases = metadata_key_aliases or {}
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
        # truncation bug below since it's the same block). Runs the
        # identity-fuzzy check once per key-tuple (2026-10, schema
        # unification) — e.g. main_topic+sub_topic always, department+
        # license_type as an additional signal — ORing every tuple's hits
        # into the same pool.
        for identity_keys in identity_key_tuples:
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
    existing_values_by_id: dict[str, dict] = {}
    for i in range(0, len(pool_items), batch_size):
        batch = pool_items[i:i + batch_size]
        candidates_for_llm = []
        for doc_id, info in batch:
            meta = info.get("metadata") or {}
            existing_values = {
                k: meta.get(metadata_key_aliases.get(k, k)) for k in set(comparison_keys) | set(diff_keys)
            }
            existing_values_by_id[doc_id] = existing_values
            diff = _field_diff(new_values, existing_values, diff_keys=diff_keys)
            preview = _comparison_text(existing_values, comparison_keys)[:300]
            candidates_for_llm.append({
                "candidate_id": doc_id,
                "preview": f"{preview} | diff={diff}" if diff else preview,
            })
        # 2026-10 fix (PDF2-N1): no membership check against the ids
        # actually sent in THIS batch — a later batch's response echoing
        # an id from an earlier batch (or any other id) used to be
        # accepted and could overwrite that id's real verdict. archive_
        # and_remove no-ops on a nonexistent id, so blast radius was
        # limited, but a genuine id from a different batch overwriting an
        # already-judged verdict was a real (if narrow) risk.
        _batch_ids = {doc_id for doc_id, _ in batch}
        _batch_judgments = _llm_judge(comparison_text, candidates_for_llm, model, cost_accumulator=cost_accumulator)
        judgments.update({k: v for k, v in _batch_judgments.items() if k in _batch_ids})

    update_ids = [doc_id for doc_id, j in judgments.items() if j["is_same_topic"] and j.get("content_status") == "update"]
    duplicate_ids = [doc_id for doc_id, j in judgments.items() if j["is_same_topic"] and j.get("content_status") == "duplicate"]

    # 2026-10 fix (live incident): the LLM's "update" verdict only judges
    # topic-identity + "did the content change", never whether the NEW
    # doc's drafted fields are a superset of what the OLD doc being retired
    # actually covers. Live-confirmed: "สถานที่ และ กำหนดเวลาการยื่นแจ้ง...pdf"
    # (a doc specifically about WHERE/WHEN to file) got judged "update" by
    # a broader, newer doc that covers the same regulatory topic in general
    # but left service_channel/operation_duration blank — archive_and_remove
    # then deleted the only place that WHERE/WHEN info lived, live, with no
    # human review to catch it (8 such cases found across one real
    # collection on first audit). Deterministic safety net, independent of
    # the LLM's judgment: for each doc_id the LLM says to retire, if the OLD
    # doc has a non-empty diff_keys field that the NEW doc leaves blank,
    # refuse to retire THAT doc — both stay live. Matches this module's own
    # stated philosophy ("Default-to-NEW on any ambiguity... a redundant doc
    # costs almost nothing to carry; a wrongly-retired live doc costs real
    # answer quality") applied at the per-field level, not just per-document.
    blocked_lost_fields: dict[str, list[str]] = {}
    for doc_id in update_ids:
        existing_values = existing_values_by_id.get(doc_id, {})
        lost = [
            key for key in diff_keys
            if str(existing_values.get(key) or "").strip() and not str(new_values.get(key) or "").strip()
        ]
        if lost:
            blocked_lost_fields[doc_id] = lost
    if blocked_lost_fields:
        for doc_id, lost in blocked_lost_fields.items():
            logger.warning(
                f"[SupersedeResolver] {source_file!r}: LLM judged {doc_id!r} as 'update' but the new "
                f"doc leaves field(s) {lost} blank that the OLD doc has content for — refusing to "
                f"retire it to avoid silently losing that data; both will stay live until manually "
                f"reviewed."
            )
        update_ids = [doc_id for doc_id in update_ids if doc_id not in blocked_lost_fields]

    # 2026-10 follow-up: the blank-field guard above only catches a field
    # going from populated to fully empty — it can't see a field that's
    # non-empty on BOTH sides but narrower/missing specific detail on the
    # new side (research-backed design: see _check_content_completeness's
    # docstring on claim decomposition vs. holistic LLM judgment). Scoped
    # to ONLY the update_ids that survived the free guard above, so this
    # extra LLM cost is paid only for the one case the deterministic check
    # structurally cannot see — not for every candidate.
    if update_ids:
        completeness_candidates = {
            doc_id: existing_values_by_id.get(doc_id, {}) for doc_id in update_ids
        }
        missing_claims = _check_content_completeness(
            completeness_candidates, new_values, diff_keys, model, cost_accumulator=cost_accumulator,
        )
        if missing_claims:
            for doc_id, claims in missing_claims.items():
                logger.warning(
                    f"[SupersedeResolver] {source_file!r}: LLM judged {doc_id!r} as 'update', and its "
                    f"fields aren't fully blank in the new doc, but a claim-level completeness check "
                    f"flagged {len(claims)} specific detail(s) that don't appear to carry over "
                    f"(e.g. {claims[0]['field']!r}: {claims[0]['claim']!r}) — refusing to retire it; "
                    f"both will stay live until manually reviewed."
                )
            update_ids = [doc_id for doc_id in update_ids if doc_id not in missing_claims]

    if duplicate_ids and not update_ids:
        return SupersedeDecision("duplicate", matched_doc_ids=duplicate_ids, reasoning="LLM confirmed duplicate")
    if update_ids:
        return SupersedeDecision("update", matched_doc_ids=update_ids, reasoning="LLM confirmed content changed")
    return SupersedeDecision("new", reasoning="no candidate confirmed as same-topic by LLM")


def resolve_topic(
    merged_topic: dict[str, Any], source_file: str, cost_accumulator: Optional[CostAccumulator] = None,
) -> SupersedeDecision:
    """merged_topic: output of pdf_direct_ingest.merge_drafted_fields() —
    main_topic/sub_topic/summary/category (always populated) plus
    department/license_type/operation_topic/answer_guideline/fees/etc.
    (populated only when the topic is genuinely a government/legal
    matter). Replaces the old resolve_regulatory_topic/resolve_knowhow_topic
    split (2026-10, PDF ingestion schema unification) — every topic now
    goes through this one function regardless of whether it ended up
    "regulatory" or "business_guide"-shaped after drafting.

    The deterministic catch-all topic is always NEW (see
    KNOWHOW_CATCHALL_MAIN_TOPIC's docstring above) — never matched, since
    its main_topic/sub_topic are generic placeholder strings shared across
    every document.

    Scoped to regulatory/business_guide/marketing candidates (every shape
    the pdf_auto pipeline can produce), deliberately NOT split by data_type
    — a topic that was previously mis-ingested under the OLD two-path
    pipeline as the wrong data_type (e.g. a law ingested as business_guide
    because it wasn't phrased as a license-application procedure) must
    still be findable and supersedable once a re-upload correctly resolves
    it as regulatory. Splitting the candidate pool by data_type would make
    that cross-shape correction impossible."""
    if str(merged_topic.get("main_topic") or "") == KNOWHOW_CATCHALL_MAIN_TOPIC:
        return SupersedeDecision("new", reasoning="catch-all topic, never matched")
    return _resolve(
        new_values=merged_topic,
        identity_key_tuples=_MERGED_IDENTITY_KEY_TUPLES,
        comparison_keys=_MERGED_COMPARISON_KEYS,
        diff_keys=_MERGED_DIFF_KEYS,
        # 2026-10 fix (PDF2-C1): scoped to ingested_via="pdf_auto" so this
        # can never match/retire a Sheet-sourced doc that happens to share
        # an identity signal — Sheet-sourced docs use the same data_type/
        # main_topic/sub_topic shape, so without this a PDF re-upload could
        # get judged "update" and archive a real Sheet row. This resolver
        # must only ever retire docs this same pdf_auto pipeline wrote.
        data_type_filter={"$and": [
            {"data_type": {"$in": ["regulatory", "business_guide", "marketing"]}},
            {"ingested_via": "pdf_auto"},
        ]},
        source_file=source_file,
        cost_accumulator=cost_accumulator,
        # 2026-10 fix (PDF2-C2): build_merged_documents now writes real
        # summary/category metadata keys directly on every new document,
        # but pre-merge live docs (ingested under the old know_how-only
        # path) only ever got them under answer_guideline/content_type —
        # kept for backward-compatible reads of those older docs until the
        # corpus fully turns over.
        metadata_key_aliases=_KNOWHOW_METADATA_KEY_ALIASES,
    )



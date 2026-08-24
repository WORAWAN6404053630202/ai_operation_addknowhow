# code/service/pdf_candidate_matching.py
"""Finds existing Sheet rows that might be the same regulatory topic as a
newly-drafted PDF review item, and checks whether the new item's topic fits
any category the bot already knows about (decision_type="new_category"
support signal) — feature/pdf-ingestion.

Read-only against the Sheet: never writes, never auto-decides a decision_type
— purely an aid surfaced in the admin UI. The reviewer always clicks the
final button themselves.

DESIGN (revised 2026-08-24, replacing the original top-3-embedding-only
version): candidate-finding now unions 3 independent signals instead of
relying on embedding similarity alone —

  1. embedding similarity — wide semantic net, dynamic cutoff (a real gap in
     the sorted score distribution) instead of a fixed top-K, so a document
     with 7 genuinely-similar rows shows 7, and one with none shows none.
  2. department+license_type fuzzy string match — independent of embeddings
     entirely, catches same-topic-different-wording cases where embeddings
     under-score (Thai regulatory text reuses so much boilerplate bureaucratic
     phrasing that embedding similarity is not fully trustworthy alone).
  3. LLM batch-scan — the regulatory Sheet is small enough right now (~178
     rows as of 2026-08-24) that a cheap LLM can just read every row in
     batches and judge topic-identity directly, which catches what both
     literal string matching AND embeddings can miss (paraphrased content).
     Revisit if the Sheet grows into the thousands — batch-scanning cost
     scales linearly with row count, unlike embedding search.

Signals are UNIONed (a row shown by any one method is included), not
intersected — biased toward recall (show more candidates) since a cheap LLM
verification-style read is exactly what a human reviewer does anyway when
looking at the list; the cost of a false-positive candidate is a reviewer
spending 2 seconds ruling it out, but the cost of a missed real duplicate is
bad data reaching the live bot.

Deliberately does NOT reuse the main app's Chroma vector store: that data is
baked into the production Docker image at build time (see Dockerfile's `COPY
local_chroma_v3/`), physically inside the running bot container — not
reachable from this isolated EC2 process. Comparing directly against the live
Sheet (already reachable here via sheet_write_back's gspread connection) is
both simpler and always current."""

from __future__ import annotations

import difflib
import json
import math
import re
from typing import Any, Optional

from openai import OpenAI

import conf
from model.pdf_review_item import ReviewItem
from service.pdf_field_drafting import DRAFTABLE_FIELDS
from service.sheet_write_back import _clean_header, _get_worksheet
from utils.logger import get_logger

logger = get_logger(__name__)

# Fields that identify WHAT the row is about — used for embedding text and
# for the LLM batch-scan's topic-identity judgment.
_COMPARISON_FIELD_KEYS = ["department", "license_type", "operation_topic", "answer_guideline"]
# Narrower subset for the cheap independent fuzzy-string safety net — only the
# two fields that are closer to categorical vocabulary than free text.
_IDENTITY_FIELD_KEYS = ["department", "license_type"]
# Everything else drafted — these are what a genuine "update" would actually
# change, so this is what gets diffed once a candidate is found.
_PROCEDURAL_FIELD_KEYS = [
    "registration_type", "terms_and_conditions", "service_channel", "operation_steps",
    "identification_documents", "operation_duration", "fees", "legal_regulatory",
]
# Every field the LLM batch-scan / category-fit check might read or diff.
_ALL_FIELD_KEYS = list(DRAFTABLE_FIELDS.keys())

_EMBEDDING_BATCH_SIZE = 100
_EMBEDDING_FLOOR = 0.45  # below this, not worth surfacing at all — pure noise
_EMBEDDING_MAX_CANDIDATES = 15  # sanity cap even when no clear gap is found
_EMBEDDING_MIN_GAP = 0.05  # a gap smaller than this isn't trustworthy as a cutoff

_IDENTITY_FUZZY_THRESHOLD = 0.82  # normalized SequenceMatcher ratio

_LLM_SCAN_BATCH_SIZE = 18  # rows per LLM call — keeps each call's output small/fast


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip().lower()


def _fuzzy_ratio(a: str, b: str) -> float:
    a, b = _normalize(a), _normalize(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _comparison_text(values: dict[str, str], keys: list[str]) -> str:
    parts = [values[key] for key in keys if values.get(key, "").strip()]
    return " | ".join(parts)


def _new_item_field_values(item: ReviewItem, keys: list[str]) -> dict[str, str]:
    fields = item.llm_drafted_fields or {}
    return {key: fields.get(DRAFTABLE_FIELDS[key][0], "") for key in keys}


def _col_index_by_key(header_row: list[str]) -> dict[str, int]:
    cleaned_headers = [_clean_header(h) for h in header_row]
    result: dict[str, int] = {}
    for key in _ALL_FIELD_KEYS:
        for idx, h in enumerate(cleaned_headers):
            if h in DRAFTABLE_FIELDS[key]:
                result[key] = idx
                break
    return result


def _row_field_values(row: list[str], col_index: dict[str, int]) -> dict[str, str]:
    return {key: (row[idx] if idx < len(row) else "") for key, idx in col_index.items()}


def _fetch_existing_rows() -> tuple[dict[str, int], list[tuple[int, dict[str, str]]]]:
    """One Sheet read, shared by every signal below and by the ④ category-fit
    check — avoids hitting the Sheets API 4 separate times per upload."""
    worksheet = _get_worksheet()
    all_values = worksheet.get_all_values()
    if len(all_values) < 2:
        return {}, []
    col_index = _col_index_by_key(all_values[0])
    rows = [
        (row_number, _row_field_values(row, col_index))
        for row_number, row in enumerate(all_values[1:], start=2)
    ]
    return col_index, rows


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Same OpenRouter embedding endpoint local_vector_store.py uses (conf.EMBEDDING_MODEL),
    but via the plain openai SDK instead of langchain_openai — it sends `input` as
    plain strings with no client-side tiktoken pre-encoding, so it doesn't need that
    module's check_embedding_ctx_length=False workaround for the same 422 bug."""
    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + _EMBEDDING_BATCH_SIZE]
        resp = client.embeddings.create(model=conf.EMBEDDING_MODEL, input=batch)
        embeddings.extend([d.embedding for d in resp.data])
    return embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _dynamic_cutoff_count(sorted_scores: list[float]) -> int:
    """How many of the descending-sorted scores to keep — the largest gap
    between consecutive scores above the floor, not a fixed top-K. Falls back
    to "everything above the floor, capped" when no gap is clearly bigger than
    the others (a smoothly-declining curve has no real cutoff point to find)."""
    above_floor = [s for s in sorted_scores if s >= _EMBEDDING_FLOOR]
    if len(above_floor) <= 1:
        return len(above_floor)
    gaps = [(above_floor[i] - above_floor[i + 1], i) for i in range(len(above_floor) - 1)]
    best_gap, best_idx = max(gaps)
    if best_gap >= _EMBEDDING_MIN_GAP:
        return min(best_idx + 1, _EMBEDDING_MAX_CANDIDATES)
    return min(len(above_floor), _EMBEDDING_MAX_CANDIDATES)


def _embedding_candidates(new_values: dict[str, str], rows: list[tuple[int, dict[str, str]]]) -> dict[int, float]:
    """Returns {row_number: similarity} for rows the embedding signal flags."""
    new_text = _comparison_text(new_values, _COMPARISON_FIELD_KEYS)
    if not new_text.strip() or not rows:
        return {}
    texts_with_rownum = [
        (row_number, _comparison_text(values, _COMPARISON_FIELD_KEYS)) for row_number, values in rows
    ]
    texts_with_rownum = [(rn, t) for rn, t in texts_with_rownum if t.strip()]
    if not texts_with_rownum:
        return {}

    embeddings = _embed_texts([new_text] + [t for _, t in texts_with_rownum])
    new_embedding, row_embeddings = embeddings[0], embeddings[1:]

    scored = [
        (round(_cosine_similarity(new_embedding, row_emb), 4), row_number)
        for (row_number, _), row_emb in zip(texts_with_rownum, row_embeddings)
    ]
    scored.sort(reverse=True)
    keep_count = _dynamic_cutoff_count([s for s, _ in scored])
    return {row_number: sim for sim, row_number in scored[:keep_count]}


def _identity_fuzzy_candidates(new_values: dict[str, str], rows: list[tuple[int, dict[str, str]]]) -> set[int]:
    """Independent of embeddings entirely — catches same-topic-different-
    wording cases where the free-text embedding signal under-scores, since
    department/license_type are closer to categorical vocabulary than the
    rest of the drafted fields."""
    flagged: set[int] = set()
    for row_number, values in rows:
        matched_fields = 0
        for key in _IDENTITY_FIELD_KEYS:
            if new_values.get(key, "").strip() and _fuzzy_ratio(new_values[key], values.get(key, "")) >= _IDENTITY_FUZZY_THRESHOLD:
                matched_fields += 1
        if matched_fields == len(_IDENTITY_FIELD_KEYS):
            flagged.add(row_number)
    return flagged


_LLM_SCAN_PROMPT = """คุณกำลังช่วยตรวจสอบว่าเอกสารกฎหมาย/ใบอนุญาตฉบับใหม่ เป็นเรื่องเดียวกัน (หน่วยงาน + ประเภทใบอนุญาต + หัวข้อการดำเนินการตรงกันจริง ไม่ใช่แค่หน่วยงานเดียวกัน) กับแถวข้อมูลเดิมแถวไหนบ้างในรายการต่อไปนี้

เอกสารใหม่:
หน่วยงาน: {new_department}
ประเภทใบอนุญาต: {new_license_type}
หัวข้อการดำเนินการ: {new_operation_topic}
แนวคำตอบ (บางส่วน): {new_answer_guideline}

รายการแถวเดิม:
{rows_block}

ตอบเป็น JSON เท่านั้น ไม่ต้องมีข้อความอื่น:
{{"matches": [{{"row_number": <int>, "reasoning": "<เหตุผลสั้นๆ ไม่เกิน 1 ประโยค>"}}]}}
ถ้าไม่มีแถวไหนตรงเลย ตอบ {{"matches": []}}
"""


def _llm_scan_batch(new_values: dict[str, str], batch: list[tuple[int, dict[str, str]]]) -> list[dict[str, Any]]:
    rows_block = "\n".join(
        f"[{row_number}] หน่วยงาน: {values.get('department', '')} | "
        f"ประเภทใบอนุญาต: {values.get('license_type', '')} | "
        f"หัวข้อ: {values.get('operation_topic', '')}"
        for row_number, values in batch
    )
    prompt = _LLM_SCAN_PROMPT.format(
        new_department=new_values.get("department", ""),
        new_license_type=new_values.get("license_type", ""),
        new_operation_topic=new_values.get("operation_topic", ""),
        new_answer_guideline=(new_values.get("answer_guideline", "") or "")[:300],
        rows_block=rows_block,
    )
    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
    resp = client.chat.completions.create(
        model=conf.OPENROUTER_MODEL_PDF_MATCHING,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        response_format={"type": "json_object"},
        extra_body={"reasoning": {"enabled": False}},
    )
    raw = (resp.choices[0].message.content or "{}").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(raw)
    matches = parsed.get("matches", [])
    return [m for m in matches if isinstance(m, dict) and isinstance(m.get("row_number"), int)]


def _llm_scan_candidates(new_values: dict[str, str], rows: list[tuple[int, dict[str, str]]]) -> dict[int, str]:
    """Returns {row_number: reasoning}. Never raises — a failed batch just
    contributes nothing (the embedding + identity signals above still stand),
    logged so it's visible without blocking the whole review item."""
    results: dict[int, str] = {}
    for i in range(0, len(rows), _LLM_SCAN_BATCH_SIZE):
        batch = rows[i : i + _LLM_SCAN_BATCH_SIZE]
        try:
            matches = _llm_scan_batch(new_values, batch)
            for m in matches:
                results[m["row_number"]] = str(m.get("reasoning", ""))
        except Exception as e:
            logger.error(f"[CandidateMatching] LLM batch-scan failed for rows {batch[0][0]}-{batch[-1][0]}, skipping this batch: {e}")
    return results


def _field_diff(new_values: dict[str, str], existing_values: dict[str, str]) -> list[dict[str, str]]:
    diffs = []
    for key in _PROCEDURAL_FIELD_KEYS:
        new_v = (new_values.get(key) or "").strip()
        old_v = (existing_values.get(key) or "").strip()
        if new_v and old_v and new_v != old_v:
            diffs.append({"field": key, "old_value": old_v, "new_value": new_v})
    return diffs


def find_candidate_matches(item: ReviewItem) -> list[dict[str, Any]]:
    """Returns every existing Sheet row flagged by ANY of the 3 signals
    (embedding similarity / identity fuzzy match / LLM batch-scan), each with
    a found_by list and a field-by-field diff against the new item's drafted
    values. No fixed top-N — see module docstring. Returns [] (never raises)
    if there's nothing to compare against or the new item has no drafted
    text — this is an aid, not a required step, and must not block saving
    the review item."""
    new_values = _new_item_field_values(item, _ALL_FIELD_KEYS)
    if not _comparison_text(new_values, _COMPARISON_FIELD_KEYS).strip():
        logger.warning(f"[CandidateMatching] {item.filename}: no comparable drafted text, skipping")
        return []

    try:
        _, rows = _fetch_existing_rows()
    except Exception as e:
        logger.error(f"[CandidateMatching] Failed to fetch existing Sheet rows for {item.filename}: {e}")
        return []
    if not rows:
        return []

    embedding_scores: dict[int, float] = {}
    try:
        embedding_scores = _embedding_candidates(new_values, rows)
    except Exception as e:
        logger.error(f"[CandidateMatching] Embedding signal failed for {item.filename}, continuing with other signals: {e}")

    identity_hits = _identity_fuzzy_candidates(new_values, rows)
    llm_hits = _llm_scan_candidates(new_values, rows)

    all_row_numbers = set(embedding_scores) | identity_hits | set(llm_hits)
    if not all_row_numbers:
        logger.info(f"[CandidateMatching] {item.filename}: no candidates from any signal across {len(rows)} existing rows")
        return []

    rows_by_number = dict(rows)
    candidates = []
    for row_number in all_row_numbers:
        existing_values = rows_by_number.get(row_number, {})
        found_by = []
        if row_number in embedding_scores:
            found_by.append("embedding")
        if row_number in identity_hits:
            found_by.append("identity_match")
        if row_number in llm_hits:
            found_by.append("llm_scan")
        candidates.append({
            "row_number": row_number,
            "found_by": found_by,
            "similarity": embedding_scores.get(row_number),
            "llm_reasoning": llm_hits.get(row_number),
            "department": existing_values.get("department", ""),
            "license_type": existing_values.get("license_type", ""),
            "operation_topic": existing_values.get("operation_topic", ""),
            "field_diffs": _field_diff(new_values, existing_values),
        })

    # Strongest evidence first: multi-signal agreement, then similarity.
    candidates.sort(key=lambda c: (len(c["found_by"]), c["similarity"] or 0), reverse=True)

    logger.info(
        f"[CandidateMatching] {item.filename}: {len(candidates)} candidate(s) across {len(rows)} existing rows "
        f"(embedding={len(embedding_scores)}, identity={len(identity_hits)}, llm_scan={len(llm_hits)})"
    )
    return candidates


_CATEGORY_FIT_PROMPT = """คุณกำลังช่วยตรวจสอบว่าเอกสารใหม่นี้ เข้าข่ายหมวดหมู่ (operation_topic) ที่มีอยู่แล้วในระบบหรือไม่

เอกสารใหม่:
หน่วยงาน: {new_department}
ประเภทใบอนุญาต: {new_license_type}
หัวข้อการดำเนินการ: {new_operation_topic}

รายชื่อหมวดหมู่ที่มีอยู่แล้วในระบบ:
{topics_block}

พิจารณาว่าเอกสารใหม่นี้เข้าข่ายหมวดใดหมวดหนึ่งในรายชื่อข้างต้นหรือไม่ (ไม่ต้องตรงคำเป๊ะๆ แค่เป็นเรื่องเดียวกันในเชิงความหมายก็นับ) หรือเป็นหมวดใหม่ที่ไม่มีในรายชื่อเลย

ตอบเป็น JSON เท่านั้น:
{{"fits_known_category": true/false, "matched_topic": "<ชื่อหมวดที่ตรง หรือ null>", "reasoning": "<เหตุผลสั้นๆ ไม่เกิน 1 ประโยค>"}}
"""


def check_category_fit(item: ReviewItem) -> Optional[dict[str, Any]]:
    """decision_type == 'new_category' support signal — a suggestion only,
    never auto-applied. Compares the new item's operation_topic against every
    distinct operation_topic already in the Sheet (the same live data the
    bot's own topic routing is built from), via LLM since 'fits an existing
    category conceptually' is a fuzzier judgment than exact/fuzzy string
    matching can reliably make. Returns None (not an error state — just 'no
    signal available') if there's nothing to compare against or the call
    fails; never raises."""
    new_values = _new_item_field_values(item, ["department", "license_type", "operation_topic"])
    if not new_values.get("operation_topic", "").strip():
        return None

    try:
        _, rows = _fetch_existing_rows()
    except Exception as e:
        logger.error(f"[CategoryFit] Failed to fetch existing Sheet rows for {item.filename}: {e}")
        return None
    if not rows:
        return None

    known_topics = sorted({values.get("operation_topic", "").strip() for _, values in rows if values.get("operation_topic", "").strip()})
    if not known_topics:
        return None

    prompt = _CATEGORY_FIT_PROMPT.format(
        new_department=new_values.get("department", ""),
        new_license_type=new_values.get("license_type", ""),
        new_operation_topic=new_values.get("operation_topic", ""),
        topics_block="\n".join(f"- {t}" for t in known_topics),
    )
    try:
        client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
        resp = client.chat.completions.create(
            model=conf.OPENROUTER_MODEL_PDF_MATCHING,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            response_format={"type": "json_object"},
            extra_body={"reasoning": {"enabled": False}},
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        result = {
            "fits_known_category": bool(parsed.get("fits_known_category", True)),
            "matched_topic": parsed.get("matched_topic"),
            "reasoning": str(parsed.get("reasoning", "")),
        }
        logger.info(f"[CategoryFit] {item.filename}: fits_known_category={result['fits_known_category']} matched={result['matched_topic']!r}")
        return result
    except Exception as e:
        logger.error(f"[CategoryFit] Failed for {item.filename}, no suggestion surfaced: {e}")
        return None

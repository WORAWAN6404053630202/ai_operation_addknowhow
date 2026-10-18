# code/service/pdf_knowhow_candidate_matching.py
"""Finds existing know-how Sheet rows that might be the same topic as a
newly-drafted PDF know-how topic (feature/pdf-ingestion, 2026-09).

Mirrors pdf_candidate_matching.py's design (3-signal union: embedding
similarity / identity fuzzy match / LLM batch-scan) but adapted for the
know-how tabs' schema (main_topic/sub_topic/category/summary/full_text,
routed across 2 possible tabs by source_type) instead of the regulatory
Sheet's (department/license_type/operation_topic/answer_guideline).

WHY THIS EXISTS: admin_pdf.py's review flow always submitted decision_type
="new" for know-how items — there was no equivalent to the regulatory Sheet's
candidate-matching aid, so a reviewer had no way to know "this document
covers the same ground as an existing row" short of manually reading through
hundreds of existing know-how rows themselves. Found during a 2026-09
production-readiness review (the SAME session that also found and fixed the
regulatory-path superseded-row read-through gap — see data_loader.py /
data_loader_general.py) — this was the other half of "duplicate/near-
duplicate detection ก่อน insert" that only existed for one of the 2
content-shape paths.

Read-only against the Sheet: never writes, never auto-decides a
decision_type — purely an aid surfaced in the admin UI, same as the
regulatory version. The reviewer always clicks the final button themselves.

Deliberately searches WITHIN the same tab the new topic is routed to (see
knowhow_write_back.py's _TAB_BY_SOURCE_TYPE) — a book-chapter topic and a
general business-advice tip are different domains even if their wording
happens to overlap, so cross-tab matches would be noise, not signal."""

from __future__ import annotations

import json
import math
import time
from typing import Any, Optional

from openai import OpenAI

import conf
from service.knowhow_write_back import _HEADER_VARIANTS, _TAB_BY_SOURCE_TYPE, _get_spreadsheet
from service.sheet_write_back import _clean_header
from utils.llm_cost_logging import CostAccumulator, log_llm_cost
from utils.logger import get_logger
from utils.page_ranges import fuzzy_ratio
from utils.prompt_safety import INJECTION_GUARD

logger = get_logger(__name__)

# Fields that identify WHAT the topic is about — used for embedding text and
# the LLM batch-scan's topic-identity judgment. summary (not full_text) is
# used deliberately: knowhow_write_back.py's own docstring already
# establishes สรุปสั้น/summary as "what a retrieval/search pass matches
# against (short, cheap to embed)" — reusing that design choice here rather
# than embedding full_text (long, expensive, and noisier for identity
# matching than a purpose-written short summary).
_COMPARISON_FIELD_KEYS = ["main_topic", "sub_topic", "summary", "category"]
# Narrower subset for the cheap independent fuzzy-string safety net — the
# categorical-vocabulary fields, mirroring department+license_type's role in
# the regulatory version.
_IDENTITY_FIELD_KEYS = ["main_topic", "sub_topic"]

_EMBEDDING_BATCH_SIZE = 100
# Reuses the regulatory version's calibrated floor/gap constants (same
# embedding model, same language/domain — Thai business/regulatory text) —
# see pdf_candidate_matching.py's live-measured noise-ceiling comment for
# the reasoning behind these exact numbers. Not re-calibrated separately for
# know-how content; revisit if this tab's false-positive rate in practice
# turns out to differ meaningfully from the regulatory tab's.
_EMBEDDING_FLOOR = 0.70
_EMBEDDING_MAX_CANDIDATES = 10
_EMBEDDING_MIN_GAP = 0.05

_IDENTITY_FUZZY_THRESHOLD = 0.82

_LLM_SCAN_BATCH_SIZE = 18


def _comparison_text(values: dict[str, str], keys: list[str]) -> str:
    parts = [values[key] for key in keys if values.get(key, "").strip()]
    return " | ".join(parts)


def _col_index_by_key(header_row: list[str]) -> dict[str, int]:
    cleaned_headers = [_clean_header(h) for h in header_row]
    result: dict[str, int] = {}
    for key, variants in _HEADER_VARIANTS.items():
        for idx, h in enumerate(cleaned_headers):
            if h in variants:
                result[key] = idx
                break
    return result


def _row_field_values(row: list[str], col_index: dict[str, int]) -> dict[str, str]:
    return {key: (row[idx] if idx < len(row) else "") for key, idx in col_index.items()}


def _fetch_existing_rows(tab_name: str) -> list[tuple[int, dict[str, str]]]:
    """One Sheet read per tab — shared by every signal below. Skips rows
    already marked superseded (see data_loader.py's matching filter) so a
    reviewer is never shown "this looks like a duplicate of row X" for a row
    that the vector store itself no longer ingests — that suggestion would be
    actively misleading."""
    spreadsheet = _get_spreadsheet()
    worksheet = spreadsheet.worksheet(tab_name)
    all_values = worksheet.get_all_values()
    if len(all_values) < 2:
        return []
    col_index = _col_index_by_key(all_values[0])
    superseded_col = next(
        (idx for idx, h in enumerate(all_values[0]) if _clean_header(h) == "superseded_note"), None
    )
    rows = []
    for row_number, row in enumerate(all_values[1:], start=2):
        if superseded_col is not None and superseded_col < len(row) and row[superseded_col].strip():
            continue
        rows.append((row_number, _row_field_values(row, col_index)))
    return rows


def _embed_texts(texts: list[str], cost_accumulator: Optional[CostAccumulator] = None) -> list[list[float]]:
    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + _EMBEDDING_BATCH_SIZE]
        _call_start = time.monotonic()
        resp = client.embeddings.create(model=conf.EMBEDDING_MODEL, input=batch)
        log_llm_cost(
            logger, f"KnowhowCandidateMatching/Embed[{len(batch)} texts]", conf.EMBEDDING_MODEL, resp,
            time.monotonic() - _call_start, accumulator=cost_accumulator,
        )
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
    above_floor = [s for s in sorted_scores if s >= _EMBEDDING_FLOOR]
    if len(above_floor) <= 1:
        return len(above_floor)
    gaps = [(above_floor[i] - above_floor[i + 1], i) for i in range(len(above_floor) - 1)]
    best_gap, best_idx = max(gaps)
    if best_gap >= _EMBEDDING_MIN_GAP:
        return min(best_idx + 1, _EMBEDDING_MAX_CANDIDATES)
    return min(len(above_floor), _EMBEDDING_MAX_CANDIDATES)


def _embedding_candidates(
    new_values: dict[str, str], rows: list[tuple[int, dict[str, str]]],
    cost_accumulator: Optional[CostAccumulator] = None,
) -> dict[int, float]:
    new_text = _comparison_text(new_values, _COMPARISON_FIELD_KEYS)
    if not new_text.strip() or not rows:
        return {}
    texts_with_rownum = [
        (row_number, _comparison_text(values, _COMPARISON_FIELD_KEYS)) for row_number, values in rows
    ]
    texts_with_rownum = [(rn, t) for rn, t in texts_with_rownum if t.strip()]
    if not texts_with_rownum:
        return {}

    embeddings = _embed_texts([new_text] + [t for _, t in texts_with_rownum], cost_accumulator=cost_accumulator)
    new_embedding, row_embeddings = embeddings[0], embeddings[1:]

    scored = [
        (round(_cosine_similarity(new_embedding, row_emb), 4), row_number)
        for (row_number, _), row_emb in zip(texts_with_rownum, row_embeddings)
    ]
    scored.sort(reverse=True)
    keep_count = _dynamic_cutoff_count([s for s, _ in scored])
    return {row_number: sim for sim, row_number in scored[:keep_count]}


def _identity_fuzzy_candidates(new_values: dict[str, str], rows: list[tuple[int, dict[str, str]]]) -> set[int]:
    flagged: set[int] = set()
    for row_number, values in rows:
        matched_fields = 0
        for key in _IDENTITY_FIELD_KEYS:
            if new_values.get(key, "").strip() and fuzzy_ratio(new_values[key], values.get(key, "")) >= _IDENTITY_FUZZY_THRESHOLD:
                matched_fields += 1
        if matched_fields == len(_IDENTITY_FIELD_KEYS):
            flagged.add(row_number)
    return flagged


_LLM_SCAN_PROMPT = """คุณกำลังช่วยตรวจสอบว่าหัวข้อความรู้ใหม่ เป็นเรื่องเดียวกัน (หัวข้อหลัก + หัวข้อย่อย ตรงกันจริงในเชิงเนื้อหา ไม่ใช่แค่หมวดเดียวกัน) กับแถวข้อมูลเดิมแถวไหนบ้างในรายการต่อไปนี้

""" + INJECTION_GUARD + """

หัวข้อใหม่:
หัวข้อหลัก: {new_main_topic}
หัวข้อย่อย: {new_sub_topic}
สรุปสั้น: {new_summary}

รายการแถวเดิม:
{rows_block}

ตอบเป็น JSON เท่านั้น ไม่ต้องมีข้อความอื่น:
{{"matches": [{{"row_number": <int>, "reasoning": "<เหตุผลสั้นๆ ไม่เกิน 1 ประโยค>"}}]}}
ถ้าไม่มีแถวไหนตรงเลย ตอบ {{"matches": []}}
"""


def _llm_scan_batch(
    new_values: dict[str, str], batch: list[tuple[int, dict[str, str]]], cost_accumulator: Optional[CostAccumulator] = None,
) -> list[dict[str, Any]]:
    rows_block = "\n".join(
        f"[{row_number}] หัวข้อหลัก: {values.get('main_topic', '')} | "
        f"หัวข้อย่อย: {values.get('sub_topic', '')} | "
        f"สรุปสั้น: {(values.get('summary', '') or '')[:150]}"
        for row_number, values in batch
    )
    prompt = _LLM_SCAN_PROMPT.format(
        new_main_topic=new_values.get("main_topic", ""),
        new_sub_topic=new_values.get("sub_topic", ""),
        new_summary=(new_values.get("summary", "") or "")[:300],
        rows_block=rows_block,
    )
    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
    _call_start = time.monotonic()
    resp = client.chat.completions.create(
        model=conf.OPENROUTER_MODEL_PDF_MATCHING,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        response_format={"type": "json_object"},
        extra_body={"reasoning": {"enabled": False}},
    )
    log_llm_cost(logger, "KnowhowCandidateMatching/LLMScan", conf.OPENROUTER_MODEL_PDF_MATCHING, resp, time.monotonic() - _call_start, accumulator=cost_accumulator)
    raw = (resp.choices[0].message.content or "{}").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(raw)
    matches = parsed.get("matches", [])
    return [m for m in matches if isinstance(m, dict) and isinstance(m.get("row_number"), int)]


def _llm_scan_candidates(
    new_values: dict[str, str], rows: list[tuple[int, dict[str, str]]], cost_accumulator: Optional[CostAccumulator] = None,
) -> dict[int, str]:
    results: dict[int, str] = {}
    for i in range(0, len(rows), _LLM_SCAN_BATCH_SIZE):
        batch = rows[i : i + _LLM_SCAN_BATCH_SIZE]
        try:
            matches = _llm_scan_batch(new_values, batch, cost_accumulator=cost_accumulator)
            for m in matches:
                results[m["row_number"]] = str(m.get("reasoning", ""))
        except Exception as e:
            logger.error(f"[KnowhowCandidateMatching] LLM batch-scan failed for rows {batch[0][0]}-{batch[-1][0]}, skipping this batch: {e}")
    return results


def find_knowhow_candidate_matches(
    topic: dict[str, Any], cost_accumulator: Optional[CostAccumulator] = None,
) -> list[dict[str, Any]]:
    """Returns every existing row (in the SAME tab this topic would route to)
    flagged by any of the 3 signals, each tagged with found_by + a simple
    full_text-changed flag. topic needs: source_type, main_topic, sub_topic,
    summary, category, full_text (same shape identify_knowhow_topics()
    produces, before knowhow_write_back.py's S3-overflow handling runs).

    Returns [] (never raises) if there's nothing to compare against or the
    topic has no comparable drafted text — this is an aid, not a required
    step, and must not block saving the review item, same contract as
    pdf_candidate_matching.py's find_candidate_matches()."""
    new_values = {
        "main_topic": topic.get("main_topic", ""), "sub_topic": topic.get("sub_topic", ""),
        "summary": topic.get("summary", ""), "category": topic.get("category", ""),
    }
    if not _comparison_text(new_values, _COMPARISON_FIELD_KEYS).strip():
        logger.warning(f"[KnowhowCandidateMatching] {topic.get('sub_topic', '?')!r}: no comparable drafted text, skipping")
        return []

    tab_name = _TAB_BY_SOURCE_TYPE.get(topic.get("source_type"), _TAB_BY_SOURCE_TYPE["document"])
    try:
        rows = _fetch_existing_rows(tab_name)
    except Exception as e:
        logger.error(f"[KnowhowCandidateMatching] Failed to fetch existing rows from {tab_name!r}: {e}")
        return []
    if not rows:
        return []

    embedding_scores: dict[int, float] = {}
    try:
        embedding_scores = _embedding_candidates(new_values, rows, cost_accumulator=cost_accumulator)
    except Exception as e:
        logger.error(f"[KnowhowCandidateMatching] Embedding signal failed, continuing with other signals: {e}")

    identity_hits = _identity_fuzzy_candidates(new_values, rows)
    llm_hits = _llm_scan_candidates(new_values, rows, cost_accumulator=cost_accumulator)

    all_row_numbers = set(embedding_scores) | identity_hits | set(llm_hits)
    if not all_row_numbers:
        logger.info(f"[KnowhowCandidateMatching] {topic.get('sub_topic', '?')!r}: no candidates across {len(rows)} rows in {tab_name!r}")
        return []

    rows_by_number = dict(rows)
    new_full_text = (topic.get("full_text") or "").strip()
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
        old_full_text = (existing_values.get("full_text") or "").strip()
        candidates.append({
            "row_number": row_number,
            "tab_name": tab_name,
            "found_by": found_by,
            "similarity": embedding_scores.get(row_number),
            "llm_reasoning": llm_hits.get(row_number),
            "main_topic": existing_values.get("main_topic", ""),
            "sub_topic": existing_values.get("sub_topic", ""),
            # full_text on the existing row may be an s3:// pointer (see
            # knowhow_write_back.py's overflow handling) rather than literal
            # text — content_changed still means something useful in that
            # case (pointer string differs from new plain text = "different"),
            # just not a meaningful diff to show verbatim, so no full diff
            # payload here unlike the regulatory version's per-field diffs.
            "content_changed": bool(new_full_text and old_full_text and new_full_text != old_full_text),
        })

    candidates.sort(key=lambda c: (len(c["found_by"]), c["similarity"] or 0), reverse=True)

    logger.info(
        f"[KnowhowCandidateMatching] {topic.get('sub_topic', '?')!r}: {len(candidates)} candidate(s) in {tab_name!r} "
        f"(embedding={len(embedding_scores)}, identity={len(identity_hits)}, llm_scan={len(llm_hits)})"
    )
    return candidates

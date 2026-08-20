# code/service/pdf_candidate_matching.py
"""Embedding-based duplicate/near-duplicate detection for the PDF review queue
(feature/pdf-ingestion) — surfaces the top-N existing Sheet rows semantically
closest to a newly-drafted PDF review item, so a reviewer doesn't have to
manually scan the whole Sheet to judge decision_type (new/duplicate/update).

Read-only against the Sheet: never writes, never auto-decides anything — this
is purely an aid surfaced in the admin UI, per ReviewItem.candidate_matches'
original design comment ("embedding-based candidate matching against the
existing Sheet data (top-3 similar rows ... per design)").

Comparison text is built from the same fields pdf_field_drafting.py drafts
(department, license_type, operation_topic, answer_guideline) — the fields
most indicative of "is this the same regulatory topic", not the procedural
ones (steps/fees/docs) that read near-identically across unrelated licenses
and would just add noise to the similarity signal.

Deliberately does NOT reuse the main app's Chroma vector store: that data is
baked into the production Docker image at build time (see Dockerfile's `COPY
local_chroma_v3/`), physically inside the running bot container — not
reachable from this isolated EC2 process, and comparing against it would risk
staleness vs. the Sheet anyway. Comparing directly against the live Sheet
(already reachable here via sheet_write_back's gspread connection) is both
simpler and always current."""

from __future__ import annotations

import math
from typing import Any

from openai import OpenAI

import conf
from model.pdf_review_item import ReviewItem
from service.pdf_field_drafting import DRAFTABLE_FIELDS
from service.sheet_write_back import _clean_header, _get_worksheet
from utils.logger import get_logger

logger = get_logger(__name__)

# department + license_type + operation_topic + answer_guideline: what the row
# is ABOUT, not how-to-do-it detail that's similar across unrelated licenses.
_COMPARISON_FIELD_KEYS = ["department", "license_type", "operation_topic", "answer_guideline"]
_EMBEDDING_BATCH_SIZE = 100
_DEFAULT_TOP_K = 3


def _comparison_text(values: dict[str, str]) -> str:
    parts = [values[key] for key in _COMPARISON_FIELD_KEYS if values.get(key, "").strip()]
    return " | ".join(parts)


def _new_item_field_values(item: ReviewItem) -> dict[str, str]:
    fields = item.llm_drafted_fields or {}
    return {key: fields.get(DRAFTABLE_FIELDS[key][0], "") for key in _COMPARISON_FIELD_KEYS}


def _col_index_by_key(header_row: list[str]) -> dict[str, int]:
    cleaned_headers = [_clean_header(h) for h in header_row]
    result: dict[str, int] = {}
    for key in _COMPARISON_FIELD_KEYS:
        for idx, h in enumerate(cleaned_headers):
            if h in DRAFTABLE_FIELDS[key]:
                result[key] = idx
                break
    return result


def _row_field_values(row: list[str], col_index: dict[str, int]) -> dict[str, str]:
    return {key: (row[idx] if idx < len(row) else "") for key, idx in col_index.items()}


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


def find_candidate_matches(item: ReviewItem, top_k: int = _DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Returns up to top_k existing Sheet rows most similar to item's drafted
    content, sorted descending by cosine similarity. Each match is
    {row_number, similarity, department, license_type, operation_topic} —
    enough for a reviewer to eyeball relevance without opening the Sheet.
    Returns [] (never raises) if there's nothing to compare against — this is
    an aid, not a required step, and must not block saving the review item."""
    new_values = _new_item_field_values(item)
    new_text = _comparison_text(new_values)
    if not new_text.strip():
        logger.warning(f"[CandidateMatching] {item.filename}: no comparable drafted text, skipping")
        return []

    worksheet = _get_worksheet()
    all_values = worksheet.get_all_values()
    if len(all_values) < 2:
        return []

    col_index = _col_index_by_key(all_values[0])
    rows: list[tuple[int, str, dict[str, str]]] = []
    for row_number, row in enumerate(all_values[1:], start=2):
        values = _row_field_values(row, col_index)
        text = _comparison_text(values)
        if text.strip():
            rows.append((row_number, text, values))

    if not rows:
        return []

    embeddings = _embed_texts([new_text] + [r[1] for r in rows])
    new_embedding, row_embeddings = embeddings[0], embeddings[1:]

    scored = [
        {
            "row_number": row_number,
            "similarity": round(_cosine_similarity(new_embedding, row_emb), 4),
            "department": values.get("department", ""),
            "license_type": values.get("license_type", ""),
            "operation_topic": values.get("operation_topic", ""),
        }
        for (row_number, _, values), row_emb in zip(rows, row_embeddings)
    ]
    scored.sort(key=lambda m: m["similarity"], reverse=True)
    top_matches = scored[:top_k]

    best = f"{top_matches[0]['similarity']:.3f}" if top_matches else "n/a"
    logger.info(f"[CandidateMatching] {item.filename}: best={best} across {len(rows)} existing rows")
    return top_matches

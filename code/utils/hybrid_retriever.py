"""
Hybrid RAG: BM25 (sparse) + Dense (embedding) fused with Reciprocal Rank Fusion (RRF).

Usage:
    from utils.hybrid_retriever import hybrid_scored_search, invalidate_bm25_cache

    docs = hybrid_scored_search(vectorstore, query, k=20)

Enable via conf: HYBRID_SEARCH_ENABLED=true
Thai tokenization requires: pythainlp, rank_bm25
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

_LOG = logging.getLogger(__name__)

_INDEX_LOCK = threading.Lock()
_INDEX_CACHE: Dict[str, "_BM25Index"] = {}  # collection_name → BM25Index
_COUNT_CACHE: Dict[str, tuple] = {}  # collection_name → (count, monotonic_timestamp)
_COUNT_TTL_S = 30.0  # refresh coll.count() at most every 30s


# ── Thai tokenizer ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    try:
        from pythainlp.tokenize import word_tokenize
        tokens = word_tokenize(str(text or ""), engine="newmm", keep_whitespace=False)
        return [t for t in tokens if t and t.strip()]
    except Exception as _e:
        _LOG.debug("[BM25/tokenize] PyThaiNLP unavailable (%s) — using whitespace split", type(_e).__name__)
        return str(text or "").split()


# ── BM25 index ────────────────────────────────────────────────────────────────

class _BM25Index:
    def __init__(self, docs: List[Document]):
        from rank_bm25 import BM25Okapi
        self._docs = docs
        self.doc_count = len(docs)
        tokenized = [_tokenize(d.page_content) for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        _LOG.info("[BM25] Index built: %d docs", len(docs))

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(self._docs[i], float(s)) for i, s in ranked[:k] if s > 0]


def _get_bm25_index(vectorstore: Any) -> Optional[_BM25Index]:
    """Lazy-load BM25 index from Chroma collection. Thread-safe singleton.

    Compares Chroma doc count on every call to detect re-ingest from another
    process — if count changed, the cached index is invalidated and rebuilt.
    """
    try:
        coll = vectorstore._collection
        coll_name: str = coll.name
    except Exception as e:
        _LOG.warning("[BM25] Cannot access collection: %s", e)
        return None

    # Cache coll.count() to avoid a SQLite query on every retrieval call.
    # invalidate_bm25_cache() clears both _INDEX_CACHE and _COUNT_CACHE,
    # so re-ingest still triggers an immediate rebuild on the next call.
    _now = time.monotonic()
    _cached_count = _COUNT_CACHE.get(coll_name)
    if _cached_count and (_now - _cached_count[1]) < _COUNT_TTL_S:
        current_count = _cached_count[0]
    else:
        try:
            current_count = coll.count()
            _COUNT_CACHE[coll_name] = (current_count, _now)
        except Exception as e:
            _LOG.warning("[BM25] Cannot access collection: %s", e)
            return None

    cached = _INDEX_CACHE.get(coll_name)
    if cached is not None:
        if cached.doc_count == current_count:
            return cached
        _LOG.info("[BM25] Doc count changed (%d→%d) — rebuilding index for %r",
                  cached.doc_count, current_count, coll_name)

    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(coll_name)
        if cached is not None and cached.doc_count == current_count:
            return cached
        try:
            result = coll.get(include=["documents", "metadatas"])
            raw_docs = result.get("documents") or []
            raw_mds = result.get("metadatas") or []
            docs = [
                Document(page_content=pc or "", metadata=md or {})
                for pc, md in zip(raw_docs, raw_mds)
                if (pc or "").strip()
            ]
            if not docs:
                _LOG.warning("[BM25] Collection %r has no docs — BM25 disabled", coll_name)
                return None
            idx = _BM25Index(docs)
            _INDEX_CACHE[coll_name] = idx
            return idx
        except Exception as e:
            _LOG.warning("[BM25] Index build failed: %s", e)
            return None


def invalidate_bm25_cache(collection_name: Optional[str] = None) -> None:
    """Force BM25 index rebuild on next search (call after re-ingest)."""
    with _INDEX_LOCK:
        if collection_name:
            _INDEX_CACHE.pop(collection_name, None)
            _COUNT_CACHE.pop(collection_name, None)
        else:
            _INDEX_CACHE.clear()
            _COUNT_CACHE.clear()
    _LOG.info("[BM25] Cache invalidated (collection=%r)", collection_name or "all")


# ── Metadata filter evaluation ───────────────────────────────────────────────

def _matches_metadata_filter(doc_metadata: dict, flt: dict) -> bool:
    """Evaluate a Chroma-style metadata filter against doc metadata.
    Supports $or, $and, $in, $eq, and simple key=value pairs."""
    if "$or" in flt:
        return any(_matches_metadata_filter(doc_metadata, sub) for sub in flt["$or"])
    if "$and" in flt:
        return all(_matches_metadata_filter(doc_metadata, sub) for sub in flt["$and"])
    md = doc_metadata or {}
    for fk, fv in flt.items():
        actual = md.get(fk)
        if isinstance(fv, dict):
            if "$in" in fv:
                if actual not in fv["$in"]:
                    return False
            elif "$eq" in fv:
                if actual != fv["$eq"]:
                    return False
            elif "$ne" in fv:
                if actual == fv["$ne"]:
                    return False
            else:
                return False  # unknown operator — safe default: no match
        else:
            if actual != fv:
                return False
    return True


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _doc_id(doc: Document) -> str:
    md = doc.metadata or {}
    return (
        md.get("id")
        or md.get("doc_id")
        or f"{md.get('license_type', '')}|{md.get('data_type', '')}|{doc.page_content[:80]}"
    )


def _rrf_fuse(
    dense_pairs: List[Tuple[Document, float]],
    bm25_pairs: List[Tuple[Document, float]],
    rrf_k: int = 60,
) -> List[Tuple[Document, float]]:
    """score(d) = Σ_i  1 / (rrf_k + rank_i(d))"""
    rrf_scores: Dict[str, float] = {}
    dense_scores: Dict[str, float] = {}  # keep original Dense sim for _sim metadata
    doc_map: Dict[str, Document] = {}

    for rank, (doc, sim) in enumerate(dense_pairs):
        did = _doc_id(doc)
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)
        dense_scores[did] = sim
        doc_map[did] = doc

    for rank, (doc, _score) in enumerate(bm25_pairs):
        did = _doc_id(doc)
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)
        if did not in doc_map:
            # BM25-only doc — flag it so downstream filters don't drop it for low Dense score.
            # Copy to avoid mutating the shared doc object cached in _BM25Index._docs.
            doc_map[did] = Document(page_content=doc.page_content, metadata={**doc.metadata, "_bm25_hit": True})

    merged = sorted(rrf_scores.items(), key=lambda x: -x[1])
    # BM25-only docs get 0.0 dense score (not None) — callers do arithmetic/comparison on score
    # and a None value raises TypeError when used with comparison operators like ">".
    return [(doc_map[did], dense_scores.get(did, 0.0)) for did, _ in merged]


# ── Filter normalization ──────────────────────────────────────────────────────

def _normalize_chroma_filter(flt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Normalize a metadata filter dict for Chroma's Dense search.

    Chroma's similarity_search requires exactly one top-level operator when
    multiple conditions exist. A flat dict with multiple fields raises:
      "Expected where to have exactly one operator, got {field1: v1, field2: v2}"

    Rules applied:
    - Drop fields with None values (Chroma rejects them).
    - Single-field dicts and dicts already using $and/$or are passed through.
    - Flat multi-field dicts are wrapped in {'$and': [{k: v}, ...]}.
    """
    if not flt:
        return flt
    cleaned = {k: v for k, v in flt.items() if v is not None}
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned
    # Multiple fields — Chroma Dense requires explicit $and
    return {"$and": [{k: v} for k, v in cleaned.items()]}


# ── Public API ────────────────────────────────────────────────────────────────

def hybrid_scored_search(
    vectorstore: Any,
    query: str,
    k: int = 20,
    metadata_filter: Optional[Dict[str, Any]] = None,
    rrf_k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Dense + BM25 fused with RRF.
    Returns List[(Document, dense_sim_score)] — same shape as
    similarity_search_with_relevance_scores so callers can attach _sim normally.

    metadata_filter is applied to Dense search (Chroma supports it).
    BM25 searches full corpus then post-filters to keep metadata semantics correct.
    Falls back to Dense-only if BM25 index is unavailable.
    """
    # Normalize filter once: Chroma Dense rejects flat multi-field dicts and None values
    _eff_filter = _normalize_chroma_filter(metadata_filter)

    # Dense search
    dense_pairs: List[Tuple[Document, float]] = []
    try:
        kwargs: Dict[str, Any] = {"k": k}
        if _eff_filter:
            kwargs["filter"] = _eff_filter
        dense_pairs = vectorstore.similarity_search_with_relevance_scores(query, **kwargs)
    except Exception as e:
        _LOG.warning("[Hybrid] Dense search failed: %s", e)

    # BM25 search
    bm25_index = _get_bm25_index(vectorstore)
    if bm25_index is None:
        _LOG.debug("[Hybrid] BM25 unavailable — Dense-only fallback")
        return dense_pairs

    bm25_pairs: List[Tuple[Document, float]] = []
    try:
        # Oversample BM25 before post-filter — use k*5 when a metadata filter is
        # present (filter may be very selective, e.g. entity_type=นิติบุคคล)
        _bm25_oversample = k * 5 if _eff_filter else k * 2
        raw_bm25 = bm25_index.search(query, k=_bm25_oversample)
        if _eff_filter:
            raw_bm25 = [
                (doc, s) for doc, s in raw_bm25
                if _matches_metadata_filter(doc.metadata or {}, _eff_filter)
            ]
        bm25_pairs = raw_bm25[:k]
    except Exception as e:
        _LOG.warning("[Hybrid] BM25 search failed: %s — Dense-only", e)
        return dense_pairs

    fused = _rrf_fuse(dense_pairs, bm25_pairs, rrf_k=rrf_k)
    _LOG.info(
        "[Hybrid] dense=%d bm25=%d → fused=%d (rrf_k=%d filter=%s)",
        len(dense_pairs), len(bm25_pairs), len(fused), rrf_k,
        str(metadata_filter) if metadata_filter else "none",
    )
    return fused[:k]

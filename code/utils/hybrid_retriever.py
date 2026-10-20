"""
Hybrid RAG: BM25 (sparse) + Dense (embedding) fused with Reciprocal Rank Fusion (RRF).

Usage:
    from utils.hybrid_retriever import hybrid_scored_search, invalidate_bm25_cache

    docs = hybrid_scored_search(vectorstore, query, k=20)

Enable via conf: HYBRID_SEARCH_ENABLED=true
Thai tokenization requires: pythainlp, rank_bm25
"""
from __future__ import annotations

import concurrent.futures
import hashlib
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

# Dedicated pool so BM25 (local CPU) can run while Dense (network call + retries)
# is in flight. Separate from persona_supervisor.py's _CLASSIFIER_POOL (LLM calls
# only) to keep pool ownership/capacity semantics independent.
_BM25_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="bm25search")


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
    def __init__(self, docs: List[Document], corpus_version: int = 0):
        from rank_bm25 import BM25Okapi
        self._docs = docs
        self.doc_count = len(docs)
        # 2026-10 fix (core2-C5): stamped with the corpus-version counter
        # (service.local_vector_store.get_corpus_version) that every other
        # corpus-derived cache in the project already checks — see
        # _get_bm25_index below for why doc count alone isn't enough.
        self.corpus_version = corpus_version
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

    2026-10 fix (core2-C5): doc count alone missed the PDF supersede
    resolver's "update" decisions — those archive one doc and add one new
    doc in the same operation, so the live collection's count is unchanged
    even though the text that was indexed is now stale. Every other
    corpus-derived cache in the project (9 of them, across all 3 personas)
    was already wired to service.local_vector_store.get_corpus_version()
    for exactly this reason; BM25 was the one cache this got missed on —
    without it, a retired regulatory doc's text stays searchable (and still
    gets the BM25-hit similarity-floor bypass) for the rest of this
    process's life.
    """
    try:
        coll = vectorstore._collection
        coll_name: str = coll.name
    except Exception as e:
        _LOG.warning("[BM25] Cannot access collection: %s", e)
        return None

    from service.local_vector_store import get_corpus_version
    try:
        current_corpus_version = get_corpus_version()
    except Exception:
        current_corpus_version = 0

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
        if cached.doc_count == current_count and cached.corpus_version == current_corpus_version:
            return cached
        _LOG.info(
            "[BM25] Doc count/corpus-version changed (%d→%d docs, v%s→v%s) — rebuilding index for %r",
            cached.doc_count, current_count, cached.corpus_version, current_corpus_version, coll_name,
        )

    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(coll_name)
        if cached is not None and cached.doc_count == current_count and cached.corpus_version == current_corpus_version:
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
            idx = _BM25Index(docs, corpus_version=current_corpus_version)
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
    """Identify "the same document" across the separately-returned Dense and
    BM25 result lists so _rrf_fuse can combine their scores correctly.

    2026-09 bug fix: the fallback used to be page_content[:80] — collides
    whenever multiple distinct documents share a long common prefix, which
    became common (and reproducibly hit in production) once PDF book content
    started prepending a repeated context header ("หนังสือ: X\\nหมวด: Y\\n
    หัวข้อย่อย: Z...") to every chunk. Confirmed live: 3 different case-study
    chapters sharing one main_topic (Y) produced an IDENTICAL 80-char prefix
    (the differentiating sub_topic, Z, never appeared before the cutoff) —
    _rrf_fuse silently summed their 3 separate RRF contributions into ONE
    entry and kept only the last-seen Document, inflating an unrelated case
    study's score high enough to outrank the genuinely relevant chapter for
    a real user query. The same collision also hits multi-chunk splits of
    one long topic (utils.pdf_text_chunking), since chunk_index isn't part
    of the old key at all.

    Fix: hash the FULL page_content instead of a fixed-length prefix — removes
    any length-dependent collision risk regardless of how future ingestion
    paths shape their content, without requiring every future document
    builder to remember to set an explicit id. Still prefers metadata['id']/
    ['doc_id'] when a caller sets one explicitly (unchanged)."""
    md = doc.metadata or {}
    explicit_id = md.get("id") or md.get("doc_id")
    if explicit_id:
        return str(explicit_id)
    content_hash = hashlib.sha256((doc.page_content or "").encode("utf-8")).hexdigest()[:16]
    return f"{md.get('license_type', '')}|{md.get('data_type', '')}|{content_hash}"


def _rrf_fuse(
    dense_pairs: List[Tuple[Document, float]],
    bm25_pairs: List[Tuple[Document, float]],
    rrf_k: int = 60,
    bm25_bypass_min_ratio: float = 0.35,
) -> List[Tuple[Document, float]]:
    """score(d) = Σ_i  1 / (rrf_k + rank_i(d))

    bm25_bypass_min_ratio (2026-09, see conf.BM25_BYPASS_MIN_RATIO for the full
    measurement): a BM25-only doc (missing from Dense entirely) only gets the
    "_bm25_hit" similarity-floor bypass when its BM25 score is at least this
    fraction of `bm25_pairs`' own top score for this query — a RELATIVE
    threshold, since raw BM25Okapi scores have no shared absolute scale across
    different queries. Docs below the ratio still participate in RRF ranking
    (rrf_scores/merged order unaffected) but are judged by the normal
    similarity floor like any other doc — closes a real asymmetry where a
    doc appearing in BM25's oversampled candidate pool for ANY reason,
    however weak, previously got an unconditional pass while a genuinely
    top-ranked Dense candidate could still be dropped for narrowly missing an
    absolute floor.
    """
    rrf_scores: Dict[str, float] = {}
    dense_scores: Dict[str, float] = {}  # keep original Dense sim for _sim metadata
    doc_map: Dict[str, Document] = {}

    for rank, (doc, sim) in enumerate(dense_pairs):
        did = _doc_id(doc)
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)
        dense_scores[did] = sim
        doc_map[did] = doc

    _top_bm25_score = bm25_pairs[0][1] if bm25_pairs else 0.0
    for rank, (doc, score) in enumerate(bm25_pairs):
        did = _doc_id(doc)
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)
        if did not in doc_map:
            # BM25-only doc. Copy to avoid mutating the shared doc object cached
            # in _BM25Index._docs. Only flag as a bypass-eligible hit when its
            # score is competitive with this query's own top BM25 result —
            # see bm25_bypass_min_ratio docstring above.
            _is_confident_bm25 = _top_bm25_score > 0 and (score / _top_bm25_score) >= bm25_bypass_min_ratio
            _md = {**doc.metadata, "_bm25_hit": True} if _is_confident_bm25 else dict(doc.metadata)
            doc_map[did] = Document(page_content=doc.page_content, metadata=_md)

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
    bm25_bypass_min_ratio: float = 0.35,
) -> List[Tuple[Document, float]]:
    """
    Dense + BM25 fused with RRF.
    Returns List[(Document, dense_sim_score)] — same shape as
    similarity_search_with_relevance_scores so callers can attach _sim normally.

    metadata_filter is applied to Dense search (Chroma supports it).
    BM25 searches full corpus then post-filters to keep metadata semantics correct.
    Falls back to Dense-only if BM25 index is unavailable.

    bm25_bypass_min_ratio: forwarded to _rrf_fuse — see conf.BM25_BYPASS_MIN_RATIO
    and that function's docstring.
    """
    # Normalize filter once: Chroma Dense rejects flat multi-field dicts and None values
    _eff_filter = _normalize_chroma_filter(metadata_filter)

    # BM25 is pure local CPU (index lookup/build + scoring) and doesn't depend on
    # Dense's result — submit it to run concurrently while Dense's network call
    # (and its retries below) is in flight, then join at the end.
    def _run_bm25() -> Optional[List[Tuple[Document, float]]]:
        bm25_index = _get_bm25_index(vectorstore)
        if bm25_index is None:
            _LOG.debug("[Hybrid] BM25 unavailable — Dense-only fallback")
            return None
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
            return raw_bm25[:k]
        except Exception as e:
            _LOG.warning("[Hybrid] BM25 search failed: %s — Dense-only", e)
            return None

    bm25_future = _BM25_POOL.submit(_run_bm25)

    # Dense search — retry on failure. The OpenRouter embedding endpoint (baai/bge-m3)
    # measured ~40-50% transient HTTP 422 ("Input should be a valid string") failure rate
    # in production, independent of query content (confirmed via direct benchmark — even
    # trivial English strings failed at the same rate). This is not a malformed-query bug
    # on our side; it's provider-side flakiness, so a few quick retries recover most calls.
    dense_pairs: List[Tuple[Document, float]] = []
    kwargs: Dict[str, Any] = {"k": k}
    if _eff_filter:
        kwargs["filter"] = _eff_filter
    _dense_max_attempts = 3
    for _attempt in range(1, _dense_max_attempts + 1):
        try:
            dense_pairs = vectorstore.similarity_search_with_relevance_scores(query, **kwargs)
            break
        except Exception as e:
            if _attempt < _dense_max_attempts:
                _LOG.info("[Hybrid] Dense search attempt %d/%d failed: %s — retrying", _attempt, _dense_max_attempts, e)
            else:
                _LOG.warning("[Hybrid] Dense search failed after %d attempts: %s", _dense_max_attempts, e)

    bm25_pairs = bm25_future.result()
    if bm25_pairs is None:
        return dense_pairs

    fused = _rrf_fuse(dense_pairs, bm25_pairs, rrf_k=rrf_k, bm25_bypass_min_ratio=bm25_bypass_min_ratio)
    _LOG.info(
        "[Hybrid] dense=%d bm25=%d → fused=%d (rrf_k=%d filter=%s)",
        len(dense_pairs), len(bm25_pairs), len(fused), rrf_k,
        str(metadata_filter) if metadata_filter else "none",
    )
    return fused[:k]

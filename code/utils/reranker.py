"""
Cross-encoder reranker utility — singleton pattern for shared inference.

Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
  - Multilingual (100+ languages including Thai)
  - ~280MB, 12-layer MiniLM, max_length=512
  - Suitable for query vs. Thai regulatory passage scoring

Usage:
    from utils.reranker import rerank

    docs = rerank(query="จะขอใบอนุญาตเปิดร้านอาหาร", docs=langchain_docs, top_k=6)

Toggle via env: RERANKER_ENABLED=true / RERANKER_MODEL=... / RERANKER_TOP_K=10
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, List, Optional

_LOG = logging.getLogger(__name__)

# Singleton: loaded once on first call, reused across all requests.
_reranker_cache: dict = {}  # model_name → CrossEncoder instance
_reranker_lock = threading.Lock()


def _get_reranker(model_name: str) -> Any:
    """Lazy-load CrossEncoder singleton. Thread-safe via double-checked locking.

    Fast path (model already loaded): no lock acquired — dict lookup only.
    Slow path (first load): lock prevents concurrent threads from loading the
    same model twice, which would waste ~280MB RAM and ~2s at startup.
    """
    if model_name not in _reranker_cache:
        with _reranker_lock:
            if model_name not in _reranker_cache:
                _LOG.info("[Reranker] Loading cross-encoder model: %s", model_name)
                t0 = time.time()
                try:
                    from sentence_transformers import CrossEncoder  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "sentence-transformers is required for reranker. "
                        "Install with: pip install sentence-transformers"
                    ) from e

                _reranker_cache[model_name] = CrossEncoder(model_name, max_length=512)
                _LOG.info("[Reranker] Model loaded in %.1fs", time.time() - t0)
    return _reranker_cache[model_name]


def rerank(
    query: str,
    docs: List[Any],
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    top_k: Optional[int] = None,
) -> List[Any]:
    """
    Rerank LangChain Document objects by cross-encoder relevance score.

    Args:
        query:      User query string (Thai or English).
        docs:       List of LangChain Document objects (must have .page_content).
        model_name: HuggingFace cross-encoder model ID.
        top_k:      If given, return only the top_k highest-scoring docs.

    Returns:
        Reranked docs list (same objects, new order). Stores _rerank_score in metadata.
        Returns original docs unchanged on error.
    """
    if not docs or not query:
        return docs

    try:
        t0 = time.time()
        reranker = _get_reranker(model_name)

        pairs = [
            (query, (getattr(d, "page_content", "") or "")[:1500])
            for d in docs
        ]
        scores: List[float] = reranker.predict(pairs).tolist()

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        # Store rerank score only on docs that are actually returned
        for doc, score in ranked:
            md = getattr(doc, "metadata", None)
            if isinstance(md, dict):
                md["_rerank_score"] = round(float(score), 4)

        elapsed_ms = (time.time() - t0) * 1000
        _LOG.info(
            "[Reranker] reranked %d → %d docs in %.0fms | top_score=%.3f",
            len(docs),
            len(ranked),
            elapsed_ms,
            ranked[0][1] if ranked else 0.0,
        )
        return [d for d, _ in ranked]

    except Exception as exc:
        _LOG.warning("[Reranker] rerank failed (%s) — returning original order", exc)
        return docs

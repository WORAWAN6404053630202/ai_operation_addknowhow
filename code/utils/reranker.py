"""
Cross-encoder reranker utility — singleton pattern for shared inference.

Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
  - ~280MB, 12-layer MiniLM, max_length=512
  - Trained on mMARCO (14 languages: en/zh/fr/de/id/it/pt/ru/es/ja/nl/vi/hi/ar —
    Thai is NOT among them, verified against arXiv:2108.13897). Thai behavior is
    zero-shot cross-lingual transfer from the base encoder, not a fine-tuned/
    benchmarked capability — treat English-derived accuracy claims (including
    quantization "no accuracy loss" claims) as unverified for Thai until measured
    on a Thai eval set (see conf.RERANKER_BACKEND for the ONNX-quantization
    measurement this project already ran).

Usage:
    from utils.reranker import rerank

    docs = rerank(query="จะขอใบอนุญาตเปิดร้านอาหาร", docs=langchain_docs, top_k=6)

Toggle via env: RERANKER_ENABLED=true / RERANKER_MODEL=... / RERANKER_TOP_K=10
Backend toggle: RERANKER_BACKEND=pytorch (default) / onnx — see conf.py for the
measured accuracy comparison behind this flag before switching it on.
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

# Singleton: (model_name, onnx_file) → (tokenizer, ORTModelForSequenceClassification).
# Separate cache from _reranker_cache since it's a different backend/object type.
_onnx_reranker_cache: dict = {}
_onnx_reranker_lock = threading.Lock()


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


def _get_onnx_reranker(model_name: str, onnx_file: str) -> Any:
    """Lazy-load (tokenizer, quantized ORT model) singleton, same double-checked
    locking pattern as _get_reranker. Loads a pre-quantized ONNX file already
    published on the model's HF Hub repo under onnx/<onnx_file> (no local export
    step needed) via Hugging Face Optimum + ONNX Runtime.

    Bypasses sentence-transformers' own ONNX CrossEncoder backend (added in
    sentence-transformers>=4.1.0) on purpose — this project pins
    sentence-transformers<4 in requirements.txt for the embedding-model code path,
    so this talks to optimum.onnxruntime directly instead of forcing that bump.
    """
    cache_key = (model_name, onnx_file)
    if cache_key not in _onnx_reranker_cache:
        with _onnx_reranker_lock:
            if cache_key not in _onnx_reranker_cache:
                _LOG.info("[Reranker/ONNX] Loading quantized model: %s (onnx/%s)", model_name, onnx_file)
                t0 = time.time()
                try:
                    from transformers import AutoTokenizer  # type: ignore
                    from optimum.onnxruntime import ORTModelForSequenceClassification  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "optimum[onnxruntime] is required for RERANKER_BACKEND=onnx. "
                        "Install with: pip install 'optimum[onnxruntime]'"
                    ) from e

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                ort_model = ORTModelForSequenceClassification.from_pretrained(
                    model_name, subfolder="onnx", file_name=onnx_file
                )
                _onnx_reranker_cache[cache_key] = (tokenizer, ort_model)
                _LOG.info("[Reranker/ONNX] Model loaded in %.1fs", time.time() - t0)
    return _onnx_reranker_cache[cache_key]


def _predict_onnx(tokenizer: Any, ort_model: Any, pairs: List[tuple]) -> List[float]:
    """Score (query, doc_text) pairs with the quantized ONNX model. Mirrors
    CrossEncoder.predict()'s tokenization (padding, truncation, max_length=512) so
    scores are comparable to the float32 path — this exact tokenization was used
    for the accuracy measurement documented in conf.py's RERANKER_BACKEND comment."""
    import torch  # already a transitive dep of sentence-transformers/langchain-openai

    enc = tokenizer(
        [p[0] for p in pairs], [p[1] for p in pairs],
        padding=True, truncation=True, max_length=512, return_tensors="pt",
    )
    with torch.no_grad():
        logits = ort_model(**enc).logits.squeeze(-1)
    return logits.tolist()


def rerank(
    query: str,
    docs: List[Any],
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    top_k: Optional[int] = None,
    backend: str = "pytorch",
    onnx_file: str = "model_quint8_avx2.onnx",
    min_score: Optional[float] = None,
) -> List[Any]:
    """
    Rerank LangChain Document objects by cross-encoder relevance score.

    Args:
        query:      User query string (Thai or English).
        docs:       List of LangChain Document objects (must have .page_content).
        model_name: HuggingFace cross-encoder model ID.
        top_k:      If given, return only the top_k highest-scoring docs.
        backend:    "pytorch" (default, float32 CrossEncoder) or "onnx" (quantized
                    INT8 via Optimum/ONNX Runtime — see conf.RERANKER_BACKEND for
                    the measured accuracy trade-off before enabling).
        onnx_file:  Which pre-quantized file to load from the model's onnx/
                    subfolder when backend="onnx".
        min_score:  If given, drop any doc scoring below this value BEFORE applying
                    top_k — mitigates quantization noise near the decision boundary
                    occasionally letting a borderline off-topic doc (already present
                    in the candidate pool from the retrieval stage) into the kept set
                    when it wouldn't have made it under float32 (see conf.py's
                    RERANKER_MIN_SCORE for the measured incident this addresses).
                    Always keeps at least the single best-scoring doc even if it's
                    below threshold — an empty rerank result would push the caller
                    into a worse "no docs found" fallback than one borderline doc.

    Returns:
        Reranked docs list (same objects, new order). Stores _rerank_score in metadata.
        Returns original docs unchanged on error.
    """
    if not docs or not query:
        return docs

    try:
        t0 = time.time()
        pairs = [
            (query, (getattr(d, "page_content", "") or "")[:1500])
            for d in docs
        ]

        if backend == "onnx":
            tokenizer, ort_model = _get_onnx_reranker(model_name, onnx_file)
            scores: List[float] = _predict_onnx(tokenizer, ort_model, pairs)
        else:
            reranker = _get_reranker(model_name)
            scores = reranker.predict(pairs).tolist()

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        if min_score is not None and ranked:
            _above = [(d, s) for d, s in ranked if s >= min_score]
            ranked = _above if _above else ranked[:1]

        if top_k is not None:
            ranked = ranked[:top_k]

        # Store rerank score only on docs that are actually returned
        for doc, score in ranked:
            md = getattr(doc, "metadata", None)
            if isinstance(md, dict):
                md["_rerank_score"] = round(float(score), 4)

        elapsed_ms = (time.time() - t0) * 1000
        _LOG.info(
            "[Reranker] backend=%s reranked %d → %d docs in %.0fms | top_score=%.3f",
            backend,
            len(docs),
            len(ranked),
            elapsed_ms,
            ranked[0][1] if ranked else 0.0,
        )
        return [d for d, _ in ranked]

    except Exception as exc:
        _LOG.warning("[Reranker] rerank failed (%s) — returning original order", exc)
        return docs


def _regulatory_bucket(doc: Any) -> str:
    """Coarse 2-way bucket for filter_by_category_concentration: "regulatory" vs
    "non_regulatory" (business_guide + marketing merged), not the full 3-way
    data_type split. Merging business_guide/marketing together is deliberate — the
    real failure mode observed live was regulatory content leaking into a
    non-regulatory question, never business_guide/marketing conflicting with each
    other. Keeping them separate buckets caused a confirmed false-positive in local
    testing: a legitimate "marketing" doc got pruned for being a numeric minority
    against a "business_guide" majority on a marketing question, which is exactly
    the kind of unwanted removal this function must not do. Docs with missing/
    unrecognized data_type return "" (never a majority, so they can't trigger
    filtering and are never themselves filtered out).
    """
    dt = (getattr(doc, "metadata", None) or {}).get("data_type") or ""
    if dt == "regulatory":
        return "regulatory"
    if dt in ("business_guide", "marketing"):
        return "non_regulatory"
    return ""


def filter_by_category_concentration(
    docs: List[Any],
    concentration_threshold: float = 0.75,
    min_pool_size: int = 5,
    min_keep: int = 3,
) -> List[Any]:
    """Drop docs from the minority bucket (regulatory vs. non_regulatory — see
    _regulatory_bucket) when the candidate pool is heavily dominated by one side —
    mitigates off-category docs (e.g. a licensing doc surfacing for a pure marketing
    question) entering the reranked/LLM-facing set. Intended to run BEFORE rerank(),
    on the raw retrieval candidate pool.

    Root cause this addresses: retrieval (embedding similarity) has no data_type
    filter at all, so an off-category doc can enter the candidate pool purely on
    semantic proximity; a reranker score threshold can only catch this when that
    doc's score is borderline — it can't help when retrieval itself surfaces the doc
    with a confidently high score. This catches it earlier: when almost every
    candidate that WAS retrieved already agrees on one side of the regulatory /
    non-regulatory line, a lone doc from the other side is very likely retrieval
    noise, not a genuine cross-category need.

    Genuinely mixed pools (e.g. "เงินลงทุนเปิดร้าน" naturally pulling both
    business_guide investment content and regulatory fee docs) are a real signal
    that the question legitimately spans categories — those pools won't clear
    concentration_threshold and are left untouched, so this does not hard-block
    cross-category answers, only prunes lopsided pools.

    Safety: no-ops below min_pool_size (not enough signal to judge concentration
    reliably) and never reduces the result below min_keep docs (a near-empty
    candidate set is worse than a few off-category docs slipping through).
    """
    if len(docs) < min_pool_size:
        return docs

    from collections import Counter

    counts = Counter(_regulatory_bucket(d) for d in docs)
    majority_bucket, majority_count = counts.most_common(1)[0]
    if not majority_bucket or majority_count / len(docs) < concentration_threshold:
        return docs

    filtered = [d for d in docs if _regulatory_bucket(d) == majority_bucket]
    if len(filtered) < min_keep:
        return docs

    if len(filtered) != len(docs):
        _LOG.info(
            "[CategoryFilter] %d → %d docs | kept bucket=%r (%.0f%% concentration)",
            len(docs), len(filtered), majority_bucket, 100 * majority_count / len(docs),
        )
    return filtered

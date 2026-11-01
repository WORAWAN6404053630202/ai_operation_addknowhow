"""
Guards the Phase 2 design decision from .claude/plans/mutable-seeking-wave.md
(2026-11 ES-parallel-project): `_sim` (the score persona_supervisor.py
attaches to doc.metadata["_sim"], consumed by utils.text_patterns.
regulatory_identity_mismatch_should_reject's hardcoded min_score=0.70 and by
RETRIEVAL_MIN_SIMILARITY/RETRIEVAL_MIN_SIMILARITY_PDF_BOOK) must ALWAYS be a
pure dense-cosine score bounded 0-1 — never a fused/hybrid/BM25 score, which
has no shared scale across queries and would silently miscalibrate those
thresholds.

Today (Phase 1+2, before Phase 3 adds hybrid search) this is automatically
true: ESCompatVectorStore.similarity_search_with_relevance_scores() only
ever runs a plain kNN search. This test file exists so that whoever
implements Phase 3's hybrid search has an explicit, named test to keep
green — if hybrid search starts returning a fused RRF/BM25+kNN score from
the SAME call these tests exercise, that's the exact regression this guards
against, and Phase 3 must instead compute `_sim` from a separate pure-kNN
sub-query (see the plan's Phase 2 section for the full rationale).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from service.es_vector_store import ESCompatCollection, ESCompatVectorStore


@pytest.mark.unit
class TestSimScoreIsBoundedCosine:
    def test_perfect_match_yields_score_one(self):
        """ES kNN _score for a cosine dense_vector field is
        (1 + cosine_similarity) / 2 — a perfect match (cosine_similarity=1.0)
        has _score=1.0, which must round-trip to a relevance score of 1.0,
        not some other scale."""
        client = MagicMock()
        client.search.return_value = {
            "hits": {"hits": [{"_id": "a", "_score": 1.0, "_source": {"page_content": "x"}}]}
        }
        embedding_model = MagicMock()
        embedding_model.embed_query.return_value = [0.1]
        vs = ESCompatVectorStore(client, "idx", embedding_model)

        _, score = vs.similarity_search_with_relevance_scores("q", k=1)[0]

        assert score == pytest.approx(1.0)

    def test_worst_match_yields_score_zero(self):
        """cosine_similarity=-1.0 (orthogonal/opposite vectors) has ES kNN
        _score=0.0, which must round-trip to relevance score 0.0 — the
        bottom of the bounded 0-1 scale every downstream threshold assumes."""
        client = MagicMock()
        client.search.return_value = {
            "hits": {"hits": [{"_id": "a", "_score": 0.0, "_source": {"page_content": "x"}}]}
        }
        embedding_model = MagicMock()
        embedding_model.embed_query.return_value = [0.1]
        vs = ESCompatVectorStore(client, "idx", embedding_model)

        _, score = vs.similarity_search_with_relevance_scores("q", k=1)[0]

        assert score == pytest.approx(0.0)

    @pytest.mark.parametrize("es_score", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_score_always_bounded_zero_to_one(self, es_score):
        """No ES kNN _score (for a cosine field, always 0-1 by construction)
        should ever produce an out-of-bounds relevance score after the
        distance-roundtrip conversion in ESCompatCollection.query()."""
        client = MagicMock()
        client.search.return_value = {
            "hits": {"hits": [{"_id": "a", "_score": es_score, "_source": {"page_content": "x"}}]}
        }
        embedding_model = MagicMock()
        embedding_model.embed_query.return_value = [0.1]
        vs = ESCompatVectorStore(client, "idx", embedding_model)

        _, score = vs.similarity_search_with_relevance_scores("q", k=1)[0]

        assert 0.0 <= score <= 1.0

    def test_query_method_never_passes_a_fused_hybrid_score_as_the_knn_score(self):
        """Documents today's contract explicitly: ESCompatCollection.query()
        (the thing similarity_search_with_relevance_scores calls) issues a
        plain `knn=` search with no `rrf`/`sub_searches` retriever body — the
        one search type guaranteed to return a pure dense-cosine _score.
        When Phase 3 adds a separate hybrid method, THAT method must not be
        what _sim is computed from; this test pins down that `query()`
        itself stays knn-only so a future hybrid implementation doesn't
        accidentally get merged into this same code path."""
        client = MagicMock()
        client.search.return_value = {"hits": {"hits": []}}
        coll = ESCompatCollection(client, "idx")

        coll.query(query_embeddings=[[0.1, 0.2]], n_results=5)

        search_kwargs = client.search.call_args.kwargs
        assert "knn" in search_kwargs
        assert "rrf" not in search_kwargs
        assert "sub_searches" not in search_kwargs

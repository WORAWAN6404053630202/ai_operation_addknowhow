"""
Unit tests for the Round 2 ES-native hybrid search (.claude/plans/
mutable-seeking-wave.md): ESCompatCollection.match_search,
ESCompatVectorStore.hybrid_search, and the module-level _rrf_fuse —
Elasticsearch's own kNN + `match` (BM25) legs fused client-side with
Reciprocal Rank Fusion, since the native `retriever.rrf` is gated to
Elastic's Enterprise license tier (confirmed via live 403 in Phase 0).

Same mocking approach as test_es_vector_store_facade.py: construct real
objects, monkeypatch only the `elasticsearch.Elasticsearch` client (the
network boundary).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from service.es_vector_store import ESCompatCollection, ESCompatVectorStore, _rrf_fuse


@pytest.mark.unit
class TestMatchSearch:
    def test_issues_match_query_on_page_content(self):
        client = MagicMock()
        client.search.return_value = {"hits": {"hits": []}}
        coll = ESCompatCollection(client, "my_index")

        coll.match_search("ค่าธรรมเนียม", k=10)

        call_kwargs = client.search.call_args.kwargs
        assert call_kwargs["size"] == 10
        assert call_kwargs["body"]["query"]["bool"]["must"] == [{"match": {"page_content": "ค่าธรรมเนียม"}}]
        assert "filter" not in call_kwargs["body"]["query"]["bool"]

    def test_filter_translated_into_bool_filter_clause(self):
        client = MagicMock()
        client.search.return_value = {"hits": {"hits": []}}
        coll = ESCompatCollection(client, "my_index")

        coll.match_search("q", k=10, where={"license_type": "x"})

        body = client.search.call_args.kwargs["body"]
        assert body["query"]["bool"]["filter"] == [{"term": {"license_type": "x"}}]

    def test_returns_doc_score_pairs_with_es_id_attached(self):
        client = MagicMock()
        client.search.return_value = {
            "hits": {"hits": [{"_id": "a", "_score": 5.5, "_source": {"page_content": "text", "license_type": "x"}}]}
        }
        coll = ESCompatCollection(client, "my_index")

        out = coll.match_search("q", k=10)

        assert len(out) == 1
        doc, score = out[0]
        assert doc.page_content == "text"
        assert doc.metadata["_es_id"] == "a"
        assert doc.metadata["license_type"] == "x"
        assert score == pytest.approx(5.5)


@pytest.mark.unit
class TestRrfFuse:
    """Hand-built example with known ranks, verified against the standard
    RRF formula score(d) = Σ 1/(k + rank(d)) — same formula/k=60 default
    utils/hybrid_retriever.py's _rrf_fuse already uses for Chroma."""

    def _docs(self):
        doc_a = Document(page_content="A", metadata={"_es_id": "a"})
        doc_b_dense = Document(page_content="B", metadata={"_es_id": "b"})
        doc_b_bm25 = Document(page_content="B", metadata={"_es_id": "b"})
        doc_c = Document(page_content="C", metadata={"_es_id": "c"})
        return doc_a, doc_b_dense, doc_b_bm25, doc_c

    def test_fusion_order_matches_rrf_formula(self):
        doc_a, doc_b_dense, doc_b_bm25, doc_c = self._docs()
        # Dense: rank0=A(0.9), rank1=B(0.5). BM25: rank0=B(10.0), rank1=C(8.0).
        dense_pairs = [(doc_a, 0.9), (doc_b_dense, 0.5)]
        bm25_pairs = [(doc_b_bm25, 10.0), (doc_c, 8.0)]

        fused = _rrf_fuse(dense_pairs, bm25_pairs, rrf_k=60, bm25_bypass_min_ratio=0.35)

        ids_in_order = [d.metadata["_es_id"] for d, _ in fused]
        # B: 1/62 (dense rank1) + 1/61 (bm25 rank0) ≈ 0.03252 — highest.
        # A: 1/61 (dense rank0 only) ≈ 0.01639.
        # C: 1/62 (bm25 rank1 only) ≈ 0.01613 — lowest.
        assert ids_in_order == ["b", "a", "c"]

    def test_sim_score_is_dense_leg_only_never_fused_or_bm25(self):
        """The whole point of keeping this a separate function from a fused
        score: _sim must stay the pure-kNN cosine value so RETRIEVAL_MIN_
        SIMILARITY*/regulatory_identity_mismatch_should_reject's 0.70
        threshold (calibrated against that scale) still mean what they did
        before hybrid search existed."""
        doc_a, doc_b_dense, doc_b_bm25, doc_c = self._docs()
        dense_pairs = [(doc_a, 0.9), (doc_b_dense, 0.5)]
        bm25_pairs = [(doc_b_bm25, 10.0), (doc_c, 8.0)]

        fused = _rrf_fuse(dense_pairs, bm25_pairs, rrf_k=60, bm25_bypass_min_ratio=0.35)
        scores_by_id = {d.metadata["_es_id"]: s for d, s in fused}

        assert scores_by_id["a"] == pytest.approx(0.9)
        assert scores_by_id["b"] == pytest.approx(0.5)
        # C is BM25-only — gets 0.0, not its raw (unbounded, incomparable) BM25 score.
        assert scores_by_id["c"] == pytest.approx(0.0)

    def test_bm25_only_doc_above_ratio_gets_bypass_flag(self):
        doc_a, doc_b_dense, doc_b_bm25, doc_c = self._docs()
        # C's BM25 score (8.0) is 80% of the top BM25 score (10.0) — above the 0.35 ratio.
        fused = _rrf_fuse([(doc_a, 0.9)], [(doc_b_bm25, 10.0), (doc_c, 8.0)], rrf_k=60, bm25_bypass_min_ratio=0.35)
        by_id = {d.metadata["_es_id"]: d for d, _ in fused}
        assert by_id["c"].metadata.get("_bm25_hit") is True

    def test_bm25_only_doc_below_ratio_does_not_get_bypass_flag(self):
        doc_a, doc_b_dense, doc_b_bm25, doc_c = self._docs()
        weak = Document(page_content="weak", metadata={"_es_id": "weak"})
        # weak's score (1.0) is only 10% of the top BM25 score (10.0) — below the 0.35 ratio.
        fused = _rrf_fuse([(doc_a, 0.9)], [(doc_b_bm25, 10.0), (weak, 1.0)], rrf_k=60, bm25_bypass_min_ratio=0.35)
        by_id = {d.metadata["_es_id"]: d for d, _ in fused}
        assert "_bm25_hit" not in by_id["weak"].metadata

    def test_doc_present_in_both_legs_does_not_get_bypass_flag(self):
        """The bypass flag only applies to docs MISSING from the dense leg
        entirely — a doc already found by dense search doesn't need it."""
        doc_a, doc_b_dense, doc_b_bm25, doc_c = self._docs()
        fused = _rrf_fuse([(doc_a, 0.9), (doc_b_dense, 0.5)], [(doc_b_bm25, 10.0)], rrf_k=60, bm25_bypass_min_ratio=0.35)
        by_id = {d.metadata["_es_id"]: d for d, _ in fused}
        assert "_bm25_hit" not in by_id["b"].metadata

    def test_empty_bm25_pairs_falls_back_to_dense_order(self):
        doc_a, doc_b_dense, _, _ = self._docs()
        fused = _rrf_fuse([(doc_a, 0.9), (doc_b_dense, 0.5)], [], rrf_k=60, bm25_bypass_min_ratio=0.35)
        assert [d.metadata["_es_id"] for d, _ in fused] == ["a", "b"]


@pytest.mark.unit
class TestHybridSearch:
    def test_runs_both_knn_and_match_legs(self):
        client = MagicMock()
        # kNN leg (via query()) then match leg (via match_search()) — same client, different calls.
        client.search.side_effect = [
            {"hits": {"hits": [{"_id": "a", "_score": 1.0, "_source": {"page_content": "dense hit"}}]}},
            {"hits": {"hits": [{"_id": "b", "_score": 3.0, "_source": {"page_content": "bm25 hit"}}]}},
        ]
        embedding_model = MagicMock()
        embedding_model.embed_query.return_value = [0.1, 0.2]
        vs = ESCompatVectorStore(client, "idx", embedding_model)

        results = vs.hybrid_search("query text", k=5)

        assert client.search.call_count == 2
        ids = [d.metadata["_es_id"] for d, _ in results]
        assert set(ids) == {"a", "b"}

    def test_oversamples_bm25_more_when_a_filter_is_present(self):
        """A metadata filter can be very selective — oversample the BM25 leg
        (k*5) before fusion so filtering doesn't starve the candidate pool,
        same reasoning hybrid_retriever.py's Chroma version already uses."""
        client = MagicMock()
        client.search.return_value = {"hits": {"hits": []}}
        embedding_model = MagicMock()
        embedding_model.embed_query.return_value = [0.1]
        vs = ESCompatVectorStore(client, "idx", embedding_model)

        vs.hybrid_search("q", k=10, filter={"license_type": "x"})

        # Second call is the match_search leg — check its `size`.
        match_call_size = client.search.call_args_list[1].kwargs["size"]
        assert match_call_size == 50  # k * 5

    def test_no_filter_uses_smaller_oversample(self):
        client = MagicMock()
        client.search.return_value = {"hits": {"hits": []}}
        embedding_model = MagicMock()
        embedding_model.embed_query.return_value = [0.1]
        vs = ESCompatVectorStore(client, "idx", embedding_model)

        vs.hybrid_search("q", k=10)

        match_call_size = client.search.call_args_list[1].kwargs["size"]
        assert match_call_size == 20  # k * 2

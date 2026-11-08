"""
Unit tests for service.es_vector_store — the Chroma-shaped Elasticsearch
facade from the 2026-11 ES-parallel-project work (see the approved plan at
.claude/plans/mutable-seeking-wave.md, Phase 1). The `elasticsearch.
Elasticsearch` client is the network boundary, mocked directly throughout —
same approach test_groundedness_safety_net.py uses for its LLM-calling
dependencies (construct the real object, monkeypatch only the network edge).

These assert request/response SHAPE correctness (what gets sent to ES, what
comes back parsed into the Chroma-compatible return shape), not real ES
behavior — a separate @pytest.mark.integration tier (not in this file) runs
against a real ES container when one is available, per the plan's Phase 7.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
from langchain_core.documents import Document

from service.es_vector_store import (
    ESCompatCollection,
    ESCompatVectorStore,
    ESVectorStoreManager,
    _index_mapping,
)


@pytest.mark.unit
class TestIndexMapping:
    """A replica can never be assigned on a single-node cluster — the index
    was silently stuck 'yellow' since Session 1 because _index_mapping()
    never set number_of_replicas. Round 3 Step 0 fix (.claude/plans/
    mutable-seeking-wave.md)."""

    def test_number_of_replicas_is_zero(self):
        mapping = _index_mapping()
        assert mapping["settings"]["index"]["number_of_replicas"] == 0

    def test_mappings_unchanged(self):
        mapping = _index_mapping()
        assert mapping["mappings"]["properties"]["page_content"] == {"type": "text"}
        assert mapping["mappings"]["properties"]["embedding"]["type"] == "dense_vector"

    def test_tokenized_field_uses_whitespace_analyzer(self):
        """2026-11: page_content_tokenized holds PyThaiNLP-pre-segmented text
        (utils/thai_tokenize.py) — it must use `whitespace`, not ES's
        default `standard` (which would re-tokenize with Thai-unaware rules,
        defeating the whole point of pre-segmenting)."""
        mapping = _index_mapping()
        assert mapping["mappings"]["properties"]["page_content_tokenized"] == {
            "type": "text", "analyzer": "whitespace",
        }


@pytest.mark.unit
class TestESCompatCollectionGet:
    def test_get_by_ids_uses_mget(self):
        client = MagicMock()
        client.mget.return_value = {
            "docs": [
                {"_id": "a", "found": True, "_source": {"page_content": "x", "license_type": "y"}},
                {"_id": "b", "found": False},
            ]
        }
        coll = ESCompatCollection(client, "my_index")

        out = coll.get(ids=["a", "b"])

        client.mget.assert_called_once()
        assert out["ids"] == ["a"]
        assert out["documents"] == ["x"]
        assert out["metadatas"] == [{"license_type": "y"}]

    def test_get_by_ids_passes_flattened_source_excludes_not_a_dict(self):
        """2026-11 regression pin: elasticsearch-py 8.19.3's .mget() silently
        returns empty _source for every doc when given source={"excludes":
        [...]} as a dict (no error) — only the flattened source_excludes=
        kwarg works, confirmed live. The mock above can't catch this on its
        own (it returns the same fixed response regardless of kwargs), so
        this test asserts the actual kwarg shape sent, not just the parsed
        output."""
        client = MagicMock()
        client.mget.return_value = {"docs": [{"_id": "a", "found": True, "_source": {}}]}
        coll = ESCompatCollection(client, "my_index")

        coll.get(ids=["a"])

        mget_kwargs = client.mget.call_args.kwargs
        assert mget_kwargs.get("source_excludes") == ["embedding"]
        assert "source" not in mget_kwargs

    def test_get_by_where_uses_search_with_translated_filter(self):
        client = MagicMock()
        client.search.return_value = {
            "hits": {"hits": [{"_id": "a", "_source": {"page_content": "x", "license_type": "y"}}]}
        }
        coll = ESCompatCollection(client, "my_index")

        out = coll.get(where={"license_type": "y"})

        search_kwargs = client.search.call_args.kwargs
        assert search_kwargs["body"]["query"] == {"term": {"license_type": "y"}}
        assert out["ids"] == ["a"]

    def test_get_by_where_also_passes_flattened_source_excludes(self):
        """Aligns with the ids= branch above — same confirmed-working kwarg
        style used for both .mget() and .search() in this one method."""
        client = MagicMock()
        client.search.return_value = {"hits": {"hits": []}}
        coll = ESCompatCollection(client, "my_index")

        coll.get(where={"license_type": "y"})

        search_kwargs = client.search.call_args.kwargs
        assert search_kwargs.get("source_excludes") == ["embedding"]

    def test_get_with_embeddings_included(self):
        client = MagicMock()
        client.mget.return_value = {
            "docs": [{"_id": "a", "found": True, "_source": {"page_content": "x", "embedding": [0.1, 0.2]}}]
        }
        coll = ESCompatCollection(client, "my_index")

        out = coll.get(ids=["a"], include=["metadatas", "documents", "embeddings"])

        assert out["embeddings"] == [[0.1, 0.2]]

    def test_get_with_embeddings_included_omits_source_excludes(self):
        """When embeddings ARE requested, no exclusion kwarg should be sent
        at all — confirms this branch is untouched by the source_excludes fix."""
        client = MagicMock()
        client.mget.return_value = {"docs": [{"_id": "a", "found": True, "_source": {}}]}
        coll = ESCompatCollection(client, "my_index")

        coll.get(ids=["a"], include=["metadatas", "documents", "embeddings"])

        mget_kwargs = client.mget.call_args.kwargs
        assert "source_excludes" not in mget_kwargs


@pytest.mark.unit
class TestESCompatCollectionQuery:
    def test_query_shape_matches_chroma_nested_list_of_one(self):
        """local_vector_store.py's query_live_with_ids unpacks Chroma's
        return shape with `(results.get("ids") or [[]])[0]` — this facade
        must produce that exact nested-list-of-one shape."""
        client = MagicMock()
        client.search.return_value = {
            "hits": {"hits": [{"_id": "a", "_score": 0.9, "_source": {"page_content": "x", "license_type": "y"}}]}
        }
        coll = ESCompatCollection(client, "my_index")

        out = coll.query(query_embeddings=[[0.1, 0.2]], n_results=5)

        assert out["ids"] == [["a"]]
        assert out["documents"] == [["x"]]
        assert out["metadatas"] == [[{"license_type": "y"}]]
        assert len(out["distances"][0]) == 1

    def test_query_cosine_score_converts_to_chroma_shaped_distance(self):
        """ES kNN score for a cosine dense_vector field is
        (1 + cosine_similarity) / 2. A perfect match (cosine_similarity=1.0)
        has ES _score=1.0 — this must convert to distance=0.0 (Chroma's
        "1 - distance = cosine similarity" contract, so callers doing
        1.0 - distance recover 1.0, a perfect match)."""
        client = MagicMock()
        client.search.return_value = {
            "hits": {"hits": [{"_id": "a", "_score": 1.0, "_source": {"page_content": "x"}}]}
        }
        coll = ESCompatCollection(client, "my_index")

        out = coll.query(query_embeddings=[[0.1, 0.2]], n_results=5)

        assert out["distances"][0][0] == pytest.approx(0.0)

    def test_query_passes_filter_and_knn_params(self):
        client = MagicMock()
        client.search.return_value = {"hits": {"hits": []}}
        coll = ESCompatCollection(client, "my_index")

        coll.query(query_embeddings=[[0.1, 0.2]], n_results=7, where={"license_type": "y"})

        knn_arg = client.search.call_args.kwargs["knn"]
        assert knn_arg["k"] == 7
        assert knn_arg["filter"] == {"term": {"license_type": "y"}}


@pytest.mark.unit
class TestESCompatCollectionMutation:
    def test_delete_by_ids_calls_delete_per_id(self):
        client = MagicMock()
        coll = ESCompatCollection(client, "my_index")

        coll.delete(ids=["a", "b"])

        client.delete.assert_has_calls(
            [call(index="my_index", id="a", ignore=[404]), call(index="my_index", id="b", ignore=[404])]
        )

    def test_delete_by_where_uses_delete_by_query(self):
        client = MagicMock()
        coll = ESCompatCollection(client, "my_index")

        coll.delete(where={"license_type": "y"})

        client.delete_by_query.assert_called_once()
        assert client.delete_by_query.call_args.kwargs["body"]["query"] == {"term": {"license_type": "y"}}

    def test_count_returns_int(self):
        client = MagicMock()
        client.count.return_value = {"count": 42}
        coll = ESCompatCollection(client, "my_index")

        assert coll.count() == 42

    def test_count_failure_returns_zero_not_raise(self):
        client = MagicMock()
        client.count.side_effect = RuntimeError("down")
        coll = ESCompatCollection(client, "my_index")

        assert coll.count() == 0


@pytest.mark.unit
class TestESCompatVectorStore:
    def test_similarity_search_with_relevance_scores_embeds_query_once(self):
        client = MagicMock()
        client.search.return_value = {
            "hits": {"hits": [{"_id": "a", "_score": 0.8, "_source": {"page_content": "real answer"}}]}
        }
        embedding_model = MagicMock()
        embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        vs = ESCompatVectorStore(client, "my_index", embedding_model)

        results = vs.similarity_search_with_relevance_scores("คำถาม", k=3)

        embedding_model.embed_query.assert_called_once_with("คำถาม")
        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert doc.page_content == "real answer"
        assert 0.0 <= score <= 1.0

    def test_as_retriever_invoke_returns_documents(self):
        client = MagicMock()
        client.search.return_value = {
            "hits": {"hits": [{"_id": "a", "_score": 1.0, "_source": {"page_content": "x"}}]}
        }
        embedding_model = MagicMock()
        embedding_model.embed_query.return_value = [0.1]
        vs = ESCompatVectorStore(client, "my_index", embedding_model)

        retriever = vs.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke("q")

        assert len(docs) == 1
        assert docs[0].page_content == "x"

    def test_as_retriever_exposes_vectorstore_attribute(self):
        """2026-11 (hardcoded-logic audit, live-reproduced): every one of ~38
        call sites across persona_supervisor.py/persona_practical.py/
        persona_academic.py does getattr(self.retriever, "vectorstore",
        None), expecting this retriever to behave like LangChain's own
        Chroma retriever wrapper (which sets .vectorstore). This class used
        to omit it entirely, so vectorstore was always None on the ES
        backend — confirmed live via a real /chat request before this fix —
        silently skipping every metadata-filtered fetch and every
        hybrid_search() call at those sites, always falling back to this
        retriever's own filterless, non-hybrid .invoke() instead."""
        client = MagicMock()
        embedding_model = MagicMock()
        vs = ESCompatVectorStore(client, "my_index", embedding_model)

        retriever = vs.as_retriever(search_kwargs={"k": 5})

        assert getattr(retriever, "vectorstore", None) is vs


@pytest.mark.unit
class TestArchiveAndRemoveOrdering:
    """Mirrors local_vector_store.py's archive_and_remove() contract: fetch
    with embeddings -> upsert into archive -> VERIFY presence in archive ->
    only then delete from live. Never delete-before-verify."""

    def test_archive_before_delete_ordering(self):
        mgr = ESVectorStoreManager()
        mgr.vectorstore = MagicMock()
        mgr.vectorstore._collection.get.side_effect = [
            {"ids": ["a"], "documents": ["doc"], "metadatas": [{"license_type": "y"}], "embeddings": [[0.1]]},
        ]
        mgr._get_client = MagicMock(return_value=MagicMock())
        # Real success path touches the filesystem (corpus-version marker
        # file) — not this test's concern, mock it out.
        mgr._bump_corpus_version = MagicMock()
        call_order = []

        def _fake_ensure_index(*a, **k):
            pass

        # Patch ESCompatCollection used internally for the archive index to
        # track call order without hitting real ES.
        import service.es_vector_store as mod
        real_coll_cls = mod.ESCompatCollection

        class _TrackingCollection(real_coll_cls):
            def upsert(self, *a, **k):
                call_order.append("archive_upsert")

            def get(self, *a, **k):
                call_order.append("archive_verify")
                return {"ids": ["a"]}

        mod.ESCompatCollection = _TrackingCollection
        try:
            n = mgr.archive_and_remove(["a"], reason="test")
        finally:
            mod.ESCompatCollection = real_coll_cls

        assert n == 1
        assert call_order == ["archive_upsert", "archive_verify"]
        mgr.vectorstore._collection.delete.assert_called_once_with(ids=["a"])

    def test_archive_verify_mismatch_refuses_to_delete_from_live(self):
        mgr = ESVectorStoreManager()
        mgr.vectorstore = MagicMock()
        mgr.vectorstore._collection.get.return_value = {
            "ids": ["a", "b"], "documents": ["d1", "d2"],
            "metadatas": [{}, {}], "embeddings": [[0.1], [0.2]],
        }
        mgr._get_client = MagicMock(return_value=MagicMock())

        import service.es_vector_store as mod
        real_coll_cls = mod.ESCompatCollection

        class _PartialVerifyCollection(real_coll_cls):
            def upsert(self, *a, **k):
                pass

            def get(self, *a, **k):
                return {"ids": ["a"]}  # only 1 of 2 verified present

        mod.ESCompatCollection = _PartialVerifyCollection
        try:
            n = mgr.archive_and_remove(["a", "b"], reason="test")
        finally:
            mod.ESCompatCollection = real_coll_cls

        assert n == 0
        mgr.vectorstore._collection.delete.assert_not_called()


@pytest.mark.unit
class TestListArchivedDocs:
    """2026-11: admin_pdf.py's "recently archived docs" panel always showed
    [] on the ES-backed deployment because this function simply didn't
    exist on this side (Chroma-only). Same pattern/since_epoch filter+sort
    contract as local_vector_store.py's LocalVectorStoreManager.
    list_archived_docs — verified here against a fake full-dump response."""

    def _mgr_with_fake_dump(self, raw_dump: dict) -> ESVectorStoreManager:
        mgr = ESVectorStoreManager()
        mgr._get_client = MagicMock(return_value=MagicMock())

        import service.es_vector_store as mod

        class _FakeCollection(mod.ESCompatCollection):
            def get(self, *a, **k):
                return raw_dump

        mod.ESCompatCollection = _FakeCollection
        return mgr

    def test_filters_by_pattern_and_since_epoch_and_sorts_newest_first(self):
        import service.es_vector_store as mod
        real_coll_cls = mod.ESCompatCollection
        mgr = self._mgr_with_fake_dump({
            "ids": ["a", "b", "c"],
            "metadatas": [
                {"source_file": "x.pdf", "superseded_at": "100"},
                {"source_file": "y.pdf", "superseded_at": "300"},
                {"source_file": "x.pdf", "superseded_at": "200"},
            ],
            "documents": ["d1", "d2", "d3"],
        })
        try:
            out = mgr.list_archived_docs(pattern="x.pdf", since_epoch=150)
        finally:
            mod.ESCompatCollection = real_coll_cls

        # b excluded by pattern (y.pdf), a excluded by since_epoch (100 < 150)
        assert [d["id"] for d in out] == ["c"]

    def test_no_filters_returns_full_dump_newest_first(self):
        import service.es_vector_store as mod
        real_coll_cls = mod.ESCompatCollection
        mgr = self._mgr_with_fake_dump({
            "ids": ["a", "b", "c"],
            "metadatas": [
                {"source_file": "x.pdf", "superseded_at": "100"},
                {"source_file": "y.pdf", "superseded_at": "300"},
                {"source_file": "x.pdf", "superseded_at": "200"},
            ],
            "documents": ["d1", "d2", "d3"],
        })
        try:
            out = mgr.list_archived_docs()
        finally:
            mod.ESCompatCollection = real_coll_cls

        assert [d["id"] for d in out] == ["b", "c", "a"]

    def test_missing_or_bad_superseded_at_sorts_last_not_crashes(self):
        import service.es_vector_store as mod
        real_coll_cls = mod.ESCompatCollection
        mgr = self._mgr_with_fake_dump({
            "ids": ["a", "b"],
            "metadatas": [{"source_file": "x.pdf"}, {"source_file": "y.pdf", "superseded_at": "bad"}],
            "documents": ["d1", "d2"],
        })
        try:
            out = mgr.list_archived_docs()
        finally:
            mod.ESCompatCollection = real_coll_cls

        assert len(out) == 2

    def test_no_op_on_empty_doc_ids(self):
        mgr = ESVectorStoreManager()
        mgr.vectorstore = MagicMock()

        assert mgr.archive_and_remove([], reason="test") == 0
        mgr.vectorstore._collection.get.assert_not_called()


@pytest.mark.unit
class TestQueryLiveWithIds:
    def test_returns_chroma_supersede_resolver_shape(self):
        """service/pdf_supersede_resolver.py consumes this exact
        {"id","score","metadata","content"} shape unmodified."""
        mgr = ESVectorStoreManager()
        mgr.vectorstore = MagicMock()
        mgr.embedding_model = MagicMock()
        mgr.embedding_model.embed_query.return_value = [0.1, 0.2]
        mgr.vectorstore._collection.query.return_value = {
            "ids": [["a"]], "distances": [[0.1]],
            "metadatas": [[{"license_type": "y"}]], "documents": [["content"]],
        }

        out = mgr.query_live_with_ids("คำถาม", k=5)

        assert out == [{"id": "a", "score": pytest.approx(0.9), "metadata": {"license_type": "y"}, "content": "content"}]

    def test_empty_text_returns_empty_list_no_embed_call(self):
        mgr = ESVectorStoreManager()
        mgr.vectorstore = MagicMock()
        mgr.embedding_model = MagicMock()

        assert mgr.query_live_with_ids("   ", k=5) == []
        mgr.embedding_model.embed_query.assert_not_called()

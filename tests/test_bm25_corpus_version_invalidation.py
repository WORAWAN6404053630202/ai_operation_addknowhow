"""Tests for the 2026-10 fix (core2-C5) to utils/hybrid_retriever.py's
_get_bm25_index — it only ever compared Chroma doc count to decide whether
to rebuild, and never checked
service.local_vector_store.get_corpus_version(), the mechanism every other
corpus-derived cache in the project (9 of them, across all 3 personas) was
already wired to.

The gap this closes: the PDF supersede resolver's "update" decisions
archive one doc and add one new doc in the same operation — net count is
unchanged, so the old count-only check kept serving the stale BM25 index
(with the retired doc's text still in it) for the rest of the process's
life. No real Chroma/pythainlp dependency — rank_bm25's tokenizer falls
back to whitespace split when pythainlp is unavailable (see _tokenize),
and the fake collection below is a plain stand-in object.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from utils import hybrid_retriever as hr


class _FakeCollection:
    def __init__(self, name: str, docs: list[Document]):
        self.name = name
        self._docs = docs

    def count(self) -> int:
        return len(self._docs)

    def get(self, include=None):
        return {
            "documents": [d.page_content for d in self._docs],
            "metadatas": [d.metadata for d in self._docs],
        }


def _fake_vectorstore(coll: _FakeCollection):
    vs = MagicMock()
    vs._collection = coll
    return vs


@pytest.fixture(autouse=True)
def _clear_bm25_caches():
    """The module-level _INDEX_CACHE/_COUNT_CACHE are process-global
    singletons — clear them before and after each test so tests don't leak
    state into each other (mirrors invalidate_bm25_cache()'s own effect)."""
    hr.invalidate_bm25_cache()
    yield
    hr.invalidate_bm25_cache()


@pytest.mark.unit
class TestBm25CorpusVersionInvalidation:
    def test_same_count_and_version_reuses_cached_index(self):
        docs = [Document(page_content="เอกสาร A", metadata={})]
        coll = _FakeCollection("c1", docs)
        vs = _fake_vectorstore(coll)
        with patch("service.local_vector_store.get_corpus_version", return_value=5):
            idx1 = hr._get_bm25_index(vs)
            idx2 = hr._get_bm25_index(vs)
        assert idx1 is idx2

    def test_corpus_version_bump_with_unchanged_count_rebuilds_index(self):
        """The actual bug this closes: a PDF supersede 'update' decision
        archives one doc and adds one new doc — count stays the same, but
        the corpus-version counter is bumped. The old doc-count-only check
        would wrongly keep serving the stale index here."""
        old_docs = [Document(page_content="เอกสารเก่า", metadata={"id": "old"})]
        new_docs = [Document(page_content="เอกสารใหม่", metadata={"id": "new"})]
        coll = _FakeCollection("c1", old_docs)
        vs = _fake_vectorstore(coll)

        with patch("service.local_vector_store.get_corpus_version", return_value=1):
            idx1 = hr._get_bm25_index(vs)
        assert idx1 is not None
        assert idx1._docs[0].page_content == "เอกสารเก่า"

        # Simulate the supersede swap: same doc count, different content,
        # corpus_version bumped.
        coll._docs = new_docs
        with patch("service.local_vector_store.get_corpus_version", return_value=2):
            idx2 = hr._get_bm25_index(vs)

        assert idx2 is not idx1
        assert idx2._docs[0].page_content == "เอกสารใหม่"
        assert idx2.corpus_version == 2

    def test_get_corpus_version_failure_does_not_crash_lookup(self):
        docs = [Document(page_content="เอกสาร A", metadata={})]
        coll = _FakeCollection("c1", docs)
        vs = _fake_vectorstore(coll)
        with patch("service.local_vector_store.get_corpus_version", side_effect=RuntimeError("boom")):
            idx = hr._get_bm25_index(vs)
        assert idx is not None

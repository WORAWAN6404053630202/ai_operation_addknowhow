"""
Unit tests for the corpus-version cache-invalidation mechanism (core-C4,
2026-10 pure-PDF-project audit).

Background: persona_supervisor.py / persona_academic.py / persona_practical.py
each cache corpus-derived metadata (distinct main_topic/sub_topic/department/
operation_topic/operation_by_department values, and the Practical topic menu)
on the persona instance for the life of the process. Since route_v1.py builds
these persona services exactly once at module-import time (a long-lived
singleton — see PersonaSupervisor(retriever=retriever) there), "cached after
first call" meant "cached forever until restart" — a PDF direct-ingest or a
Sheet re-ingest that changed the set of topics/departments was invisible to
an already-running bot process. service/local_vector_store.py now exposes a
get_corpus_version()/_bump_corpus_version() pair (a plain counter file next
to the Chroma persist dir, so it also works across two separate OS processes
sharing the same LOCAL_VECTOR_DIR, e.g. the SQS consumer and the web app on
the PDF project's shared instance) that every cache site checks before
trusting its cached value.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import conf  # noqa: E402
from service.local_vector_store import LocalVectorStoreManager  # noqa: E402


@pytest.mark.unit
class TestCorpusVersionMechanics:
    def test_starts_at_zero_when_no_version_file_exists_yet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(conf, "LOCAL_VECTOR_DIR", str(tmp_path))
        mgr = LocalVectorStoreManager()
        assert mgr.get_corpus_version() == 0

    def test_bump_increments_monotonically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(conf, "LOCAL_VECTOR_DIR", str(tmp_path))
        mgr = LocalVectorStoreManager()
        mgr._bump_corpus_version()
        assert mgr.get_corpus_version() == 1
        mgr._bump_corpus_version()
        mgr._bump_corpus_version()
        assert mgr.get_corpus_version() == 3

    def test_visible_across_separate_manager_instances_sharing_a_dir(self, tmp_path, monkeypatch):
        """Must behave the same whether the bump and the read happen in the
        same process or two different ones — simulated here with two
        separate LocalVectorStoreManager instances pointed at the same
        on-disk LOCAL_VECTOR_DIR, mirroring the SQS consumer + web app
        sharing one instance's local Chroma directory."""
        monkeypatch.setattr(conf, "LOCAL_VECTOR_DIR", str(tmp_path))
        writer = LocalVectorStoreManager()
        reader = LocalVectorStoreManager()
        assert reader.get_corpus_version() == 0
        writer._bump_corpus_version()
        assert reader.get_corpus_version() == 1

    def test_corrupt_version_file_fails_open_to_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(conf, "LOCAL_VECTOR_DIR", str(tmp_path))
        mgr = LocalVectorStoreManager()
        (tmp_path / ".corpus_version").write_text("not-a-number")
        assert mgr.get_corpus_version() == 0


@pytest.mark.unit
class TestPracticalPersonaCacheRespectsCorpusVersion:
    """_get_nonreg_main_topics (persona_practical.py) used to cache its
    result forever after the first call — this is the regression test for
    that specific fix, exercised directly rather than through the full
    message-handling pipeline."""

    @pytest.fixture
    def practical(self):
        from model.persona_practical import PracticalPersonaService
        return PracticalPersonaService(retriever=MagicMock())

    @staticmethod
    def _vectorstore_with_topics(*topic_names):
        vs = MagicMock()
        vs._collection.get.return_value = {
            "metadatas": [{"main_topic": name} for name in topic_names]
        }
        return vs

    def test_recomputes_after_corpus_version_changes(self, practical, monkeypatch):
        calls = {"v": 0}

        def fake_get_corpus_version():
            return calls["v"]

        monkeypatch.setattr(
            "service.local_vector_store.get_corpus_version", fake_get_corpus_version
        )

        vs1 = self._vectorstore_with_topics("หมวดหมู่การตลาดออนไลน์สำหรับร้านอาหาร")
        first = practical._get_nonreg_main_topics(vs1)
        assert first == ["หมวดหมู่การตลาดออนไลน์สำหรับร้านอาหาร"]

        # Corpus changes (e.g. a PDF ingest added a new chapter) — bump the version.
        calls["v"] = 1
        vs2 = self._vectorstore_with_topics("หมวดหมู่กลยุทธ์ด้านราคาสำหรับร้านอาหาร")
        second = practical._get_nonreg_main_topics(vs2)
        assert second == ["หมวดหมู่กลยุทธ์ด้านราคาสำหรับร้านอาหาร"], (
            "Cache was not invalidated after the corpus version changed — "
            f"still returned the stale value instead of recomputing: {second!r}"
        )

    def test_reuses_cache_when_corpus_version_is_unchanged(self, practical, monkeypatch):
        monkeypatch.setattr(
            "service.local_vector_store.get_corpus_version", lambda: 7
        )

        vs1 = self._vectorstore_with_topics("หมวดหมู่การตลาดออนไลน์สำหรับร้านอาหาร")
        first = practical._get_nonreg_main_topics(vs1)

        vs2 = self._vectorstore_with_topics("หมวดหมู่อื่นที่ไม่ควรถูกอ่านเพราะ cache ยังไม่หมดอายุ")
        second = practical._get_nonreg_main_topics(vs2)

        assert second == first, "Cache should be reused when the corpus version hasn't changed"
        vs2._collection.get.assert_not_called()

"""Tests for LocalVectorStoreManager's archive_and_remove / list_archived_docs
/ restore_from_archive (service/local_vector_store.py, added 2026-10 for the
PDF content supersede resolver's soft-delete mechanics). Covers the ordering
guarantee (archive-add must happen BEFORE live-delete, and a failed
archive-add must abort without touching live) and the restore round-trip.

Mocked tests here return `embeddings` as a real numpy.ndarray (2026-10
fix), not a plain Python list — the original mocks used a plain list, which
masked a 100%-reproducible crash: real Chroma always returns `embeddings`
(when requested) as an ndarray, and `ndarray or []` raises ValueError
unconditionally (every size, not just >1 element — numpy has no
well-defined truthiness for it at all here). See
_materialize_embeddings()'s docstring in local_vector_store.py. A second,
real-Chroma test class at the bottom of this file (TestArchiveRoundTripRealChroma)
doesn't depend on a mock's return type being realistic at all — add it to
guard against this exact class of bug recurring even if these mocks ever
drift out of sync with Chroma's real behavior again.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from service.local_vector_store import LocalVectorStoreManager  # noqa: E402


def _manager_with_mocked_stores():
    mgr = LocalVectorStoreManager()
    mgr.embedding_model = MagicMock()
    mgr.vectorstore = MagicMock()
    mgr.vectorstore._collection.get.return_value = {
        "ids": ["doc-1"],
        "documents": ["เนื้อหาเดิม"],
        "metadatas": [{"license_type": "ทะเบียนนายจ้าง", "department": "สำนักงานประกันสังคม"}],
        "embeddings": np.array([[0.1, 0.2, 0.3]]),
    }
    archive_store = MagicMock()
    mgr._archive_store = archive_store
    return mgr, archive_store


@pytest.mark.unit
class TestArchiveAndRemove:
    def test_empty_ids_is_noop(self):
        mgr, archive_store = _manager_with_mocked_stores()
        assert mgr.archive_and_remove([], reason="test") == 0
        archive_store._collection.add.assert_not_called()
        mgr.vectorstore._collection.delete.assert_not_called()

    def test_archive_add_happens_before_live_delete(self):
        mgr, archive_store = _manager_with_mocked_stores()
        call_order = []
        archive_store._collection.add.side_effect = lambda **kw: call_order.append("archive_add")
        mgr.vectorstore._collection.delete.side_effect = lambda **kw: call_order.append("live_delete")

        moved = mgr.archive_and_remove(["doc-1"], reason="superseded by new_law.pdf")

        assert moved == 1
        assert call_order == ["archive_add", "live_delete"]

    def test_archived_doc_is_stamped_with_superseded_metadata(self):
        mgr, archive_store = _manager_with_mocked_stores()
        mgr.archive_and_remove(["doc-1"], reason="superseded by new_law.pdf")

        _, kwargs = archive_store._collection.add.call_args
        stamped = kwargs["metadatas"][0]
        assert stamped["superseded_reason"] == "superseded by new_law.pdf"
        assert "superseded_at" in stamped
        # Original fields survive the stamp
        assert stamped["license_type"] == "ทะเบียนนายจ้าง"

    def test_archive_add_failure_aborts_without_deleting_live(self):
        mgr, archive_store = _manager_with_mocked_stores()
        archive_store._collection.add.side_effect = RuntimeError("archive write failed")

        with pytest.raises(RuntimeError):
            mgr.archive_and_remove(["doc-1"], reason="test")

        mgr.vectorstore._collection.delete.assert_not_called()

    def test_reuses_fetched_embeddings_no_extra_embedding_call(self):
        mgr, archive_store = _manager_with_mocked_stores()
        mgr.archive_and_remove(["doc-1"], reason="test")
        mgr.embedding_model.embed_documents.assert_not_called()
        _, kwargs = archive_store._collection.add.call_args
        # kwargs["embeddings"] is a list of 1D ndarrays (see
        # _materialize_embeddings) — a direct `==` against a nested plain
        # list hits the exact numpy-truthiness ValueError this whole fix is
        # about, so compare via np.allclose instead.
        assert len(kwargs["embeddings"]) == 1
        assert np.allclose(kwargs["embeddings"][0], [0.1, 0.2, 0.3])

    def test_no_ids_found_in_live_is_noop(self):
        mgr, archive_store = _manager_with_mocked_stores()
        mgr.vectorstore._collection.get.return_value = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        assert mgr.archive_and_remove(["missing-id"], reason="test") == 0
        archive_store._collection.add.assert_not_called()


@pytest.mark.unit
class TestRestoreFromArchive:
    def test_restore_strips_superseded_stamp_and_readds_to_live(self):
        mgr, archive_store = _manager_with_mocked_stores()
        archive_store._collection.get.return_value = {
            "ids": ["doc-1"],
            "documents": ["เนื้อหาเดิม"],
            "metadatas": [{
                "license_type": "ทะเบียนนายจ้าง",
                "superseded_at": 123.0,
                "superseded_reason": "superseded by new_law.pdf",
            }],
            "embeddings": np.array([[0.1, 0.2, 0.3]]),
        }

        restored = mgr.restore_from_archive(["doc-1"])

        assert restored == 1
        _, kwargs = mgr.vectorstore._collection.add.call_args
        restored_meta = kwargs["metadatas"][0]
        assert "superseded_at" not in restored_meta
        assert "superseded_reason" not in restored_meta
        assert restored_meta["license_type"] == "ทะเบียนนายจ้าง"
        archive_store._collection.delete.assert_called_once_with(ids=["doc-1"])

    def test_restore_of_missing_id_is_noop(self):
        mgr, archive_store = _manager_with_mocked_stores()
        archive_store._collection.get.return_value = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        assert mgr.restore_from_archive(["missing-id"]) == 0
        mgr.vectorstore._collection.add.assert_not_called()


@pytest.mark.unit
class TestListArchivedDocs:
    def test_list_filters_by_pattern(self):
        mgr, archive_store = _manager_with_mocked_stores()
        archive_store._collection.get.return_value = {
            "ids": ["a", "b"],
            "metadatas": [{"source_file": "labor_law.pdf"}, {"source_file": "tax_law.pdf"}],
            "documents": ["x", "y"],
        }
        result = mgr.list_archived_docs(pattern="labor")
        assert len(result) == 1
        assert result[0]["id"] == "a"

    def test_results_sorted_newest_archived_first_by_default(self):
        """2026-10 fix — before this, callers got an unsorted dump straight
        from Chroma's own internal ordering."""
        mgr, archive_store = _manager_with_mocked_stores()
        archive_store._collection.get.return_value = {
            "ids": ["oldest", "newest", "middle"],
            # superseded_at comes back as a STRING — every value is run
            # through _stringify_metadata before being written to Chroma.
            "metadatas": [
                {"source_file": "a.pdf", "superseded_at": "100"},
                {"source_file": "b.pdf", "superseded_at": "300"},
                {"source_file": "c.pdf", "superseded_at": "200"},
            ],
            "documents": ["x", "y", "z"],
        }
        result = mgr.list_archived_docs()
        assert [r["id"] for r in result] == ["newest", "middle", "oldest"]

    def test_since_epoch_filters_out_older_docs(self):
        mgr, archive_store = _manager_with_mocked_stores()
        archive_store._collection.get.return_value = {
            "ids": ["old", "recent"],
            "metadatas": [
                {"source_file": "a.pdf", "superseded_at": "100"},
                {"source_file": "b.pdf", "superseded_at": "1000"},
            ],
            "documents": ["x", "y"],
        }
        result = mgr.list_archived_docs(since_epoch=500)
        assert len(result) == 1
        assert result[0]["id"] == "recent"

    def test_malformed_superseded_at_sorts_as_oldest_not_raise(self):
        mgr, archive_store = _manager_with_mocked_stores()
        archive_store._collection.get.return_value = {
            "ids": ["malformed", "valid"],
            "metadatas": [
                {"source_file": "a.pdf", "superseded_at": "not-a-number"},
                {"source_file": "b.pdf", "superseded_at": "50"},
            ],
            "documents": ["x", "y"],
        }
        result = mgr.list_archived_docs()  # must not raise
        assert [r["id"] for r in result] == ["valid", "malformed"]


class _CollectionHolder:
    """Minimal stand-in for langchain's Chroma wrapper — archive_and_remove/
    restore_from_archive only ever touch `.vectorstore._collection` and
    `.vectorstore.persist()` (the latter wrapped in try/except: pass by the
    real code, so omitting persist() here is fine — AttributeError gets
    swallowed the same way a real persist() failure would be)."""
    def __init__(self, collection):
        self._collection = collection


@pytest.mark.unit
class TestArchiveRoundTripRealChroma:
    """2026-10: the mocked tests above use MagicMock/plain-dict returns,
    which can only ever be as realistic as whoever wrote the mock remembered
    to make them — exactly how the original embeddings-as-plain-list bug
    went undetected despite full test coverage of the ordering logic. These
    tests run archive_and_remove/restore_from_archive against a REAL
    chromadb.EphemeralClient() (in-memory, no network, no disk, no OpenRouter
    calls — embedding_model is never touched because both functions reuse
    already-fetched embeddings) so the test can't be fooled by an
    unrealistic mock shape ever again."""

    def _manager_with_real_chroma(self):
        chromadb = pytest.importorskip("chromadb")
        import uuid
        # EphemeralClient() shares collection namespace across instances
        # within one process — a fixed collection name collides between
        # tests ("already exists"). Unique suffix per call sidesteps it
        # without needing explicit teardown.
        suffix = uuid.uuid4().hex[:8]
        client = chromadb.EphemeralClient()
        live = client.create_collection(f"live-test-{suffix}")
        archive = client.create_collection(f"archive-test-{suffix}")
        live.add(
            ids=["doc-1", "doc-2"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            documents=["เนื้อหา 1", "เนื้อหา 2"],
            metadatas=[{"license_type": "ทะเบียนนายจ้าง"}, {"license_type": "ภพ.20"}],
        )
        mgr = LocalVectorStoreManager()
        mgr.embedding_model = MagicMock()
        mgr.vectorstore = _CollectionHolder(live)
        mgr._archive_store = _CollectionHolder(archive)
        return mgr, live, archive

    def test_archive_and_remove_does_not_raise_against_real_chroma(self):
        mgr, live, archive = self._manager_with_real_chroma()
        moved = mgr.archive_and_remove(["doc-1"], reason="test update")  # must not raise ValueError
        assert moved == 1
        assert live.count() == 1
        assert archive.count() == 1
        mgr.embedding_model.embed_documents.assert_not_called()  # reused fetched embeddings, no new API call

    def test_archived_doc_keeps_its_real_embedding_not_a_placeholder(self):
        mgr, live, archive = self._manager_with_real_chroma()
        mgr.archive_and_remove(["doc-1"], reason="test update")
        back = archive.get(ids=["doc-1"], include=["embeddings"])
        archived_emb = list(back["embeddings"])[0]
        assert np.allclose(archived_emb, [0.1, 0.2, 0.3])

    def test_restore_round_trips_through_real_chroma_without_raising(self):
        mgr, live, archive = self._manager_with_real_chroma()
        mgr.archive_and_remove(["doc-1"], reason="test update")
        assert live.count() == 1

        restored = mgr.restore_from_archive(["doc-1"])  # must not raise ValueError

        assert restored == 1
        assert live.count() == 2
        assert archive.count() == 0
        back = live.get(ids=["doc-1"], include=["embeddings", "metadatas"])
        assert np.allclose(list(back["embeddings"])[0], [0.1, 0.2, 0.3])
        assert "superseded_at" not in back["metadatas"][0]


@pytest.mark.unit
class TestSnapshotAndRestoreArchive:
    """PDF2-C3 (2026-10): the archive collection lives in the SAME
    persist_directory as live — create_vectorstore(reset=True) (called by
    scripts/ingest_local.py --confirm) renames the whole persist_dir away
    and deletes it, which used to destroy the archive's on-disk data too,
    since nothing snapshotted it first (unlike pdf_auto live docs, which
    Source E already carried forward). snapshot_archive/
    restore_archive_snapshot are the carry-forward mechanism for the
    archive specifically."""

    def test_snapshot_empty_archive_returns_empty_lists(self):
        mgr, archive_store = _manager_with_mocked_stores()
        archive_store._collection.get.return_value = {
            "ids": [], "documents": [], "metadatas": [], "embeddings": np.array([]),
        }
        snapshot = mgr.snapshot_archive()
        assert snapshot["ids"] == []

    def test_restore_empty_snapshot_is_noop(self):
        mgr, archive_store = _manager_with_mocked_stores()
        assert mgr.restore_archive_snapshot({"ids": []}) == 0
        archive_store._collection.add.assert_not_called()

    def test_restore_writes_snapshot_back_into_archive_collection(self):
        mgr, archive_store = _manager_with_mocked_stores()
        snapshot = {
            "ids": ["archived-1"],
            "documents": ["เนื้อหาที่ถูก archive"],
            "metadatas": [{"superseded_at": 123.0, "superseded_reason": "test"}],
            # A raw ndarray here (not yet materialized) — proves
            # restore_archive_snapshot handles it safely rather than
            # tripping the ndarray-truthiness bug this same sweep fixed in
            # archive_and_remove/restore_from_archive.
            "embeddings": np.array([[0.1, 0.2, 0.3]]),
        }
        restored = mgr.restore_archive_snapshot(snapshot)
        assert restored == 1
        call_kwargs = archive_store._collection.add.call_args.kwargs
        assert call_kwargs["ids"] == ["archived-1"]
        assert call_kwargs["documents"] == ["เนื้อหาที่ถูก archive"]
        assert call_kwargs["metadatas"] == [{"superseded_at": 123.0, "superseded_reason": "test"}]
        assert np.allclose(call_kwargs["embeddings"][0], [0.1, 0.2, 0.3])


@pytest.mark.unit
class TestArchiveSurvivesRebuildRealChroma:
    """2026-10: proves the actual bug scenario end-to-end against a REAL
    chromadb.EphemeralClient() — a doc sitting in the archive collection,
    snapshotted, the archive collection "destroyed" (simulating what
    create_vectorstore(reset=True) does to the physical directory) and
    recreated fresh, then restored — must survive with its real embedding
    intact, not a placeholder."""

    def test_archived_doc_survives_simulated_rebuild(self):
        chromadb = pytest.importorskip("chromadb")
        import uuid
        suffix = uuid.uuid4().hex[:8]
        client = chromadb.EphemeralClient()
        old_archive = client.create_collection(f"archive-old-{suffix}")
        old_archive.add(
            ids=["archived-doc-1"],
            embeddings=[[0.7, 0.8, 0.9]],
            documents=["เนื้อหาที่ถูก supersede ไปแล้ว"],
            metadatas=[{"superseded_at": 1000.0, "superseded_reason": "updated by newer PDF"}],
        )

        mgr = LocalVectorStoreManager()
        mgr.embedding_model = MagicMock()
        mgr._archive_store = _CollectionHolder(old_archive)

        snapshot = mgr.snapshot_archive()
        assert snapshot["ids"] == ["archived-doc-1"]

        # Simulate create_vectorstore(reset=True) destroying the old
        # physical directory (and therefore the old archive collection)
        # and the caller reconnecting _archive_store to a fresh one at the
        # new path, exactly as ingest_local.py now does.
        client.delete_collection(f"archive-old-{suffix}")
        new_archive = client.create_collection(f"archive-new-{suffix}")
        mgr._archive_store = _CollectionHolder(new_archive)

        restored = mgr.restore_archive_snapshot(snapshot)
        assert restored == 1
        assert new_archive.count() == 1
        back = new_archive.get(ids=["archived-doc-1"], include=["embeddings", "metadatas"])
        assert np.allclose(list(back["embeddings"])[0], [0.7, 0.8, 0.9])
        assert back["metadatas"][0]["superseded_reason"] == "updated by newer PDF"

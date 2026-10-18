"""Tests for LocalVectorStoreManager's archive_and_remove / list_archived_docs
/ restore_from_archive (service/local_vector_store.py, added 2026-10 for the
PDF content supersede resolver's soft-delete mechanics). Covers the ordering
guarantee (archive-add must happen BEFORE live-delete, and a failed
archive-add must abort without touching live) and the restore round-trip.
No real Chroma/network — vectorstore/_collection are MagicMocks.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

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
        "embeddings": [[0.1, 0.2, 0.3]],
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
        assert kwargs["embeddings"] == [[0.1, 0.2, 0.3]]

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
            "embeddings": [[0.1, 0.2, 0.3]],
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

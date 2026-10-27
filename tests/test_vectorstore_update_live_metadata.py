"""Tests for LocalVectorStoreManager.update_live_metadata (service/
local_vector_store.py, 2026-10) — in-place patch of already-live documents,
added for code/scripts/pdf_supersede_backfill.py's one-time cleanup of docs
stuck permanently blocked under the OLD pdf_supersede_resolver.py behavior
(see that module's field_merges redesign). The live ingest path never needs
this method; only the backfill does, since there the "new" sibling doc is
already live rather than about to be written.
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


def _manager_with_mocked_vectorstore():
    mgr = LocalVectorStoreManager()
    mgr.vectorstore = MagicMock()
    mgr.vectorstore._collection.get.return_value = {
        "ids": ["doc-1"],
        "documents": ["เนื้อหาเดิม"],
        "metadatas": [{"operation_duration": "ภายใน 7 วัน", "fees": "100 บาท"}],
    }
    mgr.embedding_model = MagicMock()
    mgr.embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    return mgr


@pytest.mark.unit
class TestUpdateLiveMetadata:
    def test_appends_to_existing_field_value_not_replacing_it(self):
        mgr = _manager_with_mocked_vectorstore()
        mgr.update_live_metadata(["doc-1"], {"operation_duration": " (เพิ่มเติม: กรณีพิเศษ 15 วัน)"})
        call_kwargs = mgr.vectorstore._collection.update.call_args.kwargs
        assert call_kwargs["metadatas"][0]["operation_duration"] == "ภายใน 7 วัน (เพิ่มเติม: กรณีพิเศษ 15 วัน)"
        # A field with no update must be left completely untouched.
        assert call_kwargs["metadatas"][0]["fees"] == "100 บาท"

    def test_appends_the_same_note_to_page_content_too(self):
        """Not just metadata — otherwise the merged detail would never be
        part of what similarity search actually matches against."""
        mgr = _manager_with_mocked_vectorstore()
        mgr.update_live_metadata(["doc-1"], {"operation_duration": "กรณีพิเศษ 15 วัน"})
        call_kwargs = mgr.vectorstore._collection.update.call_args.kwargs
        assert "กรณีพิเศษ 15 วัน" in call_kwargs["documents"][0]
        assert "เนื้อหาเดิม" in call_kwargs["documents"][0]

    def test_multiple_fields_all_applied_in_one_call(self):
        mgr = _manager_with_mocked_vectorstore()
        mgr.update_live_metadata(["doc-1"], {
            "operation_duration": " extra-duration-note",
            "fees": " extra-fee-note",
        })
        call_kwargs = mgr.vectorstore._collection.update.call_args.kwargs
        meta = call_kwargs["metadatas"][0]
        assert "extra-duration-note" in meta["operation_duration"]
        assert "extra-fee-note" in meta["fees"]

    def test_returns_count_of_ids_actually_found_live(self):
        mgr = _manager_with_mocked_vectorstore()
        result = mgr.update_live_metadata(["doc-1"], {"fees": " note"})
        assert result == 1

    def test_no_matching_live_ids_is_a_safe_noop(self):
        mgr = _manager_with_mocked_vectorstore()
        mgr.vectorstore._collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        result = mgr.update_live_metadata(["doc-does-not-exist"], {"fees": " note"})
        assert result == 0
        mgr.vectorstore._collection.update.assert_not_called()

    def test_empty_doc_ids_or_empty_field_updates_is_a_safe_noop_without_touching_chroma(self):
        mgr = _manager_with_mocked_vectorstore()
        assert mgr.update_live_metadata([], {"fees": " note"}) == 0
        assert mgr.update_live_metadata(["doc-1"], {}) == 0
        mgr.vectorstore._collection.get.assert_not_called()

    def test_applies_identically_across_multiple_sibling_doc_ids(self):
        """The real use case: several sub-chunks of one logical topic each
        carry their own independent copy of the structured fields
        (build_merged_documents' own design) — a merge must reach every
        one of them when the caller passes multiple ids, not just one."""
        mgr = LocalVectorStoreManager()
        mgr.vectorstore = MagicMock()
        mgr.vectorstore._collection.get.return_value = {
            "ids": ["chunk-1", "chunk-2"],
            "documents": ["เนื้อหาส่วนที่ 1", "เนื้อหาส่วนที่ 2"],
            "metadatas": [{"fees": "100 บาท"}, {"fees": "100 บาท"}],
        }
        mgr.embedding_model = MagicMock()
        mgr.embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mgr.update_live_metadata(["chunk-1", "chunk-2"], {"fees": " (เพิ่มเติม)"})
        call_kwargs = mgr.vectorstore._collection.update.call_args.kwargs
        assert call_kwargs["ids"] == ["chunk-1", "chunk-2"]
        assert all(m["fees"] == "100 บาท (เพิ่มเติม)" for m in call_kwargs["metadatas"])

    def test_embeddings_are_computed_explicitly_not_left_to_chroma_default(self):
        """2026-10 fix (live-crashed on the real corpus's first --confirm
        run): this call used to pass documents= to the raw collection's
        .update() with no embeddings= argument, trusting Chroma to auto-
        embed — but the raw collection falls back to chromadb's OWN bundled
        default embedding function (384-dim) instead of the bge-m3 (1024-
        dim) function LangChain's add_documents() explicitly computes with,
        and chromadb rejected the mismatch outright
        ("expecting embedding with dimension of 1024, got 384"). Pins that
        self.embedding_model.embed_documents() is called explicitly and its
        result is what's actually passed through."""
        mgr = _manager_with_mocked_vectorstore()
        mgr.embedding_model = MagicMock()
        mgr.embedding_model.embed_documents.return_value = [[0.9, 0.8, 0.7]]
        mgr.update_live_metadata(["doc-1"], {"fees": " note"})
        mgr.embedding_model.embed_documents.assert_called_once()
        call_kwargs = mgr.vectorstore._collection.update.call_args.kwargs
        assert call_kwargs["embeddings"] == [[0.9, 0.8, 0.7]]


class _RealChromaCollectionHolder:
    """Minimal stand-in for langchain's Chroma wrapper — update_live_metadata
    only ever touches `.vectorstore._collection` and `.vectorstore.persist()`
    (the latter wrapped in try/except: pass by the real code)."""
    def __init__(self, collection):
        self._collection = collection


@pytest.mark.unit
class TestUpdateLiveMetadataRealChroma:
    """2026-10: the mocked tests above use MagicMock, which can only ever be
    as realistic as whoever wrote the mock remembered to make it — exactly
    how the embeddings-dimension-mismatch bug above went undetected despite
    full mocked test coverage of the merge/append logic. This runs against
    a REAL chromadb.EphemeralClient() (in-memory, no network, no disk) so a
    future regression of the same class (anything that lets a write reach
    the raw collection without an explicit, correctly-shaped embeddings=)
    can't be masked by an unrealistic mock again."""

    def _manager_with_real_chroma(self, dim: int = 3):
        chromadb = pytest.importorskip("chromadb")
        import uuid
        suffix = uuid.uuid4().hex[:8]
        client = chromadb.EphemeralClient()
        live = client.create_collection(f"live-update-test-{suffix}")
        live.add(
            ids=["doc-1"],
            embeddings=[[0.1] * dim],
            documents=["เนื้อหาเดิม"],
            metadatas=[{"fees": "100 บาท"}],
        )
        mgr = LocalVectorStoreManager()
        mgr.embedding_model = MagicMock()
        mgr.embedding_model.embed_documents.return_value = [[0.2] * dim]
        mgr.vectorstore = _RealChromaCollectionHolder(live)
        return mgr, live

    def test_update_live_metadata_does_not_raise_against_real_chroma(self):
        mgr, live = self._manager_with_real_chroma()
        result = mgr.update_live_metadata(["doc-1"], {"fees": " (เพิ่มเติม)"})  # must not raise InvalidArgumentError
        assert result == 1
        back = live.get(ids=["doc-1"], include=["metadatas", "embeddings"])
        assert back["metadatas"][0]["fees"] == "100 บาท (เพิ่มเติม)"
        assert np.allclose(list(back["embeddings"][0]), [0.2, 0.2, 0.2])

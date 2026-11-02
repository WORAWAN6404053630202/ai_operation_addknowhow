"""
Unit tests for scripts/es_full_reindex.py's pagination/refresh-toggle logic
(.claude/plans/mutable-seeking-wave.md, Round 3 Step 1). Mocks the Chroma
collection and the ES client/manager directly — same "construct real
objects, monkeypatch only the network boundary" approach test_es_vector_
store_facade.py already uses (see its TestArchiveAndRemoveOrdering, which
patches service.es_vector_store.ESCompatCollection the same way).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

CODE_DIR = Path(__file__).parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import service.es_vector_store as es_mod
from scripts.es_full_reindex import _reindex_collection


def _fake_chroma_collection(total: int, batch_size: int):
    """A Chroma collection stub whose .get(limit=, offset=) returns a page
    of `batch_size` fake ids/docs/metas/embeddings starting at `offset`,
    mirroring the real shape _materialize_embeddings expects (a list, not
    a numpy array, so this test doesn't need chromadb installed)."""
    coll = MagicMock()
    coll.count.return_value = total

    def _get(limit, offset, include):
        page_ids = [str(i) for i in range(offset, min(offset + limit, total))]
        return {
            "ids": page_ids,
            "documents": [f"doc-{i}" for i in page_ids],
            "metadatas": [{"license_type": "x"} for _ in page_ids],
            "embeddings": [[0.1, 0.2] for _ in page_ids],
        }

    coll.get.side_effect = _get
    return coll


class _TrackingCollection:
    """Stands in for ESCompatCollection — records every upsert() call's ids
    instead of touching a real ES client."""
    calls = []
    fail_at_offsets = set()

    def __init__(self, client, index_name):
        self.index_name = index_name

    def upsert(self, ids, documents=None, metadatas=None, embeddings=None):
        offset = int(ids[0].split("_")[-1]) if "_" in ids[0] else int(ids[0])
        if offset in self.fail_at_offsets:
            raise RuntimeError(f"simulated ES failure at offset {offset}")
        type(self).calls.append(list(ids))


@pytest.fixture(autouse=True)
def _patch_es_compat_collection(monkeypatch):
    _TrackingCollection.calls = []
    _TrackingCollection.fail_at_offsets = set()
    monkeypatch.setattr(es_mod, "ESCompatCollection", _TrackingCollection)
    yield


@pytest.mark.unit
class TestReindexPagination:
    def test_paginates_until_chroma_exhausted(self):
        chroma_coll = _fake_chroma_collection(total=1200, batch_size=500)
        es_mgr = MagicMock()
        client = MagicMock()
        es_mgr._get_client.return_value = client

        migrated, failed = _reindex_collection(chroma_coll, es_mgr, "thai_test_es", id_prefix="full_", batch_size=500)

        assert migrated == 1200
        assert failed == []
        # 3 batches: offsets 0, 500, 1000 (last one only 200 docs).
        assert len(_TrackingCollection.calls) == 3
        assert len(_TrackingCollection.calls[0]) == 500
        assert len(_TrackingCollection.calls[1]) == 500
        assert len(_TrackingCollection.calls[2]) == 200
        assert _TrackingCollection.calls[0][0] == "full_0"

    def test_toggles_refresh_interval_off_then_on(self):
        chroma_coll = _fake_chroma_collection(total=10, batch_size=500)
        es_mgr = MagicMock()
        client = MagicMock()
        es_mgr._get_client.return_value = client

        _reindex_collection(chroma_coll, es_mgr, "thai_test_es", id_prefix="full_", batch_size=500)

        settings_calls = client.indices.put_settings.call_args_list
        assert settings_calls[0].kwargs["body"] == {"index": {"refresh_interval": "-1"}}
        assert settings_calls[-1].kwargs["body"] == {"index": {"refresh_interval": "1s"}}
        client.indices.refresh.assert_called_once_with(index="thai_test_es")

    def test_refresh_interval_restored_even_if_a_batch_fails(self, monkeypatch):
        """The refresh-interval reset lives in a `finally` — a failed batch
        must not leave the index stuck with refresh disabled forever."""
        import scripts.es_full_reindex as reindex_mod
        monkeypatch.setattr(reindex_mod.time, "sleep", lambda *_: None)
        chroma_coll = _fake_chroma_collection(total=10, batch_size=500)
        _TrackingCollection.fail_at_offsets = {0}
        es_mgr = MagicMock()
        client = MagicMock()
        es_mgr._get_client.return_value = client

        migrated, failed = _reindex_collection(chroma_coll, es_mgr, "thai_test_es", id_prefix="full_", batch_size=500)

        assert migrated == 0
        assert failed == [0]
        settings_calls = client.indices.put_settings.call_args_list
        assert settings_calls[-1].kwargs["body"] == {"index": {"refresh_interval": "1s"}}

    def test_empty_collection_skips_everything(self):
        chroma_coll = _fake_chroma_collection(total=0, batch_size=500)
        es_mgr = MagicMock()
        client = MagicMock()
        es_mgr._get_client.return_value = client

        migrated, failed = _reindex_collection(chroma_coll, es_mgr, "thai_test_es", id_prefix="full_", batch_size=500)

        assert migrated == 0
        assert failed == []
        client.indices.put_settings.assert_not_called()

"""
Unit tests for scripts/es_backfill_tokenized_field.py (.claude/plans/
mutable-seeking-wave.md — "Add PyThaiNLP pre-tokenization to Elasticsearch's
BM25/match leg"). Mocks the ES client directly, same approach as
test_es_full_reindex.py — this script never re-embeds or re-upserts whole
documents, only computes+writes the one new field via partial `_update`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

CODE_DIR = Path(__file__).parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from scripts.es_backfill_tokenized_field import _backfill_index


def _fake_es_client(docs: dict, batch_size: int):
    """`docs` maps id -> {"page_content": ..., "page_content_tokenized": ...}.
    Mirrors real ES PIT + search_after pagination: a frozen, sorted-by-id
    snapshot taken at `open_point_in_time` time, immune to any writes the
    script itself makes during the scan (same guarantee a real PIT gives —
    this is the exact property the live-caught `_doc`-sort bug lacked)."""
    client = MagicMock()
    ids_sorted = sorted(docs.keys())  # frozen "snapshot" order, fixed at PIT-open time

    def _count(index):
        return {"count": len(ids_sorted)}

    def _open_pit(index, keep_alive):
        return {"id": "fake-pit-id"}

    def _search(body):
        assert "pit" in body, "must use PIT, not a plain index= search"
        search_after = body.get("search_after")
        start = 0
        if search_after is not None:
            start = ids_sorted.index(search_after[0]) + 1
        page_ids = ids_sorted[start:start + body["size"]]
        hits = [
            {"_id": i, "_source": dict(docs[i]), "sort": [i]}
            for i in page_ids
        ]
        return {"hits": {"hits": hits}, "pit_id": body["pit"]["id"]}

    client.count.side_effect = _count
    client.open_point_in_time.side_effect = _open_pit
    client.search.side_effect = _search
    return client


@pytest.mark.unit
class TestBackfillPagination:
    def test_updates_docs_missing_the_field(self):
        docs = {
            "1": {"page_content": "การตรวจสอบสภาพอาคาร"},
            "2": {"page_content": "ค่าธรรมเนียม"},
        }
        client = _fake_es_client(docs, batch_size=10)

        updated, skipped, failed = _backfill_index(client, "my_index", batch_size=10)

        assert updated == 2
        assert skipped == 0
        assert failed == []
        update_calls = client.update.call_args_list
        assert len(update_calls) == 2
        sent_ids = {c.kwargs["id"] for c in update_calls}
        assert sent_ids == {"1", "2"}
        for c in update_calls:
            tokenized = c.kwargs["body"]["doc"]["page_content_tokenized"]
            assert isinstance(tokenized, str) and tokenized

    def test_skips_docs_that_already_have_the_field(self):
        """Idempotent: a re-run (or a doc ingested after the field existed)
        must not re-tokenize/re-write docs that already have it."""
        docs = {
            "1": {"page_content": "x", "page_content_tokenized": "x"},
            "2": {"page_content": "การตรวจสอบสภาพอาคาร"},
        }
        client = _fake_es_client(docs, batch_size=10)

        updated, skipped, failed = _backfill_index(client, "my_index", batch_size=10)

        assert updated == 1
        assert skipped == 1
        client.update.assert_called_once()
        assert client.update.call_args.kwargs["id"] == "2"

    def test_paginates_across_multiple_batches(self):
        docs = {str(i): {"page_content": f"doc {i}"} for i in range(1, 6)}
        client = _fake_es_client(docs, batch_size=2)

        updated, skipped, failed = _backfill_index(client, "my_index", batch_size=2)

        assert updated == 5
        assert failed == []
        # 5 docs at batch_size=2 → 3 search() calls (2, 2, 1).
        assert client.search.call_count == 3

    def test_empty_index_does_nothing(self):
        client = _fake_es_client({}, batch_size=10)

        updated, skipped, failed = _backfill_index(client, "my_index", batch_size=10)

        assert updated == 0
        assert skipped == 0
        assert failed == []
        client.update.assert_not_called()

    def test_batch_failure_is_recorded_not_raised(self, monkeypatch):
        import scripts.es_backfill_tokenized_field as mod
        monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
        docs = {"1": {"page_content": "x"}}
        client = _fake_es_client(docs, batch_size=10)
        client.update.side_effect = RuntimeError("simulated ES failure")

        updated, skipped, failed = _backfill_index(client, "my_index", batch_size=10)

        assert updated == 0
        assert len(failed) == 1

    def test_pit_closed_even_when_a_batch_fails(self, monkeypatch):
        """The live-caught bug this PIT rewrite fixes was a silent data-
        completeness bug (missed/duplicated docs), not a crash — but the PIT
        itself must still be released on the ES side even when something
        inside the scan fails, or it leaks until its keep_alive expires."""
        import scripts.es_backfill_tokenized_field as mod
        monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
        docs = {"1": {"page_content": "x"}}
        client = _fake_es_client(docs, batch_size=10)
        client.update.side_effect = RuntimeError("simulated ES failure")

        _backfill_index(client, "my_index", batch_size=10)

        client.close_point_in_time.assert_called_once_with(id="fake-pit-id")

    def test_pit_opened_on_the_target_index(self):
        docs = {"1": {"page_content": "x"}}
        client = _fake_es_client(docs, batch_size=10)

        _backfill_index(client, "my_index", batch_size=10)

        client.open_point_in_time.assert_called_once_with(index="my_index", keep_alive="5m")

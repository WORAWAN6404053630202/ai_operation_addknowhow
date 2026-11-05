# code/scripts/es_backfill_tokenized_field.py
"""Backfills `page_content_tokenized` (utils/thai_tokenize.py,
es_vector_store.py's _index_mapping()) onto documents that were indexed
into Elasticsearch BEFORE this field existed. Unlike es_full_reindex.py,
this does NOT re-embed or re-upsert whole documents — it reads
`page_content` from ES itself, computes the tokenized copy locally (free,
no OpenRouter calls), and writes back only that one field via a partial
`_update`, leaving embeddings/every other field untouched.

Paginates with a Point-in-Time (PIT) + `search_after` rather than relying
on ESCompatCollection.get()'s flat `size: 10000` cap — that cap happens to
cover today's ~869-doc corpus, but scale was the whole point of migrating
to Elasticsearch, so this script doesn't bake in the same ceiling.

**Live-caught bug (2026-11)**: the first version sorted by plain `_doc`
(no PIT) — since this script WRITES to the same index it's scanning, the
underlying Lucene segment this process reads from shifted mid-scan, and
`_doc`-order pagination silently re-visited some documents while skipping
others entirely (confirmed live: 274 of 869 docs never got the new field
on the first run, while others were counted twice). PIT freezes a
consistent snapshot for the whole scan's lifetime regardless of concurrent
writes to the live index — the documented correct fix for exactly this
"paginate while writing" scenario.

Usage (same container-exec shape as es_full_reindex.py):

    python3 code/scripts/es_backfill_tokenized_field.py
    python3 code/scripts/es_backfill_tokenized_field.py --index thai_food_business_v3_es_archive
    python3 code/scripts/es_backfill_tokenized_field.py --batch-size 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
code_dir = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

_BATCH_SIZE = 500
_MAX_BATCH_ATTEMPTS = 3


def _backfill_index(client, index_name: str, batch_size: int = _BATCH_SIZE):
    from utils.thai_tokenize import tokenize_thai_for_search

    total = client.count(index=index_name).get("count", 0)
    print(f"[{index_name}] {total} docs to check")
    if total == 0:
        return 0, 0, []

    updated = 0
    skipped = 0
    failed_batches = []
    search_after = None
    pit_id = client.open_point_in_time(index=index_name, keep_alive="5m")["id"]
    try:
        while True:
            body = {
                "size": batch_size,
                "pit": {"id": pit_id, "keep_alive": "5m"},
                # `_shard_doc` is the documented tiebreaker ES adds
                # automatically for PIT + search_after when no other sort
                # is needed — unlike plain `_doc` (no PIT), this stays
                # correct even though this same process is concurrently
                # WRITING to the index it's scanning (PIT freezes the view).
                "sort": [{"_shard_doc": "asc"}],
                "_source": ["page_content", "page_content_tokenized"],
            }
            if search_after is not None:
                body["search_after"] = search_after
            # PIT carries its own index context — no `index=` kwarg here.
            resp = client.search(body=body)
            pit_id = resp.get("pit_id", pit_id)  # PIT id can rotate; always use the latest
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                break

            for attempt in range(1, _MAX_BATCH_ATTEMPTS + 1):
                try:
                    for h in hits:
                        src = h.get("_source", {}) or {}
                        # Idempotent: a doc that already has the field (e.g. a
                        # re-run, or a doc ingested after this feature shipped)
                        # is left untouched rather than re-tokenized for nothing.
                        if src.get("page_content_tokenized"):
                            skipped += 1
                            continue
                        tokenized = tokenize_thai_for_search(src.get("page_content", ""))
                        client.update(
                            index=index_name, id=h["_id"],
                            body={"doc": {"page_content_tokenized": tokenized}},
                        )
                        updated += 1
                    print(f"  batch (after {search_after}): updated={updated} skipped={skipped} so far")
                    break
                except Exception as exc:
                    print(f"  batch (after {search_after}) attempt {attempt}/{_MAX_BATCH_ATTEMPTS} failed: {exc}")
                    if attempt == _MAX_BATCH_ATTEMPTS:
                        failed_batches.append(search_after)
                    else:
                        time.sleep(2 * attempt)

            search_after = hits[-1]["sort"]
            if len(hits) < batch_size:
                break
    finally:
        try:
            client.close_point_in_time(id=pit_id)
        except Exception as exc:
            print(f"  warning: failed to close PIT cleanly: {exc}")

    return updated, skipped, failed_batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default=None, help="ES index name (defaults to conf.ELASTICSEARCH_INDEX_NAME, the live index)")
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE)
    args = parser.parse_args()

    from service.es_vector_store import get_vs_manager

    es_mgr = get_vs_manager()
    es_mgr.connect_to_existing(fail_if_empty=False)
    client = es_mgr._get_client()
    index_name = args.index or es_mgr._live_index

    updated, skipped, failed = _backfill_index(client, index_name, batch_size=args.batch_size)
    print(f"\nDone. updated={updated} skipped(already had field)={skipped} failed_batches={failed}")
    print("Zero OpenRouter cost — only page_content_tokenized was computed/written, no re-embedding.")


if __name__ == "__main__":
    main()

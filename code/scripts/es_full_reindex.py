# code/scripts/es_full_reindex.py
"""Round 3 Phase 5 (.claude/plans/mutable-seeking-wave.md): migrates the REAL
Chroma corpus into Elasticsearch — not the ~58-doc smoke-test sample
`es_phase1_exit_check.py` left behind. Reuses the exact same zero-cost
pattern (pull content+metadata+ALREADY-COMPUTED embeddings from Chroma, no
OpenRouter calls at all) but paginates with `offset` across the whole
collection instead of a single `--limit` pull, since nothing else in this
codebase has ever needed to read more than one Chroma batch at a time.

Usage (same container-exec shape as es_phase1_exit_check.py):

    python3 code/scripts/es_full_reindex.py --chroma-dir /data/local_chroma_v3_copy
    python3 code/scripts/es_full_reindex.py --chroma-dir /data/local_chroma_v3_copy --archive-too
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

import conf
from service.local_vector_store import _materialize_embeddings

_BATCH_SIZE = 500
_MAX_BATCH_ATTEMPTS = 3


def _set_refresh_interval(client, index_name: str, value: str) -> None:
    client.indices.put_settings(index=index_name, body={"index": {"refresh_interval": value}})


def _reindex_collection(chroma_collection, es_mgr, es_index_name: str, id_prefix: str, batch_size: int = _BATCH_SIZE):
    """Paginates `chroma_collection.get(limit=, offset=)` — Chroma supports
    offset-based pagination; this codebase just never exercised it before,
    since every prior full-pull call site omitted `limit` and assumed one
    call returns everything. That assumption doesn't hold for a real corpus,
    hence this loop."""
    from service.es_vector_store import _stringify_metadata

    total = chroma_collection.count()
    print(f"[Chroma] collection has {total} real docs to migrate")
    if total == 0:
        return 0, []

    client = es_mgr._get_client()
    _set_refresh_interval(client, es_index_name, "-1")

    migrated = 0
    failed_batches = []
    offset = 0
    try:
        while offset < total:
            raw = chroma_collection.get(
                limit=batch_size, offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            ids = raw.get("ids") or []
            if not ids:
                break
            docs = raw.get("documents") or []
            metas = raw.get("metadatas") or []
            embeddings = _materialize_embeddings(raw)

            for attempt in range(1, _MAX_BATCH_ATTEMPTS + 1):
                try:
                    # Write through a collection bound explicitly to the
                    # target index — the live vs. archive index aren't the
                    # same thing, don't assume es_mgr.vectorstore points at
                    # whichever one this call is migrating into.
                    from service.es_vector_store import ESCompatCollection
                    ESCompatCollection(client, es_index_name).upsert(
                        ids=[f"{id_prefix}{i}" for i in ids],
                        documents=docs,
                        metadatas=[_stringify_metadata(m) for m in metas],
                        embeddings=embeddings,
                    )
                    migrated += len(ids)
                    print(f"  batch offset={offset}: migrated {len(ids)} docs ({migrated}/{total} so far)")
                    break
                except Exception as exc:
                    print(f"  batch offset={offset} attempt {attempt}/{_MAX_BATCH_ATTEMPTS} failed: {exc}")
                    if attempt == _MAX_BATCH_ATTEMPTS:
                        failed_batches.append(offset)
                    else:
                        time.sleep(2 * attempt)
            offset += batch_size
    finally:
        _set_refresh_interval(client, es_index_name, "1s")
        client.indices.refresh(index=es_index_name)

    return migrated, failed_batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chroma-dir", required=True, help="path to a (copied) Chroma persist directory, read-only")
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE)
    parser.add_argument("--archive-too", action="store_true", help="also migrate the archive (superseded-docs) collection")
    args = parser.parse_args()

    import chromadb
    from service.es_vector_store import get_vs_manager

    chroma_client = chromadb.PersistentClient(path=args.chroma_dir)

    es_mgr = get_vs_manager()
    es_mgr.connect_to_existing(fail_if_empty=False)

    live_coll = chroma_client.get_collection(conf.COLLECTION_NAME)
    migrated, failed = _reindex_collection(
        live_coll, es_mgr, es_mgr._live_index, id_prefix="full_", batch_size=args.batch_size,
    )
    final_es_count = es_mgr.vectorstore._collection.count()
    print(f"\n[LIVE] migrated {migrated} docs, {len(failed)} batch(es) failed: {failed}")
    print(f"[LIVE] final ES count: {final_es_count} (Chroma had {live_coll.count()})")

    if args.archive_too:
        archive_collection_name = getattr(conf, "ARCHIVE_COLLECTION_NAME", conf.COLLECTION_NAME + "_archive")
        try:
            archive_coll = chroma_client.get_collection(archive_collection_name)
        except Exception as exc:
            print(f"\n[Archive] could not open Chroma collection {archive_collection_name!r}: {exc} — skipping")
            archive_coll = None
        if archive_coll is not None:
            a_migrated, a_failed = _reindex_collection(
                archive_coll, es_mgr, es_mgr._archive_index, id_prefix="full_archive_", batch_size=args.batch_size,
            )
            print(f"\n[Archive] migrated {a_migrated} docs, {len(a_failed)} batch(es) failed: {a_failed}")

    print("\nDone. Zero OpenRouter cost — every embedding here was pulled straight out of Chroma, none recomputed.")


if __name__ == "__main__":
    main()

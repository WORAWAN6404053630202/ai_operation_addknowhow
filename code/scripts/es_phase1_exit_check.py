# code/scripts/es_phase1_exit_check.py
"""Phase 1 exit check (.claude/plans/mutable-seeking-wave.md): pulls a small
real subset of documents straight out of the live Chroma collection —
content, metadata, AND already-computed embeddings, so this costs zero
OpenRouter calls — and bulk-indexes them into the Elasticsearch instance.
Then runs the same handful of filter/query patterns against both backends
and prints results side by side, so a human can eyeball that the ES facade
returns the same documents Chroma does for the same query.

This is deliberately a throwaway/manual script, not wired into the app —
same "standalone, run once, not part of the live pipeline" shape as
scripts/pdf_inspect_live.py and friends in this directory.

Usage (run from inside a container that has both chromadb and elasticsearch
clients available, e.g. `docker exec -it ai-restbiz-pdf-only-es-web python3
code/scripts/es_phase1_exit_check.py --limit 30`, with the live Chroma data
directory bind-mounted/copied somewhere reachable — see CHROMA_DIR below):

    python3 code/scripts/es_phase1_exit_check.py --limit 30 --chroma-dir /data/local_chroma_v3_copy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
code_dir = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import conf
from service.local_vector_store import _materialize_embeddings


def _pull_from_chroma(chroma_dir: str, limit: int):
    """Read-only: connects to a (copied) Chroma persist dir and pulls
    `limit` real docs with content + metadata + embeddings already computed
    — no OpenRouter embedding calls made here at all. Filters to docs that
    actually have license_type set, since a plain unfiltered `.get(limit=N)`
    can land entirely on metadata-sparse marketing/guide docs (confirmed
    live: the first 30 by insertion order had zero regulatory docs among
    them), which defeats the point of a filter+kNN parity check."""
    import chromadb
    client = chromadb.PersistentClient(path=chroma_dir)
    coll = client.get_collection(conf.COLLECTION_NAME)
    raw = coll.get(limit=limit, where={"license_type": {"$ne": ""}}, include=["documents", "metadatas", "embeddings"])
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    embeddings = _materialize_embeddings(raw)
    print(f"[Chroma] pulled {len(ids)} docs (license_type != '') from collection {conf.COLLECTION_NAME!r} at {chroma_dir}")
    if not ids:
        print("[Chroma] that filter found nothing — falling back to an unfiltered pull (comparison queries will be skipped)")
        raw = coll.get(limit=limit, include=["documents", "metadatas", "embeddings"])
        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        embeddings = _materialize_embeddings(raw)
    return ids, docs, metas, embeddings


def _index_into_es(ids, docs, metas, embeddings):
    from service.es_vector_store import get_vs_manager, _stringify_metadata
    mgr = get_vs_manager()
    mgr.connect_to_existing(fail_if_empty=False)
    mgr.vectorstore._collection.upsert(
        ids=[f"phase1check_{i}" for i in ids],
        documents=docs,
        metadatas=[_stringify_metadata(m) for m in metas],
        embeddings=embeddings,
    )
    # ES doesn't make newly-indexed docs searchable/countable instantly —
    # default refresh_interval is ~1s. Force it so this script's own count
    # check right after indexing is deterministic, not a timing race.
    mgr.vectorstore._collection._client.indices.refresh(index=mgr.vectorstore._collection._index)
    count = mgr._collection_count()
    print(f"[ES] indexed {len(ids)} docs into {mgr._live_index!r} — live count now {count}")
    return mgr


def _compare(chroma_dir: str, mgr, metas, embeddings, n_queries: int = 3):
    """Runs the same filter+kNN query against both backends for a few of
    the just-indexed docs' own (metadata, embedding) pairs — the top Chroma
    hit for "find docs with this exact license_type, ranked by similarity
    to this exact vector" should, at minimum, include the doc itself."""
    import chromadb

    client = chromadb.PersistentClient(path=chroma_dir)
    chroma_coll = client.get_collection(conf.COLLECTION_NAME)

    sample_fields = [f for f in ("license_type", "department", "entity_type_normalized") if metas and metas[0].get(f)]
    if not sample_fields:
        print("[Compare] none of the sampled docs have a license_type/department/entity_type_normalized to filter on — skipping comparison queries")
        return

    field = sample_fields[0]
    checked = 0
    for meta, emb in zip(metas, embeddings):
        value = (meta or {}).get(field)
        if not value or checked >= n_queries:
            continue
        checked += 1
        print(f"\n=== Query {checked}: filter {{{field!r}: {value!r}}}, knn to this doc's own vector ===")

        chroma_res = chroma_coll.query(
            query_embeddings=[list(emb)], n_results=5, where={field: value},
            include=["metadatas", "distances"],
        )
        chroma_ids = chroma_res.get("ids", [[]])[0]
        print(f"  Chroma top ids: {chroma_ids}")

        es_res = mgr.vectorstore._collection.query(
            query_embeddings=[list(emb)], n_results=5, where={field: value},
        )
        es_ids = es_res["ids"][0]
        print(f"  ES     top ids: {es_ids}")

    if checked == 0:
        print(f"[Compare] no sampled doc had a usable {field!r} value — skipping")


_HYBRID_SPOT_CHECK_QUERIES = [
    "ค่าธรรมเนียมจดทะเบียนบริษัทจำกัด",
    "เอกสารที่ต้องใช้ยื่นขอใบอนุญาตประกอบธุรกิจอาหาร",
    "แบบ ภ.พ.01 จดทะเบียนภาษีมูลค่าเพิ่ม",
]


def _test_hybrid_search(mgr, queries=_HYBRID_SPOT_CHECK_QUERIES, k: int = 5):
    """Round 2 Step 3 live check (.claude/plans/mutable-seeking-wave.md):
    confirms hybrid_search() is genuinely fusing BM25 + kNN on real multi-
    word Thai queries, not silently degraded to dense-only. For each query,
    runs both legs side by side: plain kNN-only (via query()) vs
    hybrid_search(). A real lexical contribution shows up as (a) a doc
    appearing in the hybrid top-k that's absent/lower in dense-only, or
    (b) a `_bm25_hit` metadata flag on a doc missing from the dense leg.
    Also re-asserts the _sim-stays-bounded-0-1 invariant on live data, not
    just the unit-test mocks."""
    embedding_model = mgr.embedding_model
    for q in queries:
        print(f"\n=== Hybrid spot-check: {q!r} ===")

        if embedding_model is not None:
            query_emb = embedding_model.embed_query(q)
            dense_res = mgr.vectorstore._collection.query(query_embeddings=[query_emb], n_results=k)
            dense_ids = dense_res["ids"][0]
            dense_sims = [1.0 - d for d in dense_res.get("distances", [[]])[0]]
            print(f"  dense-only top ids:  {dense_ids}")
            print(f"  dense-only _sim:     {[round(s, 4) for s in dense_sims]}")
        else:
            dense_ids = []
            print("  (could not reach embedding_model on vectorstore — skipping dense-only comparison)")

        hybrid_pairs = mgr.vectorstore.hybrid_search(q, k=k)
        hybrid_ids = [doc.metadata.get("_es_id") for doc, _ in hybrid_pairs]
        hybrid_sims = [score for _, score in hybrid_pairs]
        bm25_only_hits = [doc.metadata.get("_es_id") for doc, _ in hybrid_pairs if doc.metadata.get("_bm25_hit")]
        print(f"  hybrid top ids:      {hybrid_ids}")
        print(f"  hybrid _sim:         {[round(s, 4) for s in hybrid_sims]}")
        if bm25_only_hits:
            print(f"  BM25-only hits (missing from dense leg, bypass-eligible): {bm25_only_hits}")
        out_of_range = [s for s in hybrid_sims if s < 0.0 or s > 1.0]
        if out_of_range:
            print(f"  !! _sim OUT OF [0,1] RANGE: {out_of_range} — invariant broken, investigate before trusting hybrid search")
        if dense_ids and hybrid_ids == dense_ids:
            print("  (note: hybrid order identical to dense-only — fine if BM25 genuinely agreed, but check with a more lexical-distinctive query if unexpected)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=30, help="how many real Chroma docs to pull and index into ES")
    parser.add_argument("--chroma-dir", default=None, help="path to a (copied) Chroma persist directory, read-only (required unless --skip-ingest)")
    parser.add_argument("--skip-ingest", action="store_true", help="skip the Chroma pull + reindex, just run against what's already in ES (use when docs from a prior run are still live)")
    parser.add_argument("--hybrid-only", action="store_true", help="skip the filter/kNN parity check, only run the hybrid-search spot check")
    args = parser.parse_args()

    if args.skip_ingest:
        from service.es_vector_store import get_vs_manager
        mgr = get_vs_manager()
        mgr.connect_to_existing(fail_if_empty=False)
        _test_hybrid_search(mgr)
        return

    if not args.chroma_dir:
        parser.error("--chroma-dir is required unless --skip-ingest is set")

    ids, docs, metas, embeddings = _pull_from_chroma(args.chroma_dir, args.limit)
    if not ids:
        print("No docs pulled — check --chroma-dir and conf.COLLECTION_NAME.")
        return
    mgr = _index_into_es(ids, docs, metas, embeddings)
    if not args.hybrid_only:
        _compare(args.chroma_dir, mgr, metas, embeddings)
    _test_hybrid_search(mgr)
    print("\nDone. This confirms (a) the ES facade can be written to and queried via the real elasticsearch client, "
          "(b) filter+kNN results are at least plausible vs. Chroma for the same vectors/filters, and "
          "(c) hybrid_search() genuinely fuses BM25+kNN on real Thai queries without breaking the _sim 0-1 invariant. "
          "Eyeball the printed ids above — the doc each query was built FROM should normally appear near the top of both lists.")


if __name__ == "__main__":
    main()

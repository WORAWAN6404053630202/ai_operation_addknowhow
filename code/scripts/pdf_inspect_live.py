# code/scripts/pdf_inspect_live.py
"""Read-only inspection tool for the LIVE (not archived) collection — shows
exactly what a PDF's direct-ingest actually produced: how it got split into
topics, what each topic got tagged as (main_topic/sub_topic/content_type),
and the real stored content, so a human can judge the LLM-assigned topic
boundaries/labels directly instead of inferring quality from chat answers
alone.

Mirrors pdf_archive_inspect.py's connection pattern (same get_vs_manager(),
same env-driven COLLECTION_NAME/LOCAL_VECTOR_DIR) but queries the live
collection filtered by source_file, not the archive collection.

Usage:
  python code/scripts/pdf_inspect_live.py --source ยื่นจดทะเบียนออนไลน์.pdf
  python code/scripts/pdf_inspect_live.py --source ยื่นจดทะเบียนออนไลน์.pdf --full
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
from service.local_vector_store import get_vs_manager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="source_file substring to filter on (e.g. the uploaded PDF's filename)")
    parser.add_argument("--full", action="store_true", help="print each doc's full stored content instead of a 300-char preview")
    args = parser.parse_args()

    mgr = get_vs_manager()
    mgr.connect_to_existing(fail_if_empty=False)
    print(f"Collection: {conf.COLLECTION_NAME}  |  Dir: {conf.LOCAL_VECTOR_DIR}\n")

    raw = mgr.vectorstore._collection.get(
        where={"source_file": {"$eq": args.source}} if "." in args.source else None,
        include=["metadatas", "documents"],
    )
    ids = raw.get("ids") or []
    metas = raw.get("metadatas") or []
    docs = raw.get("documents") or []

    # Fall back to substring match client-side if the exact-match filter
    # above found nothing (source_file may carry a path prefix or differ
    # slightly from what the user typed).
    if not ids:
        all_raw = mgr.vectorstore._collection.get(include=["metadatas", "documents"])
        for i, m in zip(all_raw.get("ids") or [], all_raw.get("metadatas") or []):
            if args.source in str((m or {}).get("source_file") or ""):
                ids.append(i)
        if ids:
            id_set = set(ids)
            metas = [m for i, m in zip(all_raw["ids"], all_raw["metadatas"]) if i in id_set]
            docs = [d for i, d in zip(all_raw["ids"], all_raw["documents"]) if i in id_set]

    if not ids:
        print(f"No live docs found with source_file matching {args.source!r}.")
        return

    print(f"{len(ids)} doc(s) found:\n")
    for i, (doc_id, meta, content) in enumerate(zip(ids, metas, docs), start=1):
        meta = meta or {}
        print(f"--- [{i}/{len(ids)}] doc_id={doc_id} ---")
        print(f"  main_topic:    {meta.get('main_topic') or ''}")
        print(f"  sub_topic:     {meta.get('sub_topic') or ''}")
        print(f"  operation_topic: {meta.get('operation_topic') or ''}")
        print(f"  content_type:  {meta.get('content_type') or ''}")
        print(f"  chunk:         {meta.get('chunk_index')}/{meta.get('chunk_total')}")
        print(f"  ingested_via:  {meta.get('ingested_via') or ''}")
        preview = content if args.full else (content[:300] + ("..." if len(content or "") > 300 else ""))
        print(f"  content:\n    {preview}\n")

    print(f"{len(ids)} doc(s) total for source_file matching {args.source!r}.")


if __name__ == "__main__":
    main()

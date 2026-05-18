"""
check_content_quality.py — Diagnostic tool for empty / thin content in the vector DB.

Usage:
    PYTHONPATH="$PWD" python -m code.scripts.check_content_quality

Output:
    - Total document chunks in Chroma
    - Per-topic summary: how many chunks have content vs. empty
    - List of topics where ALL chunks are empty (needs data in Google Sheets)
    - List of topics where content is very short (<100 chars average)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# Bootstrap path so we can import project modules
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from code.service.local_vector_store import get_retriever  # noqa: E402


_MIN_CONTENT_CHARS = 100  # threshold for "thin" content warning


def main() -> None:
    print("=" * 70)
    print("Restbiz — Content Quality Diagnostic")
    print("=" * 70)

    retriever = get_retriever(fail_if_empty=True)
    vectorstore = getattr(retriever, "vectorstore", None)
    if vectorstore is None:
        print("[ERROR] Could not access vectorstore. Aborting.")
        sys.exit(1)

    coll = getattr(vectorstore, "_collection", None)
    if coll is None:
        print("[ERROR] Could not access Chroma collection. Aborting.")
        sys.exit(1)

    result = coll.get(include=["metadatas", "documents"])
    metadatas: List[dict] = result.get("metadatas") or []
    documents: List[str] = result.get("documents") or []

    total = len(documents)
    print(f"\nTotal chunks in DB: {total}\n")

    # topic → list of (content_len, section)
    topic_chunks: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for md, doc in zip(metadatas, documents):
        md = md or {}
        topic = (md.get("operation_topic") or md.get("topic") or "").strip() or "(no topic)"
        section = (md.get("section") or md.get("doc_type") or "").strip()
        content_len = len((doc or "").strip())
        topic_chunks[topic].append((content_len, section))

    # Build report
    empty_topics: List[str] = []
    thin_topics: List[str] = []

    rows: List[Tuple[str, int, int, int, float]] = []
    for topic, chunks in sorted(topic_chunks.items()):
        n = len(chunks)
        empty_count = sum(1 for c, _ in chunks if c == 0)
        nonempty = [c for c, _ in chunks if c > 0]
        avg_len = sum(nonempty) / len(nonempty) if nonempty else 0.0
        rows.append((topic, n, empty_count, len(nonempty), avg_len))

        if empty_count == n:
            empty_topics.append(topic)
        elif avg_len < _MIN_CONTENT_CHARS and nonempty:
            thin_topics.append(topic)

    # Print table
    print(f"{'TOPIC':<50} {'CHUNKS':>6} {'EMPTY':>6} {'OK':>4} {'AVG_LEN':>8}")
    print("-" * 80)
    for topic, n, empty_count, ok, avg in rows:
        flag = " ⚠️ EMPTY" if empty_count == n else (" thin" if avg < _MIN_CONTENT_CHARS and ok > 0 else "")
        short_topic = (topic[:47] + "...") if len(topic) > 50 else topic
        print(f"{short_topic:<50} {n:>6} {empty_count:>6} {ok:>4} {avg:>8.0f}{flag}")

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(topic_chunks)} unique topics | {total} total chunks")
    if empty_topics:
        print(f"\n🔴 Topics with ALL chunks EMPTY ({len(empty_topics)}) — needs data in Google Sheets:")
        for t in empty_topics:
            print(f"   - {t}")
    else:
        print("\n✅ No fully-empty topics found.")

    if thin_topics:
        print(f"\n🟡 Topics with THIN content avg < {_MIN_CONTENT_CHARS} chars ({len(thin_topics)}):")
        for t in thin_topics:
            print(f"   - {t}")

    print("\nDone.")


if __name__ == "__main__":
    main()

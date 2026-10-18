# code/scripts/pdf_archive_inspect.py
"""Audit/rollback tool for the automated content supersede resolver
(service/pdf_supersede_resolver.py, 2026-10) — since supersede decisions are
made automatically with no human review, this is the periodic-review
mechanism that compensates for that: look at what got archived (hidden from
retrieval) and, if a decision was wrong, restore it to live.

Usage:
  python code/scripts/pdf_archive_inspect.py --list
  python code/scripts/pdf_archive_inspect.py --list --pattern labor_law.pdf
  python code/scripts/pdf_archive_inspect.py --list --since-days 7
  python code/scripts/pdf_archive_inspect.py --restore <doc_id> [<doc_id> ...]

--list shows every archived doc's id, source_file, license_type/main_topic,
superseded_at, and superseded_reason — enough to decide whether a decision
looks right without needing to restore it first. Always sorted newest-
archived-first (2026-10 — see LocalVectorStoreManager.list_archived_docs).
--since-days N limits --list to docs archived in the last N days — added
2026-10 alongside the sort fix, since "what got archived recently" used to
mean scrolling an unsorted, unfiltered dump of every archived doc ever.
--restore takes one or more doc ids straight from --list's output and moves
them back to the live collection, stripping the superseded_at/
superseded_reason stamp."""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
code_dir = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from service.local_vector_store import get_vs_manager


def _fmt_time(ts) -> str:
    try:
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts or "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List every archived (superseded) doc")
    parser.add_argument("--pattern", default="", help="With --list, only show docs whose source_file contains this substring")
    parser.add_argument("--since-days", type=float, default=None, metavar="N", help="With --list, only show docs archived in the last N days")
    parser.add_argument("--restore", nargs="+", metavar="DOC_ID", default=None, help="Restore these doc ids from the archive back to live")
    args = parser.parse_args()

    mgr = get_vs_manager()

    if args.restore:
        restored = mgr.restore_from_archive(args.restore)
        print(f"Restored {restored} doc(s) to the live collection.")
        if restored < len(args.restore):
            print(f"Note: {len(args.restore) - restored} requested id(s) were not found in the archive.")
        return

    # Default / --list
    since_epoch = (time.time() - args.since_days * 86400) if args.since_days is not None else None
    docs = mgr.list_archived_docs(pattern=args.pattern or None, since_epoch=since_epoch)
    if not docs:
        qualifiers = []
        if args.pattern:
            qualifiers.append(f"matching pattern={args.pattern!r}")
        if args.since_days is not None:
            qualifiers.append(f"in the last {args.since_days:g} day(s)")
        print("No archived docs found" + (" " + " and ".join(qualifiers) + "." if qualifiers else "."))
        return

    print(f"{'doc_id':<38} {'source_file':<30} {'topic':<30} {'superseded_at':<20} reason")
    print("-" * 150)
    for d in docs:
        meta = d.get("metadata") or {}
        source_file = str(meta.get("source_file") or "")[:30]
        topic = str(meta.get("license_type") or meta.get("main_topic") or "")[:30]
        superseded_at = _fmt_time(meta.get("superseded_at"))
        reason = str(meta.get("superseded_reason") or "")
        print(f"{d['id']:<38} {source_file:<30} {topic:<30} {superseded_at:<20} {reason}")
    print(f"\n{len(docs)} archived doc(s) total.")


if __name__ == "__main__":
    main()

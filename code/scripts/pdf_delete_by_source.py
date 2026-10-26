# code/scripts/pdf_delete_by_source.py
"""Recovery tool for the no-human-review PDF direct-ingest path
(service/pdf_direct_ingest.py, 2026-09) — since content now reaches the bot
with no review gate, this is the only way to remove a bad ingest (missing
pages, wrong chunking, etc.) after the fact.

Usage:
  python code/scripts/pdf_delete_by_source.py --list
  python code/scripts/pdf_delete_by_source.py --list --pattern labor
  python code/scripts/pdf_delete_by_source.py --delete "labor_law.pdf"
  python code/scripts/pdf_delete_by_source.py --delete "labor_law.pdf" --confirm

Without --confirm, --delete only prints what WOULD be removed (dry run) —
matches the rest of this codebase's "confirm before destructive action"
convention. --list takes no filename; it's how you find the exact
source_file string to pass to --delete (must match exactly)."""

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

from service.local_vector_store import get_vs_manager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List every PDF-auto-ingested source_file with its doc count")
    parser.add_argument("--pattern", default="", help="With --list, only show source_file names containing this substring")
    parser.add_argument("--delete", metavar="SOURCE_FILE", default="", help="Delete all docs from this exact source_file")
    parser.add_argument("--confirm", action="store_true", help="Actually perform the deletion (omit for a dry run)")
    args = parser.parse_args()

    mgr = get_vs_manager()

    if args.delete:
        counts = mgr.list_pdf_ingested_files()
        n = counts.get(args.delete, 0)
        if n == 0:
            print(f"No PDF-auto-ingested docs found for source_file={args.delete!r}. "
                  f"Run --list to see exact names (must match exactly).")
            return
        if not args.confirm:
            print(f"DRY RUN — would delete {n} doc(s) for source_file={args.delete!r}. "
                  f"Re-run with --confirm to actually delete.")
            return
        deleted = mgr.delete_by_source_file(args.delete)
        print(f"Deleted {deleted} doc(s) for source_file={args.delete!r}.")
        # 2026-10 fix (data-integrity audit, TOCTOU): `n` above was counted
        # in a SEPARATE, earlier call (often a separate CLI invocation
        # entirely — a human reads the dry run, then re-runs with
        # --confirm minutes later) — a concurrent SQS ingest adding more
        # pdf_auto docs for this exact source_file in between would get
        # silently swept into this delete without ever appearing in what
        # the operator previewed. TOCTOU research is consistent that the
        # real fix is eliminating the gap entirely (atomic check-and-use),
        # which isn't possible across two independent human-triggered CLI
        # runs — so this surfaces the mismatch loudly instead, since
        # silently deleting more than what was previewed is the one part
        # of this gap that's avoidable.
        if deleted != n:
            print(
                f"NOTE: deleted count ({deleted}) differs from the dry-run preview ({n}) — "
                f"more docs for this source_file were likely ingested between the preview and "
                f"this run. If unexpected, check --list again before assuming this was intentional."
            )
        return

    # Default / --list
    counts = mgr.list_pdf_ingested_files()
    if args.pattern:
        counts = {k: v for k, v in counts.items() if args.pattern in k}
    if not counts:
        print("No PDF-auto-ingested docs found in this collection.")
        return
    print(f"{'source_file':<50} doc_count")
    print("-" * 62)
    for src, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{src:<50} {n}")


if __name__ == "__main__":
    main()

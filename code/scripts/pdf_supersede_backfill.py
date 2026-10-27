# code/scripts/pdf_supersede_backfill.py
"""One-time cleanup for docs stuck permanently blocked under the OLD
pdf_supersede_resolver.py behavior (before its field_merges redesign,
2026-10) — confirmed against real production data: 204 of ~421 live PDF-
auto-ingested docs never got retired because the old completeness-check
blocked retirement forever on ANY flagged claim, with nothing in this
pipeline ever revisiting a blocked pair (no human review step, no sweep).

This script re-resolves each already-stuck doc_id against the rest of the
live corpus, using the SAME candidate-retrieval machinery
pdf_supersede_resolver.resolve_topic() already uses for live ingest (proven
robust to LLM topic-name drift across separate ingest runs of the same real
content — naive exact-string grouping by (source_file, main_topic,
sub_topic) was tried first and silently missed every real pair).

Direction note: resolve_topic() is built for the live-ingest shape (a fresh
"new_values" topic vs. live "old" candidates to retire) — here BOTH sides
are already-live historical docs, so resolve_topic is used only to answer
"is there exactly one confirmed same-topic live sibling for this stuck doc?"
(direction-agnostic for that question). Once a sibling is confirmed, the
sibling with the LATER ingested_at metadata timestamp (see
service/pdf_direct_ingest.py's _pdf_tags) is treated as the keeper — it was
the one that originally tried to supersede the older doc_id and got
blocked — and the completeness check is re-run in the CORRECT direction
(old=the stuck doc, new=the keeper) before merging and retiring, matching
exactly what sqs_consumer.py now does on the live path.

Any case this script can't resolve with confidence (no confirmed sibling,
more than one confirmed sibling, or the sibling's ingested_at isn't clearly
later) is reported and skipped — a one-time human-triggered backfill is
exactly the place where "leave a residual report" is fine, unlike the live
pipeline, which must never block on a human.

Usage:
  python code/scripts/pdf_supersede_backfill.py --ids-file stuck_ids.txt
  python code/scripts/pdf_supersede_backfill.py --all-live --confirm
  python code/scripts/pdf_supersede_backfill.py --all-live --limit 5 --confirm

stuck_ids.txt: one Chroma doc id per line (blank lines / lines starting
with # are ignored) — a pre-identified candidate list (e.g. grepped from a
prior run's "refusing to retire" log warnings).

--all-live: skips the pre-identified list entirely and feeds every live
ingested_via="pdf_auto" doc id straight in — safe and more robust than
relying on log lines that may have rotated away, since resolve_topic()'s
own 3-signal matcher (not a naive exact-string grouping) decides per doc
whether a sibling actually exists; a doc with no stuck sibling just comes
back "new" and is skipped at near-zero extra cost (no LLM call unless an
embedding/identity candidate pool exists at all).

Without --confirm, only prints the plan (dry run) — matches this
codebase's standing "confirm before destructive action" convention (see
pdf_delete_by_source.py)."""

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
from service.pdf_supersede_resolver import (
    _KNOWHOW_METADATA_KEY_ALIASES,
    _MERGED_COMPARISON_KEYS,
    _MERGED_DIFF_KEYS,
    _check_content_completeness,
    build_field_merges_from_missing_claims,
    resolve_topic,
)
from utils.llm_cost_logging import CostAccumulator

_RESOLVE_KEYS = sorted(set(_MERGED_COMPARISON_KEYS) | set(_MERGED_DIFF_KEYS))


def _read_ids_file(path: str) -> list[str]:
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def _values_from_live_metadata(meta: dict) -> dict:
    """Builds the same shape resolve_topic()/merged_topic expects out of a
    LIVE doc's own metadata — applying the same summary/category→
    answer_guideline/content_type aliasing _resolve() applies when reading
    an existing live doc's metadata (these are old, pre-unification docs,
    so the alias is very much still needed here, more so than on the live
    ingest path)."""
    return {k: meta.get(_KNOWHOW_METADATA_KEY_ALIASES.get(k, k)) for k in _RESOLVE_KEYS}


def _group_key(meta: dict) -> tuple:
    return (meta.get("source_file") or "", meta.get("main_topic") or "", meta.get("sub_topic") or "")


def _sibling_chunk_ids(vectorstore, group_key: tuple, known_id: str) -> list[str]:
    """Every live id sharing one logical topic's (source_file, main_topic,
    sub_topic) — a topic with chunk_total > 1 writes one independent
    Document per sub-chunk (build_merged_documents' own design), so a merge
    or a retire must reach every one of them, not just whichever single id
    resolve_topic/the ids-file happened to name.

    known_id: the doc id this group_key was itself derived FROM (always
    included in the result). 2026-10 fix (live-confirmed on the real
    corpus): a doc whose metadata is missing the main_topic/sub_topic keys
    entirely (not just blank — genuinely absent, seen on at least one old/
    legacy live doc) makes _group_key() treat it as ("", "") on the Python
    side, but Chroma's own where-equality filter does NOT treat a missing
    key as equal to "" — so a query for THIS EXACT doc's own group came
    back with zero ids, including not even itself. Without known_id, that
    silently turned a real merge+retire into a no-op (update_live_metadata/
    archive_and_remove both safely no-op on an empty id list) — not data
    loss, but a stuck doc this script exists to clean up stayed stuck."""
    source_file, main_topic, sub_topic = group_key
    raw = vectorstore._collection.get(
        where={"$and": [
            {"source_file": source_file}, {"main_topic": main_topic}, {"sub_topic": sub_topic},
        ]},
        include=[],
    )
    ids = set(raw.get("ids") or [])
    ids.add(known_id)
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--ids-file", default="", help="File with one stuck doc_id per line")
    source_group.add_argument("--all-live", action="store_true", help="Scan every live ingested_via='pdf_auto' doc instead of a pre-identified list")
    parser.add_argument("--confirm", action="store_true", help="Actually perform merges/retirements (omit for a dry run)")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N logical topics (0 = no limit) — useful for a small trial run")
    args = parser.parse_args()

    mgr = get_vs_manager()
    if mgr.vectorstore is None:
        mgr.connect_to_existing(fail_if_empty=False)
    vectorstore = mgr.vectorstore
    if vectorstore is None:
        print("Could not connect to the live vectorstore.")
        return

    if args.all_live:
        raw = vectorstore._collection.get(where={"ingested_via": "pdf_auto"}, include=[])
        stuck_ids = raw.get("ids") or []
        print(f"--all-live: found {len(stuck_ids)} live pdf_auto doc(s) to scan.")
    else:
        stuck_ids = _read_ids_file(args.ids_file)
    if not stuck_ids:
        print("No ids to process.")
        return

    fetched = vectorstore._collection.get(ids=stuck_ids, include=["metadatas"])
    meta_by_id = dict(zip(fetched.get("ids") or [], fetched.get("metadatas") or []))

    not_live = [i for i in stuck_ids if i not in meta_by_id]
    if not_live:
        print(f"{len(not_live)} id(s) are no longer live (already resolved?) — skipping: {not_live}")

    # Dedup: several grepped ids are often sub-chunks of the SAME logical
    # stuck topic — process one representative per (source_file, main_topic,
    # sub_topic) group, not once per chunk id.
    representative_by_group: dict[tuple, str] = {}
    for doc_id, meta in meta_by_id.items():
        group_key = _group_key(meta)
        representative_by_group.setdefault(group_key, doc_id)

    groups = list(representative_by_group.items())
    if args.limit:
        groups = groups[: args.limit]

    print(f"{len(meta_by_id)} live id(s) found, {len(groups)} distinct logical topic(s) to resolve.")
    if not args.confirm:
        print("DRY RUN — no changes will be made. Re-run with --confirm to actually apply.\n")

    cost_accumulator = CostAccumulator()
    model = str(getattr(conf, "OPENROUTER_MODEL_PDF_SUPERSEDE", conf.OPENROUTER_MODEL_PDF_MATCHING))

    counts = {"merged_and_retired": 0, "retired_no_merge_needed": 0, "skipped_no_sibling": 0, "skipped_ambiguous": 0}

    for group_key, rep_id in groups:
        rep_meta = meta_by_id[rep_id]
        source_file, main_topic, sub_topic = group_key
        label = f"{rep_id} ({main_topic!r}/{sub_topic!r} from {source_file!r})"

        rep_values = _values_from_live_metadata(rep_meta)
        # 2026-10 fix (live-confirmed on the first --all-live dry run against
        # the real corpus): a topic with chunk_total > 1 writes one Document
        # PER sub-chunk, every sub-chunk sharing the exact same main_topic/
        # sub_topic — excluding only rep_id left its OWN other sub-chunks in
        # the candidate pool, where they matched perfectly on identity and
        # got "confirmed" as if they were a genuinely different stuck
        # duplicate. own_group_ids (computed BEFORE resolving, from the live
        # collection, not from this run's input list) covers every one of
        # this topic's own chunks so none of them can self-match.
        own_group_ids = _sibling_chunk_ids(vectorstore, group_key, rep_id)
        try:
            decision = resolve_topic(
                rep_values, source_file, cost_accumulator=cost_accumulator,
                exclude_doc_id=rep_id, exclude_doc_ids=own_group_ids,
            )
        except Exception as e:
            print(f"[SKIP] {label}: resolve_topic failed ({e}) — needs manual review")
            counts["skipped_ambiguous"] += 1
            continue

        if decision.decision == "new" or not decision.matched_doc_ids:
            print(f"[SKIP] {label}: no confirmed live sibling found — needs manual review")
            counts["skipped_no_sibling"] += 1
            continue

        if len(decision.matched_doc_ids) != 1:
            print(f"[SKIP] {label}: {len(decision.matched_doc_ids)} confirmed sibling(s) ({decision.matched_doc_ids}) — ambiguous, needs manual review")
            counts["skipped_ambiguous"] += 1
            continue

        sibling_id = decision.matched_doc_ids[0]
        sibling_fetch = vectorstore._collection.get(ids=[sibling_id], include=["metadatas"])
        sibling_metas = sibling_fetch.get("metadatas") or []
        if not sibling_metas:
            print(f"[SKIP] {label}: confirmed sibling {sibling_id!r} is no longer live — needs manual review")
            counts["skipped_ambiguous"] += 1
            continue
        sibling_meta = sibling_metas[0]

        # Second, independent layer against the same bug the exclude_doc_ids
        # fix above addresses at the source — never trust one signal alone
        # for a decision this destructive (matches this module's own
        # "default to NEW, never guess" philosophy). If the confirmed
        # sibling still somehow belongs to this exact same logical topic,
        # merging into it and retiring it would be retiring the ONLY copy.
        if _group_key(sibling_meta) == group_key:
            print(f"[SKIP] {label}: confirmed sibling {sibling_id!r} belongs to this SAME logical topic (not a different duplicate) — refusing to self-retire, needs manual review")
            counts["skipped_ambiguous"] += 1
            continue

        if decision.decision == "duplicate":
            old_group_ids = own_group_ids
            print(f"[DUPLICATE] {label}: content already fully covered by live sibling {sibling_id!r} — retiring {old_group_ids}, no merge needed")
            if args.confirm:
                mgr.archive_and_remove(old_group_ids, reason=f"pdf_supersede_backfill: duplicate of {sibling_id}")
            counts["retired_no_merge_needed"] += 1
            continue

        # decision.decision == "update": direction is ambiguous from
        # resolve_topic's own verdict alone (see module docstring above) —
        # use ingested_at to determine which side is the keeper.
        try:
            rep_ingested_at = float(rep_meta.get("ingested_at") or 0)
            sibling_ingested_at = float(sibling_meta.get("ingested_at") or 0)
        except (TypeError, ValueError):
            print(f"[SKIP] {label}: malformed ingested_at timestamp on this or sibling {sibling_id!r} — needs manual review")
            counts["skipped_ambiguous"] += 1
            continue
        if sibling_ingested_at <= rep_ingested_at:
            print(
                f"[SKIP] {label}: sibling {sibling_id!r} is not clearly newer "
                f"(ingested_at {sibling_ingested_at} vs {rep_ingested_at}) — needs manual review"
            )
            counts["skipped_ambiguous"] += 1
            continue

        sibling_values = _values_from_live_metadata(sibling_meta)
        old_values_for_check = {rep_id: {k: rep_values.get(k) for k in _MERGED_DIFF_KEYS}}
        missing_claims = _check_content_completeness(
            old_values_for_check, sibling_values, _MERGED_DIFF_KEYS, model, cost_accumulator=cost_accumulator,
        )
        field_merges = build_field_merges_from_missing_claims(missing_claims, _MERGED_DIFF_KEYS, source_file)

        old_group_ids = own_group_ids
        sibling_group_key = _group_key(sibling_meta)
        sibling_group_ids = _sibling_chunk_ids(vectorstore, sibling_group_key, sibling_id)

        # Final, cheap sanity check: the two full id-sets about to be
        # treated as "keep and merge into" vs. "archive" must be disjoint.
        # Should be mathematically guaranteed by the group_key != group_key
        # check above (each id belongs to exactly one (source_file,
        # main_topic, sub_topic) group) — kept anyway as defense-in-depth
        # directly on the id lists this call is about to act on, not just
        # on the keys that derived them.
        overlap = set(old_group_ids) & set(sibling_group_ids)
        if overlap:
            print(f"[SKIP] {label}: old/sibling id sets overlap ({sorted(overlap)}) — refusing, needs manual review")
            counts["skipped_ambiguous"] += 1
            continue

        if field_merges:
            print(
                f"[MERGE+RETIRE] {label}: merging field(s) {list(field_merges.keys())} into sibling "
                f"{sibling_group_ids}, then retiring {old_group_ids}"
            )
            if args.confirm:
                mgr.update_live_metadata(sibling_group_ids, field_merges)
        else:
            print(f"[RETIRE] {label}: no omitted claims found — retiring {old_group_ids}, sibling {sibling_group_ids} already covers it")
        if args.confirm:
            mgr.archive_and_remove(old_group_ids, reason=f"pdf_supersede_backfill: superseded by {sibling_id}")
        counts["merged_and_retired"] += 1

    print("\nSummary:")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    print(f"  total_llm_cost_usd: {cost_accumulator.total_cost_usd:.4f}")
    if not args.confirm:
        print("\nDRY RUN — re-run with --confirm to actually apply the above.")


if __name__ == "__main__":
    main()

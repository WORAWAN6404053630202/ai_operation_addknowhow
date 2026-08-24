# code/utils/page_ranges.py
"""Shared page-range formatting/merging for the PDF review queue
(feature/pdf-ingestion) — added 2026-08-24 to fix a real bug: every call
site (sheet_write_back.py's _format_page_range, admin.py's _page_range_str,
sqs_consumer.py's know-how page_range string) independently computed
`min(page_nums)-max(page_nums)`, which is WRONG once a topic can legitimately
span non-contiguous pages (see merge_page_ranges below) — "1-3" implies page
2 is included when it might not be. format_page_ranges() below compresses
into runs correctly (e.g. [1, 3, 4] -> "1, 3-4") instead."""

from __future__ import annotations

import difflib
import re


def fuzzy_ratio(a: str, b: str) -> float:
    """Normalized (whitespace-stripped, lowercased) SequenceMatcher ratio —
    shared by pdf_candidate_matching.py's identity-field matching and this
    module's merge_topic_chunks(), so "same string comparison logic" doesn't
    silently drift into 2 different implementations over time."""
    a2 = re.sub(r"\s+", "", (a or "")).strip().lower()
    b2 = re.sub(r"\s+", "", (b or "")).strip().lower()
    if not a2 or not b2:
        return 0.0
    return difflib.SequenceMatcher(None, a2, b2).ratio()


def format_page_ranges(page_nums: list[int]) -> str:
    """"4" for a single page, "1-3" for one contiguous run, "1, 3-4" for
    disjoint runs — never claims a page is included when it isn't."""
    nums = sorted(set(page_nums))
    if not nums:
        return ""

    runs: list[tuple[int, int]] = []
    run_start = run_end = nums[0]
    for n in nums[1:]:
        if n == run_end + 1:
            run_end = n
        else:
            runs.append((run_start, run_end))
            run_start = run_end = n
    runs.append((run_start, run_end))

    return ", ".join(str(s) if s == e else f"{s}-{e}" for s, e in runs)


def merge_topic_chunks(
    chunks: list[dict], identity_keys: tuple[str, ...], fuzzy_match
) -> list[dict]:
    """Groups LLM-identified chunks (each with start_page/end_page plus
    whatever identity_keys name, e.g. ("department", "license_type")) into
    one entry per distinct topic, combining page ranges when the same topic
    appears in more than one chunk — the LLM naturally produces separate
    chunks when an unrelated chunk interrupts the same topic (e.g. pages
    1 and 3 are "ใบอนุญาตขายสุรา" but page 2 is something else), since each
    chunk is a single contiguous start_page/end_page.

    fuzzy_match(a: str, b: str) -> float should return a 0..1 similarity —
    reuses pdf_candidate_matching.py's _fuzzy_ratio so a chunk the LLM
    labeled with a slightly different string the second time around (e.g.
    appending "(ต่อ)") still merges instead of silently staying split.

    Returns each group as {**first_chunk_without_page_fields, "page_ranges":
    [(s,e), ...]} sorted ascending, groups in first-seen order — chunks with
    the SAME identity every time (the overwhelming majority case) come out
    exactly as a single-entry page_ranges list, so this is a no-op in the
    common case, not just for the non-contiguous-interruption edge case.
    Non-identity, non-page fields (e.g. know-how's summary/category) are
    kept from whichever chunk was seen FIRST for that topic — later chunks
    of the same topic only contribute their page range, not a second
    summary/category, since those describe the topic as a whole rather than
    one specific chunk of it."""
    _FUZZY_MERGE_THRESHOLD = 0.75

    groups: list[dict] = []
    for chunk in chunks:
        identity_values = {k: chunk[k] for k in identity_keys}
        match = None
        for g in groups:
            # Conservative on purpose: only merge when EVERY identity key is
            # non-blank in both AND fuzzy-matches. A missed merge just means
            # 2 rows instead of 1 (a human sees both, no data lost); a wrong
            # merge could conflate genuinely distinct topics — worse.
            same = all(
                identity_values[k] and g[k] and fuzzy_match(identity_values[k], g[k]) >= _FUZZY_MERGE_THRESHOLD
                for k in identity_keys
            )
            if same:
                match = g
                break
        if match is None:
            extra_fields = {k: v for k, v in chunk.items() if k not in ("start_page", "end_page")}
            match = {**extra_fields, "page_ranges": []}
            groups.append(match)
        match["page_ranges"].append((chunk["start_page"], chunk["end_page"]))

    for g in groups:
        g["page_ranges"] = sorted(g["page_ranges"])
    return groups

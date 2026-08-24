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


def group_into_ranges(page_nums: list[int]) -> list[tuple[int, int]]:
    """[1,2,3,5] -> [(1,3),(5,5)] — compresses a flat page-number list into
    contiguous (start,end) runs. Shared by format_page_ranges below and by
    sqs_consumer.py's uncovered-page fallback (see _build_license_items)."""
    nums = sorted(set(page_nums))
    if not nums:
        return []

    runs: list[tuple[int, int]] = []
    run_start = run_end = nums[0]
    for n in nums[1:]:
        if n == run_end + 1:
            run_end = n
        else:
            runs.append((run_start, run_end))
            run_start = run_end = n
    runs.append((run_start, run_end))
    return runs


def format_page_ranges(page_nums: list[int]) -> str:
    """"4" for a single page, "1-3" for one contiguous run, "1, 3-4" for
    disjoint runs — never claims a page is included when it isn't."""
    runs = group_into_ranges(page_nums)
    return ", ".join(str(s) if s == e else f"{s}-{e}" for s, e in runs)


def _digit_sequences(s: str) -> list[str]:
    return re.findall(r"\d+", s or "")


def _differs_only_by_digits(a: str, b: str) -> bool:
    """True when two strings are identical apart from embedded digit
    sequences that are themselves different (e.g. "...จำพวกที่ 1" vs
    "...จำพวกที่ 2", or a "รหัส 001" vs "รหัส 021" style code) — added
    2026-08-25 after live-testing found the plain fuzzy_ratio check below
    cannot tell these apart: a string differing by one digit out of 40+
    characters scores a very high similarity ratio despite the digit being
    the ENTIRE meaningful distinction (a different regulatory class/code,
    not incidental rewording like "(ต่อ)"). Real Thai regulatory patterns
    like "ใบอนุญาตประกอบกิจการโรงงาน จำพวกที่ 1/2/3" make this a genuine
    risk, not just a contrived edge case."""
    stripped_a = re.sub(r"\d+", "", a or "")
    stripped_b = re.sub(r"\d+", "", b or "")
    digits_a, digits_b = _digit_sequences(a), _digit_sequences(b)
    return stripped_a == stripped_b and digits_a != digits_b and bool(digits_a or digits_b)


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
    appending "(ต่อ)") still merges instead of silently staying split. A
    digits-only difference (see _differs_only_by_digits) blocks the merge
    regardless of how high the fuzzy score is — a differing class/code
    number is the single most likely case where 2 strings are 95%+ textually
    identical yet describe genuinely different topics.

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
            # non-blank in both, fuzzy-matches, AND isn't a digits-only
            # difference. A missed merge just means 2 rows instead of 1 (a
            # human sees both, no data lost); a wrong merge could conflate
            # genuinely distinct topics — worse.
            same = all(
                identity_values[k]
                and g[k]
                and not _differs_only_by_digits(identity_values[k], g[k])
                and fuzzy_match(identity_values[k], g[k]) >= _FUZZY_MERGE_THRESHOLD
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

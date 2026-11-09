"""Tests for cross-page table linking in service/pdf_large_extraction.py
(link_cross_page_tables, _extract_trailing_markdown_table,
_extract_leading_markdown_table) — 2026-11, PDF-extraction-pipeline
overhaul, gap #4 (second half: tables spanning 2 pages get split with no
linking step, the continuation's repeated header lands mid-table as a data
row, or the continuation is simply lost as a disconnected second table).

First-cut text-level heuristic (pages are extracted in parallel, so no
structural Table objects survive past a single page) — explicitly flagged
in the plan as needing validation against real multi-page tables from the
gold-set corpus before being trusted as fully reliable. Deliberately fails
toward leaving two separate, visible tables (never silently fusing
unrelated ones) on any ambiguity."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from service.pdf_large_extraction import (  # noqa: E402
    _extract_leading_markdown_table,
    _extract_trailing_markdown_table,
    link_cross_page_tables,
)


@pytest.mark.unit
class TestExtractTrailingMarkdownTable:
    def test_page_ending_in_table_is_detected(self):
        text = "Intro text.\n\n|Item|Price|\n|---|---|\n|A|10|"
        result = _extract_trailing_markdown_table(text)
        assert result is not None
        before, table = result
        assert before.strip() == "Intro text."
        assert table == "|Item|Price|\n|---|---|\n|A|10|"

    def test_page_not_ending_in_table_returns_none(self):
        assert _extract_trailing_markdown_table("Just ordinary prose, no table here.") is None

    def test_too_short_to_be_a_real_table_returns_none(self):
        """A single '|...|' line (no separator row) isn't a real table."""
        assert _extract_trailing_markdown_table("Some text\n|lone pipe line|") is None


@pytest.mark.unit
class TestExtractLeadingMarkdownTable:
    def test_page_starting_with_table_is_detected(self):
        text = "|Item|Price|\n|---|---|\n|C|30|\n\nFooter text."
        result = _extract_leading_markdown_table(text)
        assert result is not None
        table, after = result
        assert table == "|Item|Price|\n|---|---|\n|C|30|"
        assert after.strip() == "Footer text."

    def test_page_not_starting_with_table_returns_none(self):
        assert _extract_leading_markdown_table("Just ordinary prose, no table here.") is None


@pytest.mark.unit
class TestLinkCrossPageTables:
    def test_matching_header_tables_are_merged(self):
        pages = [
            {"page_num": 1, "typhoon_markdown": "Intro.\n\n|Item|Price|\n|---|---|\n|A|10|\n|B|20|", "claude_markdown": ""},
            {"page_num": 2, "typhoon_markdown": "|Item|Price|\n|---|---|\n|C|30|\n\nFooter.", "claude_markdown": ""},
        ]
        result = link_cross_page_tables(pages)
        assert "|A|10|" in result[0]["typhoon_markdown"]
        assert "|B|20|" in result[0]["typhoon_markdown"]
        assert "|C|30|" in result[0]["typhoon_markdown"]
        assert result[0]["typhoon_markdown"].count("|Item|Price|") == 1  # header not duplicated
        assert "|C|30|" not in result[1]["typhoon_markdown"]  # removed from page 2
        assert "Footer." in result[1]["typhoon_markdown"]

    def test_mismatched_header_tables_are_left_unmerged(self):
        pages = [
            {"page_num": 1, "typhoon_markdown": "Intro.\n\n|Name|Fee|\n|---|---|\n|A|10|", "claude_markdown": ""},
            {"page_num": 2, "typhoon_markdown": "|Completely|Different|Columns|\n|---|---|---|\n|X|Y|Z|\n\nFooter.", "claude_markdown": ""},
        ]
        result = link_cross_page_tables(pages)
        # Both tables left exactly as they were — no merge attempted.
        assert "|Name|Fee|" in result[0]["typhoon_markdown"]
        assert "|Completely|Different|Columns|" in result[1]["typhoon_markdown"]

    def test_no_table_on_either_page_is_a_noop(self):
        pages = [
            {"page_num": 1, "typhoon_markdown": "Just ordinary page text.", "claude_markdown": ""},
            {"page_num": 2, "typhoon_markdown": "More ordinary page text.", "claude_markdown": ""},
        ]
        result = link_cross_page_tables(pages)
        assert result[0]["typhoon_markdown"] == "Just ordinary page text."
        assert result[1]["typhoon_markdown"] == "More ordinary page text."

    def test_table_only_at_page_end_no_table_at_next_start_is_a_noop(self):
        pages = [
            {"page_num": 1, "typhoon_markdown": "Intro.\n\n|Item|Price|\n|---|---|\n|A|10|", "claude_markdown": ""},
            {"page_num": 2, "typhoon_markdown": "Unrelated prose, no table.", "claude_markdown": ""},
        ]
        result = link_cross_page_tables(pages)
        assert "|A|10|" in result[0]["typhoon_markdown"]
        assert result[1]["typhoon_markdown"] == "Unrelated prose, no table."

    def test_three_page_table_chains_correctly(self):
        pages = [
            {"page_num": 1, "typhoon_markdown": "|Item|Price|\n|---|---|\n|A|10|", "claude_markdown": ""},
            {"page_num": 2, "typhoon_markdown": "|Item|Price|\n|---|---|\n|B|20|", "claude_markdown": ""},
            {"page_num": 3, "typhoon_markdown": "|Item|Price|\n|---|---|\n|C|30|", "claude_markdown": ""},
        ]
        result = link_cross_page_tables(pages)
        merged = result[0]["typhoon_markdown"]
        assert "|A|10|" in merged and "|B|20|" in merged and "|C|30|" in merged
        assert merged.count("|Item|Price|") == 1

    def test_none_pages_in_list_do_not_crash(self):
        pages = [None, {"page_num": 2, "typhoon_markdown": "text", "claude_markdown": ""}]
        result = link_cross_page_tables(pages)
        assert result[0] is None

    def test_single_page_list_is_a_noop(self):
        pages = [{"page_num": 1, "typhoon_markdown": "|Item|Price|\n|---|---|\n|A|10|", "claude_markdown": ""}]
        result = link_cross_page_tables(pages)
        assert result == pages

    def test_empty_list_does_not_crash(self):
        assert link_cross_page_tables([]) == []

"""Tests for utils/pdf_text_chunking.py — the structure-aware sub-chunking
function added 2026-09 to fix a real content-truncation bug: PDF book
chapters (real topics measured up to 19,056 chars) were being silently cut
to ~1,800-2,500 chars by service/data_loader.py's generic
_build_page_content() clamp, making everything past that point invisible to
both embedding-based retrieval and the LLM's answer context. Empirically
confirmed (direct OpenRouter API calls, bypassing the langchain wrapper)
that baai/bge-m3 honors its full 8192-token / ~42,300-Thai-char context with
no silent truncation below that — the old clamp was never a real model
constraint, just an unnecessarily conservative default reused from a
different (short Sheet-row) use case.
"""
from __future__ import annotations

import pytest

from utils.pdf_text_chunking import split_into_subchunks


@pytest.mark.unit
class TestSplitIntoSubchunksBasics:
    def test_empty_text_returns_empty_list(self):
        assert split_into_subchunks("", 100) == []
        assert split_into_subchunks("   ", 100) == []

    def test_text_under_cap_returned_unchanged_as_single_chunk(self):
        text = "เนื้อหาสั้นๆ ไม่เกิน cap"
        assert split_into_subchunks(text, 1000) == [text]

    def test_text_exactly_at_cap_is_single_chunk(self):
        text = "ก" * 500
        assert split_into_subchunks(text, 500) == [text]

    def test_realistic_chapter_length_stays_single_chunk_under_default_cap(self):
        # The actual longest real topic measured in live testing (19,056
        # chars) must fit as ONE chunk under the chosen default cap (24000) —
        # this is the core motivating case for picking that cap size.
        text = "เนื้อหาบทที่ยาวมาก " * 1000  # ~19,000 chars
        assert 18000 < len(text) < 20000
        chunks = split_into_subchunks(text, 24000)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()


@pytest.mark.unit
class TestSplitOnHeaders:
    def test_splits_on_markdown_headers_when_over_cap(self):
        text = (
            "# บทที่ 1\n" + ("เนื้อหาส่วนแรก " * 50) +
            "\n## บทที่ 2\n" + ("เนื้อหาส่วนสอง " * 50)
        )
        max_chars = len("# บทที่ 1\n" + ("เนื้อหาส่วนแรก " * 50)) + 10
        chunks = split_into_subchunks(text, max_chars)
        assert len(chunks) >= 2
        assert "บทที่ 1" in chunks[0]

    def test_no_content_lost_across_chunks(self):
        text = (
            "# หัวข้อ A\n" + ("ก" * 300) +
            "\n# หัวข้อ B\n" + ("ข" * 300) +
            "\n# หัวข้อ C\n" + ("ค" * 300)
        )
        chunks = split_into_subchunks(text, 400)
        assert len(chunks) > 1
        rejoined = "".join(chunks)
        # every distinct marker character run must survive somewhere in the output
        assert "ก" * 300 in rejoined
        assert "ข" * 300 in rejoined
        assert "ค" * 300 in rejoined

    def test_small_adjacent_sections_get_packed_together(self):
        text = "# A\nสั้น\n# B\nสั้น\n# C\nสั้น"
        chunks = split_into_subchunks(text, 1000)
        # all fits comfortably under cap combined with headers, but since the
        # whole text is under max_chars it should come back as ONE chunk
        assert len(chunks) == 1


@pytest.mark.unit
class TestOversizedSectionFallback:
    def test_single_oversized_section_falls_back_to_paragraph_split(self):
        para_a = "ย่อหน้าที่หนึ่งยาวมาก " * 40
        para_b = "ย่อหน้าที่สองยาวมาก " * 40
        text = f"# หัวข้อเดียวที่ยาวมาก\n{para_a}\n\n{para_b}"
        max_chars = len(para_a) + 20
        chunks = split_into_subchunks(text, max_chars)
        assert all(len(c) <= max_chars + 50 for c in chunks)  # small slack for header repeats
        rejoined = "".join(chunks)
        assert para_a.strip() in rejoined
        assert para_b.strip() in rejoined

    def test_single_giant_paragraph_with_no_breaks_hard_slices_without_losing_content(self):
        text = "ก" * 10000  # no headers, no paragraph breaks at all
        max_chars = 3000
        chunks = split_into_subchunks(text, max_chars)
        assert len(chunks) == 4  # 10000 / 3000 -> 4 pieces
        assert all(len(c) <= max_chars for c in chunks)
        assert "".join(chunks) == text

    def test_no_chunk_ever_exceeds_max_chars_across_mixed_content(self):
        text = (
            "# หัวข้อ 1\n" + ("เนื้อหา " * 5) +
            "\n# หัวข้อ 2 ยาวมาก\n" + ("เนื้อหายาว " * 2000) +  # oversized section
            "\n# หัวข้อ 3\n" + ("เนื้อหา " * 5)
        )
        max_chars = 5000
        chunks = split_into_subchunks(text, max_chars)
        assert all(len(c) <= max_chars for c in chunks)
        assert len(chunks) > 1

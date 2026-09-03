"""
Unit tests for utils.text_patterns.topic_label_matches_query.

Background: persona_supervisor.py's "chapter retrieval" mechanism fetches
ALL docs for a main_topic/sub_topic/operation_topic via a Chroma metadata
filter (bypassing semantic top-K, which can silently drop docs below the
cutoff) whenever the user's query names that label. Before 2026-09,
main_topic's matcher only checked plain substring containment against the
(possibly LLM-rewritten) query, while sub_topic's matcher additionally
checked the raw pre-rewrite user message and an ASCII-identifier fallback —
an asymmetry that made main_topic chapter-fetch miss more often than
sub_topic for the identical class of paraphrased query. Both call sites now
share this one function. No LLM/Chroma connection required.
"""
import pytest

from utils.text_patterns import topic_label_matches_query


@pytest.mark.unit
class TestTopicLabelMatchesQuery:

    def test_exact_substring_in_query(self):
        assert topic_label_matches_query(
            "กลยุทธ์ด้านการสื่อสาร", "อยากรู้เรื่องกลยุทธ์ด้านการสื่อสารหน่อยค่ะ", ""
        )

    def test_no_match_returns_false(self):
        assert not topic_label_matches_query(
            "กลยุทธ์ด้านการสื่อสาร", "ขอใบอนุญาตขายสุราหน่อยค่ะ", "ขอใบอนุญาตขายสุราหน่อยค่ะ"
        )

    def test_label_missing_from_rewritten_query_but_present_in_raw_message(self):
        # The LLM rewriter dropped "แบบ" from the query but the raw user message
        # (state.messages[-1]) still has the full label — this is exactly the
        # asymmetry main_topic previously missed and sub_topic already handled.
        assert topic_label_matches_query(
            "การตลาดแบบ B2B",
            query_lower="การตลาด b2b สำหรับร้านอาหาร",
            raw_human_lower="การตลาดแบบ b2b คืออะไร",
        )

    def test_ascii_identifier_fallback_matches_when_thai_particles_dropped(self):
        # Neither the rewritten query nor the raw message contains the full Thai
        # label, but the distinctive ASCII token "B2B" is present in both —
        # the ASCII-id fallback should still fire.
        assert topic_label_matches_query(
            "การตลาดแบบ B2B",
            query_lower="อยากรู้เรื่อง b2b หน่อย",
            raw_human_lower="b2b คืออะไร",
        )

    def test_ascii_identifier_fallback_requires_all_tokens(self):
        # Label has two distinct ASCII tokens; only one appears in the query —
        # should NOT match (all tokens must be present, not just one).
        assert not topic_label_matches_query(
            "ระบบ ABC และ XYZ",
            query_lower="อยากรู้เรื่อง abc หน่อย",
            raw_human_lower="",
        )

    def test_no_ascii_tokens_no_fallback_available(self):
        # Pure-Thai label with no ASCII tokens: if neither substring check
        # matches, there is nothing left to fall back on.
        assert not topic_label_matches_query(
            "การบริหารจัดการพนักงาน", "ขอสอบถามเรื่องภาษีหน่อยค่ะ", "ขอสอบถามเรื่องภาษีหน่อยค่ะ"
        )

    def test_empty_raw_human_lower_does_not_crash(self):
        # Default raw_human_lower="" (e.g. called with only 2 args) must not error.
        assert not topic_label_matches_query("การตลาดแบบ B2B", "ไม่เกี่ยวข้อง")

    def test_case_insensitive_ascii_match(self):
        assert topic_label_matches_query(
            "การตลาดแบบ b2b", query_lower="b2b คืออะไร", raw_human_lower=""
        )

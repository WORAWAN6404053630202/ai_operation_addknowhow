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


@pytest.mark.unit
class TestTopicLabelMatchesQueryStrongOnly:
    """2026-10 fix (live-reproduced): a corpus where many topics are all
    chapters of the same product/system (PDF-ingested manuals tagged
    "(DBD Biz Regist)" on nearly every chapter) made the ASCII-identifier
    fallback match far too broadly — the shared product name ASCII-matched
    every chapter about it, not just the one the user actually meant.
    strong_only=True restricts to checks (1)-(3) (literal/nominalizer-
    stripped substring, specific to THIS label), skipping the ASCII
    fallback entirely — callers use it to avoid letting a weak, corpus-
    wide ASCII match out-compete a genuine substring match in a
    max(label, key=len) tie-break.
    """

    def test_strong_only_still_matches_literal_substring(self):
        assert topic_label_matches_query(
            "สมัครสมาชิกผู้ใช้งานระบบ",
            query_lower="สมัครสมาชิกผู้ใช้งานระบบ dbd biz regist ต้องทำยังไง",
            raw_human_lower="",
            strong_only=True,
        )

    def test_strong_only_rejects_ascii_only_match(self):
        # Live-reproduced shape: label only shares the generic product-name
        # ASCII tokens with the query, nothing else — strong_only must reject
        # this even though the lenient (default) call accepts it.
        label = "จดทะเบียนเลิกบริษัท/ห้างหุ้นส่วนผ่านระบบจดทะเบียนนิติบุคคลดิจิทัล (DBD Biz Regist)"
        query = "สมัครสมาชิกผู้ใช้งานระบบ dbd biz regist ต้องทำยังไง"
        assert topic_label_matches_query(label, query, "")  # lenient: still True
        assert not topic_label_matches_query(label, query, "", strong_only=True)

    def test_strong_only_still_rejects_when_lenient_also_rejects(self):
        assert not topic_label_matches_query(
            "กลยุทธ์ด้านการสื่อสาร", "ขอใบอนุญาตขายสุราหน่อยค่ะ", "", strong_only=True,
        )

    def test_strong_only_still_matches_nominalizer_stripped_form(self):
        # Check (3) — nominalizer-stripped substring — must still work under
        # strong_only, since it's specific to THIS label, not a shared token.
        assert topic_label_matches_query(
            "การจดทะเบียนพาณิชย์",
            query_lower="อยากจดทะเบียนพาณิชย์ต้องทำยังไง",
            raw_human_lower="",
            strong_only=True,
        )

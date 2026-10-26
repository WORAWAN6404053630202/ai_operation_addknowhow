"""
Regression test for utils.text_patterns.nonreg_chapter_disagreement_should_reject
— 2026-10 fix, part of a broader data-integrity audit.

Background: persona_supervisor.py's sub_topic exact-match site has two
checks in a row that reject a confident exact-match chapter fetch when
retrieval evidence disagrees. The FIRST (regulatory-dominance) already had a
score floor, added 2026-09 after a live-reproduced false positive (the SME
income-tax-bracket case: a "regulatory" top hit scoring only 0.55-0.57 — a
coincidental keyword overlap, not a genuinely better answer — incorrectly
rejected the business_guide chapter that held the real answer, breaking
answer generation entirely). The SECOND check ("disagreement": top hit's
main_topic merely differs from the exact match) had NO score floor at all —
this fix adds the same one, closing the identical failure mode for this
check specifically. An identical unfloored check was deliberately never
ported to persona_supervisor.py's op_topic call site after live-testing
showed it caused a worse failure there (total answer-generation failure
instead of a partial-but-present answer) — gating this site with a floor,
rather than leaving it unfloored or removing it, keeps its genuine catches
while closing the coincidental-weak-hit risk.
"""
import pytest

from utils.text_patterns import nonreg_chapter_disagreement_should_reject


@pytest.mark.unit
class TestNonregChapterDisagreementShouldReject:
    def test_confident_disagreement_above_floor_rejects(self):
        assert nonreg_chapter_disagreement_should_reject(
            "หัวข้อที่ขัดแย้ง", "หัวข้อเดิม", 0.80, min_score=0.70,
        ) is True

    def test_weak_coincidental_disagreement_below_floor_does_not_reject(self):
        """The exact failure mode this fix closes: a low-confidence top hit
        (the SME tax-bracket case scored 0.55-0.57) must NOT trigger
        rejection just because its main_topic differs."""
        assert nonreg_chapter_disagreement_should_reject(
            "หัวข้ออื่นที่บังเอิญคล้าย", "หัวข้อเดิม", 0.56, min_score=0.70,
        ) is False

    def test_exactly_at_floor_rejects(self):
        assert nonreg_chapter_disagreement_should_reject(
            "หัวข้อที่ขัดแย้ง", "หัวข้อเดิม", 0.70, min_score=0.70,
        ) is True

    def test_agreement_never_rejects_regardless_of_score(self):
        assert nonreg_chapter_disagreement_should_reject(
            "หัวข้อเดิม", "หัวข้อเดิม", 0.95, min_score=0.70,
        ) is False

    def test_no_nonreg_top_hit_at_all_never_rejects(self):
        """sem_top_main_topic empty means no marketing/business_guide doc
        appeared in the top-5 at all — nothing to disagree with."""
        assert nonreg_chapter_disagreement_should_reject(
            "", "หัวข้อเดิม", 0.95, min_score=0.70,
        ) is False

    def test_default_min_score_matches_the_shared_calibrated_constant(self):
        from utils.text_patterns import NONREG_CHAPTER_REGULATORY_DOMINANCE_MIN_SCORE
        assert nonreg_chapter_disagreement_should_reject(
            "หัวข้ออื่น", "หัวข้อเดิม", NONREG_CHAPTER_REGULATORY_DOMINANCE_MIN_SCORE,
        ) is True
        assert nonreg_chapter_disagreement_should_reject(
            "หัวข้ออื่น", "หัวข้อเดิม", NONREG_CHAPTER_REGULATORY_DOMINANCE_MIN_SCORE - 0.01,
        ) is False

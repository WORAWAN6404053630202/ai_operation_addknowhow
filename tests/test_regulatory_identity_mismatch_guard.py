"""
Regression test for utils.text_patterns.regulatory_identity_mismatch_should_reject
— 2026-10 addition, Phase-0 codebase audit.

Background: QA-reproduced cross-domain retrieval contamination (e.g. DBD
commerce-registration procedure content answering a VAT registration
question) — the existing dominance guard (nonreg_chapter_disagreement_
should_reject) and reranker.filter_by_category_concentration both only
distinguish regulatory from non-regulatory content; neither ever compares
license_type/department WITHIN regulatory content, which is exactly where
this contamination pattern lives. This guard is deliberately the same shape
as its sibling (bool(a) and bool(b) and a != b and score >= floor) and
reuses its sibling's score floor as a starting point, for the same reason
the sibling has one at all: an unfloored "these two values merely differ"
check is the known-unsafe degenerate case documented on the sibling's own
test file (a coincidental weak match would reject a genuinely good answer).
"""
import pytest

from utils.text_patterns import regulatory_identity_mismatch_should_reject


@pytest.mark.unit
class TestRegulatoryIdentityMismatchShouldReject:
    def test_confident_mismatch_above_floor_rejects(self):
        assert regulatory_identity_mismatch_should_reject(
            "ภาษีมูลค่าเพิ่ม", "ทะเบียนพาณิชย์", 0.80, min_score=0.70,
        ) is True

    def test_weak_coincidental_mismatch_below_floor_does_not_reject(self):
        assert regulatory_identity_mismatch_should_reject(
            "ภาษีมูลค่าเพิ่ม", "ทะเบียนพาณิชย์", 0.56, min_score=0.70,
        ) is False

    def test_exactly_at_floor_rejects(self):
        assert regulatory_identity_mismatch_should_reject(
            "ภาษีมูลค่าเพิ่ม", "ทะเบียนพาณิชย์", 0.70, min_score=0.70,
        ) is True

    def test_agreement_never_rejects_regardless_of_score(self):
        assert regulatory_identity_mismatch_should_reject(
            "ภาษีมูลค่าเพิ่ม", "ภาษีมูลค่าเพิ่ม", 0.95, min_score=0.70,
        ) is False

    def test_blank_top_value_never_rejects(self):
        """Nothing confident to compare against — most non-regulatory or
        unlabeled docs have no license_type/department at all."""
        assert regulatory_identity_mismatch_should_reject(
            "", "ทะเบียนพาณิชย์", 0.95, min_score=0.70,
        ) is False

    def test_blank_candidate_value_never_rejects(self):
        assert regulatory_identity_mismatch_should_reject(
            "ภาษีมูลค่าเพิ่ม", "", 0.95, min_score=0.70,
        ) is False

    def test_default_min_score_matches_the_shared_calibrated_constant(self):
        from utils.text_patterns import NONREG_CHAPTER_REGULATORY_DOMINANCE_MIN_SCORE
        assert regulatory_identity_mismatch_should_reject(
            "ภาษีมูลค่าเพิ่ม", "ทะเบียนพาณิชย์", NONREG_CHAPTER_REGULATORY_DOMINANCE_MIN_SCORE,
        ) is True
        assert regulatory_identity_mismatch_should_reject(
            "ภาษีมูลค่าเพิ่ม", "ทะเบียนพาณิชย์", NONREG_CHAPTER_REGULATORY_DOMINANCE_MIN_SCORE - 0.01,
        ) is False

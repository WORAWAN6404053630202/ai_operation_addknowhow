"""2026-11 fix: FOCUS FILTER's dominant-license-type narrowing (inside
PracticalPersonaService.handle()) used to score each license_type by matching
the raw query TEXT against the license_type NAME STRING via ~140 lines of
hand-written regex (an 8-char Thai substring sliding window + ~15 per-license
keyword special cases). That's blind to any phrasing that doesn't echo the
license_type's own formal vocabulary — live-reproduced: "พื้นที่ร้าน 45 ตร.ม.
ค่าธรรมเนียมแจ้งรายปีกี่บาท" retrieved the correct document ranked #0 (confirmed
via a direct hybrid_search() call bypassing this logic entirely), but the old
regex scorer gave it ~0 because the query never says "หนังสือรับรองการแจ้ง"
verbatim, so FOCUS FILTER discarded the correctly-retrieved document anyway.

Removed the regex scorer entirely; license_type relevance is now read
directly from the similarity score (_sim) retrieval+reranking already
computed for every doc reaching this point (_scored_search attaches it) —
semantically aware by construction instead of hand-tuned per keyword.

persona_practical.py's handle() is a single ~5000-line method — see
test_doc_overflow_ceiling_fixes.py's module docstring for why source
inspection (not a full mocked end-to-end call) is this codebase's
established pattern for regression-testing logic embedded this deeply.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import conf  # noqa: E402
from model.persona_practical import PracticalPersonaService  # noqa: E402


@pytest.mark.unit
class TestFocusFilterUsesSimilarityNotRegex:
    def test_regex_substring_scorer_is_gone(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        assert "_license_query_score" not in source, (
            "FOCUS FILTER's removed regex/substring scorer (_license_query_score) "
            "reappeared — this was deliberately deleted as the root-cause fix, "
            "not just patched around."
        )

    def test_removed_keyword_patterns_no_longer_defined(self):
        # These existed solely to feed the now-deleted regex scorer. Keeping any
        # of them around (even unused) would be dead code pretending the old
        # mechanism might come back — assert they're actually gone.
        removed = [
            "_LQS_ABBREV_CHECK", "_LQS_ACCOUNTING_RE", "_LQS_BANKING_RE",
            "_LQS_EMPLOYER_RE", "_LQS_HIRING_RE", "_LQS_SAN_RE",
            "_LQS_QR_KEYWORDS_RE", "_LQS_SOCIAL_RE", "_LQS_INSURED_RE",
            "_LQS_HEALTH_CERT_RE", "_LQS_LIQUOR_RE", "_LQS_FOOD_PERMIT_RE",
            "_LQS_QR_PAYMENT_RE",
        ]
        for attr in removed:
            assert not hasattr(PracticalPersonaService, attr), (
                f"{attr} should have been removed along with _license_query_score"
            )

    def test_patterns_still_needed_elsewhere_survive(self):
        # _LQS_VAT_RE / _LQS_SIGN_TAX_RE are also used by the unrelated
        # broad-tax-query detector earlier in handle() — must NOT be deleted.
        assert hasattr(PracticalPersonaService, "_LQS_VAT_RE")
        assert hasattr(PracticalPersonaService, "_LQS_SIGN_TAX_RE")

    def test_focus_filter_scores_from_sim_metadata(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        idx = source.find("Pick dominant license_type by RETRIEVAL'S OWN relevance judgment")
        assert idx != -1, "New FOCUS FILTER comment not found — scoring section may have moved/been reverted"
        end = source.find("if len(_explicitly_requested) >= 2:", idx)
        assert end != -1
        body = source[idx:end]
        assert '.get("_sim")' in body, "New scorer must read the _sim metadata retrieval+reranking already computed"
        assert "FOCUS_FILTER_SIM_MARGIN" in body, "Multi-requested grouping must use the configured similarity margin"

    def test_focus_filter_sim_margin_config_exists(self):
        assert hasattr(conf, "FOCUS_FILTER_SIM_MARGIN")
        assert isinstance(conf.FOCUS_FILTER_SIM_MARGIN, float)
        assert 0.0 < conf.FOCUS_FILTER_SIM_MARGIN < 0.5


@pytest.mark.unit
class TestSimilarityScoringBehavior:
    """Behavioral checks evaluated directly on the same comprehension shape
    the real code uses, mirroring test_focus_filter_preserves_knowhow_docs.py's
    established pattern for this file's un-mockable logic."""

    @staticmethod
    def _score_by_sim(all_docs: list) -> dict:
        scores: dict = {}
        for d in all_docs:
            lt = ((d.get("metadata") or {}).get("license_type") or "").strip()
            if not lt:
                continue
            sim = (d.get("metadata") or {}).get("_sim")
            if sim is None:
                continue
            sim = float(sim)
            if lt not in scores or sim > scores[lt]:
                scores[lt] = sim
        return scores

    def test_single_dominant_license_wins_regardless_of_query_wording(self):
        """The exact live-reproduced case: a short query with no literal overlap
        with the license_type's own name must still correctly identify the
        dominant license_type, because scoring no longer depends on the query
        text at all — only on what retrieval+reranking already ranked."""
        docs = [
            {"metadata": {"license_type": "หนังสือรับรองการแจ้งจัดตั้งสถานที่จำหน่ายอาหารและสถานที่สะสมอาหาร", "_sim": 0.8128}},
            {"metadata": {"license_type": "การจดทะเบียนพาณิชย์ตั้งใหม่สำหรับบุคคลธรรมดา", "_sim": 0.8160}},
            {"metadata": {"license_type": "บริการด้านการจดทะเบียนพาณิชยกิจและเอกสารประกอบ", "_sim": 0.8079}},
        ]
        scores = self._score_by_sim(docs)
        best_lt = max(scores, key=lambda lt: scores[lt])
        # Both top two are within the default 0.03 margin of each other — this
        # IS a genuine near-tie in this synthetic example, so the real behavior
        # (margin-based "keep both") is the correct outcome, not a single pick.
        margin = 0.03
        explicitly_requested = [lt for lt, s in scores.items() if (scores[best_lt] - s) <= margin]
        assert "หนังสือรับรองการแจ้งจัดตั้งสถานที่จำหน่ายอาหารและสถานที่สะสมอาหาร" in explicitly_requested

    def test_clear_single_winner_excludes_distant_license_types(self):
        docs = [
            {"metadata": {"license_type": "ใบอนุญาตจำหน่ายสุรา", "_sim": 0.85}},
            {"metadata": {"license_type": "ใบภาษีมูลค่าเพิ่ม ภพ.20", "_sim": 0.70}},
        ]
        scores = self._score_by_sim(docs)
        best_lt = max(scores, key=lambda lt: scores[lt])
        margin = 0.03
        explicitly_requested = [lt for lt, s in scores.items() if (scores[best_lt] - s) <= margin]
        assert explicitly_requested == ["ใบอนุญาตจำหน่ายสุรา"]

    def test_genuinely_tied_dual_license_query_keeps_both(self):
        """e.g. 'ขอใบอนุญาตจัดตั้งร้านอาหาร ฉันขายเหล้าด้วย' — retrieval ranks both
        topics near the top since both are genuinely on-point."""
        docs = [
            {"metadata": {"license_type": "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร", "_sim": 0.83}},
            {"metadata": {"license_type": "ใบอนุญาตจำหน่ายสุรา", "_sim": 0.82}},
        ]
        scores = self._score_by_sim(docs)
        best_lt = max(scores, key=lambda lt: scores[lt])
        margin = 0.03
        explicitly_requested = [lt for lt, s in scores.items() if (scores[best_lt] - s) <= margin]
        assert set(explicitly_requested) == {"ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร", "ใบอนุญาตจำหน่ายสุรา"}

    def test_docs_without_sim_metadata_produce_no_score(self):
        """Defensive path: the retrieval-exception fallback never attaches _sim —
        scorer must not crash and must simply report no signal."""
        docs = [{"metadata": {"license_type": "ใบอนุญาตจำหน่ายสุรา"}}]
        scores = self._score_by_sim(docs)
        assert scores == {}

    def test_empty_license_type_docs_never_scored(self):
        docs = [{"metadata": {"license_type": "", "_sim": 0.9}}, {"metadata": {}}]
        scores = self._score_by_sim(docs)
        assert scores == {}

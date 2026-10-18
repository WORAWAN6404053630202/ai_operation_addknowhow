"""Tests for service/pdf_supersede_resolver.py — the automated (no-human-
review) content supersede resolver added 2026-10 for the PDF-as-sole-source
architecture. Covers the decision-aggregation logic deterministically by
mocking the 2 candidate-signal functions and the LLM judgment call — no
real Chroma, no real network/LLM calls, safe to run anytime.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from service import pdf_supersede_resolver as resolver  # noqa: E402


@pytest.mark.unit
class TestResolveDecision:
    def test_empty_pool_is_new(self):
        with patch.object(resolver, "_embedding_candidates", return_value={}), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}):
            decision = resolver.resolve_regulatory_topic(
                {"department": "สำนักงานประกันสังคม", "license_type": "ทะเบียนนายจ้าง", "answer_guideline": "เนื้อหา"},
                "new_law.pdf",
            )
        assert decision.decision == "new"
        assert decision.matched_doc_ids == []

    def test_no_comparable_text_is_new_without_calling_chroma(self):
        with patch.object(resolver, "_embedding_candidates") as mock_emb:
            decision = resolver.resolve_regulatory_topic({"department": "", "license_type": ""}, "x.pdf")
        assert decision.decision == "new"
        mock_emb.assert_not_called()

    def test_matched_and_llm_says_duplicate_skips(self):
        pool = {"doc-1": {"score": 0.95, "metadata": {"department": "สำนักงานประกันสังคม", "license_type": "ทะเบียนนายจ้าง"}, "content": "เดิม"}}
        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", return_value={
                 "doc-1": {"is_same_topic": True, "content_status": "duplicate", "reasoning": "same"},
             }):
            decision = resolver.resolve_regulatory_topic(
                {"department": "สำนักงานประกันสังคม", "license_type": "ทะเบียนนายจ้าง", "answer_guideline": "เนื้อหาใหม่"},
                "reupload.pdf",
            )
        assert decision.decision == "duplicate"
        assert decision.matched_doc_ids == ["doc-1"]

    def test_matched_and_llm_says_update_returns_matched_ids_for_retirement(self):
        pool = {"doc-2": {"score": 0.91, "metadata": {"department": "สำนักงานประกันสังคม", "license_type": "ทะเบียนนายจ้าง"}, "content": "เดิม"}}
        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", return_value={
                 "doc-2": {"is_same_topic": True, "content_status": "update", "reasoning": "fee changed"},
             }):
            decision = resolver.resolve_regulatory_topic(
                {"department": "สำนักงานประกันสังคม", "license_type": "ทะเบียนนายจ้าง", "answer_guideline": "ค่าธรรมเนียมใหม่"},
                "updated_law.pdf",
            )
        assert decision.decision == "update"
        assert decision.matched_doc_ids == ["doc-2"]

    def test_llm_failure_defaults_to_new_never_infers_update(self):
        """The resolver's own _llm_judge already swallows exceptions and
        returns {} — this test pins that {} (no judgments at all) must
        resolve to "new", never "update"/"duplicate", even though a
        candidate pool existed."""
        pool = {"doc-3": {"score": 0.9, "metadata": {"department": "X", "license_type": "Y"}, "content": "c"}}
        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", return_value={}):
            decision = resolver.resolve_regulatory_topic(
                {"department": "X", "license_type": "Y", "answer_guideline": "z"}, "f.pdf",
            )
        assert decision.decision == "new"

    def test_llm_says_not_same_topic_is_new_despite_signal_hits(self):
        pool = {"doc-4": {"score": 0.9, "metadata": {"department": "X", "license_type": "Y"}, "content": "c"}}
        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", return_value={
                 "doc-4": {"is_same_topic": False, "content_status": None, "reasoning": "actually different"},
             }):
            decision = resolver.resolve_regulatory_topic(
                {"department": "X", "license_type": "Y", "answer_guideline": "z"}, "f.pdf",
            )
        assert decision.decision == "new"

    def test_identity_only_match_still_reaches_llm_judgment(self):
        """A candidate surfaced ONLY by the identity-fuzzy signal (not
        embedding) must still be sent to the LLM for judgment — signals are
        unioned, not intersected."""
        pool = {"doc-5": {"score": None, "metadata": {"department": "X", "license_type": "Y"}, "content": "c"}}
        with patch.object(resolver, "_embedding_candidates", return_value={}), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value=pool) as mock_id, \
             patch.object(resolver, "_llm_judge", return_value={
                 "doc-5": {"is_same_topic": True, "content_status": "update", "reasoning": "r"},
             }) as mock_llm:
            decision = resolver.resolve_regulatory_topic(
                {"department": "X", "license_type": "Y", "answer_guideline": "z"}, "f.pdf",
            )
        mock_id.assert_called_once()
        mock_llm.assert_called_once()
        assert decision.decision == "update"


@pytest.mark.unit
class TestKnowhowCatchAllExemption:
    def test_catchall_topic_is_always_new_without_touching_chroma(self):
        with patch.object(resolver, "_embedding_candidates") as mock_emb:
            decision = resolver.resolve_knowhow_topic(
                {
                    "main_topic": resolver.KNOWHOW_CATCHALL_MAIN_TOPIC,
                    "sub_topic": "ตรวจสอบและจัดหมวดหมู่ด้วยตนเอง",
                    "summary": "", "category": "", "full_text": "เนื้อหาที่เหลือ",
                },
                "f.pdf",
            )
        assert decision.decision == "new"
        mock_emb.assert_not_called()

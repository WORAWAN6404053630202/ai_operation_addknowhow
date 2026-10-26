"""Tests for model/persona_academic.py's _cap_docs_with_nonreg_aggregation —
2026-10 fix, data-integrity audit.

Background: the inline version of this logic required ALL of the first 20
docs sent to the Academic-mode LLM call to lack license_type before
aggregating any sub_topic with >12 rows. One mixed-in regulatory doc (a
realistic, common case — a broad Academic-mode query legitimately pulling
both a regulatory row and several marketing/business_guide rows) disabled
the safety net entirely, and the plain docs[:_MAX_DOCS_ACADEMIC] slice
downstream then silently truncated the non-reg sub_topic with no warning —
the exact failure the safety net's own comment says it exists to prevent,
just gated on a stricter condition than the actual risk.

Fix: partition into regulatory/non-regulatory subsets and aggregate only
the non-regulatory one, regardless of whether regulatory docs are mixed in.
"""
from __future__ import annotations

import pytest

from model.persona_academic import _cap_docs_with_nonreg_aggregation


def _reg_doc(license_type: str) -> dict:
    return {"content": f"เนื้อหาใบอนุญาต {license_type}", "metadata": {
        "license_type": license_type, "data_type": "regulatory",
    }}


def _nonreg_doc(sub_topic: str, content_type: str, answer_guideline: str) -> dict:
    return {"content": f"{content_type}: {answer_guideline}", "metadata": {
        "sub_topic": sub_topic, "data_type": "business_guide",
        "content_type": content_type, "answer_guideline": answer_guideline,
    }}


@pytest.mark.unit
class TestMixedRegulatoryAndNonRegDoesNotDisableAggregation:
    def test_one_mixed_in_regulatory_doc_still_aggregates_the_nonreg_subset(self):
        """The exact bug: before this fix, a SINGLE regulatory doc among
        the batch disabled aggregation entirely, silently truncating the
        14-row non-reg sub_topic down to whatever fit under max_docs=12."""
        docs = [_reg_doc("ใบอนุญาตประกอบกิจการ")] + [
            _nonreg_doc("กลยุทธ์การตลาด", f"ประเภท{i}", f"เนื้อหา{i}") for i in range(14)
        ]
        result = _cap_docs_with_nonreg_aggregation(docs, max_docs=12)
        # The regulatory doc survives untouched, and the 14 non-reg docs
        # collapse into 1 aggregated overview doc — nothing silently lost.
        assert any(d["metadata"].get("license_type") == "ใบอนุญาตประกอบกิจการ" for d in result)
        nonreg_result = [d for d in result if not d["metadata"].get("license_type")]
        assert len(nonreg_result) == 1
        assert "14" in nonreg_result[0]["metadata"]["answer_guideline"]

    def test_pure_nonreg_batch_still_aggregates_same_as_before(self):
        """Regression guard: the original pure-non-reg case must keep
        working exactly as it did before this fix."""
        docs = [_nonreg_doc("กลยุทธ์การตลาด", f"ประเภท{i}", f"เนื้อหา{i}") for i in range(14)]
        result = _cap_docs_with_nonreg_aggregation(docs, max_docs=12)
        assert len(result) == 1
        assert "14" in result[0]["metadata"]["answer_guideline"]

    def test_pure_regulatory_batch_is_returned_unchanged(self):
        """No marketing/business_guide signal at all → no-op, same as
        before this fix for the pure-regulatory case."""
        docs = [_reg_doc(f"ใบอนุญาต{i}") for i in range(14)]
        result = _cap_docs_with_nonreg_aggregation(docs, max_docs=12)
        assert result == docs

    def test_count_at_or_below_max_docs_is_a_noop(self):
        docs = [_reg_doc("X")] + [_nonreg_doc("Y", "ประเภท1", "เนื้อหา1")]
        result = _cap_docs_with_nonreg_aggregation(docs, max_docs=12)
        assert result == docs

    def test_regulatory_docs_are_never_aggregated_themselves(self):
        """Regulatory docs aren't sub_topic-groupable content — even when
        several share no identifying sub_topic, they must pass through
        individually, never collapsed into an overview doc."""
        docs = [_reg_doc(f"ใบอนุญาต{i}") for i in range(10)] + [
            _nonreg_doc("กลยุทธ์การตลาด", f"ประเภท{i}", f"เนื้อหา{i}") for i in range(14)
        ]
        result = _cap_docs_with_nonreg_aggregation(docs, max_docs=12)
        reg_result = [d for d in result if d["metadata"].get("license_type")]
        assert len(reg_result) == 10
        assert {d["metadata"]["license_type"] for d in reg_result} == {f"ใบอนุญาต{i}" for i in range(10)}

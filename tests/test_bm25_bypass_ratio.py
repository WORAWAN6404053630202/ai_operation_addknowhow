"""
Regression test for utils.hybrid_retriever._rrf_fuse's BM25-bypass ratio —
2026-09 bug fix, follow-up to the pdf_chunking_truncation_and_retrieval_fixes
chain (found while investigating the Academic-mode Top Line/Bottom Line bug,
separately calibrated and shipped afterward per explicit user request).

Background: a document found ONLY by BM25 (missing from Dense results
entirely) used to get an UNCONDITIONAL "_bm25_hit" flag, which every
downstream similarity-floor filter (persona_practical.py, persona_academic.py)
treats as "always trust this doc, skip the floor entirely" — regardless of how
weak or coincidental the actual BM25 match was. Its real BM25 score was
computed then discarded (`for rank, (doc, _score) in enumerate(bm25_pairs)`).

Live-reproduced: for the query "...Top Line...Bottom Line...", an unrelated
"ประเภทของการวางแผนผังโรงงาน" (factory layout) doc matched only via the bare
conjunction "กับ" appearing in its content by chance — real BM25 score 23.4,
but the query's own TOP BM25 score (the correct doc) was 103.3. This
coincidental match got the full unconditional bypass anyway.

An ABSOLUTE BM25 score threshold doesn't work — raw BM25Okapi scores have no
shared scale across different queries (measured top-scores of 12.7, 19.2,
22.8, 27.4, 103.3 across different real queries in this corpus). A RELATIVE
threshold does: across 7 real query/doc pairs measured live, every
CONFIRMED-relevant match (including secondary/non-#1-ranked but still
genuinely on-topic rows) scored at least 0.556 of its own query's top BM25
score. The one confirmed coincidental false positive scored only 0.227.
conf.BM25_BYPASS_MIN_RATIO=0.35 sits in that gap.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from utils.hybrid_retriever import _rrf_fuse


def _doc(sub_topic: str, body: str = "") -> Document:
    return Document(page_content=f"{sub_topic} {body}", metadata={"sub_topic": sub_topic})


@pytest.mark.unit
class TestBm25BypassRatio:
    def test_confident_bm25_only_doc_gets_bypass_flag(self):
        """A BM25-only doc scoring at/near the query's own top BM25 score
        (the common shape for genuine exact-code/rare-term matches) must
        still get the bypass — this fix must not break the mechanism's
        original purpose."""
        dense_pairs = []
        bm25_pairs = [(_doc("ใบภาษีมูลค่าเพิ่ม ภพ.20"), 22.79)]
        fused = _rrf_fuse(dense_pairs, bm25_pairs, rrf_k=60, bm25_bypass_min_ratio=0.35)
        assert fused[0][0].metadata.get("_bm25_hit") is True

    def test_secondary_relevant_doc_below_top_but_above_ratio_gets_bypass(self):
        """Pins the real measured case: a genuinely relevant doc that is NOT
        the #1 BM25 result for its query (0.676 of top) must still pass —
        multiple valid rows for one topic naturally have different scores."""
        dense_pairs = []
        bm25_pairs = [
            (_doc("ใบทะเบียนพาณิชย์ - row A"), 12.74),
            (_doc("ใบทะเบียนพาณิชย์ - row B"), 8.62),  # ratio 0.676
        ]
        fused = _rrf_fuse(dense_pairs, bm25_pairs, rrf_k=60, bm25_bypass_min_ratio=0.35)
        assert all(d.metadata.get("_bm25_hit") is True for d, _ in fused)

    def test_coincidental_weak_match_does_not_get_bypass(self):
        """Pins the exact live-reproduced false positive: a BM25-only doc
        scoring far below the query's top score (0.227 ratio) must NOT get
        an unconditional pass anymore."""
        dense_pairs = [(_doc("การวิเคราะห์รายได้และกำไร"), 0.577)]  # found by Dense too
        bm25_pairs = [
            (_doc("การวิเคราะห์รายได้และกำไร"), 103.27),  # same doc, also BM25's top hit
            (_doc("ประเภทของการวางแผนผังโรงงาน"), 23.41),  # BM25-only, coincidental (ratio 0.227)
        ]
        fused = _rrf_fuse(dense_pairs, bm25_pairs, rrf_k=60, bm25_bypass_min_ratio=0.35)
        wrong_doc = next(d for d, _ in fused if d.metadata.get("sub_topic") == "ประเภทของการวางแผนผังโรงงาน")
        assert wrong_doc.metadata.get("_bm25_hit") is not True

    def test_weak_match_still_participates_in_ranking_just_not_bypassed(self):
        """The rejected doc isn't dropped from the fused list entirely — it
        still gets its RRF rank contribution, just no floor bypass. Downstream
        sim_filter (with dense score 0.0, no bypass) will exclude it normally,
        same as any other low-relevance candidate."""
        dense_pairs = []
        bm25_pairs = [
            (_doc("correct"), 103.27),
            (_doc("wrong"), 23.41),
        ]
        fused = _rrf_fuse(dense_pairs, bm25_pairs, rrf_k=60, bm25_bypass_min_ratio=0.35)
        assert len(fused) == 2
        assert {d.metadata.get("sub_topic") for d, _ in fused} == {"correct", "wrong"}

    def test_default_ratio_matches_conf_default(self):
        import conf
        assert conf.BM25_BYPASS_MIN_RATIO == pytest.approx(0.35)

    def test_empty_bm25_pairs_no_crash(self):
        fused = _rrf_fuse([(_doc("x"), 0.5)], [], rrf_k=60, bm25_bypass_min_ratio=0.35)
        assert len(fused) == 1

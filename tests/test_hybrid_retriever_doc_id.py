"""Tests for utils/hybrid_retriever.py's _doc_id/_rrf_fuse — 2026-09 bug fix.

Root cause (found via live testing on feature/pdf-ingestion): _doc_id used to
fall back to page_content[:80] as a document identity key for combining Dense
+ BM25 ranks in _rrf_fuse. PDF book chunks all prepend a repeated context
header ("หนังสือ: X\\nหมวด: Y\\nหัวข้อย่อย: Z...") before their real content —
when Y (main_topic, shared across sibling chapters) is long enough, Z (the
actual differentiator) never appears within the first 80 characters, so
multiple DISTINCT documents produced an IDENTICAL id. Confirmed live: 3 real
case-study chapters under one main_topic collided this way, and _rrf_fuse
silently summed their 3 separate RRF contributions into one entry (keeping
only the last-seen Document as the representative) — inflating an unrelated
document's score high enough to outrank the genuinely relevant chapter for a
real user query, with no error or warning anywhere.

Fix: hash the full page_content instead of truncating to a fixed prefix.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from utils.hybrid_retriever import _doc_id, _rrf_fuse


def _pdf_doc(main_topic: str, sub_topic: str, body: str) -> Document:
    """Mirrors service/pdf_direct_ingest.py's _build_pdf_knowhow_page_content
    header shape closely enough to reproduce the real collision pattern."""
    page_content = f"หนังสือ: เจ๊ง! ในกระดาษ\nหมวด: {main_topic}\nหัวข้อย่อย: {sub_topic}\n\n{body}"
    return Document(page_content=page_content, metadata={"data_type": "business_guide", "main_topic": main_topic, "sub_topic": sub_topic})


@pytest.mark.unit
class TestDocIdNoLongerCollidesOnSharedPrefix:
    def test_distinct_docs_sharing_a_long_header_prefix_get_distinct_ids(self):
        # Long shared main_topic, short distinguishing sub_topic — the exact
        # real-world shape that collided under the old page_content[:80] rule.
        long_topic = "การวิเคราะห์ความเป็นไปได้ทางธุรกิจ (Feasibility Study)"
        d1 = _pdf_doc(long_topic, "กรณีศึกษา: ร้านกาแฟพันธุ์ไทย", "เนื้อหา A" * 200)
        d2 = _pdf_doc(long_topic, "กรณีศึกษา: ชาตันหยง", "เนื้อหา B" * 200)
        d3 = _pdf_doc(long_topic, "กรณีศึกษา: ร้านขายอุปกรณ์ไอที Remax", "เนื้อหา C" * 200)

        ids = {_doc_id(d1), _doc_id(d2), _doc_id(d3)}
        assert len(ids) == 3, "distinct documents must never produce the same id"

    def test_old_80_char_prefix_would_have_collided_here_sanity_check(self):
        """Documents this test constructs before-content genuinely share
        an 80-char prefix — otherwise this whole test module isn't actually
        exercising the bug it claims to guard against."""
        long_topic = "การวิเคราะห์ความเป็นไปได้ทางธุรกิจ (Feasibility Study)"
        d1 = _pdf_doc(long_topic, "กรณีศึกษา: ร้านกาแฟพันธุ์ไทย", "x")
        d2 = _pdf_doc(long_topic, "กรณีศึกษา: ชาตันหยง", "x")
        assert d1.page_content[:80] == d2.page_content[:80]

    def test_multi_chunk_split_of_same_topic_gets_distinct_ids(self):
        """A long topic split into multiple sub-chunks (utils.pdf_text_chunking)
        shares main_topic AND sub_topic — only the body content differs."""
        d1 = _pdf_doc("หัวข้อยาว", "ย่อยเดียวกัน", "เนื้อหาส่วนที่หนึ่ง" * 50)
        d2 = _pdf_doc("หัวข้อยาว", "ย่อยเดียวกัน", "เนื้อหาส่วนที่สอง" * 50)
        assert _doc_id(d1) != _doc_id(d2)

    def test_identical_content_produces_identical_id(self):
        d1 = _pdf_doc("หัวข้อ", "ย่อย", "เนื้อหาเดียวกัน")
        d2 = _pdf_doc("หัวข้อ", "ย่อย", "เนื้อหาเดียวกัน")
        assert _doc_id(d1) == _doc_id(d2)

    def test_explicit_doc_id_metadata_still_takes_precedence(self):
        d = Document(page_content="เนื้อหาอะไรก็ได้", metadata={"doc_id": "explicit-123"})
        assert _doc_id(d) == "explicit-123"

    def test_explicit_id_metadata_still_takes_precedence(self):
        d = Document(page_content="เนื้อหาอะไรก็ได้", metadata={"id": "explicit-456"})
        assert _doc_id(d) == "explicit-456"


@pytest.mark.unit
class TestRrfFuseNoLongerMergesDistinctDocs:
    def test_three_colliding_siblings_each_keep_their_own_rrf_score(self):
        long_topic = "การวิเคราะห์ความเป็นไปได้ทางธุรกิจ (Feasibility Study)"
        case_a = _pdf_doc(long_topic, "กรณีศึกษา: ร้านกาแฟพันธุ์ไทย", "A" * 500)
        case_b = _pdf_doc(long_topic, "กรณีศึกษา: ชาตันหยง", "B" * 500)
        case_c = _pdf_doc(long_topic, "กรณีศึกษา: ร้านขายอุปกรณ์ไอที Remax", "C" * 500)
        target = _pdf_doc("การวิเคราะห์ด้านการเงิน", "การวิเคราะห์รายได้และกำไร", "เนื้อหาที่ถูกต้อง Top Line Bottom Line")

        # target ranks #1 in BOTH dense and BM25 (the real observed pattern);
        # the 3 unrelated case studies scatter across lower ranks in each list.
        dense_pairs = [(target, 0.90), (case_a, 0.30), (case_b, 0.28), (case_c, 0.25)]
        bm25_pairs = [(target, 50.0), (case_b, 12.0), (case_c, 9.0), (case_a, 8.0)]

        fused = _rrf_fuse(dense_pairs, bm25_pairs, rrf_k=60)
        fused_by_content = {d.page_content: score for d, score in fused}

        # target must be the single highest-scoring entry — it was rank-1 in
        # both source lists, so no other single OR merged entry should beat it.
        top_doc, top_score = max(fused, key=lambda x: x[1])
        assert top_doc.page_content == target.page_content, (
            "the genuinely rank-1-in-both-lists document must win — if this "
            "fails, colliding ids are inflating a competitor's merged score again"
        )

        # exactly 4 distinct fused entries — no collision-driven merging into fewer
        assert len(fused_by_content) == 4

"""Tests for model/persona_academic.py's _aggregate_non_reg_docs — 2026-09 bug
fix, part of the pdf_chunking_truncation_and_retrieval_fixes chain.

This function groups docs sharing one sub_topic and aggregates groups of >1
into a single overview doc. It was originally built for ONE case: several
independent Sheet content_type rows under one marketing sub_topic, where a
"N ประเภท" (N types) numbered-snippet summary is the correct answer.

But a group sharing one sub_topic can also mean something completely
different: sequential CHUNKS of ONE long PDF-book topic that had to be split
at ingest time (utils/pdf_text_chunking.py) purely because it exceeded the
embedding-size cap. Those are not "different types" — collapsing them into
200-char-per-item snippets both mislabels them and re-truncates the exact
content the truncation fix (item #1 in this chain) restored.

chunk_total/chunk_index are set ONLY by the PDF ingestion path
(service/pdf_direct_ingest.py) — their presence is an unambiguous signal to
reassemble in order instead of building a lossy summary, with zero risk to
the original Sheet-based aggregation behavior (which never sets those
fields at all).
"""
from __future__ import annotations

import pytest

from model.persona_academic import _aggregate_non_reg_docs


def _sheet_doc(sub_topic: str, content_type: str, answer_guideline: str) -> dict:
    return {
        "content": f"{content_type}: {answer_guideline}",
        "metadata": {
            "sub_topic": sub_topic,
            "data_type": "business_guide",
            "content_type": content_type,
            "answer_guideline": answer_guideline,
        },
    }


def _pdf_chunk_doc(sub_topic: str, chunk_index: int, chunk_total: int, answer_guideline: str) -> dict:
    return {
        "content": f"หนังสือ: X\nหัวข้อย่อย: {sub_topic}\n\n{answer_guideline}",
        "metadata": {
            "sub_topic": sub_topic,
            "data_type": "business_guide",
            "ingested_via": "pdf_auto",
            "chunk_index": chunk_index,
            "chunk_total": chunk_total,
            "answer_guideline": answer_guideline,
        },
    }


@pytest.mark.unit
class TestSheetBasedAggregationUnchanged:
    def test_multiple_content_type_variants_still_get_types_summary(self):
        docs = [
            _sheet_doc("กลยุทธ์การตลาด", "โซเชียลมีเดีย", "เนื้อหาโซเชียล"),
            _sheet_doc("กลยุทธ์การตลาด", "อีเมล", "เนื้อหาอีเมล"),
            _sheet_doc("กลยุทธ์การตลาด", "SEO", "เนื้อหา SEO"),
        ]
        result = _aggregate_non_reg_docs(docs)
        assert len(result) == 1
        assert "ประเภท" in result[0]["content"]
        assert "3" in result[0]["metadata"]["answer_guideline"]

    def test_single_doc_sub_topic_passes_through_unchanged(self):
        docs = [_sheet_doc("หัวข้อเดี่ยว", "ทั่วไป", "เนื้อหา")]
        result = _aggregate_non_reg_docs(docs)
        assert result == docs


@pytest.mark.unit
class TestPdfMultiChunkReassembly:
    def test_chunks_reassembled_in_order_not_summarized_as_types(self):
        full_text_part1 = "เนื้อหาส่วนที่หนึ่ง " * 50
        full_text_part2 = "เนื้อหาส่วนที่สอง " * 50
        full_text_part3 = "เนื้อหาส่วนที่สาม " * 50
        docs = [
            _pdf_chunk_doc("หัวข้อยาว", 2, 3, full_text_part2),
            _pdf_chunk_doc("หัวข้อยาว", 1, 3, full_text_part1),
            _pdf_chunk_doc("หัวข้อยาว", 3, 3, full_text_part3),
        ]
        result = _aggregate_non_reg_docs(docs)
        assert len(result) == 1
        combined_ag = result[0]["metadata"]["answer_guideline"]

        # Must NOT use the "N ประเภท" types-framing — this isn't a types enumeration
        assert "ประเภท" not in combined_ag

        # Full content of all 3 chunks must survive, in chunk_index order
        pos1 = combined_ag.find(full_text_part1.strip())
        pos2 = combined_ag.find(full_text_part2.strip())
        pos3 = combined_ag.find(full_text_part3.strip())
        assert pos1 != -1 and pos2 != -1 and pos3 != -1
        assert pos1 < pos2 < pos3

        # No 200-char-per-item truncation — full text present, not just a snippet
        assert full_text_part1.strip() in combined_ag
        assert full_text_part3.strip() in combined_ag

    def test_chunk_index_and_chunk_total_stripped_from_combined_metadata(self):
        docs = [
            _pdf_chunk_doc("หัวข้อยาว", 1, 2, "ส่วนที่หนึ่ง"),
            _pdf_chunk_doc("หัวข้อยาว", 2, 2, "ส่วนที่สอง"),
        ]
        result = _aggregate_non_reg_docs(docs)
        assert "chunk_index" not in result[0]["metadata"]
        assert "chunk_total" not in result[0]["metadata"]

    def test_mismatched_chunk_totals_falls_back_to_sheet_style_aggregation(self):
        """Defensive: if chunk_total values disagree (shouldn't happen from real
        ingestion, but don't silently misbehave if it does), fall back to the
        original types-summary path rather than guessing at reassembly order."""
        docs = [
            _pdf_chunk_doc("หัวข้อยาว", 1, 2, "ส่วนที่หนึ่ง"),
            _pdf_chunk_doc("หัวข้อยาว", 1, 3, "ส่วนที่สอง"),
        ]
        result = _aggregate_non_reg_docs(docs)
        assert len(result) == 1
        assert "ประเภท" in result[0]["content"]

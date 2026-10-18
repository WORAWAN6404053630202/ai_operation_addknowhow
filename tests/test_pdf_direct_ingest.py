"""Tests for service/pdf_direct_ingest.py's build_knowhow_documents() —
2026-09 sub-chunking fix. Confirms:
- the common case (topic fits under the cap) still yields exactly one
  Document, with page_content/metadata content matching (unlike the old
  code, where metadata held the full text but page_content used for
  embedding was silently truncated — the actual root cause of the bug)
- an oversized topic gets split into multiple Documents sharing
  main_topic/sub_topic (so persona_supervisor.py's chapter-fetch groups them
  together) with correct chunk_index/chunk_total metadata and no content loss
- pre-existing fields (data_type, ingested_via tagging, etc.) are unchanged
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import conf
from service.pdf_direct_ingest import build_knowhow_documents, build_regulatory_documents_full


@pytest.mark.unit
class TestBuildKnowhowDocumentsSingleChunk:
    def test_short_topic_yields_one_document(self):
        topics = [{
            "main_topic": "การวิเคราะห์ทำเลที่ตั้ง",
            "sub_topic": "ปัจจัยด้านการจราจร",
            "full_text": "เนื้อหาสั้นๆ เกี่ยวกับการเลือกทำเล",
            "document_title": "เจ๊งในกระดาษ",
            "category": "ทำเล",
        }]
        docs = build_knowhow_documents(topics, source_file="test.pdf")
        assert len(docs) == 1
        d = docs[0]
        assert d.metadata["chunk_index"] == 1
        assert d.metadata["chunk_total"] == 1
        assert d.metadata["main_topic"] == "การวิเคราะห์ทำเลที่ตั้ง"
        assert d.metadata["sub_topic"] == "ปัจจัยด้านการจราจร"
        assert d.metadata["answer_guideline"] == "เนื้อหาสั้นๆ เกี่ยวกับการเลือกทำเล"
        assert d.metadata["data_type"] == "business_guide"
        assert d.metadata["ingested_via"] == "pdf_auto"
        assert d.metadata["source_file"] == "test.pdf"

    def test_page_content_and_metadata_answer_guideline_are_consistent(self):
        """Root-cause regression guard: the old bug was page_content (embedded)
        being truncated while metadata.answer_guideline (sent to the LLM) kept
        the full text — the two diverged. Now both must reflect the SAME
        (untruncated, for content under the cap) body."""
        full_text = "เนื้อหาเต็มของหัวข้อนี้ " * 20
        topics = [{"main_topic": "หัวข้อ", "sub_topic": "", "full_text": full_text, "document_title": "book.pdf"}]
        docs = build_knowhow_documents(topics, source_file="book.pdf")
        assert len(docs) == 1
        d = docs[0]
        assert d.metadata["answer_guideline"].strip() == full_text.strip()
        assert full_text.strip() in d.page_content

    def test_realistic_19k_char_chapter_stays_as_one_untruncated_document(self):
        # The actual longest real chapter measured during live testing.
        full_text = "เนื้อหาบทที่ยาวมาก " * 1000
        assert 18000 < len(full_text) < 20000
        topics = [{"main_topic": "บทยาว", "sub_topic": "", "full_text": full_text, "document_title": "book.pdf"}]
        docs = build_knowhow_documents(topics, source_file="book.pdf")
        assert len(docs) == 1
        assert len(docs[0].metadata["answer_guideline"]) == len(full_text.strip())

    def test_missing_full_text_topic_is_skipped(self):
        topics = [{"main_topic": "ว่างเปล่า", "sub_topic": "", "full_text": "  "}]
        docs = build_knowhow_documents(topics, source_file="test.pdf")
        assert docs == []


@pytest.mark.unit
class TestBuildKnowhowDocumentsMultiChunk:
    def test_oversized_topic_splits_into_multiple_documents_sharing_topic_metadata(self):
        full_text = "เนื้อหาที่ยาวเกินขนาด " * 3000  # comfortably over the 24000-char default cap
        topics = [{
            "main_topic": "หัวข้อใหญ่มาก",
            "sub_topic": "ย่อยของมัน",
            "full_text": full_text,
            "document_title": "long_book.pdf",
        }]
        with patch.object(conf, "PDF_KNOWHOW_CHUNK_MAX_CHARS", 24000):
            docs = build_knowhow_documents(topics, source_file="long_book.pdf")
        assert len(docs) > 1
        for i, d in enumerate(docs, start=1):
            assert d.metadata["main_topic"] == "หัวข้อใหญ่มาก"
            assert d.metadata["sub_topic"] == "ย่อยของมัน"
            assert d.metadata["chunk_index"] == i
            assert d.metadata["chunk_total"] == len(docs)
            assert len(d.metadata["answer_guideline"]) <= 24000

    def test_no_content_lost_across_split_chunks(self):
        full_text = "ก" * 30000  # single giant run, no headers/paragraphs — hard-slice path
        topics = [{"main_topic": "หัวข้อ", "sub_topic": "", "full_text": full_text, "document_title": "book.pdf"}]
        with patch.object(conf, "PDF_KNOWHOW_CHUNK_MAX_CHARS", 10000):
            docs = build_knowhow_documents(topics, source_file="book.pdf")
        assert len(docs) == 3
        rejoined = "".join(d.metadata["answer_guideline"] for d in docs)
        assert rejoined == full_text

    def test_chunk_header_included_in_page_content(self):
        full_text = "เนื้อหา " * 5000
        topics = [{
            "main_topic": "หัวข้อหลัก", "sub_topic": "หัวข้อย่อย",
            "full_text": full_text, "document_title": "หนังสือทดสอบ",
        }]
        with patch.object(conf, "PDF_KNOWHOW_CHUNK_MAX_CHARS", 10000):
            docs = build_knowhow_documents(topics, source_file="book.pdf")
        assert len(docs) > 1
        for d in docs:
            assert "หนังสือทดสอบ" in d.page_content
            assert "หัวข้อหลัก" in d.page_content
            assert "หัวข้อย่อย" in d.page_content
            assert f"{d.metadata['chunk_index']}/{d.metadata['chunk_total']}" in d.page_content


@pytest.mark.unit
class TestKnowhowResearchReferencePageRange:
    """2026-10: research_reference previously dropped page-range info in the
    direct-ingest path, inconsistent with sheet_write_back.py's established
    convention (f"{filename} (หน้า {page_range})")."""

    def test_page_range_included_when_present(self):
        topics = [{"main_topic": "หัวข้อ", "sub_topic": "", "full_text": "เนื้อหา", "page_range": "3-5"}]
        docs = build_knowhow_documents(topics, source_file="book.pdf")
        assert docs[0].metadata["research_reference"] == "book.pdf (หน้า 3-5)"

    def test_falls_back_to_plain_filename_when_page_range_missing(self):
        topics = [{"main_topic": "หัวข้อ", "sub_topic": "", "full_text": "เนื้อหา"}]
        docs = build_knowhow_documents(topics, source_file="book.pdf")
        assert docs[0].metadata["research_reference"] == "เอกสาร: book.pdf"


@pytest.mark.unit
class TestBuildRegulatoryDocumentsFullFieldGaps:
    """2026-10: location/area_size were hardcoded None in the PDF regulatory
    path, a real gap vs. DataLoader's Sheet-row derivation. Closed by reusing
    DataLoader._extract_location/_extract_area_size (pure text-sniffing, no
    Sheet dependency) against the drafted operation_topic + raw source text."""

    def test_location_extracted_from_raw_text_when_drafted_field_lacks_it(self):
        topics = [{
            "department": "เทศบาล", "license_type": "ใบอนุญาตทดสอบ",
            "operation_topic": "ขั้นตอนการขอใบอนุญาต",  # paraphrased, no literal "กรุงเทพ"
            "_raw_text": "สถานประกอบการตั้งอยู่ในเขตกรุงเทพมหานคร ต้องยื่นคำขอ...",
            "answer_guideline": "เนื้อหาคำตอบ",
        }]
        docs = build_regulatory_documents_full(topics, source_file="law.pdf")
        assert docs[0].metadata["location"] == "กรุงเทพฯ"

    def test_area_size_extracted(self):
        topics = [{
            "department": "เทศบาล", "license_type": "ใบอนุญาตทดสอบ",
            "operation_topic": "สำหรับร้านพื้นที่มากกว่า 200 ตารางเมตร",
            "answer_guideline": "เนื้อหาคำตอบ",
        }]
        docs = build_regulatory_documents_full(topics, source_file="law.pdf")
        assert docs[0].metadata["area_size"] == "มากกว่า 200 ตารางเมตร"

    def test_location_none_when_not_mentioned_anywhere(self):
        topics = [{
            "department": "เทศบาล", "license_type": "ใบอนุญาตทดสอบ",
            "operation_topic": "ขั้นตอนทั่วไป", "answer_guideline": "เนื้อหาคำตอบ",
        }]
        docs = build_regulatory_documents_full(topics, source_file="law.pdf")
        assert docs[0].metadata["location"] is None

    def test_research_reference_includes_page_range(self):
        topics = [{
            "department": "เทศบาล", "license_type": "ใบอนุญาตทดสอบ",
            "operation_topic": "ขั้นตอนทั่วไป", "answer_guideline": "เนื้อหาคำตอบ",
            "page_range": "10-12",
        }]
        docs = build_regulatory_documents_full(topics, source_file="law.pdf")
        assert docs[0].metadata["research_reference"] == "law.pdf (หน้า 10-12)"

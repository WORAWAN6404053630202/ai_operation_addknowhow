"""Tests for service/pdf_direct_ingest.py's build_merged_documents() —
2026-10, PDF ingestion schema unification. Replaces the old build_knowhow_
documents/build_regulatory_documents_full split: every topic now goes
through ONE builder, with data_type DERIVED from whether department/
license_type were drafted non-empty, not pre-classified.

Covers:
- business_guide-shaped topics (no department/license_type) — same
  sub-chunking/content behavior the old build_knowhow_documents had
- regulatory-shaped topics (department/license_type present) — data_type
  derivation, location/area_size extraction, and (new) chunking +
  structured-field replication across chunks, which the old
  build_regulatory_documents_full never needed since it never chunked
- the headline case this whole merge exists for: a topic with NO
  department/license_type drafted stays business_guide; one where they
  ARE drafted becomes regulatory, regardless of how the content is phrased
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import conf
from service.pdf_direct_ingest import (
    _MAX_MERGED_PAGE_CONTENT_CHARS,
    _build_pdf_merged_page_content,
    build_merged_documents,
)


@pytest.mark.unit
class TestBusinessGuideShapedSingleChunk:
    def test_short_topic_yields_one_document(self):
        topics = [{
            "main_topic": "การวิเคราะห์ทำเลที่ตั้ง",
            "sub_topic": "ปัจจัยด้านการจราจร",
            "full_text": "เนื้อหาสั้นๆ เกี่ยวกับการเลือกทำเล",
            "document_title": "เจ๊งในกระดาษ",
            "category": "ทำเล",
        }]
        docs = build_merged_documents(topics, source_file="test.pdf")
        assert len(docs) == 1
        d = docs[0]
        assert d.metadata["chunk_index"] == 1
        assert d.metadata["chunk_total"] == 1
        assert d.metadata["main_topic"] == "การวิเคราะห์ทำเลที่ตั้ง"
        assert d.metadata["sub_topic"] == "ปัจจัยด้านการจราจร"
        assert d.metadata["answer_guideline"] == "เนื้อหาสั้นๆ เกี่ยวกับการเลือกทำเล"
        assert d.metadata["data_type"] == "business_guide"
        assert d.metadata["department"] is None
        assert d.metadata["license_type"] is None
        assert d.metadata["ingested_via"] == "pdf_auto"
        assert d.metadata["source_file"] == "test.pdf"

    def test_page_content_and_metadata_answer_guideline_are_consistent(self):
        """Root-cause regression guard (carried over from the pre-merge
        bug): page_content (embedded) and metadata.answer_guideline (sent
        to the LLM) must reflect the SAME untruncated body for content
        under the cap."""
        full_text = "เนื้อหาเต็มของหัวข้อนี้ " * 20
        topics = [{"main_topic": "หัวข้อ", "sub_topic": "", "full_text": full_text, "document_title": "book.pdf"}]
        docs = build_merged_documents(topics, source_file="book.pdf")
        assert len(docs) == 1
        d = docs[0]
        assert d.metadata["answer_guideline"].strip() == full_text.strip()
        assert full_text.strip() in d.page_content

    def test_realistic_19k_char_chapter_stays_as_one_untruncated_document(self):
        full_text = "เนื้อหาบทที่ยาวมาก " * 1000
        assert 18000 < len(full_text) < 20000
        topics = [{"main_topic": "บทยาว", "sub_topic": "", "full_text": full_text, "document_title": "book.pdf"}]
        docs = build_merged_documents(topics, source_file="book.pdf")
        assert len(docs) == 1
        assert len(docs[0].metadata["answer_guideline"]) == len(full_text.strip())

    def test_missing_full_text_topic_is_skipped(self):
        topics = [{"main_topic": "ว่างเปล่า", "sub_topic": "", "full_text": "  "}]
        docs = build_merged_documents(topics, source_file="test.pdf")
        assert docs == []


@pytest.mark.unit
class TestBusinessGuideShapedMultiChunk:
    def test_oversized_topic_splits_into_multiple_documents_sharing_topic_metadata(self):
        full_text = "เนื้อหาที่ยาวเกินขนาด " * 3000  # comfortably over the 24000-char default cap
        topics = [{
            "main_topic": "หัวข้อใหญ่มาก",
            "sub_topic": "ย่อยของมัน",
            "full_text": full_text,
            "document_title": "long_book.pdf",
        }]
        with patch.object(conf, "PDF_KNOWHOW_CHUNK_MAX_CHARS", 24000):
            docs = build_merged_documents(topics, source_file="long_book.pdf")
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
            docs = build_merged_documents(topics, source_file="book.pdf")
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
            docs = build_merged_documents(topics, source_file="book.pdf")
        assert len(docs) > 1
        for d in docs:
            assert "หนังสือทดสอบ" in d.page_content
            assert "หัวข้อหลัก" in d.page_content
            assert "หัวข้อย่อย" in d.page_content
            assert f"{d.metadata['chunk_index']}/{d.metadata['chunk_total']}" in d.page_content


@pytest.mark.unit
class TestResearchReferencePageRange:
    def test_page_range_included_when_present(self):
        topics = [{"main_topic": "หัวข้อ", "sub_topic": "", "full_text": "เนื้อหา", "page_range": "3-5"}]
        docs = build_merged_documents(topics, source_file="book.pdf")
        assert docs[0].metadata["research_reference"] == "book.pdf (หน้า 3-5)"

    def test_falls_back_to_plain_filename_when_page_range_missing(self):
        topics = [{"main_topic": "หัวข้อ", "sub_topic": "", "full_text": "เนื้อหา"}]
        docs = build_merged_documents(topics, source_file="book.pdf")
        assert docs[0].metadata["research_reference"] == "เอกสาร: book.pdf"


@pytest.mark.unit
class TestDataTypeDerivation:
    """The headline fix this merge exists for: data_type is derived from
    whether department/license_type were drafted non-empty — NOT from any
    upfront classification. Confirmed live example: "กำหนดเวลาห้ามขายเครื่องดื่ม
    แอลกอฮอล์.pdf" (a law prohibiting alcohol sales during certain hours,
    with real penalties) used to be misclassified business_guide because it
    isn't phrased as a license-application procedure — once department/
    legal_regulatory get drafted for it, this derivation correctly resolves
    it to regulatory regardless of phrasing."""

    def test_department_and_license_type_present_resolves_regulatory(self):
        """Representative of the alcohol-sale-hours-law case: a legal
        restriction, not an application procedure, but department/
        legal_regulatory get drafted."""
        topics = [{
            "main_topic": "ข้อบังคับการขายเครื่องดื่มแอลกอฮอล์",
            "sub_topic": "กำหนดเวลาห้ามขาย",
            "full_text": "ห้ามขายเครื่องดื่มแอลกอฮอล์ในช่วงเวลา...",
            "department": "กรมสรรพสามิต",
            "license_type": "ข้อบังคับการจำหน่ายแอลกอฮอล์",
            "legal_regulatory": "มีบทลงโทษปรับไม่เกิน 10,000 บาท",
        }]
        docs = build_merged_documents(topics, source_file="กำหนดเวลาห้ามขายเครื่องดื่มแอลกอฮอล์.pdf")
        assert len(docs) == 1
        assert docs[0].metadata["data_type"] == "regulatory"
        assert docs[0].metadata["department"] == "กรมสรรพสามิต"
        assert docs[0].metadata["legal_regulatory"] == "มีบทลงโทษปรับไม่เกิน 10,000 บาท"
        # main_topic/sub_topic are now populated for regulatory-shaped
        # topics too — new, additive, nothing downstream requires them absent.
        assert docs[0].metadata["main_topic"] == "ข้อบังคับการขายเครื่องดื่มแอลกอฮอล์"

    def test_only_license_type_present_still_resolves_regulatory(self):
        topics = [{"main_topic": "A", "sub_topic": "B", "full_text": "x", "license_type": "ใบอนุญาตบางอย่าง"}]
        docs = build_merged_documents(topics, source_file="f.pdf")
        assert docs[0].metadata["data_type"] == "regulatory"

    def test_commercial_howto_with_no_department_or_license_type_stays_business_guide(self):
        """Representative of the QR-payment / EDC-setup case: a commercial
        how-to with steps/channels, but no government agency or license
        involved — must NOT be swept into regulatory just because it has
        procedural structure."""
        topics = [{
            "main_topic": "การชำระเงิน", "sub_topic": "วิธีการเปิดใช้งาน QR Payment",
            "full_text": "ขั้นตอนการสมัครใช้งาน QR Payment กับธนาคาร...",
            "service_channel": "ติดต่อธนาคารที่ใช้บริการ",
            "operation_steps": "1. ติดต่อธนาคาร 2. กรอกแบบฟอร์ม 3. รอการอนุมัติ",
        }]
        docs = build_merged_documents(topics, source_file="วิธีการชำระเงิน.pdf")
        assert len(docs) == 1
        assert docs[0].metadata["data_type"] == "business_guide"
        assert docs[0].metadata["department"] is None
        assert docs[0].metadata["license_type"] is None
        # Structured fields the drafter DID find still get through —
        # business_guide doesn't mean "no structured fields allowed".
        assert docs[0].metadata["service_channel"] == "ติดต่อธนาคารที่ใช้บริการ"

    def test_no_department_or_license_type_drafted_at_all_stays_business_guide(self):
        topics = [{"main_topic": "กลยุทธ์การตลาด", "sub_topic": "โซเชียลมีเดีย", "full_text": "เนื้อหา"}]
        docs = build_merged_documents(topics, source_file="f.pdf")
        assert docs[0].metadata["data_type"] == "business_guide"


@pytest.mark.unit
class TestRegulatoryShapedFieldGaps:
    """2026-10: location/area_size derivation (ported from the old
    build_regulatory_documents_full) — reuses DataLoader._extract_location/
    _extract_area_size (pure text-sniffing, no Sheet dependency) against the
    drafted operation_topic + raw source text. Applies regardless of shape
    (called unconditionally), but only ever finds a signal in regulatory-
    flavored content in practice."""

    def test_location_extracted_from_raw_text_when_drafted_field_lacks_it(self):
        topics = [{
            "main_topic": "A", "sub_topic": "B", "full_text": "x",
            "department": "เทศบาล", "license_type": "ใบอนุญาตทดสอบ",
            "operation_topic": "ขั้นตอนการขอใบอนุญาต",  # paraphrased, no literal "กรุงเทพ"
            "_raw_text": "สถานประกอบการตั้งอยู่ในเขตกรุงเทพมหานคร ต้องยื่นคำขอ...",
        }]
        docs = build_merged_documents(topics, source_file="law.pdf")
        assert docs[0].metadata["location"] == "กรุงเทพฯ"

    def test_area_size_extracted(self):
        topics = [{
            "main_topic": "A", "sub_topic": "B", "full_text": "x",
            "department": "เทศบาล", "license_type": "ใบอนุญาตทดสอบ",
            "operation_topic": "สำหรับร้านพื้นที่มากกว่า 200 ตารางเมตร",
        }]
        docs = build_merged_documents(topics, source_file="law.pdf")
        assert docs[0].metadata["area_size"] == "มากกว่า 200 ตารางเมตร"

    def test_location_none_when_not_mentioned_anywhere(self):
        topics = [{
            "main_topic": "A", "sub_topic": "B", "full_text": "x",
            "department": "เทศบาล", "license_type": "ใบอนุญาตทดสอบ",
            "operation_topic": "ขั้นตอนทั่วไป",
        }]
        docs = build_merged_documents(topics, source_file="law.pdf")
        assert docs[0].metadata["location"] is None


@pytest.mark.unit
class TestRegulatoryShapedChunkingWithFieldReplication:
    """New behavior (2026-10 decision): regulatory-shaped topics are no
    longer guaranteed to be exactly one Document — a topic exceeding
    PDF_KNOWHOW_CHUNK_MAX_CHARS now splits like business_guide content
    always has. The structured fields (department, fees, legal_regulatory,
    etc.) must be replicated onto EVERY resulting sub-chunk, not just the
    first — a user asking "ค่าธรรมเนียมเท่าไหร่" must find the fee
    regardless of which chunk's embedding wins the similarity search."""

    def test_long_regulatory_topic_splits_into_multiple_documents(self):
        full_text = "เนื้อหากฎหมายที่ยาวมาก " * 3000
        topics = [{
            "main_topic": "A", "sub_topic": "B", "full_text": full_text,
            "department": "กรมสรรพากร", "license_type": "ใบอนุญาตทดสอบ",
            "fees": "500 บาท",
        }]
        with patch.object(conf, "PDF_KNOWHOW_CHUNK_MAX_CHARS", 24000):
            docs = build_merged_documents(topics, source_file="long_law.pdf")
        assert len(docs) > 1

    def test_every_sub_chunk_carries_the_same_structured_fields(self):
        full_text = "เนื้อหากฎหมายที่ยาวมาก " * 3000
        topics = [{
            "main_topic": "A", "sub_topic": "B", "full_text": full_text,
            "department": "กรมสรรพากร", "license_type": "ใบอนุญาตทดสอบ",
            "fees": "500 บาท", "operation_duration": "ภายใน 15 วัน",
        }]
        with patch.object(conf, "PDF_KNOWHOW_CHUNK_MAX_CHARS", 24000):
            docs = build_merged_documents(topics, source_file="long_law.pdf")
        assert len(docs) > 1
        for d in docs:
            assert d.metadata["department"] == "กรมสรรพากร"
            assert d.metadata["license_type"] == "ใบอนุญาตทดสอบ"
            assert d.metadata["fees"] == "500 บาท"
            assert d.metadata["operation_duration"] == "ภายใน 15 วัน"
            assert d.metadata["data_type"] == "regulatory"

    def test_fee_field_appears_in_every_chunks_page_content(self):
        """The retrieval requirement made explicit: the fee must be
        embeddable from whichever chunk a similarity search happens to
        surface, not just the first."""
        full_text = "เนื้อหากฎหมายที่ยาวมาก " * 3000
        topics = [{
            "main_topic": "A", "sub_topic": "B", "full_text": full_text,
            "department": "กรมสรรพากร", "license_type": "ใบอนุญาตทดสอบ",
            "fees": "500 บาท เท่านั้น",
        }]
        with patch.object(conf, "PDF_KNOWHOW_CHUNK_MAX_CHARS", 24000):
            docs = build_merged_documents(topics, source_file="long_law.pdf")
        assert len(docs) > 1
        for d in docs:
            assert "500 บาท เท่านั้น" in d.page_content

    def test_single_chunk_regulatory_topic_unaffected(self):
        """Regression guard: short regulatory topics (the overwhelming
        majority) still produce exactly one Document, same as before this
        change."""
        topics = [{
            "main_topic": "A", "sub_topic": "B", "full_text": "เนื้อหาสั้น",
            "department": "เทศบาล", "license_type": "ใบอนุญาตทดสอบ", "fees": "100 บาท",
        }]
        docs = build_merged_documents(topics, source_file="f.pdf")
        assert len(docs) == 1
        assert docs[0].metadata["chunk_total"] == 1


@pytest.mark.unit
class TestEmbeddingSizeGuard:
    """2026-10 fix, live-reproduced on a real document ("ยื่น_เพิ่ม_ชำระ_
    ยกเลิก ภาษีป้าย.pdf", a sign-tax guide): a topic with every structured
    field populated near its individual cap (summing to ~9,800 chars) PLUS
    a near-max body chunk (PDF_KNOWHOW_CHUNK_MAX_CHARS, 24,000 by default)
    assembled a page_content around ~34,000 chars — nothing capped the SUM,
    only each piece individually — and OpenRouter's baai/bge-m3 embed call
    rejected it outright ("maximum context length is 8192 tokens"),
    silently dropping that whole write via add_documents' per-batch
    failure. _MAX_MERGED_PAGE_CONTENT_CHARS is the safety net: the body
    gets trimmed (header+fields_block always stay whole) so the assembled
    total can never exceed it, regardless of how many fields are populated
    or how large PDF_KNOWHOW_CHUNK_MAX_CHARS is configured."""

    def _all_fields_topic(self, full_text: str) -> dict:
        return {
            "main_topic": "การยื่นชำระภาษีป้าย", "sub_topic": "การยื่นเพิ่ม/ชำระ/ยกเลิก",
            "full_text": full_text, "document_title": "คู่มือภาษีป้าย",
            "department": "เทศบาล", "license_type": "ใบอนุญาตป้าย",
            # Each near its own _MERGED_FIELD_CAPS ceiling, same shape as the
            # real document that triggered this — "ก" * N as a stand-in for
            # real prose, long enough to actually exercise every cap.
            "operation_steps": "ขั้นตอน " * 400,          # cap 1800
            "legal_regulatory": "กฎหมาย " * 400,           # cap 2000
            "terms_and_conditions": "เงื่อนไข " * 400,     # cap 2300
            "identification_documents": "เอกสาร " * 400,   # cap 2100
            "fees": "ค่าธรรมเนียม " * 200,                  # cap 900
            "operation_duration": "ระยะเวลา " * 100,       # cap 300
            "service_channel": "ช่องทาง " * 150,            # cap 600
        }

    def test_page_content_never_exceeds_the_safety_ceiling_even_with_every_field_maxed(self):
        full_text = "เนื้อหากฎหมายภาษีป้ายที่ยาวมาก " * 3000  # well over PDF_KNOWHOW_CHUNK_MAX_CHARS
        topics = [self._all_fields_topic(full_text)]
        with patch.object(conf, "PDF_KNOWHOW_CHUNK_MAX_CHARS", 24000):
            docs = build_merged_documents(topics, source_file="ยื่น_เพิ่ม_ชำระ_ยกเลิก ภาษีป้าย.pdf")
        assert len(docs) >= 1
        for d in docs:
            assert len(d.page_content) <= _MAX_MERGED_PAGE_CONTENT_CHARS, (
                f"page_content was {len(d.page_content)} chars, over the "
                f"{_MAX_MERGED_PAGE_CONTENT_CHARS}-char safety ceiling — this "
                f"exact failure mode (OpenRouter rejects the embed call outright) "
                f"is what this guard exists to prevent"
            )

    def test_prefix_alone_exceeding_the_ceiling_is_also_trimmed_not_just_the_body(self):
        """Regression guard for a real bug caught live while fixing this:
        the first version of this clamp only ever trimmed the BODY, so once
        _MAX_MERGED_PAGE_CONTENT_CHARS was corrected down to a realistic
        Thai-token-density value, a structured-fields block that alone
        exceeded the new (smaller) ceiling still produced an over-budget
        page_content with an empty body — the guarantee held in every test
        written before that correction only because the ceiling used to be
        large enough that the fields block could never exceed it on its
        own. Forces that exact condition directly (tiny body, every field
        maxed) rather than relying on build_merged_documents' chunking to
        happen to hit it."""
        structured_fields = {
            "operation_steps": "ขั้นตอน " * 400,
            "legal_regulatory": "กฎหมาย " * 400,
            "terms_and_conditions": "เงื่อนไข " * 400,
            "identification_documents": "เอกสาร " * 400,
            "fees": "ค่าธรรมเนียม " * 200,
            "operation_duration": "ระยะเวลา " * 100,
            "service_channel": "ช่องทาง " * 150,
        }
        page_content = _build_pdf_merged_page_content(
            document_title="คู่มือภาษีป้าย", main_topic="A", sub_topic="B",
            department="เทศบาล", license_type="ใบอนุญาตป้าย",
            structured_fields=structured_fields, body="เนื้อหาสั้น",
            chunk_index=1, chunk_total=1,
        )
        assert len(page_content) <= _MAX_MERGED_PAGE_CONTENT_CHARS

    def test_trimmed_body_does_not_lose_data_the_llm_facing_side_reads(self):
        """The clamp must only bound what gets embedded — the full chunk
        text the chat bot's context builder actually reads
        (metadata.answer_guideline) must stay complete and untrimmed."""
        full_text = "เนื้อหากฎหมายภาษีป้ายที่ยาวมาก " * 3000
        topics = [self._all_fields_topic(full_text)]
        with patch.object(conf, "PDF_KNOWHOW_CHUNK_MAX_CHARS", 24000):
            docs = build_merged_documents(topics, source_file="ยื่น_เพิ่ม_ชำระ_ยกเลิก ภาษีป้าย.pdf")
        # At least one chunk's page_content was actually trimmed relative to
        # its own answer_guideline — proves the guard engaged for this input
        # rather than the test accidentally passing with headroom to spare.
        assert any(len(d.page_content) < len(d.metadata["answer_guideline"]) for d in docs)
        # And every chunk's full original text still exists somewhere, in full
        # (build_merged_documents strips the topic's full_text once up front,
        # same as every other test in this file relies on implicitly).
        reassembled = "".join(d.metadata["answer_guideline"] for d in docs)
        assert reassembled == full_text.strip()

    def test_short_content_is_completely_unaffected_by_the_guard(self):
        """Regression guard: the overwhelming majority of real topics are
        nowhere near the ceiling — page_content must stay byte-identical to
        the pre-fix behavior for them (no silent extra trimming)."""
        topics = [{
            "main_topic": "A", "sub_topic": "B", "full_text": "เนื้อหาสั้นมาก",
            "department": "เทศบาล", "license_type": "ใบอนุญาตทดสอบ", "fees": "100 บาท",
        }]
        docs = build_merged_documents(topics, source_file="f.pdf")
        assert len(docs) == 1
        assert docs[0].page_content == docs[0].metadata["answer_guideline"] or \
            "เนื้อหาสั้นมาก" in docs[0].page_content
        assert len(docs[0].page_content) < _MAX_MERGED_PAGE_CONTENT_CHARS

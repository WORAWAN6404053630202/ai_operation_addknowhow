"""Headline regression test for the 2026-10 PDF ingestion schema-unification
(see /Users/w.worawan/.claude/plans/happy-launching-scroll.md).

Before this change, every PDF went through service/pdf_content_shape.py's
classify_content_shape() LLM call, which picked ONE of two parallel
builders: "structured_license" (13-field schema) or "know_how" (5-field
schema). That classifier was wrong at exactly the boundary that matters:
legal/regulatory content not phrased as a step-by-step license APPLICATION
procedure. Confirmed live example — "กำหนดเวลาห้ามขายเครื่องดื่มแอลกอฮอล์.pdf"
(a law prohibiting alcohol sales during certain hours, with real penalties)
was classified "know_how" because it's a restriction, not an application
procedure, losing its department/legal_regulatory fields. Meanwhile genuine
commercial how-tos like "วิธีการชำระเงิน.pdf" (QR payment) correctly stayed
business_guide, because the axis used ("has procedural steps") was simply
the wrong axis for the real distinction ("is this a government/legal
matter").

classify_content_shape() is now gone entirely. Every document runs through
ONE pipeline (identify_knowhow_topics -> draft_fields_from_pages for every
resulting topic) and data_type is DERIVED from whether drafting actually
found a department/license_type — never pre-guessed. This test proves that
derivation end-to-end through sqs_consumer._build_items_direct (not just at
the build_merged_documents unit level, already covered by
test_pdf_direct_ingest.py's TestDataTypeDerivation) for both sides of the
original bug's contrast, with only the two genuine LLM-call boundaries
(identify_knowhow_topics, draft_fields_from_pages) mocked — no classifier
mock exists anywhere in this test, because there is no classifier left to
mock.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import service.sqs_consumer as sqs_consumer  # noqa: E402
from model.pdf_review_item import PageExtractionRecord  # noqa: E402
from service.pdf_supersede_resolver import SupersedeDecision  # noqa: E402


def _make_pages(n: int):
    pages = [PageExtractionRecord(page_num=i + 1, typhoon_markdown=f"page {i + 1}", claude_markdown="") for i in range(n)]
    markdowns = [p.typhoon_markdown for p in pages]
    return pages, markdowns


@pytest.mark.unit
class TestAlcoholLawVsQrPaymentContrast:
    def test_alcohol_law_shaped_document_resolves_regulatory_with_no_classifier_involved(self):
        """A law restricting WHEN alcohol may be sold, with real penalties —
        no step-by-step application procedure anywhere in it. The old
        classifier called this "know_how" (wrong axis: no procedural
        steps). The new pipeline never classifies at all; it just drafts
        fields, and this document genuinely has a department + legal
        basis, so data_type derives to "regulatory" by construction."""
        pages, markdowns = _make_pages(2)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [{
                "document_title": "กำหนดเวลาห้ามขายเครื่องดื่มแอลกอฮอล์",
                "source_type": "document",
                "main_topic": "ข้อกำหนดการขายเครื่องดื่มแอลกอฮอล์",
                "sub_topic": "ช่วงเวลาห้ามขาย",
                "summary": "กฎหมายกำหนดช่วงเวลาห้ามจำหน่ายเครื่องดื่มแอลกอฮอล์",
                "category": "",
                "page_ranges": [(1, 2)],
            }]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {
                "หน่วยงาน": "กรมสรรพสามิต",
                "ข้อกำหนดทางกฎหมาย และข้อบังคับ": "ห้ามขายเครื่องดื่มแอลกอฮอล์เวลา 14.00-17.00 น. และ 00.00-11.00 น. "
                                                    "ฝ่าฝืนมีโทษจำคุกไม่เกิน 6 เดือน หรือปรับไม่เกิน 10,000 บาท",
            }

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        written = {}

        def capture_add(docs):
            written["docs"] = docs
            return len(docs)

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", side_effect=capture_add):
            sqs_consumer._build_items_direct(
                "กำหนดเวลาห้ามขายเครื่องดื่มแอลกอฮอล์.pdf", "s3://bucket/alcohol_law.pdf", pages, markdowns,
            )

        assert len(written["docs"]) == 1
        doc = written["docs"][0]
        assert doc.metadata["data_type"] == "regulatory"
        assert doc.metadata["department"] == "กรมสรรพสามิต"
        assert "10,000" in doc.metadata["legal_regulatory"]

    def test_qr_payment_shaped_document_resolves_business_guide_with_no_classifier_involved(self):
        """A commercial how-to with real procedural steps (how to set up QR
        payment) but no government department or license involved anywhere
        in it. The old classifier correctly called this "know_how" already
        — this test proves the new pipeline preserves that correct outcome
        (not a regression risk from removing the classifier, since nothing
        in the drafted fields gives it a department/license_type either)."""
        pages, markdowns = _make_pages(2)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [{
                "document_title": "วิธีการชำระเงินด้วย QR Code",
                "source_type": "document",
                "main_topic": "การรับชำระเงิน",
                "sub_topic": "การตั้งค่า QR Payment",
                "summary": "ขั้นตอนการสมัครและตั้งค่าระบบรับชำระเงินด้วย QR Code ผ่านธนาคาร",
                "category": "",
                "page_ranges": [(1, 2)],
            }]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {
                "ขั้นตอนการดำเนินการ": "1. สมัครใช้บริการกับธนาคาร 2. ติดตั้งแอปพลิเคชัน 3. สร้าง QR Code ของร้าน",
                "ช่องทางการ ให้บริการ": "สาขาธนาคาร, แอปพลิเคชันมือถือ",
            }

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        written = {}

        def capture_add(docs):
            written["docs"] = docs
            return len(docs)

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", side_effect=capture_add):
            sqs_consumer._build_items_direct(
                "วิธีการชำระเงิน.pdf", "s3://bucket/qr_payment.pdf", pages, markdowns,
            )

        assert len(written["docs"]) == 1
        doc = written["docs"][0]
        assert doc.metadata["data_type"] == "business_guide"
        assert doc.metadata.get("department") is None
        assert doc.metadata.get("license_type") is None

    def test_no_content_shape_classifier_exists_anywhere_in_the_pipeline(self):
        """Structural proof, not just absence-of-mock: the module that used
        to perform the classification no longer exists at all, so there is
        no classifier left to accidentally reintroduce or bypass."""
        with pytest.raises(ModuleNotFoundError):
            import service.pdf_content_shape  # noqa: F401
        assert not hasattr(sqs_consumer, "classify_content_shape")

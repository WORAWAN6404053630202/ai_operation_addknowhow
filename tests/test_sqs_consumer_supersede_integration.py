"""Integration-style tests for the automated supersede resolver wired into
service/sqs_consumer.py's _build_license_items_direct/_build_knowhow_items_direct
(2026-10, PDF-as-sole-source architecture). Mocks the resolver's decision
and the vectorstore write/archive calls — no real Chroma, no real
network/LLM calls, safe to run anytime. Covers: a duplicate topic is
dropped before the vectorstore write; an update topic is written THEN its
old doc retired, in that order; a resolver failure defaults to writing the
topic as new rather than dropping it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
class TestLicenseItemsDirectSupersedeWiring:
    def test_duplicate_topic_is_dropped_before_write(self):
        pages, markdowns = _make_pages(4)

        def fake_identify_license_topics(page_markdowns, cost_accumulator=None):
            return [
                {"department": "Dept A", "license_type": "License A", "page_ranges": [(1, 2)]},
                {"department": "Dept B", "license_type": "License B", "page_ranges": [(3, 4)]},
            ]

        def fake_draft_fields_from_pages(topic_markdowns, cost_accumulator=None):
            return {"แนวคำตอบ": "เนื้อหาคำตอบ"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            if merged["department"] == "Dept A":
                return SupersedeDecision("duplicate", matched_doc_ids=["old-a"])
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "identify_license_topics", side_effect=fake_identify_license_topics), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft_fields_from_pages), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_regulatory_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore") as mock_add, \
             patch.object(sqs_consumer, "_archive_and_remove_in_vectorstore") as mock_archive:
            mock_add.return_value = 1
            sqs_consumer._build_license_items_direct(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
            )

        assert mock_add.call_count == 1
        written_docs = mock_add.call_args[0][0]
        assert len(written_docs) == 1
        assert written_docs[0].metadata["license_type"] == "License B"
        mock_archive.assert_not_called()

    def test_update_topic_writes_new_then_retires_old_in_order(self):
        pages, markdowns = _make_pages(2)

        def fake_identify_license_topics(page_markdowns, cost_accumulator=None):
            return [{"department": "Dept A", "license_type": "License A", "page_ranges": [(1, 2)]}]

        def fake_draft_fields_from_pages(topic_markdowns, cost_accumulator=None):
            return {"แนวคำตอบ": "เนื้อหาคำตอบใหม่"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("update", matched_doc_ids=["old-a-1"])

        call_order = []

        with patch.object(sqs_consumer, "identify_license_topics", side_effect=fake_identify_license_topics), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft_fields_from_pages), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_regulatory_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore") as mock_add, \
             patch.object(sqs_consumer, "_archive_and_remove_in_vectorstore") as mock_archive:
            mock_add.side_effect = lambda docs: (call_order.append("add"), 1)[1]
            mock_archive.side_effect = lambda ids, reason: call_order.append("archive")

            sqs_consumer._build_license_items_direct(
                "updated_law.pdf", "s3://bucket/updated_law.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
            )

        assert call_order == ["add", "archive"]
        mock_archive.assert_called_once_with(["old-a-1"], reason="superseded by updated_law.pdf")

    def test_resolver_exception_falls_back_to_writing_as_new(self):
        pages, markdowns = _make_pages(2)

        def fake_identify_license_topics(page_markdowns, cost_accumulator=None):
            return [{"department": "Dept A", "license_type": "License A", "page_ranges": [(1, 2)]}]

        def fake_draft_fields_from_pages(topic_markdowns, cost_accumulator=None):
            return {"แนวคำตอบ": "เนื้อหาคำตอบ"}

        with patch.object(sqs_consumer, "identify_license_topics", side_effect=fake_identify_license_topics), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft_fields_from_pages), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_regulatory_topic", side_effect=RuntimeError("boom")), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore") as mock_add, \
             patch.object(sqs_consumer, "_archive_and_remove_in_vectorstore") as mock_archive:
            mock_add.return_value = 1
            sqs_consumer._build_license_items_direct(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
            )

        mock_add.assert_called_once()
        written_docs = mock_add.call_args[0][0]
        assert len(written_docs) == 1
        mock_archive.assert_not_called()


@pytest.mark.unit
class TestKnowhowItemsDirectSupersedeWiring:
    def test_duplicate_knowhow_topic_is_dropped_before_write(self):
        pages, markdowns = _make_pages(2)

        def fake_identify_knowhow_topics(page_markdowns, cost_accumulator=None):
            return [{
                "document_title": "คู่มือ", "source_type": "book",
                "main_topic": "การตลาด", "sub_topic": "โปรโมชั่น",
                "summary": "สรุป", "category": "", "page_ranges": [(1, 2)],
            }]

        def fake_resolve(topic_dict, source_file, cost_accumulator=None):
            return SupersedeDecision("duplicate", matched_doc_ids=["old-kh-1"])

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify_knowhow_topics), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_knowhow_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore") as mock_add, \
             patch.object(sqs_consumer, "_archive_and_remove_in_vectorstore") as mock_archive:
            mock_add.return_value = 0
            sqs_consumer._build_knowhow_items_direct(
                "book.pdf", "s3://bucket/book.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "know_how"},
            )

        written_docs = mock_add.call_args[0][0]
        assert written_docs == []
        mock_archive.assert_not_called()

    def test_update_knowhow_topic_expands_to_full_sibling_chunks_before_retiring(self):
        pages, markdowns = _make_pages(2)

        def fake_identify_knowhow_topics(page_markdowns, cost_accumulator=None):
            return [{
                "document_title": "คู่มือ", "source_type": "book",
                "main_topic": "การตลาด", "sub_topic": "โปรโมชั่น",
                "summary": "สรุป", "category": "", "page_ranges": [(1, 2)],
            }]

        def fake_resolve(topic_dict, source_file, cost_accumulator=None):
            return SupersedeDecision("update", matched_doc_ids=["old-kh-chunk-1"])

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify_knowhow_topics), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_knowhow_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "expand_knowhow_matches_to_full_chunks",
                           return_value=["old-kh-chunk-1", "old-kh-chunk-2"]) as mock_expand, \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore") as mock_add, \
             patch.object(sqs_consumer, "_archive_and_remove_in_vectorstore") as mock_archive:
            mock_add.return_value = 1
            sqs_consumer._build_knowhow_items_direct(
                "updated_book.pdf", "s3://bucket/updated_book.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "know_how"},
            )

        mock_expand.assert_called_once_with(["old-kh-chunk-1"])
        mock_archive.assert_called_once_with(["old-kh-chunk-1", "old-kh-chunk-2"], reason="superseded by updated_book.pdf")

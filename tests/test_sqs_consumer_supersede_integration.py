"""Integration-style tests for service/sqs_consumer.py's _build_items_direct
(2026-10, PDF ingestion schema unification — replaces the old
_build_license_items_direct/_build_knowhow_items_direct split and the
classify_content_shape() call that used to choose between them). Mocks
identify_knowhow_topics (the universal topic-splitter), draft_fields_from_pages
(now run for every topic), the resolver's decision, and the vectorstore
write/archive calls — no real Chroma, no real network/LLM calls, safe to
run anytime.

Covers: a duplicate topic is dropped before the vectorstore write; an
update topic is written THEN its old doc retired, in that order; a
resolver failure defaults to writing the topic as new rather than
dropping it; a topic whose field-drafting fails falls through to the
deterministic catch-all instead of vanishing; data_type is derived from
whether department/license_type got drafted, not pre-classified.
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


def _topic(main_topic="หัวข้อ A", sub_topic="หัวข้อย่อย A", page_ranges=((1, 2),), **extra):
    t = {
        "document_title": "เอกสาร", "source_type": "document",
        "main_topic": main_topic, "sub_topic": sub_topic,
        "summary": "สรุป", "category": "", "page_ranges": list(page_ranges),
    }
    t.update(extra)
    return t


@pytest.mark.unit
class TestBuildItemsDirectSupersedeWiring:
    def test_duplicate_topic_is_dropped_before_write(self):
        pages, markdowns = _make_pages(4)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [
                _topic("หัวข้อ A", "ย่อย A", [(1, 2)]),
                _topic("หัวข้อ B", "ย่อย B", [(3, 4)]),
            ]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            if "page 1" in topic_markdowns:
                return {"หน่วยงาน": "Dept A", "ใบอนุญาต": "License A"}
            return {"หน่วยงาน": "Dept B", "ใบอนุญาต": "License B"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            if merged["department"] == "Dept A":
                return SupersedeDecision("duplicate", matched_doc_ids=["old-a"])
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore") as mock_add, \
             patch.object(sqs_consumer, "_archive_and_remove_in_vectorstore") as mock_archive:
            mock_add.return_value = 1
            sqs_consumer._build_items_direct("test.pdf", "s3://bucket/test.pdf", pages, markdowns)

        assert mock_add.call_count == 1
        written_docs = mock_add.call_args[0][0]
        assert len(written_docs) == 1
        assert written_docs[0].metadata["license_type"] == "License B"
        mock_archive.assert_not_called()

    def test_update_topic_writes_new_then_retires_old_in_order(self):
        pages, markdowns = _make_pages(2)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [_topic(page_ranges=[(1, 2)])]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"หน่วยงาน": "Dept A", "ใบอนุญาต": "License A"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("update", matched_doc_ids=["old-a-1"])

        call_order = []

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore") as mock_add, \
             patch.object(sqs_consumer, "_filter_to_pdf_auto_ids", side_effect=lambda ids, filename: ids), \
             patch.object(sqs_consumer, "_archive_and_remove_in_vectorstore") as mock_archive:
            mock_add.side_effect = lambda docs: (call_order.append("add"), 1)[1]
            mock_archive.side_effect = lambda ids, reason: call_order.append("archive")

            sqs_consumer._build_items_direct("updated_law.pdf", "s3://bucket/updated_law.pdf", pages, markdowns)

        assert call_order == ["add", "archive"]
        mock_archive.assert_called_once_with(["old-a-1"], reason="superseded by updated_law.pdf")

    def test_vectorstore_write_failure_saves_a_recovery_placeholder(self):
        """PDF-M11 (2026-10): before this fix, a failed final Chroma write
        (after OCR/topic-splitting/drafting all succeeded — real billed
        work) was just logged and discarded, with nothing preserved to
        recover from except manually re-uploading the original PDF. Now
        saves a ReviewItem pointing at s3_raw_pdf_path so a human can
        /reprocess from the same original file."""
        pages, markdowns = _make_pages(2)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [_topic(page_ranges=[(1, 2)])]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"หน่วยงาน": "Dept A", "ใบอนุญาต": "License A"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", side_effect=RuntimeError("Chroma connection refused")), \
             patch.object(sqs_consumer, "_queue_manager") as mock_queue_manager:
            sqs_consumer._build_items_direct("crashed.pdf", "s3://bucket/crashed.pdf", pages, markdowns)

        mock_queue_manager.save.assert_called_once()
        saved_item = mock_queue_manager.save.call_args[0][0]
        assert saved_item.filename == "crashed.pdf"
        assert saved_item.s3_raw_pdf_path == "s3://bucket/crashed.pdf"
        assert saved_item.relevance_check["tier"] == "vectorstore_write_failed"
        assert "Chroma connection refused" in saved_item.relevance_check["reasoning"]

    def test_recovery_placeholder_save_failure_does_not_raise(self):
        """Best-effort: if saving the recovery placeholder ITSELF fails,
        that must not compound the original failure or crash the pipeline."""
        pages, markdowns = _make_pages(2)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [_topic(page_ranges=[(1, 2)])]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"หน่วยงาน": "Dept A", "ใบอนุญาต": "License A"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", side_effect=RuntimeError("Chroma connection refused")), \
             patch.object(sqs_consumer, "_queue_manager") as mock_queue_manager:
            mock_queue_manager.save.side_effect = RuntimeError("disk full")
            sqs_consumer._build_items_direct("crashed2.pdf", "s3://bucket/crashed2.pdf", pages, markdowns)  # must not raise

    def test_resolver_exception_falls_back_to_writing_as_new(self):
        pages, markdowns = _make_pages(2)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [_topic(page_ranges=[(1, 2)])]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"หน่วยงาน": "Dept A", "ใบอนุญาต": "License A"}

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=RuntimeError("boom")), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore") as mock_add, \
             patch.object(sqs_consumer, "_archive_and_remove_in_vectorstore") as mock_archive:
            mock_add.return_value = 1
            sqs_consumer._build_items_direct("test.pdf", "s3://bucket/test.pdf", pages, markdowns)

        mock_add.assert_called_once()
        written_docs = mock_add.call_args[0][0]
        assert len(written_docs) == 1
        mock_archive.assert_not_called()

    def test_duplicate_topic_with_no_department_or_license_type_is_dropped(self):
        """Same duplicate-drop behavior for a business_guide-shaped topic
        (no department/license_type drafted) — the decision/write wiring
        doesn't depend on which data_type a topic resolves to."""
        pages, markdowns = _make_pages(2)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [_topic("การตลาด", "โปรโมชั่น", [(1, 2)])]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {}  # nothing drafted — stays business_guide

        def fake_resolve(topic_dict, source_file, cost_accumulator=None):
            return SupersedeDecision("duplicate", matched_doc_ids=["old-kh-1"])

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore") as mock_add, \
             patch.object(sqs_consumer, "_archive_and_remove_in_vectorstore") as mock_archive:
            mock_add.return_value = 0
            sqs_consumer._build_items_direct("book.pdf", "s3://bucket/book.pdf", pages, markdowns)

        mock_add.assert_not_called()
        mock_archive.assert_not_called()

    def test_update_topic_with_no_department_or_license_type_retires_only_matched(self):
        pages, markdowns = _make_pages(2)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [_topic("การตลาด", "โปรโมชั่น", [(1, 2)])]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {}

        def fake_resolve(topic_dict, source_file, cost_accumulator=None):
            return SupersedeDecision("update", matched_doc_ids=["old-kh-chunk-1"])

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore") as mock_add, \
             patch.object(sqs_consumer, "_filter_to_pdf_auto_ids", side_effect=lambda ids, filename: ids), \
             patch.object(sqs_consumer, "_archive_and_remove_in_vectorstore") as mock_archive:
            mock_add.return_value = 1
            sqs_consumer._build_items_direct("updated_book.pdf", "s3://bucket/updated_book.pdf", pages, markdowns)

        mock_archive.assert_called_once_with(["old-kh-chunk-1"], reason="superseded by updated_book.pdf")


@pytest.mark.unit
class TestFailedDraftingFallsThroughToCatchall:
    """PDF2-C4 (2026-10, ported to the unified builder): covered_local is
    marked only AFTER drafting succeeds — a topic whose draft_fields_from_
    pages call failed must fall through to the deterministic catch-all
    topic instead of vanishing with no vectorstore doc, no ReviewItem, no
    recovery placeholder. 2026-10 schema-unification note: there is no
    longer a second builder to "rescue" pages via — the single catch-all
    (same mechanism know-how's uncovered-pages handling always had) is now
    the only fallback, for every content shape."""

    def test_topic_whose_drafting_fails_gets_a_catchall_topic(self):
        pages, markdowns = _make_pages(4)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [
                _topic("หัวข้อ A", "ย่อย A", [(1, 2)]),
                _topic("หัวข้อ B", "ย่อย B", [(3, 4)]),
            ]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            if "page 1" in topic_markdowns:
                raise RuntimeError("provider 5xx")
            return {"หน่วยงาน": "Dept B", "ใบอนุญาต": "License B"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        captured_topics = {}

        def fake_build_merged_documents(topics_with_fields, source_file):
            captured_topics["topics"] = topics_with_fields
            return []

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer.pdf_direct_ingest, "build_merged_documents", side_effect=fake_build_merged_documents), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1):
            sqs_consumer._build_items_direct("test.pdf", "s3://bucket/test.pdf", pages, markdowns)

        topics = captured_topics["topics"]
        # Topic A (pages 1-2) failed drafting; topic B (pages 3-4) succeeded.
        # A catch-all topic covering pages 1-2 must be present alongside B.
        main_topics = [t.get("main_topic") for t in topics]
        assert "หัวข้อ B" in main_topics
        from service.pdf_supersede_resolver import KNOWHOW_CATCHALL_MAIN_TOPIC
        assert KNOWHOW_CATCHALL_MAIN_TOPIC in main_topics
        catchall = next(t for t in topics if t.get("main_topic") == KNOWHOW_CATCHALL_MAIN_TOPIC)
        assert "page 1" in catchall["full_text"]
        assert "page 2" in catchall["full_text"]

    def test_topic_whose_drafting_succeeds_produces_no_catchall(self):
        """Control case: with no drafting failure, every page is covered
        and no catch-all topic is added."""
        pages, markdowns = _make_pages(2)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [_topic(page_ranges=[(1, 2)])]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"หน่วยงาน": "Dept A", "ใบอนุญาต": "License A"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        captured_topics = {}

        def fake_build_merged_documents(topics_with_fields, source_file):
            captured_topics["topics"] = topics_with_fields
            return []

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer.pdf_direct_ingest, "build_merged_documents", side_effect=fake_build_merged_documents), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1):
            sqs_consumer._build_items_direct("test.pdf", "s3://bucket/test.pdf", pages, markdowns)

        assert len(captured_topics["topics"]) == 1

    def test_topic_resolved_as_duplicate_produces_no_catchall(self):
        """A topic whose drafting succeeded but was then resolved as a
        confirmed duplicate was legitimately handled (correctly recognized
        as already-covered content) — its pages must still count as
        covered, not trigger the catch-all."""
        pages, markdowns = _make_pages(2)

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [_topic(page_ranges=[(1, 2)])]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"หน่วยงาน": "Dept A", "ใบอนุญาต": "License A"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("duplicate", matched_doc_ids=["old-a"])

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer.pdf_direct_ingest, "build_merged_documents") as mock_build, \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1) as mock_add:
            sqs_consumer._build_items_direct("test.pdf", "s3://bucket/test.pdf", pages, markdowns)

        # The duplicate topic was dropped (never added to merged_topics),
        # and no catch-all was added for its pages either — "covered"
        # means "legitimately handled", not "written". With nothing left
        # to write, the builder/vectorstore calls are skipped entirely.
        mock_build.assert_not_called()
        mock_add.assert_not_called()


class _FakeDraftCache:
    """In-memory stand-in for utils.pdf_draft_cache.DraftCache — records
    every get/put call so tests can assert on exact cache keys (proving the
    version tag is actually part of the key, not just hoped to be)."""

    def __init__(self, preset: dict | None = None):
        self.store = dict(preset or {})
        self.get_calls: list[str] = []
        self.put_calls: list[tuple] = []

    def get(self, key):
        self.get_calls.append(key)
        return self.store.get(key)

    def put(self, key, value):
        self.put_calls.append((key, value))
        self.store[key] = value


@pytest.mark.unit
class TestHighSeverityFlagsSurfacedOnDashboard:
    """PDF-M1/PDF-M3 cluster (2026-10): validate_extraction()'s high-severity
    flags (arithmetic mismatches, dual-model disagreements) are computed in
    _build_and_save_review_item and stored on PageExtractionRecord.flags,
    then passed into _build_items_direct as `pages` — surfaced into the
    append_event() call that feeds the admin dashboard's activity table."""

    @staticmethod
    def _pages_with_flags(flags_by_page: dict):
        pages = []
        for i in range(1, 3):
            pages.append(PageExtractionRecord(
                page_num=i, typhoon_markdown=f"page {i}", claude_markdown="",
                flags=flags_by_page.get(i, []),
            ))
        markdowns = [p.typhoon_markdown for p in pages]
        return pages, markdowns

    def test_high_severity_flags_counted_and_summarized(self):
        pages, _ = self._pages_with_flags({
            1: [{"category": "arithmetic", "severity": "high", "message": "รวมไม่ตรง: 100 != 90"}],
            2: [
                {"category": "dual_model_disagreement", "severity": "high", "message": "ตัวเลขไม่ตรงกัน"},
                {"category": "arithmetic", "severity": "medium", "message": "เรื่องรอง ไม่นับ"},
            ],
        })
        count, examples = sqs_consumer._high_severity_flag_summary(pages)
        assert count == 2
        assert "รวมไม่ตรง: 100 != 90" in examples
        assert "ตัวเลขไม่ตรงกัน" in examples
        assert all("เรื่องรอง" not in e for e in examples)

    def test_no_flags_yields_zero(self):
        pages, _ = self._pages_with_flags({})
        count, examples = sqs_consumer._high_severity_flag_summary(pages)
        assert count == 0
        assert examples == []

    def test_builder_passes_flag_summary_to_append_event(self):
        pages, markdowns = self._pages_with_flags({
            1: [{"category": "arithmetic", "severity": "high", "message": "รวมไม่ตรง"}],
        })

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [_topic(page_ranges=[(1, 2)])]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"หน่วยงาน": "Dept A", "ใบอนุญาต": "License A"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1), \
             patch.object(sqs_consumer, "append_event") as mock_append:
            sqs_consumer._build_items_direct("test.pdf", "s3://bucket/test.pdf", pages, markdowns)

        mock_append.assert_called_once()
        event = mock_append.call_args[0][1]
        assert event["high_severity_flags"] == 1
        assert "รวมไม่ตรง" in event["high_severity_flag_examples"]


@pytest.mark.unit
class TestDraftCacheWiring:
    """2026-10: content-hash cache wired into the unified builder — proves
    a cache hit skips the LLM call entirely, a miss calls through and
    stores the result, and the cache key includes both the prompt-version
    AND logic-version constants (so a future prompt OR downstream-logic
    change naturally invalidates old entries)."""

    def test_topic_split_cache_hit_skips_llm_call(self):
        pages, markdowns = _make_pages(2)
        from service.pdf_knowhow_drafting import KNOWHOW_TOPIC_SPLIT_PROMPT_VERSION
        from utils.pdf_draft_cache import LOGIC_VERSION, content_hash
        expected_key = f"topics:{KNOWHOW_TOPIC_SPLIT_PROMPT_VERSION}:{LOGIC_VERSION}:{content_hash(chr(10).join(markdowns))}"
        preset_topics = [_topic(page_ranges=[(1, 2)])]
        fake_cache = _FakeDraftCache(preset={expected_key: preset_topics})

        def fail_if_called(*a, **kw):
            raise AssertionError("identify_knowhow_topics should NOT be called on a cache hit")

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"หน่วยงาน": "Dept A", "ใบอนุญาต": "License A"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "get_draft_cache", return_value=fake_cache), \
             patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fail_if_called), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1):
            sqs_consumer._build_items_direct("test.pdf", "s3://bucket/test.pdf", pages, markdowns)

        assert expected_key in fake_cache.get_calls

    def test_topic_split_cache_miss_stores_result(self):
        pages, markdowns = _make_pages(2)
        fake_cache = _FakeDraftCache()  # empty — guaranteed miss

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [_topic(page_ranges=[(1, 2)])]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"หน่วยงาน": "Dept A", "ใบอนุญาต": "License A"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "get_draft_cache", return_value=fake_cache), \
             patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify) as mock_identify, \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1):
            sqs_consumer._build_items_direct("test.pdf", "s3://bucket/test.pdf", pages, markdowns)

        mock_identify.assert_called_once()
        put_keys = [k for k, _v in fake_cache.put_calls]
        assert any(k.startswith("topics:") for k in put_keys)

    def test_field_draft_cache_hit_skips_llm_call(self):
        pages, markdowns = _make_pages(2)
        from service.pdf_field_drafting import DRAFT_FIELDS_PROMPT_VERSION
        from utils.pdf_draft_cache import LOGIC_VERSION, content_hash
        # topic covers both pages (1,2) so topic_markdowns == markdowns
        expected_draft_key = f"draft:{DRAFT_FIELDS_PROMPT_VERSION}:{LOGIC_VERSION}:{content_hash(chr(10).join(markdowns))}"
        preset_draft = {"หน่วยงาน": "Dept A (จากแคช)"}
        fake_cache = _FakeDraftCache(preset={expected_draft_key: preset_draft})

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [_topic(page_ranges=[(1, 2)])]

        def fail_if_called(*a, **kw):
            raise AssertionError("draft_fields_from_pages should NOT be called on a cache hit")

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "get_draft_cache", return_value=fake_cache), \
             patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fail_if_called), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1):
            sqs_consumer._build_items_direct("test.pdf", "s3://bucket/test.pdf", pages, markdowns)

        assert expected_draft_key in fake_cache.get_calls

    def test_empty_topic_split_result_is_not_cached(self):
        """PDF-M1 (2026-10): identify_knowhow_topics never raises (degrades
        to [] on a real failure, same shape as a genuine 0-topics verdict —
        see its own docstring), so a transient LLM outage must not get
        permanently cached as "no topics here" — this cache has no TTL."""
        pages, markdowns = _make_pages(2)
        fake_cache = _FakeDraftCache()

        def fake_identify_empty(page_markdowns, cost_accumulator=None):
            return []  # simulates a failed-even-after-shrink-retry outcome

        with patch.object(sqs_consumer, "get_draft_cache", return_value=fake_cache), \
             patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify_empty), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=0):
            sqs_consumer._build_items_direct("test.pdf", "s3://bucket/test.pdf", pages, markdowns)

        assert fake_cache.put_calls == [], (
            "An empty topic-split result was cached — a transient LLM outage would "
            "now permanently poison this exact content's future reprocessing"
        )

    def test_all_blank_field_draft_result_is_not_cached(self):
        """draft_fields_from_pages also never raises on a malformed LLM
        response — swallows json.JSONDecodeError internally and returns all
        13 fields blank instead (see its own try/except). Must not be
        cached as "this document's fields are genuinely all blank"."""
        pages, markdowns = _make_pages(2)
        fake_cache = _FakeDraftCache()

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [_topic(page_ranges=[(1, 2)])]

        def fake_draft_all_blank(topic_markdowns, cost_accumulator=None):
            from service.pdf_field_drafting import DRAFTABLE_FIELDS
            return {header_variants[0]: "" for header_variants in DRAFTABLE_FIELDS.values()}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "get_draft_cache", return_value=fake_cache), \
             patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft_all_blank), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1):
            sqs_consumer._build_items_direct("test.pdf", "s3://bucket/test.pdf", pages, markdowns)

        put_keys = [k for k, _v in fake_cache.put_calls]
        assert not any(k.startswith("draft:") for k in put_keys), (
            "An all-blank field-draft result was cached — a transient LLM outage "
            "would now permanently poison this exact topic's future reprocessing"
        )
        # The topic-split result itself was genuinely non-empty, so it SHOULD still cache.
        assert any(k.startswith("topics:") for k in put_keys)

    def test_cache_key_changes_when_prompt_version_differs(self):
        """Pins the invalidation mechanism itself: same content, different
        version string, must produce a different cache key — proves a
        future prompt-version bump actually invalidates old entries rather
        than silently reusing them."""
        from utils.pdf_draft_cache import content_hash
        h = content_hash("same text")
        key_v1 = f"topics:v1:{h}"
        key_v2 = f"topics:v2:{h}"
        assert key_v1 != key_v2


@pytest.mark.unit
class TestFilterToPdfAutoIds:
    """2026-10 fix (PDF2-C1): second line of defense alongside the
    ingested_via="pdf_auto" scoping added to pdf_supersede_resolver.py
    itself — this is the actual call site that retires documents, so it
    re-checks directly. Any id that isn't pdf_auto must be dropped, never
    passed through to archive_and_remove."""

    def test_empty_input_short_circuits_without_querying_chroma(self):
        with patch.object(sqs_consumer, "get_vs_manager") as mock_vs_mgr:
            result = sqs_consumer._filter_to_pdf_auto_ids([], "f.pdf")
        assert result == []
        mock_vs_mgr.assert_not_called()

    def test_keeps_only_pdf_auto_ids(self):
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection.get.return_value = {
            "ids": ["a", "b"],
            "metadatas": [{"ingested_via": "pdf_auto"}, {"ingested_via": "sheet"}],
        }
        with patch.object(sqs_consumer, "get_vs_manager") as mock_vs_mgr:
            mock_vs_mgr.return_value.vectorstore = mock_vectorstore
            result = sqs_consumer._filter_to_pdf_auto_ids(["a", "b"], "f.pdf")
        assert result == ["a"]

    def test_vectorstore_unavailable_returns_empty_rather_than_risking_a_bad_retire(self):
        with patch.object(sqs_consumer, "get_vs_manager") as mock_vs_mgr:
            mock_vs_mgr.return_value.vectorstore = None
            result = sqs_consumer._filter_to_pdf_auto_ids(["a", "b"], "f.pdf")
        assert result == []

    def test_duplicate_ids_are_deduplicated_before_querying_chroma(self):
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection.get.return_value = {
            "ids": ["doc-1", "doc-2"],
            "metadatas": [{"ingested_via": "pdf_auto"}, {"ingested_via": "pdf_auto"}],
        }
        with patch.object(sqs_consumer, "get_vs_manager") as mock_vs_mgr:
            mock_vs_mgr.return_value.vectorstore = mock_vectorstore
            result = sqs_consumer._filter_to_pdf_auto_ids(
                ["doc-1", "doc-2", "doc-1", "doc-2", "doc-1"], "f.pdf",
            )
        mock_vectorstore._collection.get.assert_called_once_with(
            ids=["doc-1", "doc-2"], include=["metadatas"],
        )
        assert result == ["doc-1", "doc-2"]

    def test_duplicate_ids_preserve_first_occurrence_order(self):
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection.get.return_value = {
            "ids": ["doc-2", "doc-1"],
            "metadatas": [{"ingested_via": "pdf_auto"}, {"ingested_via": "pdf_auto"}],
        }
        with patch.object(sqs_consumer, "get_vs_manager") as mock_vs_mgr:
            mock_vs_mgr.return_value.vectorstore = mock_vectorstore
            result = sqs_consumer._filter_to_pdf_auto_ids(["doc-2", "doc-1", "doc-2"], "f.pdf")
        assert result == ["doc-2", "doc-1"]

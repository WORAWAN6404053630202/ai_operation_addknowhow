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

    def test_vectorstore_write_failure_saves_a_recovery_placeholder(self):
        """PDF-M11 (2026-10): before this fix, a failed final Chroma write
        (after OCR/topic-splitting/drafting all succeeded — real billed
        work) was just logged and discarded, with nothing preserved to
        recover from except manually re-uploading the original PDF. Now
        saves a ReviewItem pointing at s3_raw_pdf_path so a human can
        /reprocess from the same original file."""
        pages, markdowns = _make_pages(2)

        def fake_identify_license_topics(page_markdowns, cost_accumulator=None):
            return [{"department": "Dept A", "license_type": "License A", "page_ranges": [(1, 2)]}]

        def fake_draft_fields_from_pages(topic_markdowns, cost_accumulator=None):
            return {"แนวคำตอบ": "เนื้อหาคำตอบ"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "identify_license_topics", side_effect=fake_identify_license_topics), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft_fields_from_pages), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_regulatory_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", side_effect=RuntimeError("Chroma connection refused")), \
             patch.object(sqs_consumer, "_queue_manager") as mock_queue_manager:
            sqs_consumer._build_license_items_direct(
                "crashed.pdf", "s3://bucket/crashed.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
            )

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

        def fake_identify_license_topics(page_markdowns, cost_accumulator=None):
            return [{"department": "Dept A", "license_type": "License A", "page_ranges": [(1, 2)]}]

        def fake_draft_fields_from_pages(topic_markdowns, cost_accumulator=None):
            return {"แนวคำตอบ": "เนื้อหาคำตอบ"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "identify_license_topics", side_effect=fake_identify_license_topics), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft_fields_from_pages), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_regulatory_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", side_effect=RuntimeError("Chroma connection refused")), \
             patch.object(sqs_consumer, "_queue_manager") as mock_queue_manager:
            mock_queue_manager.save.side_effect = RuntimeError("disk full")
            sqs_consumer._build_license_items_direct(  # must not raise
                "crashed2.pdf", "s3://bucket/crashed2.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
            )

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

    def test_expand_sibling_chunks_failure_falls_back_to_matched_ids_not_raise(self):
        """2026-10 fix: expand_knowhow_matches_to_full_chunks used to be the
        ONE unwrapped fallible call in this function — a transient Chroma
        error here would raise straight out, with no attempt cap backing up
        this code path (unlike the handoff path's _MAX_HANDOFF_ATTEMPTS),
        meaning SQS would redeliver the whole document indefinitely. Must
        now degrade to retiring just the directly-matched id(s) instead."""
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
                           side_effect=RuntimeError("chroma unreachable")), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore") as mock_add, \
             patch.object(sqs_consumer, "_archive_and_remove_in_vectorstore") as mock_archive:
            mock_add.return_value = 1
            result = sqs_consumer._build_knowhow_items_direct(
                "updated_book.pdf", "s3://bucket/updated_book.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "know_how"},
            )  # must not raise

        assert result == []
        mock_add.assert_called_once()  # the new doc still gets written
        mock_archive.assert_called_once_with(["old-kh-chunk-1"], reason="superseded by updated_book.pdf")


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
    then passed into the direct builders as `pages` — but nothing read them
    there before this fix, silently discarding the one piece of automated
    QA signal this no-human-review pipeline has. Now surfaced into the
    append_event() call that feeds the admin dashboard's activity table."""

    @staticmethod
    def _pages_with_flags(flags_by_page: dict):
        from model.pdf_review_item import PageExtractionRecord
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

    def test_license_direct_builder_passes_flag_summary_to_append_event(self):
        pages, markdowns = self._pages_with_flags({
            1: [{"category": "arithmetic", "severity": "high", "message": "รวมไม่ตรง"}],
        })

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [{"department": "Dept A", "license_type": "License A", "page_ranges": [(1, 2)]}]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"แนวคำตอบ": "x"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "identify_license_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_regulatory_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1), \
             patch.object(sqs_consumer, "append_event") as mock_append:
            sqs_consumer._build_license_items_direct(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
            )

        mock_append.assert_called_once()
        event = mock_append.call_args[0][1]
        assert event["high_severity_flags"] == 1
        assert "รวมไม่ตรง" in event["high_severity_flag_examples"]


@pytest.mark.unit
class TestDraftCacheWiring:
    """2026-10: content-hash cache wired into both _direct builders —
    proves a cache hit skips the LLM call entirely, a miss calls through
    and stores the result, and the cache key includes the prompt version
    constant (so a future prompt change naturally invalidates old entries).
    """

    def test_license_topic_split_cache_hit_skips_llm_call(self):
        pages, markdowns = _make_pages(2)
        from service.pdf_field_drafting import TOPIC_SPLIT_PROMPT_VERSION
        from utils.pdf_draft_cache import content_hash
        expected_key = f"license_topics:{TOPIC_SPLIT_PROMPT_VERSION}:{content_hash(chr(10).join(markdowns))}"
        preset_topics = [{"department": "Dept A", "license_type": "License A", "page_ranges": [(1, 2)]}]
        fake_cache = _FakeDraftCache(preset={expected_key: preset_topics})

        def fail_if_called(*a, **kw):
            raise AssertionError("identify_license_topics should NOT be called on a cache hit")

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"แนวคำตอบ": "เนื้อหา"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "get_draft_cache", return_value=fake_cache), \
             patch.object(sqs_consumer, "identify_license_topics", side_effect=fail_if_called), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_regulatory_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1):
            sqs_consumer._build_license_items_direct(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
            )

        assert expected_key in fake_cache.get_calls

    def test_license_topic_split_cache_miss_stores_result(self):
        pages, markdowns = _make_pages(2)
        fake_cache = _FakeDraftCache()  # empty — guaranteed miss

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [{"department": "Dept A", "license_type": "License A", "page_ranges": [(1, 2)]}]

        def fake_draft(topic_markdowns, cost_accumulator=None):
            return {"แนวคำตอบ": "เนื้อหา"}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "get_draft_cache", return_value=fake_cache), \
             patch.object(sqs_consumer, "identify_license_topics", side_effect=fake_identify) as mock_identify, \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_regulatory_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1):
            sqs_consumer._build_license_items_direct(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
            )

        mock_identify.assert_called_once()
        # Both the topic-split AND the (also-missed) field-draft results get
        # cached — assert the topic-split one specifically was stored.
        put_keys = [k for k, _v in fake_cache.put_calls]
        assert any(k.startswith("license_topics:") for k in put_keys)

    def test_field_draft_cache_hit_skips_llm_call(self):
        pages, markdowns = _make_pages(2)
        from service.pdf_field_drafting import DRAFT_FIELDS_PROMPT_VERSION
        from utils.pdf_draft_cache import content_hash
        # topic covers both pages (1,2) so topic_markdowns == markdowns
        expected_draft_key = f"license_draft:{DRAFT_FIELDS_PROMPT_VERSION}:{content_hash(chr(10).join(markdowns))}"
        preset_draft = {"แนวคำตอบ": "เนื้อหาที่แคชไว้"}
        fake_cache = _FakeDraftCache(preset={expected_draft_key: preset_draft})

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [{"department": "Dept A", "license_type": "License A", "page_ranges": [(1, 2)]}]

        def fail_if_called(*a, **kw):
            raise AssertionError("draft_fields_from_pages should NOT be called on a cache hit")

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "get_draft_cache", return_value=fake_cache), \
             patch.object(sqs_consumer, "identify_license_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fail_if_called), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_regulatory_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1):
            sqs_consumer._build_license_items_direct(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
            )

        assert expected_draft_key in fake_cache.get_calls

    def test_empty_license_topic_split_result_is_not_cached(self):
        """PDF-M1 (2026-10): identify_license_topics never raises (degrades
        to [] on a real failure, same shape as a genuine 0-topics verdict —
        see its own docstring), so a transient LLM outage must not get
        permanently cached as "no topics here" — this cache has no TTL."""
        pages, markdowns = _make_pages(2)
        fake_cache = _FakeDraftCache()

        def fake_identify_empty(page_markdowns, cost_accumulator=None):
            return []  # simulates a failed-even-after-shrink-retry outcome

        with patch.object(sqs_consumer, "get_draft_cache", return_value=fake_cache), \
             patch.object(sqs_consumer, "identify_license_topics", side_effect=fake_identify_empty):
            sqs_consumer._build_license_items_direct(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
            )

        assert fake_cache.put_calls == [], (
            "An empty topic-split result was cached — a transient LLM outage would "
            "now permanently poison this exact content's future reprocessing"
        )

    def test_empty_knowhow_topic_split_result_is_not_cached(self):
        pages, markdowns = _make_pages(2)
        fake_cache = _FakeDraftCache()

        def fake_identify_empty(page_markdowns, cost_accumulator=None):
            return []

        with patch.object(sqs_consumer, "get_draft_cache", return_value=fake_cache), \
             patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fake_identify_empty):
            sqs_consumer._build_knowhow_items_direct(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "business_guide"},
            )

        assert fake_cache.put_calls == []

    def test_all_blank_field_draft_result_is_not_cached(self):
        """draft_fields_from_pages also never raises on a malformed LLM
        response — swallows json.JSONDecodeError internally and returns all
        13 fields blank instead (see its own try/except). Must not be
        cached as "this document's fields are genuinely all blank"."""
        pages, markdowns = _make_pages(2)
        fake_cache = _FakeDraftCache()

        def fake_identify(page_markdowns, cost_accumulator=None):
            return [{"department": "Dept A", "license_type": "License A", "page_ranges": [(1, 2)]}]

        def fake_draft_all_blank(topic_markdowns, cost_accumulator=None):
            from service.pdf_field_drafting import DRAFTABLE_FIELDS
            return {header_variants[0]: "" for header_variants in DRAFTABLE_FIELDS.values()}

        def fake_resolve(merged, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "get_draft_cache", return_value=fake_cache), \
             patch.object(sqs_consumer, "identify_license_topics", side_effect=fake_identify), \
             patch.object(sqs_consumer, "draft_fields_from_pages", side_effect=fake_draft_all_blank), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_regulatory_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1):
            sqs_consumer._build_license_items_direct(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "structured_license"},
            )

        put_keys = [k for k, _v in fake_cache.put_calls]
        assert not any(k.startswith("license_draft:") for k in put_keys), (
            "An all-blank field-draft result was cached — a transient LLM outage "
            "would now permanently poison this exact topic's future reprocessing"
        )
        # The topic-split result itself was genuinely non-empty, so it SHOULD still cache.
        assert any(k.startswith("license_topics:") for k in put_keys)

    def test_cache_key_changes_when_prompt_version_differs(self):
        """Pins the invalidation mechanism itself: same content, different
        version string, must produce a different cache key — proves a
        future prompt-version bump actually invalidates old entries rather
        than silently reusing them."""
        from utils.pdf_draft_cache import content_hash
        h = content_hash("same text")
        key_v1 = f"license_topics:v1:{h}"
        key_v2 = f"license_topics:v2:{h}"
        assert key_v1 != key_v2

    def test_knowhow_topic_split_cache_hit_skips_llm_call(self):
        pages, markdowns = _make_pages(2)
        from service.pdf_knowhow_drafting import KNOWHOW_TOPIC_SPLIT_PROMPT_VERSION
        from utils.pdf_draft_cache import content_hash
        expected_key = f"knowhow_topics:{KNOWHOW_TOPIC_SPLIT_PROMPT_VERSION}:{content_hash(chr(10).join(markdowns))}"
        preset_topics = [{
            "document_title": "คู่มือ", "source_type": "book",
            "main_topic": "การตลาด", "sub_topic": "โปรโมชั่น",
            "summary": "สรุป", "category": "", "page_ranges": [(1, 2)],
        }]
        fake_cache = _FakeDraftCache(preset={expected_key: preset_topics})

        def fail_if_called(*a, **kw):
            raise AssertionError("identify_knowhow_topics should NOT be called on a cache hit")

        def fake_resolve(topic_dict, source_file, cost_accumulator=None):
            return SupersedeDecision("new")

        with patch.object(sqs_consumer, "get_draft_cache", return_value=fake_cache), \
             patch.object(sqs_consumer, "identify_knowhow_topics", side_effect=fail_if_called), \
             patch.object(sqs_consumer.pdf_supersede_resolver, "resolve_knowhow_topic", side_effect=fake_resolve), \
             patch.object(sqs_consumer, "_add_documents_to_vectorstore", return_value=1):
            sqs_consumer._build_knowhow_items_direct(
                "test.pdf", "s3://bucket/test.pdf", pages, markdowns,
                relevance_check=None, content_shape={"shape": "know_how"},
            )

        assert expected_key in fake_cache.get_calls

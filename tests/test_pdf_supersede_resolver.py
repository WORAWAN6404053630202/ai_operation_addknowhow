"""Tests for service/pdf_supersede_resolver.py — the automated (no-human-
review) content supersede resolver added 2026-10 for the PDF-as-sole-source
architecture. Covers the decision-aggregation logic deterministically by
mocking the 2 candidate-signal functions and the LLM judgment call — no
real Chroma, no real network/LLM calls, safe to run anytime.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from service import pdf_supersede_resolver as resolver  # noqa: E402


@pytest.mark.unit
class TestResolveDecision:
    def test_empty_pool_is_new(self):
        with patch.object(resolver, "_embedding_candidates", return_value={}), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}):
            decision = resolver.resolve_regulatory_topic(
                {"department": "สำนักงานประกันสังคม", "license_type": "ทะเบียนนายจ้าง", "answer_guideline": "เนื้อหา"},
                "new_law.pdf",
            )
        assert decision.decision == "new"
        assert decision.matched_doc_ids == []

    def test_no_comparable_text_is_new_without_calling_chroma(self):
        with patch.object(resolver, "_embedding_candidates") as mock_emb:
            decision = resolver.resolve_regulatory_topic({"department": "", "license_type": ""}, "x.pdf")
        assert decision.decision == "new"
        mock_emb.assert_not_called()

    def test_matched_and_llm_says_duplicate_skips(self):
        pool = {"doc-1": {"score": 0.95, "metadata": {"department": "สำนักงานประกันสังคม", "license_type": "ทะเบียนนายจ้าง"}, "content": "เดิม"}}
        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", return_value={
                 "doc-1": {"is_same_topic": True, "content_status": "duplicate", "reasoning": "same"},
             }):
            decision = resolver.resolve_regulatory_topic(
                {"department": "สำนักงานประกันสังคม", "license_type": "ทะเบียนนายจ้าง", "answer_guideline": "เนื้อหาใหม่"},
                "reupload.pdf",
            )
        assert decision.decision == "duplicate"
        assert decision.matched_doc_ids == ["doc-1"]

    def test_matched_and_llm_says_update_returns_matched_ids_for_retirement(self):
        pool = {"doc-2": {"score": 0.91, "metadata": {"department": "สำนักงานประกันสังคม", "license_type": "ทะเบียนนายจ้าง"}, "content": "เดิม"}}
        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", return_value={
                 "doc-2": {"is_same_topic": True, "content_status": "update", "reasoning": "fee changed"},
             }):
            decision = resolver.resolve_regulatory_topic(
                {"department": "สำนักงานประกันสังคม", "license_type": "ทะเบียนนายจ้าง", "answer_guideline": "ค่าธรรมเนียมใหม่"},
                "updated_law.pdf",
            )
        assert decision.decision == "update"
        assert decision.matched_doc_ids == ["doc-2"]

    def test_llm_failure_defaults_to_new_never_infers_update(self):
        """The resolver's own _llm_judge already swallows exceptions and
        returns {} — this test pins that {} (no judgments at all) must
        resolve to "new", never "update"/"duplicate", even though a
        candidate pool existed."""
        pool = {"doc-3": {"score": 0.9, "metadata": {"department": "X", "license_type": "Y"}, "content": "c"}}
        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", return_value={}):
            decision = resolver.resolve_regulatory_topic(
                {"department": "X", "license_type": "Y", "answer_guideline": "z"}, "f.pdf",
            )
        assert decision.decision == "new"

    def test_llm_says_not_same_topic_is_new_despite_signal_hits(self):
        pool = {"doc-4": {"score": 0.9, "metadata": {"department": "X", "license_type": "Y"}, "content": "c"}}
        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", return_value={
                 "doc-4": {"is_same_topic": False, "content_status": None, "reasoning": "actually different"},
             }):
            decision = resolver.resolve_regulatory_topic(
                {"department": "X", "license_type": "Y", "answer_guideline": "z"}, "f.pdf",
            )
        assert decision.decision == "new"

    def test_identity_only_match_still_reaches_llm_judgment(self):
        """A candidate surfaced ONLY by the identity-fuzzy signal (not
        embedding) must still be sent to the LLM for judgment — signals are
        unioned, not intersected."""
        pool = {"doc-5": {"score": None, "metadata": {"department": "X", "license_type": "Y"}, "content": "c"}}
        with patch.object(resolver, "_embedding_candidates", return_value={}), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value=pool) as mock_id, \
             patch.object(resolver, "_llm_judge", return_value={
                 "doc-5": {"is_same_topic": True, "content_status": "update", "reasoning": "r"},
             }) as mock_llm:
            decision = resolver.resolve_regulatory_topic(
                {"department": "X", "license_type": "Y", "answer_guideline": "z"}, "f.pdf",
            )
        mock_id.assert_called_once()
        mock_llm.assert_called_once()
        assert decision.decision == "update"


@pytest.mark.unit
class TestFieldDiffUsesContentAppropriateKeys:
    """PDF-M1 (2026-10): _field_diff defaults to _PROCEDURAL_FIELD_KEYS
    (registration_type/fees/operation_duration/...), which only matches the
    regulatory shape — previously passed unconditionally for know-how topics
    too (main_topic/sub_topic/summary/category, zero overlap), silently
    producing an empty diff for every know-how candidate. Also previously
    built existing_values from comparison_keys only, so even the regulatory
    path's diff never saw the actual old procedural field values."""

    def test_knowhow_diff_reflects_summary_change_not_empty(self):
        pool = {"doc-6": {"score": 0.9, "metadata": {
            "main_topic": "การตลาดออนไลน์", "sub_topic": "โปรโมชั่น",
            "summary": "สรุปเดิม", "category": "กลยุทธ์การตลาด",
        }, "content": "c"}}
        captured = {}

        def fake_llm_judge(new_summary, candidates, model, cost_accumulator=None):
            captured["candidates"] = candidates
            return {"doc-6": {"is_same_topic": True, "content_status": "update", "reasoning": "r"}}

        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", side_effect=fake_llm_judge):
            resolver.resolve_knowhow_topic(
                {
                    "main_topic": "การตลาดออนไลน์", "sub_topic": "โปรโมชั่น",
                    "summary": "สรุปใหม่ที่เปลี่ยนไป", "category": "กลยุทธ์การตลาด",
                    "full_text": "เนื้อหาเต็ม",
                },
                "f.pdf",
            )
        preview = captured["candidates"][0]["preview"]
        assert "diff=" in preview, f"Expected a non-empty field diff in the LLM preview, got: {preview!r}"
        assert "summary" in preview

    def test_regulatory_diff_shows_changed_not_added_for_existing_field(self):
        """Before the fix, existing_values never included the procedural
        fields at all, so old_v was always "" — every real change showed up
        as change_type="added" instead of "changed"."""
        pool = {"doc-7": {"score": 0.9, "metadata": {
            "department": "X", "license_type": "Y", "fees": "100 บาท",
        }, "content": "c"}}
        captured = {}

        def fake_llm_judge(new_summary, candidates, model, cost_accumulator=None):
            captured["candidates"] = candidates
            return {"doc-7": {"is_same_topic": True, "content_status": "update", "reasoning": "r"}}

        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", side_effect=fake_llm_judge):
            resolver.resolve_regulatory_topic(
                {"department": "X", "license_type": "Y", "answer_guideline": "z", "fees": "200 บาท"},
                "f.pdf",
            )
        preview = captured["candidates"][0]["preview"]
        assert "'change_type': 'changed'" in preview, (
            f"Expected fees' real old value (100 บาท) to be seen, giving change_type='changed', "
            f"not 'added': {preview!r}"
        )


@pytest.mark.unit
class TestKnowhowMetadataKeyAliasing:
    """PDF2-C2 (2026-10): service.pdf_direct_ingest.build_knowhow_documents
    never writes "summary"/"category" metadata keys at all — it writes the
    same logical fields under "answer_guideline"/"content_type" (the real
    field names retrieval/prompt-building code elsewhere depends on, so the
    writer wasn't changed). Before this fix, existing_values["summary"]/
    ["category"] were always None for every real live know-how doc, so a
    byte-identical re-upload's diff spuriously showed both fields as
    "added" every time, biasing the LLM judge toward "update" instead of
    "duplicate". The regression test below uses the REAL metadata shape
    (answer_guideline/content_type), unlike TestFieldDiffUsesContentAppropriateKeys
    above which predates this fix and uses literal "summary"/"category" keys
    that don't occur in real Chroma data — that test would have passed even
    with the bug present."""

    def test_identical_reupload_against_real_metadata_shape_shows_no_diff(self):
        pool = {"doc-9": {"score": 0.95, "metadata": {
            "main_topic": "การตลาดออนไลน์", "sub_topic": "โปรโมชั่น",
            "answer_guideline": "เนื้อหาเดิมเป๊ะ", "content_type": "กลยุทธ์การตลาด",
        }, "content": "c"}}
        captured = {}

        def fake_llm_judge(new_summary, candidates, model, cost_accumulator=None):
            captured["candidates"] = candidates
            return {"doc-9": {"is_same_topic": True, "content_status": "duplicate", "reasoning": "r"}}

        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", side_effect=fake_llm_judge):
            resolver.resolve_knowhow_topic(
                {
                    "main_topic": "การตลาดออนไลน์", "sub_topic": "โปรโมชั่น",
                    "summary": "เนื้อหาเดิมเป๊ะ", "category": "กลยุทธ์การตลาด",
                    "full_text": "เนื้อหาเต็ม",
                },
                "f.pdf",
            )
        preview = captured["candidates"][0]["preview"]
        assert "diff=" not in preview, (
            f"A byte-identical re-upload must show no diff once summary/category "
            f"are read from their real metadata keys (answer_guideline/content_type): {preview!r}"
        )

    def test_changed_reupload_against_real_metadata_shape_shows_diff(self):
        pool = {"doc-10": {"score": 0.95, "metadata": {
            "main_topic": "การตลาดออนไลน์", "sub_topic": "โปรโมชั่น",
            "answer_guideline": "เนื้อหาเดิม", "content_type": "กลยุทธ์การตลาด",
        }, "content": "c"}}
        captured = {}

        def fake_llm_judge(new_summary, candidates, model, cost_accumulator=None):
            captured["candidates"] = candidates
            return {"doc-10": {"is_same_topic": True, "content_status": "update", "reasoning": "r"}}

        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", side_effect=fake_llm_judge):
            resolver.resolve_knowhow_topic(
                {
                    "main_topic": "การตลาดออนไลน์", "sub_topic": "โปรโมชั่น",
                    "summary": "เนื้อหาใหม่ที่เปลี่ยนไปจริง", "category": "กลยุทธ์การตลาด",
                    "full_text": "เนื้อหาเต็ม",
                },
                "f.pdf",
            )
        preview = captured["candidates"][0]["preview"]
        assert "'change_type': 'changed'" in preview
        assert "'field': 'summary'" in preview


@pytest.mark.unit
class TestIdentityFuzzyCandidatesSkipsDocumentBodies:
    """PDF2-M4 (2026-10): this function's own docstring already said
    "metadata-only... no document bodies needed", but it fetched
    include=["metadatas", "documents"] anyway — "content" was never read
    anywhere downstream. Called once per topic from _resolve, a 40-topic
    book against a 1,500-doc corpus pulled ~40 full-partition fetches of
    real document bodies. Now fetches metadatas only."""

    def test_fetch_only_requests_metadatas(self):
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection.get.return_value = {"ids": [], "metadatas": []}
        with patch.object(resolver, "get_vs_manager") as mock_vs_mgr:
            mock_vs_mgr.return_value.vectorstore = mock_vectorstore
            resolver._identity_fuzzy_candidates(
                {"department": "X", "license_type": "Y"}, {"data_type": "regulatory"},
                resolver._REGULATORY_IDENTITY_KEYS,
            )
        call_kwargs = mock_vectorstore._collection.get.call_args.kwargs
        assert call_kwargs["include"] == ["metadatas"]

    def test_matched_candidate_still_returned_with_empty_content_placeholder(self):
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection.get.return_value = {
            "ids": ["doc-1"],
            "metadatas": [{"department": "X", "license_type": "Y"}],
        }
        with patch.object(resolver, "get_vs_manager") as mock_vs_mgr:
            mock_vs_mgr.return_value.vectorstore = mock_vectorstore
            result = resolver._identity_fuzzy_candidates(
                {"department": "X", "license_type": "Y"}, {"data_type": "regulatory"},
                resolver._REGULATORY_IDENTITY_KEYS,
            )
        assert "doc-1" in result
        assert result["doc-1"]["content"] == ""
        assert result["doc-1"]["metadata"] == {"department": "X", "license_type": "Y"}


@pytest.mark.unit
class TestKnowhowCatchAllExemption:
    def test_catchall_topic_is_always_new_without_touching_chroma(self):
        with patch.object(resolver, "_embedding_candidates") as mock_emb:
            decision = resolver.resolve_knowhow_topic(
                {
                    "main_topic": resolver.KNOWHOW_CATCHALL_MAIN_TOPIC,
                    "sub_topic": "ตรวจสอบและจัดหมวดหมู่ด้วยตนเอง",
                    "summary": "", "category": "", "full_text": "เนื้อหาที่เหลือ",
                },
                "f.pdf",
            )
        assert decision.decision == "new"
        mock_emb.assert_not_called()


@pytest.mark.unit
class TestDynamicCutoffCount:
    """2026-10 bug fix: _embedding_candidates previously kept every hit above
    the floor regardless of _MAX_CANDIDATES — these pin the now-enforced cap."""

    def test_finds_the_real_gap(self):
        scores = [0.95, 0.93, 0.75, 0.72, 0.71]
        assert resolver._dynamic_cutoff_count(scores, floor=0.70, min_gap=0.05, max_candidates=10) == 2

    def test_below_floor_returns_zero(self):
        assert resolver._dynamic_cutoff_count([0.5, 0.4, 0.3], floor=0.70, min_gap=0.05, max_candidates=10) == 0

    def test_no_clear_gap_caps_at_max_candidates(self):
        # Smoothly declining, all above floor, no real gap >= min_gap —
        # must still be capped, not returned uncapped.
        scores = [0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79]
        assert resolver._dynamic_cutoff_count(scores, floor=0.70, min_gap=0.05, max_candidates=5) == 5


@pytest.mark.unit
class TestEmbeddingCandidatesCap:
    def test_result_is_actually_capped_not_just_the_query_k(self):
        """Confirms the real bug is fixed: a Chroma query that returns MORE
        than max_candidates hits above the floor must still yield a capped
        result, not everything the query happened to return."""
        fake_hits = [
            {"id": f"doc-{i}", "score": 0.90 - i * 0.001, "metadata": {}, "content": ""}
            for i in range(20)  # all comfortably above any reasonable floor, smoothly declining (no gap)
        ]
        fake_vs_manager = type("FakeVS", (), {"query_live_with_ids": staticmethod(lambda *a, **kw: fake_hits)})()
        with patch.object(resolver, "get_vs_manager", return_value=fake_vs_manager), \
             patch.object(resolver.conf, "PDF_SUPERSEDE_EMBEDDING_MAX_CANDIDATES", 10), \
             patch.object(resolver.conf, "PDF_SUPERSEDE_EMBEDDING_FLOOR", 0.70), \
             patch.object(resolver.conf, "PDF_SUPERSEDE_EMBEDDING_MIN_GAP", 0.05):
            result = resolver._embedding_candidates("some comparison text", {"data_type": "regulatory"})
        assert len(result) <= 10


@pytest.mark.unit
class TestExhaustiveCandidateJudging:
    """2026-10 bug fix: _resolve used to silently truncate the combined
    candidate pool to PDF_SUPERSEDE_LLM_BATCH_SIZE before judging — any
    candidate past that cutoff (including identity-fuzzy-only matches,
    since embedding hits were merged in first) was never sent to the LLM at
    all. These confirm every candidate now gets judged, across as many
    _llm_judge calls as needed."""

    def test_all_candidates_judged_across_multiple_batches(self):
        # 3 candidates, batch_size=2 -> must take 2 _llm_judge calls, not 1,
        # and every candidate_id must appear in the union of what was sent.
        pool = {f"doc-{i}": {"score": 0.9, "metadata": {"department": "X", "license_type": "Y"}, "content": ""} for i in range(3)}
        seen_ids: set[str] = set()

        def fake_llm_judge(new_summary, candidates, model, cost_accumulator=None):
            for c in candidates:
                seen_ids.add(c["candidate_id"])
            return {}  # no confirmed match needed for this test — just proving coverage

        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", side_effect=fake_llm_judge) as mock_judge, \
             patch.object(resolver.conf, "PDF_SUPERSEDE_LLM_BATCH_SIZE", 2):
            resolver.resolve_regulatory_topic(
                {"department": "X", "license_type": "Y", "answer_guideline": "z"}, "f.pdf",
            )

        assert mock_judge.call_count == 2  # ceil(3/2)
        assert seen_ids == {"doc-0", "doc-1", "doc-2"}  # every candidate reached the judge, none silently dropped

    def test_match_found_only_in_the_second_batch_is_not_lost(self):
        """The specific failure mode the bug caused: a real duplicate sitting
        past the old truncation point must still be found now."""
        pool = {f"doc-{i}": {"score": 0.9, "metadata": {"department": "X", "license_type": "Y"}, "content": ""} for i in range(3)}

        def fake_llm_judge(new_summary, candidates, model, cost_accumulator=None):
            # Only "doc-2" (which would have been truncated away under the
            # old batch_size=2, take-first-2 behavior) is the real duplicate.
            ids = [c["candidate_id"] for c in candidates]
            if "doc-2" in ids:
                return {"doc-2": {"is_same_topic": True, "content_status": "duplicate", "reasoning": "same"}}
            return {}

        with patch.object(resolver, "_embedding_candidates", return_value=pool), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}), \
             patch.object(resolver, "_llm_judge", side_effect=fake_llm_judge), \
             patch.object(resolver.conf, "PDF_SUPERSEDE_LLM_BATCH_SIZE", 2):
            decision = resolver.resolve_regulatory_topic(
                {"department": "X", "license_type": "Y", "answer_guideline": "z"}, "f.pdf",
            )

        assert decision.decision == "duplicate"
        assert decision.matched_doc_ids == ["doc-2"]


@pytest.mark.unit
class TestPoolMergePreservesEmbeddingScore:
    def test_doc_matched_by_both_signals_keeps_embedding_score(self):
        with patch.object(resolver, "_embedding_candidates", return_value={
                 "doc-1": {"score": 0.93, "metadata": {"department": "X", "license_type": "Y"}, "content": ""}
             }), \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={
                 "doc-1": {"score": None, "metadata": {"department": "X", "license_type": "Y"}, "content": ""}
             }), \
             patch.object(resolver, "_llm_judge") as mock_judge:
            mock_judge.return_value = {}
            resolver.resolve_regulatory_topic(
                {"department": "X", "license_type": "Y", "answer_guideline": "z"}, "f.pdf",
            )
        # The pool passed into judging must have kept the real score, not
        # had it clobbered to None by the identity-fuzzy merge.
        _, kwargs = mock_judge.call_args
        # _llm_judge is called positionally (new_summary, candidates, model) —
        # inspect the call args directly rather than via kwargs.
        candidates_arg = mock_judge.call_args[0][1]
        assert len(candidates_arg) == 1  # doc-1 appears once, not duplicated by the merge


@pytest.mark.unit
class TestIngestedViaScoping:
    """2026-10 fix (PDF2-C1): resolve_regulatory_topic/resolve_knowhow_topic
    used to scope candidates by data_type alone — Sheet-sourced docs share
    the same data_type/main_topic/sub_topic shape as PDF-sourced ones, so a
    PDF re-upload whose topic happened to match a real Sheet row's topic
    could get judged "update" and archive that Sheet doc. Every
    data_type_filter passed to the candidate-signal functions must now also
    scope to ingested_via="pdf_auto"."""

    def test_regulatory_topic_filter_scopes_to_pdf_auto(self):
        with patch.object(resolver, "_embedding_candidates", return_value={}) as mock_emb, \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}) as mock_fuzzy:
            resolver.resolve_regulatory_topic(
                {"department": "X", "license_type": "Y", "answer_guideline": "z"}, "f.pdf",
            )
        emb_filter = mock_emb.call_args[0][1]
        fuzzy_filter = mock_fuzzy.call_args[0][1]
        assert {"ingested_via": "pdf_auto"} in emb_filter["$and"]
        assert {"ingested_via": "pdf_auto"} in fuzzy_filter["$and"]

    def test_knowhow_topic_filter_scopes_to_pdf_auto(self):
        with patch.object(resolver, "_embedding_candidates", return_value={}) as mock_emb, \
             patch.object(resolver, "_identity_fuzzy_candidates", return_value={}) as mock_fuzzy:
            resolver.resolve_knowhow_topic(
                {"main_topic": "A", "sub_topic": "B", "summary": "s", "category": "c"}, "f.pdf",
            )
        emb_filter = mock_emb.call_args[0][1]
        fuzzy_filter = mock_fuzzy.call_args[0][1]
        assert {"ingested_via": "pdf_auto"} in emb_filter["$and"]
        assert {"ingested_via": "pdf_auto"} in fuzzy_filter["$and"]

    def test_collect_full_sibling_chunk_ids_scopes_to_pdf_auto(self):
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection.get.return_value = {"ids": ["a", "b"]}
        with patch.object(resolver, "get_vs_manager") as mock_vs_mgr:
            mock_vs_mgr.return_value.vectorstore = mock_vectorstore
            result = resolver.collect_full_sibling_chunk_ids("Main", "Sub")
        where = mock_vectorstore._collection.get.call_args.kwargs["where"]
        assert {"ingested_via": "pdf_auto"} in where["$and"]
        assert result == ["a", "b"]

"""Tests for scripts/pdf_supersede_backfill.py (2026-10) — the one-time
cleanup of docs stuck permanently blocked under the OLD pdf_supersede_
resolver.py behavior (before its field_merges redesign). Mocks get_vs_manager
(no real Chroma), resolve_topic, and _check_content_completeness (no real
LLM calls) — safe to run anytime.

Covers: no-longer-live ids are skipped and reported; a confirmed "duplicate"
sibling retires the old doc with no merge; a confirmed "update" sibling
only proceeds when it's clearly newer (ingested_at); field_merges (when
non-empty) land in update_live_metadata on the sibling's FULL chunk-id
group before the old doc's full chunk-id group is archived; dry run (no
--confirm) never calls archive_and_remove/update_live_metadata; multiple
input ids from the same logical topic are deduped to one resolve_topic
call; --limit caps how many logical topics get processed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import scripts.pdf_supersede_backfill as backfill_script  # noqa: E402
from service.pdf_supersede_resolver import SupersedeDecision  # noqa: E402


class _FakeCollection:
    """docs: {doc_id: metadata_dict}. Mimics the two Chroma .get() call
    shapes this script actually uses: by explicit `ids`, and by a
    `where={"$and": [...]}` equality filter (the sibling-chunk-group
    lookup)."""

    def __init__(self, docs: dict):
        self.docs = dict(docs)

    def get(self, ids=None, where=None, include=None):
        if ids is not None:
            found = [(i, self.docs[i]) for i in ids if i in self.docs]
            return {"ids": [i for i, _ in found], "metadatas": [m for _, m in found]}
        if where is not None:
            conds = where["$and"] if "$and" in where else [where]

            def _matches(meta):
                return all(meta.get(k) == v for cond in conds for k, v in cond.items())

            matched = [i for i, m in self.docs.items() if _matches(m)]
            return {"ids": matched}
        return {"ids": list(self.docs.keys())}


def _mock_manager(docs: dict):
    mgr = MagicMock()
    mgr.vectorstore = MagicMock()
    mgr.vectorstore._collection = _FakeCollection(docs)
    return mgr


def _write_ids_file(tmp_path: Path, ids: list[str]) -> str:
    p = tmp_path / "stuck_ids.txt"
    p.write_text("\n".join(ids), encoding="utf-8")
    return str(p)


def _run(args: list[str]):
    with patch.object(sys, "argv", ["pdf_supersede_backfill.py"] + args):
        backfill_script.main()


_BASE_META = {
    "source_file": "f.pdf", "main_topic": "A", "sub_topic": "A1", "ingested_at": 1000.0,
    "ingested_via": "pdf_auto",
}


@pytest.mark.unit
class TestNoLongerLiveIds:
    def test_id_missing_from_live_collection_is_reported_and_skipped(self, tmp_path, capsys):
        ids_file = _write_ids_file(tmp_path, ["gone-1"])
        mgr = _mock_manager({})
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr):
            _run(["--ids-file", ids_file])
        out = capsys.readouterr().out
        assert "gone-1" in out
        assert "no longer live" in out


@pytest.mark.unit
class TestNoConfirmedSibling:
    def test_new_decision_is_skipped_as_no_sibling(self, tmp_path, capsys):
        ids_file = _write_ids_file(tmp_path, ["old-1"])
        mgr = _mock_manager({"old-1": dict(_BASE_META)})
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", return_value=SupersedeDecision("new")):
            _run(["--ids-file", ids_file])
        out = capsys.readouterr().out
        assert "[SKIP]" in out and "no confirmed live sibling" in out
        mgr.archive_and_remove.assert_not_called()

    def test_multiple_confirmed_siblings_is_ambiguous(self, tmp_path, capsys):
        ids_file = _write_ids_file(tmp_path, ["old-1"])
        mgr = _mock_manager({"old-1": dict(_BASE_META)})
        decision = SupersedeDecision("update", matched_doc_ids=["sib-1", "sib-2"])
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", return_value=decision):
            _run(["--ids-file", ids_file])
        out = capsys.readouterr().out
        assert "ambiguous" in out
        mgr.archive_and_remove.assert_not_called()


@pytest.mark.unit
class TestDuplicateDecisionRetiresWithNoMerge:
    def test_confirmed_duplicate_retires_old_doc_dry_run_prints_only(self, tmp_path, capsys):
        ids_file = _write_ids_file(tmp_path, ["old-1"])
        mgr = _mock_manager({
            "old-1": dict(_BASE_META),
            "sib-1": {"source_file": "f.pdf", "main_topic": "A", "sub_topic": "A1-reworded", "ingested_at": 2000.0},
        })
        decision = SupersedeDecision("duplicate", matched_doc_ids=["sib-1"])
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", return_value=decision):
            _run(["--ids-file", ids_file])
        out = capsys.readouterr().out
        assert "[DUPLICATE]" in out
        mgr.archive_and_remove.assert_not_called()  # dry run
        mgr.update_live_metadata.assert_not_called()

    def test_confirmed_duplicate_with_confirm_actually_archives(self, tmp_path):
        ids_file = _write_ids_file(tmp_path, ["old-1"])
        mgr = _mock_manager({
            "old-1": dict(_BASE_META),
            "sib-1": {"source_file": "f.pdf", "main_topic": "A", "sub_topic": "A1-reworded", "ingested_at": 2000.0},
        })
        decision = SupersedeDecision("duplicate", matched_doc_ids=["sib-1"])
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", return_value=decision):
            _run(["--ids-file", ids_file, "--confirm"])
        mgr.archive_and_remove.assert_called_once_with(["old-1"], reason="pdf_supersede_backfill: duplicate of sib-1")
        mgr.update_live_metadata.assert_not_called()


@pytest.mark.unit
class TestUpdateDecisionDirectionViaIngestedAt:
    def test_sibling_not_newer_is_ambiguous_and_skipped(self, tmp_path, capsys):
        """The stuck doc's own ingested_at >= the matched sibling's — can't
        tell which side is the real keeper, must not guess."""
        ids_file = _write_ids_file(tmp_path, ["old-1"])
        mgr = _mock_manager({
            "old-1": dict(_BASE_META, ingested_at=5000.0),
            "sib-1": {"source_file": "f.pdf", "main_topic": "A", "sub_topic": "A1-v2", "ingested_at": 1000.0},
        })
        decision = SupersedeDecision("update", matched_doc_ids=["sib-1"])
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", return_value=decision):
            _run(["--ids-file", ids_file])
        out = capsys.readouterr().out
        assert "not clearly newer" in out
        mgr.archive_and_remove.assert_not_called()

    def test_sibling_missing_from_live_collection_is_skipped(self, tmp_path, capsys):
        ids_file = _write_ids_file(tmp_path, ["old-1"])
        mgr = _mock_manager({"old-1": dict(_BASE_META)})  # sib-1 not present
        decision = SupersedeDecision("update", matched_doc_ids=["sib-1"])
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", return_value=decision):
            _run(["--ids-file", ids_file])
        out = capsys.readouterr().out
        assert "no longer live" in out
        mgr.archive_and_remove.assert_not_called()


@pytest.mark.unit
class TestUpdateDecisionWithNewerSiblingMergesThenRetires:
    def _setup(self, tmp_path, old_extra_ids=(), sib_extra_ids=()):
        ids_file = _write_ids_file(tmp_path, ["old-1"])
        docs = {
            "old-1": dict(_BASE_META, ingested_at=1000.0),
            "sib-1": {"source_file": "f.pdf", "main_topic": "A", "sub_topic": "A1-v2", "ingested_at": 2000.0},
        }
        for i in old_extra_ids:
            docs[i] = dict(_BASE_META, ingested_at=1000.0)
        for i in sib_extra_ids:
            docs[i] = {"source_file": "f.pdf", "main_topic": "A", "sub_topic": "A1-v2", "ingested_at": 2000.0}
        mgr = _mock_manager(docs)
        return ids_file, mgr

    def test_omitted_claims_merge_into_sibling_before_retiring_old(self, tmp_path):
        ids_file, mgr = self._setup(tmp_path)
        decision = SupersedeDecision("update", matched_doc_ids=["sib-1"])
        missing_claims = {"old-1": [{"field": "fees", "claim": "ค่าธรรมเนียม 100 บาท", "status": "omitted"}]}
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", return_value=decision), \
             patch.object(backfill_script, "_check_content_completeness", return_value=missing_claims):
            _run(["--ids-file", ids_file, "--confirm"])
        mgr.update_live_metadata.assert_called_once()
        call_args = mgr.update_live_metadata.call_args[0]
        assert call_args[0] == ["sib-1"]
        assert "ค่าธรรมเนียม 100 บาท" in call_args[1]["fees"]
        mgr.archive_and_remove.assert_called_once_with(["old-1"], reason="pdf_supersede_backfill: superseded by sib-1")

    def test_merge_reaches_every_sibling_sub_chunk_not_just_the_matched_id(self, tmp_path):
        """The real use case the module docstring calls out: a chunk_total>1
        topic writes one Document PER sub-chunk — a merge into the keeper
        must patch every one of them."""
        ids_file, mgr = self._setup(tmp_path, old_extra_ids=["old-2"], sib_extra_ids=["sib-2"])
        decision = SupersedeDecision("update", matched_doc_ids=["sib-1"])
        missing_claims = {"old-1": [{"field": "fees", "claim": "100 บาท", "status": "omitted"}]}
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", return_value=decision), \
             patch.object(backfill_script, "_check_content_completeness", return_value=missing_claims):
            _run(["--ids-file", ids_file, "--confirm"])
        sibling_ids_called = mgr.update_live_metadata.call_args[0][0]
        assert sorted(sibling_ids_called) == ["sib-1", "sib-2"]
        old_ids_archived = mgr.archive_and_remove.call_args[0][0]
        assert sorted(old_ids_archived) == ["old-1", "old-2"]

    def test_no_omitted_claims_retires_without_merging(self, tmp_path):
        ids_file, mgr = self._setup(tmp_path)
        decision = SupersedeDecision("update", matched_doc_ids=["sib-1"])
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", return_value=decision), \
             patch.object(backfill_script, "_check_content_completeness", return_value={}):
            _run(["--ids-file", ids_file, "--confirm"])
        mgr.update_live_metadata.assert_not_called()
        mgr.archive_and_remove.assert_called_once_with(["old-1"], reason="pdf_supersede_backfill: superseded by sib-1")

    def test_dry_run_never_calls_archive_or_update(self, tmp_path):
        ids_file, mgr = self._setup(tmp_path)
        decision = SupersedeDecision("update", matched_doc_ids=["sib-1"])
        missing_claims = {"old-1": [{"field": "fees", "claim": "100 บาท", "status": "omitted"}]}
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", return_value=decision), \
             patch.object(backfill_script, "_check_content_completeness", return_value=missing_claims):
            _run(["--ids-file", ids_file])  # no --confirm
        mgr.update_live_metadata.assert_not_called()
        mgr.archive_and_remove.assert_not_called()


@pytest.mark.unit
class TestAllLiveFlag:
    def test_all_live_scans_every_pdf_auto_doc_without_an_ids_file(self, capsys):
        mgr = _mock_manager({
            "old-1": dict(_BASE_META),
            "old-2": dict(_BASE_META, main_topic="B"),
        })
        mock_resolve = MagicMock(return_value=SupersedeDecision("new"))
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", mock_resolve):
            _run(["--all-live"])
        out = capsys.readouterr().out
        assert "--all-live: found 2 live pdf_auto doc(s)" in out
        assert mock_resolve.call_count == 2

    def test_all_live_and_ids_file_together_is_rejected(self, tmp_path):
        ids_file = _write_ids_file(tmp_path, ["old-1"])
        with pytest.raises(SystemExit):
            _run(["--ids-file", ids_file, "--all-live"])

    def test_neither_all_live_nor_ids_file_is_rejected(self):
        with pytest.raises(SystemExit):
            _run([])


@pytest.mark.unit
class TestSiblingChunkIdsAlwaysIncludesKnownId:
    """2026-10 fix (live-confirmed on the real corpus's clean dry run): a doc
    whose metadata is missing the main_topic/sub_topic keys entirely (not
    just blank — genuinely absent, seen on at least one real legacy live
    doc) makes _group_key() treat it as ("", "") on the Python side, but
    Chroma's own where-equality filter does NOT treat a missing key as
    equal to "" — so a query for that doc's own group came back with ZERO
    ids, not even its own. Silently turned "merge field(s) [...] into
    sibling [], then retiring []" — a no-op (both downstream calls safely
    no-op on an empty list, so no data was corrupted), but a genuinely
    stuck doc stayed stuck despite the LLM cost already being spent on it."""

    def test_query_returning_nothing_still_yields_the_known_id(self):
        vectorstore = MagicMock()
        vectorstore._collection = _FakeCollection({})  # empty collection -> where-query always returns []
        result = backfill_script._sibling_chunk_ids(vectorstore, ("f.pdf", "", ""), "self-id")
        assert result == ["self-id"]

    def test_query_hits_are_combined_with_the_known_id_not_replaced(self):
        vectorstore = MagicMock()
        vectorstore._collection = _FakeCollection({"sib-1": {"source_file": "f.pdf", "main_topic": "A", "sub_topic": "A1"}})
        result = backfill_script._sibling_chunk_ids(vectorstore, ("f.pdf", "A", "A1"), "self-id")
        assert sorted(result) == ["self-id", "sib-1"]

    def test_end_to_end_blank_topic_metadata_still_produces_nonempty_merge_and_retire_lists(self, tmp_path):
        """The exact live scenario: old-1's metadata has NO main_topic/
        sub_topic keys at all. Must not end up calling update_live_metadata/
        archive_and_remove with empty id lists."""
        ids_file = _write_ids_file(tmp_path, ["old-1"])
        mgr = _mock_manager({
            "old-1": {"source_file": "f.pdf", "ingested_at": 1000.0},  # no main_topic/sub_topic at all
            "sib-1": {"source_file": "g.pdf", "main_topic": "B", "sub_topic": "B1", "ingested_at": 2000.0},
        })
        decision = SupersedeDecision("update", matched_doc_ids=["sib-1"])
        missing_claims = {"old-1": [{"field": "fees", "claim": "100 บาท", "status": "omitted"}]}
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", return_value=decision), \
             patch.object(backfill_script, "_check_content_completeness", return_value=missing_claims):
            _run(["--ids-file", ids_file, "--confirm"])
        mgr.update_live_metadata.assert_called_once()
        call_args = mgr.update_live_metadata.call_args[0]
        assert call_args[0] == ["sib-1"]
        assert "100 บาท" in call_args[1]["fees"]
        mgr.archive_and_remove.assert_called_once_with(["old-1"], reason="pdf_supersede_backfill: superseded by sib-1")


@pytest.mark.unit
class TestOwnSiblingChunkSelfMatchGuard:
    """2026-10 fix (live-confirmed on the real corpus's first --all-live dry
    run): a topic with chunk_total > 1 writes one Document PER sub-chunk,
    all sharing the exact same (source_file, main_topic, sub_topic) — the
    representative's OWN other sub-chunk(s) must never be treated as "the
    confirmed sibling to merge into and retire", since that computes the
    SAME id set for both "keep" and "archive", which would delete the
    entire topic. Two independent layers are covered here: (1) resolve_topic
    is called with exclude_doc_ids covering the representative's full own
    chunk group, not just its single id; (2) even if a same-group id slipped
    through anyway, the script's own belt-and-suspenders check refuses."""

    def test_resolve_topic_is_called_with_the_full_own_chunk_group_excluded(self, tmp_path):
        ids_file = _write_ids_file(tmp_path, ["old-1"])
        mgr = _mock_manager({
            "old-1": dict(_BASE_META),
            "old-1-chunk2": dict(_BASE_META),  # same (source_file, main_topic, sub_topic) group as old-1
        })
        mock_resolve = MagicMock(return_value=SupersedeDecision("new"))
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", mock_resolve):
            _run(["--ids-file", ids_file])
        call_kwargs = mock_resolve.call_args.kwargs
        assert sorted(call_kwargs["exclude_doc_ids"]) == ["old-1", "old-1-chunk2"]

    def test_confirmed_sibling_in_the_same_group_is_refused_not_merged(self, tmp_path, capsys):
        """Defense-in-depth: simulates the exact live bug even if the root-
        cause exclusion somehow failed to prevent it upstream — resolve_topic
        here returns a "sibling" that, on inspection, shares the
        representative's own (source_file, main_topic, sub_topic)."""
        ids_file = _write_ids_file(tmp_path, ["old-1"])
        mgr = _mock_manager({
            "old-1": dict(_BASE_META),
            "old-1-chunk2": dict(_BASE_META),  # same group — the bug scenario
        })
        decision = SupersedeDecision("duplicate", matched_doc_ids=["old-1-chunk2"])
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", return_value=decision):
            _run(["--ids-file", ids_file, "--confirm"])
        out = capsys.readouterr().out
        assert "SAME logical topic" in out
        mgr.archive_and_remove.assert_not_called()
        mgr.update_live_metadata.assert_not_called()


@pytest.mark.unit
class TestDedupAndLimit:
    def test_multiple_ids_in_same_logical_topic_resolve_only_once(self, tmp_path):
        ids_file = _write_ids_file(tmp_path, ["old-1", "old-2"])
        mgr = _mock_manager({
            "old-1": dict(_BASE_META),
            "old-2": dict(_BASE_META),  # same (source_file, main_topic, sub_topic) group
        })
        mock_resolve = MagicMock(return_value=SupersedeDecision("new"))
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", mock_resolve):
            _run(["--ids-file", ids_file])
        assert mock_resolve.call_count == 1

    def test_limit_caps_number_of_logical_topics_processed(self, tmp_path):
        ids_file = _write_ids_file(tmp_path, ["old-1", "old-2"])
        mgr = _mock_manager({
            "old-1": dict(_BASE_META, main_topic="A"),
            "old-2": dict(_BASE_META, main_topic="B"),  # a DIFFERENT logical topic
        })
        mock_resolve = MagicMock(return_value=SupersedeDecision("new"))
        with patch.object(backfill_script, "get_vs_manager", return_value=mgr), \
             patch.object(backfill_script, "resolve_topic", mock_resolve):
            _run(["--ids-file", ids_file, "--limit", "1"])
        assert mock_resolve.call_count == 1

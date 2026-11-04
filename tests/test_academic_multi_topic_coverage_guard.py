"""
Regression pin for a fix in AcademicPersonaService._filter_docs_for_multi_topics
(2026-11, retrieval-safety audit — found while triaging the static-scan tool
code/scripts/filter_guard_static_scan.py's findings beyond the originally-scoped
persona_supervisor.py sites).

Background: when a user selects multiple topics at once (e.g. "3 4"), this
method retrieves docs per selected `key` via a license_type filter and merges
them. `_retrieve_docs` silently substitutes an unfiltered semantic result when
its metadata filter matches 0 docs (same documented behavior as the
single-topic selection path earlier in this file, which already guards against
it via `_has_matched_filtered` — see test_no_change_sites_regression_pin.py),
so `fresh` was never empty even when nothing in the corpus actually carried
`key`. Unlike the single-topic path, this multi-topic merge loop had no such
check — an unrelated-topic doc could silently merge into the result under a
key it doesn't actually belong to, with no way to tell. Fixed by porting the
same verify-then-retry shape into this loop.
"""
from __future__ import annotations

import inspect
import re

import pytest

from model.persona_academic import AcademicPersonaService


def _multi_topic_loop_source() -> str:
    source = inspect.getsource(AcademicPersonaService._filter_docs_for_multi_topics)
    m = re.search(
        r"for key in selected_keys:(.*?)# Include any topic-matching docs already in broad retrieval set",
        source, re.DOTALL,
    )
    assert m, "per-key retrieval loop not found in _filter_docs_for_multi_topics"
    return m.group(1)


@pytest.mark.unit
class TestMultiTopicMergeVerifiesEachKeyBeforeMerging:
    def test_checks_whether_the_license_type_filter_actually_matched(self):
        block = _multi_topic_loop_source()
        assert "_key_matched = any(" in block
        assert '.strip() == key' in block

    def test_checks_main_topic_and_sub_topic_too_not_just_license_type(self):
        block = _multi_topic_loop_source()
        assert '.get("main_topic") or "").strip() == key' in block
        assert '.get("sub_topic") or "").strip() == key' in block

    def test_retries_with_main_topic_filter_when_unmatched(self):
        block = _multi_topic_loop_source()
        assert 'metadata_filter={"main_topic": key}' in block

    def test_falls_back_to_unverified_result_rather_than_dropping_the_key_entirely(self):
        """"only add coverage, never regress" — even if nothing verifies, the
        key's best-effort semantic result is still merged (logged), not thrown
        away entirely."""
        block = _multi_topic_loop_source()
        assert "best-effort fallback" in block

    def test_is_exception_safe(self):
        block = _multi_topic_loop_source()
        assert "except Exception as _e_mtk" in block

"""
Phase 4 of the 2026-11 retrieval-safety audit (department/license_type filter
gap — see test_license_filter_coverage_cascade.py for the sites that WERE
changed). This file pins the sites the audit concluded do NOT need the same
identity-guard/coverage-adequacy treatment, as a safety net against someone
"helpfully" removing or restructuring this code later under the mistaken
impression it's redundant with the newly-wired sites.

Three categories confirmed safe, each represented by one pinned site:

1. persona_academic.py's menu-driven license_type filters already verify
   the filter actually matched before committing to it (_has_matched_filtered),
   retrying with a different filter key rather than silently accepting
   whatever the filter (or its internal unfiltered fallback) returned.

2. persona_practical.py's "established-topic guarantee" sites are additive
   only — they PREPEND docs when a known-relevant department/license_type
   is absent from the current result set; they can never exclude a doc.

3. persona_practical.py's "location-switch" site is non-short-circuiting —
   an empty filtered result just resets a flag and falls through; it never
   `return`s early the way the two Phase-2-fixed sites used to.
"""
from __future__ import annotations

import inspect
import re

import pytest

from model.persona_academic import AcademicPersonaService
from model.persona_practical import PracticalPersonaService


def _academic_handle_slot_source() -> str:
    # _has_matched_filtered lives in the topic-selection / targeted re-retrieval
    # path — search across the whole class's handle method (same "large,
    # stateful method" situation as persona_supervisor.py's documented precedent).
    source = inspect.getsource(AcademicPersonaService)
    m = re.search(
        r"_lt_filter: Optional\[Dict\[str, Any\]\] = \{\"license_type\": selected_key\}"
        r"(.*?)supervisor hint retrieval failed \(%s\)\", _e_sv\)",
        source, re.DOTALL,
    )
    assert m, "targeted re-retrieval + _has_matched_filtered block not found in AcademicPersonaService"
    return m.group(0)


def _practical_handle_source() -> str:
    return inspect.getsource(PracticalPersonaService.handle)


def _established_topic_guarantee_dept_source() -> str:
    source = _practical_handle_source()
    m = re.search(
        r"_cs_established_dept = str\((.*?)established-topic guarantee failed: %s\", _e_guarantee\)",
        source, re.DOTALL,
    )
    assert m, "established-topic guarantee (department) block not found in PracticalPersonaService.handle"
    return m.group(0)


def _location_switch_source() -> str:
    source = _practical_handle_source()
    m = re.search(
        r"# Lock to current topic's license_type(.*?)location-switch retrieval failed: %s\", _loc_err\)\n\s+_query_loc_override = \"\"",
        source, re.DOTALL,
    )
    assert m, "location-switch retrieval block not found in PracticalPersonaService.handle"
    return m.group(0)


@pytest.mark.unit
class TestAcademicMenuDrivenFilterVerifiesBeforeCommitting:
    """Menu-driven context (user already picked a topic number) is lower-risk
    than the two Phase-2-fixed free-text-keyword-triggered sites, AND this
    path additionally verifies the filter's result before trusting it —
    unlike those two sites, which committed to any non-empty result."""

    def test_checks_whether_filter_actually_matched_not_just_non_empty(self):
        block = _academic_handle_slot_source()
        assert "_has_matched_filtered = any(" in block
        assert '.strip() == selected_key' in block

    def test_retries_with_main_topic_filter_when_license_type_filter_did_not_match(self):
        block = _academic_handle_slot_source()
        assert 'metadata_filter={"main_topic": selected_key}' in block

    def test_falls_back_to_supervisor_hint_when_neither_filter_matched(self):
        block = _academic_handle_slot_source()
        assert "_sv_lt_hint" in block
        assert "_used_sv_hint" in block


@pytest.mark.unit
class TestEstablishedTopicGuaranteeIsAdditiveOnly:
    """Fetches by department+topic and PREPENDS only when that department is
    absent from the current docs — structurally cannot exclude a doc, so it
    doesn't share the Phase-2-fixed sites' risk (a wrong/missing value here
    just means a department doesn't get its extra docs added, not that a
    correct doc gets dropped)."""

    def test_skips_departments_already_present_rather_than_refetching(self):
        block = _established_topic_guarantee_dept_source()
        assert "if _target_dept in _depts_present:" in block
        assert "continue" in block

    def test_result_is_prepended_not_assigned(self):
        block = _established_topic_guarantee_dept_source()
        assert "_docs_to_process = _guar_docs + _docs_to_process" in block

    def test_is_exception_safe(self):
        block = _established_topic_guarantee_dept_source()
        assert "except Exception as _e_guarantee" in block

    def test_handles_multiple_mentioned_departments_not_just_one(self):
        """Confirms this site already generalizes beyond a single
        'established' department — relevant context for why it doesn't
        need the within-group majority-vote guard: it's not picking ONE
        reference value and comparing others against it, it's guaranteeing
        coverage for EVERY department the user actually named."""
        block = _established_topic_guarantee_dept_source()
        assert "_multi_depts_mentioned" in block
        assert "_target_depts" in block


@pytest.mark.unit
class TestLocationSwitchRetrievalDoesNotShortCircuit:
    """An empty filtered result resets _query_loc_override and falls through
    — unlike the two Phase-2-fixed sites, this never `return`s early, so a
    wrong/stray value here degrades to the pre-existing (broader) doc set
    for this turn rather than silently committing to nothing."""

    def test_filter_locks_to_the_current_topics_license_type(self):
        block = _location_switch_source()
        assert '{"$and": [{"license_type": _loc_lt_lock}, {"location": _query_loc_override}]}' in block

    def test_empty_result_resets_the_override_flag(self):
        block = _location_switch_source()
        assert '_query_loc_override = ""  # no location-specific docs' in block

    def test_does_not_return_from_handle_on_empty_result(self):
        block = _location_switch_source()
        # The reset-and-continue line must not be followed by a bare `return`
        # statement that would exit handle() early the way the Phase-2-fixed
        # sites used to.
        after_reset = block.split('_query_loc_override = ""  # no location-specific docs', 1)[-1]
        # Only the exception-handler's own reset (a separate, later line) may
        # follow — no early return anywhere in this captured block.
        assert "\n                    return\n" not in after_reset
        assert re.search(r"\n\s+return\b", after_reset) is None

    def test_is_exception_safe_and_also_resets_on_error(self):
        block = _location_switch_source()
        assert "except Exception as _loc_err:" in block
        assert block.count('_query_loc_override = ""') == 2  # empty-result path + exception path

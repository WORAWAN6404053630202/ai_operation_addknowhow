"""
Migration of Group E + the minor residual gap (2026-11, retrieval-safety
audit, Phase 5 — the final phase of this round) onto the shared helpers.

Group E ("operation_group" pending-slot handler, Fallback 1): the existing
3-stage relaxation cascade only retriggered a broader refetch on EMPTY
results, not on a result that disagrees internally on department. Fixed by
running the guard on the primary fetch and widening the Fallback 1 trigger
condition to `(not state.current_docs or _og_would_drop)` — reusing the
EXISTING relax-to-broader-filter cascade as the "supplement" mechanism,
rather than adding a new unfiltered search (unlike every other site in this
round).

Minor gap ("department" pending-slot handler): already safe on the
empty-result path (explicit "keep existing docs" fallback); this adds the
SAME guard+supplement treatment as everywhere else for consistency, lowest
priority per the project plan but included since there was time.
"""
from __future__ import annotations

import inspect
import re

import pytest

from model.persona_supervisor import PersonaSupervisor


def _route_source() -> str:
    return inspect.getsource(PersonaSupervisor._route_pending_slot_to_persona)


def _op_group_block() -> str:
    source = _route_source()
    m = re.search(
        r"_op_meta_filter: dict = \{\"entity_type_normalized\": _saved_entity\}(.*?)op-group retrieval: entity=%r",
        source, re.DOTALL,
    )
    assert m, "op-group retrieval block not found"
    return m.group(1)


def _dept_slot_block() -> str:
    source = _route_source()
    m = re.search(
        r'_dept_filter: dict = \{"department": _dept_val\}(.*?)department retrieval: dept=%r',
        source, re.DOTALL,
    )
    assert m, "department pending-slot retrieval block not found"
    return m.group(1)


@pytest.mark.unit
class TestOpGroupFallbackRetriggersOnMismatchNotJustEmptiness:
    def test_guard_is_wired_after_the_primary_fetch(self):
        block = _op_group_block()
        assert "_identity_guard_majority_vote(" in block
        assert 'identity_fields=["department"]' in block

    def test_fallback_1_condition_widened_to_include_would_drop(self):
        block = _op_group_block()
        assert "(not state.current_docs or _og_would_drop)" in block
        # must not have regressed to the old emptiness-only condition
        assert "if not state.current_docs and _saved_dept_op and _saved_license:" not in block

    def test_no_new_unfiltered_search_introduced(self):
        """Per the plan: this site reuses its OWN existing relax-to-broader-
        filter cascade as the supplement mechanism — unlike every other site
        in this round, it should NOT call _supplement_if_thin or
        similarity_search directly."""
        block = _op_group_block()
        assert "_supplement_if_thin(" not in block

    def test_fallback_2_entity_only_still_present_unchanged(self):
        block = _op_group_block()
        assert 'metadata_filter={"entity_type_normalized": _saved_entity}' in block


@pytest.mark.unit
class TestDepartmentPendingSlotMinorGapClosed:
    def test_guard_checks_license_type(self):
        block = _dept_slot_block()
        assert "_identity_guard_majority_vote(" in block
        assert 'identity_fields=["license_type"]' in block

    def test_supplement_gated_on_single_filter_flag(self):
        block = _dept_slot_block()
        assert "_supplement_if_thin(" in block
        assert 'getattr(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False)' in block

    def test_empty_result_still_falls_through_to_existing_else_branch(self):
        """The pre-existing "else: keep existing docs" safety net (the only
        thing that made this a 'minor gap' rather than a full RISKY site)
        must remain untouched."""
        source = _route_source()
        assert '_dept_filter: dict = {"department": _dept_val}' in source
        # the else branch immediately follows the guarded if-block in source
        idx = source.find('_dept_filter: dict = {"department": _dept_val}')
        tail = source[idx:idx + 3000]
        assert "else:" in tail

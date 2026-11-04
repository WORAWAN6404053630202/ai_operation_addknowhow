"""
Migration of Group C sites (2026-11, retrieval-safety audit, Phase 3) onto
_merge_value_groups_with_guard_and_supplement — the shared helper that
generalizes the multi-license merge-loop shape already fixed twice elsewhere
in this file (_ensure_practical_retrieval_for_legal's `_merged` loop and
_route_pending_slot_to_persona's "topic" slot-queue `_merged_mt2` loop, both
covered by tests/test_license_filter_coverage_cascade.py).

Group C = the "registration_type" and "entity_type" multi_license_topics
re-merge loops in _route_pending_slot_to_persona — structurally identical to
the two already-fixed loops but these two never got the fix applied. Both
reuse MULTI_TOPIC_COVERAGE_SUPPLEMENT_ENABLED (the existing multi-topic
flag), not a new one, since this is the same scenario as the already-fixed
sites, not a new single-fetch case.
"""
from __future__ import annotations

import inspect
import re

import pytest

from model.persona_supervisor import PersonaSupervisor


def _full_source() -> str:
    return inspect.getsource(PersonaSupervisor._route_pending_slot_to_persona)


def _rt_block() -> str:
    source = _full_source()
    m = re.search(
        r"def _fetch_mt_rt\(_lt: str\)(.*?)multi-topic registration_type re-merge: reg=%r",
        source, re.DOTALL,
    )
    assert m, "registration_type re-merge block not found"
    return m.group(1)


def _et_block() -> str:
    source = _full_source()
    m = re.search(
        r"def _fetch_mt_et\(_lt: str\)(.*?)multi-topic entity_type re-merge: entity=%r",
        source, re.DOTALL,
    )
    assert m, "entity_type re-merge block not found"
    return m.group(1)


@pytest.mark.unit
class TestRegistrationTypeReMergeUsesSharedHelper:
    def test_uses_the_shared_merge_helper(self):
        block = _rt_block()
        assert "_merge_value_groups_with_guard_and_supplement(" in block

    def test_guard_checks_department(self):
        block = _rt_block()
        assert 'identity_fields=["department"]' in block

    def test_guard_gated_on_shared_enforce_flag(self):
        block = _rt_block()
        assert 'getattr(conf, "REGULATORY_IDENTITY_GUARD_ENFORCE", False)' in block

    def test_supplement_reuses_multi_topic_flag_not_a_new_one(self):
        block = _rt_block()
        assert 'getattr(conf, "MULTI_TOPIC_COVERAGE_SUPPLEMENT_ENABLED", False)' in block
        assert "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED" not in block

    def test_post_filter_still_applies_entity_then_registration_type(self):
        block = _rt_block()
        assert "entity_type_normalized" in block
        assert "registration_type" in block

    def test_old_inline_loop_is_gone(self):
        """No local hand-written merge loop should remain — the shared helper
        replaces it entirely."""
        block = _rt_block()
        assert "for _lt_rt in _mt_lts:" not in block


@pytest.mark.unit
class TestEntityTypeReMergeUsesSharedHelper:
    def test_uses_the_shared_merge_helper(self):
        block = _et_block()
        assert "_merge_value_groups_with_guard_and_supplement(" in block

    def test_guard_checks_department(self):
        block = _et_block()
        assert 'identity_fields=["department"]' in block

    def test_supplement_reuses_multi_topic_flag_not_a_new_one(self):
        block = _et_block()
        assert 'getattr(conf, "MULTI_TOPIC_COVERAGE_SUPPLEMENT_ENABLED", False)' in block
        assert "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED" not in block

    def test_old_inline_loop_is_gone(self):
        block = _et_block()
        assert "for _lt_et in _mt_lts2:" not in block


@pytest.mark.unit
class TestGroupCDoesNotTouchAlreadyFixedSiblingLoops:
    """The two already-fixed merge loops elsewhere in this file
    (_ensure_practical_retrieval_for_legal, and the "topic" slot-queue
    _merged_mt2 loop in this same method) must remain untouched by this
    migration — only the registration_type/entity_type re-merge loops change."""

    def test_topic_slot_queue_merged_mt2_loop_still_present_and_unmigrated(self):
        """Per the plan, retrofitting the already-fixed sites onto the shared
        helper is explicitly deferred to a future round — this pins that it
        was NOT accidentally done as a side effect of this migration."""
        source = _full_source()
        assert "_merged_mt2: List[Dict] = []" in source
        assert "for _lt_name2 in _multi_from_input:" in source

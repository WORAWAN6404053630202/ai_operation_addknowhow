"""
Migration of Group A sites (2026-11, retrieval-safety audit, Phase 1) onto the
shared helpers added in tests/test_retrieval_safety_shared_helpers.py.

Group A = single department-filtered fetch in _ensure_practical_retrieval_for_legal
that used to commit (replace the broader doc set) on bare non-emptiness with no
identity/completeness check: the op_topic department-scope guard and the OBD
department-scope guard. Both filter on `department`, so the guard checks
`license_type` instead (the orthogonal field). Existing control flow (the
unconditional `return` later in the same try block) is preserved unchanged —
this migration only changes what value _ot_docs_e/_obd_docs_e gets assigned.
"""
from __future__ import annotations

import inspect
import re

import pytest

from model.persona_supervisor import PersonaSupervisor


def _full_source() -> str:
    return inspect.getsource(PersonaSupervisor._ensure_practical_retrieval_for_legal)


def _op_topic_dept_scope_source() -> str:
    source = _full_source()
    m = re.search(
        r"_ot_dept_res = _coll_e\.get\((.*?)_ot_docs_e = _ot_dept_docs",
        source, re.DOTALL,
    )
    assert m, "op_topic department-scope block not found"
    return m.group(1)


def _obd_dept_scope_source() -> str:
    source = _full_source()
    m = re.search(
        r"_obd_dept_res = _coll_obd_e\.get\((.*?)_obd_docs_e = _obd_dept_docs",
        source, re.DOTALL,
    )
    assert m, "OBD department-scope block not found"
    return m.group(1)


@pytest.mark.unit
class TestOpTopicDeptScopeSafety:
    def test_guard_is_wired_checking_license_type(self):
        block = _op_topic_dept_scope_source()
        assert "_identity_guard_majority_vote(" in block
        assert 'identity_fields=["license_type"]' in block

    def test_guard_gated_on_shared_enforce_flag(self):
        block = _op_topic_dept_scope_source()
        assert 'getattr(conf, "REGULATORY_IDENTITY_GUARD_ENFORCE", False)' in block

    def test_supplement_gated_on_new_single_filter_flag(self):
        block = _op_topic_dept_scope_source()
        assert "_supplement_if_thin(" in block
        assert 'getattr(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False)' in block

    def test_supplement_handles_missing_vectorstore_gracefully(self):
        block = _op_topic_dept_scope_source()
        assert "if _ot_vs_e is not None else []" in block


@pytest.mark.unit
class TestObdDeptScopeSafety:
    def test_guard_is_wired_checking_license_type(self):
        block = _obd_dept_scope_source()
        assert "_identity_guard_majority_vote(" in block
        assert 'identity_fields=["license_type"]' in block

    def test_guard_gated_on_shared_enforce_flag(self):
        block = _obd_dept_scope_source()
        assert 'getattr(conf, "REGULATORY_IDENTITY_GUARD_ENFORCE", False)' in block

    def test_supplement_gated_on_new_single_filter_flag(self):
        block = _obd_dept_scope_source()
        assert "_supplement_if_thin(" in block
        assert 'getattr(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False)' in block

    def test_supplement_handles_missing_vectorstore_gracefully(self):
        block = _obd_dept_scope_source()
        assert "if _obd_vs_e is not None else []" in block


@pytest.mark.unit
class TestGroupAPreservesExistingControlFlow:
    """Migration must not add/remove the early `return` these sites already
    had (or didn't have) — only the value assigned to _ot_docs_e/_obd_docs_e
    changes, not when the method exits."""

    def test_op_topic_block_still_returns_unconditionally_after_commit(self):
        source = _full_source()
        segment = source.split("_ot_docs_e = _ot_dept_docs", 1)[-1][:3000]
        assert re.search(r"\n\s+return\n", segment)

    def test_obd_block_does_not_introduce_a_new_early_return(self):
        """The OBD block's own commit (_obd_docs_e = _obd_dept_docs) is followed
        by further processing (entity-variety slot injection, etc.) before any
        return in the original code — migration must not short-circuit that."""
        source = _full_source()
        segment = source.split("_obd_docs_e = _obd_dept_docs", 1)[-1][:1000]
        assert "topic_slot_queue" in segment  # the next real step, proving flow continues

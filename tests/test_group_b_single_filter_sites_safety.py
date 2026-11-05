"""
Migration of Group B sites (2026-11, retrieval-safety audit, Phase 2) onto the
shared helpers added in tests/test_retrieval_safety_shared_helpers.py.

Group B = single department/license_type-filtered fetches in
_maybe_build_slot_queue_from_docs that used to commit to state.current_docs on
bare non-emptiness with no identity/completeness check, and (unlike Group A)
had NO literal early `return` immediately tied to the check — but also no
downstream correction for that field, making the risk functionally identical.
Existing control flow (return-or-fall-through) is preserved unchanged per
site; only the value assigned to state.current_docs changes.

7 sites: dept-switch re-retrieve, info-query op-inferred re-retrieve,
multi-license mismatch recovery, legal_q Step6 op-inferred re-retrieve,
Step7 op_topic switch, Step3b op_topic cascade, Step3b op-inferred re-retrieve.

Follow-up fix (2026-11, live-testing): live testing found that guard+
supplement had been nested INSIDE the same `if <raw_fetch_result>:` check
that gates the final commit — meaning a completely-empty (zero-result)
exact-match filter never got a chance at the coverage-adequacy supplement,
reproducing the original silent-failure bug for that one sub-case. All 7
sites now run guard+supplement unconditionally on the raw fetch result, and
gate ONLY the final commit (and, where present, the `return`) on the
post-supplement result — see each class's `test_guard_not_gated_on_raw_result`.
"""
from __future__ import annotations

import inspect
import re

import pytest

from model.persona_supervisor import PersonaSupervisor


def _full_source() -> str:
    return inspect.getsource(PersonaSupervisor._maybe_build_slot_queue_from_docs)


def _route_source() -> str:
    return inspect.getsource(PersonaSupervisor._route_pending_slot_to_persona)


def _block(start: str, end: str, source: str = None) -> str:
    source = source if source is not None else _full_source()
    m = re.search(re.escape(start) + r"(.*?)" + re.escape(end), source, re.DOTALL)
    assert m, f"block not found: {start!r} .. {end!r}"
    return m.group(1)


@pytest.mark.unit
class TestDeptSwitchReRetrieveSafety:
    def _blk(self):
        return _block('metadata_filter={"department": _query_dept_any},', "return  # proceed to practical.handle with updated docs")

    def test_guard_checks_license_type(self):
        block = self._blk()
        assert "_identity_guard_majority_vote(" in block
        assert 'identity_fields=["license_type"]' in block

    def test_supplement_gated_on_single_filter_flag(self):
        block = self._blk()
        assert "_supplement_if_thin(" in block
        assert 'getattr(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False)' in block

    def test_still_returns_after_commit(self):
        block = self._blk()
        assert "return  # proceed" not in block  # it's the end anchor, not inside
        assert "state.current_docs = _dept_docs" in block

    def test_guard_not_gated_on_raw_result(self):
        block = self._blk()
        guard_idx = block.index("_identity_guard_majority_vote(")
        if_idx = block.index("if _dept_docs:")
        assert guard_idx < if_idx


@pytest.mark.unit
class TestInfoQueryOpInferredReRetrieveSafety:
    def _blk(self):
        return _block(
            'metadata_filter={"license_type": _lt_infoq},',
            "state.last_retrieval_query = _enrich_iq",
        )

    def test_guard_checks_department(self):
        block = self._blk()
        assert "_identity_guard_majority_vote(" in block
        assert 'identity_fields=["department"]' in block

    def test_supplement_wired(self):
        block = self._blk()
        assert "_supplement_if_thin(" in block

    def test_guard_not_gated_on_raw_result(self):
        block = self._blk()
        guard_idx = block.index("_identity_guard_majority_vote(")
        if_idx = block.index("if _op_docs_iq:")
        assert guard_idx < if_idx


@pytest.mark.unit
class TestMultiLicenseMismatchRecoverySafety:
    def _blk(self):
        return _block(
            "_recovery_docs = self._practical._retrieve_docs(_last_q, metadata_filter=_ml_filter)",
            "return  # don't return False — let practical answer with recovered docs",
        )

    def test_guard_checks_license_type(self):
        block = self._blk()
        assert "_identity_guard_majority_vote(" in block
        assert 'identity_fields=["license_type"]' in block

    def test_supplement_wired(self):
        block = self._blk()
        assert "_supplement_if_thin(" in block

    def test_still_returns_after_commit(self):
        block = self._blk()
        assert "state.current_docs = _recovery_docs" in block

    def test_guard_not_gated_on_raw_result(self):
        block = self._blk()
        guard_idx = block.index("_identity_guard_majority_vote(")
        if_idx = block.index("if _recovery_docs:")
        assert guard_idx < if_idx


@pytest.mark.unit
class TestStep6OpInferredReRetrieveSafety:
    def _blk(self):
        return _block(
            'metadata_filter={"license_type": license_type} if license_type else None,',
            "legal_q Step6 op-inferred re-retrieve: op=%r docs=%d",
        )

    def test_guard_checks_department(self):
        block = self._blk()
        assert "_identity_guard_majority_vote(" in block
        assert 'identity_fields=["department"]' in block

    def test_supplement_wired(self):
        block = self._blk()
        assert "_supplement_if_thin(" in block

    def test_guard_not_gated_on_raw_result(self):
        block = self._blk()
        guard_idx = block.index("_identity_guard_majority_vote(")
        if_idx = block.index("if _op_docs_6:")
        assert guard_idx < if_idx


@pytest.mark.unit
class TestStep7OpTopicSwitchSafety:
    """Guard+supplement run unconditionally after the 2-stage fetch (2026-11,
    live-testing follow-up) — not nested inside `if _ot_docs_7:` — so a fully
    empty result from BOTH stages still gets a real supplement attempt."""

    def _blk(self):
        return _block(
            "_ot_docs_7 = self._practical._retrieve_docs(",
            "Step7 op_topic switch: %r → %r docs=%d",
        )

    def test_guard_checks_department(self):
        block = self._blk()
        assert "_identity_guard_majority_vote(" in block
        assert 'identity_fields=["department"]' in block

    def test_supplement_wired(self):
        block = self._blk()
        assert "_supplement_if_thin(" in block

    def test_guard_and_supplement_not_gated_on_raw_result(self):
        block = self._blk()
        guard_idx = block.index("_identity_guard_majority_vote(")
        if_idx = block.index("if _ot_docs_7:")
        assert guard_idx < if_idx


@pytest.mark.unit
class TestStep3bOpTopicCascadeSafety:
    """Lives in _route_pending_slot_to_persona, not _maybe_build_slot_queue_from_docs.
    Guard+supplement run unconditionally (2026-11, live-testing follow-up),
    not nested inside `if _ot_docs_3b:`."""

    def _blk(self):
        return _block(
            "_ot_docs_3b = self._practical._retrieve_docs(",
            'state.last_retrieval_query = f"{_full_ui} {_matched_ot_3b}".strip()',
            source=_route_source(),
        )

    def test_guard_checks_department(self):
        block = self._blk()
        assert "_identity_guard_majority_vote(" in block
        assert 'identity_fields=["department"]' in block

    def test_supplement_wired(self):
        block = self._blk()
        assert "_supplement_if_thin(" in block

    def test_guard_not_gated_on_raw_result(self):
        block = self._blk()
        guard_idx = block.index("_identity_guard_majority_vote(")
        if_idx = block.index("if _ot_docs_3b:")
        assert guard_idx < if_idx


@pytest.mark.unit
class TestStep3bOpInferredReRetrieveSafety:
    """Lives in _route_pending_slot_to_persona, not _maybe_build_slot_queue_from_docs.
    Guard+supplement run unconditionally (2026-11, live-testing follow-up),
    not nested inside `if _op_docs_3b:`."""

    def _blk(self):
        return _block(
            "_op_docs_3b = self._practical._retrieve_docs(",
            "Step3b op-inferred re-retrieve: op=%r docs=%d",
            source=_route_source(),
        )

    def test_guard_checks_department(self):
        block = self._blk()
        assert "_identity_guard_majority_vote(" in block
        assert 'identity_fields=["department"]' in block

    def test_supplement_wired(self):
        block = self._blk()
        assert "_supplement_if_thin(" in block

    def test_guard_not_gated_on_raw_result(self):
        block = self._blk()
        guard_idx = block.index("_identity_guard_majority_vote(")
        if_idx = block.index("if _op_docs_3b:")
        assert guard_idx < if_idx


@pytest.mark.unit
class TestGroupBAllSevenSitesWired:
    """Sanity count: 5 new _identity_guard_majority_vote call sites in
    _maybe_build_slot_queue_from_docs (dept-switch, info-query, multi-license
    recovery, Step6, Step7) + 2 in _route_pending_slot_to_persona (Step3b x2)
    = 7 total for Group B. _route_pending_slot_to_persona gets a 3rd guard
    call later (Group D, full-license pre-retrieve, Phase 4) — see
    test_group_d_full_license_preretrieve_safety.py — so this count is
    intentionally >= 2, not == 2, to not go stale as later phases land."""

    def test_five_guard_calls_in_maybe_build_slot_queue(self):
        source = _full_source()
        assert source.count("_identity_guard_majority_vote(") == 5

    def test_at_least_two_guard_calls_in_route_pending_slot(self):
        source = _route_source()
        assert source.count("_identity_guard_majority_vote(") >= 2

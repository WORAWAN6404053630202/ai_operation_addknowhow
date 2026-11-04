"""
Migration of Group D (2026-11, retrieval-safety audit, Phase 4) onto the
shared helpers: the "full-license pre-retrieve" site in
_route_pending_slot_to_persona (Step 2), which fetches ALL docs for one
license_type via plain .get() and used to commit to state.current_docs on
bare non-emptiness with no identity/completeness check.

Special care required (per the project plan): this site ALSO snapshots
state.context["_presaved_license_docs"] for academic-link-collection to
consume later, with an explicitly documented contract of "ALL docs for this
license" — the guard/supplement must apply only to what feeds
state.current_docs, NOT to the snapshot, or academic link collection would
silently see fewer docs than its own contract promises.
"""
from __future__ import annotations

import inspect
import re

import pytest

from model.persona_supervisor import PersonaSupervisor


def _block() -> str:
    source = inspect.getsource(PersonaSupervisor._route_pending_slot_to_persona)
    m = re.search(
        r"if _all_docs:(.*?)full-license pre-retrieved %d docs for license=%r",
        source, re.DOTALL,
    )
    assert m, "full-license pre-retrieve block not found"
    return m.group(1)


@pytest.mark.unit
class TestFullLicensePreRetrieveSafety:
    def test_guard_checks_department(self):
        block = _block()
        assert "_identity_guard_majority_vote(" in block
        assert 'identity_fields=["department"]' in block

    def test_supplement_gated_on_single_filter_flag(self):
        block = _block()
        assert "_supplement_if_thin(" in block
        assert 'getattr(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False)' in block

    def test_presaved_snapshot_is_taken_from_all_docs_before_the_guard_runs(self):
        """The snapshot line must appear BEFORE the guard call in source order
        — proving it captures the pre-guard, unfiltered _all_docs, not the
        narrowed _guarded_docs."""
        block = _block()
        snapshot_pos = block.find('state.context["_presaved_license_docs"] = list(_all_docs)')
        guard_pos = block.find("_identity_guard_majority_vote(")
        assert snapshot_pos != -1 and guard_pos != -1
        assert snapshot_pos < guard_pos

    def test_presaved_snapshot_uses_all_docs_not_guarded_docs(self):
        block = _block()
        assert 'list(_all_docs)' in block
        assert 'list(_guarded_docs)' not in block

    def test_state_current_docs_is_assigned_from_guarded_docs(self):
        block = _block()
        assert "state.current_docs = _guarded_docs" in block
        assert "state.current_docs = _all_docs" not in block

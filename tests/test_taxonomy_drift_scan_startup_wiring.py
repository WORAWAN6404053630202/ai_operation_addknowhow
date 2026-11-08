"""
Regression test for the 2026-11 hardcoded-logic audit's Phase 3: persona_
supervisor.py's _validate_license_types_at_startup now also calls scripts.
taxonomy_drift_scan.check_drift(), generalizing its license_type-only checks
to the other hand-maintained structures across all 3 persona files. Source-
inspection style, matching this file's existing precedent for logic embedded
in this same large method (see test_hardcoded_logic_audit_tier3_4.py's
test_startup_validator_rechecks_permit_lt_re).
"""
from __future__ import annotations

import inspect

import pytest

from model.persona_supervisor import PersonaSupervisor


@pytest.mark.unit
class TestStartupValidatorCallsTaxonomyDriftScan:
    def test_startup_validator_imports_and_calls_check_drift(self):
        source = inspect.getsource(PersonaSupervisor._validate_license_types_at_startup)
        assert "from scripts.taxonomy_drift_scan import check_drift" in source
        assert "check_drift(vectorstore)" in source

    def test_startup_validator_distinguishes_high_from_info_severity(self):
        source = inspect.getsource(PersonaSupervisor._validate_license_types_at_startup)
        assert '"high"' in source

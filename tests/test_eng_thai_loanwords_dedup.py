"""
Regression test for the 2026-11 dedup of persona_supervisor.py's English-
loanword Thai transliteration dict. {"PLUS": "พลัส", "SUPER": "ซุปเปอร์",
"PRO": "โปร", "MINI": "มินิ"} used to be hand-duplicated verbatim as local
variables (_ENG_THAI_7, _ENG_THAI_3B, _ENG_THAI_ELAB) at 3 separate call
sites — same drift risk class as _BANK_ID_PATTERNS, which this mirrors.
"""
from __future__ import annotations

import inspect

import pytest

from model.persona_supervisor import PersonaSupervisor


@pytest.mark.unit
class TestEngThaiLoanwordsDedup:
    def test_class_attribute_exists_with_expected_values(self):
        assert PersonaSupervisor._ENG_THAI_LOANWORDS == {
            "PLUS": "พลัส", "SUPER": "ซุปเปอร์", "PRO": "โปร", "MINI": "มินิ",
        }

    def test_no_stray_local_definitions_remain(self):
        source = inspect.getsource(PersonaSupervisor)
        for stale_name in ("_ENG_THAI_7", "_ENG_THAI_3B", "_ENG_THAI_ELAB"):
            assert stale_name not in source, f"stale local {stale_name} still present"

    def test_source_references_shared_attribute_at_least_three_times(self):
        source = inspect.getsource(PersonaSupervisor)
        assert source.count("self._ENG_THAI_LOANWORDS") >= 3

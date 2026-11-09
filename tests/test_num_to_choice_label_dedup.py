"""
Regression test for the 2026-11 dedup of persona_academic.py's numeric-token
-> choice-label map. _CHOICE_LABELS ("ก","ข","ค","ง") is the canonical source;
_num_to_label/_num_to_label_fb used to separately hardcode {"1": "ก", ...} at
2 call sites, which would silently desync if _CHOICE_LABELS ever grew.
"""
from __future__ import annotations

import inspect

import pytest

from model.persona_academic import AcademicPersonaService


@pytest.mark.unit
class TestNumToChoiceLabelDedup:
    def test_derived_correctly_from_choice_labels(self):
        expected = {str(i + 1): label for i, label in enumerate(AcademicPersonaService._CHOICE_LABELS)}
        assert AcademicPersonaService._NUM_TO_CHOICE_LABEL == expected
        assert AcademicPersonaService._NUM_TO_CHOICE_LABEL == {"1": "ก", "2": "ข", "3": "ค", "4": "ง"}

    def test_no_stray_local_definitions_remain(self):
        source = inspect.getsource(AcademicPersonaService)
        for stale_name in ("_num_to_label =", "_num_to_label_fb ="):
            assert stale_name not in source, f"stale local {stale_name!r} still present"

    def test_source_references_shared_attribute_at_both_sites(self):
        source = inspect.getsource(AcademicPersonaService)
        assert source.count("self._NUM_TO_CHOICE_LABEL") >= 2

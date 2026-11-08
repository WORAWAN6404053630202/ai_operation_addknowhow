"""
Regression test for the 2026-11 hardcoded-logic audit's dynamic-taxonomy
follow-up. persona_supervisor.py's _license_type_llm_fallback used to build
its LLM candidate list from the hardcoded _MULTI_TOPIC_LICENSE_KEYWORDS
list — the exact failure class this whole investigation found (a sibling
hardcoded list had 8/9 entries already drifted from the real corpus by the
time anyone checked). It now asks service.live_taxonomy.get_live_license_
types every time, falling back to the hardcoded list only if the live fetch
is genuinely empty (e.g. vectorstore not yet connected).
"""
from __future__ import annotations

import inspect

import pytest

from model.persona_supervisor import PersonaSupervisor


@pytest.mark.unit
class TestLicenseTypeLlmFallbackUsesLiveCandidates:
    def test_candidates_come_from_live_taxonomy_not_the_hardcoded_list_alone(self):
        source = inspect.getsource(PersonaSupervisor._license_type_llm_fallback)
        assert "get_live_license_types" in source
        assert "from service.live_taxonomy import get_live_license_types" in source

    def test_hardcoded_list_only_used_as_a_fallback_when_live_fetch_is_empty(self):
        source = inspect.getsource(PersonaSupervisor._license_type_llm_fallback)
        # The hardcoded list must appear only as an `or` fallback, never as
        # the primary/sole candidate source.
        assert "get_live_license_types(_vstore_lt) or [lt for _, lt in self._MULTI_TOPIC_LICENSE_KEYWORDS]" in source

"""persona_supervisor.py's _sv_search (nested inside the ~1750-line
_ensure_practical_retrieval_for_legal, no isolated seam for a full
behavioral drive — same constraint test_group_e_mismatch_triggered_fallback_
safety.py already works around) is one of 3 independently-duplicated sites
that used to silently fall back to an unfiltered vector search whenever a
filtered one raised any exception. See utils/text_patterns.py's
unfiltered_fallback_is_safe docstring and
tests/test_fail_closed_on_filtered_retrieval_exception.py (persona_practical.
py/persona_academic.py's behavioral equivalents of this test) for the full
incident.

Source-inspection regression pin, same established pattern as
test_group_e_mismatch_triggered_fallback_safety.py: extract _sv_search's
body via regex and assert every fallback-to-unfiltered line is guarded by
the shared helper.
"""
from __future__ import annotations

import inspect
import re

import pytest

from model.persona_supervisor import PersonaSupervisor


def _sv_search_block() -> str:
    source = inspect.getsource(PersonaSupervisor._ensure_practical_retrieval_for_legal)
    m = re.search(
        r"def _sv_search\(q: str, k: int, metadata_filter.*?\n(.*?)\n        if _is_broad_q:",
        source, re.DOTALL,
    )
    assert m, "_sv_search function body not found in _ensure_practical_retrieval_for_legal"
    return m.group(1)


@pytest.mark.unit
class TestSvSearchFailsClosedOnFilteredException:
    def test_helper_imported(self):
        import model.persona_supervisor as sup_mod
        source = inspect.getsource(sup_mod)
        assert "unfiltered_fallback_is_safe as _shared_unfiltered_fallback_is_safe" in source

    def test_every_fallback_to_unfiltered_is_guarded(self):
        block = _sv_search_block()
        # 3 fallback points: after the mid-tier filtered similarity_search
        # except, after the final unfiltered-scored-search except, and the
        # last-resort self.retriever.invoke(q) at the end of the function.
        guard_count = block.count("_shared_unfiltered_fallback_is_safe(metadata_filter)")
        assert guard_count >= 3, f"expected >=3 unfiltered_fallback_is_safe guards, found {guard_count}"

    def test_final_retriever_invoke_is_guarded_not_unconditional(self):
        block = _sv_search_block()
        final_return_idx = block.rfind("return list(self.retriever.invoke(q) or [])")
        assert final_return_idx != -1
        preceding = block[:final_return_idx]
        guard_idx = preceding.rfind("_shared_unfiltered_fallback_is_safe(metadata_filter)")
        assert guard_idx != -1 and guard_idx < final_return_idx, (
            "the final unfiltered self.retriever.invoke(q) fallback must be "
            "preceded by an unfiltered_fallback_is_safe guard, not reached unconditionally"
        )

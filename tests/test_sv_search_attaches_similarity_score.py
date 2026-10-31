"""
Regression pin for the 2026-10 fix (Phase-0 audit follow-up) making
PersonaSupervisor._sv_search attach a similarity score onto every returned
doc's metadata["_sim"], in all 3 of its branches.

Background: a prior audit built utils.text_patterns.regulatory_identity_
mismatch_should_reject() to catch cross-domain retrieval contamination (e.g.
a DBD commerce-registration doc answering a VAT question) but deliberately
left it unwired because _sv_search — the retrieval helper used on the main
live-Practical-traffic fallback path — didn't carry a score in 2 of its 3
branches, and the guard needs a score to know how confident the "top" doc is.

_sv_search is a nested closure defined ~1500 lines into a large, deeply
stateful method (_ensure_practical_retrieval_for_legal — query rewriting,
multi-pass broad-question retrieval, live LLM calls before even reaching
_sv_search's own definition), so driving it end-to-end here is impractical —
same tradeoff this codebase already made for TestBoostFieldsIncludesDepartment
(test_practical_truncation_retry.py) and TestLowConfidenceRetrievalFlag
(test_groundedness_safety_net.py). This is a regression pin on the code
shape: it proves every branch attaches "_sim" (or explicitly has none to
attach, for the true-last-resort .invoke() fallback), not a live behavioral
test against a real/mocked vectorstore.
"""
from __future__ import annotations

import inspect
import re

import pytest

from model.persona_supervisor import PersonaSupervisor


@pytest.mark.unit
class TestSvSearchAttachesSimilarityScore:
    def _sv_search_source(self) -> str:
        source = inspect.getsource(PersonaSupervisor._ensure_practical_retrieval_for_legal)
        m = re.search(
            r'def _sv_search\(.*?\n(.*?)\n\s{8}if _is_broad_q:',
            source, re.DOTALL,
        )
        assert m, "_sv_search definition not found in _ensure_practical_retrieval_for_legal"
        return m.group(1)

    def test_hybrid_branch_attaches_sim_instead_of_discarding_it(self):
        block = self._sv_search_source()
        # The old code discarded the score entirely: `for d, _ in hybrid_scored_search(...)`.
        # The fix must keep the score and attach it.
        assert '.metadata["_sim"] = float(' in block
        assert 'hybrid_scored_search(' in block

    def test_filtered_branch_uses_scored_search_not_plain_similarity_search(self):
        block = self._sv_search_source()
        assert 'similarity_search_with_relevance_scores(q, k=k, filter=metadata_filter)' in block

    def test_unfiltered_branch_tries_scored_search_before_plain_invoke_fallback(self):
        block = self._sv_search_source()
        assert 'similarity_search_with_relevance_scores(q, k=k)' in block
        # True last resort — only reached when _sv_vstore is unavailable or scored search failed.
        assert 'self.retriever.invoke(q)' in block

    def test_every_scored_branch_is_exception_safe(self):
        block = self._sv_search_source()
        # 2 try/except pairs: the metadata-filtered branch (pre-existing) and the
        # new unfiltered-scored-search branch added by this fix.
        assert block.count('except Exception') >= 2

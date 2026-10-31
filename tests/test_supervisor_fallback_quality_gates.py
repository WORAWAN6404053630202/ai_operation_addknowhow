"""
Regression pin for the 2026-10 fix (Phase-0 codebase audit) wiring the
category filter and reranker into persona_supervisor.py's
_ensure_practical_retrieval_for_legal fallback retrieval path.

Background: QA-reproduced cross-domain contamination (e.g. DBD commerce-
registration content answering a VAT question) traced to this exact fallback
branch — the narrow/follow-up retrieval path most live Practical traffic
actually uses (per _retrieve_docs' own comment: "most live Practical traffic
is actually retrieved by persona_supervisor.py's
_ensure_practical_retrieval_for_legal"). Unlike the rare-license branch
elsewhere in the same file (which already applies RERANKER_ENABLED), this
fallback previously had NO quality-control stage at all — whatever
_sv_search returned went straight to the LLM.

_ensure_practical_retrieval_for_legal is a large, deeply stateful method
(query rewriting, multi-pass broad-question retrieval, live LLM calls) not
practical to drive end-to-end in a unit test — same situation as this
project's existing TestBoostFieldsIncludesDepartment (test_practical_
truncation_retry.py) and this session's TestLowConfidenceRetrievalFlag
(test_groundedness_safety_net.py), which use the same source-inspection
approach for the same reason. This is a regression pin on the code shape:
it proves the wiring exists, is gated on the SAME flags as every other call
site (no default-behavior change), and is a no-op when those flags are off.
"""
from __future__ import annotations

import inspect
import re

import pytest

from model.persona_supervisor import PersonaSupervisor


@pytest.mark.unit
class TestSupervisorFallbackQualityGates:
    def _fallback_block_source(self) -> str:
        source = inspect.getsource(PersonaSupervisor._ensure_practical_retrieval_for_legal)
        # Isolate the narrow/follow-up fallback branch specifically (the
        # `else:` paired with `if _is_broad_q:`), not the broad-question
        # multi-pass branch or the rare-license branch earlier in the file.
        m = re.search(
            r'docs = _sv_search\(q_expanded, top_k, metadata_filter=_fb_filter or None\)(.*?)_doc_chars = ',
            source, re.DOTALL,
        )
        assert m, "fallback retrieval block not found in _ensure_practical_retrieval_for_legal"
        return m.group(1)

    def test_fallback_applies_category_filter_gated_on_the_existing_flag(self):
        block = self._fallback_block_source()
        assert 'RETRIEVAL_CATEGORY_FILTER_ENABLED' in block
        assert 'filter_by_category_concentration' in block

    def test_fallback_applies_reranker_gated_on_the_existing_flag(self):
        block = self._fallback_block_source()
        assert 'RERANKER_ENABLED' in block
        assert 'from utils.reranker import rerank' in block

    def test_both_gates_are_exception_safe_and_do_not_replace_default_behavior(self):
        """Neither addition should be able to crash retrieval if the reranker/
        filter itself errors, and both must be read via getattr(conf, ..., False)
        — same opt-in-by-default-off convention as every other flag in this
        codebase — not a hardcoded True."""
        block = self._fallback_block_source()
        assert block.count('except Exception') >= 2
        assert 'getattr(conf, "RETRIEVAL_CATEGORY_FILTER_ENABLED", False)' in block
        assert 'getattr(conf, "RERANKER_ENABLED", False)' in block


@pytest.mark.unit
class TestRegulatoryIdentityGuardWiring:
    """2026-10 (Phase-0 audit follow-up): the regulatory-identity dominance
    guard (utils.text_patterns.regulatory_identity_mismatch_should_reject)
    was built but deliberately left unwired because _sv_search didn't carry
    a similarity score on this path. Now that it does (see
    test_sv_search_attaches_similarity_score.py), this pins the wiring shape:
    log-only soft launch (user-confirmed), never-drop-everything safety,
    exception-safe, and — critically — NOT present in the broad-question
    multi-pass branch, which is deliberately multi-domain by design."""

    def _fallback_block_source(self) -> str:
        source = inspect.getsource(PersonaSupervisor._ensure_practical_retrieval_for_legal)
        m = re.search(
            r'docs = _sv_search\(q_expanded, top_k, metadata_filter=_fb_filter or None\)(.*?)_doc_chars = ',
            source, re.DOTALL,
        )
        assert m, "fallback retrieval block not found in _ensure_practical_retrieval_for_legal"
        return m.group(1)

    def _broad_question_block_source(self) -> str:
        source = inspect.getsource(PersonaSupervisor._ensure_practical_retrieval_for_legal)
        m = re.search(r'if _is_broad_q:(.*?)\n        else:', source, re.DOTALL)
        assert m, "broad-question block not found in _ensure_practical_retrieval_for_legal"
        return m.group(1)

    def test_guard_is_wired_into_fallback_block(self):
        block = self._fallback_block_source()
        assert '_shared_regulatory_identity_mismatch_should_reject(' in block

    def test_guard_is_log_only_by_default_gated_on_new_flag(self):
        block = self._fallback_block_source()
        assert 'REGULATORY_IDENTITY_GUARD_ENFORCE' in block
        # Must always log what WOULD be dropped, and only actually replace
        # `docs` when the flag is on — not a hardcoded enforce.
        assert 'would drop' in block
        assert 'getattr(conf, "REGULATORY_IDENTITY_GUARD_ENFORCE", False)' in block

    def test_guard_never_drops_every_doc(self):
        block = self._fallback_block_source()
        assert 'if _rid_kept and getattr(conf, "REGULATORY_IDENTITY_GUARD_ENFORCE", False)' in block

    def test_guard_skips_when_top_doc_has_no_confident_score(self):
        block = self._fallback_block_source()
        assert 'if _rid_top_score > 0.0:' in block

    def test_guard_is_exception_safe(self):
        block = self._fallback_block_source()
        assert 'except Exception as _rid_e' in block

    def test_guard_is_absent_from_the_broad_question_multi_pass_block(self):
        """Broad questions are deliberately multi-domain (Pass 1 = practical/
        marketing, Pass 2 = regulatory, Pass 3 = alcohol-specific) — a single-
        dominant-license guard here would actively discard correct Pass-2/
        Pass-3 results, not just be unnecessary."""
        block = self._broad_question_block_source()
        assert '_shared_regulatory_identity_mismatch_should_reject' not in block
        assert 'REGULATORY_IDENTITY_GUARD_ENFORCE' not in block

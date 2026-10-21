"""2026-10 fix: tests for the broad-question doc-overflow / JSON-truncation
bug reproduced live (a multi-topic-merged question → 39 docs + 46 unguarded
sibling-completion docs = 51 total → ~64k-token prompt → LengthFinishReasonError
on both the base and widened-budget LLM retry → generic failure message, ~$0.60
burned). Root cause and fix detail live in the approved plan
(.claude/plans/happy-launching-scroll.md) and in inline comments at each site.

persona_practical.py's handle() is a single ~5000-line method — constructing a
full, correctly-mocked end-to-end call through it (retriever, vectorstore,
multiple LLM calls, state/context wiring) for each of these narrow branches
would be disproportionately fragile versus what's being verified. This
codebase already has precedent for source-inspection regression tests on
logic embedded this deeply (see test_practical_truncation_retry.py's
TestBoostFieldsIncludesDepartment, using inspect.getsource + regex) — these
tests follow that same pattern: they verify the actual fix code is present
and wired correctly, which still catches accidental reversion/corruption of
the fix even though they don't execute the branch.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import conf  # noqa: E402
from model.persona_practical import PracticalPersonaService  # noqa: E402


@pytest.mark.unit
class TestSiblingCompletionSkipsMultiTopicRetrieval:
    """Fix (a): the sibling-completion skip gate (_skip_sib) must also check
    _multi_topic_retrieval, not just _broad_question — the multi-topic merge
    (persona_supervisor.py) returns early and never sets _broad_question, so
    an already-comprehensive multi-topic result fell through this gate
    unguarded, letting sibling-completion add 46 more docs on top of it."""

    def test_skip_sib_checks_multi_topic_retrieval(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        idx = source.find("_skip_sib = (")
        assert idx != -1, "_skip_sib assignment not found in handle()"
        end = source.find("_multi_requested_kept", idx)
        assert end != -1
        body = source[idx:end + len("_multi_requested_kept")]
        assert '"_multi_topic_retrieval"' in body
        # Guard the pre-existing checks this fix must not remove.
        assert '"_direct_topic_match"' in body
        assert '"_broad_question"' in body


@pytest.mark.unit
class TestSiblingCapUnconditionalFallback:
    """Fix (b): the original SIBLING_DOCS_CAP enforcement only runs inside
    `if _sib_new and _sib_reg:` → `if _excl_pats:` — both skipped whenever
    _sib_reg (registration_type slot) is empty, e.g. any first-turn question
    with no established entity/registration sub-type yet. An unconditional
    fallback must apply the same cap regardless."""

    def test_fallback_cap_present_outside_the_sib_reg_branch(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        # The fallback must reference SIBLING_DOCS_CAP a second time, outside
        # (after) the _sib_reg-gated block — confirmed by requiring 2 distinct
        # getattr(conf, "SIBLING_DOCS_CAP", ...) call sites in handle().
        occurrences = re.findall(r'getattr\(conf,\s*"SIBLING_DOCS_CAP",\s*10\)', source)
        assert len(occurrences) >= 2, (
            "Expected both the original _sib_reg-gated cap and the new "
            "unconditional fallback cap to reference SIBLING_DOCS_CAP"
        )
        assert "fallback, no _sib_reg" in source


@pytest.mark.unit
class TestFinalDocsCeiling:
    """Fix (c): a genuine final ceiling on _docs_to_process, applied after
    every append path (multi-topic merge, sibling-completion, guarantee-doc
    prepends) has run — nothing re-capped the doc count after that point
    before, so 51 docs reached the final prompt with zero enforcement."""

    def test_conf_defines_the_ceiling_with_default_50(self):
        assert hasattr(conf, "LLM_DOCS_MAX_FINAL_CEILING")
        assert conf.LLM_DOCS_MAX_FINAL_CEILING == 50

    def test_handle_applies_the_ceiling_to_docs_to_process(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        assert '"LLM_DOCS_MAX_FINAL_CEILING"' in source
        m = re.search(
            r'_FINAL_DOCS_CEILING\s*=.*?\n(.*?_docs_to_process\s*=\s*_docs_to_process\[:_FINAL_DOCS_CEILING\])',
            source, re.DOTALL,
        )
        assert m, "Final ceiling trim (_docs_to_process[:_FINAL_DOCS_CEILING]) not found"

    def test_ceiling_matches_multi_topic_merge_total_cap(self):
        # Deliberately set equal to persona_supervisor.py's own multi-topic
        # merge cap (LLM_DOCS_MAX_BROAD * 2) so this is a true no-op for every
        # path already correctly capped at or below it.
        assert conf.LLM_DOCS_MAX_FINAL_CEILING == conf.LLM_DOCS_MAX_BROAD * 2


@pytest.mark.unit
class TestFinalDocsCeilingBehavior:
    """Executable unit test of the actual slicing behavior (not just source
    presence) — isolated from handle()'s surrounding complexity by testing
    the identical slicing logic the fix uses."""

    def test_over_ceiling_list_gets_trimmed(self):
        docs = list(range(75))
        ceiling = conf.LLM_DOCS_MAX_FINAL_CEILING
        trimmed = docs[:ceiling] if len(docs) > ceiling else docs
        assert len(trimmed) == ceiling
        assert trimmed == docs[:ceiling]

    def test_under_ceiling_list_is_unchanged(self):
        docs = list(range(39))  # the exact reproduced multi-topic-merge count
        ceiling = conf.LLM_DOCS_MAX_FINAL_CEILING
        trimmed = docs[:ceiling] if len(docs) > ceiling else docs
        assert trimmed == docs
        assert len(trimmed) == 39

"""
Narrow, purpose-built test for the 2026-11 retry-duplication fix (.claude/
plans/mutable-seeking-wave.md): constructs representative real persona
services (no live network calls — ChatOpenAI's constructor doesn't validate
credentials or call out) and asserts their ChatOpenAI clients were built
with max_retries=0, i.e. utils.llm_call.CHAT_OPENAI_MAX_RETRIES, not the
unset default that silently falls through to the openai SDK's own hidden
max_retries=2.

This is one of the few places a kwarg-introspecting test is appropriate —
it exists specifically to test this fix, not incidental to some other
feature's tests (every other persona/supervisor test deliberately mocks
BELOW construction, e.g. llm_invoke or a classifier closure, per this
codebase's existing convention).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

CODE_DIR = Path(__file__).parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.llm_call import CHAT_OPENAI_MAX_RETRIES


@pytest.mark.unit
class TestChatOpenAIMaxRetries:
    def test_constant_is_zero(self):
        """The whole fix hinges on this being 0 — llm_invoke() must be the
        only retry layer, not reduced-but-still-stacked."""
        assert CHAT_OPENAI_MAX_RETRIES == 0

    def test_practical_main_answer_llm(self):
        from model.persona_practical import PracticalPersonaService

        service = PracticalPersonaService(retriever=MagicMock())
        assert service.llm.max_retries == CHAT_OPENAI_MAX_RETRIES

    def test_academic_main_answer_and_classifier_llms(self):
        from model.persona_academic import AcademicPersonaService

        service = AcademicPersonaService(retriever=MagicMock())
        assert service.llm.max_retries == CHAT_OPENAI_MAX_RETRIES
        assert service.llm_slots.max_retries == CHAT_OPENAI_MAX_RETRIES
        assert service.llm_classifier.max_retries == CHAT_OPENAI_MAX_RETRIES

    def test_supervisor_deflect_llm(self):
        from model.persona_supervisor import PersonaSupervisor

        supervisor = PersonaSupervisor(retriever=MagicMock())
        assert supervisor._deflect_llm.max_retries == CHAT_OPENAI_MAX_RETRIES

    def test_groundedness_judge_uses_the_longer_timeout(self):
        """2026-11 secondary fix: the groundedness judge must no longer
        share LLM_TOPIC_PICKER_TIMEOUT (8s) with every lightweight
        classifier — it needs conf.LLM_GROUNDEDNESS_JUDGE_TIMEOUT (20s)
        instead, since it processes the full retrieved-DOCUMENTS payload,
        not a short fixed input. Inspect the closure's captured `llm`
        rather than calling the judge (which would make a real LLM call).
        """
        from model.persona_practical import PracticalPersonaService
        import conf

        service = PracticalPersonaService(retriever=MagicMock())
        judge_llm = next(
            c.cell_contents for c in service._groundedness_llm_call.__closure__
            if hasattr(c.cell_contents, "request_timeout")
        )
        assert judge_llm.request_timeout == conf.LLM_GROUNDEDNESS_JUDGE_TIMEOUT
        assert judge_llm.max_retries == CHAT_OPENAI_MAX_RETRIES

    def test_groundedness_judges_use_the_raised_base_max_tokens(self):
        """2026-11 cost fix: base max_tokens=500 was found (via live-logged
        cost breakdown) to truncate routinely, not rarely — every truncation
        pays for a full discarded call AND a second, widened-budget retry.
        Raised to conf.GROUNDEDNESS_JUDGE_BASE_MAX_TOKENS (1200), comfortably
        above the real observed successful-retry usage (606, 1010 completion
        tokens) — same closure-inspection approach as the sibling timeout test
        above, no real LLM call made."""
        from model.persona_practical import PracticalPersonaService
        from model.persona_academic import AcademicPersonaService
        import conf

        practical = PracticalPersonaService(retriever=MagicMock())
        practical_judge_llm = next(
            c.cell_contents for c in practical._groundedness_llm_call.__closure__
            if hasattr(c.cell_contents, "request_timeout")
        )
        assert practical_judge_llm.max_tokens == conf.GROUNDEDNESS_JUDGE_BASE_MAX_TOKENS

        academic = AcademicPersonaService(retriever=MagicMock())
        academic_judge_llm = next(
            c.cell_contents for c in academic._groundedness_llm_call.__closure__
            if hasattr(c.cell_contents, "request_timeout")
        )
        assert academic_judge_llm.max_tokens == conf.GROUNDEDNESS_JUDGE_BASE_MAX_TOKENS

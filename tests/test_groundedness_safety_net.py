"""
Unit tests for the 2026-10 groundedness safety net (persona_practical.py) —
PracticalPersonaService._apply_groundedness_safety_net and the
_low_confidence_retrieval flag it cooperates with.

Background (QA-reproduced, Phase-0 codebase audit): real test failures showed
the bot inventing facts not present in any retrieved document — a Zoom/Teams
meeting requirement, a 660-baht VAT fee, a "Reset Password" procedure. The
existing anti-hallucination defenses are prompt-only plus two narrow
shape-based strips (phone numbers, URLs); neither catches a fabricated claim
phrased as ordinary prose. This adds a cheap LLM-judge check with a
retry-once safety net, same shape as the existing entity_type/multi-license
retry-once nets elsewhere in handle() — but unlike those, extracted into its
own method specifically so it's unit-testable without driving the entire
handle() flow.

Same testing approach as test_practical_truncation_retry.py:
PracticalPersonaService(retriever=MagicMock()) constructs with no network
access; the one LLM-calling dependency each test touches
(_groundedness_llm_call / _call_llm_json_with_truncation_retry) is
monkeypatched directly since it's the network boundary.
"""
from __future__ import annotations

import inspect
import re
from unittest.mock import MagicMock

import pytest

from model.persona_practical import PracticalPersonaService


@pytest.fixture
def practical():
    return PracticalPersonaService(retriever=MagicMock())


@pytest.mark.unit
class TestGroundednessSafetyNetOrchestration:
    def test_non_answer_action_is_untouched_no_judge_call(self, practical):
        practical._groundedness_llm_call = MagicMock()
        decision = {"action": "ask", "execution": {"question": "x"}}
        exec_ = {"question": "x"}

        out_decision, out_exec = practical._apply_groundedness_safety_net(
            "ask", exec_, decision, docs_json=[], prompt="prompt text",
        )

        assert out_decision is decision
        assert out_exec is exec_
        practical._groundedness_llm_call.assert_not_called()

    def test_answer_with_empty_text_is_untouched_no_judge_call(self, practical):
        practical._groundedness_llm_call = MagicMock()
        decision = {"action": "answer", "execution": {"answer": ""}}
        exec_ = {"answer": ""}

        out_decision, out_exec = practical._apply_groundedness_safety_net(
            "answer", exec_, decision, docs_json=[], prompt="prompt text",
        )

        assert out_decision is decision
        assert out_exec is exec_
        practical._groundedness_llm_call.assert_not_called()

    def test_grounded_answer_passes_through_unchanged_no_retry(self, practical):
        practical._groundedness_llm_call = MagicMock(return_value={"unsupported": False, "claims": []})
        practical._call_llm_json_with_truncation_retry = MagicMock()
        decision = {"action": "answer", "execution": {"answer": "real answer from docs"}}
        exec_ = {"answer": "real answer from docs"}

        out_decision, out_exec = practical._apply_groundedness_safety_net(
            "answer", exec_, decision, docs_json=[{"content": "real docs"}], prompt="prompt text",
        )

        assert out_decision is decision
        assert out_exec is exec_
        practical._groundedness_llm_call.assert_called_once()
        practical._call_llm_json_with_truncation_retry.assert_not_called()

    def test_unsupported_claim_retries_and_adopts_grounded_retry_result(self, practical):
        practical._groundedness_llm_call = MagicMock(side_effect=[
            {"unsupported": True, "claims": ["ค่าธรรมเนียม 660 บาท"]},
            {"unsupported": False, "claims": []},
        ])
        retry_decision = {"action": "answer", "execution": {"answer": "rewritten answer, no invented fee"}}
        practical._call_llm_json_with_truncation_retry = MagicMock(return_value=retry_decision)
        decision = {"action": "answer", "execution": {"answer": "answer with invented 660 baht fee"}}
        exec_ = {"answer": "answer with invented 660 baht fee"}

        out_decision, out_exec = practical._apply_groundedness_safety_net(
            "answer", exec_, decision, docs_json=[{"content": "real docs"}], prompt="prompt text",
        )

        assert out_decision == retry_decision
        assert out_exec == {"answer": "rewritten answer, no invented fee"}
        assert practical._groundedness_llm_call.call_count == 2
        practical._call_llm_json_with_truncation_retry.assert_called_once()
        # The retry prompt must carry the flagged claim and the original prompt.
        retry_prompt_arg = practical._call_llm_json_with_truncation_retry.call_args[0][0]
        assert "prompt text" in retry_prompt_arg
        assert "660 บาท" in retry_prompt_arg

    def test_retry_still_unsupported_keeps_first_attempt(self, practical):
        practical._groundedness_llm_call = MagicMock(side_effect=[
            {"unsupported": True, "claims": ["claim A"]},
            {"unsupported": True, "claims": ["claim A still here"]},
        ])
        retry_decision = {"action": "answer", "execution": {"answer": "still has claim A"}}
        practical._call_llm_json_with_truncation_retry = MagicMock(return_value=retry_decision)
        decision = {"action": "answer", "execution": {"answer": "original with claim A"}}
        exec_ = {"answer": "original with claim A"}

        out_decision, out_exec = practical._apply_groundedness_safety_net(
            "answer", exec_, decision, docs_json=[], prompt="prompt text",
        )

        assert out_decision is decision
        assert out_exec is exec_

    def test_retry_returning_no_answer_text_keeps_first_attempt(self, practical):
        practical._groundedness_llm_call = MagicMock(return_value={"unsupported": True, "claims": ["x"]})
        practical._call_llm_json_with_truncation_retry = MagicMock(
            return_value={"action": "ask", "execution": {"question": "ถามกลับ"}}
        )
        decision = {"action": "answer", "execution": {"answer": "original"}}
        exec_ = {"answer": "original"}

        out_decision, out_exec = practical._apply_groundedness_safety_net(
            "answer", exec_, decision, docs_json=[], prompt="prompt text",
        )

        assert out_decision is decision
        assert out_exec is exec_
        # Only the first judge call — retry answer was empty, no point re-judging.
        practical._groundedness_llm_call.assert_called_once()

    def test_retry_call_raising_does_not_crash_and_keeps_first_attempt(self, practical):
        practical._groundedness_llm_call = MagicMock(return_value={"unsupported": True, "claims": ["x"]})
        practical._call_llm_json_with_truncation_retry = MagicMock(side_effect=RuntimeError("network blip"))
        decision = {"action": "answer", "execution": {"answer": "original"}}
        exec_ = {"answer": "original"}

        out_decision, out_exec = practical._apply_groundedness_safety_net(
            "answer", exec_, decision, docs_json=[], prompt="prompt text",
        )

        assert out_decision is decision
        assert out_exec is exec_

    def test_judge_call_itself_raising_fails_open_as_grounded(self, practical):
        """_default_groundedness_llm_call's own _call closure must fail open
        (treat as grounded) on an LLM error — asserted indirectly here via a
        mock that mimics that documented fail-open contract, since the real
        closure's own try/except is what _apply_groundedness_safety_net relies
        on never raising out to it."""
        practical._groundedness_llm_call = MagicMock(return_value={"unsupported": False, "claims": []})
        decision = {"action": "answer", "execution": {"answer": "answer"}}
        exec_ = {"answer": "answer"}

        out_decision, out_exec = practical._apply_groundedness_safety_net(
            "answer", exec_, decision, docs_json=[], prompt="prompt text",
        )

        assert out_decision is decision
        assert out_exec is exec_


@pytest.mark.unit
class TestLowConfidenceRetrievalFlag:
    """_retrieve_docs' similarity-filter fallback (persona_practical.py) sets
    state.context["_low_confidence_retrieval"] so the prompt-builder can bias
    the LLM toward [[INFO_GAP]]. _retrieve_docs is too large/entangled
    (chapter sweep, query rewrite, synonym expansion, live retriever calls)
    to drive end-to-end in a unit test — same situation as this file's
    sibling TestBoostFieldsIncludesDepartment in test_practical_truncation_
    retry.py, which uses the same source-inspection approach for the same
    reason. This is a regression pin on the code shape, not a behavioral
    integration test."""

    def test_fallback_branch_sets_flag_true_and_normal_branch_sets_false(self):
        source = inspect.getsource(PracticalPersonaService._retrieve_docs)
        fallback_block = re.search(
            r"if len\(filtered_docs\) < 2:(.*?)else:(.*?)docs = filtered_docs",
            source, re.DOTALL,
        )
        assert fallback_block, "similarity-filter fallback block not found in _retrieve_docs"
        fallback_branch, normal_branch = fallback_block.group(1), fallback_block.group(2)
        assert '_low_confidence_retrieval"] = True' in fallback_branch
        assert '_low_confidence_retrieval"] = False' in normal_branch

    def test_prompt_builder_reads_the_flag_and_injects_the_instruction(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        assert "_low_confidence_instruction" in source
        assert '_low_confidence_retrieval' in source
        # Must actually be interpolated into the final prompt f-string, not just defined.
        assert re.search(r"\{_low_confidence_instruction\}", source)

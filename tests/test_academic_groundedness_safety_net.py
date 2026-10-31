"""
Unit tests for the 2026-10 Academic-persona port of the groundedness safety net
(persona_academic.py) — AcademicPersonaService._apply_groundedness_safety_net.

Background: the groundedness safety net shipped earlier the same session for
Practical (see test_groundedness_safety_net.py) checks every draft answer
against retrieved DOCUMENTS before it reaches the user. Academic generates
real legal answers from real DOCUMENTS the exact same way, via its own LLM
call in _finalize_answer, with no equivalent check until this port. The judge
prompt/logic (_default_groundedness_llm_call) is copied verbatim from
Practical's already-calibrated version (6/6 on held-out cases) — only the
orchestration method differs, since this file's _finalize_answer has no
action/decision/exec_ dispatch to thread through and no truncation-retry
wrapper to reuse. _apply_groundedness_safety_net here is plain string in,
string out.

Same testing approach as test_groundedness_safety_net.py:
AcademicPersonaService(retriever=MagicMock()) constructs with no network
access; the two LLM-calling dependencies each test touches
(_groundedness_llm_call / _call_llm_json) are monkeypatched directly since
they are the network boundary.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from model.persona_academic import AcademicPersonaService


@pytest.fixture
def academic():
    return AcademicPersonaService(retriever=MagicMock())


@pytest.mark.unit
class TestAcademicGroundednessSafetyNetOrchestration:
    def test_empty_answer_is_untouched_no_judge_call(self, academic):
        academic._groundedness_llm_call = MagicMock()

        out = academic._apply_groundedness_safety_net("", docs_json=[], prompt="prompt text")

        assert out == ""
        academic._groundedness_llm_call.assert_not_called()

    def test_grounded_answer_passes_through_unchanged_no_retry(self, academic):
        academic._groundedness_llm_call = MagicMock(return_value={"unsupported": False, "claims": []})
        academic._call_llm_json = MagicMock()

        out = academic._apply_groundedness_safety_net(
            "real answer from docs", docs_json=[{"content": "real docs"}], prompt="prompt text",
        )

        assert out == "real answer from docs"
        academic._groundedness_llm_call.assert_called_once()
        academic._call_llm_json.assert_not_called()

    def test_unsupported_claim_retries_and_adopts_grounded_retry_result(self, academic):
        academic._groundedness_llm_call = MagicMock(side_effect=[
            {"unsupported": True, "claims": ["ค่าธรรมเนียม 1,200 บาท"]},
            {"unsupported": False, "claims": []},
        ])
        academic._call_llm_json = MagicMock(
            return_value={"action": "answer", "execution": {"answer": "rewritten answer, no invented fee"}}
        )

        out = academic._apply_groundedness_safety_net(
            "answer with invented 1,200 baht fee", docs_json=[{"content": "real docs"}], prompt="prompt text",
        )

        assert out == "rewritten answer, no invented fee"
        assert academic._groundedness_llm_call.call_count == 2
        academic._call_llm_json.assert_called_once()
        # The retry prompt must carry the flagged claim and the original prompt.
        retry_prompt_arg = academic._call_llm_json.call_args[0][0]
        assert "prompt text" in retry_prompt_arg
        assert "1,200 บาท" in retry_prompt_arg

    def test_retry_still_unsupported_keeps_first_attempt(self, academic):
        academic._groundedness_llm_call = MagicMock(side_effect=[
            {"unsupported": True, "claims": ["claim A"]},
            {"unsupported": True, "claims": ["claim A still here"]},
        ])
        academic._call_llm_json = MagicMock(
            return_value={"action": "answer", "execution": {"answer": "still has claim A"}}
        )

        out = academic._apply_groundedness_safety_net(
            "original with claim A", docs_json=[], prompt="prompt text",
        )

        assert out == "original with claim A"

    def test_retry_returning_no_answer_text_keeps_first_attempt(self, academic):
        academic._groundedness_llm_call = MagicMock(return_value={"unsupported": True, "claims": ["x"]})
        academic._call_llm_json = MagicMock(return_value={"action": "ask", "execution": {"question": "ถามกลับ"}})

        out = academic._apply_groundedness_safety_net(
            "original", docs_json=[], prompt="prompt text",
        )

        assert out == "original"
        # Only the first judge call — retry answer was empty, no point re-judging.
        academic._groundedness_llm_call.assert_called_once()

    def test_retry_call_raising_does_not_crash_and_keeps_first_attempt(self, academic):
        academic._groundedness_llm_call = MagicMock(return_value={"unsupported": True, "claims": ["x"]})
        academic._call_llm_json = MagicMock(side_effect=RuntimeError("network blip"))

        out = academic._apply_groundedness_safety_net(
            "original", docs_json=[], prompt="prompt text",
        )

        assert out == "original"

    def test_judge_call_itself_failing_open_as_grounded_is_passed_through(self, academic):
        """_default_groundedness_llm_call's own _call closure must fail open
        (treat as grounded) on an LLM error — asserted indirectly here via a
        mock that mimics that documented fail-open contract, since the real
        closure's own try/except is what _apply_groundedness_safety_net relies
        on never raising out to it."""
        academic._groundedness_llm_call = MagicMock(return_value={"unsupported": False, "claims": []})

        out = academic._apply_groundedness_safety_net("answer", docs_json=[], prompt="prompt text")

        assert out == "answer"

    def test_retry_answer_runs_through_info_gap_tag_and_line_wrapping(self, academic):
        """The retry's answer must go through the same _shared_extract_info_gap_tag
        + _fix_line_wrapping treatment the original answer already went through
        (persona_academic.py lines ~5140/5148), so both paths are processed
        identically before the caller's URL/phone strips."""
        academic._groundedness_llm_call = MagicMock(side_effect=[
            {"unsupported": True, "claims": ["fee"]},
            {"unsupported": False, "claims": []},
        ])
        academic._call_llm_json = MagicMock(
            return_value={"action": "answer", "execution": {"answer": "ไม่พบข้อมูลค่าธรรมเนียม [[INFO_GAP]]"}}
        )
        state = MagicMock()
        state.context = {}

        out = academic._apply_groundedness_safety_net(
            "fabricated fee answer", docs_json=[], prompt="prompt text", state=state,
        )

        assert "[[INFO_GAP]]" not in out
        assert state.context.get("_info_gap_detected") is True

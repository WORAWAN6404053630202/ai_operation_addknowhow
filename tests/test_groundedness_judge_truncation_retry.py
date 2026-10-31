"""
Unit tests for the 2026-11 truncation-retry fix to both personas' groundedness
judge (_default_groundedness_llm_call in persona_practical.py and
persona_academic.py).

Background: after LangSmith label-tagging shipped (2026-10), a real trace
tagged "Practical/groundedness" showed a LengthFinishReasonError — the judge's
base max_tokens=500 was exceeded by a dense DOCUMENTS payload (a verbose
commercial-registration doc with fees/legal_regulatory/operation_steps/terms_
and_conditions all populated) plus the required chain-of-thought reasoning.
Live-observed on a routine question, not a rare edge case. Before this fix,
that exception silently fell into the generic fail-open path (treated as
grounded — verification SKIPPED, not performed). Fix mirrors this codebase's
already-established _call_llm_json_with_truncation_retry shape: keep the base
max_tokens budget cheap for the common case, retry ONCE with a widened budget
(conf.GROUNDEDNESS_JUDGE_RETRY_MAX_TOKENS, default 900) specifically when a
LengthFinishReasonError is detected, and only fail open if the retry ALSO
fails.

Both PracticalPersonaService and AcademicPersonaService build their judge's
real ChatOpenAI client inside the closure at __init__ time (no network access
until .invoke() is actually called) — these tests patch the module-level
llm_invoke import (model.persona_practical.llm_invoke /
model.persona_academic.llm_invoke) since that's the actual network boundary,
same pattern as test_deflect_slot_fidelity.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from model.persona_practical import PracticalPersonaService
from model.persona_academic import AcademicPersonaService


class FakeLengthFinishReasonError(Exception):
    """Named to match the type(e).__name__ string check in both judges'
    truncation-retry logic — doesn't need to be the real openai exception."""


def _fake_response(text: str):
    from types import SimpleNamespace
    return SimpleNamespace(content=text, usage_metadata=None, response_metadata={})


@pytest.mark.unit
class TestPracticalGroundednessJudgeTruncationRetry:
    @pytest.fixture
    def service(self):
        return PracticalPersonaService(retriever=MagicMock())

    def test_length_error_retries_with_widened_max_tokens_and_succeeds(self, service):
        ok_json = '{"reasoning": "ok", "unsupported": false, "claims": []}'
        with patch(
            "model.persona_practical.llm_invoke",
            side_effect=[FakeLengthFinishReasonError("length"), _fake_response(ok_json)],
        ) as mock_invoke:
            result = service._groundedness_llm_call("some answer", [{"content": "docs"}])

        assert result == {"unsupported": False, "claims": [], "reasoning": "ok"}
        assert mock_invoke.call_count == 2
        # Second call's llm arg must be a widened-max_tokens bind, not the base client.
        second_llm_arg = mock_invoke.call_args_list[1].args[0]
        assert second_llm_arg.kwargs.get("max_tokens") == 900

    def test_length_error_retry_also_fails_open(self, service):
        with patch(
            "model.persona_practical.llm_invoke",
            side_effect=[FakeLengthFinishReasonError("length"), FakeLengthFinishReasonError("length again")],
        ):
            result = service._groundedness_llm_call("some answer", [{"content": "docs"}])

        assert result == {"unsupported": False, "claims": [], "reasoning": ""}

    def test_non_length_error_fails_open_immediately_without_retry(self, service):
        with patch(
            "model.persona_practical.llm_invoke", side_effect=RuntimeError("network blip"),
        ) as mock_invoke:
            result = service._groundedness_llm_call("some answer", [{"content": "docs"}])

        assert result == {"unsupported": False, "claims": [], "reasoning": ""}
        mock_invoke.assert_called_once()

    def test_retry_ceiling_is_configurable_via_conf(self, service):
        with patch("model.persona_practical.conf.GROUNDEDNESS_JUDGE_RETRY_MAX_TOKENS", 1234, create=True), patch(
            "model.persona_practical.llm_invoke",
            side_effect=[FakeLengthFinishReasonError("length"), _fake_response('{"reasoning":"x","unsupported":false,"claims":[]}')],
        ) as mock_invoke:
            service._groundedness_llm_call("some answer", [{"content": "docs"}])

        second_llm_arg = mock_invoke.call_args_list[1].args[0]
        assert second_llm_arg.kwargs.get("max_tokens") == 1234


@pytest.mark.unit
class TestAcademicGroundednessJudgeTruncationRetry:
    @pytest.fixture
    def service(self):
        return AcademicPersonaService(retriever=MagicMock())

    def test_length_error_retries_with_widened_max_tokens_and_succeeds(self, service):
        ok_json = '{"reasoning": "ok", "unsupported": false, "claims": []}'
        with patch(
            "model.persona_academic.llm_invoke",
            side_effect=[FakeLengthFinishReasonError("length"), _fake_response(ok_json)],
        ) as mock_invoke:
            result = service._groundedness_llm_call("some answer", [{"content": "docs"}])

        assert result == {"unsupported": False, "claims": [], "reasoning": "ok"}
        assert mock_invoke.call_count == 2
        second_llm_arg = mock_invoke.call_args_list[1].args[0]
        assert second_llm_arg.kwargs.get("max_tokens") == 900

    def test_length_error_retry_also_fails_open(self, service):
        with patch(
            "model.persona_academic.llm_invoke",
            side_effect=[FakeLengthFinishReasonError("length"), FakeLengthFinishReasonError("length again")],
        ):
            result = service._groundedness_llm_call("some answer", [{"content": "docs"}])

        assert result == {"unsupported": False, "claims": [], "reasoning": ""}

    def test_non_length_error_fails_open_immediately_without_retry(self, service):
        with patch(
            "model.persona_academic.llm_invoke", side_effect=RuntimeError("network blip"),
        ) as mock_invoke:
            result = service._groundedness_llm_call("some answer", [{"content": "docs"}])

        assert result == {"unsupported": False, "claims": [], "reasoning": ""}
        mock_invoke.assert_called_once()

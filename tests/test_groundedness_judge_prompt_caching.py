"""
Unit tests for the 2026-11 cost optimization to both personas' groundedness
judge: the static instructions + 3 few-shot examples (byte-identical on
every call) are now sent as a cached SystemMessage via build_cached_system_
message (the same helper the main answer-generation prompts already use),
instead of being concatenated into one HumanMessage string and re-paid at
full input price on every single answer turn.

Only DOCUMENTS + the draft answer (genuinely different every call) remain
in the per-call HumanMessage.

These tests confirm the message SHAPE (SystemMessage with cache_control +
a separate, shorter HumanMessage) rather than real caching behavior —
whether Anthropic's server actually honors the cache (the static block may
be close to or under their minimum cacheable size for Haiku models) can
only be confirmed via a real trace, not a unit test; see memory for the
live-verification note.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from model.persona_practical import PracticalPersonaService
from model.persona_academic import AcademicPersonaService


def _fake_response(text: str):
    from types import SimpleNamespace
    return SimpleNamespace(content=text, usage_metadata=None, response_metadata={})


@pytest.mark.unit
class TestPracticalGroundednessJudgePromptCaching:
    @pytest.fixture
    def service(self):
        return PracticalPersonaService(retriever=MagicMock())

    def test_system_message_carries_cache_control_and_human_message_is_short(self, service):
        ok_json = '{"reasoning": "ok", "unsupported": false, "claims": []}'
        with patch(
            "model.persona_practical.llm_invoke", return_value=_fake_response(ok_json),
        ) as mock_invoke:
            service._groundedness_llm_call("some answer", [{"content": "docs"}])

        messages = mock_invoke.call_args.args[1]
        assert len(messages) == 2
        sys_msg, human_msg = messages
        assert sys_msg.__class__.__name__ == "SystemMessage"
        assert human_msg.__class__.__name__ == "HumanMessage"
        # The static instructions/few-shot text must NOT be duplicated into the
        # per-call HumanMessage — only DOCUMENTS + the draft answer belong there.
        assert "ตัวอย่างที่ 1" not in human_msg.content
        assert "some answer" in human_msg.content
        # When caching is enabled (conf default), the system content is a
        # content-block list with cache_control attached, not a plain string.
        if isinstance(sys_msg.content, list):
            assert sys_msg.content[0].get("cache_control", {}).get("type") == "ephemeral"
            assert "ตัวอย่างที่ 1" in sys_msg.content[0]["text"]
        else:
            # Caching disabled in this environment — still correct, just unguarded.
            assert "ตัวอย่างที่ 1" in sys_msg.content


@pytest.mark.unit
class TestAcademicGroundednessJudgePromptCaching:
    @pytest.fixture
    def service(self):
        return AcademicPersonaService(retriever=MagicMock())

    def test_system_message_carries_cache_control_and_human_message_is_short(self, service):
        ok_json = '{"reasoning": "ok", "unsupported": false, "claims": []}'
        with patch(
            "model.persona_academic.llm_invoke", return_value=_fake_response(ok_json),
        ) as mock_invoke:
            service._groundedness_llm_call("some answer", [{"content": "docs"}])

        messages = mock_invoke.call_args.args[1]
        assert len(messages) == 2
        sys_msg, human_msg = messages
        assert sys_msg.__class__.__name__ == "SystemMessage"
        assert human_msg.__class__.__name__ == "HumanMessage"
        assert "ตัวอย่างที่ 1" not in human_msg.content
        assert "some answer" in human_msg.content
        if isinstance(sys_msg.content, list):
            assert sys_msg.content[0].get("cache_control", {}).get("type") == "ephemeral"
            assert "ตัวอย่างที่ 1" in sys_msg.content[0]["text"]
        else:
            assert "ตัวอย่างที่ 1" in sys_msg.content

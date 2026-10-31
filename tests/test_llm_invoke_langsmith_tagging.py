"""
Unit tests for the 2026-10 fix (Phase-0 audit follow-up) making llm_invoke/
llm_invoke_async pass a LangChain `config={"tags": ..., "metadata": ...}`
dict into llm.invoke()/llm.ainvoke(), so individual calls become filterable
by label (e.g. "Practical/groundedness") in the LangSmith UI once
LANGSMITH_TRACING is enabled.

Background: investigation found LangSmith was never actually code-wired
despite an old internal note claiming it was — no config/tags/metadata ever
reached a trace. Fix is a 2-line addition at each llm.invoke()/llm.ainvoke()
call site; this test proves the config dict shape without needing a real
network call or LangSmith client (tracing activation itself is purely an
env-var concern handled by langchain-core, out of scope for a unit test —
see utils/llm_call.py's docstring comment at the new config= lines).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import utils.llm_call as llm_call_mod  # noqa: E402


def _fake_response():
    return SimpleNamespace(content="ok", usage_metadata=None, response_metadata={})


@pytest.mark.unit
class TestLlmInvokeLangsmithTagging:
    def test_llm_invoke_passes_config_with_tags_and_metadata(self):
        llm = MagicMock()
        llm.invoke.return_value = _fake_response()

        llm_call_mod.llm_invoke(
            llm, ["msg"], label="Practical/groundedness", persona="practical", operation="groundedness",
        )

        config = llm.invoke.call_args.kwargs["config"]
        assert set(config["tags"]) == {"Practical/groundedness", "practical", "groundedness"}
        assert config["metadata"] == {
            "persona": "practical", "operation": "groundedness", "label": "Practical/groundedness",
        }

    def test_tags_drop_falsy_persona_and_operation_when_not_passed(self):
        """persona/operation aren't passed at most real call sites today —
        tags must not end up with None/empty entries."""
        llm = MagicMock()
        llm.invoke.return_value = _fake_response()

        llm_call_mod.llm_invoke(llm, ["msg"], label="Academic/json")

        config = llm.invoke.call_args.kwargs["config"]
        assert config["tags"] == ["Academic/json"]
        assert config["metadata"] == {"persona": None, "operation": None, "label": "Academic/json"}

    def test_retry_attempt_also_carries_the_same_config(self):
        llm = MagicMock()
        llm.invoke.side_effect = [RuntimeError("transient"), _fake_response()]

        llm_call_mod.llm_invoke(llm, ["msg"], label="Supervisor/deflect")

        assert llm.invoke.call_count == 2
        for call in llm.invoke.call_args_list:
            assert call.kwargs["config"]["tags"] == ["Supervisor/deflect"]


@pytest.mark.unit
class TestLlmInvokeAsyncLangsmithTagging:
    @pytest.mark.asyncio
    async def test_llm_invoke_async_passes_config_with_tags_and_metadata(self):
        # A plain async function doesn't record call_args like MagicMock does,
        # so capture kwargs into a list manually.
        calls = []

        async def _tracking_ainvoke(messages, **kwargs):
            calls.append(kwargs)
            return _fake_response()

        llm = MagicMock()
        llm.ainvoke = _tracking_ainvoke

        await llm_call_mod.llm_invoke_async(
            llm, ["msg"], label="Academic/groundedness", persona="academic", operation="groundedness",
        )

        assert len(calls) == 1
        config = calls[0]["config"]
        assert set(config["tags"]) == {"Academic/groundedness", "academic", "groundedness"}
        assert config["metadata"] == {
            "persona": "academic", "operation": "groundedness", "label": "Academic/groundedness",
        }

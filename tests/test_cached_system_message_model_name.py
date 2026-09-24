"""
Regression test for a real production incident (2026-09-24): total prompt-caching
outage. PracticalPersonaService._call_llm_json built the cached system message via
`getattr(self.llm, "model", "")` — but langchain_openai as pinned in this project's
requirements.txt (0.3.35) stores the model name under `.model_name`, not `.model`.
The getattr always silently returned "" (its default), so
`build_cached_system_message`'s `"anthropic/" in model_name` check was always False,
and cache_control was never attached on any real call, ever — confirmed via a live
debug-request-body capture against the actual deployed container (0% cache_read
across many days of testing, despite the caching mechanism itself being proven
correct in isolation with a hand-built model-name string).

Fix: store the already-known-correct model name as `self._model_name` in
`_init_llm()` and use that directly, instead of re-deriving it from the ChatOpenAI
instance's attributes (whose exact name is a langchain_openai implementation detail
that has already changed across versions once).

No network access, no LLM calls.
"""
import sys
import os
from unittest.mock import MagicMock

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))

import pytest
import conf
from model.persona_practical import PracticalPersonaService
from utils.llm_call import build_cached_system_message


@pytest.mark.unit
class TestCachedSystemMessageUsesRealModelName:

    def test_model_name_stored_matches_configured_practical_model(self):
        svc = PracticalPersonaService(retriever=MagicMock())
        expected = getattr(conf, "OPENROUTER_MODEL_PRACTICAL", conf.OPENROUTER_MODEL)
        assert svc._model_name == expected

    def test_stored_model_name_is_never_empty(self):
        """The exact failure mode of the real bug: getattr(..., "") silently
        returning an empty string with no error. Guard against a regression that
        reintroduces any silent-default lookup for this value."""
        svc = PracticalPersonaService(retriever=MagicMock())
        assert svc._model_name != ""
        assert svc._model_name is not None

    def test_stored_model_name_actually_triggers_cache_control(self, monkeypatch):
        """End-to-end proof that the fix closes the real gap: build the cached
        system message exactly the way _call_llm_json does now, and confirm
        cache_control is attached — this would have FAILED before the fix, since
        getattr(self.llm, "model", "") produced "" and silently skipped caching."""
        monkeypatch.setattr(conf, "PROMPT_CACHING_ENABLED", True, raising=False)
        svc = PracticalPersonaService(retriever=MagicMock())
        assert "anthropic/" in svc._model_name

        msg = build_cached_system_message("some system prompt text", svc._model_name)
        assert isinstance(msg.content, list)
        assert any(
            isinstance(b, dict) and "cache_control" in b for b in msg.content
        )

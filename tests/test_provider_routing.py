"""
Unit tests for OpenRouter provider-routing/pinning in utils.llm_call.build_speed_extra_body.

Root-caused 2026-09: production showed cache_read_tokens == 0 on effectively every
real call despite the caching mechanism itself (build_cached_system_message) being
independently verified correct in isolation. Cause: `provider.sort` alone is a soft
preference — OpenRouter can still serve the same Anthropic model via Anthropic
direct, AWS Bedrock, or GCP Vertex interchangeably call-to-call, and prompt-cache
state does not carry over between those backends. Fix: pin Anthropic-model calls to
`provider.order=["anthropic"]` (PROVIDER_PIN_ANTHROPIC, default on), leaving
allow_fallbacks at OpenRouter's default (true) so a genuine Anthropic-side outage
still fails over instead of erroring out.

No network access, no LLM calls — pure config/dict logic.
"""
import sys
import os

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))

import pytest
import conf
from utils.llm_call import build_speed_extra_body


@pytest.mark.unit
class TestProviderPinning:

    def test_anthropic_model_pinned_by_default(self, monkeypatch):
        monkeypatch.setattr(conf, "PROVIDER_PIN_ANTHROPIC", True, raising=False)
        extra = build_speed_extra_body("anthropic/claude-sonnet-4-5")
        assert extra == {"provider": {"order": ["anthropic"]}}

    def test_pinning_does_not_set_allow_fallbacks_false(self, monkeypatch):
        """allow_fallbacks must stay at OpenRouter's default (true) — pinning must
        not remove the safety net for a genuine Anthropic-side outage."""
        monkeypatch.setattr(conf, "PROVIDER_PIN_ANTHROPIC", True, raising=False)
        extra = build_speed_extra_body("anthropic/claude-haiku-4-5")
        assert "allow_fallbacks" not in extra.get("provider", {})

    def test_non_anthropic_model_not_pinned(self, monkeypatch):
        monkeypatch.setattr(conf, "PROVIDER_PIN_ANTHROPIC", True, raising=False)
        monkeypatch.setattr(conf, "PROVIDER_ROUTING_SORT", "throughput", raising=False)
        extra = build_speed_extra_body("openai/gpt-5.1")
        assert extra == {"provider": {"sort": "throughput"}}

    def test_pin_disabled_falls_back_to_sort(self, monkeypatch):
        monkeypatch.setattr(conf, "PROVIDER_PIN_ANTHROPIC", False, raising=False)
        monkeypatch.setattr(conf, "PROVIDER_ROUTING_SORT", "throughput", raising=False)
        extra = build_speed_extra_body("anthropic/claude-sonnet-4-5")
        assert extra == {"provider": {"sort": "throughput"}}

    def test_pin_disabled_and_no_sort_returns_empty(self, monkeypatch):
        monkeypatch.setattr(conf, "PROVIDER_PIN_ANTHROPIC", False, raising=False)
        monkeypatch.setattr(conf, "PROVIDER_ROUTING_SORT", "", raising=False)
        extra = build_speed_extra_body("anthropic/claude-sonnet-4-5")
        assert extra == {}

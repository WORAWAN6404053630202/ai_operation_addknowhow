"""
Unit tests for the 2026-10 deterministic slot-fidelity check in
PersonaSupervisor._handle_deflect (persona_supervisor.py).

Background: _handle_deflect answers off-topic/greeting-adjacent/unclassifiable
turns (confirmed 3 call sites, none touch legal-answer generation — this path
never sees DOCUMENTS). It may reference state values already known exactly in
Python (last_license_type/last_topic, pending_slot key + options) inside a
free-generated LLM clarifying question. The risk isn't fabricating legal
facts (there's nothing to fabricate here) — it's the narrower, well-studied
"Slot Error Rate" problem: the LLM paraphrasing/distorting a value it was
already handed correctly. Research consensus: fix this with deterministic
verification against the known-correct value, not another LLM-judge call —
cheaper and more reliable, and matches this codebase's existing phone/URL
whitelist-and-strip pattern.

No LLM-judge call added by this fix — zero marginal cost. These tests patch
model.persona_supervisor.llm_invoke directly (the module-level import used
unqualified inside _handle_deflect) to control what the "LLM" returns,
exactly the network boundary for this method.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from model.persona_supervisor import PersonaSupervisor


@pytest.fixture
def supervisor():
    return PersonaSupervisor(retriever=MagicMock())


def _state_with_context(context: dict):
    state = MagicMock()
    state.context = context
    return state


def _fake_llm_response(text: str):
    return SimpleNamespace(content=text)


@pytest.mark.unit
class TestDeflectSlotFidelityCheck:
    def test_no_context_never_triggers_fallback_even_on_odd_reply(self, supervisor):
        state = _state_with_context({})

        with patch("model.persona_supervisor.llm_invoke", return_value=_fake_llm_response("ต้องการความช่วยเหลือเรื่องอะไรครับ")):
            _, reply = supervisor._handle_deflect(state, "สวัสดีครับ")

        assert reply == "ต้องการความช่วยเหลือเรื่องอะไรครับ"

    def test_verbatim_reference_to_last_topic_passes_through(self, supervisor):
        state = _state_with_context({"last_license_type": "ใบอนุญาตจำหน่ายอาหาร"})

        with patch(
            "model.persona_supervisor.llm_invoke",
            return_value=_fake_llm_response("ยังอยากถามเรื่องใบอนุญาตจำหน่ายอาหารอยู่ไหมครับ"),
        ):
            _, reply = supervisor._handle_deflect(state, "แล้วไงต่อ")

        assert reply == "ยังอยากถามเรื่องใบอนุญาตจำหน่ายอาหารอยู่ไหมครับ"

    def test_distorted_reference_to_last_topic_falls_back_to_safe_reply(self, supervisor):
        state = _state_with_context({"last_license_type": "ใบอนุญาตจำหน่ายอาหาร"})

        with patch(
            "model.persona_supervisor.llm_invoke",
            return_value=_fake_llm_response("ยังอยากถามเรื่องใบอนุญาตขายของอยู่ไหมครับ"),
        ):
            _, reply = supervisor._handle_deflect(state, "แล้วไงต่อ")

        assert reply == supervisor._DEFLECT_FALLBACK_REPLY

    def test_verbatim_reference_to_pending_slot_key_and_option_passes_through(self, supervisor):
        state = _state_with_context({
            "pending_slot": {"key": "ประเภทนิติบุคคล", "options": ["บุคคลธรรมดา", "นิติบุคคล"]},
        })

        with patch(
            "model.persona_supervisor.llm_invoke",
            return_value=_fake_llm_response("ประเภทนิติบุคคลของคุณเป็นบุคคลธรรมดาหรือนิติบุคคลครับ"),
        ):
            _, reply = supervisor._handle_deflect(state, "เอ่อ ไม่แน่ใจ")

        assert reply == "ประเภทนิติบุคคลของคุณเป็นบุคคลธรรมดาหรือนิติบุคคลครับ"

    def test_distorted_pending_slot_options_falls_back_to_safe_reply(self, supervisor):
        state = _state_with_context({
            "pending_slot": {"key": "ประเภทนิติบุคคล", "options": ["บุคคลธรรมดา", "นิติบุคคล"]},
        })

        with patch(
            "model.persona_supervisor.llm_invoke",
            return_value=_fake_llm_response("คุณเป็นเจ้าของคนเดียวหรือเป็นบริษัทครับ"),
        ):
            _, reply = supervisor._handle_deflect(state, "เอ่อ ไม่แน่ใจ")

        assert reply == supervisor._DEFLECT_FALLBACK_REPLY

    def test_llm_failure_still_uses_safe_fallback(self, supervisor):
        state = _state_with_context({"last_license_type": "ใบอนุญาตจำหน่ายอาหาร"})

        with patch("model.persona_supervisor.llm_invoke", side_effect=RuntimeError("network blip")):
            _, reply = supervisor._handle_deflect(state, "แล้วไงต่อ")

        assert reply == supervisor._DEFLECT_FALLBACK_REPLY

"""Tests for utils/phone_extraction.py (core-M5/M6, 2026-10) — the shared
shape-based Thai phone/hotline extraction, moved out of persona_practical.py
so persona_academic.py's newly-ported phone-hallucination defense uses the
identical logic instead of a second hand-copy. Covers: the extraction
function itself, persona_practical.py's classmethod still delegating to it
correctly after the refactor, and persona_academic.py's prompt building
actually injecting the PHONE NUMBER GUARD + post-processing strip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils.phone_extraction import extract_thai_phone_numbers  # noqa: E402


@pytest.mark.unit
class TestExtractThaiPhoneNumbers:
    def test_plain_landline_with_dashes(self):
        assert extract_thai_phone_numbers("โทร 02-123-4567") == {"021234567"}

    def test_international_plus66_normalizes_to_local_form(self):
        assert extract_thai_phone_numbers("+66 2 123 4567") == {"021234567"}

    def test_matches_02_123_4567_and_plus66_as_the_same_number(self):
        a = extract_thai_phone_numbers("02-123-4567")
        b = extract_thai_phone_numbers("+66 2 123 4567")
        assert a == b

    def test_4digit_hotline_requires_keyword_context(self):
        assert extract_thai_phone_numbers("สายด่วน 1506") == {"1506"}
        assert extract_thai_phone_numbers("ค่าธรรมเนียม 1500 บาท") == set()

    def test_no_phone_numbers_in_plain_text_returns_empty_set(self):
        assert extract_thai_phone_numbers("เอกสารนี้ไม่มีเบอร์โทรใดๆ") == set()

    def test_empty_or_none_input_is_safe(self):
        assert extract_thai_phone_numbers("") == set()
        assert extract_thai_phone_numbers(None) == set()

    def test_department_name_with_english_label_not_requiring_thai_keyword(self):
        """The actual live-reproduced bug this shape-based design fixes:
        English department-name conventions ('Merchant Call Center') never
        anticipated by a keyword-gated design."""
        assert "021234567" in extract_thai_phone_numbers("Merchant Call Center: 02-123-4567")

    def test_multiple_distinct_numbers_in_one_text(self):
        result = extract_thai_phone_numbers("โทร 02-123-4567 หรือ 081-234-5678")
        assert "021234567" in result
        assert "0812345678" in result


@pytest.mark.unit
class TestPracticalDelegatesToSharedModule:
    def test_classmethod_wrapper_returns_same_result_as_shared_function(self):
        from model.persona_practical import PracticalPersonaService
        text = "โทร 02-123-4567"
        assert PracticalPersonaService._extract_thai_phone_numbers(text) == extract_thai_phone_numbers(text)


@pytest.mark.unit
class TestAcademicPhoneGuardPromptInjection:
    """persona_academic.py's _build_final_prompt previously had NO phone
    whitelist mechanism at all — relied solely on the prompt's "ABSOLUTE
    PROHIBITION" ban (now removed, see test_academic_contact_info_prompt
    below). This pins the new Layer 2 (ported from persona_practical.py)."""

    @staticmethod
    def _academic():
        from model.persona_academic import AcademicPersonaService
        return AcademicPersonaService.__new__(AcademicPersonaService)

    def _build_prompt(self, docs):
        from model.conversation_state import ConversationState
        ac = self._academic()
        state = ConversationState(session_id="t1", persona_id="academic", context={})
        state.current_docs = docs
        return ac._build_final_prompt(state, user_question="ติดต่อยังไง")

    def test_phone_in_documents_appears_in_guard_whitelist(self):
        docs = [{"content": "c", "metadata": {
            "license_type": "ทะเบียนนายจ้าง", "department": "สำนักงานประกันสังคม",
            "service_channel": "โทร 02-123-4567",
        }}]
        prompt = self._build_prompt(docs)
        assert "PHONE NUMBER GUARD" in prompt
        assert "021234567" in prompt or "02-123-4567" in prompt

    def test_no_phone_in_documents_guard_says_none_available(self):
        docs = [{"content": "c", "metadata": {
            "license_type": "ทะเบียนนายจ้าง", "department": "สำนักงานประกันสังคม",
        }}]
        prompt = self._build_prompt(docs)
        assert "No phone numbers appear in the retrieved DOCUMENTS" in prompt


@pytest.mark.unit
class TestAcademicContactInfoPromptNoLongerContradicts:
    """core-M5/M6: the old 'ABSOLUTE PROHIBITION — contact info: NEVER
    output ... phone number ... anywhere' line directly contradicted the
    Evidence Constraint a few lines above it (which explicitly allows a
    fact verbatim in DOCUMENTS). Pins that the contradiction is gone."""

    def test_absolute_prohibition_on_all_contact_info_is_removed(self):
        from utils.prompts_academic import SYSTEM_PROMPT
        assert "ABSOLUTE PROHIBITION — contact info" not in SYSTEM_PROMPT
        assert "NEVER output department physical address" not in SYSTEM_PROMPT

    def test_evidence_constraint_contact_rule_still_present(self):
        from utils.prompts_academic import SYSTEM_PROMPT
        assert "No contact details in DOCUMENTS" in SYSTEM_PROMPT

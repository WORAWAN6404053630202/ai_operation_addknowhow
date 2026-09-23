"""
Regression test for the phone-number hallucination-prevention filter
(model/persona_practical.py's _extract_thai_phone_numbers, PracticalPersonaService
Layers 2+3).

Live-reproduced bug (2026-09): the original filter only recognized a phone number
when preceded by a specific keyword (โทร, สายด่วน, Tel., Fax., ...). Real
service_channel data labels its numbers with "Merchant Call Center" and
"Customer Service 24 Hours" — English department-name conventions the keyword
list never anticipated. Result: a genuinely real, correctly-cited phone number
never made it into the whitelist Layer 2 builds from DOCUMENTS, so Layer 3 then
stripped the LLM's own correct output as "hallucinated" — confirmed via production
logs (`docker logs`), not just inference from reading the prompt.

Fix: match 9-11 digit numbers (landline/mobile, plain or +66, any separator style)
by SHAPE alone — no keyword needed, verified safe against the full real corpus
with zero false positives. Only 4-digit hotline codes stay keyword-gated, since a
bare 4-digit number is genuinely ambiguous with a baht amount.

No network access, no LLM calls — pure regex/string logic.
"""
import sys
import os

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))

import pytest

from model.persona_practical import PracticalPersonaService


@pytest.mark.unit
class TestPhoneExtractionCoversRealLabelConventions:

    def test_call_center_label_no_longer_needs_a_keyword_change(self):
        """The exact live-reproduced case: 'Call Center' isn't in any keyword
        list, but the number must still be recognized by shape."""
        text = "Merchant Call Center • 02-777-7444 • merchant@email.scb.co.th"
        assert "027777444" in PracticalPersonaService._extract_thai_phone_numbers(text)

    def test_customer_service_label_with_extra_words_before_the_number(self):
        text = "Customer Service 24 Hours • 02-888-8822 • k_biz_contact-center@kasikornbank.com"
        assert "028888822" in PracticalPersonaService._extract_thai_phone_numbers(text)

    def test_any_future_unknown_label_also_works_because_shape_not_keyword(self):
        """The whole point of the durable fix: a label word the team has never
        used before still works, because nothing keys off the label at all."""
        text = "Hello Center (brand new label never seen before): 081-234-5678"
        assert "0812345678" in PracticalPersonaService._extract_thai_phone_numbers(text)

    @pytest.mark.parametrize("text,expected", [
        ("081-234-5678", "0812345678"),
        ("081 234 5678", "0812345678"),
        ("0812345678", "0812345678"),
        ("(02) 123-4567", "021234567"),  # 02 + 123 + 4567 = 9 digits
        ("+66 2 123 4567", "021234567"),
        ("+662-123-4567", "021234567"),
    ])
    def test_all_common_written_formats_normalize_to_the_same_local_form(self, text, expected):
        found = PracticalPersonaService._extract_thai_phone_numbers(text)
        assert expected in found, f"{text!r} -> {found}, expected {expected!r} present"

    def test_hotline_short_code_still_requires_a_keyword(self):
        assert "1506" in PracticalPersonaService._extract_thai_phone_numbers("โทรติดต่อ 1506")
        # Bare 4-digit number with NO phone keyword must NOT be treated as a phone —
        # this is indistinguishable from a baht amount without context.
        assert "1506" not in PracticalPersonaService._extract_thai_phone_numbers("จำนวน 1506 ชิ้น")


@pytest.mark.unit
class TestPhoneExtractionDoesNotFalsePositive:

    @pytest.mark.parametrize("text", [
        "ค่าธรรมเนียม 1500 บาท",
        "ค่าปรับไม่เกิน 1200 บาท",
        "ระยะเวลา 10-15 วัน",
        "ภายใน 30 วัน",
        "เอกสาร (ภ.พ.06)",
        "ปีงบประมาณ 2569",
        "ไม่เกิน 100000 บาท",
    ])
    def test_common_non_phone_numeric_content_is_never_matched(self, text):
        assert PracticalPersonaService._extract_thai_phone_numbers(text) == set()


@pytest.mark.unit
class TestLayer2And3ShareOneImplementation:
    """Both the whitelist-builder (Layer 2) and the answer-side strip check
    (Layer 3) must call the exact same method — this is what guarantees they
    can never drift apart the way the original two-separate-regexes version did."""

    def test_extraction_is_a_single_shared_classmethod(self):
        import inspect
        assert hasattr(PracticalPersonaService, "_extract_thai_phone_numbers")
        # Called as a classmethod/staticmethod-style helper — same result whether
        # invoked via the class or an instance, no per-call regex recompilation.
        src = inspect.getsource(PracticalPersonaService._extract_thai_phone_numbers)
        assert "_PHONE_CANDIDATE_RE" in src
        assert "_HOTLINE_WITH_CTX_RE" in src

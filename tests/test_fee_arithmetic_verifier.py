"""2026-11 fix: the Practical persona's FEE ARITHMETIC prompt
(prompts_practical.py:157-170) asks the LLM to show tiered-fee math inline
as prose. Live-reproduced: the exact same question, same code, different
container restarts, produced "200 + 350 = 550" (right) on one run and
"200 + 350 = 650" (wrong — operands correct, the model just added wrong) on
another. PracticalPersonaService._find_arithmetic_mismatch is a pure,
deterministic (no LLM) verifier: it scans for inline "A <op> B = C"
equations and recomputes each in Python, so a wrong sum/product is caught
regardless of model/temperature. These are direct unit tests — the function
is pure, no mocking needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model.persona_practical import PracticalPersonaService  # noqa: E402


@pytest.mark.unit
class TestFindArithmeticMismatch:
    def test_catches_reproduced_bug_200_plus_350_equals_650(self):
        answer = (
            "ค่าธรรมเนียมฐาน (ไม่เกิน 10 ตร.ม.): 200 บาท\n"
            "พื้นที่เกิน 10 ตร.ม.: (45 - 10) × 10 = 350 บาท\n"
            "รวม: 200 + 350 = 650 บาท"
        )
        result = PracticalPersonaService._find_arithmetic_mismatch(answer)
        assert result is not None
        desc, match = result
        assert "650" in desc
        assert "550" in desc

    def test_accepts_correct_200_plus_350_equals_550(self):
        answer = (
            "ค่าธรรมเนียมฐาน (ไม่เกิน 10 ตร.ม.): 200 บาท\n"
            "พื้นที่เกิน 10 ตร.ม.: (45 - 10) × 10 = 350 บาท\n"
            "รวม: 200 + 350 = 550 บาท"
        )
        assert PracticalPersonaService._find_arithmetic_mismatch(answer) is None

    def test_catches_multiplication_shape(self):
        # Different-unit tier shape: final = rate × full_value (no subtraction).
        answer = "อาคารเกิน 3 ชั้น คิดอัตรา 4 บาทต่อตารางเมตร: 4×100 = 500 บาท"
        result = PracticalPersonaService._find_arithmetic_mismatch(answer)
        assert result is not None
        desc, _ = result
        assert "400" in desc

    def test_accepts_correct_multiplication(self):
        answer = "อาคารเกิน 3 ชั้น คิดอัตรา 4 บาทต่อตารางเมตร: 4×100 = 400 บาท"
        assert PracticalPersonaService._find_arithmetic_mismatch(answer) is None

    def test_no_equation_returns_none(self):
        answer = "ค่าธรรมเนียมรายปีคือ 550 บาทครับ ไม่มีค่าใช้จ่ายเพิ่มเติม"
        assert PracticalPersonaService._find_arithmetic_mismatch(answer) is None

    def test_empty_answer_returns_none(self):
        assert PracticalPersonaService._find_arithmetic_mismatch("") is None

    def test_comma_thousands_separator(self):
        answer = "ค่าปรับ: 1,000 + 500 = 1,500 บาท"
        assert PracticalPersonaService._find_arithmetic_mismatch(answer) is None

    def test_comma_thousands_separator_catches_wrong_sum(self):
        answer = "ค่าปรับ: 1,000 + 500 = 2,000 บาท"
        result = PracticalPersonaService._find_arithmetic_mismatch(answer)
        assert result is not None

    def test_extra_whitespace_around_operator(self):
        answer = "รวม:   200   +   350   =   550   บาท"
        assert PracticalPersonaService._find_arithmetic_mismatch(answer) is None

    def test_subtraction_shape(self):
        answer = "เหลือหลังหักส่วนลด: 1000 - 200 = 800 บาท"
        assert PracticalPersonaService._find_arithmetic_mismatch(answer) is None
        bad = "เหลือหลังหักส่วนลด: 1000 - 200 = 900 บาท"
        assert PracticalPersonaService._find_arithmetic_mismatch(bad) is not None

    def test_returns_first_mismatch_only(self):
        # Two equations, first one wrong — must report the first, not the second.
        answer = "ขั้นที่ 1: 10 + 10 = 30 บาท\nขั้นที่ 2: 5 + 5 = 10 บาท"
        result = PracticalPersonaService._find_arithmetic_mismatch(answer)
        assert result is not None
        desc, _ = result
        assert "30" in desc


@pytest.mark.unit
class TestConflictingBoldAmounts:
    """2026-11, live-caught during this exact fix's own verification: a retry
    can correct the inline EQUATION while leaving an old wrong bolded amount
    sitting elsewhere in the answer (e.g. a headline), producing a
    self-contradictory answer that states two different numbers as final.
    Equations alone can't catch this — a bare "**650 บาท**" isn't an
    "A op B = C" shape at all."""

    def test_conflicting_bold_amounts_detected(self):
        answer = (
            "ค่าธรรมเนียมคิดได้ **650 บาท**\n\n"
            "สูตร: 200 + 350 = 550 บาท\n\n"
            "ขออภัยครับ ผมคำนวณใหม่: **550 บาท** ครับ"
        )
        result = PracticalPersonaService._find_arithmetic_mismatch(answer)
        assert result is not None
        desc, match = result
        assert "650" in desc and "550" in desc
        assert match is None, "no single equation span to excise for this failure shape"

    def test_single_bold_amount_not_flagged(self):
        answer = "ค่าธรรมเนียมหนังสือรับรองการแจ้งรายปีคิดเป็น **550 บาท** ครับ"
        assert PracticalPersonaService._find_arithmetic_mismatch(answer) is None

    def test_same_bold_amount_repeated_not_flagged(self):
        answer = (
            "ค่าธรรมเนียมคิดได้ **550 บาท**\n\n"
            "200 + 350 = 550 บาท\n\n"
            "สรุป: **550 บาท** ครับ"
        )
        assert PracticalPersonaService._find_arithmetic_mismatch(answer) is None


@pytest.mark.unit
class TestMissingBaseFee:
    """2026-11, live-caught during an A/B temperature test of this exact fix:
    a draft can get the tier-increment equation internally correct and still
    give the wrong final answer because it never added the base fee at all —
    "(45-10)×10 = 350 บาท" then "ต้องชำระ 350 บาทต่อปี" (should be 550).
    Equations alone can't catch this since 350 really does equal (45-10)×10;
    the gap is a missing combining step, not a wrong calculation."""

    def test_missing_base_fee_detected(self):
        answer = (
            "ค่าธรรมเนียมฐาน (ไม่เกิน 10 ตร.ม.): 200 บาท\n\n"
            "พื้นที่เกิน 10 ตร.ม.: (45 − 10) × 10 = 350 บาท\n\n"
            "ดังนั้น ต้องชำระ **350 บาทต่อปี** ครับ"
        )
        result = PracticalPersonaService._find_arithmetic_mismatch(answer)
        assert result is not None
        desc, match = result
        assert "350" in desc and "550" in desc and "200" in desc
        assert match is None, "no single equation span to excise for this failure shape"

    def test_base_fee_correctly_combined_not_flagged(self):
        answer = (
            "ค่าธรรมเนียมฐาน (ไม่เกิน 10 ตร.ม.): 200 บาท\n\n"
            "พื้นที่เกิน 10 ตร.ม.: (45 − 10) × 10 = 350 บาท\n\n"
            "รวม: 200 + 350 = 550 บาท\n\n"
            "ดังนั้น ต้องชำระ **550 บาทต่อปี** ครับ"
        )
        assert PracticalPersonaService._find_arithmetic_mismatch(answer) is None

    def test_no_base_fee_mentioned_not_flagged(self):
        # Different-unit tier shape (rate × full_value) has no base to omit.
        answer = "อาคารเกิน 3 ชั้น คิดอัตรา 4 บาทต่อตารางเมตร: 4×100 = **400 บาท**"
        assert PracticalPersonaService._find_arithmetic_mismatch(answer) is None

    def test_no_bold_amount_at_all_not_flagged(self):
        # Narrower scope by design — this check only fires when there's a single
        # clear bold final amount to compare against; no false positive otherwise.
        answer = (
            "ค่าธรรมเนียมฐาน (ไม่เกิน 10 ตร.ม.): 200 บาท\n\n"
            "พื้นที่เกิน 10 ตร.ม.: (45 − 10) × 10 = 350 บาท\n\n"
            "ดังนั้น ต้องชำระ 350 บาทต่อปี ครับ"
        )
        assert PracticalPersonaService._find_arithmetic_mismatch(answer) is None

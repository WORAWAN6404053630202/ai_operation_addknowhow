"""
Regression test for the 2026-11 fix porting Practical's FEE ARITHMETIC
safeguard to Academic. Practical's prompt got this after a live-reproduced
bug (tiered fee answered as 350 instead of the correct 550, base fee
dropped) — Academic answers the same tiered-fee documents but never
received the equivalent instruction. Source-text pin, matching this
codebase's established style for prompt-wording assertions (see
test_hardcoded_logic_audit_tier3_4.py's TestInfoGapPromptScopeClarified).
"""
from __future__ import annotations

import pytest

from utils.prompts_academic import SYSTEM_PROMPT


@pytest.mark.unit
class TestAcademicFeeArithmeticSafeguard:
    def test_fee_arithmetic_instruction_present_after_area_filter(self):
        idx = SYSTEM_PROMPT.find("AREA FILTER")
        assert idx != -1
        following = SYSTEM_PROMPT[idx:idx + 1500]
        assert "FEE ARITHMETIC" in following

    def test_covers_same_unit_and_different_unit_shapes(self):
        idx = SYSTEM_PROMPT.find("FEE ARITHMETIC")
        assert idx != -1
        following = SYSTEM_PROMPT[idx:idx + 1200]
        assert "SAME unit" in following
        assert "DIFFERENT unit" in following

    def test_worked_example_matches_practical_for_consistency(self):
        idx = SYSTEM_PROMPT.find("FEE ARITHMETIC")
        assert idx != -1
        following = SYSTEM_PROMPT[idx:idx + 1200]
        assert "200 + (45−10)×10 = 200 + 350 = 550" in following
        assert "NOT 650" in following
        assert "NOT 350 alone" in following

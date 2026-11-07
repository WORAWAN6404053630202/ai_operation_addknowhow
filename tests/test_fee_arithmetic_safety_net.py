"""2026-11 fix: source-inspection regression tests for
PracticalPersonaService._apply_fee_arithmetic_safety_net and its wiring into
handle(). persona_practical.py's handle() is a single ~5000-line method —
see test_doc_overflow_ceiling_fixes.py's module docstring for why source
inspection (not a full mocked end-to-end call) is this codebase's
established pattern for regression-testing logic embedded this deeply.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model.persona_practical import PracticalPersonaService  # noqa: E402


@pytest.mark.unit
class TestFeeArithmeticSafetyNetExistsAndWired:
    def test_safety_net_method_exists(self):
        assert hasattr(PracticalPersonaService, "_apply_fee_arithmetic_safety_net")

    def test_verifier_method_exists(self):
        assert hasattr(PracticalPersonaService, "_find_arithmetic_mismatch")

    def test_called_in_handle_after_groundedness_safety_net(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        g_idx = source.find("self._apply_groundedness_safety_net(")
        a_idx = source.find("self._apply_fee_arithmetic_safety_net(")
        assert g_idx != -1, "_apply_groundedness_safety_net call site not found in handle()"
        assert a_idx != -1, "_apply_fee_arithmetic_safety_net call site not found in handle()"
        assert a_idx > g_idx, (
            "Arithmetic safety net must run AFTER groundedness — groundedness can "
            "rewrite exec_['answer'] wholesale on its own retry, so arithmetic needs "
            "to inspect whatever text is final, not a version that might be replaced."
        )

    def test_still_wrong_fallback_excises_to_info_gap(self):
        source = inspect.getsource(PracticalPersonaService._apply_fee_arithmetic_safety_net)
        assert "[[INFO_GAP]]" in source, (
            "The still-wrong-after-retry fallback must excise the bad equation to "
            "[[INFO_GAP]] rather than silently keeping a confidently wrong number "
            "(unlike groundedness's 'keep original if still flagged', a numeric "
            "mismatch is Python ground truth with no false-positive risk)."
        )

    def test_safety_net_guards_on_action_and_empty_answer(self):
        source = inspect.getsource(PracticalPersonaService._apply_fee_arithmetic_safety_net)
        idx = source.find("if action != \"answer\"")
        assert idx != -1
        assert idx < source.find("self._find_arithmetic_mismatch(_answer)")

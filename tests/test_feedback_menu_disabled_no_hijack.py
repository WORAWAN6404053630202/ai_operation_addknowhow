"""Tests for the 2026-10 fix (core2-M2) to persona_supervisor.py's slot-
mismatch feedback-offer path.

Background: _FEEDBACK_MENU_ENABLED is False (feedback->Google Sheets menu
disabled since 2026-08-11), and the dispatcher that routes a reply to it
(in _handle_inner, guarded at "1.9) Pending feedback choice") was already
correctly gated on that flag. But _build_slot_mismatch_message appended
the feedback-menu text UNCONDITIONALLY, and its caller set
state.context["pending_feedback_choice"] = True UNCONDITIONALLY too —
neither checked the flag. Since pending_slot is deliberately left armed
in this branch, a user replying "1" (meaning the first feedback option,
which is now unroutable) fell through to whatever pending_slot handling
ran later and got silently answered as if "1" meant the FIRST OPTION OF
THE ORIGINAL SLOT QUESTION — a full wrong answer with no indication the
input was reinterpreted.

_build_slot_mismatch_message is a pure, easily-isolated method — tested
directly here. The call site inside _handle_inner is deep within an
~14,000-line method not practical to drive end-to-end in a unit test; its
fix is covered by a source-level regression check instead (same approach
already used elsewhere in this suite for structural invariants).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model.persona_supervisor import PersonaSupervisor  # noqa: E402


def _supervisor():
    return PersonaSupervisor.__new__(PersonaSupervisor)


@pytest.mark.unit
class TestBuildSlotMismatchMessageRespectsFeedbackMenuFlag:
    _PENDING = {"options": ["ธนาคารกรุงไทย", "ธนาคารกสิกรไทย"], "display_options": []}

    def test_feedback_menu_text_omitted_when_disabled(self):
        sup = _supervisor()
        with patch.object(PersonaSupervisor, "_FEEDBACK_MENU_ENABLED", False):
            msg = sup._build_slot_mismatch_message(self._PENDING, "ธนาคารทหารไทย")
        assert sup._FEEDBACK_MENU_TEXT not in msg
        assert "ธนาคารทหารไทย" in msg

    def test_feedback_menu_text_included_when_enabled(self):
        sup = _supervisor()
        with patch.object(PersonaSupervisor, "_FEEDBACK_MENU_ENABLED", True):
            msg = sup._build_slot_mismatch_message(self._PENDING, "ธนาคารทหารไทย")
        assert sup._FEEDBACK_MENU_TEXT in msg


@pytest.mark.unit
class TestSlotMismatchCallSiteStructure:
    """Source-level regression check for the call site inside _handle_inner
    (not practically unit-testable in isolation — see module docstring):
    pending_feedback_choice must only be armed inside an
    `if self._FEEDBACK_MENU_ENABLED:` guard, with an else branch that
    re-surfaces the still-armed pending_slot question instead."""

    def _call_site_block(self) -> str:
        import inspect
        src = inspect.getsource(PersonaSupervisor._route_pending_slot_to_persona) \
            if hasattr(PersonaSupervisor, "_route_pending_slot_to_persona") else None
        if src is None:
            src = inspect.getsource(PersonaSupervisor)
        idx = src.index("pending_slot_mismatch_feedback_offer")
        return src[max(0, idx - 700):idx + 100]

    def test_pending_feedback_choice_set_only_when_enabled(self):
        block = self._call_site_block()
        assert "if self._FEEDBACK_MENU_ENABLED:" in block
        assert 'state.context["pending_feedback_choice"] = True' in block

    def test_else_branch_reminds_pending_slot_instead(self):
        block = self._call_site_block()
        assert "_append_pending_slot_reminder" in block

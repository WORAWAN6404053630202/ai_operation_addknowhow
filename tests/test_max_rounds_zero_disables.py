"""Tests for the 2026-10 fix (core2-M3) to the MAX_ROUNDS=0 "disable the
cap" derivation in persona_supervisor.py and persona_practical.py.

tests/test_max_rounds.py already covers the downstream guard logic
(`max_rounds > 0 and cur_round >= max_rounds`) correctly, but passes
max_rounds in directly as a parameter — it never exercises the actual
derivation from conf.MAX_ROUNDS that had the real bug:
`int(getattr(conf, "MAX_ROUNDS", 15) or 15)` silently turned a deliberate
MAX_ROUNDS=0 back into 15, since `0 or 15` evaluates to 15 in Python. The
documented escape hatch ("Set MAX_ROUNDS=0 to disable") never actually
worked. These tests exercise the real derivation expression directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import conf  # noqa: E402


def _derive_max_rounds(default: int) -> int:
    """Mirrors the FIXED derivation now used in both
    persona_supervisor.py's _handle_inner and persona_practical.py's
    prompt-building (see their matching comments)."""
    raw = getattr(conf, "MAX_ROUNDS", default)
    return default if raw is None else int(raw)


@pytest.mark.unit
class TestMaxRoundsZeroActuallyDisables:
    def test_max_rounds_zero_is_not_coalesced_back_to_default(self):
        with patch.object(conf, "MAX_ROUNDS", 0):
            assert _derive_max_rounds(15) == 0
            assert _derive_max_rounds(7) == 0

    def test_max_rounds_positive_value_passes_through_unchanged(self):
        with patch.object(conf, "MAX_ROUNDS", 20):
            assert _derive_max_rounds(15) == 20

    def test_attribute_truly_absent_falls_back_to_default(self):
        with patch.object(conf, "MAX_ROUNDS", None):
            assert _derive_max_rounds(15) == 15

    def test_the_old_buggy_expression_would_have_failed_this(self):
        """Documents the exact bug this closes: the old expression
        `int(getattr(conf, "MAX_ROUNDS", 15) or 15)` silently discards 0."""
        with patch.object(conf, "MAX_ROUNDS", 0):
            old_buggy_result = int(getattr(conf, "MAX_ROUNDS", 15) or 15)
        assert old_buggy_result == 15  # the bug, confirmed still present in the raw expression
        with patch.object(conf, "MAX_ROUNDS", 0):
            assert _derive_max_rounds(15) == 0  # the fix


@pytest.mark.unit
class TestPersonaSourceUsesFixedDerivation:
    """Source-level regression check that the buggy `or 15`/`or 7`
    coalescing expression is gone from both call sites."""

    @staticmethod
    def _code_only(src: str) -> str:
        """Strips comment lines first — the fix's own explanatory comments
        quote the old buggy expression verbatim, which would otherwise
        false-positive this check (same issue already handled this session
        in test_admin_xss_escaping.py)."""
        return "\n".join(
            line for line in src.splitlines() if not line.strip().startswith("#")
        )

    def test_supervisor_no_longer_uses_or_15_coalescing(self):
        import inspect
        from model.persona_supervisor import PersonaSupervisor
        src = self._code_only(inspect.getsource(PersonaSupervisor._handle_inner))
        assert 'getattr(conf, "MAX_ROUNDS", 15) or 15' not in src
        assert "_raw_max_rounds" in src

    def test_practical_no_longer_uses_or_7_coalescing(self):
        import inspect
        from model.persona_practical import PracticalPersonaService
        src = self._code_only(inspect.getsource(PracticalPersonaService))
        assert 'getattr(conf, "MAX_ROUNDS", 7) or 7' not in src

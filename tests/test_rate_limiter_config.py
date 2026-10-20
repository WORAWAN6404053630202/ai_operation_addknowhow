"""Tests for the 2026-10 fix (core2-N2) to utils/rate_limiter.py's
get_rate_limiter() — used to hardcode RateLimiter(max_requests=10,
window_seconds=60) with no conf lookup at all, and conf.py's own comment
referenced a RATE_LIMIT_WINDOW_SECONDS constant that was defined nowhere
in the repo. Now reads conf.RATE_LIMIT_MAX_REQUESTS/RATE_LIMIT_WINDOW_SECONDS
(same defaults as before, so default behavior is unchanged).
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
import utils.rate_limiter as rate_limiter_module  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_global_limiter():
    rate_limiter_module._global_limiter = None
    yield
    rate_limiter_module._global_limiter = None


@pytest.mark.unit
class TestGetRateLimiterReadsConf:
    def test_conf_constants_exist_with_original_hardcoded_defaults(self):
        assert conf.RATE_LIMIT_MAX_REQUESTS == 10
        assert conf.RATE_LIMIT_WINDOW_SECONDS == 60

    def test_default_behavior_unchanged(self):
        limiter = rate_limiter_module.get_rate_limiter()
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 60

    def test_operator_can_tighten_the_limit_via_conf(self):
        with patch.object(conf, "RATE_LIMIT_MAX_REQUESTS", 3), \
             patch.object(conf, "RATE_LIMIT_WINDOW_SECONDS", 30):
            limiter = rate_limiter_module.get_rate_limiter()
        assert limiter.max_requests == 3
        assert limiter.window_seconds == 30

    def test_singleton_still_returns_the_same_instance(self):
        a = rate_limiter_module.get_rate_limiter()
        b = rate_limiter_module.get_rate_limiter()
        assert a is b

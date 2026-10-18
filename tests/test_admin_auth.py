"""Tests for utils/admin_auth.py's require_admin_key — the shared auth
dependency for router/admin.py and router/monitoring.py's debug routes.

Added 2026-09 alongside ADMIN_AUTH_DISABLED, an explicit opt-in escape hatch
built for the internal-team PDF test instance (wanted the admin dashboard
reachable with no key, for quickly eyeballing conversations). Confirms the
secure default is unchanged and the bypass only activates when deliberately
set.

require_admin_key is `async def` but never actually awaits anything inside
— run via asyncio.run() from plain sync test functions rather than adding
pytest-asyncio markers, since no other test in this suite uses those."""
import asyncio
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import conf
from utils.admin_auth import require_admin_key


def _call(key: str):
    asyncio.run(require_admin_key(x_admin_key=key))


@pytest.mark.unit
class TestRequireAdminKey:

    def test_correct_key_passes(self):
        with patch.object(conf, "ADMIN_API_KEY", "real-key"), patch.object(conf, "ADMIN_AUTH_DISABLED", False):
            _call("real-key")  # must not raise

    def test_wrong_key_raises_401(self):
        with patch.object(conf, "ADMIN_API_KEY", "real-key"), patch.object(conf, "ADMIN_AUTH_DISABLED", False):
            with pytest.raises(HTTPException) as exc_info:
                _call("wrong-key")
        assert exc_info.value.status_code == 401

    def test_missing_key_raises_401(self):
        with patch.object(conf, "ADMIN_API_KEY", "real-key"), patch.object(conf, "ADMIN_AUTH_DISABLED", False):
            with pytest.raises(HTTPException):
                _call("")

    def test_unset_admin_api_key_fails_closed_not_open(self):
        """The established fail-closed rule (predates ADMIN_AUTH_DISABLED):
        an unset key must lock the endpoint down, not open it up."""
        with patch.object(conf, "ADMIN_API_KEY", ""), patch.object(conf, "ADMIN_AUTH_DISABLED", False):
            with pytest.raises(HTTPException):
                _call("")

    def test_auth_disabled_bypasses_check_entirely(self):
        with patch.object(conf, "ADMIN_AUTH_DISABLED", True):
            _call("")  # no key at all, must not raise

    def test_auth_disabled_bypasses_even_with_a_configured_key(self):
        """Disabling auth is an intentional full bypass, not merely 'accept
        any key' — confirms it short-circuits before the key comparison."""
        with patch.object(conf, "ADMIN_API_KEY", "real-key"), patch.object(conf, "ADMIN_AUTH_DISABLED", True):
            _call("totally-wrong")  # must not raise

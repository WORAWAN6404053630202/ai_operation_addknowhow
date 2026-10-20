"""Test for the 2026-10 fix to utils/logger.py's setup_logging() — live-
reproduced during a routine PDF-consumer deploy check: at DEBUG level (the
LOG_LEVEL every env file in this repo sets), botocore.auth logs the full
StringToSign/Signature and botocore.endpoint logs the complete outgoing
request including the Authorization header and x-amz-security-token —
real AWS credentials, verbatim, to stdout and therefore to `docker logs`.
setup_logging() already silenced httpx/httpcore/openai to WARNING
regardless of the configured level; botocore/boto3/urllib3 were missing
from that list. This pins that they're now included.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils.logger import setup_logging  # noqa: E402


@pytest.mark.unit
class TestNoisyLibrariesSilencedRegardlessOfLevel:
    @pytest.mark.parametrize("lib", ["httpx", "httpcore", "openai", "botocore", "boto3", "urllib3"])
    def test_silenced_to_warning_even_at_debug_level(self, lib):
        setup_logging(level="DEBUG")
        assert logging.getLogger(lib).level == logging.WARNING

    def test_botocore_child_loggers_inherit_the_warning_ceiling(self):
        """botocore.auth/botocore.endpoint (the actual sites that logged
        credentials) don't set their own level — they must inherit the
        parent "botocore" logger's WARNING level via the standard
        effective-level lookup, same mechanism httpx's children already
        relied on."""
        setup_logging(level="DEBUG")
        assert logging.getLogger("botocore.auth").getEffectiveLevel() == logging.WARNING
        assert logging.getLogger("botocore.endpoint").getEffectiveLevel() == logging.WARNING

    def test_root_logger_itself_still_follows_the_configured_level(self):
        """The fix must not accidentally mute the application's own DEBUG
        logging — only the specific noisy third-party libraries."""
        setup_logging(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG
        assert logging.getLogger("some_app_module").getEffectiveLevel() == logging.DEBUG

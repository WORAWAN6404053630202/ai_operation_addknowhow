"""Test for the 2026-10 fix (PDF2-N7) to service/sqs_consumer.py's
_assert_dev_environment() training-wheel.

Previously this only ever checked conf.ENV_FILE_NAME == "env.properties",
inferring "safe to run" purely from the env filename — meaning the real,
reviewed, deployed pure-PDF production consumer could only ever run under
a config file misleadingly named like a dev config. Added an explicit
opt-in escape hatch, conf.PDF_CONSUMER_ACKNOWLEDGED_PRODUCTION, that
bypasses the filename check entirely when set. Defaults False, so every
existing deployment (relying on a dev-named file) is unaffected.
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
from service.sqs_consumer import _assert_dev_environment  # noqa: E402


@pytest.mark.unit
class TestAssertDevEnvironment:
    def test_raises_on_real_env_file_by_default(self):
        with patch.object(conf, "ENV_FILE_NAME", "env.properties"), \
             patch.object(conf, "PDF_CONSUMER_ACKNOWLEDGED_PRODUCTION", False):
            with pytest.raises(RuntimeError):
                _assert_dev_environment()

    def test_does_not_raise_on_dev_env_file(self):
        with patch.object(conf, "ENV_FILE_NAME", "env.dev.properties"), \
             patch.object(conf, "PDF_CONSUMER_ACKNOWLEDGED_PRODUCTION", False):
            _assert_dev_environment()  # must not raise

    def test_opt_in_flag_bypasses_the_filename_check(self):
        with patch.object(conf, "ENV_FILE_NAME", "env.properties"), \
             patch.object(conf, "PDF_CONSUMER_ACKNOWLEDGED_PRODUCTION", True):
            _assert_dev_environment()  # must not raise, even on the real file

    def test_opt_in_flag_defaults_false(self):
        assert conf.PDF_CONSUMER_ACKNOWLEDGED_PRODUCTION is False

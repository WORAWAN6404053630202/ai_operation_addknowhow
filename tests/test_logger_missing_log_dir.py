"""Tests for the 2026-10 fix (Infra2-C2) to utils/logger.py's
setup_logging() — a missing log_file parent directory used to make
RotatingFileHandler raise FileNotFoundError here, crashing the whole app
before it could even start (found on the lean prod-pdf Docker stage, which
never created /app/logs even though env.properties's global LOG_FILE
setting pointed there). setup_logging() now creates the directory
best-effort and falls back to console-only logging on any remaining
OSError rather than ever being a hard startup crash.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
import uuid
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils.logger import setup_logging  # noqa: E402


def _reset_root_logger():
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


@pytest.mark.unit
class TestMissingLogDirectoryDoesNotCrash:
    def test_creates_missing_parent_directory(self, tmp_path):
        _reset_root_logger()
        log_file = tmp_path / "nested" / "does" / "not" / "exist" / "app.log"
        assert not log_file.parent.exists()
        setup_logging(level="INFO", log_file=str(log_file))
        assert log_file.parent.exists()
        logging.getLogger(__name__).warning("probe")
        assert log_file.exists()

    def test_falls_back_to_console_only_when_directory_cannot_be_created(self, tmp_path):
        """A path that can never be a valid directory (e.g. a parent
        component that's actually a file) must degrade to console-only
        logging, not raise out of setup_logging()."""
        _reset_root_logger()
        blocking_file = tmp_path / f"not-a-dir-{uuid.uuid4().hex}"
        blocking_file.write_text("x")
        bad_log_file = blocking_file / "app.log"  # parent ("blocking_file") is a file, not a dir

        setup_logging(level="INFO", log_file=str(bad_log_file))  # must not raise

        root = logging.getLogger()
        assert not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers)
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

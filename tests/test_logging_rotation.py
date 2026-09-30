"""
Regression tests for utils.logger.setup_logging's file-rotation behavior
(2026-09).

Background: a real production incident (a customer's request failed once,
succeeded on retry) turned out to be undiagnosable after the fact — the app
only logged to stdout, which Docker discards whenever the container is
recreated (`docker stop && docker rm && docker run`, the normal redeploy
sequence used for this project). `setup_logging` already had a `log_file`
parameter, but wired it to a plain `logging.FileHandler` with no size cap at
all — enabling it as-is would have just traded "logs disappear on redeploy"
for "logs grow forever and eventually fill the disk" on a small, cost-
sensitive EC2 instance.

Fixed by switching to `RotatingFileHandler` with a small, deliberate ceiling
(default 5MB x 5 backups = ~30MB) and a separate, higher level threshold for
the file handler specifically (WARNING, vs. the console's DEBUG in
production) — real errors/warnings are what an incident investigation needs,
and skipping routine DEBUG/INFO noise means the same disk budget spans weeks
of real history instead of hours.

No network access, no LLM calls — pure logging-module behavior, using
tmp_path for real (throwaway) file I/O.
"""
import glob
import logging
import logging.handlers
import os
import sys

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))

import pytest

from utils.logger import setup_logging


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """setup_logging mutates the ROOT logger (global state) — every test here
    both starts and ends with a clean slate so tests can't leak handlers into
    each other or into the rest of the suite."""
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        h.close()
        root.removeHandler(h)
    root.setLevel(logging.WARNING)


@pytest.mark.unit
class TestLogFileRotation:

    def test_file_handler_is_rotating_not_plain(self, tmp_path):
        log_path = str(tmp_path / "app.log")
        setup_logging(level="DEBUG", log_file=log_path)
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert isinstance(file_handlers[0], logging.handlers.RotatingFileHandler)

    def test_file_handler_level_defaults_to_warning_independent_of_console(self, tmp_path):
        log_path = str(tmp_path / "app.log")
        setup_logging(level="DEBUG", log_file=log_path)
        root = logging.getLogger()
        console = next(h for h in root.handlers if not isinstance(h, logging.FileHandler))
        file_h = next(h for h in root.handlers if isinstance(h, logging.FileHandler))
        assert console.level == logging.NOTSET  # inherits root's DEBUG
        assert file_h.level == logging.WARNING

    def test_debug_and_info_never_reach_the_file(self, tmp_path):
        log_path = tmp_path / "app.log"
        setup_logging(level="DEBUG", log_file=str(log_path))
        logger = logging.getLogger("test.rotation")
        logger.debug("debug noise")
        logger.info("info noise")
        logger.warning("a real warning")
        for h in logging.getLogger().handlers:
            h.flush()
        content = log_path.read_text(encoding="utf-8")
        assert "debug noise" not in content
        assert "info noise" not in content
        assert "a real warning" in content

    def test_rotation_actually_caps_total_disk_usage(self, tmp_path):
        log_path = tmp_path / "app.log"
        # Small ceiling so the test can force multiple rotations quickly.
        setup_logging(
            level="WARNING",
            log_file=str(log_path),
            log_file_max_bytes=2000,
            log_file_backup_count=3,
        )
        logger = logging.getLogger("test.rotation.cap")
        # Each warning is padded to comfortably exceed the 2000-byte ceiling
        # after a handful of writes, forcing several rotations.
        for i in range(200):
            logger.warning("padding line %d " + ("x" * 80), i)
        for h in logging.getLogger().handlers:
            h.flush()

        rotated_files = glob.glob(str(tmp_path / "app.log*"))
        # backupCount=3 means: the active file + at most 3 rotated backups = 4 total.
        assert len(rotated_files) <= 4
        assert len(rotated_files) >= 2  # confirms rotation actually happened, not a no-op

    def test_no_log_file_means_no_file_handler(self, tmp_path):
        setup_logging(level="INFO", log_file=None)
        root = logging.getLogger()
        assert not any(isinstance(h, logging.FileHandler) for h in root.handlers)

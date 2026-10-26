"""Tests for scripts/pdf_delete_by_source.py's TOCTOU mismatch warning —
2026-10 fix, data-integrity audit.

Background: the dry-run preview count (`n`, from list_pdf_ingested_files())
and the actual delete (`deleted`, from delete_by_source_file()) are two
SEPARATE calls — often two separate CLI invocations entirely, since a human
reads the dry-run output and re-runs with --confirm minutes later. A
concurrent SQS ingest adding more pdf_auto docs for that exact source_file
in between would get silently swept into the delete without ever having
appeared in what the operator previewed. TOCTOU research is consistent
that eliminating the gap is the real fix, which isn't possible across two
independent human-triggered runs — so this surfaces the mismatch loudly
instead of staying silent about it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import scripts.pdf_delete_by_source as delete_script  # noqa: E402


def _run_with_args(args: list[str]):
    with patch.object(sys, "argv", ["pdf_delete_by_source.py"] + args):
        delete_script.main()


@pytest.mark.unit
class TestToctouMismatchWarning:
    def test_matching_counts_print_no_warning(self, capsys):
        mock_mgr = MagicMock()
        mock_mgr.list_pdf_ingested_files.return_value = {"f.pdf": 5}
        mock_mgr.delete_by_source_file.return_value = 5
        with patch.object(delete_script, "get_vs_manager", return_value=mock_mgr):
            _run_with_args(["--delete", "f.pdf", "--confirm"])
        out = capsys.readouterr().out
        assert "NOTE: deleted count" not in out

    def test_deleted_more_than_previewed_prints_a_warning(self, capsys):
        """The exact scenario: a concurrent ingest added more docs for this
        source_file between the dry-run preview and this --confirm run."""
        mock_mgr = MagicMock()
        mock_mgr.list_pdf_ingested_files.return_value = {"f.pdf": 5}
        mock_mgr.delete_by_source_file.return_value = 8
        with patch.object(delete_script, "get_vs_manager", return_value=mock_mgr):
            _run_with_args(["--delete", "f.pdf", "--confirm"])
        out = capsys.readouterr().out
        assert "NOTE: deleted count (8) differs from the dry-run preview (5)" in out

    def test_deleted_fewer_than_previewed_also_warns(self, capsys):
        """The opposite direction (e.g. another process already deleted
        some in between) should also be flagged, not just the 'more'
        direction."""
        mock_mgr = MagicMock()
        mock_mgr.list_pdf_ingested_files.return_value = {"f.pdf": 5}
        mock_mgr.delete_by_source_file.return_value = 2
        with patch.object(delete_script, "get_vs_manager", return_value=mock_mgr):
            _run_with_args(["--delete", "f.pdf", "--confirm"])
        out = capsys.readouterr().out
        assert "NOTE: deleted count (2) differs from the dry-run preview (5)" in out

    def test_dry_run_without_confirm_never_calls_delete(self, capsys):
        mock_mgr = MagicMock()
        mock_mgr.list_pdf_ingested_files.return_value = {"f.pdf": 5}
        with patch.object(delete_script, "get_vs_manager", return_value=mock_mgr):
            _run_with_args(["--delete", "f.pdf"])
        mock_mgr.delete_by_source_file.assert_not_called()

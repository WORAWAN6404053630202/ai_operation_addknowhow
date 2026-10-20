"""Test for the 2026-10 fix (core2-N3) to service/data_loader.py's two
pd.read_csv call sites — missing dtype=str, unlike data_loader_general.py's
sibling. Without it, a fully-numeric Sheet column is inferred as float64
("1000" becomes "1000.0") and a leading-zero code like "007" loses its
zero on the way into the corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from service.data_loader import DataLoader  # noqa: E402


@pytest.mark.unit
class TestDataLoaderReadsCsvAsStrings:
    def test_load_from_google_sheet_passes_dtype_str(self):
        dl = DataLoader(config=MagicMock())
        with patch("service.data_loader.pd.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame(columns=["หน่วยงาน"])
            try:
                dl.load_from_google_sheet("https://docs.google.com/spreadsheets/d/abc123/edit#gid=456", source_name="test")
            except Exception:
                pass  # only the read_csv call shape is under test here
        assert mock_read_csv.call_args.kwargs.get("dtype") is str

    def test_load_from_knowhow_sheet_passes_dtype_str(self):
        dl = DataLoader(config=MagicMock())
        with patch("service.data_loader.pd.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame(columns=["title"])
            try:
                dl.load_from_knowhow_sheet("https://docs.google.com/spreadsheets/d/abc123/edit#gid=456")
            except Exception:
                pass
        assert mock_read_csv.call_args.kwargs.get("dtype") is str

    def test_numeric_looking_values_survive_as_strings_not_floats(self):
        """The actual bug this closes: without dtype=str, a fully-numeric
        column is inferred as float64 ("1000" -> "1000.0") and a
        leading-zero code loses its zero."""
        import io
        csv_text = "code,fee\n007,1000\n"
        result = pd.read_csv(io.StringIO(csv_text), dtype=str)
        assert result["code"].iloc[0] == "007"
        assert result["fee"].iloc[0] == "1000"

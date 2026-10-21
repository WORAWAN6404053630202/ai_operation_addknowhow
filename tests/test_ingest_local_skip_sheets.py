"""Test for the 2026-10 INGEST_SKIP_SHEETS safety rail in scripts/ingest_local.py.

Added for a future genuinely PDF-only deployment (zero Sheet data by design) —
a mistaken `ingest_local.py --confirm` run (wrong terminal/directory, muscle
memory from the merged Sheet+PDF instance) must not be able to pull Sheet
content (Sources A-D) into a collection meant to stay PDF-only forever.
Source E (the PDF-content carry-forward step) must always run regardless —
it's the only thing that lets PDF content survive a reset=True rebuild.

main() is a top-level script function (argparse, real DataLoader/Sheet I/O,
a destructive ingest_documents(..., reset=True) call) not designed for
direct unit invocation — this codebase's established pattern for logic this
deeply embedded in a script entry point is source inspection (see
test_sibling_docs_cap_config.py and the handle()-based tests this same
session added), verifying the gating structure is actually present and
wired to the right flag, which still catches accidental reversion.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import conf  # noqa: E402
import scripts.ingest_local as ingest_local_mod  # noqa: E402


@pytest.mark.unit
class TestIngestSkipSheetsConfig:
    def test_conf_defines_the_flag(self):
        assert hasattr(conf, "INGEST_SKIP_SHEETS")

    def test_default_is_false(self):
        assert conf.INGEST_SKIP_SHEETS is False

    def test_env_var_actually_overrides_it(self):
        import importlib
        from unittest.mock import patch
        with patch.dict("os.environ", {"INGEST_SKIP_SHEETS": "true"}):
            importlib.reload(conf)
            assert conf.INGEST_SKIP_SHEETS is True
        importlib.reload(conf)  # restore real env for subsequent tests


@pytest.mark.unit
class TestIngestLocalGatesSheetsNotPdfContent:
    def test_sources_a_through_d_are_gated_on_the_flag(self):
        source = inspect.getsource(ingest_local_mod.main)
        assert "if not conf.INGEST_SKIP_SHEETS:" in source
        gate_idx = source.index("if not conf.INGEST_SKIP_SHEETS:")
        # Source E's own section header must come AFTER the gated block closes,
        # at the same (unindented-relative-to-the-if) top level. Search for the
        # section-header marker specifically (not the bare substring "Source E",
        # which also appears in this fix's own earlier echo-print wording).
        source_e_idx = source.index("# ── Source E:")
        assert source_e_idx > gate_idx
        gated_region = source[gate_idx:source_e_idx]
        for marker in ("load_from_google_sheet", "load_from_knowhow_sheet"):
            assert marker in gated_region, f"{marker} should be inside the gated Sheets block"

    def test_source_e_reference_not_inside_an_if_skip_sheets_block(self):
        # Belt-and-suspenders structural check: the text between the gate's
        # `if` and its matching `else:` must contain all 4 Sheet loader calls,
        # and the `else:` branch (skip message) must NOT reference any loader.
        source = inspect.getsource(ingest_local_mod.main)
        gate_idx = source.index("if not conf.INGEST_SKIP_SHEETS:")
        else_idx = source.index("\n    else:", gate_idx)
        gated_region = source[gate_idx:else_idx]
        assert gated_region.count("load_from_google_sheet") >= 3  # Sources A, B, C
        assert "load_from_knowhow_sheet" in gated_region  # Source D

"""Test for the 2026-10 fix (Infra2-L2): env.properties/env.dev.properties
both set SIBLING_DOCS_CAP, but conf.py never defined it, so
model/persona_practical.py's getattr(conf, "SIBLING_DOCS_CAP", 10) silently
fell back to the hardcoded default every time regardless of the env value.
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


@pytest.mark.unit
class TestSiblingDocsCapConfig:
    def test_conf_defines_sibling_docs_cap(self):
        assert hasattr(conf, "SIBLING_DOCS_CAP")

    def test_default_is_10(self):
        assert conf.SIBLING_DOCS_CAP == 10

    def test_env_var_actually_overrides_it(self):
        import importlib
        with patch.dict("os.environ", {"SIBLING_DOCS_CAP": "7"}):
            importlib.reload(conf)
            assert conf.SIBLING_DOCS_CAP == 7
        importlib.reload(conf)  # restore real env for subsequent tests

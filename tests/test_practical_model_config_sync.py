"""Regression test for the 2026-10 fix (Infra2-M13) — the Practical persona's
2026-09-27 Sonnet->Haiku model switch (measured ~65-68% cost reduction) was
recorded as deployed, but conf.py's own fallback default and all 3 env
files (env.properties, env.dev.properties, env.dev.properties.ec2) still
said Sonnet. A fresh rebuild/redeploy/local run with no env override would
silently revert to the 3x-more-expensive model with no error or log line,
since this repo contained no copy of config matching the actual deployed
decision. Pins all 4 to agree on Haiku going forward.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_CODE_DIR = _ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

_EXPECTED_MODEL = "anthropic/claude-haiku-4-5"


@pytest.mark.unit
class TestPracticalModelConfigAgreesEverywhere:
    def test_conf_py_default_is_haiku(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_MODEL_PRACTICAL", raising=False)
        # conf.py may already be imported (cached) by an earlier test in
        # this session with a different env var state — re-read the
        # source's os.getenv(...) call directly instead of relying on the
        # already-imported module object, which would reflect whatever
        # env var was set the first time anything imported conf.
        src = (_CODE_DIR / "conf.py").read_text(encoding="utf-8")
        match = re.search(
            r'OPENROUTER_MODEL_PRACTICAL\s*=\s*os\.getenv\("OPENROUTER_MODEL_PRACTICAL",\s*"([^"]+)"\)',
            src,
        )
        assert match, "Could not find OPENROUTER_MODEL_PRACTICAL's default in conf.py"
        assert match.group(1) == _EXPECTED_MODEL

    @pytest.mark.parametrize("env_file", [
        "env.properties", "env.dev.properties", "env.dev.properties.ec2",
    ])
    def test_env_file_sets_haiku(self, env_file):
        # These 3 files are gitignored (real/local secrets, never committed
        # — see .gitignore) — a clean checkout (including CI) won't have
        # them at all. Skip rather than fail when absent; this test's value
        # is catching drift on a machine that DOES have them (this one),
        # not asserting they exist.
        path = _ROOT / env_file
        if not path.exists():
            pytest.skip(f"{env_file} is gitignored and not present in this checkout")
        content = path.read_text(encoding="utf-8")
        match = re.search(r"^OPENROUTER_MODEL_PRACTICAL=(.+)$", content, re.MULTILINE)
        assert match, f"{env_file} has no OPENROUTER_MODEL_PRACTICAL line"
        assert match.group(1).strip() == _EXPECTED_MODEL

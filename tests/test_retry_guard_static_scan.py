"""
Unit tests for scripts/retry_guard_static_scan.py (.claude/plans/
mutable-seeking-wave.md — "Fix hidden SDK-level retry duplication across all
ChatOpenAI calls"). Unlike its precedent, filter_guard_static_scan.py (which
has no pytest wrapper and must be run manually), this one is wired into
`pytest -m unit` so the gate runs automatically on every test run, not only
when someone remembers to run the script by hand.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from scripts.retry_guard_static_scan import _scan_file


def _write(tmp_path: Path, content: str) -> str:
    p = tmp_path / "sample.py"
    p.write_text(textwrap.dedent(content))
    return str(p)


@pytest.mark.unit
class TestRetryGuardStaticScan:
    def test_flags_construction_site_with_no_max_retries(self, tmp_path):
        path = _write(tmp_path, """
            def foo():
                llm = ChatOpenAI(
                    model="x",
                    temperature=0.0,
                    request_timeout=8,
                )
                return llm
            """)
        findings = _scan_file(path)
        assert len(findings) == 1
        assert findings[0][0] == 3  # line number of "llm = ChatOpenAI("

    def test_does_not_flag_site_with_max_retries(self, tmp_path):
        path = _write(tmp_path, """
            def foo():
                llm = ChatOpenAI(
                    model="x",
                    temperature=0.0,
                    request_timeout=8,
                    max_retries=CHAT_OPENAI_MAX_RETRIES,
                )
                return llm
            """)
        assert _scan_file(path) == []

    def test_escape_hatch_marker_comment_suppresses_finding(self, tmp_path):
        path = _write(tmp_path, """
            def foo():
                # retry-guard: max-retries-intentionally-omitted
                llm = ChatOpenAI(
                    model="x",
                    temperature=0.0,
                    request_timeout=8,
                )
                return llm
            """)
        assert _scan_file(path) == []

    def test_does_not_self_flag_on_prose_mention(self, tmp_path):
        """A comment/docstring that mentions 'ChatOpenAI(' in prose (like
        this scanner's own module docstring, and utils/llm_call.py's) must
        not be mistaken for a real construction site — the regex requires
        an assignment/return shape immediately before it."""
        path = _write(tmp_path, '''
            """Every site builds its own ChatOpenAI(...) instance, see below."""
            def foo():
                pass
            ''')
        assert _scan_file(path) == []

    def test_real_repo_code_tree_has_zero_findings(self):
        """The actual regression gate: this must stay clean as new
        ChatOpenAI(...) sites get added to the real codebase."""
        import scripts.retry_guard_static_scan as scan_mod

        repo_root = Path(__file__).parent.parent
        code_root = repo_root / "code"
        all_findings = []
        for path in scan_mod._iter_source_files(str(code_root)):
            findings = scan_mod._scan_file(path)
            if findings:
                all_findings.append((path, findings))
        assert all_findings == [], f"unguarded ChatOpenAI(...) sites found: {all_findings}"

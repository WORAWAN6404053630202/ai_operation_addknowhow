"""
Static, read-only scan for ChatOpenAI(...) construction sites missing an
explicit `max_retries`.

Background (2026-11, retry-duplication audit): every one of this codebase's
50 ChatOpenAI(...) construction sites omitted `max_retries`, so each fell
through to the underlying openai SDK's own hidden default (max_retries=2,
i.e. 3 low-level HTTP attempts) — completely invisible to this codebase's
OWN deliberate retry layer, utils.llm_call.llm_invoke() (see that module's
_MAX_RETRIES/_RETRY_DELAYS). The two layers silently stacked: a call
configured with request_timeout=8 could take ~24s before llm_invoke even
saw its first failure. Live-confirmed on a groundedness judge (25.8s to
first failure, 32.5s total after one app-level retry) — see
.claude/plans/mutable-seeking-wave.md for the full incident writeup. Fixed
by adding `max_retries=utils.llm_call.CHAT_OPENAI_MAX_RETRIES` (value 0) at
every site, making llm_invoke() the single, observable retry authority.

This script does NOT re-check that fix. It exists to catch the NEXT
ChatOpenAI(...) site someone adds without this kwarg — unlike the
department/license_type filter-guard gap this scanner is modeled on
(code/scripts/filter_guard_static_scan.py, confined to 3 persona files), a
missing max_retries is equally dangerous wherever it appears — a new
script, a new service module, a new persona file — so this scanner walks
the WHOLE code/ tree rather than a fixed file list.

This is a heuristic text scan (regex + indentation matching), not an AST
analysis — it will have false positives/negatives on unusual code shapes.
A site already reviewed and confirmed intentional (e.g. test-harness code
that deliberately doesn't go through llm_invoke) should get a one-line
marker comment, `# retry-guard: max-retries-intentionally-omitted`, placed
anywhere within the call's own span, so this scanner doesn't re-flag it.

Usage: python code/scripts/retry_guard_static_scan.py [--files path1.py path2.py ...]
Exit code: 0 if no WARNs, 1 if any WARN found (suitable for a CI gate).
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_EXCLUDE_DIR_NAMES = {".git", "__pycache__", "venv", ".venv", "node_modules", "build", "dist", ".pytest_cache"}

_CONSTRUCTION_RE = re.compile(r'(?:=|return)\s*ChatOpenAI\(')
_MAX_RETRIES_RE = re.compile(r'\bmax_retries\s*=')
_ESCAPE_HATCH_RE = re.compile(r'#\s*retry-guard:\s*max-retries-intentionally-omitted')

WINDOW_LINES = 40


def _iter_source_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIR_NAMES]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _scan_file(path: str) -> list:
    """Returns a list of (line_no, snippet) for ChatOpenAI(...) sites with
    no max_retries kwarg anywhere in their own call span (or an escape-hatch
    marker comment nearby)."""
    # Never flag this scanner's own file — the construction-site regex and
    # docstring above both mention "ChatOpenAI(" in prose.
    if os.path.abspath(path) == os.path.abspath(__file__):
        return []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    findings = []
    for i, line in enumerate(lines):
        if not _CONSTRUCTION_RE.search(line):
            continue
        indent = len(line) - len(line.lstrip(" "))

        # Find the call's own closing ")" — this project's convention
        # (confirmed across all 50 known-good sites) is that it sits alone
        # on a line at the SAME indentation as the opening line.
        close_idx = None
        for j in range(i + 1, min(len(lines), i + WINDOW_LINES)):
            candidate = lines[j].rstrip("\n")
            if candidate.strip() == ")" and (len(candidate) - len(candidate.lstrip(" "))) == indent:
                close_idx = j
                break
        span_end = close_idx + 1 if close_idx is not None else min(len(lines), i + WINDOW_LINES)
        span = "".join(lines[i:span_end])

        if _MAX_RETRIES_RE.search(span):
            continue
        # Escape-hatch marker: allow it either inside the call span, or just
        # above the construction line (the natural place to annotate "this
        # one's deliberate" before the call itself, same convention as a
        # regular leading comment on any other statement).
        escape_hatch_start = max(0, i - 5)
        escape_hatch_span = "".join(lines[escape_hatch_start:span_end])
        if _ESCAPE_HATCH_RE.search(escape_hatch_span):
            continue

        findings.append((i + 1, line.strip()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="*", default=None, help="scan only these files instead of walking code/")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    code_root = os.path.join(repo_root, "code")

    if args.files:
        files = [os.path.join(repo_root, f) if not os.path.isabs(f) else f for f in args.files]
    else:
        files = sorted(_iter_source_files(code_root))

    total_findings = 0
    any_scanned = False
    for abs_path in files:
        if not os.path.isfile(abs_path):
            print(f"[WARN] file not found, skipping: {abs_path}")
            continue
        rel_path = os.path.relpath(abs_path, repo_root)
        findings = _scan_file(abs_path)
        if not findings:
            continue
        any_scanned = True
        total_findings += len(findings)
        print(f"[WARN] {rel_path} — {len(findings)} ChatOpenAI(...) site(s) missing max_retries:")
        for line_no, snippet in findings:
            print(f"       line {line_no}: {snippet}")

    print("-" * 72)
    if total_findings:
        print(
            f"{total_findings} ChatOpenAI(...) site(s) found with no explicit max_retries. "
            f"Add 'max_retries=utils.llm_call.CHAT_OPENAI_MAX_RETRIES' (import it from "
            f"utils.llm_call) — or, if this site deliberately doesn't go through llm_invoke "
            f"and that's intentional, add a '# retry-guard: max-retries-intentionally-omitted' "
            f"marker comment near the construction call so this scan doesn't re-flag it."
        )
        return 1
    if not any_scanned:
        print(f"Scanned {len(files)} file(s) under code/ — no ChatOpenAI(...) sites missing max_retries found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

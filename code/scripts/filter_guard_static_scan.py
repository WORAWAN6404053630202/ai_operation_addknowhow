"""
Static, read-only scan for unguarded department/license_type exact-match filters.

Background (2026-11, retrieval-safety audit): persona_supervisor.py had two (then a
third, found mid-fix) hardcoded-keyword-triggered filters on `department`/`license_type`
that committed to any non-empty result with no cross-check — a candidate doc with a
wrong or thin value could silently win. That gap was fixed by wiring in
utils.text_patterns.regulatory_identity_mismatch_should_reject (identity guard) and
PersonaSupervisor._is_structurally_adequate (coverage-adequacy supplement) — see
tests/test_license_filter_coverage_cascade.py for the full history.

This script does NOT re-check that fix. It exists to catch the NEXT occurrence of the
same shape — per project history, the keyword lists this gap lived in
(_MULTI_TOPIC_LICENSE_KEYWORDS) have been extended reactively 6 times after live misses
were found, each time adding a new filter target without necessarily revisiting whether
its retrieval site has the same protection. Run this after adding any new
`where={"department": ...}` / `where={"license_type": ...}` (or `metadata_filter=`/
`filter=` equivalents) call site, or periodically in CI, to flag ones that don't look
guarded.

This is a heuristic text scan (regex over inspect.getsource, same low-tech approach as
this project's existing test-suite source-shape pins), not an AST analysis — it will
have false positives/negatives on unusual code shapes. A site already reviewed and
confirmed safe (additive-only, or non-short-circuiting — see
tests/test_no_change_sites_regression_pin.py) should get a one-line marker comment,
`# coverage-guard: additive-only` or `# coverage-guard: non-short-circuiting`, placed
anywhere within WINDOW_LINES of the filter construction, so this scanner doesn't
re-flag it every run.

Usage: python code/scripts/filter_guard_static_scan.py [--files path1.py path2.py ...]
Exit code: 0 if no WARNs, 1 if any WARN found (suitable for a CI gate once the current
baseline is clean).
"""
from __future__ import annotations

import argparse
import os
import re
import sys

DEFAULT_FILES = [
    "code/model/persona_supervisor.py",
    "code/model/persona_practical.py",
    "code/model/persona_academic.py",
]

_FILTER_KEY_RE = re.compile(r'(?:where|metadata_filter|filter)\s*[:=]')
_TARGET_FIELD_RE = re.compile(r'"(department|license_type)"\s*:')
_SAFETY_SIGNAL_RE = re.compile(
    r'"\$in"'
    r'|regulatory_identity_mismatch_should_reject'
    r'|_is_structurally_adequate'
    r'|# coverage-guard: additive-only'
    r'|# coverage-guard: non-short-circuiting'
    r'|# coverage-guard: verify-then-retry'
    r'|_identity_guard_majority_vote'
    r'|_supplement_if_thin'
    r'|_merge_value_groups_with_guard_and_supplement'
)

WINDOW_LINES = 40


def _scan_file(path: str) -> list:
    """Returns a list of (line_no, field, snippet) for unguarded-looking sites."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    findings = []
    flagged_lines: set = set()
    for i, line in enumerate(lines):
        if not _FILTER_KEY_RE.search(line):
            continue
        # Look at a small window starting at this line for the target field —
        # dict literals are often multi-line, so the field key may be a few
        # lines below the where=/filter= token itself.
        window_end = min(len(lines), i + 6)
        local_window = "".join(lines[i:window_end])
        field_match = _TARGET_FIELD_RE.search(local_window)
        if not field_match:
            continue
        field = field_match.group(1)

        # Wider window (both directions) to look for a safety signal nearby.
        ctx_start = max(0, i - WINDOW_LINES)
        ctx_end = min(len(lines), i + WINDOW_LINES)
        context = "".join(lines[ctx_start:ctx_end])
        if _SAFETY_SIGNAL_RE.search(context):
            continue

        line_no = i + 1
        if line_no in flagged_lines:
            continue
        flagged_lines.add(line_no)
        snippet = line.strip()
        findings.append((line_no, field, snippet))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="*", default=None, help="override the default file list")
    args = parser.parse_args()

    files = args.files or DEFAULT_FILES
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    total_findings = 0
    for rel_path in files:
        abs_path = os.path.join(repo_root, rel_path)
        if not os.path.isfile(abs_path):
            print(f"[WARN] file not found, skipping: {rel_path}")
            continue
        findings = _scan_file(abs_path)
        if not findings:
            print(f"[OK]   {rel_path} — no unguarded department/license_type filters found")
            continue
        total_findings += len(findings)
        print(f"[WARN] {rel_path} — {len(findings)} possibly-unguarded filter site(s):")
        for line_no, field, snippet in findings:
            print(f"       line {line_no} ({field}): {snippet}")

    print("-" * 72)
    if total_findings:
        print(
            f"{total_findings} possibly-unguarded department/license_type filter site(s) found. "
            f"Review each: either wire in the identity guard / coverage-adequacy helper, or — "
            f"if already confirmed safe (additive-only / non-short-circuiting) — add a "
            f"'# coverage-guard: additive-only' or '# coverage-guard: non-short-circuiting' "
            f"marker comment near the filter so this scan doesn't re-flag it."
        )
        return 1
    print("No unguarded department/license_type filter sites found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

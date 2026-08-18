# code/scripts/test_dual_extraction.py
"""End-to-end test of the automatic dual-model extraction + validation pipeline
(feature/pdf-ingestion) — the real thing, not manually-pasted comparisons.

Usage:
    export TYPHOON_OCR_API_KEY=<your key>
    python code/scripts/test_dual_extraction.py path/to/sample.pdf --pages 38-40
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
code_dir = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from service.pdf_dual_extraction import extract_page_dual  # noqa: E402


def _parse_page_range(spec: str) -> tuple[int, int]:
    if "-" in spec:
        start, end = spec.split("-")
        return int(start), int(end)
    p = int(spec)
    return p, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", help="Path to a local sample PDF")
    parser.add_argument("--pages", default="1-3", help="Page range, e.g. '38-40'")
    parser.add_argument("--image-dim", type=int, default=1800, help="Typhoon render resolution")
    parser.add_argument(
        "--llm-comparison", action="store_true", help="Also run the general-purpose LLM catch-all check (extra API call/cost)"
    )
    args = parser.parse_args()

    if not os.getenv("TYPHOON_OCR_API_KEY"):
        print("ERROR: set TYPHOON_OCR_API_KEY before running.")
        sys.exit(1)

    start, end = _parse_page_range(args.pages)
    for page_num in range(start, end + 1):
        print(f"\n{'=' * 70}\nPage {page_num}\n{'=' * 70}")
        result = extract_page_dual(
            args.pdf_path, page_num, image_dim=args.image_dim, use_llm_comparison=args.llm_comparison
        )

        if not result.flags:
            print("✅ No flags — Typhoon and Claude agree on all checked facts, arithmetic checks out.")
        else:
            print(f"⚠️  {len(result.flags)} flag(s) — needs human review:")
            for flag in result.flags:
                print(f"  [{flag.severity}] {flag.category}: {flag.message}")
                for k, v in flag.details.items():
                    print(f"      {k}: {v}")


if __name__ == "__main__":
    main()

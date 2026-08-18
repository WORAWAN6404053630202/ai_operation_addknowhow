# code/scripts/test_pdf_extraction.py
"""Manual accuracy-validation CLI for the PDF extraction pipeline (feature/pdf-ingestion).
Run against real sample documents before trusting the pipeline — see the accuracy-testing
plan agreed for this feature (build a small known-answer test set before going live).

Usage:
    python code/scripts/test_pdf_extraction.py path/to/sample.pdf [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
code_dir = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from service.pdf_extraction_service import extract_pdf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", help="Path to a local sample PDF")
    parser.add_argument("--out", default=None, help="Optional path to dump the full JSON result")
    args = parser.parse_args()

    result = extract_pdf(args.pdf_path)

    print(f"\n=== Extraction summary: {args.pdf_path} ===")
    print(f"Confidence: avg={result.avg_confidence}  min={result.min_confidence}  low_confidence={result.low_confidence}")
    print(f"Tables found: {len(result.tables)}")
    print(f"Form fields found: {len(result.forms)}")
    print(f"Figure/diagram regions found: {len(result.figure_regions)}")
    print(f"\n--- Full text (first 1500 chars) ---\n{result.full_text[:1500]}")

    if result.tables:
        print("\n--- First table ---")
        for row in result.tables[0]:
            print(row)

    if result.forms:
        print("\n--- Form fields (first 10) ---")
        for k, v in list(result.forms.items())[:10]:
            print(f"  {k!r}: {v!r}")

    if result.figure_regions:
        print(f"\n--- Figure regions (non-table visuals, e.g. flowcharts/stamps) ---")
        for fig in result.figure_regions:
            print(f"  page {fig['page']}: bbox={fig['bbox']}")
        print("NOTE: figure regions are NOT OCR'd here — per the design, these pages should")
        print("be routed to a vision-LLM pass separately to interpret diagrams/flowcharts.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "full_text": result.full_text,
                    "tables": result.tables,
                    "forms": result.forms,
                    "figure_regions": result.figure_regions,
                    "avg_confidence": result.avg_confidence,
                    "min_confidence": result.min_confidence,
                    "low_confidence": result.low_confidence,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nFull result written to {args.out}")


if __name__ == "__main__":
    main()

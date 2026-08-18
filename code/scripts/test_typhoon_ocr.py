# code/scripts/test_typhoon_ocr.py
"""Manual accuracy-validation CLI for Typhoon OCR (feature/pdf-ingestion), as a
Thai-language alternative to AWS Textract (which does not support Thai — see
lambda/pdf_extraction/handler.py test results, avg_confidence=60.58, garbled output).

Runs entirely locally, no AWS/company-account credentials involved — only needs
TYPHOON_OCR_API_KEY from https://opentyphoon.ai (personal, free).

Usage:
    export TYPHOON_OCR_API_KEY=<your key>
    python code/scripts/test_typhoon_ocr.py path/to/sample.pdf --pages 1-3
    python code/scripts/test_typhoon_ocr.py path/to/sample.pdf --pages 38-40 --image-dim 2800

--image-dim raises the resolution Typhoon renders each PDF page to before OCR
(default 1800, per the typhoon_ocr package default). Higher = sharper detail on
small text (e.g. the tiny numbers/company names inside UI screenshots on page 39-40
that both Typhoon and Claude got wrong in the accuracy comparison) at the cost of
slower/costlier calls. Purely automatic — no validation logic, no extra API calls.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pypdf import PdfReader
from typhoon_ocr import ocr_document


def _parse_page_range(spec: str, total_pages: int) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-")
        pages = list(range(int(start), int(end) + 1))
    else:
        pages = [int(spec)]
    return [p for p in pages if 1 <= p <= total_pages]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", help="Path to a local sample PDF")
    parser.add_argument(
        "--pages", default="1-3", help="Page range to test, e.g. '1-3' or '5' (default: 1-3, keep it small/cheap)"
    )
    parser.add_argument(
        "--image-dim",
        type=int,
        default=1800,
        help="Rendered image resolution fed to OCR (default 1800, try 2400-3000 for small/dense text)",
    )
    args = parser.parse_args()

    if not os.getenv("TYPHOON_OCR_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: set TYPHOON_OCR_API_KEY (from https://opentyphoon.ai) before running.")
        sys.exit(1)

    total_pages = len(PdfReader(args.pdf_path).pages)
    pages = _parse_page_range(args.pages, total_pages)
    print(f"Document has {total_pages} pages total. Testing pages: {pages} at image_dim={args.image_dim}\n")

    for page_num in pages:
        print(f"{'=' * 60}\nPage {page_num}\n{'=' * 60}")
        markdown = ocr_document(
            args.pdf_path,
            model="typhoon-ocr",
            figure_language="Thai",
            task_type="v1.5",
            page_num=page_num,
            target_image_dim=args.image_dim,
        )
        print(markdown)
        print()


if __name__ == "__main__":
    main()

# code/scripts/test_lambda_handler_local.py
"""Local smoke test for lambda/pdf_extraction/handler.py's core logic, BEFORE
packaging + uploading to real AWS Lambda — much faster/cheaper to iterate here
first. Does NOT touch S3 at all (reads the PDF straight from local disk),
only exercises the extraction functions themselves.

Usage:
    export TYPHOON_OCR_API_KEY=<your key>
    export OPENROUTER_API_KEY=<your key>   (or omit if you've already exported it)
    python code/scripts/test_lambda_handler_local.py path/to/sample.pdf --pages 1-2
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# handler.py's module-level logger only sets its own level (logger.setLevel),
# it never attaches a handler (real Lambda runtime provides one automatically
# — a plain local script does not). Without this, logger.info(...) calls —
# including the [CostLog] lines this script exists to verify — are silently
# dropped: confirmed locally that only WARNING+ reaches the console via
# Python's "handler of last resort" when no handler is configured.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# handler.py constructs boto3.client("s3")/("sqs") at import time with no
# explicit region — fine inside a real Lambda runtime (region is always
# ambient there) but breaks import in a plain local shell with no AWS config
# active. Same NoRegionError, same fix already applied in
# tests/test_pdf_path_parity.py's lazy-import helper.
os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-2")

project_root = Path(__file__).parent.parent.parent
lambda_dir = project_root / "lambda" / "pdf_extraction"
if str(lambda_dir) not in sys.path:
    sys.path.insert(0, str(lambda_dir))

# handler.py reads its config from os.environ directly (Lambda convention) —
# make sure conf.py's dotenv loading doesn't leak in here; this must work with
# ONLY plain env vars, same as it will in the real Lambda.
os.environ.setdefault("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")

import handler  # noqa: E402


def _parse_page_range(spec: str) -> tuple[int, int]:
    if "-" in spec:
        start, end = spec.split("-")
        return int(start), int(end)
    p = int(spec)
    return p, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", help="Path to a local sample PDF")
    parser.add_argument("--pages", default="1-2", help="Page range to test, e.g. '1-2' (keep small/cheap)")
    args = parser.parse_args()

    if not os.environ.get("TYPHOON_OCR_API_KEY"):
        print("ERROR: set TYPHOON_OCR_API_KEY before running.")
        sys.exit(1)
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: set OPENROUTER_API_KEY before running.")
        sys.exit(1)

    start, end = _parse_page_range(args.pages)
    with open(args.pdf_path, "rb") as f:
        pdf_bytes = f.read()

    for page_num in range(start, end + 1):
        print(f"\n{'=' * 60}\nPage {page_num}\n{'=' * 60}")
        result = handler._extract_one_page(pdf_bytes, page_num)
        print("--- Typhoon ---")
        print(result["typhoon_markdown"][:500])
        print("--- Claude ---")
        print(result["claude_markdown"][:500])

    print("\n✅ Local handler logic ran without errors — safe to package for real Lambda upload.")


if __name__ == "__main__":
    main()

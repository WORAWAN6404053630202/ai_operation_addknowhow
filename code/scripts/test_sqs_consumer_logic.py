# code/scripts/test_sqs_consumer_logic.py
"""Tests sqs_consumer.py's PROCESSING logic directly against a known S3 key —
bypasses live SQS polling on purpose (no AWS credentials wired up for that yet,
deferred per plan: real SQS consumption will use an EC2 IAM role, not local
static keys). This exercises everything downstream of "here's a bucket/key",
which is the actual logic that matters — the SQS loop around it is a thin,
separately-testable wrapper (see poll_and_process_forever in sqs_consumer.py).

Usage:
    export RESTBIZ_ENV_FILE=env.dev.properties
    python code/scripts/test_sqs_consumer_logic.py <bucket> <key>

Typically run after manually triggering the Lambda (via a console Test event)
against a real uploaded PDF, so a real extraction-result JSON exists in S3 at
restbiz/extracted/<filename>.json to point this at.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
code_dir = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from service.sqs_consumer import process_extraction_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bucket", help="S3 bucket, e.g. chatbot-database-supercoconut")
    parser.add_argument("key", help="S3 key of the extraction-result JSON, e.g. restbiz/extracted/foo.pdf.json")
    parser.add_argument("--llm-comparison", action="store_true", help="Also run the LLM full-content comparison check")
    args = parser.parse_args()

    item = process_extraction_result(args.bucket, args.key, use_llm_comparison=args.llm_comparison)

    print(f"\n✅ Review item created: {item.id}")
    print(f"   filename: {item.filename}")
    print(f"   pages: {len(item.pages)}")
    print(f"   total flags: {item.total_flag_count} ({item.high_severity_flag_count} high severity)")
    print(f"   needs_review: {item.needs_review}")
    print(f"   drafted fields: {sum(1 for v in (item.llm_drafted_fields or {}).values() if v)} / {len(item.llm_drafted_fields or {})} populated")
    print("\n   Check it in the admin panel: /admin/ → PDF Review Queue")


if __name__ == "__main__":
    main()

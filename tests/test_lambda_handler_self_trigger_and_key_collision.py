"""Tests for the 2026-10 fixes (Infra2-M6/PDF2-M6, Infra2-M7) to
lambda/pdf_extraction/handler.py's lambda_handler.

Infra2-M7: nothing used to check that the triggering S3 key was actually a
raw PDF upload rather than one of this same Lambda's own output objects
(written back into the SAME bucket under OUTPUT_S3_PREFIX/
HANDOFF_S3_PREFIX/STATUS_S3_PREFIX) — the only thing preventing a
recursive self-trigger loop was a console-managed S3 bucket-notification
filter this repo has no record of. A 3-line guard at the top of the
per-record loop now makes that console config irrelevant to this risk.

Infra2-M6/PDF2-M6: every S3 key (handoff/output/status) and the
attempt-counter key used to be derived from the bare basename
(key.rsplit("/", 1)[-1]) — two uploads sharing a basename under different
S3 prefixes silently overwrote each other's objects and shared one
retry-attempt budget. Now derived from the full (sanitized) key instead.

This module is standalone (not part of code/) and constructs real boto3
clients at import time, which requires a region to be set — set via
AWS_DEFAULT_REGION before import, same as any Lambda runtime provides.
s3/sqs/_attempt_counter are all patched per-test; no real AWS calls.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

_LAMBDA_DIR = Path(__file__).parent.parent / "lambda" / "pdf_extraction"
if str(_LAMBDA_DIR) not in sys.path:
    sys.path.insert(0, str(_LAMBDA_DIR))

import handler  # noqa: E402


def _s3_event(key: str, bucket: str = "test-bucket") -> dict:
    return {"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]}


@pytest.mark.unit
class TestSelfTriggerGuard:
    def test_key_under_output_prefix_is_skipped_without_touching_s3(self):
        event = _s3_event(f"{handler.OUTPUT_S3_PREFIX}some_doc.pdf.json")
        with patch.object(handler, "TYPHOON_OCR_API_KEY", "x"), \
             patch.object(handler, "OPENROUTER_API_KEY", "x"), \
             patch.object(handler, "s3") as mock_s3:
            result = handler.lambda_handler(event, None)
        mock_s3.get_object.assert_not_called()
        assert result["statusCode"] == 200
        import json as _json
        assert _json.loads(result["body"]) == []

    def test_key_under_handoff_prefix_is_skipped(self):
        event = _s3_event(f"{handler.HANDOFF_S3_PREFIX}some_doc.pdf.json")
        with patch.object(handler, "TYPHOON_OCR_API_KEY", "x"), \
             patch.object(handler, "OPENROUTER_API_KEY", "x"), \
             patch.object(handler, "s3") as mock_s3:
            handler.lambda_handler(event, None)
        mock_s3.get_object.assert_not_called()

    def test_key_under_status_prefix_is_skipped(self):
        event = _s3_event(f"{handler.STATUS_S3_PREFIX}some_doc.pdf.json")
        with patch.object(handler, "TYPHOON_OCR_API_KEY", "x"), \
             patch.object(handler, "OPENROUTER_API_KEY", "x"), \
             patch.object(handler, "s3") as mock_s3:
            handler.lambda_handler(event, None)
        mock_s3.get_object.assert_not_called()

    def test_non_pdf_key_is_skipped(self):
        event = _s3_event("restbiz/raw/not_a_pdf.txt")
        with patch.object(handler, "TYPHOON_OCR_API_KEY", "x"), \
             patch.object(handler, "OPENROUTER_API_KEY", "x"), \
             patch.object(handler, "s3") as mock_s3:
            handler.lambda_handler(event, None)
        mock_s3.get_object.assert_not_called()

    def test_real_pdf_upload_key_is_not_skipped(self):
        """Control case: a genuine raw upload must still reach
        s3.get_object — the guard must not over-match and block real
        documents."""
        event = _s3_event("restbiz/raw/real_document.pdf")
        with patch.object(handler, "TYPHOON_OCR_API_KEY", "x"), \
             patch.object(handler, "OPENROUTER_API_KEY", "x"), \
             patch.object(handler, "s3") as mock_s3:
            mock_s3.get_object.side_effect = RuntimeError("stop here — just proving it was reached")
            with pytest.raises(RuntimeError):
                handler.lambda_handler(event, None)
        mock_s3.get_object.assert_called_once()


def _fake_doc(num_pages: int):
    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=num_pages)
    doc.close = MagicMock()
    return doc


@pytest.mark.unit
class TestFilenameDerivationAvoidsBasenameCollision:
    def _run_small_pdf(self, key: str, bucket: str = "test-bucket"):
        """Drives lambda_handler through its normal (non-handoff,
        ≤MAX_PAGES_FOR_LAMBDA) path for one 1-page PDF, with every real
        OCR/vision/S3/SQS/DynamoDB call point mocked out, and returns the
        output_key it computed."""
        event = _s3_event(key, bucket=bucket)
        fake_page = {"page_num": 1, "typhoon_markdown": "text", "claude_markdown": ""}
        with patch.object(handler, "TYPHOON_OCR_API_KEY", "x"), \
             patch.object(handler, "OPENROUTER_API_KEY", "x"), \
             patch.object(handler, "s3") as mock_s3, \
             patch.object(handler, "_notify_sqs"), \
             patch.object(handler, "_attempt_counter") as mock_counter, \
             patch.object(handler, "_write_status"), \
             patch("pymupdf.open", return_value=_fake_doc(1)), \
             patch.object(handler, "_extract_one_page", return_value=fake_page):
            mock_s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"%PDF-fake"))}
            mock_counter.increment_and_get.return_value = 1
            handler.lambda_handler(event, None)
            output_key = mock_s3.put_object.call_args.kwargs["Key"]
        return output_key

    def test_same_basename_under_different_prefixes_produce_different_output_keys(self):
        key_dev = self._run_small_pdf("dev/raw/report.pdf")
        key_prod = self._run_small_pdf("prod/raw/report.pdf")
        assert key_dev != key_prod, (
            f"Two uploads sharing the basename 'report.pdf' under different "
            f"prefixes must not collide on the same output key: {key_dev!r} vs {key_prod!r}"
        )

    def test_output_key_derived_from_full_path_not_bare_basename(self):
        output_key = self._run_small_pdf("dev/raw/report.pdf")
        assert "report.pdf" in output_key
        assert "dev" in output_key and "raw" in output_key

"""
Tests for the 2026-10 fix (QA-reproduced: real multi-page PDFs landing with
~224 bytes of total extracted text) to service/pdf_large_extraction.py and
lambda/pdf_extraction/handler.py's _extract_one_page().

Background: Typhoon's OCR SDK can return an empty/near-empty string with NO
exception (degraded image quality, moderation, truncation). Before this fix,
_extract_salient_tokens("") found nothing, which made the page look like it
was genuinely blank and skipped vision-verify — the one mechanism that could
have recovered real text. The page silently "succeeded" with ~0 chars, and
those near-empty pages flow into sqs_consumer.py's uncovered-pages catch-all
and get ingested as-is, with no error anywhere in the pipeline.

Fix: vision-verify now also runs when Typhoon's output is suspiciously short
(< _MIN_TYPHOON_OUTPUT_CHARS), not only when salient tokens are present. This
can only make vision-verify fire MORE often than before — it never removes
the existing salient-token trigger — so it is a pure-recall change layered
on top of the existing cost optimization, not a replacement for it.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestEC2EmptyOCRRecovery:
    def _import_module(self):
        import service.pdf_large_extraction as mod
        return mod

    def test_empty_typhoon_output_now_triggers_vision_verify(self):
        """The core regression this fix targets: Typhoon returning "" used to
        be indistinguishable from "this page is genuinely blank" and skipped
        the one recovery mechanism available."""
        mod = self._import_module()
        with patch.object(mod, "extract_page_native_text", return_value=None), \
             patch.object(mod, "_render_page_to_png_bytes", return_value=b"fake-png"), \
             patch.object(mod, "_run_typhoon", return_value=""), \
             patch.object(mod, "_run_vision_verify", return_value="recovered text from vision model") as mock_verify:
            result = mod._extract_one_page(b"fake-pdf-bytes", page_num=3)

        mock_verify.assert_called_once()
        assert result["claude_markdown"] == "recovered text from vision model"

    def test_short_typhoon_output_without_salient_tokens_still_triggers_vision_verify(self):
        mod = self._import_module()
        with patch.object(mod, "extract_page_native_text", return_value=None), \
             patch.object(mod, "_render_page_to_png_bytes", return_value=b"fake-png"), \
             patch.object(mod, "_run_typhoon", return_value="   "), \
             patch.object(mod, "_run_vision_verify", return_value="recovered") as mock_verify:
            result = mod._extract_one_page(b"fake-pdf-bytes", page_num=4)

        mock_verify.assert_called_once()
        assert result["claude_markdown"] == "recovered"

    def test_healthy_long_output_without_salient_tokens_still_skips_vision_verify(self):
        """Pre-existing cost optimization must survive unchanged: a page with
        plenty of real text but none of the 4 salient-token types should NOT
        pay for vision-verify, exactly as before this fix."""
        mod = self._import_module()
        long_prose = "เนื้อหาปกติของหน้านี้ไม่มีตัวเลขหรือรหัสฟอร์มอะไรเลย " * 5
        with patch.object(mod, "extract_page_native_text", return_value=None), \
             patch.object(mod, "_render_page_to_png_bytes", return_value=b"fake-png"), \
             patch.object(mod, "_run_typhoon", return_value=long_prose), \
             patch.object(mod, "_run_vision_verify") as mock_verify:
            result = mod._extract_one_page(b"fake-pdf-bytes", page_num=5)

        mock_verify.assert_not_called()
        assert result["claude_markdown"] == ""
        assert result["typhoon_markdown"] == long_prose

    def test_salient_tokens_still_trigger_vision_verify_regardless_of_length(self):
        """Pre-existing trigger must still work even on a short page."""
        mod = self._import_module()
        with patch.object(mod, "extract_page_native_text", return_value=None), \
             patch.object(mod, "_render_page_to_png_bytes", return_value=b"fake-png"), \
             patch.object(mod, "_run_typhoon", return_value="ค่าธรรมเนียม 500 บาท"), \
             patch.object(mod, "_run_vision_verify", return_value="verified") as mock_verify:
            mod._extract_one_page(b"fake-pdf-bytes", page_num=6)

        mock_verify.assert_called_once()

    def test_both_outputs_still_empty_after_vision_verify_logs_warning(self, caplog):
        mod = self._import_module()
        with patch.object(mod, "extract_page_native_text", return_value=None), \
             patch.object(mod, "_render_page_to_png_bytes", return_value=b"fake-png"), \
             patch.object(mod, "_run_typhoon", return_value=""), \
             patch.object(mod, "_run_vision_verify", return_value=""), \
             caplog.at_level(logging.WARNING):
            mod._extract_one_page(b"fake-pdf-bytes", page_num=8)

        assert any("BOTH Typhoon and vision-verify returned empty" in r.message for r in caplog.records)


@pytest.mark.unit
class TestLambdaEmptyOCRRecovery:
    def _import_module(self):
        import os
        import sys
        from pathlib import Path

        os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-2")
        lambda_dir = Path(__file__).resolve().parent.parent / "lambda" / "pdf_extraction"
        if str(lambda_dir) not in sys.path:
            sys.path.insert(0, str(lambda_dir))
        import handler
        return handler

    def test_empty_typhoon_output_now_triggers_vision_verify(self):
        handler = self._import_module()
        with patch.object(handler, "_extract_page_native_text", return_value=None), \
             patch.object(handler, "_render_page_to_png_bytes", return_value=b"fake-png"), \
             patch.object(handler, "_run_typhoon", return_value=""), \
             patch.object(handler, "_run_vision_verify", return_value="recovered text from vision model") as mock_verify:
            result = handler._extract_one_page(b"fake-pdf-bytes", page_num=3)

        mock_verify.assert_called_once()
        assert result["claude_markdown"] == "recovered text from vision model"

    def test_healthy_long_output_without_salient_tokens_still_skips_vision_verify(self):
        handler = self._import_module()
        long_prose = "เนื้อหาปกติของหน้านี้ไม่มีตัวเลขหรือรหัสฟอร์มอะไรเลย " * 5
        with patch.object(handler, "_extract_page_native_text", return_value=None), \
             patch.object(handler, "_render_page_to_png_bytes", return_value=b"fake-png"), \
             patch.object(handler, "_run_typhoon", return_value=long_prose), \
             patch.object(handler, "_run_vision_verify") as mock_verify:
            result = handler._extract_one_page(b"fake-pdf-bytes", page_num=5)

        mock_verify.assert_not_called()
        assert result["claude_markdown"] == ""

    def test_both_outputs_still_empty_after_vision_verify_logs_warning(self, caplog):
        handler = self._import_module()
        with patch.object(handler, "_extract_page_native_text", return_value=None), \
             patch.object(handler, "_render_page_to_png_bytes", return_value=b"fake-png"), \
             patch.object(handler, "_run_typhoon", return_value=""), \
             patch.object(handler, "_run_vision_verify", return_value=""), \
             caplog.at_level(logging.WARNING):
            handler._extract_one_page(b"fake-pdf-bytes", page_num=8)

        assert any("BOTH Typhoon and vision-verify returned empty" in r.message for r in caplog.records)

"""
Tests for the per-page fault isolation added 2026-09 to service/pdf_large_
extraction.py and lambda/pdf_extraction/handler.py's _extract_one_page().

Background: before this fix, one page's unrecoverable extraction failure
(e.g. Typhoon retries exhausted, a vision-verify error, a corrupted page
pymupdf can't render) would raise out of the ThreadPoolExecutor future and
crash the ENTIRE document — discarding every other page already
successfully (and expensively) extracted in the same run. Now a failure is
caught and turned into a placeholder page record instead, so the rest of
the document's work survives.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestEC2FaultIsolation:
    def _import_module(self):
        import service.pdf_large_extraction as mod
        return mod

    def test_page_failure_returns_placeholder_instead_of_raising(self):
        mod = self._import_module()
        with patch.object(mod, "extract_page_native_text", return_value=None), \
             patch.object(mod, "_render_page_to_png_bytes", side_effect=RuntimeError("simulated corrupted page")):
            result = mod._extract_one_page(b"fake-pdf-bytes", page_num=7)

        assert result["page_num"] == 7
        assert "สกัดข้อมูลหน้านี้ไม่สำเร็จ" in result["typhoon_markdown"]
        assert result["claude_markdown"] == ""

    def test_typhoon_failure_after_retries_returns_placeholder(self):
        mod = self._import_module()
        with patch.object(mod, "extract_page_native_text", return_value=None), \
             patch.object(mod, "_render_page_to_png_bytes", return_value=b"fake-png"), \
             patch.object(mod, "_run_typhoon", side_effect=RuntimeError("simulated Typhoon failure after exhausting retries")):
            result = mod._extract_one_page(b"fake-pdf-bytes", page_num=12)

        assert result["page_num"] == 12
        assert "สกัดข้อมูลหน้านี้ไม่สำเร็จ" in result["typhoon_markdown"]

    def test_healthy_page_is_unaffected(self):
        mod = self._import_module()
        with patch.object(mod, "extract_page_native_text", return_value="native text content"):
            result = mod._extract_one_page(b"fake-pdf-bytes", page_num=1)

        assert result == {"page_num": 1, "typhoon_markdown": "native text content", "claude_markdown": "native text content"}

    def test_one_bad_page_does_not_lose_other_pages_in_a_batch(self):
        """The actual regression this fix targets: extract_full_document()
        must not lose already-completed pages just because one page's
        future would otherwise have raised."""
        import concurrent.futures as cf

        mod = self._import_module()

        def fake_extract(pdf_bytes, page_num):
            if page_num == 2:
                raise RuntimeError("simulated failure — should be caught INSIDE _extract_one_page, not here")
            return {"page_num": page_num, "typhoon_markdown": f"page {page_num} ok", "claude_markdown": ""}

        # Simulate what extract_full_document()'s ThreadPoolExecutor loop does,
        # using the REAL _extract_one_page wrapped around a fake OCR failure —
        # proves the try/except inside _extract_one_page is what prevents the
        # future.result() crash, not some property of the test harness itself.
        with patch.object(mod, "extract_page_native_text", return_value=None), \
             patch.object(mod, "_render_page_to_png_bytes", return_value=b"fake-png"), \
             patch.object(mod, "_run_typhoon", side_effect=lambda png, page_num: (
                 (_ for _ in ()).throw(RuntimeError("boom")) if page_num == 2 else f"page {page_num} typhoon text"
             )):
            pages = [None] * 3
            with cf.ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(mod._extract_one_page, b"fake-pdf-bytes", p): p for p in (1, 2, 3)}
                for future in cf.as_completed(futures):
                    page_num = futures[future]
                    pages[page_num - 1] = future.result()  # must NOT raise for any page

        assert pages[0]["typhoon_markdown"] == "page 1 typhoon text"
        assert "สกัดข้อมูลหน้านี้ไม่สำเร็จ" in pages[1]["typhoon_markdown"]
        assert pages[2]["typhoon_markdown"] == "page 3 typhoon text"


@pytest.mark.unit
class TestLambdaFaultIsolation:
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

    def test_page_failure_returns_placeholder_instead_of_raising(self):
        handler = self._import_module()
        with patch.object(handler, "_extract_page_native_text", return_value=None), \
             patch.object(handler, "_render_page_to_png_bytes", side_effect=RuntimeError("simulated corrupted page")):
            result = handler._extract_one_page(b"fake-pdf-bytes", page_num=9)

        assert result["page_num"] == 9
        assert "สกัดข้อมูลหน้านี้ไม่สำเร็จ" in result["typhoon_markdown"]
        assert result["claude_markdown"] == ""

    def test_healthy_page_is_unaffected(self):
        handler = self._import_module()
        with patch.object(handler, "_extract_page_native_text", return_value="native text content"):
            result = handler._extract_one_page(b"fake-pdf-bytes", page_num=1)

        assert result == {"page_num": 1, "typhoon_markdown": "native text content", "claude_markdown": "native text content"}

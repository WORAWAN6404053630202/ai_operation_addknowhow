"""Tests for the 2026-11 (PDF pipeline overhaul, lower-priority add-on)
document-anchoring fix: when vision-verify runs, whatever native text a
page had — even if extract_page_native_text() rejected it — is passed
along as grounding context (olmOCR's measured hallucination-reduction
technique: a model given SOME reference text hallucinates less than one
given a bare image).

Only _run_vision_verify (EC2: service/pdf_large_extraction.py, Lambda:
lambda/pdf_extraction/handler.py) can receive this — Typhoon's own OCR
call (ocr_document(), a third-party library function) has no prompt-
injection hook, confirmed by inspecting its source this session, so
region-OCR (_run_typhoon) is deliberately left unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from pdf_fixtures import _to_bytes, make_blank_page  # noqa: E402

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


def _page_with_text(text: str) -> bytes:
    doc = make_blank_page()
    try:
        doc[0].insert_text((50, 50), text)
        return _to_bytes(doc)
    finally:
        doc.close()


@pytest.mark.unit
class TestExtractRawPageTextBestEffortEC2:
    def test_returns_page_text_unvalidated(self):
        from utils.pdf_text_layer import extract_raw_page_text_best_effort

        pdf_bytes = _page_with_text("Hello anchoring world")
        result = extract_raw_page_text_best_effort(pdf_bytes, page_num=1)
        assert "Hello anchoring world" in result

    def test_truncates_to_max_chars(self):
        from utils.pdf_text_layer import _ANCHOR_TEXT_MAX_CHARS, extract_raw_page_text_best_effort

        pdf_bytes = _page_with_text("A" * 2000)
        result = extract_raw_page_text_best_effort(pdf_bytes, page_num=1)
        assert len(result) <= _ANCHOR_TEXT_MAX_CHARS

    def test_corrupt_bytes_never_raises_returns_empty(self):
        from utils.pdf_text_layer import extract_raw_page_text_best_effort

        result = extract_raw_page_text_best_effort(b"not a real pdf", page_num=1)
        assert result == ""

    def test_out_of_range_page_never_raises_returns_empty(self):
        from utils.pdf_text_layer import extract_raw_page_text_best_effort

        pdf_bytes = _page_with_text("only one page")
        result = extract_raw_page_text_best_effort(pdf_bytes, page_num=99)
        assert result == ""


@pytest.mark.unit
class TestRunVisionVerifyAnchorTextEC2:
    def _fake_openai_client(self, captured: dict):
        def fake_create(**kwargs):
            captured["prompt_text"] = kwargs["messages"][0]["content"][0]["text"]
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content="vision output"))]
            return resp

        client = MagicMock()
        client.chat.completions.create.side_effect = fake_create
        return client

    def test_anchor_text_appended_to_prompt_when_given(self):
        from service import pdf_large_extraction as mod

        captured: dict = {}
        with patch.object(mod, "OpenAI", return_value=self._fake_openai_client(captured)), \
             patch.object(mod, "get_shared_pdf_http_client", return_value=None), \
             patch.object(mod, "log_llm_cost"):
            mod._run_vision_verify(b"\x89PNG fake", anchor_text="partial rejected native text")

        assert "partial rejected native text" in captured["prompt_text"]

    def test_no_anchor_text_leaves_prompt_unchanged(self):
        from service import pdf_large_extraction as mod

        captured: dict = {}
        with patch.object(mod, "OpenAI", return_value=self._fake_openai_client(captured)), \
             patch.object(mod, "get_shared_pdf_http_client", return_value=None), \
             patch.object(mod, "log_llm_cost"):
            mod._run_vision_verify(b"\x89PNG fake", anchor_text="")

        assert captured["prompt_text"] == mod._CLAUDE_PROMPT

    def test_whitespace_only_anchor_text_leaves_prompt_unchanged(self):
        from service import pdf_large_extraction as mod

        captured: dict = {}
        with patch.object(mod, "OpenAI", return_value=self._fake_openai_client(captured)), \
             patch.object(mod, "get_shared_pdf_http_client", return_value=None), \
             patch.object(mod, "log_llm_cost"):
            mod._run_vision_verify(b"\x89PNG fake", anchor_text="   \n  ")

        assert captured["prompt_text"] == mod._CLAUDE_PROMPT


@pytest.mark.unit
class TestExtractOnePagePassesAnchorTextEC2:
    def test_vision_verify_called_with_raw_page_text_as_anchor(self):
        from service import pdf_large_extraction as mod

        with patch.object(mod, "_render_page_to_png_bytes", return_value=b"\x89PNG fake"), \
             patch.object(mod, "_run_typhoon", return_value="short"), \
             patch.object(mod, "_run_vision_verify", return_value="vision result") as mock_vision, \
             patch.object(mod, "extract_page_native_text", return_value=None), \
             patch.object(mod, "extract_page_supplementary_text", return_value=""), \
             patch.object(mod, "extract_raw_page_text_best_effort", return_value="rejected native text here") as mock_anchor, \
             patch.object(mod, "_is_length_implausible", return_value=True):
            mod._extract_one_page(b"fake pdf bytes", page_num=1)

        mock_anchor.assert_called_once_with(b"fake pdf bytes", 1)
        mock_vision.assert_called_once()
        assert mock_vision.call_args.kwargs["anchor_text"] == "rejected native text here"

    def test_skipped_vision_verify_does_not_bother_extracting_anchor_text(self):
        from service import pdf_large_extraction as mod

        with patch.object(mod, "_render_page_to_png_bytes", return_value=b"\x89PNG fake"), \
             patch.object(mod, "_run_typhoon", return_value="perfectly normal clean output, no special tokens"), \
             patch.object(mod, "_run_vision_verify") as mock_vision, \
             patch.object(mod, "extract_page_native_text", return_value=None), \
             patch.object(mod, "extract_page_supplementary_text", return_value=""), \
             patch.object(mod, "extract_raw_page_text_best_effort") as mock_anchor, \
             patch.object(mod, "_is_likely_degenerate_repetition", return_value=False), \
             patch.object(mod, "_is_length_implausible", return_value=False), \
             patch.object(mod, "_should_random_audit_sample", return_value=False):
            mod._extract_one_page(b"fake pdf bytes", page_num=1)

        mock_vision.assert_not_called()
        mock_anchor.assert_not_called()


@pytest.mark.unit
class TestExtractRawPageTextBestEffortLambda:
    def _import_handler(self):
        import os

        os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-2")
        lambda_dir = Path(__file__).resolve().parent.parent / "lambda" / "pdf_extraction"
        if str(lambda_dir) not in sys.path:
            sys.path.insert(0, str(lambda_dir))
        import handler
        return handler

    def test_returns_page_text_unvalidated(self):
        handler = self._import_handler()
        pdf_bytes = _page_with_text("Hello anchoring world")
        result = handler._extract_raw_page_text_best_effort(pdf_bytes, page_num=1)
        assert "Hello anchoring world" in result

    def test_corrupt_bytes_never_raises_returns_empty(self):
        handler = self._import_handler()
        result = handler._extract_raw_page_text_best_effort(b"not a real pdf", page_num=1)
        assert result == ""


@pytest.mark.unit
class TestExtractOnePagePassesAnchorTextLambda:
    def _import_handler(self):
        import os

        os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-2")
        lambda_dir = Path(__file__).resolve().parent.parent / "lambda" / "pdf_extraction"
        if str(lambda_dir) not in sys.path:
            sys.path.insert(0, str(lambda_dir))
        import handler
        return handler

    def test_vision_verify_called_with_raw_page_text_as_anchor(self):
        handler = self._import_handler()
        with patch.object(handler, "_render_page_to_png_bytes", return_value=b"\x89PNG fake"), \
             patch.object(handler, "_run_typhoon", return_value="short"), \
             patch.object(handler, "_run_vision_verify", return_value="vision result") as mock_vision, \
             patch.object(handler, "_extract_page_native_text", return_value=None), \
             patch.object(handler, "_extract_page_supplementary_text", return_value=""), \
             patch.object(handler, "_extract_raw_page_text_best_effort", return_value="rejected native text here") as mock_anchor, \
             patch.object(handler, "_is_length_implausible", return_value=True):
            handler._extract_one_page(b"fake pdf bytes", page_num=1)

        mock_anchor.assert_called_once_with(b"fake pdf bytes", 1)
        mock_vision.assert_called_once()
        assert mock_vision.call_args.kwargs["anchor_text"] == "rejected native text here"

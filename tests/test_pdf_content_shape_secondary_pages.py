"""Test for the 2026-10 fix (PDF2-N2) to service/pdf_content_shape.py's
classify_content_shape — overlapping secondary_pages ranges from the
model's answer (e.g. [[6,10],[8,12]] for one real secondary section) used
to be kept as-is; the overlapping pages (8-10 here) then get processed
twice by the secondary builder downstream, writing duplicate vectorstore
docs for the same content. Now collapsed into disjoint ranges.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from service.pdf_content_shape import classify_content_shape  # noqa: E402


def _fake_openai_response(shape: str, secondary_pages: list, reasoning: str = "r"):
    payload = json.dumps({"shape": shape, "reasoning": reasoning, "secondary_pages": secondary_pages})
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=payload))]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    return resp


@pytest.mark.unit
class TestSecondaryPagesCollapsedToDisjointRanges:
    def test_overlapping_ranges_are_merged(self):
        pages = [f"page {i}" for i in range(1, 13)]
        fake_resp = _fake_openai_response("mixed", [[6, 10], [8, 12]])
        with patch("service.pdf_content_shape.OpenAI") as mock_openai_cls, \
             patch("service.pdf_content_shape.log_llm_cost"):
            mock_openai_cls.return_value.chat.completions.create.return_value = fake_resp
            result = classify_content_shape(pages)

        assert result["secondary_pages"] == [(6, 12)]

    def test_non_overlapping_ranges_are_left_separate(self):
        pages = [f"page {i}" for i in range(1, 16)]
        fake_resp = _fake_openai_response("mixed", [[2, 4], [10, 12]])
        with patch("service.pdf_content_shape.OpenAI") as mock_openai_cls, \
             patch("service.pdf_content_shape.log_llm_cost"):
            mock_openai_cls.return_value.chat.completions.create.return_value = fake_resp
            result = classify_content_shape(pages)

        assert result["secondary_pages"] == [(2, 4), (10, 12)]

    def test_no_secondary_pages_stays_empty(self):
        pages = [f"page {i}" for i in range(1, 4)]
        fake_resp = _fake_openai_response("structured_license", [])
        with patch("service.pdf_content_shape.OpenAI") as mock_openai_cls, \
             patch("service.pdf_content_shape.log_llm_cost"):
            mock_openai_cls.return_value.chat.completions.create.return_value = fake_resp
            result = classify_content_shape(pages)

        assert result["secondary_pages"] == []

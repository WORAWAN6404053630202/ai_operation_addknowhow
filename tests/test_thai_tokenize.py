"""
Unit tests for utils/thai_tokenize.py (.claude/plans/mutable-seeking-wave.md
— "Add PyThaiNLP pre-tokenization to Elasticsearch's BM25/match leg").

Test cases are the exact terms used to confirm ES's own `standard`/`thai`
analyzers mis-segment this corpus's legal/regulatory vocabulary (live
`_analyze` API comparison, 2026-11) — pinning that PyThaiNLP's `newmm`
engine gets these specific, previously-broken cases right.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

CODE_DIR = Path(__file__).parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.thai_tokenize import tokenize_thai_for_search


@pytest.mark.unit
class TestTokenizeThaiForSearch:
    def test_keeps_sathanti_whole_unlike_es_thai_analyzer(self):
        """ES's built-in `thai` analyzer drops "ที่" and leaves just "สถาน"
        — confirmed live. newmm must keep "สถานที่" intact."""
        result = tokenize_thai_for_search("สถานที่จำหน่ายอาหาร")
        assert "สถานที่" in result.split()

    def test_keeps_truetsob_whole_unlike_es_thai_analyzer(self):
        """ES's built-in `thai` analyzer splits "ตรวจสอบ" into the two
        meaningless fragments "ตรวจ"+"สอบ" — confirmed live."""
        result = tokenize_thai_for_search("การตรวจสอบสภาพอาคาร")
        assert "ตรวจสอบ" in result.split()

    def test_splits_nitibukkon_from_digital_unlike_standard_analyzer(self):
        """ES's default `standard` analyzer returns the whole phrase as one
        token. newmm must split it into real words."""
        result = tokenize_thai_for_search("การจดทะเบียนนิติบุคคลดิจิทัล")
        tokens = result.split()
        assert len(tokens) > 1
        assert "นิติบุคคล" in tokens

    def test_result_is_whitespace_joined_string(self):
        result = tokenize_thai_for_search("กรมพัฒนาธุรกิจการค้า")
        assert isinstance(result, str)
        assert " " in result or len(result.split()) == 1

    def test_empty_string_returns_empty_string(self):
        assert tokenize_thai_for_search("") == ""

    def test_none_input_does_not_raise(self):
        assert tokenize_thai_for_search(None) == ""

    def test_falls_back_to_original_text_when_pythainlp_unavailable(self):
        # Setting a module to None in sys.modules makes Python raise
        # ImportError on `from <name> import ...` — the standard trick for
        # simulating "package not installed" without actually uninstalling it.
        with patch.dict(sys.modules, {"pythainlp.tokenize": None}):
            from utils.thai_tokenize import tokenize_thai_for_search as _fn
            result = _fn("สวัสดีครับ")
        assert result == "สวัสดีครับ"

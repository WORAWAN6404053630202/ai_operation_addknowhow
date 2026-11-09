"""Tests for the output-derived uncertainty signals added to
service/pdf_large_extraction.py (_is_likely_degenerate_repetition,
_is_length_implausible, _should_random_audit_sample,
_texts_substantially_agree) and the rewired vision-verify trigger in
_extract_one_page — 2026-11, PDF-extraction-pipeline overhaul, gap #5.

Background: the second OCR pass (_run_vision_verify) was gated by
_extract_salient_tokens matching one of only 4 content categories (baht
amounts, dates, form codes, license numbers). A prose-heavy page with none
of those — a marketing chapter, a procedural narrative — could never
trigger verification no matter how garbled Typhoon's output was, confirmed
this session to be ~67% of OCR'd pages. These signals are computed from
Typhoon's own OUTPUT instead, so they fire regardless of content category."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from service.pdf_large_extraction import (  # noqa: E402
    _is_length_implausible,
    _is_likely_degenerate_repetition,
    _should_random_audit_sample,
    _texts_substantially_agree,
)

_NORMAL_THAI = "คู่มือการขอใบอนุญาตขายสุรา ยาสูบ ไพ่ ทางอินเทอร์เน็ต กรมสรรพสามิต เอกสารนี้จัดทำขึ้นเพื่อให้ความรู้แก่ผู้ประกอบการที่สนใจ โดยมีขั้นตอนดังต่อไปนี้"


@pytest.mark.unit
class TestDegenerateRepetitionDetector:
    def test_normal_prose_is_not_flagged(self):
        assert _is_likely_degenerate_repetition(_NORMAL_THAI) is False

    def test_highly_repetitive_text_is_flagged(self):
        repetitive = "ใบอนุญาต ใบอนุญาต ใบอนุญาต ใบอนุญาต " * 20
        assert _is_likely_degenerate_repetition(repetitive) is True

    def test_realistic_degenerate_pattern_normal_then_loops(self):
        """The actual VLM OCR failure mode this targets: starts coherent,
        degrades into a repetition loop partway through."""
        text = (
            "เอกสารนี้เป็นคำขออนุญาตประกอบกิจการร้านอาหารตามที่กฎหมายกำหนด ผู้ยื่นคำขอต้องเตรียมเอกสารดังนี้ "
            + "และเอกสารอื่นๆ และเอกสารอื่นๆ และเอกสารอื่นๆ " * 30
        )
        assert _is_likely_degenerate_repetition(text) is True

    def test_short_text_is_never_flagged(self):
        """zlib's fixed per-call overhead makes the ratio meaningless below
        the minimum length — must not false-positive on short legitimate
        pages."""
        assert _is_likely_degenerate_repetition("อนุมัติ") is False

    def test_empty_text_is_not_flagged(self):
        assert _is_likely_degenerate_repetition("") is False

    def test_malformed_input_fails_safe_to_not_flagged(self):
        assert _is_likely_degenerate_repetition(None) is False


@pytest.mark.unit
class TestLengthImplausibilityCheck:
    def test_dense_image_with_short_output_is_flagged(self):
        large_png = b"\x89PNG" + b"\x00" * 20000  # simulates a large (visually complex) rendered page
        assert _is_length_implausible(large_png, "short") is True

    def test_dense_image_with_substantial_output_is_not_flagged(self):
        large_png = b"\x89PNG" + b"\x00" * 20000
        assert _is_length_implausible(large_png, _NORMAL_THAI) is False

    def test_small_blank_ish_image_with_short_output_is_not_flagged(self):
        """A genuinely near-blank page SHOULD return little text — not
        suspicious on its own."""
        small_png = b"\x89PNG" + b"\x00" * 100
        assert _is_length_implausible(small_png, "short") is False

    def test_malformed_input_fails_safe_to_not_flagged(self):
        assert _is_length_implausible(None, "text") is False


@pytest.mark.unit
class TestRandomAuditSample:
    def test_fires_at_approximately_the_configured_rate(self):
        random.seed(42)
        n = 10000
        fired = sum(1 for _ in range(n) if _should_random_audit_sample())
        rate = fired / n
        assert 0.04 < rate < 0.10, f"audit sample rate {rate} outside expected ~0.07 band"


@pytest.mark.unit
class TestTextsSubstantiallyAgree:
    def test_identical_text_agrees(self):
        assert _texts_substantially_agree(_NORMAL_THAI, _NORMAL_THAI) is True

    def test_both_empty_agrees(self):
        assert _texts_substantially_agree("", "") is True

    def test_one_empty_one_not_disagrees(self):
        assert _texts_substantially_agree(_NORMAL_THAI, "") is False
        assert _texts_substantially_agree("", _NORMAL_THAI) is False

    def test_completely_different_text_disagrees(self):
        assert _texts_substantially_agree(_NORMAL_THAI, "A totally unrelated English sentence about something else entirely.") is False

    def test_minor_variation_agrees(self):
        """Robust to small OCR noise (a character or two different) — this
        reconciliation check is about catching SUBSTANTIAL divergence, not
        demanding byte-identical output."""
        minor_variant = _NORMAL_THAI[:-1] + "ง"
        assert _texts_substantially_agree(_NORMAL_THAI, minor_variant) is True


@pytest.mark.unit
class TestExtractOnePageUncertaintyGating:
    """End-to-end: confirms each new signal actually triggers vision-verify
    via _extract_one_page, and that the salient-token trigger still works
    (kept, not replaced)."""

    def _run_with_mocks(self, typhoon_output, *, degenerate=False, length_implausible=False, audit_sample=False, vision_output="vision result"):
        from service.pdf_large_extraction import _extract_one_page

        with patch("service.pdf_large_extraction._render_page_to_png_bytes", return_value=b"\x89PNG fake"), \
             patch("service.pdf_large_extraction._run_typhoon", return_value=typhoon_output), \
             patch("service.pdf_large_extraction._run_vision_verify", return_value=vision_output) as mock_vision, \
             patch("service.pdf_large_extraction.extract_page_native_text", return_value=None), \
             patch("service.pdf_large_extraction.extract_page_supplementary_text", return_value=""), \
             patch("service.pdf_large_extraction._is_likely_degenerate_repetition", return_value=degenerate), \
             patch("service.pdf_large_extraction._is_length_implausible", return_value=length_implausible), \
             patch("service.pdf_large_extraction._should_random_audit_sample", return_value=audit_sample), \
             patch("service.pdf_large_extraction._texts_substantially_agree", return_value=True):
            result = _extract_one_page(b"fake pdf bytes", 1)
        return result, mock_vision

    def test_degenerate_repetition_triggers_vision_verify(self):
        _, mock_vision = self._run_with_mocks("repetitive junk output", degenerate=True)
        mock_vision.assert_called_once()

    def test_length_implausible_triggers_vision_verify(self):
        _, mock_vision = self._run_with_mocks("short", length_implausible=True)
        mock_vision.assert_called_once()

    def test_random_audit_sample_triggers_vision_verify(self):
        _, mock_vision = self._run_with_mocks("perfectly normal clean output", audit_sample=True)
        mock_vision.assert_called_once()

    def test_clean_output_with_no_triggers_skips_vision_verify(self):
        _, mock_vision = self._run_with_mocks("perfectly normal clean output, no special tokens here")
        mock_vision.assert_not_called()

    def test_salient_token_trigger_still_works(self):
        """The ORIGINAL trigger (kept, not retired) — a baht amount in
        Typhoon's output must still trigger verification."""
        _, mock_vision = self._run_with_mocks("ค่าธรรมเนียม 500 บาท ต้องชำระภายในกำหนด")
        mock_vision.assert_called_once()

    def test_substantial_disagreement_prefers_longer_output_when_neither_is_degenerate(self):
        from service.pdf_large_extraction import _extract_one_page

        short_typhoon = "short typhoon output"
        long_vision = "a much longer and more complete vision-verify output with substantially more content than typhoon produced"
        with patch("service.pdf_large_extraction._render_page_to_png_bytes", return_value=b"\x89PNG fake"), \
             patch("service.pdf_large_extraction._run_typhoon", return_value=short_typhoon), \
             patch("service.pdf_large_extraction._run_vision_verify", return_value=long_vision), \
             patch("service.pdf_large_extraction.extract_page_native_text", return_value=None), \
             patch("service.pdf_large_extraction.extract_page_supplementary_text", return_value=""), \
             patch("service.pdf_large_extraction._is_likely_degenerate_repetition", return_value=False), \
             patch("service.pdf_large_extraction._should_random_audit_sample", return_value=True), \
             patch("service.pdf_large_extraction._texts_substantially_agree", return_value=False):
            result = _extract_one_page(b"fake pdf bytes", 1)
        assert result["typhoon_markdown"] == long_vision

    def test_disagreement_prefers_non_degenerate_side_even_if_shorter(self):
        """2026-11-10 real-corpus validation catch: Typhoon looped into a
        32,771-char repetition of "- " on a form's ID-number blank line
        (real page, real bug) — "prefer longer" picked that garbage over
        vision-verify's correct short answer. Now the non-degenerate side
        wins regardless of length."""
        from service.pdf_large_extraction import _extract_one_page

        degenerate_typhoon = "- " * 16000  # much longer, but garbage
        correct_vision = "เลขบัตรประจำตัวประชาชน: _-____-_____-__-_"
        with patch("service.pdf_large_extraction._render_page_to_png_bytes", return_value=b"\x89PNG fake"), \
             patch("service.pdf_large_extraction._run_typhoon", return_value=degenerate_typhoon), \
             patch("service.pdf_large_extraction._run_vision_verify", return_value=correct_vision), \
             patch("service.pdf_large_extraction.extract_page_native_text", return_value=None), \
             patch("service.pdf_large_extraction.extract_page_supplementary_text", return_value=""), \
             patch("service.pdf_large_extraction._is_likely_degenerate_repetition", side_effect=lambda t: t == degenerate_typhoon), \
             patch("service.pdf_large_extraction._texts_substantially_agree", return_value=False):
            result = _extract_one_page(b"fake pdf bytes", 1)
        assert result["typhoon_markdown"] == correct_vision

    def test_disagreement_prefers_non_degenerate_typhoon_over_degenerate_vision(self):
        """Same fix, opposite side: vision-verify is the one that looped."""
        from service.pdf_large_extraction import _extract_one_page

        correct_typhoon = "ข้อมูลที่ถูกต้องจาก Typhoon"
        degenerate_vision = "repeat " * 5000
        with patch("service.pdf_large_extraction._render_page_to_png_bytes", return_value=b"\x89PNG fake"), \
             patch("service.pdf_large_extraction._run_typhoon", return_value=correct_typhoon), \
             patch("service.pdf_large_extraction._run_vision_verify", return_value=degenerate_vision), \
             patch("service.pdf_large_extraction.extract_page_native_text", return_value=None), \
             patch("service.pdf_large_extraction.extract_page_supplementary_text", return_value=""), \
             patch("service.pdf_large_extraction._is_likely_degenerate_repetition", side_effect=lambda t: t == degenerate_vision), \
             patch("service.pdf_large_extraction._texts_substantially_agree", return_value=False):
            result = _extract_one_page(b"fake pdf bytes", 1)
        assert result["typhoon_markdown"] == correct_typhoon

    def test_disagreement_prefers_shorter_when_both_sides_look_degenerate(self):
        """Rare, pathological double-failure — prefer the shorter of the
        two as the lesser-damage guess rather than compounding the
        inflated-length bias of a repetition loop."""
        from service.pdf_large_extraction import _extract_one_page

        shorter_degenerate = "x " * 100
        longer_degenerate = "y " * 10000
        with patch("service.pdf_large_extraction._render_page_to_png_bytes", return_value=b"\x89PNG fake"), \
             patch("service.pdf_large_extraction._run_typhoon", return_value=longer_degenerate), \
             patch("service.pdf_large_extraction._run_vision_verify", return_value=shorter_degenerate), \
             patch("service.pdf_large_extraction.extract_page_native_text", return_value=None), \
             patch("service.pdf_large_extraction.extract_page_supplementary_text", return_value=""), \
             patch("service.pdf_large_extraction._is_likely_degenerate_repetition", return_value=True), \
             patch("service.pdf_large_extraction._texts_substantially_agree", return_value=False):
            result = _extract_one_page(b"fake pdf bytes", 1)
        assert result["typhoon_markdown"] == shorter_degenerate

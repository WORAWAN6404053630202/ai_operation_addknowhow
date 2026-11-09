"""Tests for the per-block Thai linguistic text-layer validation added to
utils/pdf_text_layer.py (thai_linguistic_score, text_block_composite_score,
classify_page_text_blocks) — 2026-11, PDF-extraction-pipeline overhaul,
gap #3.

Background: the old checks (text_looks_valid, _salient_regions_look_valid)
only looked at Unicode CHARACTER RANGES. In-script Thai mojibake — wrong
tone-mark order, a bad legacy scanner's hidden OCR layer — stays entirely
within the Thai Unicode block and is invisible to a range check no matter
how scrambled the actual words are. Confirmed empirically this session:
pythainlp's word_tokenize separates clean Thai (mean_token_length ~5.5-11
chars) from fully character-shuffled Thai (~2.5-3 chars) far more reliably
than raw known-word-rate (newmm's greedy longest-match "launders" garbled
input into short-but-real dictionary fragments).

Two real bugs found and fixed DURING this session's own implementation
(kept here as regression tests, not just design notes):
1. Averaging the unicode-range and linguistic components into one blended
   score dilutes the linguistic signal to uselessness for in-script
   corruption — shuffled Thai characters are still valid Thai Unicode
   codepoints, so the unicode component stays ~1.0 regardless, and a 50/50
   blend floors around 0.75-0.77 even for maximally-scrambled text. Fixed
   by using independent gates (both components must individually clear
   their own threshold) instead of one weighted average.
2. pythainlp's word_tokenize does NOT leave pure-English text untokenized —
   it happily splits "This is real page body text" into short English
   word-chunks that then score as if they were Thai (near-zero dictionary
   matches, short mean token length): a legitimate pure-English block
   scored 0.44, well below any reasonable validity threshold, before this
   fix. Fixed by gating on the THAI-SCRIPT character ratio first."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils.pdf_text_layer import (  # noqa: E402
    _block_is_valid,
    classify_page_text_blocks,
    text_block_composite_score,
    thai_linguistic_score,
)

_CLEAN_THAI = "คู่มือการขอใบอนุญาตขายสุรา ยาสูบ ไพ่ ทางอินเทอร์เน็ต กรมสรรพสามิต เอกสารนี้จัดทำขึ้นเพื่อให้ความรู้แก่ผู้ประกอบการ"


def _scramble(text: str, seed: int) -> str:
    chars = list(text.replace(" ", ""))
    random.Random(seed).shuffle(chars)
    return "".join(chars)


@pytest.mark.unit
class TestThaiLinguisticScore:
    def test_clean_thai_scores_high(self):
        assert thai_linguistic_score(_CLEAN_THAI) > 0.9

    def test_severely_scrambled_thai_scores_low(self):
        for seed in range(5):
            scrambled = _scramble(_CLEAN_THAI, seed)
            assert thai_linguistic_score(scrambled) < 0.80, f"seed {seed} scored too high: {thai_linguistic_score(scrambled)}"

    def test_mild_single_character_noise_still_scores_high(self):
        """Explicit non-goal: mild noise is Phase 5's territory (output-
        derived signals on real OCR output), not this check's job — this
        guards against the threshold ever being tuned so tight it starts
        false-positiving on realistic minor noise."""
        mild = _CLEAN_THAI[:30] + _CLEAN_THAI[31] + _CLEAN_THAI[30] + _CLEAN_THAI[32:]
        assert thai_linguistic_score(mild) > 0.85

    def test_pure_english_text_is_not_penalized(self):
        """Regression test for the real bug found this session: pythainlp's
        tokenizer does not leave English text alone."""
        english = "This is real page body text content here. " * 5
        assert thai_linguistic_score(english) == 1.0

    def test_pure_numeric_text_is_not_penalized(self):
        assert thai_linguistic_score("1234567890 " * 10) == 1.0

    def test_empty_text_is_not_penalized(self):
        assert thai_linguistic_score("") == 1.0

    def test_mixed_thai_english_with_enough_thai_is_scored(self):
        """A block that's mostly Thai with some English terms embedded
        (common in Thai regulatory docs — "ใบอนุญาต SAN Plus") must still
        go through the Thai-specific check, not get waved through just
        because some English is present."""
        mixed_clean = _CLEAN_THAI + " SAN Plus"
        mixed_garbled = _scramble(_CLEAN_THAI, 0) + " SAN Plus"
        assert thai_linguistic_score(mixed_clean) > thai_linguistic_score(mixed_garbled)


@pytest.mark.unit
class TestIndependentGateDesign:
    """Regression test for the real bug found this session: a 0.5/0.5
    weighted average of the unicode and linguistic components dilutes the
    linguistic signal to uselessness for in-script corruption (scrambled
    Thai characters are still valid Thai Unicode, so the unicode component
    never drops). Independent gates fix this."""

    def test_clean_text_passes_both_gates(self):
        assert _block_is_valid(_CLEAN_THAI) is True

    def test_scrambled_text_fails_the_gate_despite_valid_unicode_range(self):
        scrambled = _scramble(_CLEAN_THAI, 0)
        unicode_component, linguistic_component = text_block_composite_score(scrambled)
        assert unicode_component > 0.95, "scrambled text should still be valid Thai Unicode codepoints"
        assert linguistic_component < 0.80, "but the linguistic component must catch the scrambling"
        assert _block_is_valid(scrambled) is False

    def test_a_simple_weighted_average_would_have_missed_this(self):
        """Explicit demonstration of the bug this design avoids — kept as a
        permanent regression guard, not just a one-off finding."""
        scrambled = _scramble(_CLEAN_THAI, 0)
        unicode_component, linguistic_component = text_block_composite_score(scrambled)
        naive_average = 0.5 * unicode_component + 0.5 * linguistic_component
        assert naive_average > 0.70, "confirms a naive blend would NOT have rejected this scrambled text"


@pytest.mark.unit
class TestClassifyPageTextBlocks:
    def test_short_blocks_are_skipped_entirely(self):
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 50), "หน้า 6", fontsize=10)  # well under _MIN_BLOCK_CHARS_FOR_VALIDATION
        try:
            blocks = classify_page_text_blocks(page)
        finally:
            doc.close()
        assert blocks == []

    def test_long_clean_block_is_classified_valid(self):
        """Uses English text, not Thai, for the real-PDF round-trip —
        pymupdf's default built-in font (no explicit fontfile) can't render
        Thai glyphs and round-trips Thai as placeholder dots (confirmed
        this session: identical to the documented limitation in
        tests/test_pdf_text_layer_image_detection.py's own _make_pdf_page
        docstring) — that's a test-fixture limitation of insert_text's
        default font, not a real validity failure, and would make this
        test assert the wrong thing. The Thai-specific scoring logic itself
        is already covered directly (no PDF round-trip needed) by
        TestThaiLinguisticScore and TestIndependentGateDesign above; this
        test only needs to confirm the block-classification MECHANISM
        (bbox extraction, per-block iteration) works end-to-end through a
        real PDF."""
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 50), "This is real page body text content here. " * 5, fontsize=10)
        try:
            blocks = classify_page_text_blocks(page)
        finally:
            doc.close()
        assert len(blocks) >= 1
        assert all(b["valid"] for b in blocks)
        assert all(b["bbox"] is not None for b in blocks)

    def test_malformed_page_fails_safe_to_empty_list(self):
        class _BrokenPage:
            def get_text(self, *_a, **_k):
                raise RuntimeError("boom")

        assert classify_page_text_blocks(_BrokenPage()) == []

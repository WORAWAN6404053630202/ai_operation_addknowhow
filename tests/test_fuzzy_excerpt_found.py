"""Tests for utils/page_ranges.py's fuzzy_excerpt_found — 2026-10, added for
service/pdf_supersede_resolver.py's completeness-check: the one deterministic
cross-check standing between an LLM's bare "contradicted" assertion and
auto-releasing a retired document's old value with no further safety net.
fuzzy_ratio (the module's existing function) answers a different question —
whole-string similarity — and returns near-0 for a genuine short-excerpt-
inside-a-long-blob match, which is exactly the shape this check needs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils.page_ranges import fuzzy_excerpt_found  # noqa: E402


@pytest.mark.unit
class TestFuzzyExcerptFound:
    def test_exact_substring_match_is_found(self):
        content = "เนื้อหายาวมากมาย " * 50 + "ค่าธรรมเนียม 150 บาท" + " เนื้อหาต่อ" * 50
        assert fuzzy_excerpt_found("ค่าธรรมเนียม 150 บาท", content) is True

    def test_whitespace_and_case_differences_still_match(self):
        content = "   ค่าธรรมเนียม   150   บาท   "
        assert fuzzy_excerpt_found("ค่าธรรมเนียม 150 บาท", content) is True

    def test_minor_wording_drift_still_matches_above_threshold(self):
        content = "เนื้อหาเดิม " * 30 + "ชำระค่าธรรมเนียมจำนวน 150 บาทถ้วน" + " เนื้อหาต่อ" * 30
        assert fuzzy_excerpt_found("ค่าธรรมเนียม 150 บาท", content, threshold=0.5) is True

    def test_excerpt_genuinely_absent_is_not_found(self):
        content = "เนื้อหานี้ไม่เกี่ยวข้องกับค่าธรรมเนียมเลยแม้แต่น้อย " * 20
        assert fuzzy_excerpt_found("ค่าธรรมเนียม 999,999 บาท สำหรับใบอนุญาตพิเศษ", content) is False

    def test_empty_excerpt_is_not_found(self):
        assert fuzzy_excerpt_found("", "เนื้อหาอะไรก็ได้") is False

    def test_empty_content_is_not_found(self):
        assert fuzzy_excerpt_found("ข้อความ", "") is False

    def test_short_fake_excerpt_against_long_unrelated_content_rejected(self):
        """Regression guard for the actual bug this function exists to
        avoid: fuzzy_ratio's own whole-string comparison would score a
        real excerpt genuinely present inside a long blob as near-0
        similarity purely due to length mismatch — this function must not
        make that mistake (false negative on a real match), but must also
        not go so loose it accepts an unrelated excerpt just because the
        blob is long (false positive)."""
        content = "ขั้นตอนการยื่นคำขอมีดังนี้ หนึ่งกรอกแบบฟอร์ม สองแนบเอกสาร สามยื่นที่เคาน์เตอร์ " * 10
        assert fuzzy_excerpt_found("ค่าปรับ 5000 บาท กรณีเกินกำหนด", content) is False

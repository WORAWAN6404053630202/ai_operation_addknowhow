"""
Regression test for the 2026-11 fix to utils/query_synonyms.py's marketing
synonym pattern. The old pattern (market(?:ing)?(?:\\s+strategy)?) matched
bare "market" as a substring — not just embedded ("supermarket") but also
as a legitimate standalone word ("flea market", "night market", "stock
market"), none of which are about marketing strategy. This bot serves food-
business vendors who may ask about operating in a night/flea market, so the
false match biased retrieval toward the wrong topic_group. Fixed by dropping
the ambiguous bare-"market" alternative entirely.
"""
from __future__ import annotations

import re

import pytest

from utils.query_synonyms import SYNONYM_PATTERNS


def _marketing_pattern() -> str:
    for pattern, expansion in SYNONYM_PATTERNS:
        if "กลยุทธ์การตลาด การตลาด marketing" == expansion:
            return pattern
    raise AssertionError("marketing synonym pattern not found in SYNONYM_PATTERNS")


@pytest.mark.unit
class TestMarketingPatternDropsAmbiguousBareMarket:
    def test_false_positives_no_longer_match(self):
        pattern = _marketing_pattern()
        for text in ["flea market", "night market", "supermarket", "stock market"]:
            assert not re.search(pattern, text, re.IGNORECASE), f"{text!r} should not match"

    def test_genuine_marketing_mentions_still_match(self):
        pattern = _marketing_pattern()
        for text in ["marketing", "market strategy", "การตลาด", "digital marketing plan"]:
            assert re.search(pattern, text, re.IGNORECASE), f"{text!r} should still match"

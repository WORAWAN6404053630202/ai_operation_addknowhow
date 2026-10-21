"""2026-10 fix: _ensure_practical_retrieval_for_legal's main_topic and
sub_topic/operation_topic "chapter retrieval" blocks each picked the
LONGEST matched label via max(candidates, key=len) with no regard for HOW
each candidate matched. topic_label_matches_query's ASCII-identifier
fallback (checked when a candidate doesn't literally substring-match) can
fire on a token shared across MANY unrelated topics — a PDF-ingested
manual's product/system name tagged onto nearly every chapter (e.g.
"(DBD Biz Regist)") — so a query merely mentioning that shared name let an
unrelated-but-longer chapter label out-rank the correct, shorter,
genuinely substring-matched chapter.

Live-reproduced: "สมัครสมาชิกผู้ใช้งานระบบ DBD Biz Regist ต้องทำยังไง"
ASCII-matched both the correct "สมัครสมาชิกผู้ใช้งานระบบ" chapter (24 chars,
literal substring) and an unrelated "dissolve the company" chapter (70
chars, matched only via the shared "DBD Biz Regist" ASCII tokens) — the
longer, wrong one won.

Fix: when ANY candidate matches via topic_label_matches_query's new
strong_only=True (checks (1)-(3): literal/nominalizer-stripped substring,
specific to THIS label), restrict the max(key=len) selection to strong
candidates only, ignoring weak ASCII-only matches entirely — falls back to
the original lenient behavior, unchanged, whenever nothing matched
strongly (the heavily-tested restaurant-domain ASCII-fallback cases are
untouched).

persona_supervisor.py's _ensure_practical_retrieval_for_legal is a single
very long method — see test_doc_overflow_ceiling_fixes.py's module
docstring for why source inspection (not a full mocked end-to-end call) is
this codebase's established pattern for regression-testing logic embedded
this deeply.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model.persona_supervisor import PersonaSupervisor  # noqa: E402


def _source() -> str:
    return inspect.getsource(PersonaSupervisor._ensure_practical_retrieval_for_legal)


@pytest.mark.unit
class TestMainTopicChapterRetrievalPrefersStrongMatch:
    def test_strong_mts_computed_and_preferred_over_plain_max(self):
        source = _source()
        idx = source.find("if _matched_mts:")
        assert idx != -1, "main_topic matched-candidates block not found"
        end = source.find("try:", idx)
        assert end != -1
        body = source[idx:end]
        # The strong subset must be computed via strong_only=True...
        assert "strong_only=True" in body
        # ...and preferred first: max(_strong_mts, ...) wins when non-empty.
        assert "if _strong_mts:" in body
        assert "_target_mt_sv = max(_strong_mts, key=len)" in body
        # Falls back to the full lenient list only in the final else branch.
        assert "_target_mt_sv = max(_matched_mts, key=len)" in body


@pytest.mark.unit
class TestSubTopicOperationTopicChapterRetrievalPrefersStrongMatch:
    def test_strong_sts_and_ots_computed_and_preferred(self):
        source = _source()
        idx = source.find("if _matched_sts or _matched_ots:")
        assert idx != -1, "sub_topic/operation_topic matched-candidates block not found"
        end = source.find("if len(_best_ot) > len(_best_st):", idx)
        assert end != -1
        body = source[idx:end]
        # Both strong subsets must be computed via strong_only=True.
        assert body.count("strong_only=True") == 2
        assert "_strong_sts" in body and "_strong_ots" in body
        # Selection must branch on "any strong match exists" before falling
        # back to the original lenient max(key=len) behavior.
        assert "if _strong_sts or _strong_ots:" in body


@pytest.mark.unit
class TestChapterRetrievalSkipsAmbiguousWeakMatches:
    """Round 2 (live-reproduced): the strong/weak split alone only helps when a
    strong match exists to prefer. "ยื่นรายงานการชำระบัญชี ลช.3 ผ่านระบบ DBD Biz
    Regist ต้องทำยังไง" had no strong match for EITHER the correct chapter (its
    label's parens/wording differ from the query) or the unrelated "dissolve the
    company" chapter — both matched only weakly via the shared "DBD Biz Regist"
    tokens, and max(key=len) picked the wrong one by a 2-character margin. Fix:
    when 2+ distinct labels all matched only weakly, skip the exact-match
    shortcut entirely (leave the target empty) instead of guessing by length."""

    def test_main_topic_ambiguous_weak_matches_skip_the_shortcut(self):
        source = _source()
        idx = source.find("if _matched_mts:")
        assert idx != -1
        end = source.find("try:", idx)
        assert end != -1
        body = source[idx:end]
        assert "elif len(set(_matched_mts)) >= 2:" in body
        assert '_target_mt_sv = ""' in body

    def test_sub_operation_topic_ambiguous_weak_matches_skip_the_shortcut(self):
        source = _source()
        idx = source.find("if _matched_sts or _matched_ots:")
        assert idx != -1
        end = source.find("if len(_best_ot) > len(_best_st):", idx)
        body = source[idx:end]
        assert "_weak_pool = list(dict.fromkeys(_matched_sts + _matched_ots))" in body
        assert "if len(_weak_pool) >= 2:" in body
        assert "_best_st = _best_ot = \"\"" in body

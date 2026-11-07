"""2026-11 fix: the _prompt_max_docs cap in persona_practical.py's handle()
used to slice _all_docs in whatever order it happened to be in — confirmed
via grep that zero .sort() calls ever touched this list. Every doc reaching
this point already carries doc["metadata"]["_sim"] (attached by
_scored_search), so a cap that has to drop documents should keep the
highest-_sim ones, not arbitrary first-seen ones.

2026-11 fix #2 (live-reproduced doc-replacement bug, then root-caused and
fixed differently): raw _sim alone wasn't enough to prevent FOCUS FILTER's
single-dominant branch from losing its one regulatory doc — live case:
_focused_docs = 1 regulatory doc for the dominant license_type + 5
blank-license "know-how" docs the 2026-10 fix always preserves = 6 docs,
but _prompt_max_docs was reset to 5 (LLM_DOCS_MAX_PRACTICAL on that
deployment), so the later generic cap had to drop 1 doc — and because _sim
can be a per-retrieval-batch-normalized score (not reliably comparable
across docs merged from different upstream retrieval calls), it dropped
the regulatory doc and kept all 5 filler docs — "ไม่พบข้อมูล" even though
the fee doc was retrieved.

First attempted fix was a blanket "any non-blank license_type beats any
blank one" tiebreak in the shared _sim_key — REVERTED. That rule creates a
mirror-image regression: a purely general know-how query that happens to
retrieve one irrelevant "stray" named-license doc (retrieval applies no
license_type filter at all) would let that doc unconditionally survive
the cap ahead of genuinely relevant know-how content, since the tiebreak
doesn't distinguish "verified dominant via real head-to-head comparison"
from "the only named doc that happened to show up, for any reason
including noise." _sim_key is back to plain _sim, exactly as before that
bug was ever found.

The real fix instead targets the actual root cause at its source: right
after FOCUS FILTER builds _focused_docs (the curated 1-dominant +
know-how set), the cap restoration now accounts for that set's own size
(`max(LLM_DOCS_MAX_PRACTICAL, len(_focused_docs))`), so the generic cap
downstream never has to drop anything from this already-curated,
query-local, RERANKER_TOP_K-bounded set at all. This fix is scoped inside
the `_is_multi_license_docs` branch (requires >=2 competing named license
types to begin with) — a pure know-how query with at most 1 stray named
doc never enters it, so it can't regress that scenario.

persona_practical.py's handle() is a single ~5000-line method — see
test_doc_overflow_ceiling_fixes.py's module docstring for why source
inspection (not a full mocked end-to-end call) is this codebase's
established pattern for regression-testing logic embedded this deeply.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model.persona_practical import PracticalPersonaService  # noqa: E402


def _sim_key(d):
    s = (d.get("metadata") or {}).get("_sim")
    return s if s is not None else -1


@pytest.mark.unit
class TestDocCapSortKeyBehavior:
    """Direct test of the sort key logic against hand-built doc lists —
    mirrors the exact lambda shape used at all 3 cap sites."""

    def test_keeps_highest_sim_docs_when_capped(self):
        docs = [
            {"metadata": {"license_type": "A", "_sim": 0.70}},
            {"metadata": {"license_type": "A", "_sim": 0.90}},
            {"metadata": {"license_type": "A", "_sim": 0.80}},
        ]
        capped = sorted(docs, key=_sim_key, reverse=True)[:2]
        assert [d["metadata"]["_sim"] for d in capped] == [0.90, 0.80]

    def test_missing_sim_sorts_last_not_first(self):
        docs = [
            {"metadata": {"license_type": "A"}},  # no _sim at all
            {"metadata": {"license_type": "A", "_sim": 0.60}},
        ]
        capped = sorted(docs, key=_sim_key, reverse=True)[:1]
        assert capped[0]["metadata"].get("_sim") == 0.60

    def test_per_license_counter_on_presorted_list_yields_top_n_per_license(self):
        """A subsequence of a globally-sorted list is itself sorted — the
        existing per-license counter (unchanged) should naturally keep each
        license_type's top-N once _all_docs is pre-sorted."""
        docs = [
            {"metadata": {"license_type": "A", "_sim": 0.60}},
            {"metadata": {"license_type": "B", "_sim": 0.95}},
            {"metadata": {"license_type": "A", "_sim": 0.85}},
            {"metadata": {"license_type": "B", "_sim": 0.55}},
            {"metadata": {"license_type": "A", "_sim": 0.90}},
        ]
        sorted_docs = sorted(docs, key=_sim_key, reverse=True)
        cap = 1
        lt_counts: dict = {}
        kept = []
        for d in sorted_docs:
            lt = d["metadata"]["license_type"]
            lt_counts[lt] = lt_counts.get(lt, 0) + 1
            if lt_counts[lt] <= cap:
                kept.append(d)
        kept_by_lt = {d["metadata"]["license_type"]: d["metadata"]["_sim"] for d in kept}
        assert kept_by_lt == {"A": 0.90, "B": 0.95}

    def test_sim_key_has_no_license_type_tiebreak(self):
        """Guard against reintroducing the reverted blanket tiebreak: _sim_key
        must depend only on _sim, never on license_type, so a purely general
        know-how query's cap behavior can't be skewed by an irrelevant stray
        named-license doc outranking genuinely relevant blank-license docs."""
        source = inspect.getsource(PracticalPersonaService.handle)
        start = source.find("def _sim_key(_d):")
        assert start != -1
        end = source.find("\n\n", start)
        sim_key_body = source[start:end if end != -1 else start + 400]
        assert "license_type" not in sim_key_body


@pytest.mark.unit
class TestFocusFilterCapRestorationAccountsForKnowHowDocs:
    """Live-reproduced regression, fixed at the source rather than via a
    sort tiebreak: FOCUS FILTER's single-dominant branch must never reset
    _prompt_max_docs to a value smaller than the curated _focused_docs set
    it just built (1 dominant regulatory doc + however many blank-license
    know-how docs the 2026-10 fix preserves alongside it)."""

    def test_cap_restoration_uses_max_of_default_and_focused_docs_size(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        assert (
            '_prompt_max_docs = max(int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 6)), len(_focused_docs))'
            in source
        )

    def test_cap_restoration_follows_focused_docs_construction(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        focused_idx = source.find("_focused_docs = [")
        restore_idx = source.find(
            '_prompt_max_docs = max(int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 6)), len(_focused_docs))'
        )
        assert focused_idx != -1
        assert restore_idx != -1
        assert focused_idx < restore_idx

    def test_restoration_formula_keeps_all_6_docs_when_cap_is_5(self):
        """Direct arithmetic check against the exact live-reproduced shape:
        6 focused docs (1 regulatory + 5 know-how), default cap 5."""
        llm_docs_max_practical = 5
        focused_docs = list(range(6))
        restored_cap = max(llm_docs_max_practical, len(focused_docs))
        assert restored_cap == 6
        assert restored_cap >= len(focused_docs)


@pytest.mark.unit
class TestDocCapSortWiredIntoHandle:
    def test_sim_key_helper_defined_before_first_cap_site(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        helper_idx = source.find("def _sim_key(_d):")
        first_cap_idx = source.find("if _is_multi_license_docs:")
        assert helper_idx != -1, "_sim_key helper not found in handle()"
        assert first_cap_idx != -1
        assert helper_idx < first_cap_idx

    def test_all_three_cap_sites_sort_before_slicing(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        # Site 1: multi-license path sorts _all_docs itself before the counter loop.
        assert "_all_docs = sorted(_all_docs, key=_sim_key, reverse=True)" in source
        # Sites 2 and 3: direct slices now sort-then-slice instead of a bare slice.
        assert source.count("sorted(_all_docs, key=_sim_key, reverse=True)[:_prompt_max_docs]") == 2

    def test_uncapped_chapter_overview_send_all_path_untouched(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        assert '_LOG.info("[Practical] Chapter overview: sending all %d docs (no dedup)", len(_all_docs))' in source

"""
Regression tests for utils.text_patterns.rescue_top_sim_doc — 2026-11,
hardcoded-logic audit.

Background: persona_practical.py's/persona_academic.py's operation-intent
doc filters classify "new registration vs renewal vs cancel vs edit vs
replacement" from a handful of hardcoded regex patterns against the query
text, then drop any doc whose operation_topic/operation_by_department
disagrees with the matched pattern — the same brittle-classification
failure class as this codebase's own removed _license_query_score (now
replaced with scoring from doc["metadata"]["_sim"]). Those filters already
guard against wiping out ALL docs, but nothing previously guarded against
silently dropping the single BEST doc while lower-relevance ones survive.
This helper closes that gap with a log-only (then optionally enforced)
rescue, mirroring REGULATORY_IDENTITY_GUARD_ENFORCE's soft-launch shape.
"""
from __future__ import annotations

import logging

import pytest

from utils.text_patterns import rescue_top_sim_doc


def _doc(content: str, sim=None) -> dict:
    meta = {} if sim is None else {"_sim": sim}
    return {"content": content, "metadata": meta}


@pytest.mark.unit
class TestRescueTopSimDoc:
    def test_top_doc_already_present_is_a_true_noop(self):
        top = _doc("the best answer content here", sim=0.95)
        pre = [top, _doc("other", sim=0.40)]
        post = [top]
        out = rescue_top_sim_doc(pre, post, enforce=True)
        assert out == post

    def test_top_doc_missing_and_enforce_true_prepends_it(self, caplog):
        top = _doc("the best answer content here", sim=0.95)
        other = _doc("survivor doc", sim=0.40)
        pre = [top, other]
        post = [other]  # top got filtered out
        with caplog.at_level(logging.WARNING):
            out = rescue_top_sim_doc(pre, post, enforce=True, log_label="[Test]")
        assert out[0] == top
        assert other in out
        assert any("dropped top-_sim doc" in r.message for r in caplog.records)

    def test_top_doc_missing_and_enforce_false_logs_but_leaves_unchanged(self, caplog):
        top = _doc("the best answer content here", sim=0.95)
        other = _doc("survivor doc", sim=0.40)
        pre = [top, other]
        post = [other]
        with caplog.at_level(logging.WARNING):
            out = rescue_top_sim_doc(pre, post, enforce=False, log_label="[Test]")
        assert out == post
        assert any("would rescue (enforce=False)" in r.message for r in caplog.records)

    def test_all_sim_none_is_a_noop_no_signal_to_rescue_on(self, caplog):
        a = _doc("doc a")  # no _sim at all
        b = _doc("doc b")
        pre = [a, b]
        post = [b]
        with caplog.at_level(logging.WARNING):
            out = rescue_top_sim_doc(pre, post, enforce=True)
        assert out == post
        assert not any("dropped top-_sim doc" in r.message for r in caplog.records)

    def test_empty_pre_filter_docs_is_a_noop(self):
        post = [_doc("whatever", sim=0.5)]
        assert rescue_top_sim_doc([], post, enforce=True) == post

    def test_empty_post_filter_docs_is_a_noop(self):
        """Mirrors the existing 'don't apply the filter if it wipes out
        everything' safety net already present at every call site — this
        helper must never try to rescue into an intentionally-empty result,
        that's a different, already-handled case."""
        pre = [_doc("whatever", sim=0.5)]
        assert rescue_top_sim_doc(pre, [], enforce=True) == []

    def test_partial_sim_scored_docs_still_finds_the_real_max(self):
        scored = _doc("scored doc", sim=0.80)
        unscored = _doc("unscored doc")  # no _sim
        pre = [unscored, scored]
        post = [unscored]  # the actually-best (scored) doc got dropped
        out = rescue_top_sim_doc(pre, post, enforce=True)
        assert scored in out

    def test_exception_inside_helper_leaves_post_filter_docs_unchanged(self, caplog):
        """A malformed doc (e.g. metadata is not a dict-like) must never
        crash the caller — fail safe to the pre-existing filtered result."""
        bad_pre = [{"content": "x", "metadata": object()}]
        post = [_doc("survivor", sim=0.2)]
        with caplog.at_level(logging.WARNING):
            out = rescue_top_sim_doc(bad_pre, post, enforce=True, log_label="[Test]")
        assert out == post

"""2026-11: 3 independently-duplicated call sites (persona_practical.py's
_scored_search, persona_academic.py's _ac_search, persona_supervisor.py's
_sv_search) all silently fell back to a fully UNFILTERED vector search
whenever a FILTERED search raised any exception — returning the wrong-scope
document set with no visible error (only a _LOG.debug line). Not a live bug
(every filter operator in use today translates correctly via
utils/es_filter_translate.py), but a landmine: any future filter shape the
ES translator doesn't yet handle would silently widen retrieval scope
instead of erroring, in a legal/regulatory-fee advisory bot with a
documented history of exactly this failure class (5 rounds of "entity-type
retrieval contamination" fixes).

Fix: utils.text_patterns.unfiltered_fallback_is_safe(flt) — only safe to
widen to unfiltered when no filter was ever meaningfully requested. When a
filter WAS requested and every filtered attempt failed, fail closed (loud
ERROR log, return [] instead of substituting wrong-scope results) — letting
each persona's own pre-existing, deliberate "filter matched 0 docs -> widen"
cascade (a separate, correct, already-tested mechanism) decide what to do
next, rather than the low-level search helper silently deciding it alone.

persona_practical.py's downstream cascade never calls self.retriever.invoke
directly for its own widening steps (confirmed in
test_retrieve_docs_entity_type_fallback.py — always re-tries via
similarity_search_with_relevance_scores with a different filter arg), so
asserting it's never called is a valid, meaningful signal there.
persona_academic.py's cascade DOES call self.retriever.invoke directly for
its own legitimate unfiltered retry, so that assertion doesn't apply there
— the ERROR log is the direct, unambiguous signal that the fix's code path
ran instead of the old silent-swallow behavior.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import conf
from model.persona_academic import AcademicPersonaService
from model.persona_practical import PracticalPersonaService


@pytest.fixture
def practical_service():
    svc = PracticalPersonaService(retriever=MagicMock())
    svc.retriever.vectorstore = MagicMock()
    return svc


@pytest.fixture
def academic_service():
    svc = AcademicPersonaService(retriever=MagicMock())
    svc.retriever.vectorstore = MagicMock()
    return svc


@pytest.mark.unit
class TestPracticalFailsClosedOnFilteredException:
    def test_filtered_search_exception_fails_closed_not_silently_unfiltered(self, practical_service, caplog, monkeypatch):
        # Force the plain (non-hybrid) path deterministically — this
        # environment's local .env may have HYBRID_SEARCH_ENABLED=true,
        # which would route through utils.hybrid_retriever instead of the
        # similarity_search_with_relevance_scores/similarity_search chain
        # this test is specifically about.
        monkeypatch.setattr(conf, "HYBRID_SEARCH_ENABLED", False)

        def _raise(*a, **k):
            raise RuntimeError("simulated filter translation failure")

        practical_service.retriever.vectorstore.similarity_search_with_relevance_scores.side_effect = _raise
        practical_service.retriever.vectorstore.similarity_search.side_effect = _raise

        with patch("model.persona_practical._qr_needs_rewrite", return_value=False), caplog.at_level("ERROR"):
            practical_service._retrieve_docs(
                "some query",
                metadata_filter={"license_type": "X"},
                max_docs=5,
            )

        assert any("failing closed" in r.message for r in caplog.records), (
            "expected the fail-closed ERROR log when every filtered attempt "
            "raised while a real filter was requested"
        )
        # The ERROR log above proves _scored_search itself returned [] rather
        # than silently substituting unfiltered results FROM WITHIN its own
        # exception handler. _retrieve_docs' own OUTER cascade is a separate,
        # legitimate, already-tested mechanism (test_retrieve_docs_entity_
        # type_fallback.py) that may still retry unfiltered afterward — that
        # retry being visible/deliberate (vs. silent) is the whole point of
        # this fix, not something this test needs to suppress.

    def test_no_filter_requested_fail_closed_path_never_triggers(self, practical_service, monkeypatch, caplog):
        """Sanity check: the fix must not change behavior for the pre-
        existing, perfectly safe case where no filter was ever requested —
        the new fail-closed ERROR log must never fire here."""
        monkeypatch.setattr(conf, "HYBRID_SEARCH_ENABLED", False)

        def _raise(*a, **k):
            raise RuntimeError("transient failure")

        practical_service.retriever.vectorstore.similarity_search_with_relevance_scores.side_effect = _raise
        practical_service.retriever.vectorstore.similarity_search.side_effect = _raise
        practical_service.retriever.invoke.return_value = []

        with patch("model.persona_practical._qr_needs_rewrite", return_value=False), caplog.at_level("ERROR"):
            practical_service._retrieve_docs("some query", metadata_filter=None, max_docs=5)

        assert not any("failing closed" in r.message for r in caplog.records), (
            "the fail-closed path must never trigger when no filter was ever requested"
        )


@pytest.mark.unit
class TestAcademicFailsClosedOnFilteredException:
    def test_filtered_search_exception_fails_closed_not_silently_unfiltered(self, academic_service, caplog, monkeypatch):
        monkeypatch.setattr(conf, "HYBRID_SEARCH_ENABLED", False)
        _retriever_mock = MagicMock()
        _retriever_mock.invoke.side_effect = RuntimeError("simulated filter translation failure")
        academic_service.retriever.vectorstore.as_retriever.return_value = _retriever_mock

        with caplog.at_level("ERROR"):
            academic_service._retrieve_docs(
                "some query",
                metadata_filter={"license_type": "X"},
            )

        assert any("failing closed" in r.message for r in caplog.records), (
            "expected the fail-closed ERROR log when the filtered "
            "as_retriever().invoke() attempt raised while a real filter was requested"
        )

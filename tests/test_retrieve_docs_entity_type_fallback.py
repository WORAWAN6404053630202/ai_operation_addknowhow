"""
Behavioral tests for the 2026-11 fix to persona_practical.py's _retrieve_docs:
when the full metadata filter (entity_type_normalized + license_type [+
department]) matches zero documents, the method now tries an intermediate
"entity_type_normalized only" fallback BEFORE giving up and going fully
unfiltered.

Background: a real, live-reproduced bug — a user who explicitly selected
"ห้างหุ้นส่วนจำกัด" (which normalizes to entity_type_normalized="นิติบุคคล")
still got "บุคคลธรรมดา" documents in their final answer. Two earlier fix
attempts this session (both verified live via LangSmith traces, not just
assumed) targeted entity_type *derivation* and were confirmed correct —
collected_slots["entity_type"] really was "นิติบุคคล" and the caller
(persona_supervisor.py's operation_group handler) really did build a filter
that included entity_type_normalized. Neither fix changed the outcome,
because the REAL leak was one layer deeper: _retrieve_docs's own "if not
docs: drop ALL filtering" fallback (persona_practical.py, right above where
this fix is inserted) discarded entity_type_normalized along with
license_type/department whenever the combined filter returned zero results
— even though the caller had already deliberately ordered ITS OWN fallback
cascade to drop license_type/department before entity_type, specifically to
avoid exactly this cross-entity-type contamination. This test drives
_retrieve_docs directly (not source-inspection) with a mocked vectorstore,
to prove the actual runtime behavior — a real mocked-data regression test
after two assumption-based misses, not another assumption.

PracticalPersonaService(retriever=MagicMock()) with retriever.vectorstore
also mocked — no real network/Chroma access. HYBRID_SEARCH_ENABLED defaults
to False in this test environment, so _scored_search's plain
vectorstore.similarity_search_with_relevance_scores(...) path is what's
exercised (confirmed via conf.py's default).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from model.persona_practical import PracticalPersonaService


def _doc(entity_type: str, content: str = "doc content"):
    return SimpleNamespace(page_content=content, metadata={"entity_type_normalized": entity_type})


@pytest.fixture
def service():
    svc = PracticalPersonaService(retriever=MagicMock())
    svc.retriever.vectorstore = MagicMock()
    return svc


@pytest.mark.unit
class TestEntityTypeOnlyFallback:
    def test_full_filter_empty_falls_back_to_entity_only_not_fully_unfiltered(self, service):
        """The exact bug scenario: entity_type_normalized + license_type filter
        finds nothing, but entity_type_normalized alone WOULD find real
        นิติบุคคล docs — the fix must use those, not fall through to a fully
        unfiltered search that could return any entity type."""
        _juristic_doc = _doc("นิติบุคคล", "ห้างหุ้นส่วนจำกัด content")

        call_filters = []

        def _side_effect(q, **kwargs):
            call_filters.append(kwargs.get("filter"))
            if len(call_filters) == 1:
                return []  # full filter (entity_type + license_type): nothing matches
            return [(_juristic_doc, 0.8)]  # entity-only retry: real matches

        service.retriever.vectorstore.similarity_search_with_relevance_scores.side_effect = _side_effect

        with patch("model.persona_practical._qr_needs_rewrite", return_value=False):
            docs = service._retrieve_docs(
                "ยื่นขอใหม่ใบทะเบียนพาณิชย์",
                metadata_filter={"entity_type_normalized": "นิติบุคคล", "license_type": "ใบทะเบียนพาณิชย์"},
                max_docs=5,
            )

        assert len(docs) == 1
        assert docs[0]["metadata"]["entity_type_normalized"] == "นิติบุคคล"
        # Exactly 2 calls: the full filter (empty), then the entity-only retry
        # that succeeded — must NOT reach a 3rd fully-unfiltered call.
        assert service.retriever.vectorstore.similarity_search_with_relevance_scores.call_count == 2

    def test_entity_only_also_empty_still_falls_through_to_fully_unfiltered(self, service):
        """Safety net preserved: if even entity-only genuinely finds nothing,
        the method must still fall back to a fully unfiltered search rather
        than returning zero docs and breaking the answer entirely."""
        _any_doc = _doc("บุคคลธรรมดา", "fallback content")

        call_filters = []

        def _side_effect(q, **kwargs):
            call_filters.append(kwargs.get("filter"))
            if len(call_filters) <= 2:
                return []  # full filter, then entity-only — both empty
            return [(_any_doc, 0.5)]  # fully unfiltered retry

        service.retriever.vectorstore.similarity_search_with_relevance_scores.side_effect = _side_effect

        with patch("model.persona_practical._qr_needs_rewrite", return_value=False):
            docs = service._retrieve_docs(
                "ยื่นขอใหม่ใบทะเบียนพาณิชย์",
                metadata_filter={"entity_type_normalized": "นิติบุคคล", "license_type": "ใบทะเบียนพาณิชย์"},
                max_docs=5,
            )

        assert len(docs) == 1
        assert service.retriever.vectorstore.similarity_search_with_relevance_scores.call_count == 3

    def test_filter_without_entity_type_skips_the_new_fallback_entirely(self, service):
        """A filter with no entity_type_normalized at all (e.g. license_type-only
        callers elsewhere) must go straight to the existing fully-unfiltered
        fallback — the new intermediate stage must not fire/change anything
        for callers that never had an entity_type to preserve in the first place."""
        _any_doc = _doc("บุคคลธรรมดา", "fallback content")

        call_filters = []

        def _side_effect(q, **kwargs):
            call_filters.append(kwargs.get("filter"))
            if len(call_filters) == 1:
                return []
            return [(_any_doc, 0.5)]

        service.retriever.vectorstore.similarity_search_with_relevance_scores.side_effect = _side_effect

        with patch("model.persona_practical._qr_needs_rewrite", return_value=False):
            docs = service._retrieve_docs(
                "ยื่นขอใหม่ใบทะเบียนพาณิชย์",
                metadata_filter={"license_type": "ใบทะเบียนพาณิชย์"},
                max_docs=5,
            )

        assert len(docs) == 1
        # Exactly 2 calls: original filter (empty), straight to fully unfiltered.
        assert service.retriever.vectorstore.similarity_search_with_relevance_scores.call_count == 2

    def test_entity_type_inside_and_wrapped_filter_is_also_found(self, service):
        """Some callers pass an already-$and-wrapped filter (no location key,
        so the existing location-stripping fallback stays a no-op and doesn't
        confound this test) — the fix must extract entity_type_normalized
        from inside the $and list too, not only from a flat top-level key."""
        _juristic_doc = _doc("นิติบุคคล", "ห้างหุ้นส่วนจำกัด content")

        call_filters = []

        def _side_effect(q, **kwargs):
            call_filters.append(kwargs.get("filter"))
            if len(call_filters) == 1:
                return []
            return [(_juristic_doc, 0.8)]

        service.retriever.vectorstore.similarity_search_with_relevance_scores.side_effect = _side_effect

        with patch("model.persona_practical._qr_needs_rewrite", return_value=False):
            docs = service._retrieve_docs(
                "ยื่นขอใหม่ใบทะเบียนพาณิชย์",
                metadata_filter={"$and": [
                    {"entity_type_normalized": "นิติบุคคล"},
                    {"license_type": "ใบทะเบียนพาณิชย์"},
                    {"department": "กรมพัฒนาธุรกิจการค้า"},
                ]},
                max_docs=5,
            )

        assert len(docs) == 1
        assert docs[0]["metadata"]["entity_type_normalized"] == "นิติบุคคล"
        # No location/area_size key present, so the existing location-stripping
        # fallback must stay a no-op — exactly 2 calls: original (empty), then
        # this fix's entity-only retry (found).
        assert service.retriever.vectorstore.similarity_search_with_relevance_scores.call_count == 2

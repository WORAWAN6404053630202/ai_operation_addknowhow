"""
Unit tests for utils.es_filter_translate.chroma_filter_to_es_query — pure
dict-in/dict-out translation, no Elasticsearch connection required. Part of
the 2026-11 Elasticsearch-parallel-project work (see the approved plan at
.claude/plans/mutable-seeking-wave.md, Phase 1): this translator is what lets
service/es_vector_store.py's Chroma-shaped facade accept the exact same
filter dicts persona_supervisor.py/persona_practical.py/hybrid_retriever.py
already build, unmodified.
"""
from __future__ import annotations

import pytest

from utils.es_filter_translate import chroma_filter_to_es_query


@pytest.mark.unit
class TestNoFilter:
    def test_none_filter_matches_everything(self):
        assert chroma_filter_to_es_query(None) == {"match_all": {}}

    def test_empty_dict_filter_matches_everything(self):
        assert chroma_filter_to_es_query({}) == {"match_all": {}}


@pytest.mark.unit
class TestFlatExactMatch:
    def test_single_field_becomes_term_query(self):
        assert chroma_filter_to_es_query({"license_type": "ใบทะเบียนพาณิชย์"}) == {
            "term": {"license_type": "ใบทะเบียนพาณิชย์"}
        }

    def test_multi_field_flat_dict_is_implicit_and(self):
        """Chroma requires wrapping multi-field flat dicts in "$and" itself
        (exactly-one-top-level-operator rule) — this translator accepts the
        UNWRAPPED flat shape directly too, since ES's bool.filter has no such
        restriction, so a caller that forgets the $and wrapper still works."""
        out = chroma_filter_to_es_query({"license_type": "ใบทะเบียนพาณิชย์", "department": "DBD"})
        assert out == {
            "bool": {
                "filter": [
                    {"term": {"license_type": "ใบทะเบียนพาณิชย์"}},
                    {"term": {"department": "DBD"}},
                ]
            }
        }


@pytest.mark.unit
class TestAndOperator:
    def test_explicit_and_wrapping_same_as_flat_multi_field(self):
        wrapped = chroma_filter_to_es_query(
            {"$and": [{"license_type": "ใบทะเบียนพาณิชย์"}, {"department": "DBD"}]}
        )
        flat = chroma_filter_to_es_query({"license_type": "ใบทะเบียนพาณิชย์", "department": "DBD"})
        assert wrapped == flat

    def test_and_with_three_clauses(self):
        out = chroma_filter_to_es_query(
            {"$and": [{"license_type": "x"}, {"department": "y"}, {"entity_type_normalized": "z"}]}
        )
        assert out == {
            "bool": {
                "filter": [
                    {"term": {"license_type": "x"}},
                    {"term": {"department": "y"}},
                    {"term": {"entity_type_normalized": "z"}},
                ]
            }
        }


@pytest.mark.unit
class TestOrOperator:
    def test_or_becomes_should_with_minimum_should_match(self):
        out = chroma_filter_to_es_query({"$or": [{"license_type": "a"}, {"license_type": "b"}]})
        assert out == {
            "bool": {
                "should": [{"term": {"license_type": "a"}}, {"term": {"license_type": "b"}}],
                "minimum_should_match": 1,
            }
        }


@pytest.mark.unit
class TestInOperator:
    def test_in_becomes_terms_query(self):
        out = chroma_filter_to_es_query({"license_type": {"$in": ["a", "b", "c"]}})
        assert out == {"terms": {"license_type": ["a", "b", "c"]}}

    def test_entity_neutral_widening_pattern(self):
        """The exact real-world shape persona_supervisor.py builds to widen
        an entity-specific filter to also include entity-neutral docs (empty
        string = applies to every entity type) — this is the single most
        load-bearing filter pattern in the live retrieval path, carried over
        directly from Chroma's $in usage."""
        out = chroma_filter_to_es_query({"entity_type_normalized": {"$in": ["นิติบุคคล", ""]}})
        assert out == {"terms": {"entity_type_normalized": ["นิติบุคคล", ""]}}

    def test_in_combined_with_other_fields_via_and(self):
        out = chroma_filter_to_es_query(
            {"$and": [{"license_type": "ใบทะเบียนพาณิชย์"}, {"entity_type_normalized": {"$in": ["นิติบุคคล", ""]}}]}
        )
        assert out == {
            "bool": {
                "filter": [
                    {"term": {"license_type": "ใบทะเบียนพาณิชย์"}},
                    {"terms": {"entity_type_normalized": ["นิติบุคคล", ""]}},
                ]
            }
        }


@pytest.mark.unit
class TestEqNeOperators:
    def test_eq_becomes_term_query(self):
        assert chroma_filter_to_es_query({"license_type": {"$eq": "x"}}) == {"term": {"license_type": "x"}}

    def test_ne_becomes_must_not_term_query(self):
        out = chroma_filter_to_es_query({"license_type": {"$ne": "x"}})
        assert out == {"bool": {"must_not": [{"term": {"license_type": "x"}}]}}


@pytest.mark.unit
class TestUnsupportedOperator:
    def test_unknown_field_operator_raises_clear_error(self):
        with pytest.raises(ValueError, match="Unsupported field-level operator"):
            chroma_filter_to_es_query({"license_type": {"$gt": "x"}})

# code/utils/es_filter_translate.py
"""Translates this project's Chroma-style metadata filter dicts into
Elasticsearch query DSL — the shared piece `service/es_vector_store.py`'s
`ESCompatCollection.get()`/`.query()` both need to accept the exact same
filter shapes every existing call site already builds (persona_supervisor.py,
persona_practical.py, hybrid_retriever.py, service/local_vector_store.py's
own delete_by_source_file) unmodified.

Vocabulary covered (confirmed via a full repo grep, 2026-11 ES-parallel-
project planning — see the plan's "Chroma filter vocabulary" section):
- flat {field: value} exact match, single or multiple fields (multiple
  fields are an implicit AND — Chroma requires wrapping these in "$and"
  itself since it allows only one top-level operator per filter; ES's
  bool.filter has no such restriction, so this translator accepts BOTH the
  pre-wrapped {"$and": [...]} shape AND a flat multi-key dict directly).
- {"$and": [{...}, {...}]} — combined AND.
- {"$or": [{...}, {...}]} — combined OR (not observed built by persona code
  as of this writing, but hybrid_retriever.py's own filter-matcher supports
  it defensively; kept here for the same reason).
- {field: {"$in": [v1, v2, ...]}} — "any of these values", including the
  entity-neutral-widening pattern {"entity_type_normalized": {"$in":
  [eff_entity, ""]}} so a doc with an empty string (= applies to every
  entity type) is never excluded by an entity-specific filter.
- {field: {"$eq": v}} / {field: {"$ne": v}} — explicit equality/inequality.

Pure function, no ES connection involved — exercised directly by
tests/test_es_filter_translation.py without a running Elasticsearch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_OPERATOR_KEYS = {"$and", "$or"}
_FIELD_OPERATOR_KEYS = {"$in", "$eq", "$ne"}


def _translate_field_clause(field: str, value: Any) -> Dict[str, Any]:
    """A single {field: value} or {field: {"$op": ...}} pair → one ES leaf query."""
    if isinstance(value, dict):
        if "$in" in value:
            return {"terms": {field: list(value["$in"])}}
        if "$eq" in value:
            return {"term": {field: value["$eq"]}}
        if "$ne" in value:
            return {"bool": {"must_not": [{"term": {field: value["$ne"]}}]}}
        raise ValueError(f"Unsupported field-level operator in filter: {field}={value!r}")
    return {"term": {field: value}}


def _translate_subfilter(sub: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one filter dict (no outer wrapping assumed) into a single ES
    query clause — recurses for nested $and/$or, otherwise treats every key
    as a field (implicit AND across multiple keys, same semantics Chroma's
    own "$and" wrapping already encodes)."""
    if "$and" in sub:
        return {"bool": {"filter": [_translate_subfilter(s) for s in sub["$and"]]}}
    if "$or" in sub:
        return {"bool": {"should": [_translate_subfilter(s) for s in sub["$or"]], "minimum_should_match": 1}}
    clauses = [_translate_field_clause(field, value) for field, value in sub.items()]
    if len(clauses) == 1:
        return clauses[0]
    return {"bool": {"filter": clauses}}


def chroma_filter_to_es_query(filter: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Entry point. `filter=None` or `{}` → match everything (no filter).
    Otherwise returns a single ES query-DSL dict suitable for either the
    `query` clause of a filtered-only search, or combined into a larger
    bool query's `filter`/`must` array by the caller."""
    if not filter:
        return {"match_all": {}}
    return _translate_subfilter(filter)

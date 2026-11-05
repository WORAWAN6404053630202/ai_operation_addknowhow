"""
Shared Thai word-segmentation helper for Elasticsearch's BM25/match leg.

2026-11: ES's own `standard` analyzer barely recognizes Thai word
boundaries at all (a whole Thai sentence can come back as one token), and
its built-in `thai` analyzer (JVM's java.text.BreakIterator) produces real
segmentation errors on this corpus's legal/regulatory vocabulary (confirmed
live via ES's `_analyze` API — e.g. "สถานที่" loses "ที่" and becomes
"สถาน"; "ตรวจสอบ" wrongly splits into "ตรวจ"+"สอบ"). PyThaiNLP's `newmm`
engine segments both of these correctly and is already proven in this
codebase for the Chroma-backed BM25 index (utils/hybrid_retriever.py) — ES's
`match_search()` was written fresh for scale and missed this step entirely
when the ES branch was built. This module exists so the SAME tokenization
runs at both index time (service/pdf_direct_ingest.py, writing
`page_content_tokenized`) and query time
(service/es_vector_store.py:ESCompatCollection.match_search), guaranteeing
the two sides always agree on word boundaries.

Not merged with the two pre-existing, near-identical private tokenizers
(utils/hybrid_retriever.py:_tokenize, utils/text_patterns.py:
extract_thai_keywords) — both already work correctly for their own
call sites; refactoring them carries regression risk this change doesn't
need to take on.
"""
from __future__ import annotations

import logging

_LOG = logging.getLogger(__name__)


def tokenize_thai_for_search(text: str) -> str:
    """Segment `text` into whitespace-joined words for an ES field analyzed
    with `"analyzer": "whitespace"`. Falls back to the original text
    unchanged if PyThaiNLP is unavailable — ES's `whitespace` analyzer then
    just sees it as however many words were already space-separated (same
    degraded-but-safe behavior as before this field existed)."""
    try:
        from pythainlp.tokenize import word_tokenize
        tokens = word_tokenize(str(text or ""), engine="newmm", keep_whitespace=False)
        tokens = [t for t in tokens if t and t.strip()]
        return " ".join(tokens)
    except Exception as _e:
        _LOG.debug("[thai_tokenize] PyThaiNLP unavailable (%s) — using text unchanged", type(_e).__name__)
        return str(text or "")

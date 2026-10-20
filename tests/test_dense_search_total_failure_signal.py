"""Test for the 2026-10 fix (core2-N4) to utils/hybrid_retriever.py's
hybrid_scored_search — a total dense-retrieval failure (all 3 attempts)
used to be logged at WARNING, the same level as a routine single-attempt
retry just above it, and nothing distinguished "this query answered from
a thin/degraded BM25-only pool" from a normal successful hybrid search.
Upgraded to ERROR with a distinct, greppable marker
(DENSE_SEARCH_TOTAL_FAILURE) and a DENSE_DEGRADED flag on the final
fused-result summary log line.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils import hybrid_retriever as hr  # noqa: E402


def _fake_vectorstore_always_failing():
    vs = MagicMock()
    vs.similarity_search_with_relevance_scores.side_effect = RuntimeError("OpenRouter 422")
    return vs


@pytest.mark.unit
class TestDenseSearchTotalFailureIsLoudlySignaled:
    def test_total_dense_failure_logs_at_error_with_distinct_marker(self, caplog):
        vs = _fake_vectorstore_always_failing()
        with patch.object(hr, "_get_bm25_index", return_value=None):
            with caplog.at_level(logging.ERROR, logger=hr._LOG.name):
                result = hr.hybrid_scored_search(vs, "คำถามทดสอบ", k=5)

        assert result == []  # BM25 unavailable too -> empty, but must not raise
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("DENSE_SEARCH_TOTAL_FAILURE" in r.getMessage() for r in error_records)

    def test_total_dense_failure_marks_fused_summary_as_degraded(self, caplog):
        vs = _fake_vectorstore_always_failing()
        fake_bm25_index = MagicMock()
        fake_bm25_index.search.return_value = [(Document(page_content="doc1", metadata={}), 10.0)]
        with patch.object(hr, "_get_bm25_index", return_value=fake_bm25_index):
            with caplog.at_level(logging.INFO, logger=hr._LOG.name):
                result = hr.hybrid_scored_search(vs, "คำถามทดสอบ", k=5)

        assert len(result) == 1  # BM25-only result still returned, degraded but not empty
        info_records = [r for r in caplog.records if "dense=" in r.getMessage()]
        assert info_records, "expected the fused-result summary log line"
        assert "DENSE_DEGRADED=True" in info_records[-1].getMessage()

    def test_successful_dense_search_is_not_marked_degraded(self, caplog):
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="doc1", metadata={}), 0.9),
        ]
        fake_bm25_index = MagicMock()
        fake_bm25_index.search.return_value = [(Document(page_content="doc1", metadata={}), 10.0)]
        with patch.object(hr, "_get_bm25_index", return_value=fake_bm25_index):
            with caplog.at_level(logging.INFO, logger=hr._LOG.name):
                hr.hybrid_scored_search(vs, "คำถามทดสอบ", k=5)

        info_records = [r for r in caplog.records if "dense=" in r.getMessage()]
        assert info_records
        assert "DENSE_DEGRADED" not in info_records[-1].getMessage()

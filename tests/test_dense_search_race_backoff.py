"""2026-10 fix: hybrid_scored_search's dense-search retry loop used to retry
with zero delay — fine for the stateless OpenRouter-422 flakiness it was
originally built for (see the loop's header comment), but live-reproduced
against this project's own local (SQLite-backed) Chroma store, a dense
search running at the exact same moment sqs_consumer.py was writing a new
PDF-ingested doc into the SAME file failed with a Pydantic "Document
page_content must be a valid string [input_value=None]" error — a
cross-process read/write race, not provider flakiness. A zero-delay retry
gives a concurrent writer's transaction no extra wall-clock time to land
before the next attempt reads the same inconsistent state again.

Fix: a small growing backoff (0.15s × attempt number) before each retry
(not before the first attempt, not after the final exhausted attempt).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils import hybrid_retriever as hr  # noqa: E402


@pytest.mark.unit
class TestDenseSearchRetryBackoff:
    def test_sleeps_with_growing_backoff_between_retries(self):
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.side_effect = [
            RuntimeError("Document page_content must be a valid string [input_value=None]"),
            RuntimeError("Document page_content must be a valid string [input_value=None]"),
            [(Document(page_content="doc1", metadata={}), 0.9)],  # 3rd attempt succeeds
        ]
        with patch.object(hr, "_get_bm25_index", return_value=None), \
             patch("utils.hybrid_retriever.time.sleep") as mock_sleep:
            result = hr.hybrid_scored_search(vs, "คำถามทดสอบ", k=5)

        assert len(result) == 1
        # Exactly 2 sleeps (before retry 2 and retry 3), growing: 0.15*1, 0.15*2.
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.15 * 1)
        mock_sleep.assert_any_call(0.15 * 2)

    def test_no_sleep_when_first_attempt_succeeds(self):
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="doc1", metadata={}), 0.9),
        ]
        with patch.object(hr, "_get_bm25_index", return_value=None), \
             patch("utils.hybrid_retriever.time.sleep") as mock_sleep:
            hr.hybrid_scored_search(vs, "คำถามทดสอบ", k=5)

        mock_sleep.assert_not_called()

    def test_no_sleep_after_final_exhausted_attempt(self):
        # All 3 attempts fail — sleep should happen only before retries 2 and 3,
        # never after the 3rd (final, non-retried) failure.
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.side_effect = RuntimeError("still failing")
        with patch.object(hr, "_get_bm25_index", return_value=None), \
             patch("utils.hybrid_retriever.time.sleep") as mock_sleep:
            result = hr.hybrid_scored_search(vs, "คำถามทดสอบ", k=5)

        assert result == []
        assert mock_sleep.call_count == 2

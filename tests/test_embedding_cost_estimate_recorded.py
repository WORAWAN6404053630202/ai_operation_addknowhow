"""Test for the 2026-10 fix (PDF2-N6) to service/local_vector_store.py's
two real billed OpenRouter embedding call sites (embed_documents in
add_documents, embed_query in query_live_with_ids) — neither recorded
into the cost dashboard, unlike every other LLM call in the PDF pipeline.
LangChain's embedding wrapper doesn't surface token usage from the
underlying API call, so this estimates tokens from character count and
records that estimate via utils.pdf_cost_guard.get_cost_guard() the same
way a real usage-based call would.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from service.local_vector_store import _record_embedding_cost_estimate  # noqa: E402


@pytest.mark.unit
class TestRecordEmbeddingCostEstimate:
    def test_records_a_nonzero_cost_for_real_text(self):
        mock_guard = MagicMock()
        with patch("utils.pdf_cost_guard.get_cost_guard", return_value=mock_guard):
            _record_embedding_cost_estimate("Test/embed", ["a" * 4000])
        mock_guard.record_cost.assert_called_once()
        recorded_cost = mock_guard.record_cost.call_args[0][0]
        assert recorded_cost > 0.0

    def test_longer_text_costs_more(self):
        mock_guard = MagicMock()
        with patch("utils.pdf_cost_guard.get_cost_guard", return_value=mock_guard):
            _record_embedding_cost_estimate("Test/embed", ["a" * 1000])
        short_cost = mock_guard.record_cost.call_args[0][0]

        mock_guard2 = MagicMock()
        with patch("utils.pdf_cost_guard.get_cost_guard", return_value=mock_guard2):
            _record_embedding_cost_estimate("Test/embed", ["a" * 100_000])
        long_cost = mock_guard2.record_cost.call_args[0][0]

        assert long_cost > short_cost

    def test_multiple_texts_are_summed(self):
        mock_guard = MagicMock()
        with patch("utils.pdf_cost_guard.get_cost_guard", return_value=mock_guard):
            _record_embedding_cost_estimate("Test/embed", ["a" * 1000, "b" * 1000, "c" * 1000])
        cost_many = mock_guard.record_cost.call_args[0][0]

        mock_guard2 = MagicMock()
        with patch("utils.pdf_cost_guard.get_cost_guard", return_value=mock_guard2):
            _record_embedding_cost_estimate("Test/embed", ["a" * 1000])
        cost_one = mock_guard2.record_cost.call_args[0][0]

        assert cost_many == pytest.approx(cost_one * 3, rel=0.01)

    def test_failure_is_swallowed_not_raised(self):
        with patch("utils.pdf_cost_guard.get_cost_guard", side_effect=RuntimeError("boom")):
            _record_embedding_cost_estimate("Test/embed", ["text"])  # must not raise

    def test_add_documents_records_cost_for_embedded_text(self):
        from service.local_vector_store import LocalVectorStoreManager
        from langchain_core.documents import Document

        mgr = LocalVectorStoreManager()
        mgr.vectorstore = MagicMock()
        mgr.vectorstore.add_documents = MagicMock()
        mgr.vectorstore.persist = MagicMock()
        mgr._collection_count = MagicMock(return_value=1)

        with patch("service.local_vector_store._record_embedding_cost_estimate") as mock_record:
            mgr.add_documents([Document(page_content="เนื้อหาเอกสาร", metadata={})])

        mock_record.assert_called_once()
        assert mock_record.call_args[0][0] == "VectorStore/embed_documents"

    def test_query_live_with_ids_records_cost_for_the_query(self):
        from service.local_vector_store import LocalVectorStoreManager

        mgr = LocalVectorStoreManager()
        mgr.vectorstore = MagicMock()
        mgr.embedding_model = MagicMock()
        mgr.embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mgr.vectorstore._collection.query.return_value = {
            "ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]],
        }

        with patch("service.local_vector_store._record_embedding_cost_estimate") as mock_record:
            mgr.query_live_with_ids("คำถามทดสอบ", k=5)

        mock_record.assert_called_once()
        assert mock_record.call_args[0][0] == "VectorStore/embed_query"

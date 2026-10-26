"""Test for the 2026-10 fix to service/local_vector_store.py's add_documents
— live-reproduced on a real PDF ("ยื่น_เพิ่ม_ชำระ_ยกเลิก ภาษีป้าย.pdf"):
each Document's page_content was already kept under baai/bge-m3's
8192-token ceiling by its own caller (pdf_direct_ingest.py's
_MAX_MERGED_PAGE_CONTENT_CHARS), confirmed live by logging every
document's length right before the embed call — none exceeded the cap.
Yet the write still failed with the model's own "8192 tokens... at least
8193 input tokens" error, because initialize_embeddings()'s
chunk_size=25 batches up to 25 documents into ONE OpenRouter request, and
this provider applies the ceiling to the SUM of the whole batch, not
per-document — a handful of large-but-individually-valid documents
landing in the same batch of 25 can still blow past it even though no
single one of them would on its own.

add_documents now retries once with chunk_size=1 (one document per
request — the smallest ANY already-capped single document could ever
need) specifically on that error, leaving the fast batched path
untouched for the overwhelming majority of calls that never hit it.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from langchain_core.documents import Document  # noqa: E402

from service.local_vector_store import LocalVectorStoreManager  # noqa: E402


def _manager_with_mocked_vectorstore(persist_dir: str, chunk_size: int = 25) -> tuple[LocalVectorStoreManager, MagicMock]:
    mgr = LocalVectorStoreManager()
    mgr._persist_dir = lambda: persist_dir  # type: ignore[method-assign]
    mock_vectorstore = MagicMock()
    mgr.vectorstore = mock_vectorstore
    mgr.embedding_model = MagicMock()
    mgr.embedding_model.chunk_size = chunk_size
    return mgr, mock_vectorstore


_CONTEXT_LENGTH_ERROR = RuntimeError(
    "Error code: 400 - {'error': {'message': \"This model's maximum context "
    "length is 8192 tokens. However, you requested 0 output tokens and your "
    "prompt contains at least 8193 input tokens, for a total of at least "
    "8193 tokens...\", 'code': 400}}"
)


@pytest.mark.unit
class TestAddDocumentsRetriesOnBatchContextLengthOverflow:
    def test_context_length_failure_retries_with_chunk_size_one_and_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, mock_vs = _manager_with_mocked_vectorstore(tmp, chunk_size=25)
            mock_vs.add_documents.side_effect = [_CONTEXT_LENGTH_ERROR, None]  # fail once, then succeed
            mock_vs._collection.count.return_value = 1

            docs = [Document(page_content="เนื้อหาทดสอบ", metadata={"source_file": "f.pdf"})]
            result = mgr.add_documents(docs)

        assert result == 1
        assert mock_vs.add_documents.call_count == 2

    def test_chunk_size_is_set_to_1_during_the_retry_call(self):
        """The actual mechanism under test: chunk_size must be 1 at the
        moment the SECOND add_documents call happens, proving the retry
        really does force one-document-per-request rather than just
        calling the exact same thing twice and hoping it works."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr, mock_vs = _manager_with_mocked_vectorstore(tmp, chunk_size=25)
            observed_chunk_sizes = []

            def _side_effect(docs):
                observed_chunk_sizes.append(mgr.embedding_model.chunk_size)
                if len(observed_chunk_sizes) == 1:
                    raise _CONTEXT_LENGTH_ERROR
                return None

            mock_vs.add_documents.side_effect = _side_effect
            mock_vs._collection.count.return_value = 1

            docs = [Document(page_content="เนื้อหาทดสอบ", metadata={})]
            mgr.add_documents(docs)

        assert observed_chunk_sizes == [25, 1]

    def test_chunk_size_is_restored_to_its_original_value_after_a_successful_retry(self):
        """The retry's chunk_size=1 override must not leak into every
        SUBSEQUENT call for the rest of the process's life — only this one
        retry needs it."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr, mock_vs = _manager_with_mocked_vectorstore(tmp, chunk_size=25)
            mock_vs.add_documents.side_effect = [_CONTEXT_LENGTH_ERROR, None]
            mock_vs._collection.count.return_value = 1

            mgr.add_documents([Document(page_content="x", metadata={})])

        assert mgr.embedding_model.chunk_size == 25

    def test_chunk_size_is_restored_even_if_the_retry_itself_also_fails(self):
        """finally-block guard: a retry that STILL fails (e.g. a genuinely
        oversized single document, a different error entirely) must not
        leave the manager permanently stuck at chunk_size=1."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr, mock_vs = _manager_with_mocked_vectorstore(tmp, chunk_size=25)
            mock_vs.add_documents.side_effect = [_CONTEXT_LENGTH_ERROR, _CONTEXT_LENGTH_ERROR]

            with pytest.raises(RuntimeError):
                mgr.add_documents([Document(page_content="x", metadata={})])

        assert mgr.embedding_model.chunk_size == 25

    def test_an_unrelated_error_is_not_retried_and_propagates_immediately(self):
        """Only the specific context-length failure gets the retry — any
        other add_documents failure (network error, auth error, ...) must
        surface immediately, not be silently masked behind a pointless
        chunk_size=1 retry that would fail for the same unrelated reason."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr, mock_vs = _manager_with_mocked_vectorstore(tmp, chunk_size=25)
            mock_vs.add_documents.side_effect = ConnectionError("network unreachable")

            with pytest.raises(ConnectionError):
                mgr.add_documents([Document(page_content="x", metadata={})])

        assert mock_vs.add_documents.call_count == 1  # no retry attempted
        assert mgr.embedding_model.chunk_size == 25  # never touched

    def test_the_common_case_never_retries(self):
        """Regression guard: when the batch embeds fine the first time
        (the overwhelming majority of real calls), nothing about this fix
        changes behavior — one call, no retry, no chunk_size mutation."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr, mock_vs = _manager_with_mocked_vectorstore(tmp, chunk_size=25)
            mock_vs.add_documents.return_value = None
            mock_vs._collection.count.return_value = 1

            mgr.add_documents([Document(page_content="x", metadata={})])

        assert mock_vs.add_documents.call_count == 1
        assert mgr.embedding_model.chunk_size == 25

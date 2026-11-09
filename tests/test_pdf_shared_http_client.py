"""Tests for utils/llm_call.py's get_shared_pdf_http_client (2026-11 fix).

Background: every raw OpenAI(...) construction across the PDF ingestion
pipeline (pdf_large_extraction.py, pdf_field_drafting.py, pdf_supersede_
resolver.py, pdf_candidate_matching.py, pdf_knowhow_drafting.py,
pdf_knowhow_candidate_matching.py, pdf_dual_extraction.py,
pdf_extraction_validation.py — 13 call sites total) used to construct a
brand-new client (and thus a brand-new httpx.Client/socket) on every single
call, never reused, never explicitly closed, relying on GC's non-
deterministic __del__ to ever release the connection — a real contributor to
a live incident where the SQS consumer accumulated CLOSE_WAIT sockets and
eventually went silent for 76+ minutes. get_shared_http_client() already
solved this exact problem for the chatbot's ~45 ChatOpenAI(...) sites;
get_shared_pdf_http_client() mirrors that fix for the PDF pipeline, as a
SEPARATE singleton so a connection-pool issue on one side can never affect
the other."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import httpx
import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import utils.llm_call as llm_call  # noqa: E402


@pytest.mark.unit
class TestGetSharedPdfHttpClient:
    def setup_method(self):
        # Reset the module-level singleton before each test so tests don't
        # leak state into each other via import-time caching.
        llm_call._shared_pdf_http_client = None

    def test_returns_an_httpx_client(self):
        client = llm_call.get_shared_pdf_http_client()
        assert isinstance(client, httpx.Client)

    def test_singleton_identity_across_calls(self):
        first = llm_call.get_shared_pdf_http_client()
        second = llm_call.get_shared_pdf_http_client()
        assert first is second

    def test_separate_singleton_from_chatbot_shared_client(self):
        llm_call._shared_http_client = None
        pdf_client = llm_call.get_shared_pdf_http_client()
        chatbot_client = llm_call.get_shared_http_client()
        assert pdf_client is not chatbot_client

    def test_thread_safe_construction_yields_one_instance(self):
        """Double-checked locking under real concurrent first-access —
        every thread must observe the same client, never two different
        ones racing to construct it."""
        llm_call._shared_pdf_http_client = None
        results = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(llm_call.get_shared_pdf_http_client())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 8
        assert all(r is results[0] for r in results)

    def test_configured_with_generous_connection_limits(self):
        client = llm_call.get_shared_pdf_http_client()
        pool = client._transport._pool
        assert pool._max_connections == 1000
        assert pool._max_keepalive_connections == 100

"""Tests for code/scripts/pdf_completeness_reconciliation.py (2026-11, PDF
pipeline overhaul, Phase 6 / gap #6) — formalizes this session's ad-hoc
S3-vs-corpus audit (the one that found the 28-document silent-gap
incident) into a permanent, testable tool.

All I/O (S3 client, vectorstore manager) is mocked — no real AWS
credentials or network calls, matching this codebase's established
dependency-injection test pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from scripts.pdf_completeness_reconciliation import (  # noqa: E402
    find_missing_documents,
    find_thin_documents,
    get_indexed_filenames,
    list_s3_raw_pdf_filenames,
    run_reconciliation,
)


def _mock_s3_client(keys: list[str]):
    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [{"Key": k} for k in keys]}]
    client.get_paginator.return_value = paginator
    return client


@pytest.mark.unit
class TestListS3RawPdfFilenames:
    def test_lists_raw_pdfs_under_prefix(self):
        client = _mock_s3_client([
            "restbiz/a.pdf", "restbiz/b.jpg", "restbiz/sub/c.pdf",
        ])
        result = list_s3_raw_pdf_filenames(client, "bucket", "restbiz/")
        assert sorted(result) == ["a.pdf", "b.jpg", "sub/c.pdf"]

    def test_excludes_pipeline_derived_subpaths(self):
        client = _mock_s3_client([
            "restbiz/a.pdf",
            "restbiz/extracted/a.json",
            "restbiz/pending_large/a.marker",
            "restbiz/status/a.json",
            "restbiz/ingestion_log/2026-11-09.json",
        ])
        result = list_s3_raw_pdf_filenames(client, "bucket", "restbiz/")
        assert result == ["a.pdf"]

    def test_excludes_directory_placeholder_keys(self):
        client = _mock_s3_client(["restbiz/", "restbiz/a.pdf"])
        result = list_s3_raw_pdf_filenames(client, "bucket", "restbiz/")
        assert result == ["a.pdf"]


@pytest.mark.unit
class TestGetIndexedFilenames:
    def test_elasticsearch_backend_returns_list_as_set(self):
        with patch("service.es_vector_store.list_pdf_ingested_files", return_value=["a.pdf", "b.pdf"]):
            result = get_indexed_filenames(use_elasticsearch=True)
        assert result == {"a.pdf", "b.pdf"}

    def test_local_chroma_backend_returns_dict_keys_as_set(self):
        with patch("service.local_vector_store.list_pdf_ingested_files", return_value={"a.pdf": 5, "b.pdf": 3}):
            result = get_indexed_filenames(use_elasticsearch=False)
        assert result == {"a.pdf", "b.pdf"}


@pytest.mark.unit
class TestFindMissingDocuments:
    def test_exact_match_is_not_missing(self):
        assert find_missing_documents(["a.pdf"], {"a.pdf"}) == []

    def test_substring_match_against_indexed_metadata_is_not_missing(self):
        # indexed source_file metadata can carry extra path/whitespace,
        # per this session's own audit findings — substring match tolerates it.
        assert find_missing_documents(["a.pdf"], {"restbiz/a.pdf"}) == []

    def test_whitespace_differences_are_normalized_before_matching(self):
        assert find_missing_documents(["a.pdf"], {"a.pdf  "}) == []

    def test_truly_absent_document_is_flagged_missing(self):
        assert find_missing_documents(["a.pdf", "b.pdf"], {"a.pdf"}) == ["b.pdf"]


@pytest.mark.unit
class TestFindThinDocuments:
    def test_document_under_threshold_is_flagged_thin(self):
        result = find_thin_documents([("a.pdf", 10)], {"a.pdf": 500}, threshold=200)
        assert result == [("a.pdf", 50.0)]

    def test_document_at_or_above_threshold_is_not_flagged(self):
        result = find_thin_documents([("a.pdf", 10)], {"a.pdf": 3000}, threshold=200)
        assert result == []

    def test_zero_page_count_is_skipped_not_flagged(self):
        result = find_thin_documents([("a.pdf", 0)], {"a.pdf": 0}, threshold=200)
        assert result == []

    def test_document_with_no_known_char_count_is_skipped_not_flagged(self):
        result = find_thin_documents([("a.pdf", 10)], {}, threshold=200)
        assert result == []


@pytest.mark.unit
class TestRunReconciliation:
    def test_flags_both_missing_and_thin_documents_in_one_report(self):
        client = _mock_s3_client(["restbiz/a.pdf", "restbiz/b.pdf", "restbiz/c.pdf"])
        with patch("scripts.pdf_completeness_reconciliation.get_indexed_filenames", return_value={"a.pdf", "b.pdf"}):
            report = run_reconciliation(
                client, "bucket", "restbiz/",
                filenames_and_page_counts=[("a.pdf", 10), ("b.pdf", 10)],
                char_counts_by_filename={"a.pdf": 5000, "b.pdf": 100},
            )
        assert report["total_s3_documents"] == 3
        assert report["missing_filenames"] == ["c.pdf"]
        assert report["missing_count"] == 1
        assert report["thin_filenames"] == [("b.pdf", 10.0)]
        assert report["thin_count"] == 1

    def test_all_accounted_for_and_healthy_yields_empty_report(self):
        client = _mock_s3_client(["restbiz/a.pdf"])
        with patch("scripts.pdf_completeness_reconciliation.get_indexed_filenames", return_value={"a.pdf"}):
            report = run_reconciliation(
                client, "bucket", "restbiz/",
                filenames_and_page_counts=[("a.pdf", 10)],
                char_counts_by_filename={"a.pdf": 5000},
            )
        assert report["missing_count"] == 0
        assert report["thin_count"] == 0

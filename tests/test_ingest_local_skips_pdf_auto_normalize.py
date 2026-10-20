"""Test for the 2026-10 fix (PDF2-M1) to scripts/ingest_local.py's
normalize_documents — used to run over Source E's carried-forward
pdf_auto docs too, collapsing newlines (breaking tables/numbered steps in
page_content) and re-running the 4-pass Thai dup-token regex over
book-length answer_guideline text it was never calibrated for (the same
regex class that already corrupted "จัดการการเงิน" once live). PDF-sourced
docs are already normalized at ingest time and are now skipped here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from scripts.ingest_local import normalize_documents  # noqa: E402


def _doc(page_content: str, metadata: dict):
    return SimpleNamespace(page_content=page_content, metadata=metadata)


@pytest.mark.unit
class TestNormalizeDocumentsSkipsPdfAuto:
    def test_pdf_auto_doc_page_content_is_left_untouched(self):
        multiline = "ขั้นตอนที่ 1\nขั้นตอนที่ 2\nขั้นตอนที่ 3"
        doc = _doc(multiline, {"ingested_via": "pdf_auto"})
        normalize_documents([doc])
        assert doc.page_content == multiline

    def test_pdf_auto_doc_metadata_is_left_untouched(self):
        doc = _doc("content", {"ingested_via": "pdf_auto", "answer_guideline": "line1\nline2\nline3"})
        original_guideline = doc.metadata["answer_guideline"]
        normalize_documents([doc])
        assert doc.metadata["answer_guideline"] == original_guideline

    def test_sheet_sourced_doc_is_still_normalized(self):
        """Control case: a Sheet-sourced doc (no ingested_via, or a
        different value) must still go through normalization as before —
        this fix is scoped to pdf_auto only."""
        multiline = "line1\nline2"
        doc = _doc(multiline, {"department": "X"})
        normalize_documents([doc])
        assert "\n" not in doc.page_content  # _normalize_text collapses newlines

    def test_mixed_batch_only_skips_pdf_auto_entries(self):
        pdf_doc = _doc("a\nb", {"ingested_via": "pdf_auto"})
        sheet_doc = _doc("c\nd", {"department": "Y"})
        normalize_documents([pdf_doc, sheet_doc])
        assert pdf_doc.page_content == "a\nb"
        assert "\n" not in sheet_doc.page_content

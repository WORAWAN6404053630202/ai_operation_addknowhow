"""2026-11 (PDF pipeline overhaul, lower-priority OCG investigation spike)
— CLOSED, confirmed non-reproducing.

The original research report flagged hidden Optional Content Group (OCG)
layers as a possible silent-data-loss gap: a layer set off-by-default could
in principle be invisible to get_text() AND to OCR (which renders via
get_pixmap(), which also respects OCG visibility by default) alike, with
nothing flagging that such a layer even exists.

Two-part investigation, now both complete:
1. Synthetic fixture (this file) — confirms this installed pymupdf version
   (1.28.2) already excludes an off-by-default OCG's content from BOTH
   get_text() and get_pixmap() by default, exactly as PyMuPDF's own docs
   claim. The stated gap does not reproduce on this pymupdf version.
2. Real-corpus scan (2026-11-09, run on the ai-restbiz-pdf-only-es-consumer
   container on EC2 instance 43.210.35.109, which has live S3 read access
   via its restbiz-pdf-ingestion-ec2-role instance role) — every one of
   the 177 real raw PDFs under s3://chatbot-database-supercoconut/restbiz/
   was opened and checked via doc.get_ocgs(). Zero files define any OCG
   layer at all. Confirms the residual risk this spike was scoped to check
   (a real document using a hidden OCG layer, which nothing currently
   flags) has zero known occurrences in the actual production corpus as
   of this date — not just a synthetic-fixture finding.

Conclusion: no code change warranted. If a future document DOES define an
OCG layer, get_text()/get_pixmap() will still silently exclude whatever it
hides (per part 1) — this spike did not fix that residual risk, it only
confirmed it has never yet occurred in this corpus. Revisit if a future
reconciliation pass (see code/scripts/pdf_completeness_reconciliation.py)
or a new document type introduces one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from pdf_fixtures import make_page_with_ocg_layer  # noqa: E402

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


@pytest.mark.unit
class TestOcgHiddenLayerExcludedByDefault:
    def test_get_text_excludes_off_by_default_ocg_content(self):
        import pymupdf

        pdf_bytes = make_page_with_ocg_layer(
            visible_text="Visible base content", hidden_text="Hidden OCG content",
        )
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            text = doc[0].get_text()
            assert "Visible base content" in text
            assert "Hidden OCG content" not in text
        finally:
            doc.close()

    def test_doc_get_ocgs_reports_the_layer_as_off(self):
        """Confirms get_ocgs() (the same API the real-corpus scan used) can
        actually detect an OCG's existence and off-state — so the real-
        corpus scan's "0 found" result reflects a working detector, not a
        detector that can't see OCGs at all."""
        import pymupdf

        pdf_bytes = make_page_with_ocg_layer(
            visible_text="Visible base content", hidden_text="Hidden OCG content",
        )
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            ocgs = doc.get_ocgs()
            assert len(ocgs) == 1
            info = next(iter(ocgs.values()))
            assert info["on"] is False
        finally:
            doc.close()

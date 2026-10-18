"""
Regression test for utils.text_patterns.effective_min_similarity — 2026-09
bug fix, part of the pdf_chunking_truncation_and_retrieval_fixes chain.

Background: RETRIEVAL_MIN_SIMILARITY=0.6 was calibrated for short, structured
regulatory/marketing Sheet rows. Live measurement against the real test PDF
("เจ๊ง! ในกระดาษ") found genuinely-correct PDF-book narrative content scores
systematically LOWER on bge-m3 cosine similarity than that floor — 5
independently-verified correct query/doc pairs scored 0.487-0.577, ALL below
0.6, live-reproduced as a Top Line/Bottom Line financial-analysis question
never surfacing its own (objectively best-matching, rank #1 on both dense AND
BM25 signals) source document at all, landing on an unrelated topic instead.

A 3-query negative-control check (regulatory-only queries with zero relation
to the PDF book) found the highest ANY pdf_auto doc scored was 0.3998 — the
0.45 floor chosen for RETRIEVAL_MIN_SIMILARITY_PDF_BOOK sits in the gap
between that ceiling and the lowest confirmed-correct score, not a guess.

conf.RETRIEVAL_MIN_SIMILARITY (0.6) is left completely untouched for every
other content type — this fix is scoped strictly to ingested_via="pdf_auto".
"""
import pytest

from utils.text_patterns import effective_min_similarity


@pytest.mark.unit
class TestEffectiveMinSimilarity:
    def test_pdf_book_doc_gets_lower_floor(self):
        md = {"ingested_via": "pdf_auto", "data_type": "business_guide"}
        assert effective_min_similarity(md, 0.6, 0.45) == 0.45

    def test_regulatory_doc_keeps_general_floor(self):
        md = {"license_type": "ใบทะเบียนพาณิชย์", "data_type": "regulatory"}
        assert effective_min_similarity(md, 0.6, 0.45) == 0.6

    def test_legacy_sheet_business_guide_keeps_general_floor(self):
        """Sheet-based business_guide content (no ingested_via at all) must NOT
        get the PDF-book floor — this fix is scoped to the PDF ingestion path
        only, never measured/confirmed for legacy Sheet content."""
        md = {"data_type": "business_guide", "main_topic": "กลยุทธ์การตลาด"}
        assert effective_min_similarity(md, 0.6, 0.45) == 0.6

    def test_ingested_via_case_insensitive_and_whitespace_tolerant(self):
        md = {"ingested_via": "  PDF_AUTO  "}
        assert effective_min_similarity(md, 0.6, 0.45) == 0.45

    def test_empty_metadata_keeps_general_floor(self):
        assert effective_min_similarity({}, 0.6, 0.45) == 0.6
        assert effective_min_similarity(None, 0.6, 0.45) == 0.6

    def test_real_measured_scores_pass_pdf_floor_but_not_general_floor(self):
        """Pins the actual live-measured scores from the 2026-09 investigation:
        all 5 genuinely-correct PDF-book matches clear 0.45 but none clear 0.6."""
        measured_correct_scores = [0.487, 0.548, 0.558, 0.565, 0.577]
        floor = effective_min_similarity({"ingested_via": "pdf_auto"}, 0.6, 0.45)
        for score in measured_correct_scores:
            assert score >= floor, f"{score} should clear the PDF-book floor"
            assert score < 0.6, f"{score} should NOT have cleared the general floor (sanity check on the measurement itself)"

    def test_real_measured_false_positive_ceiling_stays_below_pdf_floor(self):
        """Pins the negative-control ceiling: the highest score any PDF-book
        doc got for a completely unrelated regulatory query. Must stay below
        the chosen floor, or the floor is too permissive."""
        false_positive_ceiling = 0.3998
        floor = effective_min_similarity({"ingested_via": "pdf_auto"}, 0.6, 0.45)
        assert false_positive_ceiling < floor

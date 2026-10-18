"""
Regression test for utils.text_patterns.extract_thai_keywords — 2026-09 bug
fix, third and deepest layer of the pdf_chunking_truncation_and_retrieval_fixes
chain (after the dead sub_topic-retrieval fallback and the PDF-book similarity
floor, both fixed the same night).

Background: persona_academic.py's _detect_distinct_topics extracted query
"keywords" via plain whitespace splitting (`re.sub(r'[^Thai-range a-zA-Z0-9]',
' ', query).split()`), which does nothing useful for Thai — there are no
spaces between words in normal Thai writing. Live-reproduced against the
query "ในงบกำไรขาดทุน Top Line กับ Bottom Line ต่างกันยังไง":

- The real keyword "กำไร" (profit) was trapped inside one giant unsplittable
  blob ("ในงบกำไรขาดทุน") and could never match the correct topic's title
  ("การวิเคราะห์รายได้และกำไร"), even though the substring is right there.
- The bare conjunction "กับ" ("and/with", meaningless alone) survived as its
  own 3-character "word" purely because it happened to have a space on both
  sides in this particular query, and coincidentally appeared in an unrelated
  topic's content — handing that wrong topic the ONLY nonzero relevance score
  in the whole candidate set, so the topic-menu filter locked onto it.

Fix: real Thai segmentation via pythainlp (same word_tokenize engine="newmm"
already used by utils.hybrid_retriever's BM25 index — kept consistent on
purpose) plus pythainlp's maintained Thai stopword corpus. Both halves are
necessary: tokenization alone rescues "กำไร" but doesn't suppress "กับ".
"""
import pytest

from utils.text_patterns import extract_thai_keywords


@pytest.mark.unit
class TestExtractThaiKeywords:
    def test_real_word_trapped_in_blob_is_correctly_segmented(self):
        """The exact live-reproduced failure: 'กำไร' must come out as its own
        token, not be trapped inside 'ในงบกำไรขาดทุน'."""
        keywords = extract_thai_keywords("ในงบกำไรขาดทุน Top Line กับ Bottom Line ต่างกันยังไง")
        assert "กำไร" in keywords

    def test_common_conjunction_is_filtered_as_stopword(self):
        """The exact live-reproduced false-positive cause: 'กับ' must NOT
        survive as a "keyword" — it's a bare conjunction with no topical
        meaning of its own."""
        keywords = extract_thai_keywords("ในงบกำไรขาดทุน Top Line กับ Bottom Line ต่างกันยังไง")
        assert "กับ" not in keywords

    def test_correct_topic_title_now_matches_via_substring(self):
        """End-to-end shape of the actual bug: with real tokenization, the
        correct topic's title should now score a hit via plain substring
        check — this is the exact mechanism _detect_distinct_topics uses."""
        keywords = extract_thai_keywords("ในงบกำไรขาดทุน Top Line กับ Bottom Line ต่างกันยังไง")
        correct_title = "การวิเคราะห์รายได้และกำไร"
        wrong_title = "ประเภทของการวางแผนผังโรงงาน"
        assert any(kw in correct_title.lower() for kw in keywords)
        assert not any(kw in wrong_title.lower() for kw in keywords if kw not in ("top", "line", "bottom"))

    def test_english_tokens_preserved(self):
        keywords = extract_thai_keywords("Top Line กับ Bottom Line ต่างกันยังไง")
        assert "top" in keywords
        assert "line" in keywords
        assert "bottom" in keywords

    def test_empty_and_none_input(self):
        assert extract_thai_keywords("") == []
        assert extract_thai_keywords(None) == []

    def test_min_len_filters_short_tokens(self):
        keywords = extract_thai_keywords("ในงบกำไรขาดทุน Top Line กับ Bottom Line ต่างกันยังไง", min_len=5)
        assert all(len(kw) >= 5 for kw in keywords)

    def test_common_stopwords_filtered(self):
        """Sanity check against a few other everyday Thai function words —
        not just the one that happened to cause tonight's bug."""
        keywords = extract_thai_keywords("ร้านของฉันอยู่ที่ไหนก็ได้ แต่ต้องมีคนมา")
        for stopword in ("ของ", "ที่", "แต่", "ก็"):
            assert stopword not in keywords

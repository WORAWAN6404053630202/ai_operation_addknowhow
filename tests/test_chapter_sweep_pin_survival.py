"""
Regression test for the "runtime retrieval-pipeline silently drops content"
bug class (2026-09, pricing-strategy chapter case) — distinct from the
ingest-time field-coverage bug (see test_ingest_field_coverage.py).

Background: PracticalPersonaService._retrieve_docs's Round 4 chapter-
completeness sweep (persona_practical.py) deterministically fetches every
sub-topic of a detected chapter and tags each doc `_chapter_sweep_pin=True`
so it survives the final `docs[:max_docs]` truncation regardless of
similarity ranking. Two real, separately-discovered bugs meant that
guarantee didn't actually hold end-to-end:
  1. The cross-encoder reranker (utils.reranker.rerank) re-sorts AND
     truncates to its own RERANKER_TOP_K purely by relevance score, with no
     notion of "pinned" — a pinned doc could be dropped there even though it
     survived the min_similarity filter. Fixed by re-admitting any pinned
     doc the reranker output is missing.
  2. The metadata-boost step (also inside _retrieve_docs) re-sorts `docs` by
     blend score AGAIN afterward — silently undoing the front-pinned order
     from #1. Fixed by re-asserting the pinned/non-pinned grouping right
     before the final truncation, regardless of each group's internal order.

Live-verified case: 9 pinned sub-topics went in, only 7 came out — twice,
at two different pipeline stages. This test proves both stages can no
longer eat a pinned doc, using deliberately adversarial mocks (a fake
reranker that drops every pinned doc, and real metadata-boost math that
would otherwise rank them last) — no network access, no real embeddings,
no LLM calls, no real cross-encoder model load.
"""
import sys
import os

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import conf
from model.persona_practical import PracticalPersonaService

_TEST_MAIN_TOPIC = "ทดสอบหมวดตัวอย่างเชิงลึก"  # >= _NONREG_TOPIC_MIN_LEN (12 chars)
_SUB_TOPICS = [f"หัวข้อย่อยทดสอบที่ {i}" for i in range(1, 8)]  # 7 distinct sub-topics


def _make_chapter_docs():
    """The full chapter the sweep should find via coll.get(where={'main_topic': ...})."""
    return {
        "documents": [f"เนื้อหาเต็มของ {st}" for st in _SUB_TOPICS],
        "metadatas": [
            {
                "main_topic": _TEST_MAIN_TOPIC,
                "sub_topic": st,
                "data_type": "business_guide",
            }
            for st in _SUB_TOPICS
        ],
    }


def _make_noise_docs():
    """Round 1's initial (unrelated, low-similarity) results — deliberately
    share no common license_type/operation_topic/main_topic, so Round 2's
    majority-vote anchor finds nothing and Round 4's own detection is what
    has to do the work, matching the real live-reproduced failure mode."""
    pairs = []
    for i in range(3):
        d = Document(
            page_content=f"เนื้อหาที่ไม่เกี่ยวข้องเลข {i}",
            metadata={"main_topic": f"หมวดอื่นที่ไม่เกี่ยวข้อง {i}", "data_type": "marketing"},
        )
        pairs.append((d, 0.3))  # below RETRIEVAL_MIN_SIMILARITY (0.6) — filtered out later
    return pairs


@pytest.fixture
def practical(monkeypatch):
    fake_vectorstore = MagicMock()
    fake_vectorstore.similarity_search_with_relevance_scores = MagicMock(
        return_value=_make_noise_docs()
    )

    def _fake_collection_get(where=None, include=None):
        if where == {"data_type": {"$in": ["marketing", "business_guide"]}}:
            return {"metadatas": [{"main_topic": _TEST_MAIN_TOPIC}]}
        if where == {"main_topic": _TEST_MAIN_TOPIC}:
            return _make_chapter_docs()
        return {"documents": [], "metadatas": []}

    fake_vectorstore._collection.get = MagicMock(side_effect=_fake_collection_get)

    fake_retriever = MagicMock()
    fake_retriever.vectorstore = fake_vectorstore

    svc = PracticalPersonaService(retriever=fake_retriever)

    # No LLM calls at all — the query rewriter is a real network/LLM boundary.
    monkeypatch.setattr("model.persona_practical._qr_needs_rewrite", lambda q: False)

    monkeypatch.setattr(conf, "RERANKER_ENABLED", True)
    monkeypatch.setattr(conf, "RETRIEVAL_CATEGORY_FILTER_ENABLED", False)

    return svc


def _adversarial_rerank_drops_all_pins(query, docs, **kwargs):
    """Simulates the real, live-reproduced failure: the cross-encoder ranks
    every pinned doc below the top_k cutoff and drops them entirely (not
    just reorders) — proving the fix doesn't just get lucky with scoring."""
    survivors = [d for d in docs if not (getattr(d, "metadata", {}) or {}).get("_chapter_sweep_pin")]
    return survivors


@pytest.mark.unit
class TestChapterSweepPinSurvival:

    def test_all_pinned_subtopics_survive_an_adversarial_reranker(self, practical, monkeypatch):
        monkeypatch.setattr("utils.reranker.rerank", _adversarial_rerank_drops_all_pins)

        query = f"อยากรู้เรื่อง {_TEST_MAIN_TOPIC} แบบละเอียดทุกหัวข้อ"
        results = practical._retrieve_docs(query)

        found_sub_topics = {
            r["metadata"].get("sub_topic")
            for r in results
            if r["metadata"].get("main_topic") == _TEST_MAIN_TOPIC
        }
        missing = set(_SUB_TOPICS) - found_sub_topics
        assert not missing, (
            f"{len(missing)}/{len(_SUB_TOPICS)} pinned sub-topics were lost even after the "
            f"reranker-drop + final-truncation fixes: {missing}"
        )

    def test_pin_survival_holds_even_with_a_well_behaved_reranker(self, practical, monkeypatch):
        """Same guarantee must hold when the reranker behaves normally (just
        reorders, doesn't drop) — the fix must not depend on the adversarial
        case specifically."""
        def _passthrough_rerank(query, docs, **kwargs):
            return list(reversed(docs))
        monkeypatch.setattr("utils.reranker.rerank", _passthrough_rerank)

        query = f"อยากรู้เรื่อง {_TEST_MAIN_TOPIC} แบบละเอียดทุกหัวข้อ"
        results = practical._retrieve_docs(query)

        found_sub_topics = {
            r["metadata"].get("sub_topic")
            for r in results
            if r["metadata"].get("main_topic") == _TEST_MAIN_TOPIC
        }
        assert set(_SUB_TOPICS) == found_sub_topics

    def test_no_sweep_when_query_does_not_mention_any_known_chapter(self, practical, monkeypatch):
        """Negative control: an unrelated query must NOT trigger the sweep —
        proves the detection is query-driven, not unconditional."""
        monkeypatch.setattr("utils.reranker.rerank", lambda query, docs, **kwargs: docs)

        results = practical._retrieve_docs("คำถามที่ไม่เกี่ยวข้องกับหมวดทดสอบเลย")

        found = {
            r["metadata"].get("sub_topic")
            for r in results
            if r["metadata"].get("main_topic") == _TEST_MAIN_TOPIC
        }
        assert not found

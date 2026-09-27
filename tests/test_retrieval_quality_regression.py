"""
Retrieval-quality regression suite (2026-09).

Background: this project had no automated way to detect "did retrieval quality
regress" — every gap found this session (embedding boilerplate dominating Dense
similarity, topic_label_matches_query missing colloquial phrasing,
_is_chapter_overview breaking on stray cross-chapter docs) was found by a human
live-reproducing one customer complaint at a time. That doesn't scale as the
knowledge base grows (more Sheet rows, eventually PDF-ingested content sharing
the same Chroma/BM25/embedding pipeline). This suite is a standing, growable set
of (question, expected main_topic) pairs, checked against the REAL retrieval
pipeline on every run.

Marked @pytest.mark.integration (same convention as test_rag_faithfulness.py) —
makes real embedding API calls (a few cents total for the whole file), so it is
EXCLUDED from the default `pytest` run (see pytest.ini's `-m "not integration"`).
Run deliberately with:

    pytest tests/test_retrieval_quality_regression.py -m integration -v

No LLM generation calls are made — only `PracticalPersonaService._retrieve_docs`,
which runs the real Dense+BM25+RRF+rerank pipeline and returns ranked documents.
This is deliberately cheaper and more directly diagnostic than a full
PersonaSupervisor.handle() round-trip: it isolates retrieval quality from
classification/prompt-construction behavior, which is what this suite exists to
catch regressions in.

How to add a new case: append a dict to RETRIEVAL_CASES with `question` and
`expected_main_topics` (a set/list of any ONE of which is an acceptable top-K
hit — use more than one value when a question could reasonably surface either
of two related chapters). Pull real customer questions from QA review, not
synthetic ones — this suite's whole value is pinning REAL previously-broken
cases so they can't silently regress.
"""
import sys
import os

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))

import pytest

from service.local_vector_store import get_vs_manager
from model.persona_practical import PracticalPersonaService


# Each real case pinned here was a live customer complaint this session, fixed,
# and verified on production — see project memory (chapter_overview_dominant_topic_fix_2026-09.md,
# topic_label_nominalizer_fix_2026-09.md, embedding_boilerplate_fix_2026-09.md)
# for the full root-cause history of each.
RETRIEVAL_CASES = [
    {
        "question": "ร้านอาหารควรมีกลยุทธ์การสื่อสารการตลาดยังไงบ้างให้คนรู้จักร้านมากขึ้น",
        "expected_main_topics": {"กลยุทธ์ด้านการสื่อสาร"},
        "note": "chapter-overview dominant-topic fix — stray cross-chapter docs must not hide this chapter",
    },
    {
        "question": "ร้านเราเพิ่งเปิด อยากได้คำแนะนำเรื่องการตั้งราคาเมนูให้เหมาะสม มีหลักคิดยังไงบ้าง",
        "expected_main_topics": {"กลยุทธ์ด้านราคา", "การเข้าใจพื้นฐานการขายสินค้า"},
        "note": "exact-match chapter-fetch case",
    },
    {
        "question": "ร้านอาหารเล็กๆ ควรตั้งราคาเมนูยังไงให้แข่งขันได้ มีกลยุทธ์แนะนำไหมคะ",
        "expected_main_topics": {"กลยุทธ์ด้านราคา"},
        "note": (
            "the original embedding-boilerplate case — colloquial phrasing "
            "('ควรตั้งราคา', no การ nominalizer), must resolve via hybrid "
            "retrieval alone since exact-match may or may not fire"
        ),
    },
]


@pytest.fixture(scope="module")
def practical():
    mgr = get_vs_manager()
    mgr.connect_to_existing(fail_if_empty=True)
    return PracticalPersonaService(retriever=mgr)


@pytest.mark.integration
class TestRetrievalQualityRegression:

    @pytest.mark.parametrize(
        "case", RETRIEVAL_CASES, ids=[c["question"][:40] for c in RETRIEVAL_CASES]
    )
    def test_expected_main_topic_reachable_in_top_k(self, practical, case):
        docs = practical._retrieve_docs(case["question"])
        found_topics = {
            (d.get("metadata") or {}).get("main_topic", "").strip() for d in docs
        }
        expected = case["expected_main_topics"]
        assert found_topics & expected, (
            f"None of the expected main_topics {expected!r} appeared in the "
            f"top-{len(docs)} retrieved docs for question {case['question']!r} "
            f"({case.get('note', '')}). Actually found: {sorted(t for t in found_topics if t)}"
        )

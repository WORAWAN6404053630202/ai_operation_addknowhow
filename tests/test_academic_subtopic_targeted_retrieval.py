"""
Regression test for AcademicPersonaService._filter_docs_by_selected_topic's
sub_topic Chroma get() fallback — 2026-09 bug fix, part of the
pdf_chunking_truncation_and_retrieval_fixes chain.

Background: this function's targeted-re-retrieval step always tries a
`license_type` metadata filter first (line ~2021: `{"license_type":
selected_key}`), because for regulatory content the topic-menu key IS the
license_type. But for non-regulatory content (marketing/business_guide,
including PDF-book chunks), the topic-menu key is the doc's `sub_topic`
field instead — license_type is always empty for that content. A dedicated
sub_topic Chroma get() fallback existed for exactly this case, but it was
gated on `if not fresh_docs:`. That condition can never actually be true:
`_retrieve_docs`'s own contract (documented two blocks above it in the same
function) is to silently substitute an UNFILTERED semantic-search result
whenever its metadata filter matches 0 docs — so `fresh_docs` is always a
non-empty (but wrong-topic) list for a sub_topic-keyed selection. The correct
sub_topic get() fallback was therefore dead code: business_guide/PDF-book
topic selections silently kept whatever wrong-topic docs the unfiltered
semantic fallback ranked highest, live-reproduced as a Top Line/Bottom Line
financial-analysis question landing on an unrelated factory layout-planning
sub_topic and staying there on every retry.

Fix: gate the sub_topic get() fallback on the same "did any doc in fresh_docs
actually carry selected_key" signal (`_has_matched_filtered`, extended to
also check the `sub_topic` field) already used for the license_type/main_topic
retries in this same function, instead of on emptiness.

This test exercises the real `_filter_docs_by_selected_topic` method with a
mocked Chroma collection and a mocked `_retrieve_docs` standing in for the
"filter finds nothing → substitutes unfiltered wrong-topic docs" behavior —
no live retriever/LLM required, following the same pattern as
test_entity_type_confidence_tiering.py (`AcademicPersonaService.__new__`).
"""
import pytest

from model.persona_academic import AcademicPersonaService
from model.conversation_state import ConversationState

CORRECT_SUB_TOPIC = "การวิเคราะห์ Top Line และ Bottom Line ในงบกำไรขาดทุน"
WRONG_SUB_TOPIC = "ประเภทของการวางแผนผังโรงงาน"


def _doc(sub_topic: str, content: str) -> dict:
    return {
        "content": content,
        "metadata": {
            "sub_topic": sub_topic,
            "main_topic": "การวิเคราะห์ผลประกอบการ" if sub_topic == CORRECT_SUB_TOPIC else "การบริหารการผลิต",
            "data_type": "business_guide",
            "ingested_via": "pdf_auto",
        },
    }


class _FakeCollection:
    """Mimics Chroma collection.get(): exact `where` match only, mirroring
    real Chroma semantics (never substitutes unrelated content)."""

    def __init__(self, correct_docs):
        self._correct_docs = correct_docs

    def get(self, where=None, include=None):
        where = where or {}
        if where.get("sub_topic") == CORRECT_SUB_TOPIC:
            return {
                "documents": [d["content"] for d in self._correct_docs],
                "metadatas": [d["metadata"] for d in self._correct_docs],
            }
        # license_type (or any other) filter never matches business_guide content.
        return {"documents": [], "metadatas": []}


class _FakeVectorstore:
    def __init__(self, collection):
        self._collection = collection


@pytest.mark.unit
class TestSubTopicTargetedRetrievalFallback:
    def test_correct_subtopic_docs_win_over_wrong_topic_unfiltered_fallback(self):
        correct_docs = [
            _doc(CORRECT_SUB_TOPIC, "Top Line คือยอดขายรวม ส่วน Bottom Line คือกำไรสุทธิ " * 5),
            _doc(CORRECT_SUB_TOPIC, "สูตรคำนวณ Bottom Line = Top Line - ค่าใช้จ่ายทั้งหมด " * 5),
        ]
        wrong_topic_docs = [
            _doc(WRONG_SUB_TOPIC, "การวางแผนผังโรงงานแบ่งเป็นหลายประเภท " * 5),
        ]

        ac = AcademicPersonaService.__new__(AcademicPersonaService)
        ac.retriever = _FakeVectorstore.__new__(_FakeVectorstore)
        ac.retriever.vectorstore = _FakeVectorstore(_FakeCollection(correct_docs))

        # Simulate _retrieve_docs's real contract: a metadata_filter matching 0 real
        # docs silently substitutes unfiltered (here: wrong-topic) semantic results.
        def _fake_retrieve_docs(query, metadata_filter=None, state=None):
            return list(wrong_topic_docs)

        ac._retrieve_docs = _fake_retrieve_docs

        # Broad set only had 1 doc for the correct sub_topic (< 2) — forces reliance
        # on the targeted re-retrieval + get() fallback, matching the real scenario
        # where a niche PDF-book sub_topic barely surfaces in the broad top-K.
        state = ConversationState(
            session_id="t", persona_id="academic",
            context={"academic_question": "Top Line กับ Bottom Line ต่างกันยังไง"},
        )
        state.current_docs = [correct_docs[0], wrong_topic_docs[0]]

        ac._filter_docs_by_selected_topic(state, CORRECT_SUB_TOPIC)

        result_sub_topics = {
            (d.get("metadata") or {}).get("sub_topic") for d in state.current_docs
        }
        assert CORRECT_SUB_TOPIC in result_sub_topics
        assert WRONG_SUB_TOPIC not in result_sub_topics, (
            "wrong-topic doc must not survive once the correct sub_topic's real docs "
            "were found via the get() fallback"
        )

    def test_has_matched_filtered_recognizes_subtopic_match(self):
        """Direct pin on the extended _has_matched_filtered signal itself, via source
        inspection — guards against the check silently narrowing back to only
        license_type/main_topic in a future edit."""
        import inspect
        src = inspect.getsource(AcademicPersonaService._filter_docs_by_selected_topic)
        assert '.get("sub_topic") or "").strip() == selected_key' in src

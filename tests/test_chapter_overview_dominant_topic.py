"""
Regression tests for the _is_chapter_overview dominant-topic fix (2026-09,
persona_practical.py, ~line 4108 detection + ~line 4258 elif branch + ~line
6017 mandate construction).

Background: a real customer question ("ร้านอาหารควรมีกลยุทธ์การสื่อสารการตลาด
ยังไงบ้าง...") got answered with only 1 of 10 sub_topics under main_topic
"กลยุทธ์ด้านการสื่อสาร" (23 docs). Retrieval actually returned 23 correct docs
plus a handful of stray docs from an unrelated main_topic. The OLD detection
required ALL retrieved docs to share exactly one main_topic
(`len(main_topics) == 1`) before activating the "cover every sub_topic"
mandate — 2 distinct values present meant it never fired, so the LLM fell
back to answering only the most literally-matching sub_topic.

The fix: pick the DOMINANT main_topic by doc count (>= 50% of all retrieved
docs) instead of requiring 100% purity, mirroring an existing precedent
(_informational_but_nonreg_chapter, a few dozen lines below, which fixed the
identical shape of bug once already for a different branch).

Two rounds of self-caught over-reach during review (see project memory
chapter_overview_dominant_topic_fix_2026-09.md for the full narrative) shaped
the final design:
  1. Detection must NOT drop stray docs itself — _direct_topic_match's own
     branch (_informational_but_nonreg_chapter) deliberately KEEPS
     cross-chapter docs merged in by _merge_cross_chapter_docs_for_nonreg
     (they can be a deliberate whole-second-chapter merge, not noise).
  2. The "Required sections" mandate text must be scoped to the dominant
     topic's sub_topics regardless of which branch built _docs_to_process
     (Supervisor's own classification isn't perfectly deterministic call to
     call, so either branch can end up handling the exact same query).

These tests bypass retrieval entirely (retriever=None, _retrieve_docs/
_retrieve_multi_topic monkeypatched to hand back a fixed doc list regardless
of query) and monkeypatch _call_llm_json to capture the constructed prompt
without any network call — same zero-cost pattern as
test_auto_retrieve_loop_cap.py / test_practical_truncation_retry.py.
"""
from unittest.mock import MagicMock

import pytest

from model.persona_practical import PracticalPersonaService


def _doc(main_topic, sub_topic, data_type="marketing", content=None):
    return {
        "content": content or f"หัวข้อหลัก: {main_topic} หัวข้อย่อย: {sub_topic}",
        "metadata": {
            "main_topic": main_topic,
            "sub_topic": sub_topic,
            "data_type": data_type,
        },
    }


COMM_SUBTOPICS = [
    "การสื่อสารการตลาด",
    "ลักษณะของการสื่อสารการตลาด",
    "การสื่อสารแบบวางแผน (Planned Communication)",
    "การสื่อสารแบบไม่ได้วางแผน (Unplanned Communication)",
    "ปัญหาในการสื่อสาร (Communication Problems)",
    "การสื่อสารการตลาดบูรณาการ",
    "คุณสมบัติแนวคิดการสื่อสารการตลาดบูรณาการ (IMC)",
    "การตลาดผ่านสื่อสังคมออนไลน์",
    "ประโยชน์ของการตลาดออนไลน์ต่อผู้ซื้อ",
    "ประโยชน์ของการตลาดออนไลน์ต่อผู้ขาย",
]
PROMO_SUBTOPICS = [
    "ความหมายของการส่งเสริมการขาย",
    "ลักษณะของการส่งเสริมการขาย",
    "วัตถุประสงค์ของการส่งเสริมการขาย",
    "ผู้รับสารเป้าหมายของการส่งเสริมการขาย",
    "ประเภทและวิธีการส่งเสริมการขาย",
    "วิธีการส่งเสริมการขายของการส่งเสริมการขายแต่ละประเภท",
]
PRICING_SUBTOPICS = [
    "ความหมายกลยุทธ์ด้านราคา",
    "การกำหนดกลยุทธ์ทางด้านราคา",
    "การใช้กลยุทธ์ในการตั้งราคา",
    "กลยุทธ์การตั้งราคาโดยกำหนดส่วนลด",
    "กลยุทธ์การตั้งราคาเชิงจิตวิทยา",
]


def _comm_docs():
    return [_doc("กลยุทธ์ด้านการสื่อสาร", st) for st in COMM_SUBTOPICS]


def _promo_docs():
    return [_doc("กลยุทธ์ด้านการส่งเสริมการขาย", st) for st in PROMO_SUBTOPICS]


def _pricing_docs():
    return [_doc("กลยุทธ์ด้านราคา", st) for st in PRICING_SUBTOPICS]


@pytest.fixture
def practical():
    svc = PracticalPersonaService(retriever=None)
    svc._retrieve_docs = MagicMock(return_value=[])
    svc._retrieve_multi_topic = MagicMock(return_value=[])
    return svc


def _capture_prompt(practical):
    """Monkeypatch _call_llm_json to short-circuit with a canned answer and
    record the exact prompt text it was called with (zero network calls)."""
    captured = {}

    def _fake(prompt, *a, **kw):
        captured["prompt"] = prompt
        return {
            "input_type": "new_question",
            "analysis": "test",
            "action": "answer",
            "execution": {"answer": "ok", "context_update": {}},
        }

    practical._call_llm_json = _fake
    return captured


def _mandate_block(prompt: str) -> str:
    """Extract just the 'Required sections' mandate instruction, not the raw
    DOCUMENTS JSON dump above it — a stray doc's sub_topic legitimately
    appears in the DOCUMENTS listing itself (that's fine/expected), the thing
    that must NOT leak is the mandate telling the LLM which sections to
    write."""
    start = prompt.find("CHAPTER OVERVIEW")
    assert start != -1, "CHAPTER OVERVIEW mandate not found in prompt"
    end = prompt.find("Format: emoji header", start)
    return prompt[start:end]


@pytest.mark.unit
class TestChapterOverviewDominantTopic:

    def test_dominant_topic_with_minority_strays_triggers_full_mandate(self, practical, fresh_state):
        """The original bug: 10 real chapter docs + 3 stray docs from another
        main_topic must still activate the 'cover every sub_topic' mandate,
        scoped to the dominant (real) chapter only."""
        docs = _comm_docs() + _promo_docs()[:3]
        practical._retrieve_docs.return_value = docs
        fresh_state.current_docs = docs
        fresh_state.last_retrieval_query = "กลยุทธ์ด้านการสื่อสาร"
        captured = _capture_prompt(practical)

        practical.handle(fresh_state, "กลยุทธ์การสื่อสารการตลาดร้านอาหารมีอะไรบ้าง")

        prompt = captured["prompt"]
        mandate = _mandate_block(prompt)
        assert f"Required sections ({len(COMM_SUBTOPICS)} total)" in mandate
        for st in COMM_SUBTOPICS:
            assert st in mandate
        # Must NOT leak the stray topic's sub_topics into the mandate (they
        # can still legitimately appear in the raw DOCUMENTS JSON above it).
        for st in PROMO_SUBTOPICS:
            assert st not in mandate

    def test_pure_single_topic_unaffected_by_the_fix(self, practical, fresh_state):
        """Zero contamination case — must behave identically to the old
        strict len(main_topics)==1 check (still trivially true here)."""
        docs = _comm_docs()
        practical._retrieve_docs.return_value = docs
        fresh_state.current_docs = docs
        fresh_state.last_retrieval_query = "กลยุทธ์ด้านการสื่อสาร"
        captured = _capture_prompt(practical)

        practical.handle(fresh_state, "กลยุทธ์การสื่อสารการตลาดร้านอาหารมีอะไรบ้าง")

        prompt = captured["prompt"]
        assert "CHAPTER OVERVIEW" in prompt
        assert f"Required sections ({len(COMM_SUBTOPICS)} total)" in prompt
        assert f"DOCUMENTS ({len(COMM_SUBTOPICS)} found)" in prompt

    def test_no_dominant_topic_does_not_force_completeness(self, practical, fresh_state):
        """Genuinely ambiguous 3-way mix (5+5+5-ish, no topic >= 50%) must NOT
        trigger the mandate — falls back to the normal capped default rather
        than forcing completeness onto an arbitrary 'winner'."""
        mixed = _comm_docs()[:4] + _promo_docs()[:4] + _pricing_docs()[:4]
        practical._retrieve_docs.return_value = mixed
        fresh_state.current_docs = mixed
        fresh_state.last_retrieval_query = "กลยุทธ์การตลาดร้านอาหารมีอะไรบ้าง"
        captured = _capture_prompt(practical)

        practical.handle(fresh_state, "กลยุทธ์การตลาดร้านอาหารมีอะไรบ้าง")

        prompt = captured["prompt"]
        assert "CHAPTER OVERVIEW" not in prompt

    def test_direct_topic_match_keeps_all_docs_but_scopes_mandate_to_dominant(self, practical, fresh_state):
        """Cross-chapter-merge shape (_direct_topic_match branch,
        _informational_but_nonreg_chapter): a small 'wrongly matched' primary
        chapter plus a much larger genuinely-relevant merged chapter. That
        branch must still send EVERY doc uncapped (never drops the merge —
        _merge_cross_chapter_docs_for_nonreg's docs can be deliberately
        relevant, not noise), while the mandate text is scoped to whichever
        topic dominates by count."""
        small_primary = _promo_docs()[:2]  # stand-in for a thin wrongly-matched chapter
        large_merged = _pricing_docs()  # stand-in for the actually-relevant merged chapter
        all_docs = small_primary + large_merged
        practical._retrieve_docs.return_value = all_docs
        fresh_state.current_docs = all_docs
        fresh_state.last_retrieval_query = "กลยุทธ์ด้านการส่งเสริมการขาย"
        captured = _capture_prompt(practical)

        fresh_state.context["_direct_topic_match"] = True
        fresh_state.context["_informational_query"] = True
        practical.handle(fresh_state, "ตั้งราคาเมนูอาหารในร้านมีหลักคิดยังไงบ้างคะ")

        prompt = captured["prompt"]
        # _informational_but_nonreg_chapter must still send every doc, uncapped.
        assert f"DOCUMENTS ({len(all_docs)} found)" in prompt
        # But the mandate must be scoped to the dominant (pricing) chapter only.
        mandate = _mandate_block(prompt)
        assert f"Required sections ({len(PRICING_SUBTOPICS)} total)" in mandate
        for st in PRICING_SUBTOPICS:
            assert st in mandate
        for st in PROMO_SUBTOPICS[:2]:
            assert st not in mandate

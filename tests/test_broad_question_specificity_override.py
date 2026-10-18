"""
Regression tests for PersonaSupervisor._is_broad_question's specificity override
(2026-09 fix).

Background: a live QA rerun found the LLM broad_q classifier
(_broad_question_llm_check) firing 0.95-confidence "broad" on single-item
questions that name one specific license/device — e.g. "จดทะเบียนพาณิชย์แบบ
บุคคลธรรมดา ต้องเตรียมเอกสารอะไรบ้าง" and "ขั้นตอนการสมัครเครื่อง EDC...
เรียงลำดับ" — despite each naming exactly one thing. Misclassifying these as
"broad" triggers a 2-pass wide retrieval that pulls in ~20+ mostly-unrelated
docs (other business_guide chapters, unrelated license types), which measurably
caused unstable/incomplete/hallucinated answers (see qa_12case rerun, cases 1
and 9) and, in case 9, drove the model to ask a clarifying question instead of
ever answering.

The fix makes _SPECIFIC_LICENSE_INDICATOR_RE a hard short-circuit (return
False immediately) rather than "let the LLM decide" — a query naming one
specific license/device by name can never legitimately be a broad multi-topic
overview question. This is checked via the regex alone (no LLM call), so
these tests run free and fast.

PersonaSupervisor is constructed with retriever=MagicMock() — no network
(confirmed elsewhere in this suite; see test_multi_entity_comparison.py).
"""
from unittest.mock import MagicMock

import pytest

from model.persona_supervisor import PersonaSupervisor
from utils.prompts_supervisor import build_broad_question_detect_prompt


@pytest.fixture(scope="module")
def supervisor():
    return PersonaSupervisor(retriever=MagicMock())


@pytest.mark.unit
class TestSpecificLicenseOverrideShortCircuits:

    @pytest.mark.parametrize("query", [
        "จดทะเบียนพาณิชย์แบบบุคคลธรรมดา ต้องเตรียมเอกสารอะไรบ้าง",
        "ขั้นตอนการสมัครเครื่อง EDC มีอะไรบ้าง เรียงลำดับให้หน่อยคะ",
        "ค่าธรรมเนียมรายเดือนหรือค่าเช่าเครื่อง EDC ประมาณเท่าไหร่คะ",
        "อยากเปิดรับชำระเงินแบบ QR ในร้าน ต้องสมัครกับธนาคารไหน ใช้เอกสารอะไรบ้างคะ",
        "ขอใบภาษีมูลค่าเพิ่ม ภพ.20 ต้องใช้เอกสารอะไรบ้าง",
    ])
    def test_specific_item_queries_are_never_broad(self, supervisor, query):
        # state=None so this can't fall through to the LLM fallback even if the
        # override were somehow bypassed — isolates the regex-only short-circuit.
        assert supervisor._is_broad_question(query, state=None) is False

    def test_genuinely_broad_query_without_specific_item_still_detected_by_regex(self, supervisor):
        # Sanity check the fix didn't break the still-needed regex path for
        # actually-broad questions naming no specific license.
        assert supervisor._is_broad_question("จะเปิดร้านอาหารต้องทำอะไรบ้าง", state=None) is True

    def test_specific_indicator_regex_covers_edc_and_qr_terms(self, supervisor):
        # Regression pin: these terms were entirely absent before the fix,
        # so the override never had a chance to apply for EDC/QR questions.
        for term in ("เครื่องรูดบัตร EDC", "EDC", "QR Payment", "คิวอาร์", "ระบบชำระเงินออนไลน์"):
            assert supervisor._SPECIFIC_LICENSE_INDICATOR_RE.search(term), (
                f"{term!r} should match _SPECIFIC_LICENSE_INDICATOR_RE"
            )


@pytest.mark.unit
class TestBroadQuestionPromptDoesNotOverfireOnAroBang:
    """
    2026-09 follow-up bug fix: this class covers queries that name NO specific
    license/device at all (a non-regulatory business_guide/PDF-book topic), so
    _SPECIFIC_LICENSE_INDICATOR_RE above can never help — these fall straight
    through to the LLM classifier. Live-reproduced: "การหาพื้นที่ฐานลูกค้าทำได้
    กี่วิธี อะไรบ้าง" (a narrow question about ONE topic, methods for finding a
    customer catchment area) got misclassified is_broad=true (conf=0.95),
    triggering the same expensive 2-pass sweep — this time bad enough to blow
    the session token budget (147k tokens, 62.8k on a single call) and produce
    a broken 26-character answer.

    Root cause: every original is_broad=true prompt example ended in
    "...อะไรบ้าง" ("what are they"), so the classifier appears to have latched
    onto that ENDING as the broad signal, rather than the actual thing that
    makes those examples broad (spanning multiple regulatory/business
    dimensions). "อะไรบ้าง" alone carries no breadth signal in ordinary Thai —
    it's the everyday way to ask for a list of sub-items within ONE topic too.

    Since there's no closed/enumerable vocabulary here (unlike license names),
    the fix is at the prompt level, not a new regex override — this test pins
    the prompt's own content so the counter-examples can't silently regress,
    since the real classification behavior itself requires a live LLM call
    (verified separately, live, not part of this offline unit suite).
    """

    def test_prompt_states_aro_bang_alone_is_not_a_breadth_signal(self):
        prompt = build_broad_question_detect_prompt("test query")
        assert "อะไรบ้าง" in prompt
        assert "ไม่ใช่สัญญาณว่าเป็นคำถามภาพรวม" in prompt

    def test_prompt_includes_the_live_reproduced_counter_example(self):
        prompt = build_broad_question_detect_prompt("test query")
        assert "การหาพื้นที่ฐานลูกค้าทำได้กี่วิธี อะไรบ้าง" in prompt
        assert "ปัจจัยสำคัญในการเลือกทำเลมีอะไรบ้าง" in prompt

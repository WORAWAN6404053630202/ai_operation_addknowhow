"""
tests/test_rag_faithfulness.py — RAG answer-faithfulness eval (deepeval).

COSTS REAL MONEY — fires live LLM calls: the real Practical-mode answer call
AND a judge-LLM call to score it (deepeval's FaithfulnessMetric), once per
question. This is NOT a regular pytest test (not free, not deterministic, not
meant to run on every push) — it is marked @pytest.mark.integration and
excluded from both the default CI run and a plain local `pytest tests/` run
(see pytest.ini's `-m "not integration"` default). Run it deliberately:

    pytest tests/test_rag_faithfulness.py -m integration -v          # all 9
    pytest tests/test_rag_faithfulness.py -m "integration and not slow" -v  # smoke tier (3)

Requires a real OPENROUTER_API_KEY in env.properties.

What this checks: whether the real answer for each question is (a) faithful
to (grounded in) the documents actually retrieved, and (b) free of the
project's own known compliance violations (FORBIDDEN_PHRASES) — an automated
hallucination + compliance check. Questions and forbidden phrases are
imported directly from code/scripts/answer_type_pilot.py's ANSWER_TYPES
rather than invented here, so this eval has the exact same taxonomy coverage
(9 documented Practical-mode answer shapes) as the team's existing manual
regression pilot, instead of a smaller ad hoc set that drifts from it over
time. The faithfulness score is the automated, claim-by-claim counterpart to
that script's fixed-phrase scan — catches partial/subtle hallucination a
regex or a human skim can miss; the phrase scan is kept alongside it because
it catches a different, complementary class of issue (exact known-bad
wording, not ungrounded claims).

Judge model: Haiku via the project's own OpenRouter setup (conf.OPENROUTER_
SWITCH_MODEL), not deepeval's OpenAI-API-key default — this project already
requires OPENROUTER_API_KEY and nothing else, and Haiku is cheap enough that
scoring doesn't meaningfully add to the cost of the eval.
"""
import os
import re
import sys

import pytest

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))

deepeval = pytest.importorskip("deepeval")

from deepeval.metrics import FaithfulnessMetric  # noqa: E402
from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from scripts.answer_type_pilot import ANSWER_TYPES, SMOKE_TIER_IDS, find_forbidden_violations  # noqa: E402


class _OpenRouterJudge(DeepEvalBaseLLM):
    """Wraps this project's own OpenRouter/ChatOpenAI setup as deepeval's judge
    model so the eval only needs the OPENROUTER_API_KEY this project already
    requires, not a separate OPENAI_API_KEY (deepeval's default judge)."""

    def __init__(self):
        import conf
        self._model_name = getattr(conf, "OPENROUTER_SWITCH_MODEL", "anthropic/claude-haiku-4-5")
        super().__init__(model=self._model_name)

    def load_model(self):
        from langchain_openai import ChatOpenAI
        import conf
        return ChatOpenAI(
            model=self._model_name,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=1500,
            request_timeout=60,
        )

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        resp = await self.model.ainvoke(prompt)
        return resp.content

    def get_model_name(self) -> str:
        return self._model_name


@pytest.fixture(scope="module")
def _practical_service():
    """Real PracticalPersonaService wired to the real local Chroma retriever —
    the same construction the app itself uses (see router/route_v1.py)."""
    from model.persona_practical import PracticalPersonaService
    from service.local_vector_store import get_retriever
    retriever = get_retriever(fail_if_empty=False)
    return PracticalPersonaService(retriever=retriever)


@pytest.fixture(scope="module")
def _supervisor_service():
    """Real PersonaSupervisor wired to the real local Chroma retriever. Some
    regression cases below depend on supervisor's own retrieval orchestration
    (_ensure_practical_retrieval_for_legal — broad-question two-pass retrieval,
    exact-match chapter fetch, entity-injection) which _practical_service alone
    bypasses entirely — calling PracticalPersonaService.handle() directly skips
    the state.context flags/docs supervisor would have set up first. Using the
    wrong one silently gives an unrepresentative result (confirmed live during
    the 2026-09 QA sheet audit: identical queries passed via the direct-call
    shortcut and failed via this real path, and vice versa)."""
    from model.persona_supervisor import PersonaSupervisor
    from service.local_vector_store import get_retriever
    retriever = get_retriever(fail_if_empty=False)
    return PersonaSupervisor(retriever=retriever)


def _param_for(at) -> "pytest.param":
    """Smoke-tier questions (ids in SMOKE_TIER_IDS) get just @integration, so
    `-m "integration and not slow"` runs a cheap 3-question check; the rest
    are additionally @slow so the full 9-question sweep is an explicit opt-in
    (`-m integration`), mirroring answer_type_pilot.py's own smoke/full split.
    Marks must be attached here, at parametrize-list build time — adding them
    dynamically inside the test body via request.node.add_marker() is too
    late, since -m filtering happens at collection time, before the test body
    ever runs."""
    marks = [pytest.mark.integration]
    if at.id not in SMOKE_TIER_IDS:
        marks.append(pytest.mark.slow)
    return pytest.param(at, marks=marks, id=f"{at.id}_{at.name}")


@pytest.mark.parametrize("at", [_param_for(a) for a in ANSWER_TYPES])
def test_practical_answer_is_faithful_to_retrieved_docs(_practical_service, at):
    """The final answer must (a) not claim things the retrieved documents
    don't support — FaithfulnessMetric extracts individual claims and checks
    each against retrieval_context, catching partial hallucination a human
    skim or a fixed regex both easily miss — and (b) avoid the project's own
    documented forbidden phrases (utils/prompts_practical.py's "NEVER say"
    rules), the same check code/scripts/answer_type_pilot.py already does
    manually."""
    from model.conversation_state import ConversationState

    state = ConversationState(
        session_id=f"eval_{at.id}_{abs(hash(at.question))}", persona_id="practical", context={}
    )
    _, reply = _practical_service.handle(state, at.question, _internal=False)

    violations = find_forbidden_violations(reply)
    assert not violations, (
        f"[{at.name}] Forbidden phrase(s) found in answer: {violations}\n"
        f"Question: {at.question!r}\nAnswer: {reply[:500]}"
    )

    retrieved = [d.get("content", "") for d in (state.current_docs or []) if d.get("content")]
    assert retrieved, (
        f"[{at.name}] No documents retrieved for {at.question!r} — eval is "
        "meaningless without retrieval_context (check the question still matches the corpus)"
    )

    test_case = LLMTestCase(input=at.question, actual_output=reply, retrieval_context=retrieved)
    metric = FaithfulnessMetric(threshold=0.7, model=_OpenRouterJudge(), include_reason=True)
    metric.measure(test_case)

    assert metric.score >= metric.threshold, (
        f"[{at.name}] Faithfulness score {metric.score:.2f} below threshold "
        f"{metric.threshold}.\nQuestion: {at.question!r}\nReason: {metric.reason}\n"
        f"Answer: {reply[:500]}"
    )


@pytest.mark.integration
def test_compound_entity_variant_question_does_not_silently_pick_one_entity(_practical_service):
    """Regression pin for the "Case 6" bug (2026-09 QA sheet, real question below —
    reported directly by the user, not reconstructed): a compound question asks one
    thing that is entity-INVARIANT (which provider — NTT DATA, not a bank) and one
    thing that is entity-VARIANT (which documents — differs by บุคคลธรรมดา/นิติบุคคล).

    Root cause: plain semantic retrieval for this question also pulls in unrelated
    QR Payment / bank docs (word-overlap on "ธนาคาร") alongside the correct NTT DATA
    EDC docs — none of persona_supervisor.py's exact-match chapter-fetch guards fire
    for this query (the EDC entity-specific rows have no operation_topic, and the
    query doesn't contain the operation_by_department string verbatim), so this is
    purely a prompts_practical.py COMPOUND QUESTIONS-rule behavior, not a retrieval
    fix — confirmed live during the 2026-09 audit that produced this test.

    Before the fix, the model answered the entity-invariant part correctly but then
    silently picked ONE entity type (นิติบุคคล) for the document list — neither
    asking nor labeling both, contradicting the rule's actual intent. The fix
    replaced the rule's ambiguous "either ask or show both" phrasing (which the
    model was collapsing into a third, wrong behavior) with a single mandatory
    outcome: always show both entity variants, each labeled, never ask and never
    silently pick one.
    """
    from model.conversation_state import ConversationState

    question = "อยากติดตั้งเครื่องรูดบัตร EDC ในร้าน ต้องขอผ่านธนาคารไหน ใช้เอกสารอะไรบ้างคะ"
    state = ConversationState(
        session_id=f"eval_case6_{abs(hash(question))}", persona_id="practical", context={}
    )
    _, reply = _practical_service.handle(state, question, _internal=False)

    violations = find_forbidden_violations(reply)
    assert not violations, f"Forbidden phrase(s) found: {violations}\nAnswer: {reply[:800]}"

    assert "NTT DATA" in reply, (
        f"Entity-invariant part (provider = NTT DATA, not a bank) missing or wrong.\n"
        f"Answer: {reply[:800]}"
    )

    shows_both_labeled = "บุคคลธรรมดา" in reply and "นิติบุคคล" in reply
    asks_entity_type = bool(re.search(r"บุคคลธรรมดา.*หรือ.*นิติบุคคล|นิติบุคคล.*หรือ.*บุคคลธรรมดา", reply))
    assert shows_both_labeled or asks_entity_type, (
        "Compound-question entity-variant part was neither shown for both entity "
        "types nor asked about — likely silently picked one entity type again.\n"
        f"Answer: {reply[:800]}"
    )
    # The bug was specifically a SILENT single pick — enforce that if only one
    # entity type's label is present without the other, that's a failure even if
    # the word "บุคคลธรรมดา"/"นิติบุคคล" appears somewhere incidentally.
    assert not (("นิติบุคคล" in reply) and ("บุคคลธรรมดา" not in reply) and not asks_entity_type), (
        f"Answer mentions นิติบุคคล but not บุคคลธรรมดา and doesn't ask — silent single-entity pick.\n"
        f"Answer: {reply[:800]}"
    )


@pytest.mark.integration
def test_steps_question_does_not_ask_entity_type_when_steps_are_identical(_supervisor_service):
    """Regression pin for the "Case 9" bug (2026-09 QA sheet): a pure "steps, in
    order" question about EDC machine registration was live-reproduced
    non-deterministically asking "บุคคลธรรมดาหรือนิติบุคคล" before answering, even
    though operation_steps is byte-identical across both entity types for this
    procedure (only identification_documents differs) — confirmed directly
    against the corpus. The entity_type slot-queue injection in
    persona_practical.py fired purely because ≥2 entity-specific docs were in the
    candidate pool, with no check for whether the FIELD the user actually asked
    about (steps) varies by entity — contradicting the persona's own "Judge this
    by the SPECIFIC thing the user asked about" rule, since this is a
    deterministic slot-queue injection that runs before the LLM ever gets a
    chance to apply that rule. Fixed by gating the injection on operation_steps
    actually differing (or the query asking about documents) — see the
    "Field-awareness guard" comment at persona_practical.py's Level 1.6.

    Uses _supervisor_service (not _practical_service) because this bug's fix
    reads _op_pass_docs, which only exists via _ensure_practical_retrieval_for_legal
    + handle(_internal=True) — the direct _practical_service.handle() shortcut
    does not reproduce this code path at all.
    """
    from model.conversation_state import ConversationState

    question = "ขั้นตอนการสมัครเครื่อง EDC มีอะไรบ้าง เรียงลำดับให้หน่อยค่ะ"
    state = ConversationState(
        session_id=f"eval_case9_{abs(hash(question))}", persona_id="practical", context={}
    )
    state.messages = [{"role": "user", "content": question}]
    _supervisor_service._ensure_practical_retrieval_for_legal(state, question)
    _, reply = _supervisor_service._practical.handle(state, question, _internal=True)

    asks_entity_type = bool(re.search(r"บุคคลธรรมดา.*หรือ.*นิติบุคคล|นิติบุคคล.*หรือ.*บุคคลธรรมดา", reply))
    assert not asks_entity_type, (
        f"Steps question wrongly asked entity_type even though operation_steps "
        f"is identical across entity types for this procedure.\nAnswer: {reply[:500]}"
    )
    assert "กรอกข้อมูล" in reply or "ฟอร์ม" in reply, (
        f"Expected the actual EDC application steps in the answer.\nAnswer: {reply[:500]}"
    )


@pytest.mark.integration
def test_which_provider_question_shows_all_providers_per_entity(_supervisor_service):
    """Regression pin for "Case 4" (2026-09 QA sheet): "which bank, which
    documents" for QR Payment registration. The corpus genuinely offers this
    service through 2 banks (SCB, KBank) whose document requirements differ
    even for the SAME entity_type (บุคคลธรรมดา) — confirmed directly against the
    corpus (SCB's บุคคลธรรมดา list additionally requires "หนังสือจดทะเบียนพาณิชย์
    อิเล็กทรอนิกส์", which KBank's list does not). Before the fix, the answer
    picked only ONE bank's document list and presented it as if it applied
    universally to "บุคคลธรรมดา" regardless of bank — a second, un-named
    dimension of the same "compound question / entity-variant" bug class
    prompts_practical.py's COMPOUND QUESTIONS rule addresses, needing its own
    EXCEPTION C since here the "which provider" answer is itself multi-valued
    (no single established department to switch from, unlike EXCEPTION A/B).
    """
    from model.conversation_state import ConversationState

    question = "อยากเปิดรับชำระเงินแบบ QR ในร้าน ต้องสมัครกับธนาคารไหน ใช้เอกสารอะไรบ้างคะ"
    state = ConversationState(
        session_id=f"eval_case4_{abs(hash(question))}", persona_id="practical", context={}
    )
    state.messages = [{"role": "user", "content": question}]
    _supervisor_service._ensure_practical_retrieval_for_legal(state, question)
    _, reply = _supervisor_service._practical.handle(state, question, _internal=True)

    violations = find_forbidden_violations(reply)
    assert not violations, f"Forbidden phrase(s) found: {violations}\nAnswer: {reply[:800]}"

    assert "กสิกรไทย" in reply and "ไทยพาณิชย์" in reply, (
        f"Answer must name both providers this service is genuinely offered through.\n"
        f"Answer: {reply[:800]}"
    )
    assert "บุคคลธรรมดา" in reply, f"Answer dropped entity-variant framing entirely.\nAnswer: {reply[:800]}"


@pytest.mark.integration
@pytest.mark.slow
def test_multi_entity_comparison_on_first_turn_does_not_info_gap(_supervisor_service):
    """Regression pin for "Case 11" (2026-09 QA sheet): a genuine multi-entity
    comparison question ("ห้างหุ้นส่วนจำกัด กับบุคคลธรรมดา ... ต่างกันไหม") asked as
    the FIRST message in a session. persona_supervisor.py correctly detects and
    sets `_multi_entity_mentioned`, but persona_practical.py's compensating
    "established-topic guarantee (entity)" required `state.context["last_topic"]`
    to already be set — which only happens on a FOLLOW-UP turn, never on a
    fresh first message. Root cause traced deeper than that alone: the query's
    initial retrieval happened to surface only renewal/edit/cancel variants of
    the target license (all correctly excluded by the op-intent filter on their
    own terms), leaving that license with ZERO surviving docs by the time any
    entity guarantee could run — collapsing to a 40-doc pool of a completely
    unrelated license_type (ใบทะเบียนพาณิชย์, DBD commerce registration) via
    sibling-completion, and a blanket [[INFO_GAP]] despite the correct docs
    having been retrieved at all (just filtered out under this specific
    combination of query framing + entity-comparison phrasing). Fixed via three
    changes in persona_practical.py: (1) deriving a fallback established_license
    from a PRE-op-intent-filter snapshot when none is pre-established, (2) a
    trigram-overlap scorer for that fallback since Thai has no word spaces, (3) a
    blanket per-license guarantee fetch (not just per-entity_type) for licenses
    whose entity-comparison content lives as running text within a single row
    rather than split across entity_type_normalized-tagged rows. Marked slow —
    this is the most expensive/fragile case in this file, held to a lighter bar
    (no forbidden-phrase check) since the compound fix here is retrieval-shape,
    not wording.
    """
    from model.conversation_state import ConversationState

    question = "ร้านเราเป็นห้างหุ้นส่วนจำกัด กับร้านที่เป็นบุคคลธรรมดา เอกสารที่ต้องใช้ขอใบอนุญาตอาหารต่างกันไหม"
    state = ConversationState(
        session_id=f"eval_case11_{abs(hash(question))}", persona_id="practical", context={}
    )
    state.messages = [{"role": "user", "content": question}]
    _supervisor_service._ensure_practical_retrieval_for_legal(state, question)
    _, reply = _supervisor_service._practical.handle(state, question, _internal=True)

    assert "ยังไม่พบข้อมูลในเอกสาร" not in reply, (
        f"First-turn multi-entity comparison fell back to a blanket info-gap "
        f"despite the corpus having the relevant documents.\nAnswer: {reply[:500]}"
    )
    assert "บุคคลธรรมดา" in reply, f"Answer dropped the บุคคลธรรมดา side of the comparison.\nAnswer: {reply[:500]}"


@pytest.mark.integration
@pytest.mark.slow
def test_broad_nonreg_overview_is_not_truncated_by_informational_cap(_supervisor_service):
    """Regression pin for "Case 3" (2026-09 QA sheet): a broad "what strategies should
    I use" marketing-communication question. Two rounds of prompt-strengthening
    ("cover all distinct sub_topics", then a forceful checklist-style framing) did
    NOT fix this — the answer kept covering only 2-3 of the 6+ distinct sub_topics
    genuinely present in the retrieved chapter, no matter how directive the wording
    got. Root cause turned out to be retrieval-shape, not LLM obedience: this query
    matches via Step 0's operation_topic exact-match on a single definition row
    ("การสื่อสารการตลาด"), which persona_supervisor.py's own "non-reg completeness"
    logic correctly expands to the full 23+ doc "กลยุทธ์ด้านการสื่อสาร" chapter — but
    persona_supervisor.py's LLM-based informational/action classifier ALSO tagged
    this query as `_informational_query=True` (a defensible call in isolation — "what
    strategies should I use" reads as an overview ask), and persona_practical.py's
    pre-existing "informational query → cap to _prompt_max_docs (~5)" rule fired
    unconditionally whenever `_direct_topic_match` was set, silently discarding 20+
    of the very sub_topics the chapter expansion existed to surface. No prompt
    wording can recover content the LLM was never shown. Fixed by skipping that cap
    specifically when the doc set is a non-regulatory (marketing/business_guide)
    chapter overview — detected by data_type alone (not requiring a single shared
    main_topic, since the separate "cross-chapter merge" step can legitimately mix
    in a couple of docs from another main_topic on top of the primary chapter).
    Marked slow: this is a large, expensive multi-topic answer.
    """
    from model.conversation_state import ConversationState

    question = "ร้านอาหารควรมีกลยุทธ์การสื่อสารการตลาดยังไงบ้างให้คนรู้จักร้านมากขึ้น"
    state = ConversationState(
        session_id=f"eval_case3_{abs(hash(question))}", persona_id="practical", context={}
    )
    state.messages = [{"role": "user", "content": question}]
    _supervisor_service._ensure_practical_retrieval_for_legal(state, question)
    _, reply = _supervisor_service._practical.handle(state, question, _internal=True)

    # Distinct sub_topics confirmed present in the retrieved chapter (see corpus check in
    # the fix's own investigation) — require most of them, not literally all 6+, since
    # exact section framing/wording can still vary run to run even with full doc access.
    _expected_subtopics = [
        "Planned Communication", "Unplanned Communication", "IMC", "Social Media",
    ]
    _found = [s for s in _expected_subtopics if s in reply]
    assert len(_found) >= 3, (
        f"Broad chapter-overview answer covered only {len(_found)}/{len(_expected_subtopics)} "
        f"expected distinct sub_topics ({_found}) — likely the informational-cap regression "
        f"discarding the chapter expansion again.\nAnswer: {reply[:1500]}"
    )

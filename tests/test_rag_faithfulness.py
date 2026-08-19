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
import sys

import pytest

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))

deepeval = pytest.importorskip("deepeval")

from deepeval.metrics import FaithfulnessMetric  # noqa: E402
from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from scripts.answer_type_pilot import ANSWER_TYPES, FORBIDDEN_PHRASES, SMOKE_TIER_IDS  # noqa: E402


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

    violations = [p for p in FORBIDDEN_PHRASES if p in reply]
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

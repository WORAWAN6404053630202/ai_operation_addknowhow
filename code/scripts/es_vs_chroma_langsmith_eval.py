# code/scripts/es_vs_chroma_langsmith_eval.py
"""Round 3 Phase 7 (.claude/plans/mutable-seeking-wave.md): runs the same
LangSmith dataset (see es_qa_sheet_to_langsmith_dataset.py) against both the
Chroma-backed baseline bot and the Elasticsearch-backed candidate bot via
their real live /chat HTTP endpoints, scores both with 3 evaluators
(Correctness, Groundedness, Relevance), and uploads two separate LangSmith
experiments for side-by-side comparison in the UI.

COSTS REAL MONEY — each run makes N x (1 chat call + up to 3 judge calls)
real LLM calls (N = dataset size, 175 real rows as of 2026-11 in
pdf-only-qa-175 — the "113" in this file's own name/defaults is stale,
left over from an earlier, smaller sheet). Confirm with the user before
running for real; nothing here is free like the earlier reindex/hardening
scripts. Use --limit for a cheap smoke-test before committing to a full run.

2026-11, live-caught: the default max_concurrency=4 reliably trips the
target bot's own per-IP rate limiter (conf.RATE_LIMIT_MAX_REQUESTS=10 per
conf.RATE_LIMIT_WINDOW_SECONDS=60) within the first few iterations — a
real 175-row run lost 165/175 rows to HTTP 429 with real (if partial, see
route_v1.py's rate limiter — it rejects BEFORE calling the LLM, so no
chat/judge cost was incurred on the rejected rows) money spent on nothing.
Default lowered to 1: at this bot's typical ~20-35s per /chat call, even
fully sequential requests land at ~2 req/60s, comfortably under the limit
without needing to touch the live bot's rate-limiter config at all. Raise
back up only if you've confirmed the target's rate limit is disabled/high
enough for the concurrency you want.

Correctness and Relevance are fresh lightweight claude-haiku-4-5 judges
(same model/client-construction pattern this codebase already uses for its
own cheap classifier calls). Groundedness deliberately REUSES this
project's own already-calibrated judge — model.persona_practical.
PracticalPersonaService._groundedness_llm_call (see that method's own
docstring for its calibration history / few-shot examples) — rather than a
generic prompt, instantiated with retriever=None since the judge closures
built in __init__ never touch the retriever at all.

Usage:

    python3 code/scripts/es_vs_chroma_langsmith_eval.py \
        --dataset-name pdf-only-qa-175 \
        --es-url http://43.210.35.109:3001 --skip-chroma \
        --limit 5   # smoke-test a handful of rows before a full paid run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent.parent
code_dir = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import conf

_CHAT_TIMEOUT_S = 60


def _make_target(base_url: str):
    """Each call uses a FRESH session_id — comparing cold-session answers
    on equal footing across both backends, not accumulating multi-turn
    state across the 113 questions (which would make later questions'
    answers depend on earlier ones, confounding the comparison)."""
    import requests

    def target(inputs: dict) -> dict:
        session_id = f"s_eval_{uuid.uuid4().hex[:12]}"
        resp = requests.post(
            f"{base_url}/api/v1/chat",
            json={"message": inputs["question"], "session_id": session_id, "include_sources": True},
            timeout=_CHAT_TIMEOUT_S,
        )
        resp.raise_for_status()
        body = resp.json()
        return {
            "answer": body.get("response", ""),
            "retrieved_sources": body.get("retrieved_sources"),
            "total_cost": body.get("total_cost"),
        }

    return target


def _judge_llm():
    from langchain_openai import ChatOpenAI
    from utils.llm_call import get_shared_http_client, CHAT_OPENAI_MAX_RETRIES

    # 2026-11: ESEVAL_JUDGE_MODEL lets correctness/relevance be A/B tested
    # against a different judge model without touching conf.py's real
    # OPENROUTER_SWITCH_MODEL (which other live call sites also depend on).
    # Deliberately never defaults to a Google/Gemini-family model here now
    # that OPENROUTER_MODEL_PRACTICAL is gemini-2.5-flash-lite — judge
    # self-preference bias is well-documented (a judge scores its own
    # model family's outputs 10-25% more favorably, even on objective
    # rubrics) and this judge's whole job is catching THIS bot's mistakes.
    model = os.environ.get("ESEVAL_JUDGE_MODEL") or getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
    return ChatOpenAI(
        model=model,
        openai_api_key=conf.OPENROUTER_API_KEY,
        openai_api_base=conf.OPENROUTER_BASE_URL,
        http_client=get_shared_http_client(),
        temperature=0.0,
        # 2026-11, pilot-run-caught bug: 400 was too small once prompt_tokens
        # grow past ~1,600 (long question + reference + candidate answer) —
        # the reasoning field alone can eat the whole budget before the
        # JSON closes, and response_format=json_object makes OpenAI's
        # client raise LengthFinishReasonError on a truncated body instead
        # of returning partial text. Confirmed live: a real judge call hit
        # exactly this (1,657 prompt tokens, 400/400 completion tokens
        # used, parse failed) on the very first pilot question.
        max_tokens=1000,
        request_timeout=20,
        max_retries=CHAT_OPENAI_MAX_RETRIES,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _parse_judge_json(text: str) -> dict:
    """2026-11, live-caught bug: response_format={"type": "json_object"} on
    _judge_llm() does NOT reliably stop claude-haiku-4-5 (via OpenRouter)
    from wrapping its JSON in a ```json ... ``` markdown fence — confirmed
    live, every single judge call in a 175-question run came back fenced.
    A bare json.loads() on that text fails immediately (char 0), and the
    broad except silently returned {} — meaning correctness_evaluator and
    relevance_evaluator scored EVERY example 0.0 with an empty reasoning
    comment, with no error surfaced anywhere, since they were never
    designed to expect a parse failure on a response_format=json_object
    call. Strip the fence first, same pattern already used elsewhere in
    this codebase (e.g. persona_academic.py's _call methods) for models
    that aren't forced into true JSON mode."""
    t = text.strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(t)
    except Exception:
        return {}


def correctness_evaluator(run, example) -> dict:
    """LLM-judge: does the candidate answer match the reference answer's
    actual facts? Reference-based — needs example.outputs['answer']
    (confirmed present, the sheet has a reference-answer column)."""
    answer = (run.outputs or {}).get("answer", "")
    reference = (example.outputs or {}).get("answer", "")
    question = (example.inputs or {}).get("question", "")
    if not reference.strip():
        return {"key": "correctness", "score": None, "comment": "no reference answer in dataset row — skipped"}

    llm = _judge_llm()
    prompt = (
        "เปรียบเทียบคำตอบของบอทกับคำตอบอ้างอิง ว่าตรงกันในข้อเท็จจริงหลักหรือไม่ "
        "(ไม่ต้องตรงคำต่อคำ แค่ข้อมูลสำคัญ เช่น ค่าธรรมเนียม ขั้นตอน เอกสาร ต้องตรงกัน)\n\n"
        f"คำถาม: {question}\n\nคำตอบอ้างอิง: {reference}\n\nคำตอบของบอท: {answer}\n\n"
        'ตอบเป็น JSON เท่านั้น: {"reasoning": "...", "correct": true/false}'
    )
    # 2026-11, pilot-run-caught: a single judge-call failure (truncation,
    # rate limit, transient network error) must not risk aborting a paid
    # 113-175-question x 2-backend evaluate() run — report it as a skipped
    # score for this one example instead of raising. Routed through
    # llm_invoke (not a bare llm.invoke()) for the same retry/logging/
    # metrics parity every other LLM call in this codebase gets — this was
    # previously the one real bypass found during the retry-duplication
    # audit (.claude/plans/mutable-seeking-wave.md).
    from langchain_core.messages import HumanMessage
    from utils.llm_call import llm_invoke, extract_llm_text
    try:
        text = extract_llm_text(
            llm_invoke(llm, [HumanMessage(content=prompt)], label="ESEval/correctness")
        ).strip()
    except Exception as exc:
        return {"key": "correctness", "score": None, "comment": f"judge call failed: {exc}"}
    result = _parse_judge_json(text)
    return {
        "key": "correctness",
        "score": 1.0 if result.get("correct") else 0.0,
        "comment": result.get("reasoning", ""),
    }


def relevance_evaluator(run, example) -> dict:
    """Reference-free: does the answer actually address the question asked
    (not a deflection / off-topic / 'ไม่พบข้อมูล' when the info existed)."""
    answer = (run.outputs or {}).get("answer", "")
    question = (example.inputs or {}).get("question", "")

    llm = _judge_llm()
    prompt = (
        "คำตอบของบอทตอบตรงคำถามที่ถามหรือไม่ (ไม่ใช่ตรวจความถูกต้อง แค่ตรวจว่าตอบตรงประเด็นที่ถามหรือหลบคำถาม)\n\n"
        f"คำถาม: {question}\n\nคำตอบ: {answer}\n\n"
        'ตอบเป็น JSON เท่านั้น: {"reasoning": "...", "relevant": true/false}'
    )
    from langchain_core.messages import HumanMessage
    from utils.llm_call import llm_invoke, extract_llm_text
    try:
        text = extract_llm_text(
            llm_invoke(llm, [HumanMessage(content=prompt)], label="ESEval/relevance")
        ).strip()
    except Exception as exc:
        return {"key": "relevance", "score": None, "comment": f"judge call failed: {exc}"}
    result = _parse_judge_json(text)
    return {
        "key": "relevance",
        "score": 1.0 if result.get("relevant") else 0.0,
        "comment": result.get("reasoning", ""),
    }


def _make_groundedness_evaluator():
    """Instantiates the real PracticalPersonaService judge once (not per
    call — same ChatOpenAI client/closure reused across all 113 questions,
    matching how the live app itself reuses it across a session)."""
    from model.persona_practical import PracticalPersonaService

    service = PracticalPersonaService(retriever=None)

    def groundedness_evaluator(run, example) -> dict:
        answer = (run.outputs or {}).get("answer", "")
        docs = (run.outputs or {}).get("retrieved_sources")
        if not docs:
            return {"key": "groundedness", "score": None, "comment": "no retrieved_sources on this run (cache hit, or include_sources wasn't honored) — skipped"}
        try:
            result = service._groundedness_llm_call(answer, docs, state=None)
        except Exception as exc:
            return {"key": "groundedness", "score": None, "comment": f"judge call failed: {exc}"}
        return {
            "key": "groundedness",
            "score": 0.0 if result.get("unsupported") else 1.0,
            "comment": f"{result.get('reasoning', '')} claims={result.get('claims', [])}",
        }

    return groundedness_evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", default="pdf-only-qa-175")
    parser.add_argument("--chroma-url", default="http://43.210.64.119:3001")
    parser.add_argument("--es-url", default="http://43.210.35.109:3001")
    # 2026-11, live-caught: 4 reliably trips the target bot's own per-IP
    # rate limiter (see module docstring) — 1 is safe at this bot's typical
    # per-call latency without needing to touch the live rate-limiter config.
    parser.add_argument("--max-concurrency", type=int, default=1, help="parallel requests — keep modest, these hit live bot instances serving real traffic too; raise only if you've confirmed the target's rate limit allows it")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N dataset rows — use this for a cheap smoke-test before a full paid run")
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument("--skip-es", action="store_true")
    args = parser.parse_args()

    from langsmith import evaluate, Client

    evaluators = [correctness_evaluator, relevance_evaluator, _make_groundedness_evaluator()]

    # --limit slices the dataset client-side into an explicit list of
    # examples — evaluate()'s `data` param accepts either a dataset name
    # (string) or an iterable of Examples, so this is a drop-in substitute
    # for the full dataset name string below, not a separate code path.
    data = args.dataset_name
    if args.limit:
        client = Client()
        ds = client.read_dataset(dataset_name=args.dataset_name)
        data = list(client.list_examples(dataset_id=ds.id, limit=args.limit))
        print(f"--limit {args.limit}: running only {len(data)} row(s) of {args.dataset_name}")

    if not args.skip_chroma:
        print(f"\n=== Running Chroma baseline experiment against {args.chroma_url} ===")
        chroma_results = evaluate(
            _make_target(args.chroma_url),
            data=data,
            evaluators=evaluators,
            experiment_prefix="chroma-baseline",
            max_concurrency=args.max_concurrency,
        )
        print(f"Chroma experiment done: {chroma_results}")

    if not args.skip_es:
        print(f"\n=== Running ES candidate experiment against {args.es_url} ===")
        es_results = evaluate(
            _make_target(args.es_url),
            data=data,
            evaluators=evaluators,
            experiment_prefix="es-hybrid",
            max_concurrency=args.max_concurrency,
        )
        print(f"ES experiment done: {es_results}")

    print("\nDone. Compare the two experiments side-by-side in the LangSmith UI (same dataset, two experiment_prefix values).")


if __name__ == "__main__":
    main()

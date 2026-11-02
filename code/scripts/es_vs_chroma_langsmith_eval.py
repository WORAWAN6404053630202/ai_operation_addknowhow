# code/scripts/es_vs_chroma_langsmith_eval.py
"""Round 3 Phase 7 (.claude/plans/mutable-seeking-wave.md): runs the same
LangSmith dataset (see es_qa_sheet_to_langsmith_dataset.py) against both the
Chroma-backed baseline bot and the Elasticsearch-backed candidate bot via
their real live /chat HTTP endpoints, scores both with 3 evaluators
(Correctness, Groundedness, Relevance), and uploads two separate LangSmith
experiments for side-by-side comparison in the UI.

COSTS REAL MONEY — each run makes 113 x (1 chat call + 3 judge calls) real
LLM calls. Confirm with the user before running for real; nothing here is
free like the earlier reindex/hardening scripts.

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
        --dataset-name pdf-only-qa-113 \
        --chroma-url http://43.210.64.119:3001 \
        --es-url http://43.210.35.109:3001
"""
from __future__ import annotations

import argparse
import json
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
    from utils.llm_call import get_shared_http_client

    return ChatOpenAI(
        model=getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL),
        openai_api_key=conf.OPENROUTER_API_KEY,
        openai_api_base=conf.OPENROUTER_BASE_URL,
        http_client=get_shared_http_client(),
        temperature=0.0,
        max_tokens=400,
        request_timeout=20,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _parse_judge_json(text: str) -> dict:
    try:
        return json.loads(text)
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
    text = llm.invoke(prompt).content
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
    text = llm.invoke(prompt).content
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
        result = service._groundedness_llm_call(answer, docs, state=None)
        return {
            "key": "groundedness",
            "score": 0.0 if result.get("unsupported") else 1.0,
            "comment": f"{result.get('reasoning', '')} claims={result.get('claims', [])}",
        }

    return groundedness_evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", default="pdf-only-qa-113")
    parser.add_argument("--chroma-url", default="http://43.210.64.119:3001")
    parser.add_argument("--es-url", default="http://43.210.35.109:3001")
    parser.add_argument("--max-concurrency", type=int, default=4, help="parallel requests — keep modest, these hit live bot instances serving real traffic too")
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument("--skip-es", action="store_true")
    args = parser.parse_args()

    from langsmith import evaluate

    evaluators = [correctness_evaluator, relevance_evaluator, _make_groundedness_evaluator()]

    if not args.skip_chroma:
        print(f"\n=== Running Chroma baseline experiment against {args.chroma_url} ===")
        chroma_results = evaluate(
            _make_target(args.chroma_url),
            data=args.dataset_name,
            evaluators=evaluators,
            experiment_prefix="chroma-baseline",
            max_concurrency=args.max_concurrency,
        )
        print(f"Chroma experiment done: {chroma_results}")

    if not args.skip_es:
        print(f"\n=== Running ES candidate experiment against {args.es_url} ===")
        es_results = evaluate(
            _make_target(args.es_url),
            data=args.dataset_name,
            evaluators=evaluators,
            experiment_prefix="es-hybrid",
            max_concurrency=args.max_concurrency,
        )
        print(f"ES experiment done: {es_results}")

    print("\nDone. Compare the two experiments side-by-side in the LangSmith UI (same dataset, two experiment_prefix values).")


if __name__ == "__main__":
    main()

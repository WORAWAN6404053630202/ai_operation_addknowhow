"""
Reranker backend (pytorch vs onnx) comparison harness — READY TO RUN ON EC2.

Usage:
    # Pass 1: start the server with RERANKER_BACKEND=pytorch (or unset), then:
    python3 reranker_backend_compare.py --tag pytorch_baseline --url http://localhost:3000

    # Pass 2: restart the server with RERANKER_BACKEND=onnx, then:
    python3 reranker_backend_compare.py --tag onnx_backend --url http://localhost:3000

    # After both passes exist as chat_results_<tag>.json:
    python3 reranker_backend_compare.py --report pytorch_baseline onnx_backend

Background: RERANKER_BACKEND=onnx (see conf.py) swaps the float32 CrossEncoder
reranker for a quantized ONNX Runtime path. It was validated for RANKING ACCURACY
on a 23-query Thai golden eval set (built from this project's own Chroma license_type
metadata — see conf.py's RERANKER_BACKEND comment for the numbers) with negligible
loss. It was NOT validated for LATENCY on real hardware — a local Mac (Apple Silicon)
test showed it was actually SLOWER there, because (a) the float32 path auto-uses
Apple's MPS GPU while ONNX Runtime here is CPU-only, and (b) the quantized kernels are
built for x86 AVX2/AVX512-VNNI instructions that don't exist on ARM64 at all. Neither
of those conditions holds on the real x86 EC2 deploy target, so this script exists to
get an honest answer there instead of guessing from an unrepresentative machine.

Scenario coverage (an earlier, narrower pass only covered "regulatory, single-turn,
Practical" and missed everything below — this version is the complete replacement):
  - regulatory            : license_type-bearing docs (Practical)
  - regulatory_multiturn  : slot-filling follow-up turn in the SAME session (does the
                             reranker fire correctly again on turn 2, not just turn 1?)
  - business_guide        : the LARGEST data_type in the corpus (542/941 docs, no
                             license_type)
  - marketing             : second data_type (222/941 docs)
  - broad_question        : triggers the multi-query/large-candidate-pool retrieval
                             path (up to LLM_DOCS_MAX_BROAD=25 docs) — the case where
                             reranker latency matters most
  - greeting_no_rag       : must NOT touch the reranker at all regardless of backend
                             flag — sanity check that the flag can't affect a path
                             that shouldn't reach it
  - academic              : the OTHER persona (GPT-5.1, separate retrieval/rerank call
                             site in persona_academic.py)

The --report mode only diffs latency + response length per category/step — it cannot
judge whether the actual answer text is still correct/complete. Re-read the saved
chat_results_<tag>.json files (or paste specific query pairs to an LLM judge) for that.
"""
import argparse
import json
import os
import sys
import time

import requests

HERE = os.path.dirname(__file__)


SCENARIOS = [
    # ---- regulatory (Practical) — single turn ----
    {"category": "regulatory", "persona_id": "practical", "steps": [
        "ขายเบียร์ในร้านอาหารต้องขอใบอนุญาตอะไร"]},
    {"category": "regulatory", "persona_id": "practical", "steps": [
        "ค่าธรรมเนียมใบอนุญาตขายอาหารเท่าไหร่"]},
    {"category": "regulatory", "persona_id": "practical", "steps": [
        "จดทะเบียนพาณิชย์ร้านอาหารใช้เอกสารอะไรบ้าง"]},
    {"category": "regulatory", "persona_id": "practical", "steps": [
        "อยากติดเครื่องรูดบัตรเครดิตในร้านต้องขอที่ไหน"]},
    {"category": "regulatory", "persona_id": "practical", "steps": [
        "ร้านอาหารต้องจดภาษีมูลค่าเพิ่มตอนไหน"]},

    # ---- regulatory — multi-turn slot-filling (does turn 2's rerank still work?) ----
    {"category": "regulatory_multiturn", "persona_id": "practical", "steps": [
        "จดทะเบียนพาณิชย์", "บุคคลธรรมดา"]},
    {"category": "regulatory_multiturn", "persona_id": "practical", "steps": [
        "ขอใบอนุญาตขายสุรา", "นิติบุคคล"]},

    # ---- business_guide (542/941 docs — biggest data_type) ----
    {"category": "business_guide", "persona_id": "practical", "steps": [
        "อยากรู้วิธีจัดการสต๊อกวัตถุดิบในร้านอาหารไม่ให้ของเสีย"]},
    {"category": "business_guide", "persona_id": "practical", "steps": [
        "เปิดร้านอาหารใหม่ต้องเตรียมเงินลงทุนเท่าไหร่"]},
    {"category": "business_guide", "persona_id": "practical", "steps": [
        "อยากตั้งราคาเมนูอาหารให้เหมาะสมทำยังไงดี"]},
    {"category": "business_guide", "persona_id": "practical", "steps": [
        "ควรวางผังร้านอาหารยังไงให้ลูกค้าเดินสะดวก"]},

    # ---- marketing (222/941 docs) ----
    {"category": "marketing", "persona_id": "practical", "steps": [
        "อยากทำการตลาดออนไลน์ให้ร้านอาหารทำยังไงดี"]},
    {"category": "marketing", "persona_id": "practical", "steps": [
        "อยากเพิ่มยอดขายร้านอาหารแบบง่ายๆทำอะไรได้บ้าง"]},

    # ---- broad question (large candidate pool, up to 25 docs) ----
    {"category": "broad_question", "persona_id": "practical", "steps": [
        "อยากเปิดร้านกาแฟต้องเตรียมอะไรบ้างตั้งแต่ต้นจนจบ"]},
    {"category": "broad_question", "persona_id": "practical", "steps": [
        "เปิดร้านอาหารต้องขอใบอนุญาตอะไรบ้าง"]},

    # ---- greeting/smalltalk — MUST NOT reach the reranker at all ----
    {"category": "greeting_no_rag", "persona_id": "practical", "steps": [
        "สวัสดีค่ะ"]},
    {"category": "greeting_no_rag", "persona_id": "practical", "steps": [
        "ขอบคุณมากค่ะ"]},

    # ---- academic persona — completely different retrieval/rerank call site ----
    {"category": "academic", "persona_id": "academic", "steps": [
        "อยากทราบขั้นตอนขอใบอนุญาตจำหน่ายสุราแบบละเอียด"]},
    {"category": "academic", "persona_id": "academic", "steps": [
        "อยากรู้รายละเอียดการจดทะเบียนพาณิชย์แบบครบถ้วน"]},
]


def new_session(base_url: str, http: requests.Session, persona_id: str) -> str:
    resp = http.post(f"{base_url}/api/v1/greeting", json={"persona_id": persona_id}, timeout=60)
    resp.raise_for_status()
    return resp.json()["session_id"]


def run(base_url: str, tag: str):
    http = requests.Session()
    results = []
    for i, sc in enumerate(SCENARIOS):
        session_id = new_session(base_url, http, sc["persona_id"])
        step_results = []
        for msg in sc["steps"]:
            t0 = time.time()
            resp = http.post(f"{base_url}/api/v1/chat", json={"message": msg, "session_id": session_id}, timeout=180)
            elapsed = time.time() - t0
            data = resp.json()
            step_results.append({
                "message": msg,
                "elapsed_s": round(elapsed, 2),
                "response": data.get("response", ""),
                "persona_id": data.get("persona_id"),
                "cached": data.get("cached", False),
            })
            print(f"[{tag}] [{sc['category']:22s}] {elapsed:6.2f}s  {msg[:40]:40s} -> {len(data.get('response',''))} chars (persona={data.get('persona_id')})")
        results.append({
            "category": sc["category"],
            "persona_id_requested": sc["persona_id"],
            "session_id": session_id,
            "steps": step_results,
        })

    out_path = os.path.join(HERE, f"chat_results_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} scenarios to {out_path}")


def report(tags):
    all_runs = {}
    for tag in tags:
        path = os.path.join(HERE, f"chat_results_{tag}.json")
        with open(path, encoding="utf-8") as f:
            all_runs[tag] = json.load(f)

    n_scenarios = min(len(r) for r in all_runs.values())
    if any(len(r) != n_scenarios for r in all_runs.values()):
        print(f"WARNING: tag result counts differ ({[len(all_runs[t]) for t in tags]}) — "
              f"comparing only the first {n_scenarios} scenarios (same SCENARIOS list order assumed).")

    # Track a running per-category index (1-based) for readable labels — relies on
    # every tag's results list being produced by the same SCENARIOS order (true as
    # long as both runs used this same script), so zipping by position is valid.
    cat_counts: dict = {}
    print(f"{'category':24s} {'step':4s}", *[f"{t:>18s}" for t in tags])
    for i in range(n_scenarios):
        cat = all_runs[tags[0]][i]["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        label = f"{cat} #{cat_counts[cat]}"
        n_steps = max(len(all_runs[t][i]["steps"]) for t in tags)
        for step_i in range(n_steps):
            row = [f"{label:24s}", f"{step_i+1:4d}"]
            for tag in tags:
                steps = all_runs[tag][i]["steps"]
                if step_i < len(steps):
                    st = steps[step_i]
                    row.append(f"{st['elapsed_s']:6.1f}s/{len(st['response']):5d}c")
                else:
                    row.append(" " * 18)
            print(" ".join(row))

    print("\nRe-read chat_results_<tag>.json manually for full response text diffing —")
    print("this report only compares latency + response length per category/step, since")
    print("content-correctness needs a human (or LLM-judge) read, not just a number.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="Label for this run's output file, e.g. pytorch_baseline")
    ap.add_argument("--url", default="http://localhost:3000", help="Base URL of the running server")
    ap.add_argument("--report", nargs="+", help="Compare N previously-saved tags instead of running")
    args = ap.parse_args()

    if args.report:
        report(args.report)
        return

    if not args.tag:
        print("Need --tag (or --report <tag1> <tag2>)", file=sys.stderr)
        sys.exit(1)

    run(args.url.rstrip("/"), args.tag)


if __name__ == "__main__":
    main()

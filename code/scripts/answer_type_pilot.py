"""
answer_type_pilot.py — Manual regression/pilot suite for Practical mode's 9 answer types.

⚠️ COSTS REAL MONEY — fires live LLM calls against a running server. This is NOT a
pytest test (it is not free, not deterministic, not meant for CI) — run it deliberately,
by hand, when you've changed a prompt or want to sanity-check speed/behavior.

Prerequisite: the server must already be running and reachable (uvicorn app:app).

Usage:
    PYTHONPATH="$PWD" python -m code.scripts.answer_type_pilot --tier smoke
    PYTHONPATH="$PWD" python -m code.scripts.answer_type_pilot --tier full
    PYTHONPATH="$PWD" python -m code.scripts.answer_type_pilot --tier full --url http://127.0.0.1:3000

Background: Restbiz_answer_types-7.pdf documents 9 distinct Practical-mode answer types,
each with an example question and a measured/expected wall-clock time range under normal
conditions. This script fires one representative question per type, measures actual time,
flags anything outside the documented range, and scans each answer for the exact
forbidden-phrase patterns already listed in prompts_practical.py's own "NEVER say X" rules
(e.g. "เอกสารที่ผมมีให้ข้อมูลว่า", "ไม่พบในเอกสาร") — this is a direct, cheap way to catch
the kind of evidence-policy violation found during real testing (the VAT-timing
regression), without needing to read every answer by hand.

Tiers:
  smoke — types 3, 5, 7 (3 questions): fast + cheap, run often after small prompt tweaks.
          Picked to span the speed spectrum (fastest/slowest/mid) and includes type 3,
          the exact category where a real regression was caught during manual testing.
  full  — all 9 types: run before shipping any non-trivial prompt change.
"""
from __future__ import annotations

import argparse
import time
import uuid
from dataclasses import dataclass
from typing import List

import requests


@dataclass
class AnswerType:
    id: int
    name: str
    question: str
    expected_min_s: float
    expected_max_s: float


# Source: Restbiz_answer_types-7.pdf — example question + documented timing range per type.
ANSWER_TYPES: List[AnswerType] = [
    AnswerType(1, "ตอบครบทุกขั้นตอน", "ขอขั้นตอนการขอใบอนุญาตจัดตั้งร้านอาหารทั้งหมดเลย", 25, 55),
    AnswerType(2, "ตอบความหมาย", "ใบอนุญาตสุขาภิบาลอาหารคืออะไร", 20, 35),
    # NOTE: the PDF's own example ("ค่าธรรมเนียม...ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร") was
    # tried first but dropped — verified against actual Chroma metadata that neither
    # entity_type nor location changes the fee for an 8-sqm shop (flat 200 baht either way),
    # yet the model still asked a clarifying question (entity_type, or location, varying
    # across runs) even when both were supplied up front. That's a real minor over-asking
    # inconsistency worth flagging separately — but it makes that question unreliable for a
    # *speed* regression test. Using the VAT-timing question instead, since it has answered
    # directly (no clarifying question) across every test this session.
    AnswerType(3, "ตอบข้อมูลเฉพาะด้าน", "ต้องจดทะเบียนภาษีมูลค่าเพิ่ม VAT ตอนไหนครับ", 15, 25),
    AnswerType(4, "ตอบเฉพาะส่วนที่ถาม", "มีใบอนุญาตไหนบ้างที่ต้องต่ออายุ", 25, 35),
    AnswerType(5, "ตอบภาพรวมกว้าง", "จะเปิดร้านอาหารต้องทำอะไรบ้าง", 45, 65),
    AnswerType(6, "ตอบเนื้อหาทั้งบท", "อยากรู้เรื่องกลยุทธ์ด้านราคาทั้งหมดเลย", 50, 80),
    AnswerType(7, "ตอบแนะนำหัวข้อ", "ใบทะเบียนพาณิชย์", 10, 20),
    AnswerType(8, "ตอบหลายเรื่องพร้อมกัน (≤3)", "ภาษี VAT กับประกันสังคมต้องทำยังไง", 20, 50),
    AnswerType(
        9, "สรุปรายการเมื่อถามหลายเรื่อง (≥4)",
        "ต้องการข้อมูลเรื่องทะเบียนพาณิชย์ VAT ประกันสังคม และใบอนุญาตจัดตั้งร้านอาหาร",
        25, 35,
    ),
]

SMOKE_TIER_IDS = {3, 5, 7}

# Exact "NEVER say" phrases already declared forbidden in prompts_practical.py.
# A hit here means the model violated an existing rule — a real compliance regression,
# not a style nitpick.
FORBIDDEN_PHRASES = [
    "ไม่พบในเอกสาร", "เอกสารที่ผมมีไม่ระบุ", "เอกสารที่ผมมีให้ข้อมูลว่า",
    "เอกสารระบุว่า", "จากเอกสาร", "ข้อมูลระบุว่า", "ในเอกสารที่ผมมี",
    "เอกสารที่มีอยู่", "ตามเอกสาร", "ข้อมูลในเอกสาร",
    "เท่าที่รู้", "เท่าที่ทราบ", "ตามที่ผมทราบ", "ข้อมูลที่ผมมี", "ในข้อมูลที่มี",
    "ผมเพิ่งอธิบายไปแล้ว", "ตอบไปแล้ว", "ดูจากข้อความก่อนหน้า", "ดังที่กล่าวไว้",
]


def run_one(base_url: str, at: AnswerType) -> dict:
    session_id = f"pilot_{at.id}_{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/api/v1/chat",
            json={"message": at.question, "session_id": session_id},
            timeout=120,
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        answer = resp.json().get("response", "")
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed": time.perf_counter() - t0}

    violations = [p for p in FORBIDDEN_PHRASES if p in answer]
    return {
        "ok": True,
        "elapsed": elapsed,
        "within_range": at.expected_min_s <= elapsed <= at.expected_max_s,
        "answer": answer,
        "answer_len": len(answer),
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--show-answers", action="store_true", help="Print full answer text, not just a length/status summary")
    args = parser.parse_args()

    types = [a for a in ANSWER_TYPES if args.tier == "full" or a.id in SMOKE_TIER_IDS]

    print("=" * 78)
    print(f"Restbiz Practical answer-type pilot — tier={args.tier} ({len(types)} questions)")
    print("Costs real money (live LLM calls). Make sure the server is already running.")
    print("=" * 78)

    results = []
    for at in types:
        print(f"\n[{at.id}] {at.name}: {at.question!r}")
        r = run_one(args.url, at)
        results.append((at, r))
        if not r["ok"]:
            print(f"  FAILED: {r['error']}")
            continue
        speed_flag = "OK" if r["within_range"] else "OUT-OF-RANGE"
        print(f"  {speed_flag} — {r['elapsed']:.1f}s (expected {at.expected_min_s}-{at.expected_max_s}s), answer_len={r['answer_len']}")
        if r["violations"]:
            print(f"  COMPLIANCE VIOLATION — forbidden phrase(s) found: {r['violations']}")
        if args.show_answers:
            print(f"  --- answer ---\n{r['answer']}\n  --------------")

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    n_fail = sum(1 for _, r in results if not r["ok"])
    n_out = sum(1 for _, r in results if r["ok"] and not r["within_range"])
    n_violation = sum(1 for _, r in results if r["ok"] and r["violations"])
    for at, r in results:
        if not r["ok"]:
            status = "FAIL"
        elif r["violations"]:
            status = "VIOLATION"
        elif not r["within_range"]:
            status = "OUT-OF-RANGE"
        else:
            status = "OK"
        elapsed_str = f"{r['elapsed']:.1f}s" if "elapsed" in r else "-"
        print(f"  [{at.id}] {at.name:35s} {status:14s} {elapsed_str}")

    n_ok = len(results) - n_fail - n_out - n_violation
    print(f"\n{n_ok}/{len(results)} OK, {n_out} out-of-range, {n_violation} compliance violations, {n_fail} failed.")
    if n_violation or n_fail:
        print("⚠️  Review the flagged answers above before shipping this prompt change.")


if __name__ == "__main__":
    main()

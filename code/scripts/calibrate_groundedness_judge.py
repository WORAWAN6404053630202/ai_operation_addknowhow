"""
calibrate_groundedness_judge.py — Manual calibration check for the groundedness
LLM-judge (PracticalPersonaService._groundedness_llm_call, persona_practical.py).

⚠️ COSTS REAL MONEY — fires live Haiku calls via OpenRouter. This is NOT a pytest
test (not free, not part of `pytest tests/ -m "not integration"`, not run in CI) —
run it deliberately, by hand, whenever the judge prompt in
_default_groundedness_llm_call changes.

Usage:
    venv/bin/python3 code/scripts/calibrate_groundedness_judge.py

Background: 2026-10 research pass on LLM-as-judge best practice flagged "shipping
an uncalibrated judge" — one never checked against cases with a known-correct
answer — as the single most common production mistake. This script is that check.
Part 1 calls the REAL production judge function (not a reimplementation)
against 6 held-out cases, deliberately worded differently from the 3 few-shot
examples embedded in the judge's own prompt — the point is confirming the
judge generalizes the PATTERN, not that it can recognize its own examples
verbatim. Part 2 goes further: runs the REAL _apply_groundedness_safety_net
end to end (real judge call -> real retry-generation call -> real re-check
call) against one deliberately fabricated answer, to prove the full
detect-and-correct round trip actually works, not just the classifier alone.

Cost: ~9 calls total to the OPENROUTER_SWITCH_MODEL (claude-haiku-4-5 by
default) plus one call to the main answer model for the retry generation —
well under $0.05 total.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))


@dataclass
class Case:
    name: str
    expect_unsupported: bool
    docs_json: List[dict]
    answer_text: str


CASES: List[Case] = [
    # ── Should flag (3) — same PATTERN as the 3 real QA-reproduced fabrication
    # cases, deliberately different wording/values from the judge's own few-shot
    # examples so this tests generalization, not memorization.
    Case(
        name="invented tool/platform requirement",
        expect_unsupported=True,
        docs_json=[{
            "metadata": {"operation_steps": "ยื่นแบบคำขอที่เคาน์เตอร์สำนักงานเขต แนบสำเนาบัตรประชาชนและทะเบียนบ้าน"},
            "content": "ยื่นแบบคำขอที่เคาน์เตอร์สำนักงานเขต แนบสำเนาบัตรประชาชนและทะเบียนบ้าน",
        }],
        answer_text="ยื่นแบบคำขอที่เคาน์เตอร์สำนักงานเขต แนบสำเนาบัตรประชาชนและทะเบียนบ้าน จากนั้นต้องยืนยันตัวตนผ่านแอป ThaiD ก่อนเจ้าหน้าที่จะรับเรื่อง",
    ),
    Case(
        name="invented fee amount",
        expect_unsupported=True,
        docs_json=[{
            "metadata": {"fees": "ไม่พบข้อมูลค่าธรรมเนียมในเอกสารที่มี"},
            "content": "ไม่พบข้อมูลค่าธรรมเนียมในเอกสารที่มี",
        }],
        answer_text="ค่าธรรมเนียมในการขอใบอนุญาตนี้อยู่ที่ 1,200 บาทครับ",
    ),
    Case(
        name="invented named procedure",
        expect_unsupported=True,
        docs_json=[{
            "metadata": {"operation_steps": "ติดต่อเจ้าหน้าที่เพื่อขอแก้ไขข้อมูลในระบบ"},
            "content": "ติดต่อเจ้าหน้าที่เพื่อขอแก้ไขข้อมูลในระบบ",
        }],
        answer_text="ติดต่อเจ้าหน้าที่เพื่อขอแก้ไขข้อมูล โดยต้องทำ 'คำร้องขอรีเซ็ตรหัสผ่าน' ในระบบก่อนถึงจะแก้ไขข้อมูลได้",
    ),
    # ── Should NOT flag (3)
    Case(
        name="correct summary of real DOCUMENTS content",
        expect_unsupported=False,
        docs_json=[{
            "metadata": {"operation_steps": "ยื่นคำขอที่สำนักงานเขต รอผลพิจารณา 15 วันทำการ", "fees": "500 บาท"},
            "content": "ยื่นคำขอที่สำนักงานเขต รอผลพิจารณา 15 วันทำการ ค่าธรรมเนียม 500 บาท",
        }],
        answer_text="ยื่นคำขอที่สำนักงานเขตครับ ใช้เวลาพิจารณาประมาณ 15 วันทำการ ค่าธรรมเนียม 500 บาท",
    ),
    Case(
        name="honest info-gap refusal",
        expect_unsupported=False,
        docs_json=[{
            "metadata": {"operation_steps": "ไม่พบข้อมูลในเอกสารที่มี"},
            "content": "ไม่พบข้อมูลในเอกสารที่มี",
        }],
        answer_text="ขออภัยครับ ยังไม่พบข้อมูลเรื่องนี้ในเอกสารที่มี [[INFO_GAP]] แนะนำให้สอบถามหน่วยงานที่เกี่ยวข้องโดยตรงเพิ่มเติมครับ",
    ),
    Case(
        name="generic framing, no specific numbers/names",
        expect_unsupported=False,
        docs_json=[{
            "metadata": {"operation_steps": "ยื่นคำขอที่สำนักงานเขต"},
            "content": "ยื่นคำขอที่สำนักงานเขต",
        }],
        answer_text="โดยทั่วไปแล้วการขอใบอนุญาตประเภทนี้จะต้องเตรียมเอกสารให้ครบถ้วนก่อนยื่นคำขอครับ มีคำถามอื่นเพิ่มเติมไหมครับ",
    ),
]


def run_judge_calibration(service) -> bool:
    print("=" * 78)
    print(f"Part 1 — judge classifier alone, {len(CASES)} held-out cases")
    print("=" * 78)

    correct = 0
    for case in CASES:
        result = service._groundedness_llm_call(case.answer_text, case.docs_json)
        got = bool(result.get("unsupported"))
        ok = got == case.expect_unsupported
        correct += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {case.name}")
        print(f"  expected unsupported={case.expect_unsupported}  got unsupported={got}")
        print(f"  reasoning: {result.get('reasoning', '')!r}")
        if result.get("claims"):
            print(f"  claims flagged: {result['claims']}")

    print(f"\nAccuracy: {correct}/{len(CASES)} ({100 * correct / len(CASES):.0f}%)")
    return correct == len(CASES)


def run_end_to_end_retry_check(service) -> bool:
    """Part 2 — the part Part 1 can't test: when the judge flags something,
    does the REAL retry-once mechanism (a real _call_llm_json_with_truncation_
    retry call, not a mock) actually produce a corrected answer? Part 1 only
    proves the classifier works in isolation; this proves the full safety net
    (detect -> retry -> re-verify -> replace decision) works end to end."""
    print("\n" + "=" * 78)
    print("Part 2 — full retry-once safety net, real round trip")
    print("=" * 78)

    docs_json = [{
        "metadata": {"fees": "ไม่พบข้อมูลค่าธรรมเนียมในเอกสารที่มี"},
        "content": "ไม่พบข้อมูลค่าธรรมเนียมในเอกสารที่มี",
    }]
    fabricated_answer = "ค่าธรรมเนียมในการขอใบอนุญาตนี้อยู่ที่ 1,200 บาทครับ"
    prompt = (
        "USER INPUT:\nค่าธรรมเนียมขอใบอนุญาตนี้เท่าไหร่ครับ\n\n"
        f"DOCUMENTS:\n{json_dumps_safe(docs_json)}\n\n"
        "ROUND: 1/7"
    )
    decision = {"action": "answer", "execution": {"answer": fabricated_answer}}
    exec_ = {"answer": fabricated_answer}

    print(f"\nStarting (deliberately fabricated) answer:\n  {fabricated_answer!r}")

    out_decision, out_exec = service._apply_groundedness_safety_net(
        "answer", exec_, decision, docs_json, prompt,
    )
    final_answer = str(out_exec.get("answer") or "")
    was_corrected = final_answer != fabricated_answer
    still_has_fake_number = "1,200" in final_answer or "1200" in final_answer

    print(f"\nFinal answer after safety net:\n  {final_answer!r}")
    print(f"\nWas the decision replaced by a retry? {was_corrected}")
    print(f"Does the final answer still contain the fabricated 1,200-baht claim? {still_has_fake_number}")

    ok = was_corrected and not still_has_fake_number
    print(f"\n[{'PASS' if ok else 'FAIL'}] end-to-end retry {'corrected' if ok else 'did NOT correct'} the fabricated answer")
    return ok


def json_dumps_safe(obj) -> str:
    import json as _json
    return _json.dumps(obj, ensure_ascii=False)


def main() -> int:
    from model.persona_practical import PracticalPersonaService

    service = PracticalPersonaService(retriever=MagicMock())

    part1_ok = run_judge_calibration(service)
    part2_ok = run_end_to_end_retry_check(service)

    print("\n" + "=" * 78)
    print(f"Part 1 (classifier): {'PASS' if part1_ok else 'FAIL'}")
    print(f"Part 2 (end-to-end retry): {'PASS' if part2_ok else 'FAIL'}")
    print("=" * 78)
    return 0 if (part1_ok and part2_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

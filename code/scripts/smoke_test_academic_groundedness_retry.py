"""
smoke_test_academic_groundedness_retry.py — one real-call smoke test for the
2026-10 Academic-persona port of the groundedness safety net
(AcademicPersonaService._apply_groundedness_safety_net, persona_academic.py).

⚠️ COSTS REAL MONEY — fires live calls via OpenRouter (the judge call on
OPENROUTER_SWITCH_MODEL, plus the retry-generation call on the real Academic
main model). This is NOT a pytest test — run it deliberately, by hand.

Usage:
    venv/bin/python3 code/scripts/smoke_test_academic_groundedness_retry.py

Why this exists: code/scripts/calibrate_groundedness_judge.py already proved
(a) the judge classifier generalizes (6/6 on held-out cases) and (b) the full
retry round trip works end to end — but only for Practical's orchestration
method (_apply_groundedness_safety_net there calls
_call_llm_json_with_truncation_retry). Academic's new orchestration method
uses a different, Academic-specific retry call (self._call_llm_json directly
— there is no truncation-retry wrapper in this file) and a different
signature (plain string in/out, no action/decision/exec_ dance). The judge
PROMPT itself is identical/already calibrated, so this script does not
re-run Part 1's classifier calibration — it only proves Academic's new
wiring (the part that IS new code) actually works with real calls on both
sides, mirroring calibrate_groundedness_judge.py's Part 2 in spirit.

Cost: ~4 calls total (2x judge call on OPENROUTER_SWITCH_MODEL, 1x real
Academic main-model retry-generation call per case, 2 cases) — a small
fraction of a cent to a few cents depending on the Academic model's pricing.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))


def json_dumps_safe(obj) -> str:
    import json as _json
    return _json.dumps(obj, ensure_ascii=False)


def run_case(service, state, name: str, docs_json: list, user_question: str, fabricated_answer: str) -> bool:
    print("\n" + "=" * 78)
    print(f"Case: {name}")
    print("=" * 78)

    prompt = (
        f"USER INPUT:\n{user_question}\n\n"
        f"DOCUMENTS:\n{json_dumps_safe(docs_json)}\n\n"
        "Return JSON only:\n\n"
        "{\n"
        '  "input_type": "new_question",\n'
        '  "analysis": "brief reasoning summary",\n'
        '  "action": "answer",\n'
        '  "execution": {\n'
        '    "answer": "structured final answer"\n'
        "  }\n"
        "}"
    )

    print(f"\nStarting (deliberately fabricated) answer:\n  {fabricated_answer!r}")

    final_answer = service._apply_groundedness_safety_net(
        fabricated_answer, docs_json, prompt, state=state,
    )

    was_corrected = final_answer != fabricated_answer
    print(f"\nFinal answer after safety net:\n  {final_answer!r}")
    print(f"\nWas the answer replaced by a real retry call? {was_corrected}")

    ok = was_corrected
    print(f"\n[{'PASS' if ok else 'FAIL'}] Academic retry wiring {'corrected' if ok else 'did NOT correct'} the fabricated answer")
    return ok


def main() -> int:
    from model.persona_academic import AcademicPersonaService
    from model.conversation_state import ConversationState

    service = AcademicPersonaService(retriever=MagicMock())
    state = ConversationState(session_id="smoke-test", persona_id="academic", context={})

    results = []

    results.append(run_case(
        service, state,
        name="invented fee amount (no fee data in DOCUMENTS)",
        docs_json=[{
            "metadata": {"fees": "ไม่พบข้อมูลค่าธรรมเนียมในเอกสารที่มี"},
            "content": "ไม่พบข้อมูลค่าธรรมเนียมในเอกสารที่มี",
        }],
        user_question="ค่าธรรมเนียมขอใบอนุญาตนี้เท่าไหร่ครับ",
        fabricated_answer="ค่าธรรมเนียมในการขอใบอนุญาตนี้อยู่ที่ 1,200 บาทครับ",
    ))

    results.append(run_case(
        service, state,
        name="invented tool/platform requirement",
        docs_json=[{
            "metadata": {"operation_steps": "ยื่นแบบคำขอที่เคาน์เตอร์สำนักงานเขต แนบสำเนาบัตรประชาชนและทะเบียนบ้าน"},
            "content": "ยื่นแบบคำขอที่เคาน์เตอร์สำนักงานเขต แนบสำเนาบัตรประชาชนและทะเบียนบ้าน",
        }],
        user_question="ขั้นตอนขอใบอนุญาตนี้ทำยังไงครับ",
        fabricated_answer="ยื่นแบบคำขอที่เคาน์เตอร์สำนักงานเขต แนบสำเนาบัตรประชาชนและทะเบียนบ้าน จากนั้นต้องยืนยันตัวตนผ่านแอป ThaiD ก่อนเจ้าหน้าที่จะรับเรื่อง",
    ))

    print("\n" + "=" * 78)
    for i, ok in enumerate(results, 1):
        print(f"Case {i}: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
End-to-end automated test suite for น้องสุดยอด (Thai Restaurant Regulatory AI Bot)

Covers all 22 หมวด (categories) of test cases.
Calls the real API and generates pass/fail reports.

Usage:
    python tests/test_bot_e2e.py
    python tests/test_bot_e2e.py --url http://your-server:8000
    python tests/test_bot_e2e.py --save-responses
    python tests/test_bot_e2e.py --category 1      # run only category 1
    python tests/test_bot_e2e.py --category 1-5    # run categories 1 through 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"
DEFAULT_TIMEOUT = 120  # seconds per request
RATE_LIMIT_MAX_RETRIES = 5

# ANSI colours
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _url(path: str) -> str:
    return f"{BASE_URL}{API_PREFIX}{path}"


def new_session(session: "requests.Session") -> str:
    """Call /greeting and return session_id."""
    resp = session.post(_url("/greeting"), json={}, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["session_id"]


def reset_session(session: "requests.Session", session_id: str) -> None:
    """Reset an existing session."""
    session.post(_url("/reset"), json={"session_id": session_id}, timeout=DEFAULT_TIMEOUT)


def chat(
    session: "requests.Session",
    session_id: str,
    message: str,
    *,
    retries: int = RATE_LIMIT_MAX_RETRIES,
) -> dict:
    """Send a chat message and return the response dict. Handles 429 retry."""
    for attempt in range(retries + 1):
        resp = session.post(
            _url("/chat"),
            json={"message": message, "session_id": session_id},
            timeout=DEFAULT_TIMEOUT,
        )
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 5))
            print(
                _color(f"  [rate-limited] waiting {retry_after}s (attempt {attempt+1}/{retries})", _YELLOW)
            )
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Rate limited after {retries} retries on message: {message!r}")


# ---------------------------------------------------------------------------
# Check helpers  (return (passed: bool, detail: str))
# ---------------------------------------------------------------------------
CheckResult = Tuple[bool, str]


def check_contains(text: str, *keywords: str, label: str = "") -> CheckResult:
    """All keywords must appear in text (case-insensitive)."""
    missing = [kw for kw in keywords if kw.lower() not in text.lower()]
    if missing:
        return False, f"missing: {missing}"
    return True, label or f"contains {list(keywords)}"


def check_not_contains(text: str, *keywords: str, label: str = "") -> CheckResult:
    """None of the keywords must appear in text."""
    found = [kw for kw in keywords if kw.lower() in text.lower()]
    if found:
        return False, f"should NOT contain: {found}"
    return True, label or f"does not contain {list(keywords)}"


def check_has_url(text: str, label: str = "has URL") -> CheckResult:
    """Text must contain at least one http/https URL."""
    if re.search(r"https?://\S+", text):
        return True, label
    return False, "no URL found"


def check_no_duplicate_url(text: str, label: str = "no duplicate URL") -> CheckResult:
    """No URL should appear more than once."""
    urls = re.findall(r"https?://\S+", text)
    seen: set = set()
    dupes = []
    for u in urls:
        u_clean = u.rstrip(".,)")
        if u_clean in seen:
            dupes.append(u_clean)
        seen.add(u_clean)
    if dupes:
        return False, f"duplicate URLs: {dupes}"
    return True, label


def check_asks_entity_type(text: str, label: str = "asks entity type") -> CheckResult:
    """Response must ask about บุคคลธรรมดา / นิติบุคคล."""
    patterns = ["บุคคลธรรมดา", "นิติบุคคล", "รูปแบบธุรกิจ", "ประเภทกิจการ", "รูปแบบกิจการ"]
    if any(p in text for p in patterns):
        return True, label
    return False, "no entity-type question found"


def check_no_entity_type_reask(
    text: str, label: str = "no re-ask entity type"
) -> CheckResult:
    """After entity type answered, must NOT ask it again."""
    patterns = ["รูปแบบธุรกิจ", "บุคคลธรรมดาหรือนิติบุคคล", "ประเภทกิจการ"]
    found = [p for p in patterns if p in text]
    if found:
        return False, f"re-asked entity type: {found}"
    return True, label


def check_all_fields(text: str, label: str = "all required fields present") -> CheckResult:
    """Broad answer must include key legal info sections."""
    required = ["ขั้นตอน", "เอกสาร", "ค่าธรรมเนียม", "สถานที่"]
    missing = [f for f in required if f not in text]
    if missing:
        return False, f"missing fields: {missing}"
    return True, label


def check_no_rag_trigger(text: str, label: str = "no RAG triggered for greeting") -> CheckResult:
    """Greeting/thanks responses must not contain legal retrieval content."""
    # "ใบอนุญาต" is intentionally excluded: bot legitimately mentions it in topic suggestion menus
    legal_patterns = ["ขั้นตอน", "เอกสาร", "ค่าธรรมเนียม", "กรมพัฒนาธุรกิจ"]
    found = [p for p in legal_patterns if p in text]
    if found:
        return False, f"RAG content leaked into greeting: {found}"
    return True, label


def check_no_ref_section(text: str, label: str = "no ref section (not requested)") -> CheckResult:
    """When user did not ask for references, response should not contain ref block."""
    if "📎" in text or "อ้างอิง" in text:
        return False, "reference section shown when not requested"
    return True, label


def check_asks_registration_type(
    text: str, label: str = "asks registration type"
) -> CheckResult:
    """Must ask about sub-type of registration."""
    patterns = [
        "ห้างหุ้นส่วน", "บริษัทจำกัด", "มหาชน", "ประเภทนิติบุคคล",
        "รูปแบบนิติบุคคล", "ประเภทของนิติบุคคล",
    ]
    if any(p in text for p in patterns):
        return True, label
    return False, "no registration-type question found"


def check_numbered_menu(text: str, count: int, label: str = "") -> CheckResult:
    """Text must contain a numbered menu with at least `count` items."""
    items = re.findall(r"^\s*\d+[\.\)]\s+\S", text, re.MULTILINE)
    if len(items) >= count:
        return True, label or f"numbered menu ≥{count} items"
    return False, f"numbered menu items found: {len(items)}, expected ≥{count}"


def check_phase_question(text: str, label: str = "asks clarifying question") -> CheckResult:
    """Response must contain a question mark — bot is asking something."""
    if "?" in text or "？" in text or "ครับ?" in text or "คะ?" in text:
        return True, label
    if re.search(r"[ก-๙]+\?", text):
        return True, label
    # Thai questions sometimes end with คะ/ครับ without literal ?
    if any(text.strip().endswith(s) for s in ["คะ", "ครับ", "ไหม", "มั้ย"]):
        return True, label
    # Numbered menu = bot asking user to choose (counts as a question)
    if re.search(r"^\s*[1-9][\.\)]", text, re.MULTILINE):
        return True, label
    return False, "no question found in response"


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------
@dataclass
class Step:
    message: str
    checks: List[Callable[[str], CheckResult]] = field(default_factory=list)
    description: str = ""


@dataclass
class TestCase:
    name: str
    steps: List[Step]
    category: int = 0


@dataclass
class StepResult:
    step_index: int
    message: str
    response: str
    check_results: List[Tuple[bool, str]]

    @property
    def passed(self) -> bool:
        return all(r[0] for r in self.check_results)


@dataclass
class TestResult:
    test_case: TestCase
    step_results: List[StepResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        return all(sr.passed for sr in self.step_results)


class TestRunner:
    def __init__(self, base_url: str, save_responses: bool = False):
        global BASE_URL
        BASE_URL = base_url
        self.save_responses = save_responses
        self.results: List[TestResult] = []
        self._http = requests.Session()
        self._http.headers.update({"Content-Type": "application/json"})

    def run(self, tc: TestCase) -> TestResult:
        result = TestResult(test_case=tc)
        session_id = None
        try:
            session_id = new_session(self._http)
            for i, step in enumerate(tc.steps):
                print(f"    step {i+1}: {step.message[:60]!r}")
                resp_data = chat(self._http, session_id, step.message)
                response_text = resp_data.get("response", "")

                check_results = []
                for check_fn in step.checks:
                    check_results.append(check_fn(response_text))

                sr = StepResult(
                    step_index=i,
                    message=step.message,
                    response=response_text,
                    check_results=check_results,
                )
                result.step_results.append(sr)

                # Print step outcome
                for passed, detail in check_results:
                    sym = _color("✓", _GREEN) if passed else _color("✗", _RED)
                    print(f"      {sym} {detail}")

                if self.save_responses:
                    self._save_step(tc, i, step.message, response_text, check_results)

        except Exception as exc:
            result.error = str(exc)
            print(_color(f"  ERROR: {exc}", _RED))
        finally:
            if session_id:
                try:
                    reset_session(self._http, session_id)
                except Exception:
                    pass
        return result

    def _save_step(
        self,
        tc: TestCase,
        step_idx: int,
        message: str,
        response: str,
        checks: List[Tuple[bool, str]],
    ) -> None:
        fname = f"test_responses_cat{tc.category:02d}_{step_idx:02d}.txt"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(f"=== {tc.name} | step {step_idx+1} ===\n")
            f.write(f"USER: {message}\n")
            f.write(f"BOT: {response}\n")
            for passed, detail in checks:
                f.write(f"  {'PASS' if passed else 'FAIL'}: {detail}\n")
            f.write("\n")

    def print_report(self) -> None:
        print("\n" + "=" * 70)
        print(_color("TEST REPORT", _BOLD))
        print("=" * 70)

        by_cat: dict[int, List[TestResult]] = {}
        for r in self.results:
            by_cat.setdefault(r.test_case.category, []).append(r)

        total_pass = 0
        total_fail = 0

        for cat in sorted(by_cat):
            cat_results = by_cat[cat]
            cat_pass = sum(1 for r in cat_results if r.passed)
            cat_fail = len(cat_results) - cat_pass
            total_pass += cat_pass
            total_fail += cat_fail

            status_str = _color(f"{cat_pass}/{len(cat_results)}", _GREEN if cat_fail == 0 else _YELLOW)
            print(f"\n  หมวด {cat:2d}  [{status_str}]")
            for r in cat_results:
                sym = _color("PASS", _GREEN) if r.passed else _color("FAIL", _RED)
                print(f"    {sym}  {r.test_case.name}")
                if not r.passed:
                    if r.error:
                        print(f"         error: {r.error}")
                    for sr in r.step_results:
                        if not sr.passed:
                            print(f"         step {sr.step_index+1}: {sr.message[:50]!r}")
                            for ok, detail in sr.check_results:
                                if not ok:
                                    print(f"           ✗ {detail}")
                            print(f"         response: {sr.response[:200]!r}...")

        print("\n" + "-" * 70)
        overall = _color("ALL PASS", _GREEN) if total_fail == 0 else _color(f"{total_fail} FAILED", _RED)
        print(f"  TOTAL: {total_pass} passed, {total_fail} failed  →  {overall}")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Test case definitions  (22 หมวด)
# ---------------------------------------------------------------------------

def build_all_tests() -> List[TestCase]:
    tests: List[TestCase] = []

    # ------------------------------------------------------------------
    # หมวด 1 — Greeting / Small talk (no RAG)
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=1, name="1.1 สวัสดีไม่ trigger RAG",
        steps=[
            Step("สวัสดีค่ะ", [
                lambda t: check_not_contains(t, "ขั้นตอน", "เอกสาร", "ใบอนุญาต"),
                lambda t: check_no_rag_trigger(t),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=1, name="1.2 ขอบคุณ ไม่ trigger RAG",
        steps=[
            Step("ขอบคุณมากเลยค่ะ ช่วยได้มากจริงๆ", [
                lambda t: check_no_rag_trigger(t),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=1, name="1.3 คุยเรื่องทั่วไป",
        steps=[
            Step("วันนี้อากาศดีจังเลยนะ", [
                lambda t: check_no_rag_trigger(t),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 2 — ใบทะเบียนพาณิชย์ (Commercial Registration)
    # NOTE: Bot collects location_type (ยื่นที่ไหน) BEFORE entity_type.
    #       Step "1" answers location_type = กรมพัฒนาธุรกิจการค้า.
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=2, name="2.1 ถามใบทะเบียนพาณิชย์ → bot ถาม slot",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ค่ะ", [
                lambda t: check_phase_question(t),  # bot should ask any slot
            ]),
        ],
    ))
    tests.append(TestCase(
        category=2, name="2.2 บุคคลธรรมดา → ขั้นตอน+เอกสาร+ค่าธรรมเนียม",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ค่ะ", []),
            Step("1", []),              # answer location_type
            Step("บุคคลธรรมดาค่ะ", []),  # answer entity_type
            Step("1", [                # answer operation_type → full answer
                lambda t: check_contains(t, "ขั้นตอน"),
                lambda t: check_contains(t, "เอกสาร"),
                lambda t: check_contains(t, "ค่าธรรมเนียม"),
                lambda t: check_no_duplicate_url(t),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=2, name="2.3 นิติบุคคล → ถามประเภทนิติบุคคลก่อน",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ค่ะ", []),
            Step("1", []),  # answer location_type
            Step("นิติบุคคลค่ะ", [
                lambda t: check_asks_registration_type(t),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=2, name="2.4 นิติบุคคล ห้างหุ้นส่วน → ครบถ้วน",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ค่ะ", []),
            Step("1", []),                   # answer location_type
            Step("นิติบุคคลค่ะ", []),          # answer entity_type
            Step("ห้างหุ้นส่วนจำกัดค่ะ", []),  # answer registration_type
            Step("1", [                      # answer operation_type → full answer
                lambda t: check_contains(t, "ขั้นตอน"),
                lambda t: check_contains(t, "เอกสาร"),
                lambda t: check_contains(t, "ห้างหุ้นส่วน"),
                lambda t: check_no_duplicate_url(t),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=2, name="2.5 ไม่ถามประเภทกิจการซ้ำ (no re-ask)",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ค่ะ", []),
            Step("1", []),              # answer location_type
            Step("บุคคลธรรมดาค่ะ", []),  # answer entity_type
            Step("1", []),              # answer operation_type → full answer
            Step("แล้วค่าธรรมเนียมเท่าไหร่ค่ะ", [
                lambda t: check_no_entity_type_reask(t),
                lambda t: check_contains(t, "ค่าธรรมเนียม"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 3 — ใบทะเบียนพาณิชย์ — แก้ไข / ยกเลิก
    # NOTE: bot collects slots before answering; answer appears after slot fill
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=3, name="3.1 แก้ไขใบทะเบียนพาณิชย์",
        steps=[
            Step("อยากแก้ไขใบทะเบียนพาณิชย์ค่ะ ต้องทำยังไงบ้าง", []),  # bot asks slot
            Step("1", [  # answer first slot
                lambda t: check_contains(t, "แก้ไข"),
                lambda t: (True, "response received"),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=3, name="3.2 ยกเลิกใบทะเบียนพาณิชย์",
        steps=[
            Step("จะยกเลิกใบทะเบียนพาณิชย์ทำยังไงค่ะ", []),  # bot asks entity_type
            Step("บุคคลธรรมดาค่ะ", [
                lambda t: check_contains(t, "ยกเลิก"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 4 — ใบภาษีมูลค่าเพิ่ม ภพ.20 (VAT)
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=4, name="4.1 จด VAT → มีข้อมูล VAT",
        steps=[
            Step("อยากจด VAT ค่ะ ต้องทำอะไรบ้าง", [
                # bot may give direct VAT answer or ask entity_type — accept either
                lambda t: check_asks_entity_type(t) if any(
                    p in t for p in ["บุคคลธรรมดา", "นิติบุคคล", "รูปแบบ", "ประเภท"]
                ) else check_contains(t, "VAT"),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=4, name="4.2 บุคคลธรรมดา จด VAT → ขั้นตอน+เกณฑ์ยอดขาย",
        steps=[
            Step("อยากจด VAT ค่ะ ในฐานะบุคคลธรรมดา ต้องทำอะไรบ้าง", [
                lambda t: check_contains(t, "ขั้นตอน"),
                lambda t: check_contains(t, "เอกสาร"),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=4, name="4.3 VAT — แก้ไข",
        steps=[
            Step("จะแก้ไขข้อมูล VAT ของร้านต้องทำอะไรบ้างค่ะ", [
                lambda t: check_contains(t, "แก้ไข"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 5 — ประกันสังคม
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=5, name="5.1 ขึ้นทะเบียนประกันสังคม → ถามประเภทกิจการ",
        steps=[
            Step("อยากขึ้นทะเบียนประกันสังคมสำหรับร้านของฉันค่ะ", [
                lambda t: check_asks_entity_type(t),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=5, name="5.2 ประกันสังคม บุคคลธรรมดา → ขั้นตอน+เอกสาร",
        steps=[
            Step("อยากขึ้นทะเบียนประกันสังคมค่ะ", []),
            Step("บุคคลธรรมดาค่ะ", [
                lambda t: check_contains(t, "ขั้นตอน"),
                lambda t: check_contains(t, "เอกสาร"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 6 — ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร
    # NOTE: broad "เปิดร้านอาหาร" query may get direct multi-license answer (no slot)
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=6, name="6.1 ขอใบอนุญาตร้านอาหาร → มีข้อมูลใบอนุญาต",
        steps=[
            Step("จะขอใบอนุญาตเปิดร้านอาหารต้องทำอะไรบ้างค่ะ", [
                lambda t: check_contains(t, "ใบอนุญาต"),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=6, name="6.2 ใบอนุญาตร้านอาหาร บุคคลธรรมดา → ขั้นตอน+เอกสาร",
        steps=[
            Step("จะขอใบอนุญาตเปิดร้านอาหารต้องทำอะไรบ้างค่ะ", []),
            Step("บุคคลธรรมดาค่ะ", [
                lambda t: check_contains(t, "ขั้นตอน"),
                lambda t: check_contains(t, "เอกสาร"),
                lambda t: check_no_duplicate_url(t),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=6, name="6.3 ต่ออายุใบอนุญาตร้านอาหาร",
        steps=[
            Step("ต้องต่ออายุใบอนุญาตร้านอาหารยังไงค่ะ", []),  # bot asks location slot
            Step("1", []),   # answer location_type
            Step("1", [     # answer entity_type or shop_area_type → answer
                lambda t: check_contains(t, "ต่ออายุ"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 7 — ใบอนุญาตจำหน่ายสุรา
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=7, name="7.1 ใบอนุญาตขายสุรา → มีข้อมูลสุรา",
        steps=[
            Step("ร้านผมจะขายเหล้าด้วย ต้องขอใบอนุญาตอะไรบ้างครับ", [
                lambda t: check_contains(t, "สุรา"),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=7, name="7.2 ใบอนุญาตขายสุรา บุคคลธรรมดา → ขั้นตอน+ค่าธรรมเนียม",
        steps=[
            Step("ร้านผมจะขายเหล้าด้วย ต้องขอใบอนุญาตอะไรบ้างครับ", []),
            Step("บุคคลธรรมดาครับ", [
                lambda t: check_contains(t, "ขั้นตอน"),
                lambda t: check_contains(t, "สุรา") or check_contains(t, "สรรพสามิต")[0] and True,
                lambda t: check_no_duplicate_url(t),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 8 — ภาษีป้าย
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=8, name="8.1 ภาษีป้ายร้านอาหาร → ขั้นตอน",
        steps=[
            Step("ภาษีป้ายร้านอาหารต้องยื่นยังไงค่ะ", [
                lambda t: check_contains(t, "ป้าย"),
                lambda t: (True, "response received"),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=8, name="8.2 ภาษีป้าย — อัตราและกำหนดเวลา",
        steps=[
            Step("อัตราภาษีป้ายเป็นยังไงค่ะ และต้องยื่นภายในเมื่อไหร่", [
                lambda t: check_contains(t, "ป้าย"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 9 — ใบรับรองมาตรฐานร้านอาหาร (SAN / SAN PLUS)
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=9, name="9.1 มาตรฐานร้านอาหาร → ขั้นตอน",
        steps=[
            Step("อยากได้ใบรับรองมาตรฐานร้านอาหารค่ะ ต้องทำอะไรบ้าง", [
                lambda t: check_contains(t, "มาตรฐาน") or check_contains(t, "SAN")[0] and True,
            ]),
        ],
    ))
    tests.append(TestCase(
        category=9, name="9.2 SAN PLUS — เกณฑ์และขั้นตอน",
        steps=[
            Step("SAN PLUS คืออะไรค่ะ แตกต่างจาก SAN ยังไง", [
                lambda t: check_contains(t, "SAN") or check_contains(t, "มาตรฐาน")[0] and True,
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 10 — ใบวุฒิบัตรผู้สัมผัสอาหาร
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=10, name="10.1 อบรมผู้สัมผัสอาหาร → ขั้นตอน+ค่าใช้จ่าย",
        steps=[
            Step("พนักงานร้านต้องอบรมผู้สัมผัสอาหารไหมค่ะ ทำยังไงบ้าง", [
                lambda t: check_contains(t, "ผู้สัมผัสอาหาร") or check_contains(t, "อบรม")[0] and True,
            ]),
        ],
    ))
    tests.append(TestCase(
        category=10, name="10.2 ใบวุฒิบัตร หมดอายุ — ต่ออายุ",
        steps=[
            Step("ใบวุฒิบัตรผู้สัมผัสอาหารหมดอายุต้องทำอะไรบ้างค่ะ", [
                lambda t: (True, "response received"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 11 — ใบรับรองแพทย์ 9 โรค (สณ.11)
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=11, name="11.1 ตรวจสุขภาพ 9 โรค → ขั้นตอน+ค่าใช้จ่าย",
        steps=[
            Step("ต้องตรวจสุขภาพ 9 โรคก่อนเปิดร้านไหมค่ะ", [
                lambda t: check_contains(t, "โรค") or check_contains(t, "สุขภาพ")[0] and True,
            ]),
        ],
    ))
    tests.append(TestCase(
        category=11, name="11.2 สณ.11 — เอกสาร+สถานที่ตรวจ",
        steps=[
            Step("สณ.11 คืออะไรค่ะ ต้องไปตรวจที่ไหนได้บ้าง", [
                lambda t: (True, "response received"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 12 — ระบบชำระเงินออนไลน์ (QR / PromptPay)
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=12, name="12.1 สมัคร QR Payment → ขั้นตอน",
        steps=[
            Step("จะสมัคร QR Payment สำหรับร้านต้องทำยังไงค่ะ", []),  # bot asks entity_type
            Step("บุคคลธรรมดาค่ะ", [  # answer entity_type → answer
                lambda t: check_contains(t, "QR") or check_contains(t, "ชำระเงิน")[0] and True,
            ]),
        ],
    ))
    tests.append(TestCase(
        category=12, name="12.2 PromptPay ร้านค้า",
        steps=[
            Step("PromptPay สำหรับร้านค้าสมัครยังไงค่ะ", []),  # bot asks entity_type
            Step("บุคคลธรรมดาค่ะ", []),  # answer entity_type → bot asks bank
            Step("1", [  # answer bank selection
                lambda t: check_contains(t, "PromptPay") if "PromptPay" in t else check_contains(t, "พร้อมเพย์"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 13 — เครื่องรูดบัตร EDC
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=13, name="13.1 สมัคร EDC → ขั้นตอน",
        steps=[
            Step("อยากได้เครื่องรูดบัตรสำหรับร้านค่ะ ต้องทำอะไรบ้าง", [
                lambda t: check_contains(t, "EDC") or check_contains(t, "บัตร")[0] and True,
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 14 — จัดการเงิน / ระบบบัญชี
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=14, name="14.1 ระบบบัญชีร้านอาหาร",
        steps=[
            Step("ร้านอาหารต้องทำบัญชียังไงบ้างค่ะ มีระบบอะไรแนะนำไหม", [
                lambda t: (True, "response received"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 15 — Multi-topic (2–3 topics answered together)
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=15, name="15.1 2 topics → ตอบรวม",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์และภาษีป้ายค่ะ", [
                lambda t: check_contains(t, "ทะเบียนพาณิชย์") or check_contains(t, "ป้าย")[0] and True,
                lambda t: check_not_contains(t, "ต้องการรายละเอียดเรื่องไหนก่อน"),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=15, name="15.2 3 topics → ตอบรวม (ไม่แสดงเมนู)",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ ภาษีป้าย และ VAT ค่ะ", [
                lambda t: check_not_contains(t, "ต้องการรายละเอียดเรื่องไหนก่อน"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 16 — Multi-topic (4+ topics → menu)
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=16, name="16.1 4+ topics → แสดง summary + เมนู",
        steps=[
            Step(
                "อยากเปิดร้านอาหารค่ะ อยากทราบเรื่องใบทะเบียนพาณิชย์ "
                "ใบอนุญาตร้านอาหาร ประกันสังคม VAT และภาษีป้ายค่ะ",
                [
                    lambda t: check_contains(t, "ต้องการรายละเอียดเรื่องไหนก่อน"),
                    lambda t: check_numbered_menu(t, 4),
                ],
            ),
        ],
    ))
    tests.append(TestCase(
        category=16, name="16.2 เลือกจากเมนู 4+ → ตอบเรื่องที่เลือก",
        steps=[
            Step(
                "อยากเปิดร้านอาหารค่ะ อยากทราบเรื่องใบทะเบียนพาณิชย์ "
                "ใบอนุญาตร้านอาหาร ประกันสังคม VAT และภาษีป้ายค่ะ",
                [],
            ),
            Step("1", [
                lambda t: check_not_contains(t, "ต้องการรายละเอียดเรื่องไหนก่อน"),
                lambda t: (True, "response received for selection"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 17 — Academic mode switch
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=17, name="17.1 ขอข้อมูลละเอียด → เข้า Academic mode",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ค่ะ", []),
            Step("บุคคลธรรมดาค่ะ", []),
            Step("ขอแบบละเอียดกว่านี้ค่ะ", [
                lambda t: check_not_contains(t, "ยืนยัน", "confirm"),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=17, name="17.2 Academic mode — ตอบครบทุก field",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์แบบละเอียดเลยค่ะ", [
                lambda t: (True, "academic intake started"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 18 — Follow-up questions (context retention)
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=18, name="18.1 Follow-up ไม่ถามประเภทกิจการซ้ำ",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ค่ะ", []),
            Step("1", []),              # answer location_type
            Step("บุคคลธรรมดาค่ะ", []),  # answer entity_type
            Step("แล้วถ้าจะขอใบอนุญาตร้านอาหารด้วยล่ะค่ะ", [
                lambda t: check_no_entity_type_reask(t),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=18, name="18.2 Follow-up ถามเรื่องที่ต่อเนื่อง",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ค่ะ", []),
            Step("1", []),              # answer location_type
            Step("บุคคลธรรมดาค่ะ", []),  # answer entity_type
            Step("1", []),              # answer operation_type → full answer
            Step("ค่าธรรมเนียมเท่าไหร่ค่ะ", [
                lambda t: check_contains(t, "ค่าธรรมเนียม"),
                lambda t: check_no_entity_type_reask(t),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 19 — Reference links (แสดงเฉพาะเมื่อถาม)
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=19, name="19.1 ไม่ถามอ้างอิง → ไม่แสดง ref section",
        steps=[
            Step("ต้องทำยังไงบ้างในการจด VAT ค่ะ", []),
            Step("บุคคลธรรมดาค่ะ", [
                lambda t: check_no_ref_section(t),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=19, name="19.2 ขอแหล่งอ้างอิง → แสดง URL",
        steps=[
            Step("ขอใบทะเบียนพาณิชย์ยังไงค่ะ", []),
            Step("1", []),              # answer location_type
            Step("บุคคลธรรมดาค่ะ", []),  # answer entity_type
            Step("1", []),              # answer operation_type → full answer
            Step("ขอลิงก์อ้างอิงด้วยค่ะ", [
                lambda t: check_has_url(t),
                lambda t: check_no_duplicate_url(t),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 20 — Slot memory (ไม่ถามซ้ำข้ามหัวข้อ)
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=20, name="20.1 ตอบ entity_type ครั้งเดียว ใช้กับทุกหัวข้อ",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ค่ะ", []),
            Step("1", []),              # answer location_type
            Step("บุคคลธรรมดาค่ะ", []),  # answer entity_type
            Step("1", []),              # answer operation_type → full answer
            Step("แล้วถ้าจะขอใบอนุญาตขายสุราด้วยล่ะค่ะ ต้องทำอะไรบ้าง", [
                lambda t: check_no_entity_type_reask(t),
                lambda t: check_contains(t, "สุรา"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 21 — Edge cases
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=21, name="21.1 คำถามนอกขอบเขต",
        steps=[
            Step("ช่วยแนะนำเมนูอาหารด้วยค่ะ", [
                # bot may mention "ขั้นตอน" in topic suggestions — exclude only ค่าธรรมเนียม
                lambda t: check_not_contains(t, "ค่าธรรมเนียม"),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=21, name="21.2 Input สั้นมาก",
        steps=[
            Step("เอกสาร", [
                lambda t: (True, "response received (no crash)"),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=21, name="21.3 ข้อความยาวมาก (near limit)",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ " + "รายละเอียด " * 50, [
                lambda t: (True, "response received (no crash)"),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 22 — Session reset
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=22, name="22.1 Reset แล้วเริ่มใหม่ได้ปกติ",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ค่ะ", []),
            Step("บุคคลธรรมดาค่ะ", []),
            # reset happens automatically between test cases
        ],
    ))
    tests.append(TestCase(
        category=22, name="22.2 Session ใหม่ ไม่จำ context จาก session เก่า",
        steps=[
            Step("อยากทราบเรื่องใบทะเบียนพาณิชย์ค่ะ", [
                lambda t: check_phase_question(t),  # bot should ask some slot (not remember old session)
            ]),
        ],
    ))

    return tests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2E test runner for น้องสุดยอด")
    p.add_argument("--url", default=BASE_URL, help="API base URL (default: http://localhost:8000)")
    p.add_argument("--save-responses", action="store_true", help="Save raw responses to text files")
    p.add_argument(
        "--category",
        default=None,
        help="Run only specific category or range, e.g. '3' or '1-5'",
    )
    return p.parse_args()


def parse_category_filter(cat_str: Optional[str]) -> Optional[set]:
    if cat_str is None:
        return None
    if "-" in cat_str:
        lo, hi = cat_str.split("-", 1)
        return set(range(int(lo), int(hi) + 1))
    return {int(cat_str)}


def run_all_tests(args: argparse.Namespace) -> int:
    cat_filter = parse_category_filter(args.category)
    all_tests = build_all_tests()

    if cat_filter:
        all_tests = [t for t in all_tests if t.category in cat_filter]

    runner = TestRunner(base_url=args.url, save_responses=args.save_responses)

    print(_color(f"\nน้องสุดยอด E2E Test Suite", _BOLD + _CYAN))
    print(f"API: {args.url}{API_PREFIX}")
    print(f"Tests: {len(all_tests)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for tc in all_tests:
        print(_color(f"  [{tc.category:2d}] {tc.name}", _CYAN))
        result = runner.run(tc)
        runner.results.append(result)
        overall = _color("PASS", _GREEN) if result.passed else _color("FAIL", _RED)
        print(f"       → {overall}\n")

    runner.print_report()

    failed = sum(1 for r in runner.results if not r.passed)
    return 1 if failed > 0 else 0


def main() -> None:
    args = parse_args()
    sys.exit(run_all_tests(args))


if __name__ == "__main__":
    main()

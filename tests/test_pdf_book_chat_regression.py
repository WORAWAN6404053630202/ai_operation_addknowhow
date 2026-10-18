"""
Chat-answer regression suite for PDF book content (feature/pdf-ingestion).

Companion to test_bot_e2e.py (same TestCase/Step/check-function pattern,
same standalone-runner design — hits a REAL running bot, costs real LLM
money per run, NOT auto-collected by plain `pytest`). Where test_bot_e2e.py
covers the general regulatory-content flows, this file covers PDF
book/know-how content specifically — the layer where a chain of 9 real bugs
was found and fixed in one investigation (2026-09, see the
pdf_chunking_truncation_and_retrieval_fixes memory for full history):
page_content truncation, a doc_id RRF-fusion collision, a generic sub_topic
label hijacking chapter-fetch, a MULTI-LICENSE hint firing on retrieval
noise, a FEE/DURATION instruction with too-narrow scanning scope, THREE
independent duplicate/missed answer_guideline caps across
persona_practical.py / persona_supervisor.py / persona_academic.py, and a
multi-chunk-topic aggregation bug that mislabeled sequential chunks as
"different types". None of these bugs threw an exception or returned an
HTTP error — every one of them silently produced a wrong or incomplete
answer. A generic "no crash" E2E check would not have caught ANY of them.
That's what this suite checks instead: specific FACTS that must appear
(proving full content reached the LLM, not just a truncated prefix) and
specific FAILURE SIGNATURES that must NOT appear (the literal info-gap
marker, wrong-topic contamination from a retrieval-noise regression, etc).

Usage:
    python tests/test_pdf_book_chat_regression.py
    python tests/test_pdf_book_chat_regression.py --url http://43.210.64.119:3000
    python tests/test_pdf_book_chat_regression.py --category 1

HOW TO ADD A NEW PDF BOOK'S TEST CASES (do this every time a new book is
ingested, before trusting it in production):
    1. Ask the bot 5-10 real questions covering: (a) a short chapter/topic,
       (b) the LONGEST chapter/topic in the book (the truncation-guard
       case), (c) anything with numbers/tables/formulas, (d) at least one
       question using a generic/common word that might collide with other
       books' or the regulatory Sheet's topic labels (the hijack-guard
       case — reuse this session's exact pattern: a word like "ภาษี" or
       "ค่าธรรมเนียม" that appears in many unrelated places).
    2. For each, manually verify the CORRECT answer by reading the source
       PDF directly (never assume the bot's first answer is right — that's
       exactly the assumption that let all 9 bugs above ship silently).
    3. Add a new TestCase below with category = next integer, checks
       against the specific facts you verified (not just "did it answer").
    4. Run this file against the live instance BEFORE the new book goes
       into real production use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_bot_e2e import (  # noqa: E402 — reuse the exact same infra, no drift
    CheckResult,
    Step,
    TestCase,
    TestRunner,
    check_contains,
    check_not_contains,
)

_GREEN = "\033[92m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def check_min_length(text: str, min_chars: int, label: str = "") -> CheckResult:
    """Structural proxy for 'wasn't cut off to a short stub' — a weak but
    stable signal, useful when the exact wording of a broad/open question's
    answer varies too much between calls to anchor on specific vocabulary
    (LLMs reorganize/reword open "list the factors" answers noticeably
    between calls, confirmed empirically — an earlier version of this file
    anchored on specific words like "แผนที่"/"ประชากร" and flaked when a
    real, complete, correct answer simply organized itself differently)."""
    if len(text) >= min_chars:
        return True, label or f"response length {len(text)} >= {min_chars}"
    return False, f"response only {len(text)} chars, expected >= {min_chars}"


def check_no_info_gap(text: str, label: str = "no info-gap marker") -> CheckResult:
    """The literal display text substituted for the LLM's [[INFO_GAP]] marker
    (utils/text_patterns.py's INFO_GAP_DISPLAY_TEXT). A question this suite
    asks is one already verified answerable from the real PDF content — if
    this string appears, real content silently failed to reach/convince the
    LLM, regardless of which of the 9 known bug classes is responsible or
    whether it's a brand new one."""
    if "ยังไม่พบข้อมูลในเอกสาร" in text or "ไม่มี context" in text:
        return False, "response fell back to an info-gap / no-context answer"
    return True, label


def build_all_tests() -> list[TestCase]:
    tests: list[TestCase] = []

    # ------------------------------------------------------------------
    # หมวด 1 — เจ๊งในกระดาษ.pdf: การวิเคราะห์ด้านทำเล (Location Analysis)
    # 19,314 chars — the single longest real chapter measured. Old bug #1
    # truncated page_content at ~1,800-2,500 chars; "เซ้งร้าน" only appears
    # in the back half of this chapter, well past that old cutoff.
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=1, name="1.1 ปัจจัยเลือกทำเล — เนื้อหาส่วนท้ายบทต้องรอด (truncation guard)",
        steps=[
            Step("ปัจจัยสำคัญในการเลือกทำเลร้านมีอะไรบ้างคะ", [
                lambda t: check_contains(t, "ทำเล"),
                # Open "list the factors" questions get genuinely reworded/
                # reorganized between calls (confirmed empirically — see
                # check_min_length's docstring) — anchor on length + at least
                # 2 of several stable recurring themes instead of exact words.
                lambda t: check_min_length(t, 800, "substantial answer, not a truncated stub"),
                lambda t: (
                    sum(1 for kw in ("ค่าเช่า", "คู่แข่ง", "ที่จอดรถ", "สัญญาเช่า", "ลูกค้า") if kw in t) >= 2,
                    "covers >=2 of several recurring factor themes",
                ),
                lambda t: check_no_info_gap(t),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 2 — เจ๊งในกระดาษ.pdf: การวิเคราะห์รายได้และกำไร
    # 3,867 chars. Was hit by bug #2 (doc_id collision — merged with 2 other
    # case-study chapters), #3 (sub_topic label "ภาษีเงินได้" hijacked to a
    # DIFFERENT book), #4 (MULTI-LICENSE hint), #5 (fee-info instruction
    # scanning only structured fields), #6 (duplicate cap dict in
    # persona_supervisor.py). The single hardest-fought fix in this chain.
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=2, name="2.1 Top Line/Bottom Line — doc_id collision guard",
        steps=[
            Step("ในงบกำไรขาดทุน Top Line กับ Bottom Line ต่างกันยังไง", [
                lambda t: check_contains(t, "Top Line"),
                lambda t: check_contains(t, "Bottom Line"),
                lambda t: check_no_info_gap(t),
            ]),
        ],
    ))
    tests.append(TestCase(
        category=2, name="2.2 ภาษีเงินได้ SME — sub_topic hijack + MULTI-LICENSE + fee-info guard (hardest case)",
        steps=[
            Step(
                "ธุรกิจ SME (ทุนไม่เกิน 5 ล้านบาท รายได้ไม่เกิน 30 ล้านบาทต่อปี) "
                "เสียภาษีเงินได้นิติบุคคลอัตราเท่าไหร่",
                [
                    lambda t: check_contains(t, "300,000") if "300,000" in t else check_contains(t, "300000"),
                    lambda t: check_contains(t, "15%"),
                    lambda t: check_contains(t, "20%"),
                    lambda t: check_no_info_gap(t),
                    # Regression guard: must NOT get hijacked into an unrelated
                    # regulatory chapter's content (the exact bug #3/#4 shape).
                    lambda t: check_not_contains(t, "ภาษีมูลค่าเพิ่ม", "ใบอนุญาตจำหน่ายสุรา", "ภาษีป้าย"),
                ],
            ),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 3 — เจ๊งในกระดาษ.pdf: Payback Period chapters
    # 4,000+ chars each — formula sits past the old ~1,800-2,500 char cutoff.
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=3, name="3.1 สูตรคำนวณระยะเวลาคืนทุน — truncation guard (formula past old cutoff)",
        steps=[
            Step("สูตรคำนวณระยะเวลาคืนทุนคืออะไร", [
                lambda t: check_contains(t, "เงินลงทุน"),
                lambda t: check_contains(t, "กำไร"),
                lambda t: check_no_info_gap(t),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 4 — เจ๊งในกระดาษ.pdf: พื้นที่ฐานลูกค้า (Customer Catchment)
    # Was a TOTAL non-answer ("ไม่มี context เกี่ยวกับร้านคุณ") before bug #1's
    # fix — the model asked for user-specific data instead of answering from
    # the document, because the real content was truncated away entirely.
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=4, name="4.1 พื้นที่ฐานลูกค้า ประเภท+สัดส่วนรายได้ — was a total non-answer before fix #1",
        steps=[
            Step(
                "พื้นที่ฐานลูกค้าแบ่งเป็นกี่ประเภท อะไรบ้าง "
                "และแต่ละประเภทสร้างรายได้กี่เปอร์เซ็นต์หรอคะ",
                [
                    lambda t: check_not_contains(t, "ไม่มี context", "ไม่ทราบร้านของคุณ"),
                    lambda t: check_no_info_gap(t),
                ],
            ),
        ],
    ))
    tests.append(TestCase(
        category=4, name="4.2 วิธีหาพื้นที่ฐานลูกค้า — baseline, unaffected by any fix (short chapter)",
        steps=[
            Step("การหาพื้นที่ฐานลูกค้าทำได้กี่วิธี อะไรบ้าง", [
                lambda t: check_no_info_gap(t),
            ]),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 5 — เจ๊งในกระดาษ.pdf: ค่าเช่า + ร้านสะดวกซัก (deep-content specificity)
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=5, name="5.1 ค่าเช่า+ร้านสะดวกซัก — specific numbers from deep in a long chapter",
        steps=[
            Step(
                "ค่าเช่าร้านอาหารทั่วไปไม่ควรเกินกี่เปอร์เซ็นต์ของรายได้ต่อเดือน "
                "แล้วร้านสะดวกซักต่างจากนี้ยังไง",
                [
                    lambda t: check_contains(t, "20%"),
                    lambda t: check_no_info_gap(t),
                ],
            ),
        ],
    ))

    # ------------------------------------------------------------------
    # หมวด 6 — Academic mode parity
    # Was KNOWN FAILING as of 2026-09-16. Took THREE independent bug fixes,
    # each uncovering the next layer, to fully resolve — live-verified
    # passing on the ai-restbiz-pdf instance the same night:
    #   1. model/persona_academic.py's _filter_docs_by_selected_topic — the
    #      sub_topic Chroma get() fallback was gated on `if not fresh_docs:`,
    #      a condition that can never be true (see
    #      tests/test_academic_subtopic_targeted_retrieval.py).
    #   2. conf.RETRIEVAL_MIN_SIMILARITY (0.6) silently dropped genuinely-
    #      correct PDF-book narrative content (measured 0.487-0.577 across 5
    #      confirmed-correct queries) before it could even become a
    #      topic-menu candidate — fixed with a separate, evidence-based
    #      RETRIEVAL_MIN_SIMILARITY_PDF_BOOK=0.45 scoped to
    #      ingested_via="pdf_auto" (see
    #      tests/test_pdf_book_similarity_floor.py).
    #   3. persona_academic.py's _detect_distinct_topics extracted query
    #      "keywords" via plain whitespace splitting, which does nothing
    #      useful for Thai (no spaces between words) — the real keyword
    #      "กำไร" was trapped inside an unsplittable blob while the bare
    #      conjunction "กับ" survived as a "keyword" and coincidentally won.
    #      Fixed with real pythainlp segmentation + stopword filtering (see
    #      utils.text_patterns.extract_thai_keywords and
    #      tests/test_thai_keyword_extraction.py).
    # ------------------------------------------------------------------
    tests.append(TestCase(
        category=6, name="6.1 Top Line/Bottom Line ใน Academic mode",
        persona_id="academic",
        steps=[
            Step("ในงบกำไรขาดทุน Top Line กับ Bottom Line ต่างกันยังไง", [
                lambda t: check_no_info_gap(t),
                lambda t: check_not_contains(t, "ผังโรงงาน"),
            ]),
        ],
    ))

    return tests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PDF book content regression suite")
    p.add_argument("--url", default="http://localhost:3000", help="API base URL")
    p.add_argument("--category", default=None, help="Run only category N, e.g. '2'")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    all_tests = build_all_tests()
    if args.category:
        all_tests = [t for t in all_tests if t.category == int(args.category)]

    runner = TestRunner(base_url=args.url)
    print(f"{_BOLD}{_CYAN}PDF Book Content Regression Suite{_RESET}")
    print(f"API: {args.url}/api/v1  |  tests: {len(all_tests)} (category 6 uses persona_id=academic, others practical)\n")

    for tc in all_tests:
        print(f"{_CYAN}  [{tc.category}] {tc.name}{_RESET}")
        result = runner.run(tc)
        runner.results.append(result)
        overall = f"{_GREEN}PASS{_RESET}" if result.passed else f"{_RED}FAIL{_RESET}"
        print(f"       -> {overall}\n")

    runner.print_report()
    failed = sum(1 for r in runner.results if not r.passed)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

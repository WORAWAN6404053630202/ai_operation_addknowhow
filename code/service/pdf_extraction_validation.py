# code/service/pdf_extraction_validation.py
"""Automatic accuracy validation for PDF-extraction output (feature/pdf-ingestion),
built from findings in the Typhoon OCR vs Claude vision vs ground-truth comparison:

1. Arithmetic consistency — a fee table's stated total must equal the sum of its
   line items. Caught a real error: Typhoon read "รวมค่าธรรมเนียม 3,000 บาท" when the
   line items (660+1,200+1,200) actually sum to 3,060 — confirmed against the source PDF.
   Document-type-agnostic: this is a math property, not tied to any specific form/agency.
2. Dual-model disagreement — running two independently-failing tools (different
   error patterns) and flagging where they disagree on money/dates/codes/license
   numbers catches real errors: license numbers, totals, and form-code prefixes
   (e.g. "ภส." misread as "กส."/"กล."/"สล." — never the same wrong answer twice)
   all diverged between Typhoon and Claude at exactly the spots later confirmed
   wrong or risky. Also document-type-agnostic: it doesn't need to know in advance
   which prefix/format is "correct", only that the two extractors disagree.
3. LLM-judged disagreement (compare_extractions_llm) — the general catch-all.
   Checks 1 and 2 above only catch the SHAPES of error already seen in this one
   test document (baht amounts, dates, form-code-looking tokens, license-number-
   looking tokens). A future document with a different kind of important field —
   an ID number, a percentage, a person's name — would slip past both silently.
   This check asks an LLM to read both extractors' full output for the page and
   report ANY factual disagreement in its own words, with no pre-defined pattern
   to match against. Costs a real API call (money + latency) per page, so it's
   opt-in via validate_extraction(use_llm_comparison=True), not always-on.

Deliberately NOT included: a hardcoded "form code must start with ภส." check.
Tried it, then dropped it — it would false-positive on any future document from a
different agency/form family, and the dual-model check above already catches the
same failure mode (the two extractors never agreed on the wrong prefix either)
without needing to know the "correct" prefix in advance. General rule for this
module: prefer checks that need no per-document-type configuration.

None of this replaces human review — it only decides which items get a loud
warning attached before they reach the review queue. Logprobs-based confidence
was investigated and ruled out: neither Claude (via OpenRouter) nor Typhoon
return logprobs when requested (empirically confirmed, both return None)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

import conf
from utils.llm_cost_logging import log_llm_cost
from utils.logger import get_logger

logger = get_logger(__name__)

# Matches Thai-consonant-cluster form codes like "ภส.08-05", "กส.08-05", "สล.08-22"
# — used only for the dual-model token comparison below (does NOT assert which
# prefix is "correct" — see module docstring for why that check was dropped).
_FORM_CODE_RE = re.compile(r"[ก-ฮ]{1,3}\.?\s?\d{2}-\d{2}")

# Matches Thai-baht amounts: "660 บาท", "3,060 บาท", "1,200.00 บาท"
_BAHT_AMOUNT_RE = re.compile(r"[\d,]+(?:\.\d{1,2})?\s*บาท")

# Matches dd/mm/yyyy (Thai Buddhist-era) dates, e.g. "14/06/2567"
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")

# Matches license/receipt-style numbers seen in this doc family:
# "67L0006170", "67-L-0006179", "2567-00341243"
_LICENSE_NUMBER_RE = re.compile(r"\b\d{2,4}-?[A-Zก-ฮ]?-?\d{6,8}\b")

_TOTAL_ROW_KEYWORDS = ("รวม",)


@dataclass
class ValidationFlag:
    category: str  # "arithmetic" | "form_code" | "dual_model_disagreement"
    severity: str  # "high" | "medium"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _clean_number(raw: str) -> float:
    return float(raw.replace(",", "").replace("บาท", "").strip())


class _HTMLTableParser(HTMLParser):
    """Minimal parser for Typhoon's <table><tr><td>...</td></tr></table> output."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table":
            self._current_table = []
        elif tag == "tr" and self._current_table is not None:
            self._current_row = []
        elif tag in ("td", "th") and self._current_row is not None:
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._current_table is not None:
            self.tables.append(self._current_table)
            self._current_table = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_table is not None:
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag in ("td", "th") and self._current_cell is not None:
            if self._current_row is not None:
                self._current_row.append("".join(self._current_cell).strip())
            self._current_cell = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)


def _extract_html_tables(markdown: str) -> list[list[list[str]]]:
    parser = _HTMLTableParser()
    parser.feed(markdown)
    return parser.tables


def _extract_markdown_pipe_tables(markdown: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue  # header separator row, e.g. |---|---|
            current.append(cells)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)
    return tables


def check_table_arithmetic(markdown: str) -> list[ValidationFlag]:
    """Flags any table where a "รวม..." (total) row's amount doesn't equal the
    sum of the other baht-amount rows in the same table."""
    flags: list[ValidationFlag] = []
    tables = _extract_html_tables(markdown) + _extract_markdown_pipe_tables(markdown)

    for table in tables:
        line_item_amounts: list[float] = []
        total_amount: float | None = None
        total_row_text = ""

        for row in table:
            row_text = " ".join(row)
            amounts = _BAHT_AMOUNT_RE.findall(row_text)
            if not amounts:
                continue
            amount = _clean_number(amounts[-1])
            is_total_row = any(kw in row_text for kw in _TOTAL_ROW_KEYWORDS)
            if is_total_row:
                total_amount = amount
                total_row_text = row_text
            else:
                line_item_amounts.append(amount)

        if total_amount is not None and line_item_amounts:
            computed = round(sum(line_item_amounts), 2)
            if abs(computed - total_amount) > 0.01:
                flags.append(
                    ValidationFlag(
                        category="arithmetic",
                        severity="high",
                        message=(
                            f"Table total ({total_amount}) does not match sum of line items "
                            f"({computed}) — off by {round(total_amount - computed, 2)}"
                        ),
                        details={
                            "stated_total": total_amount,
                            "computed_total": computed,
                            "line_items": line_item_amounts,
                            "total_row_text": total_row_text,
                        },
                    )
                )
    return flags


def _normalize_date(raw: str) -> str:
    """Zero-pads day/month so "4/6/2567" and "04/06/2567" compare equal — same
    real date, just written differently. Only reformats, never reinterprets the
    numbers (no Buddhist/Gregorian-year guessing), so it can't turn a genuinely
    different date into a false match."""
    day, month, year = raw.split("/")
    return f"{int(day):02d}/{int(month):02d}/{year}"


def _extract_salient_tokens(markdown: str) -> dict[str, set[str]]:
    return {
        "baht_amounts": {_BAHT_AMOUNT_RE.sub(lambda m: str(_clean_number(m.group())), a) for a in _BAHT_AMOUNT_RE.findall(markdown)},
        "dates": {_normalize_date(d) for d in _DATE_RE.findall(markdown)},
        "form_codes": {m.group().replace(" ", "") for m in _FORM_CODE_RE.finditer(markdown)},
        "license_numbers": set(_LICENSE_NUMBER_RE.findall(markdown)),
    }


def compare_extractions(markdown_a: str, markdown_b: str, label_a: str = "model_a", label_b: str = "model_b") -> list[ValidationFlag]:
    """Flags salient facts (money, dates, form codes, license numbers) present in
    one extractor's output but not the other's — a disagreement worth a human
    look, regardless of which one (if either) is actually correct."""
    flags: list[ValidationFlag] = []
    tokens_a = _extract_salient_tokens(markdown_a)
    tokens_b = _extract_salient_tokens(markdown_b)

    for kind in tokens_a:
        only_in_a = tokens_a[kind] - tokens_b[kind]
        only_in_b = tokens_b[kind] - tokens_a[kind]
        if only_in_a or only_in_b:
            flags.append(
                ValidationFlag(
                    category="dual_model_disagreement",
                    severity="medium",
                    message=f"{kind}: {label_a} and {label_b} disagree",
                    details={f"only_in_{label_a}": sorted(only_in_a), f"only_in_{label_b}": sorted(only_in_b)},
                )
            )
    return flags


_LLM_COMPARISON_PROMPT = """ต่อไปนี้คือผลลัพธ์การสกัดข้อความจากเอกสารหน้าเดียวกัน โดยเครื่องมือ 2 ตัวที่ทำงานอิสระจากกัน (ตัว A และตัว B)

อ่านทั้งสองฉบับ แล้วหาจุดที่ทั้งสองฉบับ "ขัดแย้งกันในข้อเท็จจริง" (ตัวเลข, ชื่อ, วันที่, รหัส, หรือข้อความสำคัญอื่นๆ ที่ควรจะตรงกัน แต่กลับไม่ตรงกัน)

ไม่ต้องสนใจความแตกต่างที่ไม่สำคัญ เช่น การจัดรูปแบบ/เว้นวรรค/โครงสร้าง markdown ที่ต่างกัน — สนใจแค่ "ข้อเท็จจริงที่ขัดแย้งกันจริงๆ" เท่านั้น

ตอบกลับเป็น JSON array เท่านั้น ไม่ต้องมีข้อความอื่นนอกเหนือจาก JSON โดยแต่ละรายการมีโครงสร้าง:
{{"fact": "คำอธิบายสั้นๆ ว่าเรื่องอะไร", "value_a": "ค่าที่ฉบับ A บอก", "value_b": "ค่าที่ฉบับ B บอก", "severity": "high หรือ medium"}}

ใช้ severity "high" สำหรับตัวเลขทางการเงิน/รหัสสำคัญ/เลขที่เอกสาร และ "medium" สำหรับจุดอื่นๆ

ถ้าไม่พบข้อขัดแย้งเลย ให้ตอบ [] (array ว่าง)

=== ฉบับ A ===
{markdown_a}

=== ฉบับ B ===
{markdown_b}
"""


def compare_extractions_llm(
    markdown_a: str, markdown_b: str, label_a: str = "model_a", label_b: str = "model_b"
) -> list[ValidationFlag]:
    """General-purpose catch-all: asks an LLM to read both full extractions and
    report any factual disagreement in its own words — no pre-defined pattern to
    match against, unlike check_table_arithmetic/compare_extractions above. Exists
    specifically to catch error shapes we haven't seen yet in a new document type.
    Costs a real API call — see validate_extraction(use_llm_comparison=...)."""
    from openai import OpenAI

    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
    prompt = _LLM_COMPARISON_PROMPT.format(markdown_a=markdown_a, markdown_b=markdown_b)

    try:
        _call_start = time.monotonic()
        resp = client.chat.completions.create(
            model=conf.OPENROUTER_MODEL_PRACTICAL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        log_llm_cost(logger, "ExtractionValidation/LLMComparison", conf.OPENROUTER_MODEL_PRACTICAL, resp, time.monotonic() - _call_start)
        raw = (resp.choices[0].message.content or "[]").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        disagreements = json.loads(raw)
    except Exception as e:
        logger.error(f"[pdf_extraction_validation] LLM comparison call failed: {e}")
        return [
            ValidationFlag(
                category="llm_comparison_error",
                severity="medium",
                message=f"LLM comparison call failed ({type(e).__name__}) — falls back to regex-only checks for this page, flag it for review manually",
                details={"error": str(e)},
            )
        ]

    flags: list[ValidationFlag] = []
    for d in disagreements:
        if not isinstance(d, dict):
            continue
        flags.append(
            ValidationFlag(
                category="llm_disagreement",
                severity=d.get("severity", "medium") if d.get("severity") in ("high", "medium") else "medium",
                message=f"{d.get('fact', 'disagreement')}: {label_a}='{d.get('value_a')}' vs {label_b}='{d.get('value_b')}'",
                details={"label_a": label_a, "label_b": label_b, **d},
            )
        )
    return flags


def validate_extraction(
    markdown: str,
    compare_with: str | None = None,
    compare_label: str = "second_model",
    use_llm_comparison: bool = False,
) -> list[ValidationFlag]:
    """Entry point: runs every automatic check available. `compare_with` is an
    optional second extractor's output over the SAME page for the dual-model
    checks. `use_llm_comparison` additionally runs the general-purpose LLM catch-all
    (costs a real API call — off by default)."""
    flags = check_table_arithmetic(markdown)
    if compare_with is not None:
        flags += compare_extractions(markdown, compare_with, "primary", compare_label)
        if use_llm_comparison:
            flags += compare_extractions_llm(markdown, compare_with, "primary", compare_label)
    return flags

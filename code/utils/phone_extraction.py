# code/utils/phone_extraction.py (2026-10, core-M5/M6)
"""Shared shape-based Thai phone/hotline number extraction — used to build a
verbatim-from-DOCUMENTS whitelist for the phone-hallucination defense
(persona_practical.py's PHONE NUMBER GUARD, now also persona_academic.py's).

Extracted out of persona_practical.py, where it originally lived as a
classmethod, specifically so persona_academic.py can reuse the EXACT same
extraction logic instead of a second hand-copy — the whole reason Practical
keeps its own Layer 2 (prompt whitelist) and Layer 3 (post-processing strip)
on one shared implementation is "so the two layers can never again drift
apart" (see the original docstring below); duplicating this across files
would reintroduce precisely that risk one level up.

Live-reproduced bug (2026-09) this design fixes: the original version
required a specific keyword immediately before a number (โทร/สายด่วน/Tel/
Fax/...) to recognize it as a phone number — real data labels its numbers
with "Merchant Call Center" / "Customer Service 24 Hours" (English
department-name conventions never anticipated), so a genuinely real,
correctly-cited phone number never made it into the whitelist, and the LLM's
correct output was then stripped as "hallucinated". Patching in more
keywords is the same whack-a-mole that caused the bug — any FUTURE label
wording the team adds would break it again the same way. Durable fix: 9-11
digit numbers (landline/mobile, with or without +66, with any mix of
dashes/dots/spaces/parens) are matched by SHAPE alone, no keyword required —
the shape itself is specific enough that it essentially never collides with
fee amounts, dates, or reference numbers in this data (verified against the
full real corpus, zero false positives). Only the much shorter 4-digit
hotline codes (e.g. "1506") stay keyword-gated, since a bare 4-digit number
IS genuinely ambiguous with a baht amount ("ค่าธรรมเนียม 1500 บาท")."""

from __future__ import annotations

import re

_PHONE_CANDIDATE_RE = re.compile(r'\+?\d[\d\-\.\s]{6,15}\d')
_HOTLINE_WITH_CTX_RE = re.compile(
    r'(?:📞|โทร(?:ศัพท์|สาร)?|สายด่วน|Tel\.?|Fax\.?|แฟกซ์|Call Center|Customer Service|Hotline|ติดต่อ)'
    r'.{0,15}?(?<!\d)(1\d{3})(?!\d)',
    re.IGNORECASE,
)


def extract_thai_phone_numbers(text: str) -> set:
    """Returns numbers normalized to local form (leading "0", no
    separators) so "+66 2 123 4567" and "02-123-4567" compare equal.
    Shared by both the doc-side whitelist builder and the answer-side strip
    check in each persona — see module docstring."""
    text = (text or "").replace("(", " ").replace(")", " ")
    found: set = set()
    for m in _PHONE_CANDIDATE_RE.finditer(text):
        has_plus = m.group(0).strip().startswith("+")
        digits = re.sub(r"\D", "", m.group(0))
        if has_plus and digits.startswith("66") and len(digits) in (10, 11):
            found.add("0" + digits[2:])
        elif digits.startswith("0") and len(digits) in (9, 10):
            found.add(digits)
    for m in _HOTLINE_WITH_CTX_RE.finditer(text):
        found.add(m.group(1))
    return found

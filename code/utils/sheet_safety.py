# code/utils/sheet_safety.py
"""Shared spreadsheet-formula-injection guard (feature/pdf-ingestion, added
2026-08-25) — both sheet_write_back.py and knowhow_write_back.py write
cells via value_input_option="USER_ENTERED" (needed so genuinely-typed-
looking content behaves like a human typed it), but that mode also means
Google Sheets evaluates any value starting with =, +, -, or @ as a live
formula — a well-known CSV/spreadsheet injection class (OWASP-recognized).

Live-verified this is exploitable here: every field written to the Sheet
comes from LLM-drafted text extracted from an uploaded PDF — attacker-
influenceable content, same as the prompt-injection surface. Testing wrote
"+1+1" and it landed in the Sheet cell as "2" (executed as a formula), and
"=HYPERLINK(\"http://evil.example.com\",\"...\")" became a real clickable
link to an attacker-controlled URL — a live example of exactly the
phishing/data-exfiltration vector this vulnerability class is known for."""

from __future__ import annotations

_FORMULA_TRIGGER_CHARS = ("=", "+", "-", "@", "\t", "\r")


def neutralize_formula(value: str) -> str:
    """Prefixes with a single quote if the value would otherwise be
    interpreted as a formula — Google Sheets treats a leading apostrophe as
    "force literal text" and does not display the apostrophe itself, so
    this is safe even for the rare legitimate value that happens to start
    with one of these characters coincidentally."""
    if not value:
        return value
    if value[0] in _FORMULA_TRIGGER_CHARS:
        return "'" + value
    return value

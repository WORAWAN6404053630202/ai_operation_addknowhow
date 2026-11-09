"""Runner for the real-document gold-set fixtures under
tests/fixtures/pdf_gold_set/ (2026-11, PDF-extraction-pipeline overhaul —
see tests/fixtures/pdf_gold_set/README.md for the assertion schema and
/Users/w.worawan/.claude/plans/distributed-noodling-ladybug.md for the
overall plan this supports).

Distinct from tests/pdf_fixtures.py's synthetic builders: those check "does
the API behave the way a fix assumes" on documents built to exercise one
specific mechanism. This file checks "does a fix actually help on a real
page from the production corpus" — the thing a synthetic test structurally
cannot tell you, since a synthetic PDF never has unseen edge cases.

Starts with an empty assertions.json (see that file) — populated
incrementally as each phase of the pipeline overhaul needs real-document
validation, per the plan's Phase 0 cost-discipline note: adding a fixture
costs nothing (local pymupdf.open() inspection) unless it's specifically an
ocr_uncertainty_gating-gap fixture, which needs a real paid OCR call to
validate and is only added with explicit go-ahead at that point."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

_GOLD_SET_DIR = Path(__file__).parent / "fixtures" / "pdf_gold_set"
_ASSERTIONS_FILE = _GOLD_SET_DIR / "assertions.json"


def _load_assertions() -> list[dict]:
    if not _ASSERTIONS_FILE.exists():
        return []
    with open(_ASSERTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _ids(assertions: list[dict]) -> list[str]:
    return [f"{a.get('gap', '?')}:{a.get('doc_id', '?')}#p{a.get('page_num', '?')}:{a.get('assertion_type', '?')}" for a in assertions]


_ASSERTIONS = _load_assertions()


@pytest.mark.unit
class TestGoldSetSchemaIsValid:
    """Even with zero real assertions yet, the harness itself and the
    scaffold must be structurally sound — this is what would fail loudly if
    someone adds a malformed assertion record."""

    def test_assertions_file_is_valid_json_list(self):
        assert isinstance(_ASSERTIONS, list)

    def test_every_assertion_has_required_keys(self):
        required = {"doc_id", "page_num", "fixture_file", "gap", "assertion_type", "expected"}
        for a in _ASSERTIONS:
            missing = required - set(a.keys())
            assert not missing, f"assertion missing keys {missing}: {a}"

    def test_every_assertion_type_is_recognized(self):
        known = {"text_contains", "text_not_contains", "table_cell_equals", "widget_value_equals", "no_regression_vs_baseline"}
        for a in _ASSERTIONS:
            assert a["assertion_type"] in known, f"unrecognized assertion_type: {a['assertion_type']!r}"

    def test_every_fixture_file_referenced_actually_exists(self):
        for a in _ASSERTIONS:
            fixture_path = _GOLD_SET_DIR / a["fixture_file"]
            assert fixture_path.exists(), f"missing fixture file: {fixture_path}"

    def test_table_cell_equals_assertions_have_row_and_col(self):
        for a in _ASSERTIONS:
            if a["assertion_type"] == "table_cell_equals":
                assert a.get("row") is not None and a.get("col") is not None, f"table_cell_equals missing row/col: {a}"


def _run_assertion(assertion: dict) -> None:
    import pymupdf

    from service.pdf_large_extraction import _extract_one_page
    from utils.pdf_text_layer import extract_page_native_text

    fixture_path = _GOLD_SET_DIR / assertion["fixture_file"]
    pdf_bytes = fixture_path.read_bytes()
    page_num = assertion["page_num"]
    atype = assertion["assertion_type"]

    if atype in ("text_contains", "text_not_contains", "no_regression_vs_baseline"):
        native = extract_page_native_text(pdf_bytes, page_num)
        if native is None:
            result = _extract_one_page(pdf_bytes, page_num)
            text = result["typhoon_markdown"]
        else:
            text = native
        if atype == "text_contains":
            assert assertion["expected"] in text
        elif atype == "text_not_contains":
            assert assertion["expected"] not in text
        else:
            assert len(text) >= int(assertion["expected"])

    elif atype == "widget_value_equals":
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            page = doc[page_num - 1]
            values = {w.field_name: w.field_value for w in page.widgets()}
        finally:
            doc.close()
        assert values.get(assertion.get("notes", {}).get("field_name") if isinstance(assertion.get("notes"), dict) else None) == assertion["expected"] \
            or assertion["expected"] in values.values()

    elif atype == "table_cell_equals":
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            page = doc[page_num - 1]
            tables = page.find_tables().tables
            assert tables, "no table detected on this page"
            cell = tables[0].extract()[assertion["row"]][assertion["col"]]
        finally:
            doc.close()
        assert cell == assertion["expected"]

    else:
        pytest.fail(f"no runner implemented for assertion_type {atype!r}")


@pytest.mark.unit
@pytest.mark.parametrize("assertion", _ASSERTIONS, ids=_ids(_ASSERTIONS))
def test_gold_set_assertion(assertion):
    _run_assertion(assertion)

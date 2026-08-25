# code/service/knowhow_write_back.py
"""Additive-only write-back for approved know-how topics (feature/pdf-ingestion).

REVISED 2026-08-24: originally wrote to a brand-new "know_how" tab invented
for this feature (never actually used, deleted). The real Sheet already has 2
established know-how tabs the bot's data_loader.py already reads from:
  - "Know how_ร้านอาหาร" — general business/marketing know-how
  - "Know How_ข้อมูลหนังสือ" — content sourced from actual published books
Routes each topic to the right one via pdf_knowhow_drafting.py's source_type
classification, and writes by matching each tab's REAL header text (both tabs
share the same 6-column shape: เรื่อง/ชื่อหนังสือ, หัวข้อหลัก,
หัวข้อการดำเนินการย่อย, ประเภท, แนวคำตอบ, อ้างอิง) rather than assuming a fixed
column order — same defensive pattern sheet_write_back.py already uses for
the regulatory Sheet, so a column getting reordered by hand doesn't silently
write data into the wrong field.

Two-tier storage per topic, per the design agreed for this feature: a new
"สรุปสั้น" column (added idempotently to both tabs if missing) is what a
retrieval/search pass matches against (short, cheap to embed); the existing
"แนวคำตอบ" column keeps its original meaning (full answer content, unchanged
for the 764 pre-existing rows) and is what actually gets read at answer-
generation time. A Google Sheet cell has a hard 50,000-character limit —
full_text that would exceed a safe margin under that (_FULL_TEXT_CELL_LIMIT)
is uploaded to S3 instead, with the cell holding an `s3://bucket/key` pointer
rather than truncated (i.e. silently lossy) text.

Additive-only, same as sheet_write_back.py: appends new rows only, never
edits/deletes existing ones."""

from __future__ import annotations

from pathlib import Path

import boto3
import gspread

import conf
from service.sheet_write_back import _RESEARCH_REFERENCE_HEADER_VARIANTS, _clean_header, _parse_sheet_url
from utils.logger import get_logger
from utils.sheet_safety import neutralize_formula

logger = get_logger(__name__)

# Comfortably under Google Sheets' hard 50,000-char/cell limit — leaves margin
# for the odd extra character from encoding/formatting quirks.
_FULL_TEXT_CELL_LIMIT = 40_000
_KNOWHOW_FULLTEXT_S3_PREFIX = "restbiz/knowhow_fulltext/"

_TAB_BOOK = "Know How_ข้อมูลหนังสือ"
_TAB_GENERAL = "Know how_ร้านอาหาร"
_TAB_BY_SOURCE_TYPE = {"book": _TAB_BOOK, "document": _TAB_GENERAL}

_SUMMARY_HEADER = "สรุปสั้น"

# Real tabs use เรื่อง (general) or ชื่อหนังสือ (book) for the same document-
# level column — both map to the same logical field here.
_HEADER_VARIANTS = {
    "document_title": ["เรื่อง", "ชื่อหนังสือ"],
    "main_topic": ["หัวข้อหลัก"],
    "sub_topic": ["หัวข้อการดำเนินการย่อย"],
    "category": ["ประเภท"],
    "full_text": ["แนวคำตอบ"],
    "summary": [_SUMMARY_HEADER],
}


def _get_spreadsheet():
    if not conf.PDF_INGESTION_GOOGLE_CREDENTIALS_PATH:
        raise RuntimeError("Know-how write-back not configured: PDF_INGESTION_GOOGLE_CREDENTIALS_PATH is unset.")
    if not conf.SHEET_URL_REGULATORY:
        raise RuntimeError("Know-how write-back not configured: SHEET_URL_REGULATORY is unset.")

    cred_path = conf.PDF_INGESTION_GOOGLE_CREDENTIALS_PATH
    if not Path(cred_path).is_absolute():
        cred_path = str(Path(conf.BASE_DIR) / cred_path)

    client = gspread.service_account(filename=cred_path)
    # Same spreadsheet FILE as the regulatory data (both real know-how tabs
    # live here too) — reuses the one gspread sharing/auth setup already
    # granted for this feature rather than needing a second spreadsheet
    # shared separately.
    spreadsheet_id, _gid = _parse_sheet_url(conf.SHEET_URL_REGULATORY)
    return client.open_by_key(spreadsheet_id)


def _ensure_summary_column(worksheet: gspread.Worksheet) -> None:
    """Idempotent — the 2 real know-how tabs predate this feature and don't
    have a short-summary column; adds one (appended as the last column, never
    inserted in the middle, so nothing already reading these tabs by position
    gets silently shifted) if it isn't already there."""
    header_row = worksheet.row_values(1)
    cleaned = [_clean_header(h) for h in header_row]
    if _SUMMARY_HEADER in cleaned:
        return
    next_col = len(header_row) + 1
    if next_col > worksheet.col_count:
        worksheet.add_cols(next_col - worksheet.col_count)
    worksheet.update_cell(1, next_col, _SUMMARY_HEADER)
    logger.info(f"[KnowhowWriteBack] Added {_SUMMARY_HEADER!r} column to {worksheet.title!r} at column {next_col}")


def _store_full_text(full_text: str, source_file: str, label: str) -> str:
    """Returns either the full_text itself (fits in a cell) or an s3://
    pointer (doesn't fit) — never truncates."""
    if len(full_text) <= _FULL_TEXT_CELL_LIMIT:
        return full_text

    if not conf.PDF_INGESTION_S3_BUCKET:
        # No bucket configured to overflow into — truncating would silently
        # lose content, so fail loudly instead of guessing.
        raise RuntimeError(
            f"full_text for {label!r} is {len(full_text)} chars (limit {_FULL_TEXT_CELL_LIMIT}) "
            "but PDF_INGESTION_S3_BUCKET is unset — cannot store the overflow."
        )

    s3 = boto3.client("s3", region_name=conf.AWS_REGION or None)
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:60]
    key = f"{_KNOWHOW_FULLTEXT_S3_PREFIX}{source_file}/{safe_label}.txt"
    s3.put_object(Bucket=conf.PDF_INGESTION_S3_BUCKET, Key=key, Body=full_text.encode("utf-8"), ContentType="text/plain; charset=utf-8")
    pointer = f"s3://{conf.PDF_INGESTION_S3_BUCKET}/{key}"
    logger.info(f"[KnowhowWriteBack] {label!r}: full_text {len(full_text)} chars exceeds cell limit, stored at {pointer}")
    return pointer


def _build_row(header_row: list[str], values: dict[str, str], reference_text: str) -> list[str]:
    """Matches by REAL header text (like sheet_write_back.py's
    _resolve_value_for_header), not fixed column position — a column that
    exists in the tab but isn't one we write (or vice versa) just comes out
    blank instead of misaligning every other column."""
    row = []
    for raw_header in header_row:
        cleaned = _clean_header(raw_header)
        if cleaned in _RESEARCH_REFERENCE_HEADER_VARIANTS:
            row.append(neutralize_formula(reference_text))
            continue
        matched_key = next((key for key, variants in _HEADER_VARIANTS.items() if cleaned in variants), None)
        row.append(neutralize_formula(values.get(matched_key, "")) if matched_key else "")
    return row


def append_knowhow_topics(topics: list[dict], source_file: str) -> dict:
    """Each topic dict needs: document_title, source_type ("book"|"document"),
    main_topic, sub_topic, summary, full_text, category, page_range (display
    string like "3-5"). Appends ONE row per topic, routed to whichever of the
    2 real know-how tabs source_type points at — a single reviewed PDF can
    produce several rows across either or both tabs, unlike the regulatory
    path's one-PDF-one-row model."""
    if not topics:
        raise ValueError("append_knowhow_topics called with an empty topic list — caller bug")

    spreadsheet = _get_spreadsheet()
    worksheets_cache: dict[str, gspread.Worksheet] = {}
    rows_by_tab: dict[str, int] = {}

    for topic in topics:
        tab_name = _TAB_BY_SOURCE_TYPE.get(topic.get("source_type"), _TAB_GENERAL)
        if tab_name not in worksheets_cache:
            worksheet = spreadsheet.worksheet(tab_name)
            _ensure_summary_column(worksheet)
            worksheets_cache[tab_name] = worksheet
        worksheet = worksheets_cache[tab_name]

        label = topic.get("sub_topic") or topic.get("main_topic") or "topic"
        stored_full_text = _store_full_text(topic["full_text"], source_file, label)
        page_range = topic.get("page_range", "")
        reference_text = f"{source_file} (หน้า {page_range})" if page_range else source_file

        row_values = {
            "document_title": topic.get("document_title", ""),
            "main_topic": topic.get("main_topic", ""),
            "sub_topic": topic.get("sub_topic", ""),
            "category": topic.get("category", ""),
            "full_text": stored_full_text,
            "summary": topic.get("summary", ""),
        }
        row = _build_row(worksheet.row_values(1), row_values, reference_text)
        # stored_full_text is already S3-overflowed below the Sheets limit
        # by _store_full_text above — this guards the OTHER fields
        # (document_title/main_topic/sub_topic/category/summary), which are
        # short by prompt design but not literally length-capped anywhere.
        oversized = [(i, len(v)) for i, v in enumerate(row) if isinstance(v, str) and len(v) > _FULL_TEXT_CELL_LIMIT]
        if oversized:
            raise ValueError(
                f"know-how field(s) too long for a Sheet cell (>{_FULL_TEXT_CELL_LIMIT:,} chars): "
                f"column index(es) {[i for i, _ in oversized]} — topic {label!r} from {source_file!r}"
            )
        worksheet.append_row(row, value_input_option="USER_ENTERED")
        rows_by_tab[tab_name] = rows_by_tab.get(tab_name, 0) + 1
        logger.info(f"[KnowhowWriteBack] Appended row for {label!r} from {source_file} to tab {tab_name!r}")

    return {"rows_appended": sum(rows_by_tab.values()), "rows_by_tab": rows_by_tab}

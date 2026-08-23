# code/service/knowhow_write_back.py
"""Additive-only write-back for approved know-how topics (feature/pdf-ingestion)
— the know-how counterpart to sheet_write_back.py, writing to a separate
"know_how" tab (lighter schema: title/summary/full_text/category/source_file/
page_range) instead of the main regulatory tab's fixed 13 fields.

Two-tier storage per topic, per the design agreed for this feature: `summary`
is what a normal retrieval/search pass matches against (short, cheap to
embed); `full_text` is what actually gets read at answer-generation time once
a topic is selected, so nothing gets lost the way forcing long content into
the structured Sheet's fields would. A Google Sheet cell has a hard 50,000-
character limit — full_text that would exceed a safe margin under that
(_FULL_TEXT_CELL_LIMIT) is uploaded to S3 instead, with the cell holding an
`s3://bucket/key` pointer rather than truncated (i.e. silently lossy) text.

Additive-only, same as sheet_write_back.py: appends new rows only, never
edits/deletes existing ones."""

from __future__ import annotations

from pathlib import Path

import boto3
import gspread

import conf
from utils.logger import get_logger

logger = get_logger(__name__)

# Comfortably under Google Sheets' hard 50,000-char/cell limit — leaves margin
# for the odd extra character from encoding/formatting quirks.
_FULL_TEXT_CELL_LIMIT = 40_000
_KNOWHOW_FULLTEXT_S3_PREFIX = "restbiz/knowhow_fulltext/"
_KNOWHOW_TAB_TITLE = "know_how"

_KNOWHOW_HEADERS = ["title", "summary", "full_text", "category", "source_file", "page_range"]


def _get_spreadsheet():
    if not conf.PDF_INGESTION_GOOGLE_CREDENTIALS_PATH:
        raise RuntimeError("Know-how write-back not configured: PDF_INGESTION_GOOGLE_CREDENTIALS_PATH is unset.")
    if not conf.SHEET_URL_REGULATORY:
        raise RuntimeError("Know-how write-back not configured: SHEET_URL_REGULATORY is unset.")

    cred_path = conf.PDF_INGESTION_GOOGLE_CREDENTIALS_PATH
    if not Path(cred_path).is_absolute():
        cred_path = str(Path(conf.BASE_DIR) / cred_path)

    client = gspread.service_account(filename=cred_path)
    # Same spreadsheet FILE as the regulatory data (just a different tab within
    # it) — reuses the one gspread sharing/auth setup already granted for this
    # feature rather than needing a second spreadsheet shared separately.
    from service.sheet_write_back import _parse_sheet_url

    spreadsheet_id, _gid = _parse_sheet_url(conf.SHEET_URL_REGULATORY)
    return client.open_by_key(spreadsheet_id)


def ensure_knowhow_tab_exists() -> gspread.Worksheet:
    """Idempotent — creates the know_how tab with headers if it doesn't exist
    yet, otherwise just returns the existing one untouched. Safe to call every
    time (e.g. at the start of append_knowhow_topics) rather than requiring a
    separate manual setup step someone could forget."""
    spreadsheet = _get_spreadsheet()
    try:
        worksheet = spreadsheet.worksheet(_KNOWHOW_TAB_TITLE)
        logger.info(f"[KnowhowWriteBack] Using existing '{_KNOWHOW_TAB_TITLE}' tab")
        return worksheet
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=_KNOWHOW_TAB_TITLE, rows=1000, cols=len(_KNOWHOW_HEADERS))
        worksheet.append_row(_KNOWHOW_HEADERS, value_input_option="RAW")
        logger.info(f"[KnowhowWriteBack] Created new '{_KNOWHOW_TAB_TITLE}' tab with headers")
        return worksheet


def _store_full_text(full_text: str, source_file: str, topic_title: str) -> str:
    """Returns either the full_text itself (fits in a cell) or an s3://
    pointer (doesn't fit) — never truncates."""
    if len(full_text) <= _FULL_TEXT_CELL_LIMIT:
        return full_text

    if not conf.PDF_INGESTION_S3_BUCKET:
        # No bucket configured to overflow into — truncating would silently
        # lose content, so fail loudly instead of guessing.
        raise RuntimeError(
            f"full_text for {topic_title!r} is {len(full_text)} chars (limit {_FULL_TEXT_CELL_LIMIT}) "
            "but PDF_INGESTION_S3_BUCKET is unset — cannot store the overflow."
        )

    s3 = boto3.client("s3", region_name=conf.AWS_REGION or None)
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic_title)[:60]
    key = f"{_KNOWHOW_FULLTEXT_S3_PREFIX}{source_file}/{safe_title}.txt"
    s3.put_object(Bucket=conf.PDF_INGESTION_S3_BUCKET, Key=key, Body=full_text.encode("utf-8"), ContentType="text/plain; charset=utf-8")
    pointer = f"s3://{conf.PDF_INGESTION_S3_BUCKET}/{key}"
    logger.info(f"[KnowhowWriteBack] {topic_title!r}: full_text {len(full_text)} chars exceeds cell limit, stored at {pointer}")
    return pointer


def append_knowhow_topics(topics: list[dict], source_file: str) -> dict:
    """Each topic dict needs: title, summary, full_text, category, page_range
    (page_range as a display string like "3-5"). Appends ONE row per topic —
    a single reviewed PDF can produce several know-how rows, unlike the
    regulatory path's one-PDF-one-row model."""
    if not topics:
        raise ValueError("append_knowhow_topics called with an empty topic list — caller bug")

    worksheet = ensure_knowhow_tab_exists()
    rows_appended = 0
    for topic in topics:
        stored_full_text = _store_full_text(topic["full_text"], source_file, topic["title"])
        row = [
            topic["title"],
            topic["summary"],
            stored_full_text,
            topic.get("category", ""),
            source_file,
            topic.get("page_range", ""),
        ]
        worksheet.append_row(row, value_input_option="USER_ENTERED")
        rows_appended += 1
        logger.info(f"[KnowhowWriteBack] Appended row for topic {topic['title']!r} from {source_file}")

    return {"rows_appended": rows_appended}

# code/service/sheet_write_back.py
"""Additive-only Google Sheet write-back for approved PDF review items
(feature/pdf-ingestion). Appends a NEW row only — never edits or deletes an
existing row. This is the single most important safety property of this whole
feature (agreed at the very start of the staging/review design): a bug here can
at worst leave a stray extra row a human can delete by hand; it can never
silently corrupt or remove real production data.

decision_type == "update" is handled the same way — a new row is appended, and
the OLD row gets a note appended in its own dedicated column if the Sheet has
one (best-effort: logs and continues if not, never fails the whole write over
a missing note). The actual deletion/replacement of the old row is left to a
human, on purpose — see the original design discussion.

Every row written carries `source_review_id` (if that column exists in the
Sheet) so it can always be traced back to which PDF/review decision produced
it — the audit trail this design has relied on since the beginning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import gspread

import conf
from model.pdf_review_item import ReviewItem
from service.pdf_field_drafting import DRAFTABLE_FIELDS
from utils.logger import get_logger
from utils.page_ranges import format_page_ranges, fuzzy_ratio
from utils.sheet_safety import neutralize_formula

logger = get_logger(__name__)


def _clean_header(name: str) -> str:
    """Must match data_loader.py's DataLoader.clean_header() exactly — real Sheet
    headers have embedded newlines (e.g. "การดำเนินการ\\nตามหน่วยงาน") that don't
    match DRAFTABLE_FIELDS' plain-space values unless normalized the same way the
    read path already does. Without this, columns silently write blank instead
    of erroring — worse than a crash, since nothing would look wrong until a
    human noticed missing data in the Sheet."""
    if not isinstance(name, str):
        return name
    name = name.replace("\n", " ").replace("\r", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()

_SOURCE_REVIEW_ID_HEADER = "source_review_id"
_SUPERSEDED_NOTE_HEADER = "superseded_note"

# Google Sheets' actual hard limit — live-tested 2026-08-25: an oversized
# single field (not full_text, which already has S3-overflow protection in
# knowhow_write_back.py) raised a raw gspread APIError that propagated all
# the way to the reviewer as a cryptic "[400] Your input contains more than
# the maximum..." with no indication of WHICH field or how to fix it. The
# write failed atomically (no half-written row), so this was never silent
# data corruption — but it WAS a dead-end the reviewer had no path out of.
_SHEET_CELL_CHAR_LIMIT = 50_000


def _parse_sheet_url(sheet_url: str) -> tuple[str, str]:
    """Returns (spreadsheet_id, gid) — same parsing logic as
    data_loader.py's _build_csv_export_url, reused here for the write path."""
    u = urlparse(sheet_url)
    base = f"{u.scheme}://{u.netloc}{u.path}".split("/edit")[0].rstrip("/")
    spreadsheet_id = base.rstrip("/").split("/")[-1]
    q = parse_qs(u.query)
    gid = (q.get("gid", [None])[0]) or (parse_qs(u.fragment).get("gid", [None])[0])
    if not gid:
        raise ValueError(f"Sheet URL missing gid: {sheet_url}")
    return spreadsheet_id, gid


def _get_worksheet():
    """Opens the regulatory Sheet's target tab. Only "regulatory" is wired up —
    routing new_category/marketing/bakery content to the right tab needs a
    classification step that doesn't exist yet (flagged, not silently assumed).
    Uses PDF_INGESTION_GOOGLE_CREDENTIALS_PATH, NOT GOOGLE_CREDENTIALS_PATH —
    the latter is a different feature's (feedback logging) service account,
    a different GCP project owned by someone else's work; reusing it here
    would silently cross-wire two unrelated credentials."""
    if not conf.PDF_INGESTION_GOOGLE_CREDENTIALS_PATH:
        raise RuntimeError("Sheet write-back not configured: PDF_INGESTION_GOOGLE_CREDENTIALS_PATH is unset.")
    if not conf.SHEET_URL_REGULATORY:
        raise RuntimeError("Sheet write-back not configured: SHEET_URL_REGULATORY is unset.")

    # A relative value must not depend on the current working directory (same
    # bug class fixed in pdf_review_queue_manager.py — a script run from the
    # repo root and a server run from code/ would resolve it differently).
    # google_sheets_service.py already does this for GOOGLE_CREDENTIALS_PATH;
    # mirrored here for consistency.
    cred_path = conf.PDF_INGESTION_GOOGLE_CREDENTIALS_PATH
    if not Path(cred_path).is_absolute():
        cred_path = str(Path(conf.BASE_DIR) / cred_path)

    client = gspread.service_account(filename=cred_path)
    spreadsheet_id, gid = _parse_sheet_url(conf.SHEET_URL_REGULATORY)
    spreadsheet = client.open_by_key(spreadsheet_id)
    return spreadsheet.get_worksheet_by_id(int(gid))


# Copied from data_loader.py's research_reference alias list — this column means
# "which document is this row based on", and for a PDF-ingested row the answer is
# simply the source PDF itself (plus which page(s) of it, so a reviewer can jump
# straight to the source without re-reading the whole PDF). Filled in
# deterministically from item.filename/item.pages, NOT via the LLM — there's
# nothing to guess, so routing it through the drafting prompt would only add
# hallucination risk for zero benefit.
_RESEARCH_REFERENCE_HEADER_VARIANTS = [
    "อ้างอิง Research",
    "อ้างอิง (Research) เอกสาร (Document)",
    "อ้างอิง (Research)",
    "อ้างอิง (Research) เอกสาร",
    "อ้างอิง (Research) เอกสาร(Document)",
]


def _format_page_range(item: ReviewItem) -> str:
    """"หน้า 4" for a single page, "หน้า 1-3" for one contiguous range, "หน้า
    1-2, 5" for disjoint pages — REVISED 2026-08-24: an item's pages are no
    longer guaranteed contiguous (structured_license topic-splitting can
    merge a topic that resumes after an interruption, see
    utils/page_ranges.py), so a plain min/max would misleadingly claim
    in-between pages are included when they aren't."""
    page_nums = [p.page_num for p in item.pages]
    formatted = format_page_ranges(page_nums)
    return f"หน้า {formatted}" if formatted else ""


def _resolve_value_for_header(raw_header: str, fields: dict[str, str], item: ReviewItem) -> str:
    """Matches one real (raw, possibly newline-containing) Sheet header against
    known column variants (cleaned the same way data_loader.py cleans headers on
    read) and returns the value for whichever field it belongs to. Returns "" for
    any header this schema doesn't know how to fill (e.g. "แนวคำตอบ" — an
    editorial/style decision with no source-of-truth in the PDF at all) — never
    guesses."""
    cleaned = _clean_header(raw_header)

    if cleaned in _RESEARCH_REFERENCE_HEADER_VARIANTS:
        page_range = _format_page_range(item)
        return f"{item.filename} ({page_range})" if page_range else item.filename

    for header_variants in DRAFTABLE_FIELDS.values():
        if cleaned in header_variants:
            return fields.get(header_variants[0], "")
    return ""


def append_review_item_to_sheet(item: ReviewItem) -> dict:
    """Writes ONE new row for an approved, non-duplicate ReviewItem.
    Returns {"row_appended": True, "row_number": int} on success."""
    if item.review_status != "approved":
        raise ValueError(f"append_review_item_to_sheet called on review_status={item.review_status!r}, expected 'approved'")
    if item.decision_type == "duplicate":
        raise ValueError("decision_type='duplicate' means nothing should be written to the Sheet — caller bug")
    if not item.llm_drafted_fields:
        raise ValueError("item.llm_drafted_fields is empty — draft fields before approving")

    worksheet = _get_worksheet()
    header_row = worksheet.row_values(1)

    fields = item.llm_drafted_fields
    new_row = []
    for raw_col_name in header_row:
        if _clean_header(raw_col_name) == _SOURCE_REVIEW_ID_HEADER:
            new_row.append(item.id)
        else:
            new_row.append(neutralize_formula(_resolve_value_for_header(raw_col_name, fields, item)))

    matched_count = sum(1 for v in new_row if v)
    logger.info(f"[SheetWriteBack] Matched {matched_count}/{len(header_row)} Sheet columns to drafted fields for item {item.id}")

    oversized = [
        (_clean_header(raw_col_name), len(value))
        for raw_col_name, value in zip(header_row, new_row)
        if isinstance(value, str) and len(value) > _SHEET_CELL_CHAR_LIMIT
    ]
    if oversized:
        details = ", ".join(f"{name!r} ({length:,} ตัวอักษร)" for name, length in oversized)
        raise ValueError(
            f"ฟิลด์ยาวเกินขีดจำกัดของ Google Sheets ({_SHEET_CELL_CHAR_LIMIT:,} ตัวอักษรต่อช่อง): {details} — "
            f"กรุณาแก้ไขฟิลด์นี้ให้สั้นลงก่อนอนุมัติอีกครั้ง (item {item.id})"
        )

    worksheet.append_row(new_row, value_input_option="USER_ENTERED")
    new_row_number = len(worksheet.get_all_values())  # append_row doesn't return the row number itself
    logger.info(
        f"[SheetWriteBack] Appended row {new_row_number} for review item {item.id} "
        f"({item.filename}, decision_type={item.decision_type})"
    )

    old_row_ref_warning = None
    if item.decision_type == "update" and item.old_row_ref:
        old_row_ref_warning = _note_superseded_row(worksheet, header_row, item)

    result = {"row_appended": True, "row_number": new_row_number}
    if old_row_ref_warning:
        result["old_row_ref_warning"] = old_row_ref_warning
    return result


def _note_superseded_row(worksheet, header_row: list[str], item: ReviewItem) -> Optional[str]:
    """Best-effort: writes a note on the OLD row pointing at the new one, if the
    Sheet has a dedicated column for it. Never deletes/overwrites the old row's
    actual content — and never raises, since a missing note column shouldn't
    fail the whole approval (the new row is already safely written by this point).

    Returns a warning string (never blocks the write — the reviewer typed this
    row number deliberately and might know something the system doesn't) when
    old_row_ref's own department+license_type don't relate at all to the new
    item's — live-testing found this catches a real risk with zero protection
    before: a typo'd/misremembered row number silently mislabels a completely
    unrelated real Sheet row as "superseded" with no error or signal anywhere,
    since update_cell() only fails on an out-of-range row number, not a
    valid-but-wrong one."""
    cleaned_headers = [_clean_header(h) for h in header_row]
    if _SUPERSEDED_NOTE_HEADER not in cleaned_headers:
        logger.warning(
            f"[SheetWriteBack] Sheet has no '{_SUPERSEDED_NOTE_HEADER}' column — "
            f"skipping old-row note for review item {item.id}. Add that column manually "
            f"if you want this automated; old row ref was: {item.old_row_ref!r}"
        )
        return None
    try:
        old_row_number = int(item.old_row_ref)
    except (TypeError, ValueError):
        logger.warning(
            f"[SheetWriteBack] old_row_ref={item.old_row_ref!r} is not a row number — "
            f"skipping old-row note for review item {item.id}"
        )
        return None

    warning = None
    try:
        old_row_values = worksheet.row_values(old_row_number)
        col_index_map = {_clean_header(h): idx for idx, h in enumerate(header_row)}
        for key in ("department", "license_type"):
            for variant in DRAFTABLE_FIELDS[key]:
                if variant in col_index_map:
                    idx = col_index_map[variant]
                    old_val = old_row_values[idx] if idx < len(old_row_values) else ""
                    new_val = (item.llm_drafted_fields or {}).get(DRAFTABLE_FIELDS[key][0], "")
                    if old_val and new_val and fuzzy_ratio(old_val, new_val) < 0.5:
                        warning = (
                            f"เลขแถวเดิม ({old_row_number}) ดูเหมือนจะเป็นคนละเรื่องกับเอกสารใหม่ — "
                            f"แถวเดิม {key}={old_val!r} แต่เอกสารใหม่ {key}={new_val!r} กรุณาตรวจสอบว่าใส่เลขแถวถูกต้อง"
                        )
                        break
            if warning:
                break
    except Exception as e:
        logger.warning(f"[SheetWriteBack] Could not check old_row_ref={old_row_number} relatedness for item {item.id}: {e}")

    col_index = cleaned_headers.index(_SUPERSEDED_NOTE_HEADER) + 1  # gspread columns are 1-indexed
    note = f"มีข้อมูลใหม่มาแทนที่ — ดูแถวใหม่ (source_review_id={item.id})"
    try:
        worksheet.update_cell(old_row_number, col_index, note)
        logger.info(f"[SheetWriteBack] Noted row {old_row_number} as superseded by review item {item.id}")
        if warning:
            logger.warning(f"[SheetWriteBack] {warning}")
    except Exception as e:
        logger.error(f"[SheetWriteBack] Failed to write superseded-note on row {old_row_number}: {e}")
    return warning

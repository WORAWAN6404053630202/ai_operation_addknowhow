# code/service/google_sheets_service.py
"""Feedback logging to Google Sheets. Soft-disables (no-ops) if GOOGLE_CREDENTIALS_PATH
or FEEDBACK_SHEET_ID is unset — never raises at import/startup time."""

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import gspread

import conf
from utils.logger import get_logger

logger = get_logger(__name__)

_BRANCH_ID_FALLBACK = "0" * 33


class GoogleSheetsService:
    def __init__(self):
        self.client = None
        self.sheet = None
        try:
            if conf.GOOGLE_CREDENTIALS_PATH and conf.FEEDBACK_SHEET_ID:
                cred_path = conf.GOOGLE_CREDENTIALS_PATH
                if not Path(cred_path).is_absolute():
                    cred_path = str(Path(conf.BASE_DIR) / cred_path)
                logger.info(f"[GoogleSheets] Looking for credentials at: {cred_path}")
                self.client = gspread.service_account(filename=cred_path)
                self.sheet = self.client.open_by_key(conf.FEEDBACK_SHEET_ID).sheet1
                logger.info("[GoogleSheets] Service initialized successfully.")
            else:
                logger.warning(
                    "[GoogleSheets] GOOGLE_CREDENTIALS_PATH or FEEDBACK_SHEET_ID not set "
                    "— feedback logging disabled."
                )
        except Exception as e:
            logger.error(f"[GoogleSheets] Failed to initialize: {e}")

    def save_feedback(
        self,
        category_abbr: str,
        query: str,
        branch_id: Optional[str] = None,
        branch_name: Optional[str] = None,
    ) -> bool:
        """Synchronous, blocking write — call via save_feedback_async from request-handling code."""
        if not self.sheet:
            logger.error("[GoogleSheets] Cannot save feedback: service not configured.")
            return False
        try:
            now = datetime.now(ZoneInfo("Asia/Bangkok"))
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            date_code = now.strftime("%y%m%d")
            safe_branch_id = branch_id or _BRANCH_ID_FALLBACK

            col_values = self.sheet.col_values(1)
            running_number = f"{len(col_values):04d}"

            feedback_code = f"{category_abbr} 01 {safe_branch_id} {date_code} {running_number}"
            row = [timestamp, conf.BOT_TYPE, feedback_code, query, branch_id or "", branch_name or ""]

            self.sheet.append_row(row)
            return True
        except Exception as e:
            logger.error(f"[GoogleSheets] Failed to append row: {e}")
            return False

    def save_feedback_async(
        self,
        category_abbr: str,
        query: str,
        branch_id: Optional[str] = None,
        branch_name: Optional[str] = None,
    ) -> None:
        """Fire-and-forget. PersonaSupervisor.handle() runs on a plain executor thread with
        no asyncio event loop, so this uses threading.Thread rather than asyncio.to_thread."""
        threading.Thread(
            target=self.save_feedback,
            args=(category_abbr, query, branch_id, branch_name),
            daemon=True,
            name="gsheets-feedback",
        ).start()


gsheets_service = GoogleSheetsService()

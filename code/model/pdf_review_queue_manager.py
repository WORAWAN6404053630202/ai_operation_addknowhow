# code/model/pdf_review_queue_manager.py
"""Persistence for PDF review-queue items (feature/pdf-ingestion) — one JSON file
per item, mirroring model/state_manager.py's pattern exactly (atomic write via
temp-file-then-replace, best-effort cross-process file locking with staleness
detection) rather than introducing a new storage technology. This project has no
SQL database anywhere; matching the existing convention beats inventing a second
persistence style for one feature."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from model.pdf_review_item import ReviewItem

_LOG = logging.getLogger("restbiz.pdf_review_queue")

try:
    import conf
except Exception:
    conf = None


class PdfReviewQueueManager:
    def __init__(self, persist_dir: str | None = None):
        if persist_dir:
            base = Path(persist_dir)
        elif conf is not None and getattr(conf, "PDF_REVIEW_QUEUE_DIR", None):
            base = Path(getattr(conf, "PDF_REVIEW_QUEUE_DIR"))
            # A relative value (e.g. "data/pdf_review_queue_dev" in env.dev.properties)
            # must not depend on the current working directory — a script run from
            # the repo root and a server run from code/ would otherwise silently
            # write to two different physical directories. Anchor it to conf.BASE_DIR,
            # same as every other path conf.py resolves.
            if not base.is_absolute():
                base = Path(getattr(conf, "BASE_DIR", ".")) / base
        else:
            base = Path(__file__).resolve().parent.parent / "data" / "pdf_review_queue"

        self.dir = base
        self.dir.mkdir(parents=True, exist_ok=True)

        self._lock_timeout_s = float(getattr(conf, "STATE_LOCK_TIMEOUT_S", 2.0) if conf is not None else 2.0)
        self._lock_poll_s = float(getattr(conf, "STATE_LOCK_POLL_S", 0.05) if conf is not None else 0.05)
        self._lock_stale_s = float(getattr(conf, "STATE_LOCK_STALE_S", 15.0) if conf is not None else 15.0)

    def _item_path(self, item_id: str) -> Path:
        safe_id = (item_id or "").replace("/", "_").replace("\\", "_").strip()
        return self.dir / f"{safe_id}.json"

    def _lock_path(self, item_id: str) -> Path:
        return self._item_path(item_id).with_suffix(".lock")

    def _acquire_lock(self, item_id: str) -> None:
        lock_path = self._lock_path(item_id)
        deadline = time.time() + self._lock_timeout_s

        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, json.dumps({"pid": os.getpid(), "ts": time.time()}).encode("utf-8"))
                finally:
                    os.close(fd)
                return
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > self._lock_stale_s:
                        lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass
                if time.time() >= deadline:
                    raise TimeoutError(f"Could not acquire review-queue lock for item_id={item_id!r}")
                time.sleep(self._lock_poll_s)

    def _release_lock(self, item_id: str) -> None:
        self._lock_path(item_id).unlink(missing_ok=True)

    def save(self, item: ReviewItem) -> None:
        path = self._item_path(item.id)
        tmp_path = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")

        self._acquire_lock(item.id)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(item.model_dump(), f, ensure_ascii=False, indent=2)
            tmp_path.replace(path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._release_lock(item.id)

    def load(self, item_id: str) -> Optional[ReviewItem]:
        path = self._item_path(item_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            _LOG.warning(f"[PdfReviewQueue] Corrupt item file for id={item_id}: {e}")
            return None
        return ReviewItem.model_validate(data)

    def _iter_all(self) -> list[ReviewItem]:
        items: list[ReviewItem] = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items.append(ReviewItem.model_validate(data))
            except Exception as e:
                _LOG.warning(f"[PdfReviewQueue] Skipping unreadable item file {path.name}: {e}")
                continue
        return items

    def list_all(self) -> list[ReviewItem]:
        return self._iter_all()

    def list_pending(self) -> list[ReviewItem]:
        return [item for item in self._iter_all() if item.review_status == "pending"]

    def delete(self, item_id: str) -> None:
        """Deletes the review-queue record itself (the workflow tracking entry) —
        never touches the real Google Sheet data. Used for cleaning up test/junk
        uploads, not for undoing an approval (that's a Sheet-side deactivation,
        per the additive-only design — a separate, not-yet-built concern)."""
        self._item_path(item_id).unlink(missing_ok=True)

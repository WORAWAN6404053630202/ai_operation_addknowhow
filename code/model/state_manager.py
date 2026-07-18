"""
State Manager Service
Handles persistence of conversation states

PRODUCTION FIXES:
- Persist directory is stable (not dependent on current working directory).
- Supports env override via conf.STATE_DIR (if present).
- Best-effort cross-process file locking to prevent concurrent write clobber
- Payload trimming on save to reduce latency/state bloat (messages + internal_messages)
- NEW: list sessions
- NEW: purge sessions older than N days
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

_LOG = logging.getLogger("restbiz.state_manager")

from model.conversation_state import ConversationState

try:
    import conf
except Exception:
    conf = None


class StateManager:
    def __init__(self, persist_dir: str | None = None):
        if persist_dir:
            base = Path(persist_dir)
        elif conf is not None and getattr(conf, "STATE_DIR", None):
            base = Path(getattr(conf, "STATE_DIR"))
        else:
            base = Path(__file__).resolve().parent.parent / "data" / "states"

        self.dir = base
        self.dir.mkdir(parents=True, exist_ok=True)

        self._lock_timeout_s = float(getattr(conf, "STATE_LOCK_TIMEOUT_S", 2.0) if conf is not None else 2.0)
        self._lock_poll_s = float(getattr(conf, "STATE_LOCK_POLL_S", 0.05) if conf is not None else 0.05)

        self._default_max_recent = int(getattr(conf, "MAX_RECENT_MESSAGES_SAVE", 18) if conf is not None else 18)
        self._default_max_internal = int(getattr(conf, "MAX_INTERNAL_MESSAGES_SAVE", 40) if conf is not None else 40)

    def _safe_session_id(self, session_id: str) -> str:
        return (session_id or "").replace("/", "_").replace("\\", "_").strip()

    def _state_path(self, session_id: str) -> Path:
        safe_id = self._safe_session_id(session_id)
        return self.dir / f"{safe_id}.json"

    def _lock_path(self, session_id: str) -> Path:
        safe_id = self._safe_session_id(session_id)
        return self.dir / f"{safe_id}.lock"

    def _acquire_lock(self, session_id: str) -> None:
        lock_path = self._lock_path(session_id)
        deadline = time.time() + self._lock_timeout_s

        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    payload = {"pid": os.getpid(), "ts": time.time()}
                    os.write(fd, json.dumps(payload).encode("utf-8"))
                finally:
                    os.close(fd)
                return
            except FileExistsError:
                try:
                    stat = lock_path.stat()
                    age = time.time() - float(stat.st_mtime)
                    stale_after = float(getattr(conf, "STATE_LOCK_STALE_S", 15.0) if conf is not None else 15.0)
                    if age > stale_after:
                        lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass

                if time.time() >= deadline:
                    raise TimeoutError(f"Could not acquire state lock for session_id={session_id!r}")
                time.sleep(self._lock_poll_s)

    def _release_lock(self, session_id: str) -> None:
        lock_path = self._lock_path(session_id)
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    def cleanup_orphan_locks(self) -> int:
        """
        Remove all .lock files in the state directory whose modification time exceeds
        the STATE_LOCK_STALE_S threshold. Safe to call at startup before any requests
        are accepted. Returns the number of lock files removed.
        """
        stale_after = float(
            getattr(conf, "STATE_LOCK_STALE_S", 15.0) if conf is not None else 15.0
        )
        removed = 0
        try:
            for lock_path in self.dir.glob("*.lock"):
                try:
                    age = time.time() - float(lock_path.stat().st_mtime)
                    if age > stale_after:
                        lock_path.unlink(missing_ok=True)
                        removed += 1
                except Exception:
                    continue
        except Exception:
            pass
        return removed

    # Context keys that should never be persisted to disk — they are only valid within
    # the single request that set them and become stale immediately after.
    _EPHEMERAL_CTX_KEYS = frozenset({
        "_broad_retrieval_docs",     # large doc list; set for Academic handoff, stale afterwards
        "_broad_retrieval_query",    # companion query; same lifecycle as _broad_retrieval_docs
        "_multi_topic_retrieval",    # flag for multi-topic DC suppression; per-turn only
        "_style_pw_cache",           # style pre-warm result; keyed by (raw_stripped, last_q), per-turn only
        # Retrieve-loop guards (practical.py) — per-turn counters; stale values across turns
        # cause premature hard-abort ("ขอโทษครับ ไม่พบข้อมูล") on the next legitimate question.
        "_retrieve_blocked_count",   # counts consecutive retrieve-blocked events within one turn
        "_force_answer_count",       # counts force-answer attempts within one turn (BUG-2 fix)
        # Auto-fill guard (practical.py) — set before recursive handle(), popped after.
        # If an exception escapes the recursive call the pop never runs; without ephemeral
        # treatment the guard sticks to disk and blocks auto-fill permanently. (BUG-3 fix)
        "_autofill_guard",
        # Supervisor-set retrieval-done flag (supervisor.py → read in practical.py).
        # Cleared at start of _ensure_practical_retrieval_for_legal() on legal-question turns,
        # but greeting/thanks turns leave it True. Adding here prevents stale True leaking
        # into next turn (confirmed in 6 persisted state files in /data/states/).
        "_supervisor_retrieval_done",
    })

    def _trim_state_for_save(self, state: ConversationState) -> None:
        max_recent = None
        try:
            sp = getattr(state, "strict_profile", None) or {}
            if isinstance(sp, dict):
                v = sp.get("max_recent_messages")
                if v is not None:
                    max_recent = int(v)
        except Exception:
            max_recent = None

        if not max_recent or max_recent <= 0:
            max_recent = self._default_max_recent

        if isinstance(state.messages, list) and len(state.messages) > max_recent:
            state.messages = state.messages[-max_recent:]

        max_internal = self._default_max_internal
        if isinstance(state.internal_messages, list) and max_internal > 0 and len(state.internal_messages) > max_internal:
            state.internal_messages = state.internal_messages[-max_internal:]

        # Strip ephemeral context keys before persisting — they must not survive across requests.
        ctx = getattr(state, "context", None)
        if isinstance(ctx, dict):
            for _ek in self._EPHEMERAL_CTX_KEYS:
                ctx.pop(_ek, None)

    def save(self, session_id: str, state: ConversationState) -> None:
        if not session_id:
            raise ValueError("session_id is required")

        state.session_id = session_id
        self._trim_state_for_save(state)

        path = self._state_path(session_id)
        tmp_path = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")

        self._acquire_lock(session_id)
        try:
            payload = state.model_dump()
            payload.setdefault("_meta", {})
            payload["_meta"]["schema_version"] = payload["_meta"].get("schema_version", "v1")
            payload["_meta"]["saved_at"] = time.time()

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))

            tmp_path.replace(path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._release_lock(session_id)

    def load(self, session_id: str) -> Optional[ConversationState]:
        if not session_id:
            return None

        path = self._state_path(session_id)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, ValueError) as _je:
                _LOG.warning(
                    "[StateManager] Corrupt state file for session %s — starting fresh (%s)",
                    session_id, _je,
                )
                return None

        data.pop("_meta", None)

        # Sanitize context: pending_slot must always be a dict or absent
        _ctx = data.get("context") or {}
        if isinstance(_ctx, dict):
            _ps = _ctx.get("pending_slot")
            if _ps is not None and not isinstance(_ps, dict):
                _ctx.pop("pending_slot", None)
                data["context"] = _ctx

        state = ConversationState(**data)
        # Backward compat: old sessions saved before display_messages field existed
        # will have display_messages=[] — sync from messages so UI still shows history.
        state.sync_display_messages()
        return state

    def delete(self, session_id: str) -> None:
        if not session_id:
            return

        path = self._state_path(session_id)

        _lock_acquired = False
        try:
            self._acquire_lock(session_id)
            _lock_acquired = True
        except Exception:
            pass

        try:
            if path.exists():
                path.unlink()
        finally:
            if _lock_acquired:
                self._release_lock(session_id)

    # session listing
    def list_session_ids(self) -> List[str]:
        """Return all active session IDs without loading full state (used for lock cleanup)."""
        return [p.stem for p in self.dir.glob("*.json")]

    def list_sessions(self, limit: int = 20, client_key: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        client_key = (client_key or "").strip()

        for path in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                data.pop("_meta", None)

                context = data.get("context") or {}
                if client_key:
                    owner_key = str(context.get("client_key") or "").strip()
                    if owner_key != client_key:
                        continue

                session_id = str(data.get("session_id") or path.stem)
                persona_id = str(data.get("persona_id") or "practical")
                messages = data.get("messages") or []

                first_user = ""
                for m in messages:
                    if m.get("role") == "user" and (m.get("content") or "").strip():
                        first_user = (m.get("content") or "").strip()
                        break

                preview = first_user[:80] if first_user else f"Session {session_id}"
                updated_at = path.stat().st_mtime

                out.append(
                    {
                        "session_id": session_id,
                        "persona_id": persona_id,
                        "preview": preview,
                        "updated_at": updated_at,
                    }
                )
            except Exception:
                continue

            if len(out) >= limit:
                break

        return out

    # purge old sessions
    def purge_older_than_days(self, days: int = 7) -> int:
        deleted = 0
        now = time.time()
        cutoff = now - (max(1, int(days)) * 86400)

        for path in self.dir.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    session_id = path.stem
                    self.delete(session_id)
                    deleted += 1
            except Exception:
                continue

        return deleted
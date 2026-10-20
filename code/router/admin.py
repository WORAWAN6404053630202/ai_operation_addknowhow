"""
Admin Dashboard Router
Endpoints for monitoring bot usage, sessions, and logs.
"""

import asyncio
import os
import time
from pathlib import Path

try:
    import conf as _conf
except Exception:
    _conf = None

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from model.state_manager import StateManager
from utils.admin_auth import require_admin_key
from utils.logger import get_logger
from utils.simple_cache import get_cache

_LOG = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])
# API routes need the admin key; the dashboard shell below stays open since it
# renders no data itself (a bare page load can't attach a custom header) —
# static/admin.html's apiGet() attaches the key to every /admin/api/* call.
_auth = Depends(require_admin_key)

_state_manager = StateManager()

# Path to uvicorn log file (adjust if needed via env)
_LOG_FILE = Path(os.getenv("LOG_FILE", str(Path(__file__).resolve().parent.parent.parent / "uvicorn.log")))

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard():
    html_path = Path(__file__).resolve().parent.parent / "static" / "admin.html"
    if not html_path.exists():
        return HTMLResponse("<h1>admin.html not found</h1>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@router.get("/api/sessions", dependencies=[_auth])
async def admin_sessions(limit: int = Query(default=50, le=200)):
    """List all sessions with message count and preview.

    2026-10 fix (core2-M5): this used to be `async def` doing real
    synchronous disk I/O directly on the single-threaded asyncio event
    loop — `list_sessions` globs + opens + json.load's every session file,
    then this function used to ALSO call `_state_manager.load(sid)` per
    row (N+1: re-opening and re-parsing the exact same file list_sessions
    had already fully parsed, just to re-extract token/cost fields it
    already had in hand). Opening this page with a few hundred real
    sessions could block the event loop for seconds — stalling every
    concurrent user's /chat/stream SSE connection, which route_v1.py
    documents extensively and already guards against for the exact same
    reason. list_sessions now returns the token/cost/message-count fields
    directly (see model/state_manager.py), and the whole call goes through
    run_in_executor so the blocking work runs on a worker thread instead.
    """
    loop = asyncio.get_running_loop()
    sessions = await loop.run_in_executor(None, _state_manager.list_sessions, limit)

    result = []
    for s in sessions:
        prompt_tokens = s.get("prompt_tokens", 0)
        completion_tokens = s.get("completion_tokens", 0)
        result.append({
            "session_id": s["session_id"],
            "persona_id": s.get("persona_id", "practical"),
            "preview": s.get("preview", ""),
            "updated_at": s.get("updated_at"),
            "total_messages": s.get("total_messages", 0),
            "user_messages": s.get("user_messages", 0),
            "bot_messages": s.get("bot_messages", 0),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            # Real accumulated cost (see model/conversation_state.py's total_cost
            # field) — NOT re-estimated from token counts. Re-deriving cost from
            # prompt/completion tokens would blend every model used in the session
            # (Sonnet/GPT-5.1 AND cheap Haiku classifier calls) priced at a single
            # assumed model for the whole persona — total_cost is accumulated
            # per-call at the correct model's rate at the source (llm_call.py).
            "total_cost": s.get("total_cost", 0.0),
        })

    return JSONResponse({"sessions": result, "total": len(result)})


@router.get("/api/session/{session_id}", dependencies=[_auth])
async def admin_session_detail(session_id: str):
    """Full message history for a session.

    2026-10 fix (core2-M5): see admin_sessions' matching comment — this
    does the same real blocking disk I/O directly on the event loop.
    """
    loop = asyncio.get_running_loop()
    state = await loop.run_in_executor(None, _state_manager.load, session_id)
    if state is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    messages = state.messages or []
    context  = state.context or {}

    prompt_tokens     = getattr(state, "total_prompt_tokens", 0) or 0
    completion_tokens = getattr(state, "total_completion_tokens", 0) or 0
    # Real accumulated cost, not re-estimated — see matching comment in admin_sessions() above.
    total_cost        = round(getattr(state, "total_cost", 0.0) or 0.0, 6)

    return JSONResponse({
        "session_id": session_id,
        "persona_id": state.persona_id,
        "messages": messages,
        "collected_slots": context.get("collected_slots", {}),
        "last_topic": context.get("last_topic", ""),
        "fsm_state": context.get("fsm_state", ""),
        "total_messages": len(messages),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "total_cost": total_cost,
    })


@router.get("/api/stats", dependencies=[_auth])
async def admin_stats():
    """Overall stats: session count, cache, log summary.

    2026-10 fix (core2-M5): see admin_sessions' matching comment.
    """
    loop = asyncio.get_running_loop()
    all_sessions = await loop.run_in_executor(None, _state_manager.list_sessions, 500)

    now = time.time()
    today_cutoff   = now - 86400
    week_cutoff    = now - 7 * 86400

    sessions_today = [s for s in all_sessions if (s.get("updated_at") or 0) >= today_cutoff]
    sessions_week  = [s for s in all_sessions if (s.get("updated_at") or 0) >= week_cutoff]

    cache = get_cache()
    cache_stats = cache.get_stats()

    return JSONResponse({
        "sessions": {
            "total": len(all_sessions),
            "today": len(sessions_today),
            "this_week": len(sessions_week),
        },
        "cache": cache_stats,
    })


def _read_log_tail(lines: int) -> dict:
    if not _LOG_FILE.exists():
        # Try relative path (when running from code/ dir)
        alt = Path(__file__).resolve().parent.parent.parent / "uvicorn.log"
        if alt.exists():
            log_path = alt
        else:
            return {"lines": [], "error": f"Log file not found: {_LOG_FILE}"}
    else:
        log_path = _LOG_FILE

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:]
        return {"lines": [l.rstrip() for l in tail], "total_lines": len(all_lines)}
    except Exception as e:
        return {"lines": [], "error": str(e)}


@router.get("/api/logs", dependencies=[_auth])
async def admin_logs(lines: int = Query(default=100, le=500)):
    """Return last N lines from the server log file.

    2026-10 fix (core2-M5): see admin_sessions' matching comment — reading
    (up to 500 lines of) a log file is real blocking disk I/O too.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _read_log_tail, lines)
    return JSONResponse(result)


# PDF review queue (feature/pdf-ingestion) — REMOVED 2026-10, this was a
# stale duplicate of router/admin_pdf.py's /api/pdf-queue* routes, left
# behind when that file was split out in 2026-09. The two implementations
# had already drifted: this copy never got decision_type/old_row_ref support
# for know-how items (added 2026-09 to knowhow_write_back.py's
# append_knowhow_topics, only ever wired up on the admin_pdf.py side) — a
# reviewer approving a know-how "update"/"duplicate" decision through
# whichever process happened to serve this copy would have silently always
# been treated as "new". app.py now includes admin_pdf_router directly
# (see app.py's own comment) instead of duplicating these routes here, so
# there is exactly one implementation again.

"""
Admin Dashboard Router — bot session monitoring only.

Split off from the old router/admin.py (2026-09) as part of separating the
production chat bot from the PDF ingestion pipeline into two independently
deployable services — a PDF-side dependency crash (see the boto3 incident,
2026-09-06/07) must never be able to take this service down again. The PDF
review-queue endpoints live in router/admin_pdf.py now, served by a separate
process/port; this file keeps only the routes the bot service itself needs.

static/admin.html is still served from here (the combined dashboard shell) —
its PDF-related fetch calls are pointed at the pdf-admin service's own port
via a JS constant, so the same single page still shows both Sessions and PDF
Queue tabs even though two different backends now serve them.
"""

import os
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse

from model.state_manager import StateManager
from utils.admin_auth import require_admin_key
from utils.simple_cache import get_cache

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
    """List all sessions with message count and preview."""
    sessions = _state_manager.list_sessions(limit=limit)

    result = []
    for s in sessions:
        sid = s["session_id"]
        state = _state_manager.load(sid)
        messages = state.messages if state else []

        user_msgs = [m for m in messages if m.get("role") == "user"]
        bot_msgs  = [m for m in messages if m.get("role") == "assistant"]

        prompt_tokens     = getattr(state, "total_prompt_tokens", 0) or 0
        completion_tokens = getattr(state, "total_completion_tokens", 0) or 0
        total_tokens      = prompt_tokens + completion_tokens
        # Real accumulated cost (see model/conversation_state.py's total_cost field) —
        # NOT re-estimated from token counts. The old code here re-derived cost from
        # total_prompt_tokens/total_completion_tokens (which blend every model used in
        # the session — Sonnet/GPT-5.1 AND cheap Haiku classifier calls) priced at a
        # single assumed model for the whole persona. That overcounted every session
        # (Haiku tokens billed at Sonnet/GPT-5.1 rates) and carried its own separate,
        # already-stale pricing table. total_cost is accumulated per-call at the
        # correct model's rate at the source (llm_call.py) and is not subject to
        # either problem.
        total_cost        = round(getattr(state, "total_cost", 0.0) or 0.0, 6)

        result.append({
            "session_id": sid,
            "persona_id": s.get("persona_id", "practical"),
            "preview": s.get("preview", ""),
            "updated_at": s.get("updated_at"),
            "total_messages": len(messages),
            "user_messages": len(user_msgs),
            "bot_messages": len(bot_msgs),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        })

    return JSONResponse({"sessions": result, "total": len(result)})


@router.get("/api/session/{session_id}", dependencies=[_auth])
async def admin_session_detail(session_id: str):
    """Full message history for a session."""
    state = _state_manager.load(session_id)
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
    """Overall stats: session count, cache, log summary."""
    all_sessions = _state_manager.list_sessions(limit=500)

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


@router.get("/api/logs", dependencies=[_auth])
async def admin_logs(lines: int = Query(default=100, le=500)):
    """Return last N lines from the server log file."""
    if not _LOG_FILE.exists():
        # Try relative path (when running from code/ dir)
        alt = Path(__file__).resolve().parent.parent.parent / "uvicorn.log"
        if alt.exists():
            log_path = alt
        else:
            return JSONResponse({"lines": [], "error": f"Log file not found: {_LOG_FILE}"})
    else:
        log_path = _LOG_FILE

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:]
        return JSONResponse({"lines": [l.rstrip() for l in tail], "total_lines": len(all_lines)})
    except Exception as e:
        return JSONResponse({"lines": [], "error": str(e)})

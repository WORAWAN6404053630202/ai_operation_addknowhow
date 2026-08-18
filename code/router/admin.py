"""
Admin Dashboard Router
Endpoints for monitoring bot usage, sessions, and logs.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

try:
    import conf as _conf
except Exception:
    _conf = None

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from model.state_manager import StateManager
from model.pdf_review_queue_manager import PdfReviewQueueManager
from utils.logger import get_logger
from utils.simple_cache import get_cache

router = APIRouter(prefix="/admin", tags=["admin"])
_LOG = get_logger(__name__)

_state_manager = StateManager()
_pdf_queue_manager = PdfReviewQueueManager()

# Path to uvicorn log file (adjust if needed via env)
_LOG_FILE = Path(os.getenv("LOG_FILE", str(Path(__file__).resolve().parent.parent.parent / "uvicorn.log")))

# Pricing per million tokens (USD) — keep in sync with llm_call.py PRICING_USD_PER_MILLION_TOKENS
_PRICING = {
    "anthropic/claude-sonnet-4-5":             {"input": 3.00,  "output": 15.00},
    "anthropic/claude-sonnet-4":               {"input": 3.00,  "output": 15.00},
    "anthropic/claude-haiku-4-5":              {"input": 0.80,  "output": 4.00},
    "anthropic/claude-haiku-4":                {"input": 0.25,  "output": 1.25},
    "anthropic/claude-3.5-haiku-20241022":     {"input": 0.25,  "output": 1.25},
    "openai/gpt-5.1":                          {"input": 2.00,  "output": 8.00},
    "openai/gpt-4o":                           {"input": 5.00,  "output": 15.00},
}

def _estimate_cost(prompt_tokens: int, completion_tokens: int, model: str = "") -> float:
    p = _PRICING.get(model, {"input": 2.0, "output": 8.0})  # fallback average
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1_000_000


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard():
    html_path = Path(__file__).resolve().parent.parent / "static" / "admin.html"
    if not html_path.exists():
        return HTMLResponse("<h1>admin.html not found</h1>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@router.get("/api/sessions")
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
        _persona          = s.get("persona_id", "practical")
        _model_for_cost   = {
            "academic":  getattr(_conf, "OPENROUTER_MODEL_ACADEMIC",  "") if _conf else "",
            "practical": getattr(_conf, "OPENROUTER_MODEL_PRACTICAL", "") if _conf else "",
        }.get(_persona, "")
        cost_usd          = _estimate_cost(prompt_tokens, completion_tokens, model=_model_for_cost)

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
            "cost_usd": round(cost_usd, 6),
        })

    return JSONResponse({"sessions": result, "total": len(result)})


@router.get("/api/session/{session_id}")
async def admin_session_detail(session_id: str):
    """Full message history for a session."""
    state = _state_manager.load(session_id)
    if state is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    messages = state.messages or []
    context  = state.context or {}

    prompt_tokens     = getattr(state, "total_prompt_tokens", 0) or 0
    completion_tokens = getattr(state, "total_completion_tokens", 0) or 0
    _persona          = state.persona_id or "practical"
    _model_for_cost   = {
        "academic":  getattr(_conf, "OPENROUTER_MODEL_ACADEMIC",  "") if _conf else "",
        "practical": getattr(_conf, "OPENROUTER_MODEL_PRACTICAL", "") if _conf else "",
    }.get(_persona, "")

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
        "cost_usd": round(_estimate_cost(prompt_tokens, completion_tokens, model=_model_for_cost), 6),
    })


@router.get("/api/stats")
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


@router.get("/api/logs")
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


# ── PDF review queue (feature/pdf-ingestion) ─────────────────────────────
# NOTE: unauthenticated, matching every other endpoint in this router as it
# exists on this branch — admin_auth.py (from wip/bot-behavior-fixes) is a
# separate, not-yet-merged piece of work and out of scope here.

class PdfReviewDecision(BaseModel):
    review_status: str  # "approved" | "rejected"
    decision_type: Optional[str] = None  # "new" | "duplicate" | "update" | "new_category" — required when approving
    reviewer_id: Optional[str] = None
    reviewer_notes: Optional[str] = None
    old_row_ref: Optional[str] = None  # decision_type == "update": which existing Sheet row this supersedes
    edited_fields: Optional[dict[str, str]] = None  # reviewer's corrections to llm_drafted_fields, if any —
    # what actually gets written to the Sheet is these values (merged over the original draft),
    # never the raw LLM draft blindly — the whole point of the review step.


@router.get("/api/pdf-queue")
async def pdf_queue_list():
    """Summary list for the left-hand panel — newest upload first."""
    items = _pdf_queue_manager.list_all()
    items.sort(key=lambda i: i.uploaded_at, reverse=True)
    return JSONResponse({
        "items": [
            {
                "id": i.id,
                "filename": i.filename,
                "uploaded_at": i.uploaded_at,
                "extraction_completed_at": i.extraction_completed_at,
                "review_status": i.review_status,
                "decision_type": i.decision_type,
                "page_count": len(i.pages),
                "total_flag_count": i.total_flag_count,
                "high_severity_flag_count": i.high_severity_flag_count,
                "needs_review": i.needs_review,
            }
            for i in items
        ],
        "total": len(items),
    })


@router.get("/api/pdf-queue/{item_id}")
async def pdf_queue_detail(item_id: str):
    """Full record — every page's Typhoon/Claude text and flags, for the review UI."""
    item = _pdf_queue_manager.load(item_id)
    if item is None:
        return JSONResponse({"error": "Item not found"}, status_code=404)
    return JSONResponse(item.model_dump())


@router.post("/api/pdf-queue/{item_id}/decision")
async def pdf_queue_decide(item_id: str, decision: PdfReviewDecision):
    """Records a human review decision. For review_status="approved" with
    decision_type in (new, update, new_category), this WRITES a new row to the
    real (additive-only, never edited/deleted) Google Sheet before saving the
    decision — if that write fails, the item is left "pending" and the error is
    returned, rather than silently recording an "approved" status that never
    actually reached the Sheet. decision_type="duplicate" and "rejected" never
    touch the Sheet at all — nothing new to write."""
    item = _pdf_queue_manager.load(item_id)
    if item is None:
        return JSONResponse({"error": "Item not found"}, status_code=404)

    if decision.review_status not in ("approved", "rejected"):
        return JSONResponse({"error": "review_status must be 'approved' or 'rejected'"}, status_code=400)
    if decision.review_status == "approved" and not decision.decision_type:
        return JSONResponse({"error": "decision_type is required when approving"}, status_code=400)
    if decision.decision_type not in (None, "new", "duplicate", "update", "new_category"):
        return JSONResponse({"error": "invalid decision_type"}, status_code=400)

    item.review_status = decision.review_status
    item.decision_type = decision.decision_type
    item.reviewer_id = decision.reviewer_id
    item.reviewer_notes = decision.reviewer_notes
    item.old_row_ref = decision.old_row_ref
    item.reviewed_at = time.time()

    # Reviewer's corrections win over the raw LLM draft — this is the actual
    # review step, not a formality. Merge (not replace) so any field the
    # reviewer left untouched in the UI still has its drafted value.
    if decision.edited_fields:
        item.llm_drafted_fields = {**(item.llm_drafted_fields or {}), **decision.edited_fields}

    sheet_result = None
    if decision.review_status == "approved" and decision.decision_type != "duplicate":
        try:
            from service.sheet_write_back import append_review_item_to_sheet

            sheet_result = append_review_item_to_sheet(item)
        except Exception as e:
            _LOG.error(f"[pdf_queue_decide] Sheet write-back failed for item {item_id}: {e}")
            return JSONResponse(
                {"error": f"Sheet write-back failed, decision NOT saved: {e}"}, status_code=502
            )

    _pdf_queue_manager.save(item)
    return JSONResponse({"ok": True, "item": item.model_dump(), "sheet_result": sheet_result})

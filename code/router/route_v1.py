"""
FastAPI Router
API endpoints for Thai Regulatory AI
"""

import asyncio
import datetime
import json
import logging
import queue
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from adapter.response.response_custom import HandleSuccess
from model.conversation_state import ConversationState
from model.state_manager import StateManager
from model.persona_supervisor import PersonaSupervisor
from utils.simple_cache import get_cache
from utils.rate_limiter import get_rate_limiter
from utils.progress import set_progress_callback, clear_progress_callback

import conf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_v1 = APIRouter()

SESSION_RETENTION_DAYS = 7


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=5000, description="User message to chatbot")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    # Always sent by the caller (can be ""). A non-empty value updates the session's
    # stored branch identity; "" leaves whatever was captured on a prior turn untouched
    # (see _merge_branch_fields in chat()/_stream_reply()).
    branch_id: str = Field(default="", description="Branch identifier of the caller (may be empty string)")
    branch_name: str = Field(default="", description="Branch display name of the caller (may be empty string)")


class SessionRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID")


class NewSessionRequest(BaseModel):
    branch_id: str = Field(default="", description="Branch identifier of the caller (may be empty string)")
    branch_name: str = Field(default="", description="Branch display name of the caller (may be empty string)")
    persona_id: str = Field(default="practical", description="practical or academic")


logger.info("Initializing services...")

try:
    conf.validate_config()
    if conf.USE_ZILLIZ:
        from service.vector_store import VectorStoreManager
        vs_manager = VectorStoreManager()
        retriever = vs_manager.connect_to_existing()
        logger.info("Using Milvus/Zilliz retriever")
    else:
        from service.local_vector_store import get_retriever
        retriever = get_retriever(fail_if_empty=False)
        logger.info("Using local Chroma retriever")

    supervisor = PersonaSupervisor(retriever=retriever)
    state_manager = StateManager()

    try:
        _cleaned = state_manager.cleanup_orphan_locks()
        if _cleaned:
            logger.info("Startup: removed %d orphan lock file(s)", _cleaned)
    except Exception:
        logger.warning("Startup: cleanup_orphan_locks failed", exc_info=True)

    logger.info("Services initialized successfully")

    # Reranker preload lives in app.py's lifespan hook only (was duplicated here too —
    # both called _get_reranker() for the same model, racing at startup; harmless under
    # normal single-process boot since _get_reranker double-checks its cache, but wasteful
    # and confusing to have two independent trigger points for the same one-time load).

except Exception:
    logger.error("Failed to initialize services", exc_info=True)
    supervisor = None
    state_manager = None
    raise


def start_prewarm_threads() -> None:
    """Starts the background cache-warming threads (topic pool, BM25 index, op-group
    classifier). Must be called from app.py's lifespan startup hook — NOT run at
    module import time like the rest of this file's init. uvicorn's --reload does an
    throwaway import of the app module to validate it before forking the real worker
    process, so anything with paid side effects at module level (op-group pre-warm
    fires real Haiku LLM calls, one per license_type/entity combo) ran twice per dev
    server start — once in that validation import, once for real in the worker
    (confirmed live: 16 distinct op_group_classify calls each appearing exactly twice,
    ~$0.035 wasted per restart). Matches the pattern already used for the reranker
    preload in app.py's lifespan, which does not have this problem.
    """
    import threading as _th

    # Pre-warm topic pool in background — populates global cross-session cache so the
    # first user greeting doesn't pay the 5-retrieval build cost (~1-2s).
    def _prewarm_topic_pool():
        try:
            _dummy = ConversationState(session_id="__prewarm__", persona_id="practical", context={})
            supervisor._build_topic_pool_from_corpus(_dummy)
            logger.info("Topic pool pre-warmed: %d items", len(_dummy.context.get("topic_pool") or []))
        except Exception as _tp_e:
            logger.warning("Topic pool pre-warm failed (non-blocking): %s", _tp_e)
    _th.Thread(target=_prewarm_topic_pool, daemon=True, name="topic_pool_prewarm").start()

    # Pre-warm BM25 index — builds tokenized corpus index so the first
    # hybrid_scored_search() call doesn't pay the PyThaiNLP tokenization cost (~0.3-0.5s).
    # topic_pool_prewarm uses retriever.invoke() (pure vector) and does NOT trigger this.
    def _prewarm_bm25():
        try:
            from utils.hybrid_retriever import _get_bm25_index
            from service.local_vector_store import get_vs_manager
            _vs = get_vs_manager().vectorstore
            if _vs:
                _idx = _get_bm25_index(_vs)
                logger.info("BM25 index pre-warmed: %d docs", _idx.doc_count if _idx else 0)
            else:
                logger.warning("BM25 pre-warm skipped — vectorstore not ready")
        except Exception as _bm_e:
            logger.warning("BM25 pre-warm failed (non-blocking): %s", _bm_e)
    _th.Thread(target=_prewarm_bm25, daemon=True, name="bm25_prewarm").start()

    # Pre-warm op-group classifier cache — so the first user who asks about
    # แก้ไข/ต่ออายุ/ยกเลิก doesn't pay the Haiku LLM call cost (~3-5s).
    # Iterates every license_type found in Chroma × 3 entity variants.
    def _prewarm_op_groups():
        try:
            _vs = getattr(getattr(supervisor, "_practical", None), "retriever", None)
            _vs = getattr(_vs, "vectorstore", None)
            _coll = getattr(_vs, "_collection", None)
            if _coll is None:
                logger.warning("Op-group pre-warm skipped — vectorstore not ready")
                return
            _result = _coll.get(include=["metadatas"])
            _license_types = {
                (md.get("license_type") or "").strip()
                for md in (_result.get("metadatas") or [])
                if (md.get("license_type") or "").strip()
            }
            for _lt in sorted(_license_types):
                for _entity in ("นิติบุคคล", "บุคคลธรรมดา", ""):
                    supervisor._get_operation_groups_for_entity(_lt, _entity)
            logger.info("Op-group cache pre-warmed: %d license types", len(_license_types))
        except Exception as _og_e:
            logger.warning("Op-group pre-warm failed (non-blocking): %s", _og_e)
    _th.Thread(target=_prewarm_op_groups, daemon=True, name="op_group_prewarm").start()


_CLEANUP_INTERVAL_S = 3600  # run at most once per hour
_last_cleanup_ts: float = 0.0

# Per-session asyncio locks prevent concurrent load→handle→save races when the
# stream endpoint runs supervisor.handle() in a thread pool executor.
from typing import Dict
_session_locks: Dict[str, asyncio.Lock] = {}


def _client_identifier(http_request: Request) -> str:
    """Identifier for rate limiting — must NOT be something the caller can
    cheaply rotate (session_id is client-supplied and was previously used
    here, which let a caller bypass the limit by minting a new session_id
    per request). X-Forwarded-For's first hop is trusted only because this
    service sits behind a reverse proxy/load balancer that sets it; if
    deployed with the app port directly internet-facing, that header is
    itself attacker-controlled and this would need a trusted-proxy check.
    """
    xff = http_request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return http_request.client.host if http_request.client else "unknown"


def _get_session_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]


# state_manager.load/save/delete/list_sessions do real blocking disk I/O (file open,
# json parse/dump, os.rename, and — for save — a cross-process lock acquired via a
# busy-poll loop with time.sleep). FastAPI/uvicorn runs on a single-threaded asyncio
# event loop, so calling these directly inside an `async def` handler (unlike
# supervisor.handle(), which already correctly goes through run_in_executor) stalls
# the ENTIRE event loop for that duration — blocking every other concurrent session's
# request, not just this one. These wrappers move the exact same calls onto the
# executor thread pool; behavior/return values are unchanged, only which thread runs
# them.
async def _load_state_async(session_id: str) -> Optional[ConversationState]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, state_manager.load, session_id)


async def _save_state_async(session_id: str, state: ConversationState) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, state_manager.save, session_id, state)


async def _delete_state_async(session_id: str) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, state_manager.delete, session_id)


async def _list_sessions_async(limit: int = 20) -> list:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, state_manager.list_sessions, limit)


async def _cleanup_old_sessions_async() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _cleanup_old_sessions)


def _consume_pending_token_budget(state: ConversationState, session_id: str) -> bool:
    """Runs the deferred summarize/trim check (see llm_call.py's
    run_pending_token_budget_check) on a worker thread. Returns True if it actually
    ran, so the caller knows the state changed and needs re-saving."""
    ctx = state.context or {}
    if not ctx.get("_needs_token_budget_check"):
        return False
    ctx.pop("_needs_token_budget_check", None)
    from utils.llm_call import run_pending_token_budget_check
    run_pending_token_budget_check(state, label=f"TokenBudget/{session_id}", log=logger)
    return True


# asyncio.create_task() only keeps a weak reference to the task it returns — if
# nothing else references it, the task can be garbage-collected mid-run before it
# finishes (a documented asyncio pitfall). Hold a strong reference here for the
# deferred summarize/lock-release tasks below, since a GC'd task would leak the
# session lock forever (never released).
_background_tasks: set = set()


def _spawn_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _release_session_lock_after_deferred_work(
    session_id: str, lock: asyncio.Lock, state: ConversationState
) -> None:
    """Fire-and-forget task: finishes the deferred token-budget summarize/trim (an
    LLM call that can take 1-3s) in the background AFTER the turn's response has
    already been returned to the client, persists the result, then releases the
    per-session lock. The lock is deliberately held for this whole window so a
    second message arriving for the SAME session can't start mutating
    state.messages concurrently with this background summarize/trim — see the
    manual acquire/release (not `async with`) in chat() and _stream_reply()."""
    try:
        loop = asyncio.get_running_loop()
        changed = await loop.run_in_executor(None, _consume_pending_token_budget, state, session_id)
        if changed:
            await _save_state_async(session_id, state)
    except Exception:
        logger.warning("[%s] Deferred token-budget summarize failed (non-blocking)", session_id, exc_info=True)
    finally:
        lock.release()


def _merge_branch_fields(state: ConversationState, branch_id: str, branch_name: str) -> None:
    """The caller sends branch_id/branch_name on every /chat turn (may be "").
    A non-empty value updates the session's stored branch identity; "" is a no-op
    so a turn that doesn't carry it (e.g. a mid-conversation slot answer) never
    erases what an earlier turn already captured. Branch is conversation-scoped,
    not per-turn, so this is a merge, not a replace."""
    bid = (branch_id or "").strip()
    bname = (branch_name or "").strip()
    if bid:
        state.branch_id = bid
    if bname:
        state.branch_name = bname


def _cleanup_old_sessions():
    global _last_cleanup_ts
    import time
    now = time.time()
    if now - _last_cleanup_ts < _CLEANUP_INTERVAL_S:
        return
    _last_cleanup_ts = now
    try:
        state_manager.purge_older_than_days(SESSION_RETENTION_DAYS)
    except Exception:
        logger.warning("Session cleanup failed", exc_info=True)
    # Remove locks for sessions that are no longer in state_manager to prevent unbounded growth.
    try:
        active_ids: set = set(state_manager.list_session_ids()) if hasattr(state_manager, "list_session_ids") else set()
        stale_locks = [sid for sid in list(_session_locks) if sid not in active_ids]
        for sid in stale_locks:
            _session_locks.pop(sid, None)
        if stale_locks:
            logger.debug("Cleaned up %d stale session locks", len(stale_locks))
    except Exception:
        pass  # best-effort; locks are small objects


def _build_topics_from_state(state: ConversationState):
    topics_raw: list = (state.context or {}).get("last_menu_topics") or []
    descs_raw: list = (state.context or {}).get("last_menu_topic_descs") or []

    selected = topics_raw[:5]
    topics = [
        {
            "title": t,
            "description": descs_raw[i] if i < len(descs_raw) else f"ผมจะแนะนำ{t} ตั้งแต่ต้นจนจบ พร้อมเอกสารที่ต้องใช้ ให้คุณทำตามได้ง่ายที่สุดครับ",
        }
        for i, t in enumerate(selected)
    ]
    return topics


@api_v1.post("/greeting")
async def start_session(payload: Optional[NewSessionRequest] = None):
    if supervisor is None or state_manager is None:
        raise HTTPException(status_code=503, detail="Services not initialized")

    await _cleanup_old_sessions_async()

    persona_id = "practical"
    if payload and payload.persona_id in {"practical", "academic"}:
        persona_id = payload.persona_id

    session_id = f"s_{uuid.uuid4().hex[:8]}"
    branch_id = ((payload.branch_id if payload else "") or "").strip() or None
    branch_name = ((payload.branch_name if payload else "") or "").strip() or None
    state = ConversationState(
        session_id=session_id,
        persona_id=persona_id,
        context={},
        branch_id=branch_id,
        branch_name=branch_name,
    )

    try:
        loop = asyncio.get_running_loop()
        state, greeting_text = await loop.run_in_executor(None, supervisor.handle, state, "")
        await _save_state_async(session_id, state)
    except Exception:
        logger.error(f"[{session_id}] Greeting failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Greeting failed. Please try again.",
        )

    topics = _build_topics_from_state(state)

    return HandleSuccess(
        message="Session created",
        session_id=session_id,
        response=greeting_text,
        topics=topics,
        persona_id=persona_id,
        retention_days=SESSION_RETENTION_DAYS,
    )


@api_v1.post("/reset")
async def reset_session(request: SessionRequest):
    if supervisor is None or state_manager is None:
        raise HTTPException(status_code=503, detail="Services not initialized")

    await _cleanup_old_sessions_async()

    session_id = request.session_id or f"s_{uuid.uuid4().hex[:8]}"
    state = ConversationState(session_id=session_id, persona_id="practical", context={})

    try:
        loop = asyncio.get_running_loop()
        state, greeting_text = await loop.run_in_executor(None, supervisor.handle, state, "")
        await _save_state_async(session_id, state)
    except Exception:
        logger.error(f"[{session_id}] Reset failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Reset failed. Please try again.",
        )

    topics = _build_topics_from_state(state)

    return HandleSuccess(
        message="Session reset",
        session_id=session_id,
        response=greeting_text,
        topics=topics,
        retention_days=SESSION_RETENTION_DAYS,
    )


@api_v1.get("/sessions")
async def list_sessions():
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    await _cleanup_old_sessions_async()

    sessions = await _list_sessions_async(limit=20)
    return HandleSuccess(
        message="Sessions loaded",
        sessions=sessions,
        retention_days=SESSION_RETENTION_DAYS,
    )


@api_v1.post("/session/load")
async def load_session(request: SessionRequest):
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    state = await _load_state_async(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return HandleSuccess(
        message="Session loaded",
        session_id=state.session_id,
        persona_id=state.persona_id,
        messages=state.display_messages or [],
    )


@api_v1.post("/session/delete")
async def delete_session(request: SessionRequest):
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    await _delete_state_async(request.session_id)

    return HandleSuccess(
        message="Session deleted",
        session_id=request.session_id,
    )


@api_v1.get("/healthcheck")
async def health_check():
    cache = get_cache()
    cache_stats = cache.get_stats()
    
    rate_limiter = get_rate_limiter()
    rate_stats = rate_limiter.get_stats()
    
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "service": "Thai Regulatory AI - น้องสุดยอด",
        "version": "1.0.0",
        "supervisor_initialized": supervisor is not None,
        "state_manager_initialized": state_manager is not None,
        "use_zilliz": conf.USE_ZILLIZ,
        "collection_name": conf.COLLECTION_NAME,
        "session_retention_days": SESSION_RETENTION_DAYS,
        "cache": cache_stats,
        "rate_limit": rate_stats,
    }


@api_v1.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    if supervisor is None or state_manager is None:
        logger.error("Services not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services not initialized. Check server logs.",
        )

    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty",
        )

    await _cleanup_old_sessions_async()

    session_id = request.session_id or f"s_{uuid.uuid4().hex[:8]}"
    client_id = _client_identifier(http_request)

    # Rate limiting check — keyed by client IP, not session_id (session_id is
    # client-supplied and freely rotatable, see _client_identifier).
    rate_limiter = get_rate_limiter()
    allowed, rate_info = rate_limiter.is_allowed(client_id)

    if not allowed:
        logger.warning(f"[{session_id}] Rate limit exceeded - blocking request")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many requests. Please wait {rate_info['retry_after']} seconds.",
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(rate_info["reset_in"]),
                "Retry-After": str(rate_info["retry_after"])
            }
        )

    logger.info(f"[{session_id}] Rate limit OK - {rate_info['remaining']}/{rate_info['limit']} remaining")

    lock = _get_session_lock(session_id)
    await lock.acquire()
    _lock_deferred = False
    try:
        # Load state
        saved = await _load_state_async(session_id)
        state = saved if saved else ConversationState(session_id=session_id, persona_id="practical", context={})
        _merge_branch_fields(state, request.branch_id, request.branch_name)

        # Session-level token budget: monitor-only (no blocking).
        _pre_prompt = getattr(state, "total_prompt_tokens", 0) or 0
        _pre_completion = getattr(state, "total_completion_tokens", 0) or 0
        _pre_cost = getattr(state, "total_cost_usd", 0.0) or 0.0
        _pre_tokens = _pre_prompt + _pre_completion
        _session_budget = int(getattr(conf, "TOKEN_BUDGET_PER_SESSION", 0) or 0)
        if _session_budget > 0 and _pre_tokens >= _session_budget:
            logger.warning(
                "[%s] Session token budget alert: used=%d limit=%d (continuing — monitoring only)",
                session_id, _pre_tokens, _session_budget,
            )

        # Per-window token rate check (burst protection).
        _window_token_limit = int(getattr(conf, "TOKEN_RATE_LIMIT_PER_WINDOW", 0) or 0)
        if _window_token_limit > 0:
            _token_rate_ok, _window_tokens_used = rate_limiter.is_token_rate_allowed(client_id, _window_token_limit)
            if not _token_rate_ok:
                logger.warning(
                    "[%s] Token rate limit exceeded: window_tokens=%d limit=%d",
                    session_id, _window_tokens_used, _window_token_limit,
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Token rate limit exceeded. Please wait before sending more messages.",
                    headers={"Retry-After": "60"},
                )

        cache = get_cache()
        has_pending_slot = bool((state.context or {}).get("pending_slot"))
        _collected_slots = (state.context or {}).get("collected_slots") or {}
        # First-turn queries with no prior user messages and no collected slots
        # produce identical LLM context across sessions (same system prompt +
        # same greeting + same retrieved docs). Use a shared cache key so
        # different users asking the same first question share one cache entry.
        _prior_user_msgs = [
            m for m in (state.messages or [])
            if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "user"
        ]
        _is_context_free = not _collected_slots and not _prior_user_msgs and not has_pending_slot
        _cache_session_id = "__shared__" if _is_context_free else session_id
        cached_result = (
            None
            if has_pending_slot
            else cache.get(_cache_session_id, request.message, state.persona_id, _collected_slots)
        )

        if cached_result is not None:
            logger.info(f"[{session_id}] Cache HIT! Skipping LLM call (saved ${cached_result.get('cost', 0):.3f})")
            state.add_user_message_once(request.message)
            state.add_assistant_message_once(cached_result["response"])
            await _save_state_async(session_id, state)
            return HandleSuccess(
                message="Chat completed (cached)",
                response=cached_result["response"],
                session_id=session_id,
                persona_id=state.persona_id,
                cached=True,
                cache_stats=cache.get_stats()
            )

        logger.info(f"[{session_id}] Cache MISS - Calling LLM")
        loop = asyncio.get_running_loop()
        state, bot_reply = await loop.run_in_executor(None, supervisor.handle, state, request.message)
        await _save_state_async(session_id, state)

        # Record token delta and cache inside the lock — prevents stale reads
        # when two requests for the same session overlap.
        _post_tokens = (getattr(state, "total_prompt_tokens", 0) or 0) + (getattr(state, "total_completion_tokens", 0) or 0)
        _delta_tokens = max(0, _post_tokens - _pre_tokens)
        if _delta_tokens > 0:
            rate_limiter.record_token_usage(client_id, _delta_tokens)

        # Real cost delta, not reconstructed from token counts: a turn often mixes
        # one main-persona call (Sonnet/GPT-5.1) with several cheap Haiku classifier
        # calls, and re-pricing that blended token delta at a single assumed model
        # would misprice every Haiku token at the main model's (higher) rate. Each
        # LLM call already accumulates its own correctly-priced cost onto
        # state.total_cost_usd at the source (llm_call.py, where the real model for
        # THAT call is known) — so the delta here is already accurate as-is.
        _post_cost = getattr(state, "total_cost_usd", 0.0) or 0.0
        _actual_cost = max(0.0, _post_cost - _pre_cost)
        # Don't cache "no answer found" fallbacks — a transient retrieval/model
        # hiccup would otherwise poison the shared (__shared__) cache key and
        # serve the same wrong "not found" reply to every other user asking
        # this question until the entry expires. Reuses the same
        # state.context["_info_gap_detected"] flag the supervisor sets to gate its
        # feedback-menu detection (see persona_supervisor.py's _maybe_offer_feedback) —
        # already computed for this turn, no extra work needed here.
        _is_no_answer_reply = bool((state.context or {}).get("_info_gap_detected"))
        # Never cache a reply that leaves a pending_slot behind — the cache only stores
        # the reply text, not the state.context side-effects (pending_slot, main_menu_shown,
        # etc.) that a cache HIT skips by never calling supervisor.handle(). Caching such a
        # reply under the shared key poisons every other brand-new session's first turn with
        # a reply that looks identical but has no pending_slot, so a follow-up numeric menu
        # pick ("1"/"2") can't be resolved and falls through to the unknown-input fallback.
        _reply_has_pending_slot = bool((state.context or {}).get("pending_slot"))
        if not _is_no_answer_reply and not _reply_has_pending_slot:
            cache.set(
                session_id=_cache_session_id,
                question=request.message,
                value={
                    "response": bot_reply,
                    "cost": _actual_cost,
                    "persona": state.persona_id
                },
                persona=state.persona_id,
                collected_slots=_collected_slots,
            )

        response = HandleSuccess(
            message="Chat completed",
            response=bot_reply,
            session_id=session_id,
            persona_id=state.persona_id,
            cached=False,
            cache_stats=cache.get_stats()
        )
        # If the turn's answer call crossed the session token budget, the actual
        # summarize/trim work (an LLM call) was deferred (see llm_call.py's
        # run_pending_token_budget_check) — run it in the background now, AFTER
        # the response is ready, instead of adding its latency to this turn.
        if bool((state.context or {}).get("_needs_token_budget_check")):
            _lock_deferred = True
            _spawn_background(_release_session_lock_after_deferred_work(session_id, lock, state))
        return response

    except HTTPException:
        raise
    except Exception:
        logger.error(f"[{session_id}] Chat failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat failed. Please try again.",
        )
    finally:
        if not _lock_deferred:
            lock.release()


def _run_supervisor_with_progress(state, message, progress_cb):
    """Runs supervisor.handle() on the executor's worker thread with the progress
    callback set/cleared on that SAME thread — thread-locals set on the event-loop
    thread would never be visible inside the worker thread that actually runs the
    pipeline, so the set/clear must happen here, wrapping the call itself."""
    set_progress_callback(progress_cb)
    try:
        return supervisor.handle(state, message)
    finally:
        clear_progress_callback()


async def _stream_reply(
    session_id: str, message: str, branch_id: str = "", branch_name: str = "", client_id: str = ""
) -> AsyncGenerator[str, None]:
    """
    Generator ที่ส่งคำตอบทีละ chunk แบบ SSE (Server-Sent Events)
    Format: data: <json>\n\n
    Events:
      - {"type": "status", "text": "..."}  ← สถานะการทำงานจริงระหว่างรอ (ไม่ใช่การเดา)
      - {"type": "chunk", "text": "..."}   ← ตัวอักษรที่ทยอยส่ง
      - {"type": "done", "session_id": "...", "persona_id": "..."}  ← จบ
      - {"type": "error", "message": "..."}  ← กรณี error
    """
    if supervisor is None or state_manager is None:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Services not initialized'})}\n\n"
        return

    # Falls back to session_id only if a caller forgets to pass client_id —
    # the real chat_stream() route above always passes the request's IP.
    client_id = client_id or session_id

    # Compute phase: load → handle → save, serialized per session to prevent
    # concurrent requests from overwriting each other's state.
    _full_text: str = ""
    _stream_persona_id: str = "practical"
    _stream_cached: bool = False
    _stream_error: str = ""
    _stream_cost: float = 0.0

    lock = _get_session_lock(session_id)
    await lock.acquire()
    _lock_deferred = False
    state: Optional[ConversationState] = None
    try:
        saved = await _load_state_async(session_id)
        state = saved if saved else ConversationState(
            session_id=session_id, persona_id="practical", context={}
        )
        _merge_branch_fields(state, branch_id, branch_name)

        # Session-level token budget: monitor-only (no blocking).
        _pre_prompt_st = getattr(state, "total_prompt_tokens", 0) or 0
        _pre_completion_st = getattr(state, "total_completion_tokens", 0) or 0
        _pre_cost_st = getattr(state, "total_cost_usd", 0.0) or 0.0
        _pre_tokens_st = _pre_prompt_st + _pre_completion_st
        _session_budget_st = int(getattr(conf, "TOKEN_BUDGET_PER_SESSION", 0) or 0)
        if _session_budget_st > 0 and _pre_tokens_st >= _session_budget_st:
            logger.warning(
                "[%s] Session token budget alert (stream): used=%d limit=%d (continuing — monitoring only)",
                session_id, _pre_tokens_st, _session_budget_st,
            )

        # Per-window token rate check (burst protection).
        _window_token_limit_st = int(getattr(conf, "TOKEN_RATE_LIMIT_PER_WINDOW", 0) or 0)
        if _window_token_limit_st > 0:
            _rl_st = get_rate_limiter()
            _token_ok_st, _window_used_st = _rl_st.is_token_rate_allowed(client_id, _window_token_limit_st)
            if not _token_ok_st:
                logger.warning(
                    "[%s] Token rate limit exceeded (stream): window_tokens=%d limit=%d",
                    session_id, _window_used_st, _window_token_limit_st,
                )
                _stream_error = "Token rate limit exceeded. Please wait before sending more messages."

        if not _stream_error:
            # Skip cache only when a slot question is pending (mid-collection).
            # collected_slots is included in the cache key so different slot combos
            # never share a cache entry — no need to skip cache when slots are present.
            # First-turn + no-slot queries use a shared key so different users
            # asking the same question share one cache entry (see /chat endpoint).
            cache = get_cache()
            has_pending_slot = bool((state.context or {}).get("pending_slot"))
            _collected_slots = (state.context or {}).get("collected_slots") or {}
            _prior_user_msgs_st = [
                m for m in (state.messages or [])
                if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "user"
            ]
            _is_context_free_st = not _collected_slots and not _prior_user_msgs_st and not has_pending_slot
            _cache_session_id_st = "__shared__" if _is_context_free_st else session_id
            cached_result = (
                None
                if has_pending_slot
                else cache.get(_cache_session_id_st, message, state.persona_id, _collected_slots)
            )

            if cached_result is not None:
                logger.info(f"[{session_id}] Cache HIT (stream)")
                _full_text = cached_result["response"]
                state.add_user_message_once(message)
                state.add_assistant_message_once(_full_text)
                await _save_state_async(session_id, state)
                _stream_persona_id = state.persona_id
                _stream_cached = True
            else:
                logger.info(f"[{session_id}] Cache MISS (stream) - Calling LLM")
                loop = asyncio.get_running_loop()
                progress_q: "queue.Queue[str]" = queue.Queue()
                future = loop.run_in_executor(
                    None, _run_supervisor_with_progress, state, message,
                    lambda t: progress_q.put(t),
                )
                while not future.done():
                    try:
                        _progress_text = progress_q.get_nowait()
                        yield f"data: {json.dumps({'type': 'status', 'text': _progress_text}, separators=(',', ':'))}\n\n"
                    except queue.Empty:
                        await asyncio.sleep(0.15)
                while True:
                    try:
                        _progress_text = progress_q.get_nowait()
                        yield f"data: {json.dumps({'type': 'status', 'text': _progress_text}, separators=(',', ':'))}\n\n"
                    except queue.Empty:
                        break
                state, bot_reply = future.result()
                await _save_state_async(session_id, state)

                # Token delta and cost — computed inside lock while state is still valid
                _post_tokens_st = (getattr(state, "total_prompt_tokens", 0) or 0) + (getattr(state, "total_completion_tokens", 0) or 0)
                _delta_tokens_st = max(0, _post_tokens_st - _pre_tokens_st)
                if _delta_tokens_st > 0:
                    get_rate_limiter().record_token_usage(client_id, _delta_tokens_st)

                # Real cost delta — see matching comment in the non-streaming /chat
                # handler above (a turn often mixes one main-persona call with several
                # cheap Haiku classifier calls; each already accumulates its own
                # correctly-priced cost onto state.total_cost_usd at the source).
                _post_cost_st = getattr(state, "total_cost_usd", 0.0) or 0.0
                _stream_cost = max(0.0, _post_cost_st - _pre_cost_st)

                # See matching comment in the non-streaming /chat handler above —
                # never cache a "no answer found" fallback under the shared key.
                _is_no_answer_reply_st = bool((state.context or {}).get("_info_gap_detected"))
                # See matching comment in the non-streaming /chat handler above — never
                # cache a reply that leaves a pending_slot behind (cache HIT skips
                # supervisor.handle() entirely, so the pending_slot side-effect is lost).
                _reply_has_pending_slot_st = bool((state.context or {}).get("pending_slot"))
                if not _is_no_answer_reply_st and not _reply_has_pending_slot_st:
                    cache.set(
                        session_id=_cache_session_id_st,
                        question=message,
                        value={"response": bot_reply, "cost": _stream_cost, "persona": state.persona_id},
                        persona=state.persona_id,
                        collected_slots=_collected_slots,
                    )

                _full_text = bot_reply
                _stream_persona_id = state.persona_id
                _stream_cached = False

    except Exception:
        logger.error(f"[{session_id}] Stream failed", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': 'Stream failed. Please try again.'})}\n\n"
        return
    finally:
        # If the turn's answer call crossed the session token budget, run the
        # deferred summarize/trim (see llm_call.py's run_pending_token_budget_check)
        # in the background, holding the lock until it's done, instead of adding
        # its latency here. See _release_session_lock_after_deferred_work.
        _needs_defer_st = False
        try:
            _needs_defer_st = bool(state is not None and (state.context or {}).get("_needs_token_budget_check"))
        except Exception:
            pass
        if _needs_defer_st:
            _lock_deferred = True
            _spawn_background(_release_session_lock_after_deferred_work(session_id, lock, state))
        if not _lock_deferred:
            lock.release()
    # lock released (or handed off to the deferred task) — now stream the reply

    if _stream_error:
        yield f"data: {json.dumps({'type': 'error', 'message': _stream_error})}\n\n"
        return

    chunk_size = 20
    for i in range(0, len(_full_text), chunk_size):
        chunk = _full_text[i:i + chunk_size]
        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk}, separators=(',', ':'))}\n\n"
        await asyncio.sleep(0.008 if not _stream_cached else 0.003)

    yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'persona_id': _stream_persona_id, 'cached': _stream_cached}, separators=(',', ':'))}\n\n"


@api_v1.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    """
    Streaming version ของ /chat
    ส่งคำตอบทีละ chunk แบบ SSE ทำให้ user เห็นข้อความทยอยขึ้น
    ไม่ต้องรอจนครบก่อนแสดง
    """
    if supervisor is None or state_manager is None:
        raise HTTPException(status_code=503, detail="Services not initialized")

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    await _cleanup_old_sessions_async()

    session_id = request.session_id or f"s_{uuid.uuid4().hex[:8]}"
    client_id = _client_identifier(http_request)

    # Rate limiting — keyed by client IP, see _client_identifier.
    rate_limiter = get_rate_limiter()
    allowed, rate_info = rate_limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many requests. Please wait {rate_info['retry_after']} seconds.",
        )

    return StreamingResponse(
        _stream_reply(session_id, request.message.strip(), request.branch_id, request.branch_name, client_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # บอก nginx ไม่ให้ buffer
        },
    )
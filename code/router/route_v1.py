"""
FastAPI Router
API endpoints for Thai Regulatory AI
"""

import asyncio
import datetime
import json
import logging
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from adapter.response.response_custom import HandleSuccess
from model.conversation_state import ConversationState
from model.state_manager import StateManager
from model.persona_supervisor import PersonaSupervisor
from utils.simple_cache import get_cache
from utils.rate_limiter import get_rate_limiter
from utils.llm_call import estimate_cost

import conf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_v1 = APIRouter()

SESSION_RETENTION_DAYS = 7


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=5000, description="User message to chatbot")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class SessionRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID")


class NewSessionRequest(BaseModel):
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

except Exception:
    logger.error("Failed to initialize services", exc_info=True)
    supervisor = None
    state_manager = None
    raise


_CLEANUP_INTERVAL_S = 3600  # run at most once per hour
_last_cleanup_ts: float = 0.0


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

    _cleanup_old_sessions()

    persona_id = "practical"
    if payload and payload.persona_id in {"practical", "academic"}:
        persona_id = payload.persona_id

    session_id = f"s_{uuid.uuid4().hex[:8]}"
    state = ConversationState(session_id=session_id, persona_id=persona_id, context={})

    try:
        state, greeting_text = supervisor.handle(state, "")
        state_manager.save(session_id, state)
    except Exception as e:
        logger.error(f"[{session_id}] Greeting failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Greeting failed: {str(e)}",
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

    _cleanup_old_sessions()

    session_id = request.session_id or f"s_{uuid.uuid4().hex[:8]}"
    state = ConversationState(session_id=session_id, persona_id="practical", context={})

    try:
        state, greeting_text = supervisor.handle(state, "")
        state_manager.save(session_id, state)
    except Exception as e:
        logger.error(f"[{session_id}] Reset failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reset failed: {str(e)}",
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

    _cleanup_old_sessions()

    sessions = state_manager.list_sessions(limit=20)
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

    state = state_manager.load(request.session_id)
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

    state_manager.delete(request.session_id)

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
async def chat(request: ChatRequest):
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

    _cleanup_old_sessions()

    session_id = request.session_id or f"s_{uuid.uuid4().hex[:8]}"
    
    # Rate limiting check
    rate_limiter = get_rate_limiter()
    allowed, rate_info = rate_limiter.is_allowed(session_id)
    
    if not allowed:
        logger.warning(f"[{session_id}] 🚫 Rate limit exceeded - blocking request")
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

    try:
        # Load state
        saved = state_manager.load(session_id)
        state = saved if saved else ConversationState(session_id=session_id, persona_id="practical", context={})

        # Session-level token budget check.
        # Session-level token budget: monitor-only (no blocking).
        # Logs a warning when the session exceeds TOKEN_BUDGET_PER_SESSION so the
        # operator can identify high-usage sessions without interrupting the user.
        _pre_prompt = getattr(state, "total_prompt_tokens", 0) or 0
        _pre_completion = getattr(state, "total_completion_tokens", 0) or 0
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
            _token_rate_ok, _window_tokens_used = rate_limiter.is_token_rate_allowed(session_id, _window_token_limit)
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

        # Skip cache if pending_slot (stateful slot-filling) or if slot-sensitive values
        # are collected. entity_type and registration_type affect answer content, so a
        # cached response from before those slots were filled must not be reused.
        cache = get_cache()
        has_pending_slot = bool((state.context or {}).get("pending_slot"))
        _CACHE_SKIP_SLOTS = {"entity_type", "registration_type", "location", "area_size"}
        _collected_slots = (state.context or {}).get("collected_slots") or {}
        has_slot_context = bool(_CACHE_SKIP_SLOTS & set(_collected_slots.keys()))
        cached_result = (
            None
            if (has_pending_slot or has_slot_context)
            else cache.get(session_id, request.message, state.persona_id)
        )
        
        if cached_result is not None:
            logger.info(f"[{session_id}] Cache HIT! Skipping LLM call (saved ${cached_result.get('cost', 0):.3f})")

            # Update state with cached message (but don't call LLM)
            # Use dedup helpers to avoid duplicate messages when same question asked repeatedly
            state.add_user_message_once(request.message)
            state.add_assistant_message_once(cached_result["response"])
            state_manager.save(session_id, state)
            
            return HandleSuccess(
                message="Chat completed (cached)",
                response=cached_result["response"],
                session_id=session_id,
                persona_id=state.persona_id,
                cached=True,
                cache_stats=cache.get_stats()
            )
        
        # Cache miss - call LLM
        logger.info(f"[{session_id}] Cache MISS - Calling LLM")
        state, bot_reply = supervisor.handle(state, request.message)
        state_manager.save(session_id, state)

        # Record token delta for window-level rate tracking.
        _post_tokens = (getattr(state, "total_prompt_tokens", 0) or 0) + (getattr(state, "total_completion_tokens", 0) or 0)
        _delta_tokens = max(0, _post_tokens - _pre_tokens)
        if _delta_tokens > 0:
            rate_limiter.record_token_usage(session_id, _delta_tokens)

        # Compute actual cost from token delta and persona-appropriate model
        _post_prompt = getattr(state, "total_prompt_tokens", 0) or 0
        _post_completion = getattr(state, "total_completion_tokens", 0) or 0
        _model_for_cost = {
            "academic": conf.OPENROUTER_MODEL_ACADEMIC,
            "practical": conf.OPENROUTER_MODEL_PRACTICAL,
        }.get(state.persona_id, conf.OPENROUTER_MODEL)
        _actual_cost = estimate_cost(
            _model_for_cost,
            max(0, _post_prompt - _pre_prompt),
            max(0, _post_completion - _pre_completion),
        )

        # Store in cache for future use
        cache.set(
            session_id=session_id,
            question=request.message,
            value={
                "response": bot_reply,
                "cost": _actual_cost,
                "persona": state.persona_id
            },
            persona=state.persona_id
        )

        return HandleSuccess(
            message="Chat completed",
            response=bot_reply,
            session_id=session_id,
            persona_id=state.persona_id,
            cached=False,
            cache_stats=cache.get_stats()
        )

    except Exception as e:
        logger.error(f"[{session_id}] Chat failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}",
        )


async def _stream_reply(session_id: str, message: str) -> AsyncGenerator[str, None]:
    """
    Generator ที่ส่งคำตอบทีละ chunk แบบ SSE (Server-Sent Events)
    Format: data: <json>\n\n
    Events:
      - {"type": "chunk", "text": "..."}   ← ตัวอักษรที่ทยอยส่ง
      - {"type": "done", "session_id": "...", "persona_id": "..."}  ← จบ
      - {"type": "error", "message": "..."}  ← กรณี error
    """
    if supervisor is None or state_manager is None:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Services not initialized'})}\n\n"
        return

    try:
        saved = state_manager.load(session_id)
        state = saved if saved else ConversationState(
            session_id=session_id, persona_id="practical", context={}
        )

        # Session-level token budget: monitor-only (no blocking).
        _pre_prompt_st = getattr(state, "total_prompt_tokens", 0) or 0
        _pre_completion_st = getattr(state, "total_completion_tokens", 0) or 0
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
            _token_ok_st, _window_used_st = _rl_st.is_token_rate_allowed(session_id, _window_token_limit_st)
            if not _token_ok_st:
                logger.warning(
                    "[%s] Token rate limit exceeded (stream): window_tokens=%d limit=%d",
                    session_id, _window_used_st, _window_token_limit_st,
                )
                yield f"data: {json.dumps({'type': 'error', 'message': 'Token rate limit exceeded. Please wait before sending more messages.'})}\n\n"
                return

        # Skip cache if pending_slot (stateful slot-filling) or if slot-sensitive values
        # are collected. entity_type and registration_type affect answer content, so a
        # cached response from before those slots were filled must not be reused.
        cache = get_cache()
        has_pending_slot = bool((state.context or {}).get("pending_slot"))
        _CACHE_SKIP_SLOTS = {"entity_type", "registration_type", "location", "area_size"}
        _collected_slots = (state.context or {}).get("collected_slots") or {}
        has_slot_context = bool(_CACHE_SKIP_SLOTS & set(_collected_slots.keys()))
        cached_result = (
            None
            if (has_pending_slot or has_slot_context)
            else cache.get(session_id, message, state.persona_id)
        )

        if cached_result is not None:
            # Cache hit → stream ตัวอักษรจาก cache ทีละ chunk เพื่อให้ดูเหมือน typewriter
            logger.info(f"[{session_id}] Cache HIT (stream)")
            full_text = cached_result["response"]
            # Use dedup helpers to avoid duplicate messages when same question asked repeatedly
            state.add_user_message_once(message)
            state.add_assistant_message_once(full_text)
            state_manager.save(session_id, state)

            # ส่งทีละ ~5 ตัวอักษร เพื่อให้ดู smooth
            chunk_size = 5
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i + chunk_size]
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                await asyncio.sleep(0.01)  # หน่วงเล็กน้อยให้เห็น effect

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'persona_id': state.persona_id, 'cached': True})}\n\n"
            return

        # Cache miss → เรียก LLM จริง (blocking แต่ stream ผลลัพธ์หลังได้คำตอบ)
        logger.info(f"[{session_id}] Cache MISS (stream) - Calling LLM")

        # เรียก supervisor ใน thread pool ไม่บล็อก event loop
        loop = asyncio.get_running_loop()
        state, bot_reply = await loop.run_in_executor(
            None, supervisor.handle, state, message
        )
        state_manager.save(session_id, state)

        # Record token delta for window-level rate tracking.
        _post_tokens_st = (getattr(state, "total_prompt_tokens", 0) or 0) + (getattr(state, "total_completion_tokens", 0) or 0)
        _delta_tokens_st = max(0, _post_tokens_st - _pre_tokens_st)
        if _delta_tokens_st > 0:
            get_rate_limiter().record_token_usage(session_id, _delta_tokens_st)

        # Compute actual cost from token delta and persona-appropriate model
        _post_prompt_st = getattr(state, "total_prompt_tokens", 0) or 0
        _post_completion_st = getattr(state, "total_completion_tokens", 0) or 0
        _model_for_cost_st = {
            "academic": conf.OPENROUTER_MODEL_ACADEMIC,
            "practical": conf.OPENROUTER_MODEL_PRACTICAL,
        }.get(state.persona_id, conf.OPENROUTER_MODEL)
        _actual_cost_st = estimate_cost(
            _model_for_cost_st,
            max(0, _post_prompt_st - _pre_prompt_st),
            max(0, _post_completion_st - _pre_completion_st),
        )

        # เก็บ cache
        cache.set(
            session_id=session_id,
            question=message,
            value={"response": bot_reply, "cost": _actual_cost_st, "persona": state.persona_id},
            persona=state.persona_id,
        )

        # Stream คำตอบทีละ chunk
        chunk_size = 5
        for i in range(0, len(bot_reply), chunk_size):
            chunk = bot_reply[i:i + chunk_size]
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
            await asyncio.sleep(0.008)

        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'persona_id': state.persona_id, 'cached': False})}\n\n"

    except Exception as e:
        logger.error(f"[{session_id}] Stream failed: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@api_v1.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming version ของ /chat
    ส่งคำตอบทีละ chunk แบบ SSE ทำให้ user เห็นข้อความทยอยขึ้น
    ไม่ต้องรอจนครบก่อนแสดง
    """
    if supervisor is None or state_manager is None:
        raise HTTPException(status_code=503, detail="Services not initialized")

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    _cleanup_old_sessions()

    session_id = request.session_id or f"s_{uuid.uuid4().hex[:8]}"

    # Rate limiting
    rate_limiter = get_rate_limiter()
    allowed, rate_info = rate_limiter.is_allowed(session_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many requests. Please wait {rate_info['retry_after']} seconds.",
        )

    return StreamingResponse(
        _stream_reply(session_id, request.message.strip()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # บอก nginx ไม่ให้ buffer
        },
    )
"""
llm_call.py — Enhanced LLM wrapper with comprehensive metrics tracking

Thin wrapper around LangChain LLM .invoke() that:
- Logs token usage and wall-clock time with structured data
- Records metrics for monitoring (cost, latency, tokens)
- Handles retries with exponential backoff
- Tracks per-persona usage
- AI Engineer metrics: cost estimation, performance tracking

Usage:
    from code.utils.llm_call import llm_invoke
    response = llm_invoke(
        llm, messages, 
        logger=_LOG, 
        label="Practical/answer", 
        state=state,
        persona="practical",
        operation="answer"
    )
    text = (response.content or "").strip()
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional

import httpx

_MAX_RETRIES = 2
_RETRY_DELAYS = (0.3, 1.0)  # exponential backoff in seconds

if TYPE_CHECKING:
    from code.model.conversation_state import ConversationState

_ROOT_LOG = logging.getLogger("restbiz.llm")

# Every persona/supervisor classifier builds its own ChatOpenAI(...) instance (~45
# call sites across persona_supervisor.py/persona_practical.py/persona_academic.py),
# each normally opening its own httpx connection pool to OpenRouter. Sharing one
# httpx.Client across all of them means concurrent calls (e.g. via _CLASSIFIER_POOL)
# reuse warm TCP/TLS connections instead of each paying a fresh handshake.
#
# Safe to share despite each ChatOpenAI configuring a different request_timeout
# (8s classifiers vs 90-120s main answer calls): openai-python applies `timeout`
# per-request in _build_request() (see openai._base_client.SyncAPIClient), which
# overrides whatever default timeout the shared httpx.Client itself was built
# with — so every call site keeps its own configured timeout unchanged.
#
# Limits match openai-python's own per-client default (openai._constants.
# DEFAULT_CONNECTION_LIMITS) so pooling behavior for any single call is unchanged;
# only the number of separate pools drops from ~45 to 1.
_shared_http_client: Optional[httpx.Client] = None
_shared_http_client_lock = threading.Lock()


def get_shared_http_client() -> httpx.Client:
    """Singleton httpx.Client to pass as `http_client=` to every ChatOpenAI(...)
    instance. Thread-safe via double-checked locking (same pattern as the reranker
    singleton in utils/reranker.py). Never explicitly closed — lives for the
    process lifetime, same as the per-instance clients it replaces."""
    global _shared_http_client
    if _shared_http_client is None:
        with _shared_http_client_lock:
            if _shared_http_client is None:
                _shared_http_client = httpx.Client(
                    limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
                )
    return _shared_http_client


def _is_length_error(exc: Exception) -> bool:
    """Return True when exc is a LengthFinishReasonError (token limit hit).
    Retrying with identical input always reproduces the same error, so callers
    should skip all retry attempts and re-raise immediately."""
    n = type(exc).__name__
    return "LengthFinishReasonError" in n or "LengthFinishReason" in str(exc)[:80]


def _extract_token_counts(response: Any, log: logging.Logger, label: str) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from a LangChain response object.
    Returns (0, 0) on any failure and emits a debug log so the gap is visible."""
    try:
        um = getattr(response, "usage_metadata", None) or {}
        if um:
            return (
                int(um.get("input_tokens") or um.get("prompt_tokens") or 0),
                int(um.get("output_tokens") or um.get("completion_tokens") or 0),
            )
        rm = getattr(response, "response_metadata", None) or {}
        tu = rm.get("token_usage") or rm.get("usage") or {}
        return (
            int(tu.get("prompt_tokens") or tu.get("input_tokens") or 0),
            int(tu.get("completion_tokens") or tu.get("output_tokens") or 0),
        )
    except Exception as _e_tok:
        log.debug("[%s] Token extraction failed (continuing with 0 tokens): %s", label, _e_tok)
        return 0, 0


def _extract_cache_tokens(response: Any) -> tuple[int, int]:
    """Extract (cache_read_tokens, cache_write_tokens) if present, else (0, 0).
    Kept as permanent lightweight observability — this is the only signal that shows
    whether build_cached_system_message's cache_control is actually being honored by
    the provider on a given call, without needing to re-add temporary debug logging.
    Also feeds estimate_cost() so logged/tracked cost reflects the real cache discount
    (~90% off reads, ~25% premium on writes for Anthropic) instead of charging every
    prompt token at the full uncached rate."""
    try:
        um = getattr(response, "usage_metadata", None) or {}
        itd = um.get("input_token_details") or {}
        rm = getattr(response, "response_metadata", None) or {}
        tu = rm.get("token_usage") or rm.get("usage") or {}
        ptd = tu.get("prompt_tokens_details") or {}
        cache_read = int(itd.get("cache_read") or ptd.get("cached_tokens") or 0)
        cache_write = int(ptd.get("cache_write_tokens") or 0)
        return cache_read, cache_write
    except Exception:
        return 0, 0


def _safe_log_with_data(log_obj: logging.Logger, level: str, message: str, payload: dict) -> None:
    """Structured-log when available, otherwise fallback to plain logging."""
    method = getattr(log_obj, "log_with_data", None)
    if callable(method):
        try:
            method(level, message, payload)
            return
        except Exception as exc:
            _ROOT_LOG.warning("Structured log failed (%s): %s", message, exc)

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    try:
        log_obj.log(numeric_level, "%s | %s", message, payload)
    except Exception:
        _ROOT_LOG.log(numeric_level, "%s", message)

# Import structured logger
try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    _STRUCTURED_LOG = True
except ImportError:
    logger = _ROOT_LOG
    _STRUCTURED_LOG = False

# Token Management: session-level cumulative token threshold.
# When session_total exceeds this value after an LLM call, auto-summarize or trim is triggered.
# The practical system prompt alone is ~15K tokens per call. A threshold of 20K fires on the
# first LLM call in a fresh session — useless and adds latency. Raised to 80K so that
# summarize/trim only kicks in after ~4-5 full Q&A rounds of real message accumulation.
_TOKEN_WARN_THRESHOLD = 80_000  # trigger summarize/trim when session total exceeds this


def _handle_post_call_token_budget(
    state: Any,
    prompt_tokens: int,
    completion_tokens: int,
    label: str,
    log: logging.Logger,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Track cumulative token usage. If the session budget is exceeded, this does NOT
    run summarize/trim inline anymore — that used to fire a blocking Haiku call
    (summarize_messages(), ~1-3s) directly in the critical path of whichever answer
    call happened to cross the threshold, adding to the user's wait time on that turn
    and (since total_tokens never decreases) on roughly every 2-3 turns for the rest
    of any long conversation.

    Instead this just flags state.context["_needs_token_budget_check"] = True — cheap,
    no LLM call. The caller (route_v1.py) runs run_pending_token_budget_check() below
    in a background thread AFTER the turn's response has already been built, while
    still holding that session's asyncio lock, so no concurrent turn for the same
    session can mutate state.messages while the deferred work runs. See
    route_v1.py's _release_session_lock_after_deferred_work.

    Shared by llm_invoke (sync) and llm_invoke_async (async) — no async operations here.
    Wrapped in a broad except so any failure is silent and never breaks the main call.
    """
    try:
        state.add_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost_usd,
        )
        session_total = getattr(state, "total_tokens", 0)
        if session_total <= _TOKEN_WARN_THRESHOLD:
            return

        log.warning(
            "[%s] Session token budget exceeded %d (total=%d) — flagging for deferred summarize",
            label, _TOKEN_WARN_THRESHOLD, session_total,
        )
        try:
            state.context = getattr(state, "context", None) or {}
            state.context["_needs_token_budget_check"] = True
        except Exception:
            pass
    except Exception:
        pass


def run_pending_token_budget_check(
    state: Any,
    label: str = "TokenBudget",
    log: Optional[logging.Logger] = None,
) -> None:
    """Runs the actual summarize-or-trim work that _handle_post_call_token_budget
    used to run inline. Identical logic/thresholds, just relocated.

    Safe to call from a background thread ONLY while the caller still holds the
    per-session lock for `state`'s session — this mutates state.messages wholesale
    (via auto_summarize_if_needed → state.summarize_old_messages, or trim_messages),
    which would race with a concurrent turn for the same session appending its own
    messages. route_v1.py enforces this by not releasing the session lock until
    this function returns.
    """
    log = log or _ROOT_LOG
    try:
        _academic_stage = (
            (getattr(state, "context", None) or {})
            .get("_academic_flow", {})
            .get("stage", "")
        )
        _skip_summarize = _academic_stage in ("awaiting_slots", "awaiting_sections", "awaiting_topic")

        try:
            from utils.conversation_summarizer import auto_summarize_if_needed

            if _skip_summarize:
                log.info("[%s] Academic stage=%r — skipping summarize, using trim only", label, _academic_stage)
                summarized = False
            else:
                summarized = auto_summarize_if_needed(
                    state,
                    threshold=6,   # summarize when 6+ messages (~3 Q&A turns)
                    keep_recent=4  # keep last 4 messages (2 Q&A turns) + summary
                )

            if summarized:
                log.info("[%s] Auto-summarized old messages → token reduced", label)
            else:
                log.info("[%s] Summarization not needed, using trim instead", label)
                if hasattr(state, "trim_messages"):
                    state.trim_messages(keep_last=4)
        except Exception as e:
            log.warning("[%s] Summarization failed: %s, falling back to trim", label, e)
            if hasattr(state, "trim_messages"):
                state.trim_messages(keep_last=4)
    except Exception:
        pass


# Import metrics if available
try:
    from utils.metrics import metrics
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _ROOT_LOG.warning("Metrics module not available, metrics collection disabled")

# Import config for budget thresholds
try:
    import conf
    _CONF_AVAILABLE = True
except ImportError:
    _CONF_AVAILABLE = False


def build_speed_extra_body(model_name: str) -> dict:
    """Extra OpenRouter request-body fields that reduce latency without changing what
    the model produces: provider routing (`provider.sort`) — send the request to
    whichever OpenRouter provider currently has the best throughput/latency for this
    model. Same model weights, same output; only the serving path changes.

    Prompt caching is NOT set here — a top-level `cache_control` in extra_body was
    tried and measured (see conversation notes): it never produced a single cache
    read (every call showed cache_write_tokens == input_tokens), and once the correct
    content-block-level cache_control (see build_cached_system_message) was added
    alongside it, the two conflicted outright — OpenRouter/Anthropic rejects a
    request with two cache_control breakpoints at different default TTLs ("ttl='1h'
    block must not come after a ttl='5m' block"). Caching is applied solely via
    build_cached_system_message on the actual system-message content block now.
    """
    extra: dict = {}
    if not _CONF_AVAILABLE:
        return extra
    _sort = getattr(conf, "PROVIDER_ROUTING_SORT", "")
    if _sort:
        extra["provider"] = {"sort": _sort}
    return extra


def build_thinking_extra_body(model_name: str, budget_tokens: int) -> dict:
    """OpenRouter `reasoning` field to enable Claude extended thinking, gated by
    OPENROUTER_PRACTICAL_THINKING_BUDGET (0 = disabled, untouched behavior).

    OpenRouter uses its own `reasoning: {enabled, max_tokens}` wrapper rather than
    Anthropic's native `thinking` field for models routed through it. Only applies
    to Anthropic models with budget_tokens > 0 configured.

    Caller must also force temperature=1.0 on the same call — Sonnet 4.5 rejects
    any other temperature with a 400 error when thinking is enabled (a hard
    Anthropic API constraint, not an OpenRouter quirk); see _init_llm in
    persona_practical.py for where that's applied.
    """
    if budget_tokens > 0 and "anthropic/" in (model_name or ""):
        return {"reasoning": {"enabled": True, "max_tokens": budget_tokens}}
    return {}


def build_cached_system_message(text: str, model_name: str) -> Any:
    """Build a SystemMessage with Anthropic prompt caching correctly applied.

    Measured directly against the real API (see conversation notes): the top-level
    `cache_control` in build_speed_extra_body's extra_body was NOT taking effect —
    every call showed cache_write_tokens == input_tokens and cached_tokens == 0,
    meaning we paid the 1.25x write premium on every single call and never once
    got the 90% read discount. Root cause: OpenRouter's OpenAI-wire chat_completions
    format only honors `cache_control` when it's attached to the actual system-message
    CONTENT BLOCK, not as a bare top-level request field. Fix: send the system prompt
    as a one-element content-block list with cache_control on that block instead of
    a plain string. Falls back to a plain SystemMessage for non-Anthropic models
    (OpenAI/GPT-5.1 already gets automatic caching with no request change needed).

    ttl="1h" (the extended cache window, vs Anthropic's 5-minute default when ttl is
    omitted) is set explicitly — this is the only cache_control in the request now
    (the old top-level one that used to conflict with this has been removed from
    build_speed_extra_body), so there's no risk of the "two breakpoints, different
    TTL" 400 error hit earlier while both existed simultaneously.
    """
    from langchain_core.messages import SystemMessage

    if _CONF_AVAILABLE and getattr(conf, "PROMPT_CACHING_ENABLED", False) and "anthropic/" in (model_name or ""):
        return SystemMessage(content=[{"type": "text", "text": text, "cache_control": {"type": "ephemeral", "ttl": "1h"}}])
    return SystemMessage(content=text)


def _check_token_budget(total: int, model: str) -> None:
    """Check token budget and log warnings with severity levels"""
    if not _CONF_AVAILABLE:
        return
    
    if total >= conf.TOKEN_BUDGET_CRITICAL:
        logger.error(
            "CRITICAL: Token budget exceeded",
            extra={
                "tokens": total,
                "threshold": conf.TOKEN_BUDGET_CRITICAL,
                "model": model,
                "severity": "critical",
                "detail": f"Token usage {total:,} exceeds CRITICAL threshold {conf.TOKEN_BUDGET_CRITICAL:,}!"
            }
        )
    elif total >= conf.TOKEN_BUDGET_WARNING:
        logger.warning(
            "WARNING: Token budget exceeded",
            extra={
                "tokens": total,
                "threshold": conf.TOKEN_BUDGET_WARNING,
                "target": conf.TOKEN_BUDGET_PER_CALL,
                "model": model,
                "severity": "warning",
                "detail": f"Token usage {total:,} exceeds WARNING threshold. Target: {conf.TOKEN_BUDGET_PER_CALL:,}"
            }
        )


# Cost Estimation (ราคาโดยประมาณ - ตรวจสอบราคาจริงจาก OpenRouter)
PRICING_USD_PER_MILLION_TOKENS = {
    # Claude Sonnet family
    "anthropic/claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "anthropic/claude-4-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    # Claude Haiku family
    "anthropic/claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    "anthropic/claude-haiku-4": {"input": 0.25, "output": 1.25},
    "anthropic/claude-3.5-haiku-20241022": {"input": 0.25, "output": 1.25},
    # GPT-5.x family (verify pricing at openrouter.ai/models)
    "openai/gpt-5.1": {"input": 2.00, "output": 8.00},
    # GPT-4o family
    "openai/gpt-4o": {"input": 5.00, "output": 15.00},
    "openai/chatgpt-4o-latest": {"input": 5.00, "output": 15.00},
}

_COST_LOG = logging.getLogger(__name__)


def _cache_multipliers(model_name: str) -> tuple[float, float]:
    """Returns (cache_read_multiplier, cache_write_multiplier), relative to a model's
    normal input price. Approximate, provider-level figures (not billing-grade — the
    pricing table above is itself already an estimate per its own comment):
      - Anthropic: cache reads ~0.1x (90% off), cache writes ~1.25x (25% premium).
      - Everyone else (e.g. OpenAI/GPT-5.1, which gets automatic caching with no
        request change needed): reads ~0.5x, no write premium — matches OpenAI's
        published "50% off cached input" behavior, distinct from Anthropic's scheme.
    """
    if "anthropic/" in (model_name or ""):
        return 0.1, 1.25
    return 0.5, 1.0


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """คำนวณค่าใช้จ่ายโดยประมาณ (USD).

    cache_read_tokens/cache_write_tokens are a SUBSET of prompt_tokens (not
    additional) — pass 0/0 (the default) for calls where caching doesn't apply or
    wasn't measured, which reduces to the original flat-rate calculation exactly.
    """
    pricing = PRICING_USD_PER_MILLION_TOKENS.get(model)
    if pricing is None:
        _COST_LOG.warning("[Cost] Model %r not in pricing table — cost logged as $0. Add it to PRICING_USD_PER_MILLION_TOKENS.", model)
        return 0.0
    _read_mult, _write_mult = _cache_multipliers(model)
    _regular_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    _input_price = pricing["input"]
    cost = (
        _regular_tokens * _input_price / 1_000_000
        + cache_read_tokens * _input_price * _read_mult / 1_000_000
        + cache_write_tokens * _input_price * _write_mult / 1_000_000
        + completion_tokens * pricing["output"] / 1_000_000
    )
    return cost


def extract_llm_text(response: Any) -> str:
    """
    Extract the final text content from an LLM response.

    Handles two formats:
    - Normal models: response.content is a plain string
    - Gemini 2.5 Flash (thinking mode): response.content is a list of blocks
      e.g. [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]

    Always returns a plain string (may be empty).
    """
    content = getattr(response, "content", None)

    # Plain string — the common case
    if isinstance(content, str):
        return content

    # Gemini thinking mode: content is a list of typed blocks
    if isinstance(content, list):
        # Prefer the first "text"-type block
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block.get("text") or "")
        # Fallback: first string element in the list
        for block in content:
            if isinstance(block, str):
                return block
        # Last resort: join all string values found
        parts = []
        for block in content:
            if isinstance(block, dict):
                v = block.get("text") or block.get("content") or block.get("thinking") or ""
                if v:
                    parts.append(str(v))
        return " ".join(parts)

    # None or unknown — return empty string
    return ""


def llm_invoke(
    llm: Any,
    messages: List[Any],
    *,
    logger: Optional[logging.Logger] = None,
    label: str = "LLM",
    state: Optional["ConversationState"] = None,
    persona: Optional[str] = None,  # "academic", "practical", "supervisor"
    operation: Optional[str] = None,  # "greet", "topic_picker", "answer", etc
) -> Any:
    """
    Call llm.invoke(messages), log elapsed time + token usage, and
    accumulate tokens into state.add_token_usage() if state is provided.

    Args:
        llm: LangChain LLM instance
        messages: List of messages to send
        logger: Logger instance for logging
        label: Label for log messages
        state: ConversationState for token tracking
        persona: Which persona is making the call (for metrics)
        operation: What operation is being performed (for metrics)

    Returns the raw LangChain response object (same as llm.invoke would).
    """
    log = logger or _ROOT_LOG
    t0 = time.perf_counter()
    last_exc: Optional[Exception] = None
    
    # Extract model name for metrics
    model_name = getattr(llm, "model", getattr(llm, "model_name", "unknown"))

    for attempt in range(1 + _MAX_RETRIES):
        try:
            response = llm.invoke(messages)
            break
        except Exception as exc:
            last_exc = exc
            elapsed = time.perf_counter() - t0
            log.warning("[%s] exception: %s — %s", label, type(exc).__name__, str(exc)[:300])

            # BUG-G fix: LengthFinishReasonError means the model hit the token limit.
            # Retrying with identical input always produces the same error — skip all retries.
            if _is_length_error(exc):
                log.warning(
                    "[%s] LengthFinishReasonError — skip retries (token limit, same input = same result)",
                    label,
                )
                raise

            # Record failed attempt in metrics
            if _METRICS_AVAILABLE:
                metrics.record_llm_call(
                    model=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    elapsed_ms=elapsed * 1000,
                    success=False,
                    error=str(exc)[:200],
                    persona=persona,
                    operation=operation
                )

            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAYS[attempt]
                log.warning(
                    "[%s] LLM call FAILED after %.2fs (attempt %d/%d) — retrying in %.0fs",
                    label, elapsed, attempt + 1, 1 + _MAX_RETRIES, delay,
                )
                time.sleep(delay)
            else:
                log.warning("[%s] LLM call FAILED after %.2fs (all %d attempts exhausted)", label, elapsed, 1 + _MAX_RETRIES)
                raise
    else:
        # loop exhausted without break (should not happen due to raise above)
        raise RuntimeError(f"[{label}] LLM call failed") from last_exc

    elapsed = time.perf_counter() - t0

    # Extract token counts via shared helper (logs debug on failure)
    prompt_tokens, completion_tokens = _extract_token_counts(response, log, label)
    cache_read_tokens, cache_write_tokens = _extract_cache_tokens(response)

    total_call = prompt_tokens + completion_tokens

    # Calculate cost (cache-aware — see estimate_cost/_cache_multipliers)
    cost_usd = estimate_cost(model_name, prompt_tokens, completion_tokens, cache_read_tokens, cache_write_tokens)

    # Check token budget and log warnings
    _check_token_budget(total_call, model_name)

    # Enhanced structured logging for AI Engineers
    if _STRUCTURED_LOG:
        _safe_log_with_data(log, "info", f"{label} สำเร็จ", {
            "action": "llm_call",
            "label": label,
            "model": model_name,
            "persona": persona,
            "operation": operation,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_call,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "duration_ms": round(elapsed * 1000, 2),
            "cost_usd": round(cost_usd, 6),
            "session_id": getattr(state, "session_id", None) if state else None,
            "temperature": getattr(llm, "temperature", None),
        })

        # Performance warning
        if elapsed > 10.0:
            _safe_log_with_data(log, "warning", "LLM ช้าเกินไป", {
                "label": label,
                "duration_ms": round(elapsed * 1000, 2),
                "threshold_ms": 10000,
                "model": model_name
            })
    else:
        # Fallback to old logging
        log.info(
            "[%s] tokens=%d (in=%d out=%d) time=%.2fs cost=$%.6f model=%s",
            label, total_call, prompt_tokens, completion_tokens, elapsed, cost_usd, model_name,
        )
    
    # Record metrics
    if _METRICS_AVAILABLE:
        metrics.record_llm_call(
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_ms=elapsed * 1000,
            success=True,
            persona=persona,
            operation=operation
        )

    if state is not None:
        _handle_post_call_token_budget(state, prompt_tokens, completion_tokens, label, log, cache_read_tokens, cache_write_tokens, cost_usd)

    return response


async def llm_invoke_async(
    llm: Any,
    messages: List[Any],
    *,
    logger: Optional[logging.Logger] = None,
    label: str = "LLM",
    state: Optional["ConversationState"] = None,
    persona: Optional[str] = None,
    operation: Optional[str] = None,
) -> Any:
    """
    Async version of llm_invoke. Calls llm.ainvoke() for async LLM operations.
    
    Args:
        llm: LangChain LLM instance (must support ainvoke)
        messages: List of messages to send
        logger: Logger instance for logging
        label: Label for log messages
        state: ConversationState for token tracking
        persona: Which persona is making the call (for metrics)
        operation: What operation is being performed (for metrics)
    
    Returns the raw LangChain response object (async version).
    """
    log = logger or _ROOT_LOG
    t0 = time.perf_counter()
    last_exc: Optional[Exception] = None
    
    # Extract model name for metrics
    model_name = getattr(llm, "model", getattr(llm, "model_name", "unknown"))

    for attempt in range(1 + _MAX_RETRIES):
        try:
            # Use ainvoke for async call
            response = await llm.ainvoke(messages)
            break
        except Exception as exc:
            last_exc = exc
            elapsed = time.perf_counter() - t0
            log.warning("[%s] exception: %s — %s", label, type(exc).__name__, str(exc)[:300])

            # BUG-G fix: same as llm_invoke — skip retries for LengthFinishReasonError
            if _is_length_error(exc):
                log.warning(
                    "[%s] LengthFinishReasonError — skip retries (token limit, same input = same result)",
                    label,
                )
                raise

            # Record failed attempt in metrics
            if _METRICS_AVAILABLE:
                metrics.record_llm_call(
                    model=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    elapsed_ms=elapsed * 1000,
                    success=False,
                    error=str(exc)[:200],
                    persona=persona,
                    operation=operation
                )

            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAYS[attempt]
                log.warning(
                    "[%s] LLM call FAILED after %.2fs (attempt %d/%d) — retrying in %.0fs",
                    label, elapsed, attempt + 1, 1 + _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)  # Use async sleep
            else:
                log.warning("[%s] LLM call FAILED after %.2fs (all %d attempts exhausted)", label, elapsed, 1 + _MAX_RETRIES)
                raise
    else:
        raise RuntimeError(f"[{label}] LLM call failed") from last_exc

    elapsed = time.perf_counter() - t0

    # Extract token counts via shared helper (logs debug on failure)
    prompt_tokens, completion_tokens = _extract_token_counts(response, log, label)
    cache_read_tokens, cache_write_tokens = _extract_cache_tokens(response)

    total_call = prompt_tokens + completion_tokens
    cost_usd = estimate_cost(model_name, prompt_tokens, completion_tokens, cache_read_tokens, cache_write_tokens)

    _check_token_budget(total_call, model_name)

    # Enhanced structured logging
    if _STRUCTURED_LOG:
        _safe_log_with_data(log, "info", f"{label} สำเร็จ (async)", {
            "action": "llm_call_async",
            "label": label,
            "model": model_name,
            "persona": persona,
            "operation": operation,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_call,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "duration_ms": round(elapsed * 1000, 2),
            "cost_usd": round(cost_usd, 6),
            "session_id": getattr(state, "session_id", None) if state else None,
            "temperature": getattr(llm, "temperature", None),
        })
        
        if elapsed > 10.0:
            _safe_log_with_data(log, "warning", "LLM ช้าเกินไป (async)", {
                "label": label,
                "duration_ms": round(elapsed * 1000, 2),
                "threshold_ms": 10000,
                "model": model_name
            })
    else:
        log.info(
            "[%s] (async) tokens=%d (in=%d out=%d) time=%.2fs cost=$%.6f model=%s",
            label, total_call, prompt_tokens, completion_tokens, elapsed, cost_usd, model_name,
        )
    
    # Record metrics
    if _METRICS_AVAILABLE:
        metrics.record_llm_call(
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_ms=elapsed * 1000,
            success=True,
            persona=persona,
            operation=operation
        )

    if state is not None:
        _handle_post_call_token_budget(state, prompt_tokens, completion_tokens, label, log, cache_read_tokens, cache_write_tokens, cost_usd)

    return response

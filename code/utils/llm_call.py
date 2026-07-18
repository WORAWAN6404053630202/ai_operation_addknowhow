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
import time
from typing import TYPE_CHECKING, Any, List, Optional

_MAX_RETRIES = 2
_RETRY_DELAYS = (0.3, 1.0)  # exponential backoff in seconds

if TYPE_CHECKING:
    from code.model.conversation_state import ConversationState

_ROOT_LOG = logging.getLogger("restbiz.llm")


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
) -> None:
    """Track cumulative token usage and auto-summarize/trim when session total is high.

    Shared by llm_invoke (sync) and llm_invoke_async (async) — no async operations here.
    Wrapped in a broad except so any failure is silent and never breaks the main call.
    """
    try:
        state.add_token_usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        session_total = getattr(state, "total_tokens", 0)
        if session_total <= _TOKEN_WARN_THRESHOLD:
            return

        log.warning(
            "[%s] Session token budget exceeded %d (total=%d) — trying summarization first",
            label, _TOKEN_WARN_THRESHOLD, session_total,
        )
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

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """คำนวณค่าใช้จ่ายโดยประมาณ (USD)"""
    pricing = PRICING_USD_PER_MILLION_TOKENS.get(model)
    if pricing is None:
        _COST_LOG.warning("[Cost] Model %r not in pricing table — cost logged as $0. Add it to PRICING_USD_PER_MILLION_TOKENS.", model)
        return 0.0
    cost = (prompt_tokens * pricing["input"] / 1_000_000) + (completion_tokens * pricing["output"] / 1_000_000)
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

    total_call = prompt_tokens + completion_tokens

    # Calculate cost
    cost_usd = estimate_cost(model_name, prompt_tokens, completion_tokens)
    
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
        _handle_post_call_token_budget(state, prompt_tokens, completion_tokens, label, log)

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

    total_call = prompt_tokens + completion_tokens
    cost_usd = estimate_cost(model_name, prompt_tokens, completion_tokens)
    
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
        _handle_post_call_token_budget(state, prompt_tokens, completion_tokens, label, log)

    return response

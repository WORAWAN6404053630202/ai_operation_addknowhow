# code/utils/llm_cost_logging.py
"""Token/cost logging for the PDF ingestion pipeline (feature/pdf-ingestion).

Background: unlike the main chat bot (persona_supervisor.py etc., which all
go through utils/llm_call.py's wrapper and log prompt_tokens/completion_
tokens/cost_usd on every call), every file under service/pdf_*.py and
lambda/pdf_extraction/handler.py calls the OpenAI SDK client directly and
never captured that data — real numbers were only ever available from the
OpenRouter dashboard, not from this project's own logs. This module is the
minimal fix: one function, called once after each LLM response is received,
that reads the token counts already present on the response object and logs
them alongside a computed cost.

Model pricing below was verified directly against openrouter.ai's model
pages (2026-08/09, during the same investigation that added this module) —
deliberately NOT copied from utils/llm_call.py's or router/admin.py's
pricing tables, both of which were found to have a stale entry for
claude-haiku-4-5 ($0.80/$4.00 there vs the actual $1.00/$5.00) during that
same check. Re-verify against openrouter.ai before trusting old values if
this table is ever copied elsewhere."""

from __future__ import annotations

import threading
from typing import Any, Optional

# {"input": $ per million input tokens, "output": $ per million output tokens}
PDF_PIPELINE_MODEL_PRICING: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "anthropic/claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "qwen/qwen3.7-flash": {"input": 0.03, "output": 0.13},
    "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}


class CostAccumulator:
    """Thread-safe running total for ONE document's processing — added
    2026-09 so the admin review UI can show a per-document total instead of
    cost only ever being visible one log line at a time. Pass the same
    instance into every log_llm_cost() call across every stage of one
    document's pipeline (extraction/vision-verify, content-shape
    classification, topic drafting, candidate matching) via the optional
    `accumulator` parameter — thread-safe because per-page extraction calls
    happen concurrently via ThreadPoolExecutor."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_cost_usd = 0.0

    def add(self, cost_usd: float) -> None:
        with self._lock:
            self.total_cost_usd += cost_usd


def _elapsed_suffix(elapsed_seconds: float | None) -> str:
    return "" if elapsed_seconds is None else f" elapsed_s={elapsed_seconds:.2f}"


def log_llm_cost(
    logger: Any,
    label: str,
    model: str,
    response: Any,
    elapsed_seconds: float | None = None,
    accumulator: Optional[CostAccumulator] = None,
) -> float:
    """Call once, right after receiving a chat-completion (or embedding)
    response, with a short label identifying which call site this was (e.g.
    "ClassifyContentShape", "DraftFields/topic-3") — the label is what makes
    the resulting log line searchable later, since every call site in this
    pipeline currently uses a generic response variable name.

    elapsed_seconds is optional (added 2026-09, same investigation that
    surfaced Typhoon's undocumented-until-checked 2 RPS / 20 RPM rate limit —
    a slow/retried call is the first visible symptom of hitting that ceiling,
    and there was previously no timing data anywhere in this pipeline to
    notice it happening) — pass time.monotonic() before/after the call
    yourself; this function does no timing of its own since it only ever
    sees the response, not the call.

    Reads response.usage.prompt_tokens/completion_tokens (present on any
    successful, non-streaming OpenAI-SDK chat-completion OR embedding
    response, regardless of which OpenRouter-routed model actually served
    it). Never raises: a logging failure must not break the real extraction
    work it's just observing, and a missing/malformed usage object degrades
    to logging zeros rather than crashing.

    Returns the computed cost in USD (0.0 if pricing for this model is
    unknown), so a caller that wants to accumulate a running total across
    multiple calls for one document can do so without re-deriving it from
    the log line — or pass a CostAccumulator via `accumulator` to have this
    function do that bookkeeping for you (added when the accumulator itself
    was added; 0-cost outcomes are added too, harmlessly, so the accumulator
    stays accurate without the caller needing to branch on which path ran)."""
    elapsed_str = _elapsed_suffix(elapsed_seconds)
    try:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    except Exception as e:
        logger.warning(f"[CostLog] {label}: could not read token usage from the response, logging nothing{elapsed_str}: {e}")
        if accumulator is not None:
            accumulator.add(0.0)
        return 0.0

    total_tokens = prompt_tokens + completion_tokens
    pricing = PDF_PIPELINE_MODEL_PRICING.get(model)

    if pricing is None:
        logger.warning(
            f"[CostLog] {label} | model={model} prompt_tokens={prompt_tokens} "
            f"completion_tokens={completion_tokens} total_tokens={total_tokens} "
            f"cost_usd=unknown (no pricing entry for this model in PDF_PIPELINE_MODEL_PRICING){elapsed_str}"
        )
        if accumulator is not None:
            accumulator.add(0.0)
        return 0.0

    cost_usd = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000
    logger.info(
        f"[CostLog] {label} | model={model} prompt_tokens={prompt_tokens} "
        f"completion_tokens={completion_tokens} total_tokens={total_tokens} "
        f"cost_usd={cost_usd:.6f}{elapsed_str}"
    )
    if accumulator is not None:
        accumulator.add(cost_usd)
    return cost_usd


def log_call_duration(logger: Any, label: str, elapsed_seconds: float) -> None:
    """For external calls with no token/cost data to report at all (Typhoon
    OCR's ocr_document() returns a plain string, not a response object with
    a .usage field) — added alongside log_llm_cost's new elapsed_seconds
    parameter so EVERY external call in this pipeline shows timing, not just
    the OpenRouter-routed ones. Never raises, same as log_llm_cost."""
    logger.info(f"[CostLog] {label} | elapsed_s={elapsed_seconds:.2f} (no token/cost data for this call)")

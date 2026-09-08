# code/utils/model_pricing.py
"""Single source of truth for OpenRouter model pricing and cache-discount
multipliers — shared by the chat bot (utils/llm_call.py), the PDF ingestion
pipeline (utils/llm_cost_logging.py), and the admin dashboard (router/admin.py).

Before this module existed, all 3 kept their own separate copies of this
table and had already drifted out of sync — router/admin.py had a stale
Haiku rate ($0.80/$4.00 vs the real $1.00/$5.00) and a GPT-5.1 rate that
didn't match utils/llm_call.py's ($2.00/$8.00 vs $1.25/$10.00).

Every rate below for a model with confirmed live traffic (see conf.py's
OPENROUTER_MODEL_*/OPENROUTER_SWITCH_MODEL env-var defaults) was
independently re-verified against openrouter.ai/<provider>/<model> on
2026-09-05 — do not trust old values if this table is ever copied
elsewhere; re-verify against openrouter.ai first.

The remaining entries (claude-sonnet-4, claude-4-5-sonnet-20241022,
claude-haiku-4, claude-3.5-haiku-20241022, gpt-4o, chatgpt-4o-latest) have
NO confirmed live caller in this codebase as of 2026-09-05 (checked via
repo-wide grep) — carried over unverified from the old tables, kept only
so an old/alias model string doesn't silently fall back to "unknown
pricing". Re-verify against openrouter.ai before trusting one of these if
it ever becomes live again.
"""

from __future__ import annotations

# {"input": $ per million input tokens, "output": $ per million output tokens}
MODEL_PRICING: dict[str, dict[str, float]] = {
    # --- Verified live against openrouter.ai on 2026-09-05 ---
    "anthropic/claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "anthropic/claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "openai/gpt-5.1": {"input": 1.25, "output": 10.00},
    "qwen/qwen3.7-flash": {"input": 0.03, "output": 0.13},
    "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "baai/bge-m3": {"input": 0.01, "output": 0.0},  # embedding model — no completion tokens

    # --- NOT independently re-verified this round — no confirmed live caller ---
    "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "anthropic/claude-4-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "anthropic/claude-haiku-4": {"input": 0.25, "output": 1.25},
    "anthropic/claude-3.5-haiku-20241022": {"input": 0.25, "output": 1.25},
    "openai/gpt-4o": {"input": 5.00, "output": 15.00},
    "openai/chatgpt-4o-latest": {"input": 5.00, "output": 15.00},
}


def get_pricing(model: str) -> dict[str, float] | None:
    return MODEL_PRICING.get(model)


def cache_multipliers(model_name: str) -> tuple[float, float]:
    """Returns (cache_read_multiplier, cache_write_multiplier), relative to a
    model's normal input price. Verified directly against openrouter.ai on
    2026-09-05:

      - Anthropic: cache reads are always 0.1x (90% off) regardless of TTL.
        Cache writes depend on TTL: 1.25x at the default 5-minute TTL, but
        2.0x at the 1-hour TTL (confirmed on claude-sonnet-4-5's page: $3.75/M
        for the 5m write vs $6.00/M for the 1h write, against a $3.00/M input
        price). This codebase's only cache_control call site
        (build_cached_system_message in llm_call.py) always requests
        ttl="1h", so 2.0x is the correct write multiplier here — the old
        table used 1.25x unconditionally, undercounting every cache-write
        call by ~1.6x.

      - OpenAI (e.g. gpt-5.1): cache reads are 0.1x (90% off) — confirmed on
        gpt-5.1's OpenRouter page ($0.125/M read vs $1.25/M input) and
        cross-checked against independent 2026 pricing writeups (both show a
        90% discount). The old table assumed 0.5x ("50% off"), which is
        stale/wrong for this model — that undercounted the true discount,
        i.e. OVERcounted cost on every cache-read call for the Academic
        persona (GPT-5.1). OpenAI's caching is automatic with no explicit
        write charge, so the write multiplier here is a moot 1.0x
        placeholder — no call in this codebase ever bills a cache write for
        a non-Anthropic model.
    """
    if "anthropic/" in (model_name or ""):
        return 0.1, 2.0
    return 0.1, 1.0

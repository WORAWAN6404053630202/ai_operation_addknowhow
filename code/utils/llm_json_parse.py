# code/utils/llm_json_parse.py
"""Shared lenient JSON parsing for LLM responses across the PDF pipeline
(classify_content_shape, identify_license_topics, identify_knowhow_topics,
and their batch-splitting siblings) — extracted 2026-09 after a real,
reproduced failure: asked to "answer JSON only, no other text", a model
that legitimately found nothing (topics=[]) still added a one-sentence Thai
explanation AFTER the JSON — e.g. "```json\n[]\n```\n\nเอกสารนี้เป็น..." — a
plain json.loads() on the whole string correctly raises "Extra data" even
though the JSON portion itself is perfectly valid, and the caller's
adaptive shrink-and-retry logic (built for a DIFFERENT failure mode — an
empty response from a token/reasoning-budget issue) can't fix this at all:
it's not size-related, so halving all the way down to 2-page batches still
fails identically, just at 6x the API cost for zero benefit (reproduced
live: 250-page real document, $2.56 in wasted retries, 41% of pages left
uncovered by the safety-net fallback instead of properly split).

Every call site in this pipeline independently did the same 2-step dance
(strip a markdown code fence, then json.loads the rest) — centralizing
here means fixing this once fixes it everywhere, and any future call site
gets the fix for free."""
from __future__ import annotations

import json
import re
from typing import Any


def parse_leading_json(raw: str) -> Any:
    """Strips a leading markdown code fence if present, then parses only the
    FIRST valid JSON value in what remains — any trailing text after it (a
    model's own explanation, disclaimer, stray closing fence, etc.) is
    ignored rather than raising. Raises json.JSONDecodeError if no valid
    JSON value starts the (fence-stripped) string at all — callers should
    catch that exactly as they already catch json.loads()'s JSONDecodeError."""
    text = re.sub(r"^```(?:json)?\s*", "", (raw or "").strip())
    obj, _end_index = json.JSONDecoder().raw_decode(text)
    return obj

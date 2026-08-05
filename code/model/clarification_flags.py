# code/model/clarification_flags.py
"""
Shared registry of context flags that suppress slot-based clarification.

Background: multiple independent code paths in this project each decide, on their
own, whether the bot should ask the user a clarifying question before answering:
  - persona_supervisor.py's _maybe_dynamic_clarification() — scans state.current_docs
    for metadata divergence (entity_type/registration_type/etc) and asks about it.
  - persona_practical.py's own entity_type-injection check (~"Level 1.6" in
    _retrieve_docs) — independently injects an entity_type question into
    topic_slot_queue when the docs it sees look entity-split.

Each used to maintain its own separate, hand-written list of "an earlier stage
already decided this turn shouldn't be asked about, so don't ask either" flags. That
caused a real, repeated bug (found and fixed 2026-08-03/04, see project memory,
e.g. "สุขาภิบาล"): a new flag added to suppress ONE of these mechanisms was never
wired into the OTHER, so the mechanism nobody updated would independently re-ask a
wrong or redundant clarifying question the first mechanism had already ruled out.

Keeping both lists HERE, side by side, in one shared module (rather than duplicated
inline at each call site across two different files) means anyone adding a new flag
sees both lists at once and must make a deliberate choice about which consumer(s) it
should affect — instead of the flag silently existing only in whichever file they
happened to be editing when they added it.

This module intentionally does NOT try to unify the two lists into one — they are
allowed to differ, because the two mechanisms have genuinely different blast radii
(see the comments on each list below). The value here is visibility, not sameness.
"""
from __future__ import annotations

from typing import Any, FrozenSet, Optional

# Checked by persona_supervisor.py's _maybe_dynamic_clarification() before it scans
# retrieved docs for metadata divergence. Narrower list than ENTITY_INJECTION_SKIP_
# FLAGS below by design: e.g. "_informational_query" (any info-style query, including
# fee/duration questions) is deliberately NOT in this set — those questions already
# get their own more targeted fee/duration doc-supplement handling upstream in
# _maybe_build_slot_queue_from_docs, and a genuine entity/area fee difference should
# still surface as a real clarifying question here, not be silently suppressed.
DYNAMIC_CLARIFICATION_SKIP_FLAGS: FrozenSet[str] = frozenset({
    "_multi_topic_retrieval",   # docs deliberately span multiple topics — choices would mix them
    "multi_license_topics",     # same, list form (non-empty = active)
    "_broad_question",          # user wants an overview, not a menu
    "_definition_query",        # "X คืออะไร" — entity/location/area irrelevant to the answer
    "_non_reg_dominant_query",  # doc mix is mostly non-regulatory — unreliable for divergence checks
})

# Checked by persona_practical.py before it independently injects an entity_type
# question into topic_slot_queue. Broader than DYNAMIC_CLARIFICATION_SKIP_FLAGS above
# by design: "_informational_query" and "_link_request_query" are included here
# because this specific injection point has always been considered too intrusive to
# fire during an info-style answer or a link request, regardless of the narrower
# per-mechanism reasoning that gates the dynamic-clarification scanner above.
#
# "_multi_topic_retrieval"/"multi_license_topics" added 2026-08-04 as a precaution:
# live-tested (multiple real queries spanning 2 entity-split licenses, e.g. "VAT
# สุรา") and found NO case where this injection currently fires wrongly in that
# scenario — but the reasoning for suppressing it here is the same one already
# trusted for "_broad_question" ("the answer covers multiple topics at once, so a
# single targeted question doesn't make sense"), and checking two more dict keys
# has no runtime cost, so there's no reason to leave the gap open for a future
# corpus/retrieval change to fall into.
ENTITY_INJECTION_SKIP_FLAGS: FrozenSet[str] = frozenset({
    "_informational_query",
    "_link_request_query",
    "_broad_question",
    "_direct_topic_match",
    "_definition_query",
    "_non_reg_dominant_query",
    "_multi_topic_retrieval",
    "multi_license_topics",
})


def has_any_flag(context: Optional[dict], flag_names: FrozenSet[str]) -> bool:
    """True if `context` (state.context) has any of `flag_names` set to a truthy
    value. Handles both boolean flags (True/absent) and list-valued flags like
    "multi_license_topics" (truthy iff non-empty) uniformly via plain truthiness."""
    ctx = context or {}
    return any(ctx.get(name) for name in flag_names)

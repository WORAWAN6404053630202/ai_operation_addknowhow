"""tests/test_pdf_drafting_regression.py — golden-fixture regression suite for
PDF drafting quality (feature/pdf-ingestion item 2.5 / "4" from the
production-readiness follow-up work).

Background: this pipeline has no automated way to catch a model/prompt change
silently degrading topic-splitting ("chunking") quality — the exact failure
mode item "1" fixed (identify_knowhow_topics()'s max_tokens truncating output
on documents with many topics, silently dropping pages). That fix was
verified once, by hand, with a throwaway synthetic document. This file makes
that kind of check repeatable: it re-runs the real drafting functions against
real OCR text captured in data/pdf_review_queue_dev/*.json (the local dev
persistence store every review item already gets written to — see
model/pdf_review_queue_manager.py) and compares the result against whatever
topics/fields a reviewer already saw for that same input, treated as the
golden baseline.

This is intentionally NOT a fixed, hand-curated golden set: as real documents
get processed through the dev pipeline and reviewed, this directory
accumulates more cases for free, and this suite picks them all up
automatically (see discover_knowhow_fixtures below). Only 1 fixture exists as
of this writing (a hand-built synthetic 3-page/2-topic doc — its
`knowhow_topics` entries predate the current schema and use `title` instead
of `main_topic`/`sub_topic`; _golden_label below tolerates both so older
fixtures don't need to be rewritten when the schema evolves).

Fires real, paid LLM calls (identify_knowhow_topics's own model) — marked
integration, excluded from the default `pytest` run. Run deliberately with
`pytest -m integration tests/test_pdf_drafting_regression.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from service.pdf_knowhow_drafting import identify_knowhow_topics  # noqa: E402
from utils.page_ranges import fuzzy_ratio  # noqa: E402

_FIXTURE_DIR = Path(__file__).parent.parent / "data" / "pdf_review_queue_dev"

# Below this, two topic labels are treated as unrelated rather than "the same
# topic, reworded" — loose on purpose since re-running the same LLM prompt
# never reproduces identical wording, only the same underlying split.
_LABEL_SIMILARITY_FLOOR = 0.3


def _parse_page_range(range_str: str) -> set[int]:
    """Inverse of utils/page_ranges.py's format_page_ranges: "1, 3-4" -> {1,3,4}."""
    pages: set[int] = set()
    for part in (range_str or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            pages.update(range(int(start.strip()), int(end.strip()) + 1))
        else:
            pages.add(int(part))
    return pages


def _golden_label(topic: dict[str, Any]) -> str:
    """Fixtures written before the main_topic/sub_topic schema split (see
    sqs_consumer.py's _build_knowhow_items) used a single `title` field
    instead — support both so old fixtures keep working untouched."""
    if "title" in topic:
        return str(topic.get("title") or "")
    return f"{topic.get('main_topic', '')} {topic.get('sub_topic', '')}".strip()


def discover_knowhow_fixtures() -> list[tuple[str, dict[str, Any]]]:
    """Every data/pdf_review_queue_dev/*.json item that was routed to the
    know-how path and already has reviewer-visible topics — i.e. every case
    this suite can hold today's drafting output accountable to."""
    if not _FIXTURE_DIR.is_dir():
        return []
    found = []
    for path in sorted(_FIXTURE_DIR.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        shape = (item.get("content_shape") or {}).get("shape")
        if shape == "know_how" and item.get("knowhow_topics"):
            found.append((path.stem, item))
    return found


_FIXTURES = discover_knowhow_fixtures()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not _FIXTURES, reason="no reviewed know-how fixtures in data/pdf_review_queue_dev yet")
@pytest.mark.parametrize("fixture_id, item", _FIXTURES, ids=[f[0] for f in _FIXTURES])
def test_knowhow_topic_split_matches_golden(fixture_id, item):
    pages_markdown = [p["typhoon_markdown"] for p in item["pages"]]
    golden_topics = item["knowhow_topics"]

    actual_topics = identify_knowhow_topics(pages_markdown)

    # The failure mode this guards against: max_tokens (or any other future
    # truncation/prompt regression) silently drops pages from the split
    # entirely, rather than mis-labeling them — that's the part that must
    # never regress unnoticed.
    golden_pages: set[int] = set()
    for t in golden_topics:
        golden_pages |= _parse_page_range(t.get("page_range", ""))
    actual_pages: set[int] = set()
    for t in actual_topics:
        for start, end in t["page_ranges"]:
            actual_pages.update(range(start, end + 1))
    missing_pages = golden_pages - actual_pages
    assert not missing_pages, (
        f"[{fixture_id}] {len(missing_pages)} page(s) present in the golden split "
        f"vanished from today's output: {sorted(missing_pages)} "
        f"(golden covered {sorted(golden_pages)}, actual covered {sorted(actual_pages)})"
    )

    # Topic count is a coarse but sensitive signal for "the split fell apart"
    # (e.g. truncation collapsing 22 topics into 1) — loose tolerance because
    # exact topic boundaries are allowed to drift, just not collapse.
    assert len(actual_topics) >= max(1, len(golden_topics) - 1), (
        f"[{fixture_id}] golden split had {len(golden_topics)} topic(s), "
        f"today's output only produced {len(actual_topics)}: "
        f"{[_golden_label(t) or t.get('main_topic') for t in actual_topics]}"
    )

    # Every golden topic should still be recognizable as *some* topic in
    # today's output — matched by page overlap first (the more reliable
    # signal), falling back to label wording only as a diagnostic.
    unmatched = []
    for gt in golden_topics:
        gt_pages = _parse_page_range(gt.get("page_range", ""))
        gt_label = _golden_label(gt)
        best_ratio = 0.0
        matched_by_pages = False
        for at in actual_topics:
            at_pages = {p for start, end in at["page_ranges"] for p in range(start, end + 1)}
            if gt_pages & at_pages:
                matched_by_pages = True
            at_label = f"{at.get('main_topic', '')} {at.get('sub_topic', '')}"
            best_ratio = max(best_ratio, fuzzy_ratio(gt_label, at_label))
        if not matched_by_pages and best_ratio < _LABEL_SIMILARITY_FLOOR:
            unmatched.append((gt_label, best_ratio))

    assert not unmatched, (
        f"[{fixture_id}] golden topic(s) have no matching page overlap or label "
        f"similarity in today's output: {unmatched}"
    )

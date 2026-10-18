# code/service/pdf_knowhow_drafting.py
"""Topic-identification for know-how-shaped PDFs (feature/pdf-ingestion) — for
documents pdf_content_shape.py routed away from the structured 13-field path
(advisory content, multi-topic documents, anything that doesn't have a single
clean license-procedure shape).

REVISED 2026-08-24: originally wrote to a brand-new "know_how" tab invented
for this feature — turned out the real Sheet already has 2 established
know-how tabs the bot's data_loader.py already reads from ("Know how_ร้านอาหาร"
for general business/marketing know-how, "Know How_ข้อมูลหนังสือ" for content
sourced from actual published books), each with a 2-level topic hierarchy
(หัวข้อหลัก main_topic -> หัวข้อการดำเนินการย่อย sub_topic), not the flat
title/category this module originally produced. This module now identifies
BOTH which of those 2 tabs a document belongs in (source_type: "book" if the
PDF is itself a book with a clear title/author, else "document") AND the
2-level topic split, in one call — see knowhow_write_back.py for how this
routes to the right tab and column layout.

Deliberately does NOT ask the LLM to reproduce full topic text — only topic
BOUNDARIES (which pages, main/sub topic, short summary, category). The actual
full_text for each topic is sliced directly from the already-extracted
per-page markdown by the caller (see knowhow_write_back.py), guaranteeing it
matches the source document verbatim instead of risking an LLM paraphrasing,
truncating, or subtly hallucinating while trying to reproduce a long passage
inside a single JSON response — the same "don't ask an LLM to echo back
long text faithfully" caution that shaped pdf_field_drafting.py's
research_reference field (filled deterministically, not drafted)."""

from __future__ import annotations

import time
from typing import Optional, TypedDict

from openai import OpenAI

import conf
from utils.llm_cost_logging import CostAccumulator, log_llm_cost
from utils.llm_json_parse import parse_leading_json
from utils.logger import get_logger
from utils.page_ranges import fuzzy_ratio, group_into_ranges, merge_topic_chunks
from utils.prompt_safety import INJECTION_GUARD

logger = get_logger(__name__)


class KnowhowTopicBounds(TypedDict):
    document_title: str
    source_type: str  # "book" | "document"
    main_topic: str
    sub_topic: str
    summary: str
    category: str
    page_ranges: list[tuple[int, int]]


_PROMPT_TEMPLATE = """เอกสารต่อไปนี้ (มีทั้งหมด {num_pages} หน้า) ถูกจัดว่าเป็นเนื้อหา know-how/ให้ความรู้
สำหรับฐานความรู้ของ Restbiz (ผู้ช่วย AI ด้านธุรกิจร้านอาหารในไทย) ซึ่งอาจมีหลายหัวข้อปนกันอยู่ในไฟล์เดียว

""" + INJECTION_GUARD + """

หน้าที่ของคุณมี 2 ส่วน:

**ส่วนที่ 1 — จำแนกประเภทเอกสารทั้งฉบับ (ทำครั้งเดียว):**
- "document_title": ชื่อเรื่อง/ชื่อหนังสือของเอกสารทั้งฉบับ (จากหน้าปก/หัวเรื่อง ถ้าไม่มีให้สรุปสั้นๆ จากเนื้อหา)
- "source_type": "book" ถ้าเอกสารนี้เป็น**หนังสือจริงๆ** (มีชื่อหนังสือ/ผู้แต่ง/สำนักพิมพ์ชัดเจน คล้ายหนังสือที่วางขาย)
  หรือ "document" ถ้าเป็นเอกสาร/บทความ/คู่มือ/ประกาศทั่วไปที่ไม่ใช่หนังสือทั้งเล่มโดยเฉพาะ

**ส่วนที่ 2 — แบ่งเอกสารออกเป็นหัวข้อย่อย 2 ระดับ:**
- "main_topic" (หัวข้อหลัก): **บังคับต้องมีเสมอ** ห้ามเว้นว่าง — เป็นหัวข้อกว้างๆ ที่หัวข้อย่อยหลายอันอาจแชร์ร่วมกันได้
  (เช่น "การจัดการด้านการตลาด", "ขั้นตอนการเตรียมเปิดร้าน")
- "sub_topic" (หัวข้อการดำเนินการย่อย): **บังคับต้องมีเสมอ** ประเด็นเฉพาะเจาะจงของหัวข้อนี้
  (เช่น "ความหมายกลยุทธ์ด้านผลิตภัณฑ์", "ศึกษาตลาด (Market Research)")
- "category" (ประเภท): **ไม่บังคับ** ถ้าเนื้อหาไม่ได้ระบุหมวดหมู่ชัดเจน ให้ใส่ค่าว่าง "" ห้ามเดา/แต่งขึ้นเอง
- แต่ละหน้าต้องอยู่ในหัวข้อย่อยใดหัวข้อย่อยหนึ่งเท่านั้น ห้ามข้ามหน้าหรือซ้อนทับกัน
- ถ้าเอกสารทั้งฉบับเป็นเรื่องเดียวต่อเนื่องกัน ให้ตอบมาแค่ 1 หัวข้อย่อยที่ครอบคลุมทุกหน้าก็ได้
- **สำคัญ**: ถ้าหัวข้อย่อยเดียวกันปรากฏอีกครั้งหลังถูกคั่นด้วยหัวข้ออื่น ให้ตอบเป็นคนละรายการที่แยกกันได้
  ตามปกติ (แต่ละรายการยังต้องเป็นช่วงหน้าต่อเนื่อง) แต่ต้องใช้ค่า "main_topic" และ "sub_topic" เป็น
  **ข้อความเดียวกันเป๊ะๆ** กับรายการแรกที่พูดถึงเรื่องนี้ — **ห้ามเติมคำต่อท้ายเช่น "(ต่อ)"** เพราะระบบ
  จะรวมรายการที่ main_topic+sub_topic ตรงกันเป๊ะให้เป็นเรื่องเดียวโดยอัตโนมัติในภายหลัง

**ห้ามคัดลอกเนื้อหาเต็มมาใส่ — ระบุแค่ขอบเขตหน้าและสรุปสั้นๆ เท่านั้น** (เนื้อหาเต็มจริงจะถูกตัดมาจาก
ต้นฉบับโดยตรงทีหลัง ไม่ต้องพิมพ์ซ้ำ)

ตอบเป็น JSON object เท่านั้น ไม่ต้องมีข้อความอื่น:
{{
  "document_title": "...",
  "source_type": "book หรือ document",
  "topics": [
    {{
      "main_topic": "...",
      "sub_topic": "...",
      "summary": "สรุปเนื้อหาของหัวข้อนี้ 2-4 ประโยค สำหรับใช้ค้นหา",
      "category": "หมวดหมู่สั้นๆ หรือค่าว่างถ้าไม่มี",
      "start_page": 1,
      "end_page": 3
    }}
  ]
}}

=== เนื้อหาเอกสารทุกหน้า (มีเลขหน้ากำกับแต่ละส่วน) ===
{combined_text}
"""


_OVERLAP_RESOLUTION_PROMPT = """หน้าต่อไปนี้ถูกระบบแบ่งหัวข้อจัดให้อยู่ใน 2 หัวข้อพร้อมกันโดยผิดพลาด (ห้ามเกิดขึ้น
แต่เกิดขึ้นจริง) ต้องเลือกว่าเนื้อหาหน้านี้ควรอยู่หัวข้อไหนมากกว่ากัน — เลือกแค่ 1 หัวข้อ

หัวข้อ A: "{topic_a}" — {sub_a}
หัวข้อ B: "{topic_b}" — {sub_b}

=== เนื้อหาหน้าที่มีปัญหา (หน้า {page_nums}) ===
{combined_text}

ตอบเป็น JSON เท่านั้น: {{"winner": "A" หรือ "B"}}
"""


def _resolve_page_overlaps(
    topics: list[KnowhowTopicBounds], pages_markdown: list[str], cost_accumulator: Optional[CostAccumulator] = None,
) -> list[KnowhowTopicBounds]:
    """merge_topic_chunks() above can still leave an overlap if the LLM's raw
    page_ranges answer claimed the same page for 2+ topics — a real,
    reproduced failure (2026-08/09 live example: adjacent topics claiming
    pages 6-7 in common) despite the main prompt already saying "ห้ามข้ามหน้า
    หรือซ้อนทับกัน". There's a deterministic check for the OPPOSITE failure
    (pages nobody claimed) in sqs_consumer.py's catch-all fallback, but
    nothing caught this direction until now.

    For each contiguous run of pages overlapping between exactly 2 topics,
    asks one small targeted follow-up (just those pages' text + the 2
    competing topic descriptions) rather than guessing with a heuristic —
    cheap (a couple pages of context, ~50 output tokens) and actually
    grounded in the disputed content, unlike a blind "first topic wins"
    rule which has no way to know which topic is really right. Falls back
    to first-topic-wins only if that follow-up call itself fails, so this
    can never hang or retry indefinitely — same bounded-fallback pattern as
    the rest of this pipeline's error handling."""
    def _page_set(t: KnowhowTopicBounds) -> set[int]:
        s: set[int] = set()
        for a, b in t["page_ranges"]:
            s.update(range(a, b + 1))
        return s

    page_owners: dict[int, list[int]] = {}
    for idx, t in enumerate(topics):
        for p in _page_set(t):
            page_owners.setdefault(p, []).append(idx)

    overlapping = sorted(p for p, owners in page_owners.items() if len(owners) > 1)
    if not overlapping:
        return topics

    # Group contiguous pages that share the exact same pair of competing
    # topics into one dispute (one follow-up call per dispute, not per page).
    runs: list[tuple[list[int], tuple[int, int]]] = []
    for p in overlapping:
        owners = tuple(sorted(page_owners[p])[:2])  # 3+-way overlap: just take the first 2, rare enough not to warrant a bespoke multi-way prompt
        if runs and runs[-1][1] == owners and runs[-1][0][-1] == p - 1:
            runs[-1][0].append(p)
        else:
            runs.append(([p], owners))

    losers: dict[int, set[int]] = {}
    client: OpenAI | None = None
    for pages, (idx_a, idx_b) in runs:
        winner_idx, loser_idx = idx_a, idx_b  # safe default if the resolution call below fails
        try:
            if client is None:
                client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
            combined = "\n\n---\n\n".join(f"[หน้า {p}]\n{pages_markdown[p - 1]}" for p in pages)
            prompt = _OVERLAP_RESOLUTION_PROMPT.format(
                topic_a=topics[idx_a]["main_topic"], sub_a=topics[idx_a]["sub_topic"],
                topic_b=topics[idx_b]["main_topic"], sub_b=topics[idx_b]["sub_topic"],
                page_nums=", ".join(str(p) for p in pages),
                combined_text=combined,
            )
            _call_start = time.monotonic()
            resp = client.chat.completions.create(
                model=conf.OPENROUTER_MODEL_PDF_TOPIC_SPLIT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0,
            )
            log_llm_cost(logger, "KnowhowDrafting/ResolveOverlap", conf.OPENROUTER_MODEL_PDF_TOPIC_SPLIT, resp, time.monotonic() - _call_start, accumulator=cost_accumulator)
            raw = resp.choices[0].message.content or "{}"
            winner = parse_leading_json(raw).get("winner", "A")
            winner_idx, loser_idx = (idx_a, idx_b) if winner == "A" else (idx_b, idx_a)
        except Exception as e:
            logger.warning(f"[KnowhowDrafting] Overlap resolution call failed for pages {pages}, defaulting to first-topic-wins: {e}")

        losers.setdefault(loser_idx, set()).update(pages)

    resolved: list[KnowhowTopicBounds] = []
    for idx, t in enumerate(topics):
        if idx not in losers:
            resolved.append(t)
            continue
        remaining_pages = sorted(_page_set(t) - losers[idx])
        if not remaining_pages:
            logger.info(f"[KnowhowDrafting] Topic {t['main_topic']!r}/{t['sub_topic']!r} lost all its pages to overlap resolution, dropping it")
            continue
        resolved.append({**t, "page_ranges": group_into_ranges(remaining_pages)})

    logger.info(f"[KnowhowDrafting] Resolved {len(overlapping)} overlapping page(s) across {len(runs)} dispute(s)")
    return resolved


# Real-world evidence (2026-09): a 242-page book's full combined OCR text
# hit a hard provider error — "prompt is too long: 227307 tokens > 200000
# maximum" — reproduced live against the exact real OCR text that triggered
# it. Unlike pdf_content_shape.py's classification call (which can tolerate
# truncating each page, since it only needs a coarse shape judgment), topic-
# splitting genuinely needs full page content to draw correct boundaries —
# so large documents are split into sequential BATCHES that each stay safely
# under the limit, each run through the same single-call prompt below, and
# the resulting chunks concatenated and merged exactly as multiple chunks
# from one call already were (see merge_topic_chunks — a topic split across
# a batch boundary merges the same way a topic split by an interrupting
# topic already did). Ratio observed on the real failing document: ~290k
# chars ≈ 227k tokens (~1.28 tokens/char for Thai) — 100k chars keeps each
# batch comfortably under 200k tokens even with prompt-template overhead.
_MAX_BATCH_CHARS = 100_000


def _split_into_batches(pages_markdown: list[str], max_chars: int) -> list[tuple[int, list[str]]]:
    """[(page_offset, batch_pages_markdown), ...] — contiguous, page-aligned
    (never splits a single page's text across 2 batches). page_offset is how
    many pages precede this batch in the full document, for remapping this
    batch's own local 1-based page numbers back to global ones."""
    batches: list[tuple[int, list[str]]] = []
    current: list[str] = []
    current_start = 0
    current_chars = 0
    for i, md in enumerate(pages_markdown):
        if current and current_chars + len(md) > max_chars:
            batches.append((current_start, current))
            current, current_start, current_chars = [], i, 0
        current.append(md)
        current_chars += len(md)
    if current:
        batches.append((current_start, current))
    return batches


# Batching (see identify_knowhow_topics) means a single document now makes
# several of these calls instead of one — multiplying how often a lone
# transient failure (a momentary empty response, a network blip) can drop
# a whole batch's pages into the "uncategorized, needs manual review"
# catch-all for no real reason. One retry is enough to smooth that over
# without risking a genuine, repeatable failure retrying forever.
_BATCH_MAX_ATTEMPTS = 2


def _identify_knowhow_topics_batch(
    batch_pages_markdown: list[str], page_offset: int, cost_accumulator: Optional[CostAccumulator] = None,
) -> tuple[list[dict], str, str, bool]:
    """Runs the single-call topic-identification prompt on ONE batch of
    pages (retried once on failure — see _BATCH_MAX_ATTEMPTS). Returns
    (chunks, document_title, source_type, failed) with start_page/end_page
    already remapped to GLOBAL document page numbers via page_offset — never
    raises. `failed` distinguishes "the call itself broke" from "the LLM
    legitimately found 0 topics in this batch" (both otherwise look like an
    empty chunks list) — see _identify_knowhow_topics_batch_adaptive, which
    only shrinks-and-retries on a real failure, not a legitimate empty
    result (that would just double cost for the same answer)."""
    numbered_pages = [f"[หน้า {i + 1}]\n{md}" for i, md in enumerate(batch_pages_markdown)]
    combined_text = "\n\n---\n\n".join(numbered_pages)

    parsed = None
    for attempt in range(1, _BATCH_MAX_ATTEMPTS + 1):
        try:
            client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
            _call_start = time.monotonic()
            resp = client.chat.completions.create(
                model=conf.OPENROUTER_MODEL_PDF_TOPIC_SPLIT,
                messages=[{
                    "role": "user",
                    "content": _PROMPT_TEMPLATE.format(num_pages=len(batch_pages_markdown), combined_text=combined_text),
                }],
                # 2500 (original) was never revisited after pdf_field_drafting.py's
                # identify_license_topics() — the sibling function this one mirrors —
                # found and fixed the EXACT same truncation failure mode (2026-08-25:
                # live-tested a 45-topic document, got a real JSONDecodeError from
                # output cut off mid-string at 1500, raised progressively to 6000).
                # This function's per-topic schema is even LARGER than that one's
                # (adds "summary" — 2-4 sentences — and "category", which
                # identify_license_topics doesn't have), so it needs MORE headroom
                # per topic, not the same ceiling. Live-reproduced 2026-09: a real
                # marketing/business_guide chapter ("กลยุทธ์ด้านการสื่อสาร") has 23
                # distinct sub_topics on its own — comfortably capable of exceeding
                # 2500 tokens once each entry's summary is included, meaning topics
                # could have been silently truncated at ingestion time with no
                # error surfaced anywhere (the caller only checks for a JSON parse
                # failure, not a suspiciously-short topic list). 8000 gives
                # headroom beyond the proven 6000 baseline for the smaller schema.
                max_tokens=8000,
                response_format={"type": "json_object"},
                # Live-tested 2026-08-25 (same finding as pdf_content_shape.py /
                # pdf_field_drafting.py): default sampling temperature made a
                # borderline classification call flip run-to-run; temperature=0
                # made repeated identical calls consistent.
                temperature=0,
                # qwen3.7-flash defaults reasoning ON — without disabling it, the
                # model's own chain-of-thought can consume the token budget
                # before ever writing the JSON answer (message.content=None,
                # finish_reason "length"). Found live 2026-08-25 switching this
                # call off OPENROUTER_MODEL_PRACTICAL.
                extra_body={"reasoning": {"enabled": False}},
            )
            log_llm_cost(logger, "KnowhowDrafting/IdentifyTopics", conf.OPENROUTER_MODEL_PDF_TOPIC_SPLIT, resp, time.monotonic() - _call_start, accumulator=cost_accumulator)
            raw = resp.choices[0].message.content or "{}"
            parsed = parse_leading_json(raw)
            break
        except Exception as e:
            logger.error(f"[KnowhowDrafting] LLM call/parse failed for batch starting at page {page_offset + 1} (attempt {attempt}/{_BATCH_MAX_ATTEMPTS}): {e}")
            parsed = None

    if parsed is None:
        return [], "", "document", True

    if not isinstance(parsed, dict) or not isinstance(parsed.get("topics"), list):
        logger.error(f"[KnowhowDrafting] LLM returned unexpected shape for batch starting at page {page_offset + 1}: {type(parsed)}")
        return [], "", "document", True

    document_title = str(parsed.get("document_title", "")).strip()
    source_type = str(parsed.get("source_type", "")).strip().lower()
    if source_type not in ("book", "document"):
        source_type = "document"  # fail toward the more general tab, not a guess at "book"

    num_local_pages = len(batch_pages_markdown)
    chunks: list[dict] = []
    for raw_topic in parsed["topics"]:
        try:
            start_page = int(raw_topic["start_page"])
            end_page = int(raw_topic["end_page"])
        except (KeyError, TypeError, ValueError):
            logger.warning(f"[KnowhowDrafting] Skipping topic with invalid page bounds: {raw_topic!r}")
            continue

        # Clamp rather than reject outright — an LLM off-by-one at the very
        # edges (e.g. end_page = num_pages + 1) shouldn't lose an otherwise-
        # valid topic; a page range that's backwards or fully outside the
        # document is a real error worth dropping instead of guessing.
        start_page = max(1, start_page)
        end_page = min(num_local_pages, end_page)
        if start_page > end_page or start_page > num_local_pages or end_page < 1:
            logger.warning(f"[KnowhowDrafting] Skipping topic with out-of-range pages {raw_topic.get('start_page')}-{raw_topic.get('end_page')} (batch has {num_local_pages} pages)")
            continue

        main_topic = str(raw_topic.get("main_topic", "")).strip()
        sub_topic = str(raw_topic.get("sub_topic", "")).strip()
        if not main_topic or not sub_topic:
            logger.warning(f"[KnowhowDrafting] Skipping topic missing required main_topic/sub_topic: {raw_topic!r}")
            continue

        chunks.append({
            "document_title": document_title,
            "source_type": source_type,
            "main_topic": main_topic,
            "sub_topic": sub_topic,
            "summary": str(raw_topic.get("summary", "")).strip(),
            "category": str(raw_topic.get("category", "")).strip(),
            "start_page": start_page + page_offset,
            "end_page": end_page + page_offset,
        })

    return chunks, document_title, source_type, False


def _identify_knowhow_topics_batch_adaptive(
    batch_pages_markdown: list[str], page_offset: int, cost_accumulator: Optional[CostAccumulator] = None,
) -> tuple[list[dict], str, str]:
    """Wraps _identify_knowhow_topics_batch with shrink-and-retry on a real
    failure: live-reproduced 2026-09 that an 85-page batch failed with an
    EMPTY response from the model twice in a row with identical parameters
    (deterministic, not a transient blip a same-size retry could fix) — most
    likely the same "model's own reasoning silently eats the whole token
    budget" failure mode already documented elsewhere in this file, just
    resurfacing for this specific content at this specific size. Since a
    same-size retry can't out-run a deterministic failure, split the batch
    in half and recurse instead — each half is smaller/simpler, more likely
    to actually complete. Bounded by construction: the batch halves every
    recursion, so this terminates in at most log2(len(batch)) steps even in
    the pathological case where every half keeps failing all the way down
    to individual pages (at which point there's nothing smaller to try, so
    it just returns whatever that single page's own attempt got — the
    existing uncovered-pages catch-all in sqs_consumer.py still nets it)."""
    chunks, doc_title, src_type, failed = _identify_knowhow_topics_batch(batch_pages_markdown, page_offset, cost_accumulator=cost_accumulator)
    if not failed or len(batch_pages_markdown) <= 1:
        return chunks, doc_title, src_type

    mid = len(batch_pages_markdown) // 2
    logger.warning(f"[KnowhowDrafting] Batch at page {page_offset + 1} ({len(batch_pages_markdown)} pages) failed outright even after retry — splitting in half and trying again")
    left_chunks, left_title, left_type = _identify_knowhow_topics_batch_adaptive(
        batch_pages_markdown[:mid], page_offset, cost_accumulator=cost_accumulator
    )
    right_chunks, right_title, right_type = _identify_knowhow_topics_batch_adaptive(
        batch_pages_markdown[mid:], page_offset + mid, cost_accumulator=cost_accumulator
    )
    title, src = (left_title, left_type) if left_title else (right_title, right_type)
    return left_chunks + right_chunks, title, src


def identify_knowhow_topics(pages_markdown: list[str], cost_accumulator: Optional[CostAccumulator] = None) -> list[KnowhowTopicBounds]:
    """Returns [] (never raises) on total failure — the caller treats an
    empty list as "could not split into topics" and should fall back to
    treating the whole document as one topic rather than losing it entirely;
    see knowhow_write_back.py. document_title/source_type are denormalized
    onto every returned topic dict (same values repeated) so downstream code
    (knowhow_write_back.py) doesn't need a separate wrapper shape — every
    topic is independently routable/writable.

    Documents small enough to fit _MAX_BATCH_CHARS run as a single call
    (unchanged behavior/cost from before batching existed — the common
    case). Larger documents split into sequential batches (see
    _split_into_batches), each independently topic-split, then merged
    together exactly as multiple same-call chunks already were."""
    total_chars = sum(len(md) for md in pages_markdown)
    if total_chars <= _MAX_BATCH_CHARS:
        batches = [(0, pages_markdown)]
    else:
        batches = _split_into_batches(pages_markdown, _MAX_BATCH_CHARS)
        logger.warning(f"[KnowhowDrafting] {len(pages_markdown)}-page document ({total_chars} chars) exceeds the single-call budget — splitting into {len(batches)} batch(es)")

    all_chunks: list[dict] = []
    fallback_document_title, fallback_source_type = "", "document"
    for offset, batch in batches:
        chunks, doc_title, src_type = _identify_knowhow_topics_batch_adaptive(batch, offset, cost_accumulator=cost_accumulator)
        if not fallback_document_title and doc_title:
            fallback_document_title, fallback_source_type = doc_title, src_type
        all_chunks.extend(chunks)

    # A book's title/source_type shouldn't fragment across batches just
    # because a later batch (without the cover page in view) guessed
    # differently — normalize every chunk to whichever batch found a real
    # title first (usually the first, since it has the cover/title pages).
    if fallback_document_title:
        for c in all_chunks:
            c["document_title"] = fallback_document_title
            c["source_type"] = fallback_source_type

    topics: list[KnowhowTopicBounds] = merge_topic_chunks(all_chunks, ("main_topic", "sub_topic"), fuzzy_ratio)
    topics = _resolve_page_overlaps(topics, pages_markdown, cost_accumulator=cost_accumulator)
    logger.info(
        f"[KnowhowDrafting] {fallback_document_title!r} (source_type={fallback_source_type}): "
        f"identified {len(topics)} topic(s) (from {len(all_chunks)} chunk(s) across {len(batches)} batch(es)) across {len(pages_markdown)} page(s)"
    )
    return topics

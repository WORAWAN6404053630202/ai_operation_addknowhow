# code/service/pdf_field_drafting.py
"""LLM field-drafting for the PDF review queue (feature/pdf-ingestion) — maps raw
extracted page text into the REAL Google Sheet's column structure (matches
data_loader.py's _build_column_map() exactly, so a drafted row lines up with
what ingest_local.py already expects when it later reads this Sheet for RAG).

Drafts fields plausibly derivable FROM a source PDF's content (department,
license type, steps, documents, fees, duration, conditions, legal basis, service
channel, registration type, answer_guideline). answer_guideline ("แนวคำตอบ") looks
like an editorial/style field by name but per data_loader.py it's actually
substantive content — for business_guide docs it's the SOLE content field, and
it's injected verbatim into RAG context ("แนวคำตอบ:\n{value}") — so it's a
prose summary drawn from the document, same category as the other content
fields, not a subjective invention.

Deliberately leaves 2 columns blank (restaurant_ai_document, combined_links) —
both are LINKS to internal team-hosted resources/forms, not content a source
PDF would state. A wrong drafted URL is worse than a wrong drafted sentence (a
dead/misleading link, not just an inaccurate summary) — this is exactly the
category of risk that produced a hallucinated URL in the Typhoon OCR accuracy
testing earlier in this feature's development. A reviewer fills those in by hand.

research_reference is the one exception: it's NOT drafted here at all (not an
LLM guess) — sheet_write_back.py fills it in deterministically with the source
PDF's filename, since "which document is this row based on" has an exact,
100%-certain answer for PDF-ingested content and asking an LLM to reproduce a
fact we already have would only add risk for no benefit."""

from __future__ import annotations

import json
import time
from typing import Optional, TypedDict

from openai import OpenAI

import conf
from utils.llm_cost_logging import CostAccumulator, log_llm_cost
from utils.pdf_cost_guard import get_cost_guard
from utils.llm_json_parse import parse_leading_json
from utils.logger import get_logger
from utils.page_ranges import fuzzy_ratio as _fuzzy_ratio
from utils.page_ranges import group_into_ranges, merge_topic_chunks
from utils.prompt_safety import INJECTION_GUARD

logger = get_logger(__name__)

# internal_key -> [primary header, *aliases] — copied exactly from data_loader.py's
# _build_column_map() so this matches whichever header variant the real Sheet
# actually uses (confirmed live: this test Sheet uses the ALIAS "หัวข้อการดำเนินการย่อย"
# for operation_topic, not the primary name — a single hardcoded string would have
# silently written that column blank). Index 0 (primary) is the key used in the
# dict this module returns; sheet_write_back.py matches against every variant.
DRAFTABLE_FIELDS: dict[str, list[str]] = {
    "department": ["หน่วยงาน"],
    "license_type": ["ใบอนุญาต"],
    "operation_by_department": ["การดำเนินการ ตามหน่วยงาน", "การดำเนินการตามหน่วยงาน", "การดำเนินการ  ตามหน่วยงาน"],
    "operation_topic": ["หัวข้อการดำเนินการ", "หัวข้อการดำเนินการย่อย", "หัวข้อการดำเนินการ (ย่อย)"],
    "registration_type": ["ประเภทการจดทะเบียน", "ประเภท การจดทะเบียน"],
    "terms_and_conditions": ["เงื่อนไขและหลักเกณฑ์", "เงื่อนไข", "หลักเกณฑ์"],
    "service_channel": ["ช่องทางการ ให้บริการ", "ช่องทางการให้บริการ", "ช่องทางให้บริการ", "ช่องทาง"],
    "operation_steps": ["ขั้นตอนการดำเนินการ", "ขั้นตอน"],
    "identification_documents": ["เอกสาร ยืนยันตัวตน", "เอกสารยืนยันตัวตน", "เอกสารยืนยัน"],
    "operation_duration": ["ระยะเวลา การดำเนินการ", "ระยะเวลา", "ระยะเวลาการดำเนินการ"],
    "fees": ["ค่าธรรมเนียม"],
    "legal_regulatory": ["ข้อกำหนดทางกฎหมาย และข้อบังคับ", "ข้อกำหนดทางกฎหมายและข้อบังคับ", "ข้อกำหนดทางกฎหมาย", "ข้อบังคับ"],
    "answer_guideline": ["แนวคำตอบ", "แนวทางคำตอบ", "แนวตอบ"],
}

# Link/URL columns pointing at internal team-hosted resources — never drafted.
# research_reference deliberately excluded from this list too — it IS filled in,
# just deterministically by sheet_write_back.py (source PDF filename) rather
# than by the LLM here.
NOT_DRAFTED_FIELDS = ["restaurant_ai_document", "combined_links"]

# Bump this whenever _PROMPT_TEMPLATE below changes — it's the cache-
# invalidation key for utils/pdf_draft_cache.py's content-hash cache
# (service/sqs_consumer.py includes it in the cache key it builds), so an
# old cached draft from before a prompt improvement never gets silently
# served after the improvement ships. Declared here, next to the prompt it
# versions, specifically so editing one naturally prompts editing the other.
DRAFT_FIELDS_PROMPT_VERSION = "v1"

_PROMPT_TEMPLATE = """ต่อไปนี้คือข้อความที่สกัดได้จากเอกสารราชการ (PDF) ทุกหน้า

""" + INJECTION_GUARD + """

หน้าที่ของคุณ: กรอกข้อมูลลงในฟิลด์ต่อไปนี้ **เฉพาะข้อมูลที่มีอยู่จริงในเอกสารเท่านั้น**
ห้ามเดาหรือแต่งเติมข้อมูลที่ไม่มีในเอกสารเด็ดขาด — ถ้าฟิลด์ไหนไม่มีข้อมูลในเอกสาร ให้ใส่ค่าว่าง ""

ฟิลด์ที่ต้องกรอก (ตอบเป็น JSON object เท่านั้น ไม่ต้องมีข้อความอื่น):
{{
  "department": "หน่วยงานที่ออกเอกสาร/รับผิดชอบ",
  "license_type": "ประเภทใบอนุญาต",
  "operation_by_department": "การดำเนินการตามหน่วยงาน",
  "operation_topic": "หัวข้อการดำเนินการ (สรุปสั้นๆ)",
  "registration_type": "ประเภทการจดทะเบียน (บุคคลธรรมดา/นิติบุคคล ฯลฯ)",
  "terms_and_conditions": "เงื่อนไขและหลักเกณฑ์",
  "service_channel": "ช่องทางการให้บริการ",
  "operation_steps": "ขั้นตอนการดำเนินการ (สรุปเป็นลำดับ)",
  "identification_documents": "เอกสารที่ต้องใช้ยืนยันตัวตน/ประกอบการยื่นคำขอ",
  "operation_duration": "ระยะเวลาดำเนินการ",
  "fees": "ค่าธรรมเนียม (ระบุตัวเลขให้ครบถ้วนตามที่ปรากฏ)",
  "legal_regulatory": "ข้อกำหนดทางกฎหมายและข้อบังคับที่เกี่ยวข้อง",
  "answer_guideline": "สรุปเนื้อหาสำคัญของเอกสารที่เป็นประโยชน์ต่อการตอบคำถามผู้ใช้ แต่ไม่ได้ครอบคลุมในฟิลด์ข้างต้น (ถ้ามี) — เขียนเป็นเนื้อหาสรุป ไม่ใช่คำแนะนำเชิงสไตล์"
}}

=== เนื้อหาเอกสารทุกหน้า ===
{combined_text}
"""


# Identity-like fields expected to be THE SAME value across every batch of a
# single topic (a topic's department/license_type/etc. don't change mid-topic)
# — used by _merge_drafted_batches below to pick first-non-empty rather than
# concatenate, unlike the content fields (fees/operation_steps/...), which can
# legitimately have real content spread across more than one batch.
_IDENTITY_LIKE_HEADER_KEYS = {
    DRAFTABLE_FIELDS[k][0] for k in ("department", "license_type", "operation_by_department", "operation_topic", "registration_type")
}


def _draft_fields_single_batch(combined_text: str, cost_accumulator: Optional[CostAccumulator] = None) -> dict[str, str]:
    """One LLM call over already-combined text, returning a dict keyed by the
    REAL Sheet header text (via DRAFTABLE_FIELDS, 13 fields). Fields the LLM
    couldn't find are empty strings, not omitted."""
    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)

    prompt = _PROMPT_TEMPLATE.format(combined_text=combined_text)
    _call_start = time.monotonic()
    resp = client.chat.completions.create(
        model=conf.OPENROUTER_MODEL_PDF_DRAFTING,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
    )
    log_llm_cost(logger, "FieldDrafting/DraftFields", conf.OPENROUTER_MODEL_PDF_DRAFTING, resp, time.monotonic() - _call_start, accumulator=cost_accumulator, cost_guard=get_cost_guard())
    raw = resp.choices[0].message.content or "{}"

    try:
        drafted = parse_leading_json(raw)
    except json.JSONDecodeError as e:
        logger.error(f"[pdf_field_drafting] LLM returned invalid JSON: {e} — raw: {raw[:200]}")
        drafted = {}

    # Key by the primary header text (DRAFTABLE_FIELDS[key][0]), always present
    # (empty string if missing), and silently ignore any key the LLM invented
    # outside DRAFTABLE_FIELDS. sheet_write_back.py matches this against whichever
    # header variant the real Sheet actually uses.
    result: dict[str, str] = {}
    for internal_key, header_variants in DRAFTABLE_FIELDS.items():
        value = drafted.get(internal_key, "")
        result[header_variants[0]] = value if isinstance(value, str) else str(value)
    return result


def _merge_drafted_batches(batch_results: list[dict[str, str]]) -> dict[str, str]:
    """Combines 2+ batches' header-keyed drafted-field dicts into one.
    Identity-like fields (department/license_type/...) take the first
    non-empty value — repeating them per batch would be redundant at best,
    contradictory at worst. Content fields (fees/operation_steps/...)
    concatenate every batch's non-empty value instead (deduped, order
    preserved) — a single large topic split across batches can legitimately
    have its real content (e.g. steps 1-5 in batch 1, 6-10 in batch 2) spread
    across more than one batch; first-wins would silently drop the rest."""
    merged: dict[str, str] = {}
    for header in (headers[0] for headers in DRAFTABLE_FIELDS.values()):
        values = [b.get(header, "").strip() for b in batch_results if b.get(header, "").strip()]
        if not values:
            merged[header] = ""
        elif header in _IDENTITY_LIKE_HEADER_KEYS:
            merged[header] = values[0]
        else:
            merged[header] = "\n".join(dict.fromkeys(values))
    return merged


def draft_fields_from_pages(pages_markdown: list[str], cost_accumulator: Optional[CostAccumulator] = None) -> dict[str, str]:
    """Runs the field-drafting prompt over all pages and returns a dict keyed
    by the REAL Sheet header text (via DRAFTABLE_FIELDS, 13 fields), ready to
    hand to sheet_write_back.py. Fields the LLM couldn't find are empty
    strings, not omitted — callers should always see all 13 draftable keys
    present.

    2026-10 fix (PDF-M1): previously always a single LLM call over the whole
    combined text, unlike its sibling identify_license_topics — which already
    needed _MAX_BATCH_CHARS/_split_into_batches after a live-reproduced
    failure ("prompt is too long: 227142 tokens > 200000 maximum" on a real
    242-page document, see that constant's comment). A single TOPIC's pages
    are usually much smaller than a whole multi-topic document, but nothing
    structurally prevents one huge single-topic document from hitting the
    exact same ceiling — this was the one call left with no size guard at all."""
    total_chars = sum(len(md) for md in pages_markdown)
    if total_chars <= _MAX_BATCH_CHARS:
        result = _draft_fields_single_batch("\n\n---\n\n".join(pages_markdown), cost_accumulator=cost_accumulator)
    else:
        batches = _split_into_batches(pages_markdown, _MAX_BATCH_CHARS)
        logger.warning(f"[pdf_field_drafting] Topic text ({total_chars} chars across {len(pages_markdown)} pages) exceeds the single-call budget — drafting in {len(batches)} batch(es)")
        batch_results = [
            _draft_fields_single_batch("\n\n---\n\n".join(batch_pages), cost_accumulator=cost_accumulator)
            for _offset, batch_pages in batches
        ]
        result = _merge_drafted_batches(batch_results)

    logger.info(f"[pdf_field_drafting] Drafted {sum(1 for v in result.values() if v)} / {len(result)} fields")
    return result


class LicenseTopicBounds(TypedDict):
    department: str
    license_type: str
    page_ranges: list[tuple[int, int]]


# Same purpose as DRAFT_FIELDS_PROMPT_VERSION above, for _TOPIC_SPLIT_PROMPT
# specifically — bump when that prompt changes.
TOPIC_SPLIT_PROMPT_VERSION = "v1"

_TOPIC_SPLIT_PROMPT = """เอกสารต่อไปนี้ (มีทั้งหมด {num_pages} หน้า) ถูกจัดว่าเป็นเอกสารเกี่ยวกับใบอนุญาต/
ขั้นตอนราชการ ซึ่งอาจพูดถึงใบอนุญาตหรือขั้นตอนมากกว่า 1 เรื่องปนกันอยู่ในไฟล์เดียวก็ได้ (เช่น เอกสารรวม
ประกาศหลายฉบับ) หรืออาจไม่เกี่ยวกับใบอนุญาต/ขั้นตอนราชการเลยก็ได้

""" + INJECTION_GUARD + """

หน้าที่ของคุณ: **ระบุว่าเอกสารนี้พูดถึงใบอนุญาต/ขั้นตอนราชการกี่เรื่อง แต่ละเรื่องอยู่หน้าไหนถึงหน้าไหน**
(ไม่ต้องกรอกรายละเอียดฟิลด์ทั้งหมดตรงนี้ แค่ระบุหน่วยงาน+ประเภทใบอนุญาตคร่าวๆ พอให้แยกเรื่องออกจากกันได้)

- ถ้าทั้งเอกสารพูดถึงเรื่องเดียวต่อเนื่องกัน ให้ตอบมาแค่ 1 รายการที่ครอบคลุมทุกหน้า
- ถ้าเอกสารไม่เกี่ยวกับใบอนุญาต/ขั้นตอนราชการเลย ให้ตอบ list ว่างเปล่า [] ห้ามฝืนสร้างรายการที่ไม่มีจริง
- แต่ละหน้าควรอยู่ในเรื่องใดเรื่องหนึ่งเท่านั้น ห้ามข้ามหน้าหรือซ้อนทับกันโดยไม่จำเป็น
- **สำคัญ**: ถ้าเรื่องเดียวกันปรากฏอีกครั้งหลังถูกคั่นด้วยเรื่องอื่น (เช่น หน้า 1 กับหน้า 3 เป็นเรื่อง
  เดียวกัน แต่หน้า 2 เป็นเรื่องอื่นคั่นอยู่) ให้ตอบเป็น**คนละรายการที่แยกกัน**ได้ตามปกติ (แต่ละรายการ
  ยังต้องเป็นช่วงหน้าต่อเนื่อง) แต่ต้องใช้ค่า "department" และ "license_type" เป็น**ข้อความเดียวกันเป๊ะๆ**
  กับรายการแรกที่พูดถึงเรื่องนี้ — **ห้ามเติมคำต่อท้ายเช่น "(ต่อ)" หรือเปลี่ยนคำแม้เล็กน้อย** เพราะระบบ
  จะรวมรายการที่ department+license_type ตรงกันเป๊ะให้เป็นเรื่องเดียวโดยอัตโนมัติในภายหลัง
- **สำคัญมาก**: ถ้าเอกสารมีใบอนุญาต/เรื่องหลายรายการที่ชื่อคล้ายกันมากแต่จริงๆ เป็นคนละเรื่องกัน
  (เช่น "ใบอนุญาตประกอบกิจการโรงงานจำพวกที่ 1" กับ "จำพวกที่ 2", หรือมีรหัส/หมายเลข/ประเภทย่อย
  กำกับไว้ต่อท้ายเพื่อแยกความแตกต่าง) **ต้องคงคำหรือรหัสที่แยกความแตกต่างนั้นไว้ใน "license_type" เสมอ
  ห้ามตัดออกแม้จะทำให้ข้อความสั้นลง** — การตัดรายละเอียดที่แยกความแตกต่างออกจะทำให้ระบบเข้าใจผิดว่า
  เป็นเรื่องเดียวกันแล้วรวมเข้าด้วยกันโดยไม่ตั้งใจ

ตอบเป็น JSON array เท่านั้น ไม่ต้องมีข้อความอื่น:
[
  {{
    "department": "หน่วยงานที่รับผิดชอบเรื่องนี้",
    "license_type": "ประเภทใบอนุญาต/เรื่องนี้สั้นๆ",
    "start_page": 1,
    "end_page": 3
  }}
]

=== เนื้อหาเอกสารทุกหน้า (มีเลขหน้ากำกับแต่ละส่วน) ===
{combined_text}
"""


# Real-world evidence (2026-09): a 242-page real document's full combined
# OCR text made this same style of call fail outright — "prompt is too long:
# 227142 tokens > 200000 maximum", reproduced live. Same fix as
# pdf_knowhow_drafting.py's identify_knowhow_topics: split into sequential,
# page-aligned batches that each stay safely under the limit, run each
# through the same single-call prompt, concatenate the resulting chunks, and
# merge exactly as multiple chunks from one call already were. See that
# module for the token/char ratio this budget is based on.
_MAX_BATCH_CHARS = 100_000
_BATCH_MAX_ATTEMPTS = 2  # one retry — batching multiplies how many calls can hit a lone transient failure


def _split_into_batches(pages_markdown: list[str], max_chars: int) -> list[tuple[int, list[str]]]:
    """[(page_offset, batch_pages_markdown), ...] — contiguous, page-aligned.
    Identical logic to pdf_knowhow_drafting.py's helper of the same name;
    not shared because each module's chunk shape differs enough (department/
    license_type vs main_topic/sub_topic/summary/category) that a shared
    helper would need to take a callback anyway — not worth the indirection
    for ~10 lines of pure list-grouping logic."""
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


def _identify_license_topics_batch(batch_pages_markdown: list[str], page_offset: int, cost_accumulator: Optional[CostAccumulator] = None) -> tuple[list[dict], bool]:
    """Runs the single-call topic-split prompt on ONE batch, retried once on
    failure. Returns (chunks, failed) with department/license_type/
    start_page/end_page already remapped to GLOBAL document page numbers —
    never raises. `failed` distinguishes "the call itself broke" from "the
    LLM legitimately found 0 topics" — see _identify_license_topics_batch_adaptive,
    which only shrinks-and-retries on a real failure."""
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
                    "content": _TOPIC_SPLIT_PROMPT.format(num_pages=len(batch_pages_markdown), combined_text=combined_text),
                }],
                # 6000 (raised from 1500, then 4000, then 6000 — all 2026-08-25):
                # live-tested a 45-topic combined bulletin and reproduced a real
                # JSONDecodeError from output getting cut off mid-string at 1500 —
                # the caller's except-clause fallback (treat the whole slice as one
                # topic) meant this degraded gracefully rather than crashing, but
                # silently produced a garbled single row blending 45 departments
                # together, not a clean split. 4000 fixed that, but then the SAME
                # sweep found the prompt instruction below (preserve distinguishing
                # codes/classes so similarly-named-but-different topics don't get
                # merge_topic_chunks'd together) makes each entry longer, which
                # itself re-hit the ceiling at 45 topics — 6000 covers both.
                max_tokens=6000,
                # Live-tested 2026-08-25 (same finding as pdf_content_shape.py):
                # default sampling temperature made a borderline topic-boundary call
                # flip run-to-run; temperature=0 made repeated identical calls
                # consistent. Topic splitting is a classification task, not a
                # creative one — determinism matters more than variety here.
                temperature=0,
                # qwen3.7-flash defaults reasoning ON — without disabling it, the
                # model's own chain-of-thought can consume the token budget before
                # ever writing the JSON answer (message.content=None, finish_reason
                # "length"). Found live 2026-08-25 switching this call off
                # OPENROUTER_MODEL_PRACTICAL.
                extra_body={"reasoning": {"enabled": False}},
            )
            log_llm_cost(logger, "FieldDrafting/IdentifyLicenseTopics", conf.OPENROUTER_MODEL_PDF_TOPIC_SPLIT, resp, time.monotonic() - _call_start, accumulator=cost_accumulator, cost_guard=get_cost_guard())
            raw = resp.choices[0].message.content or "[]"
            parsed = parse_leading_json(raw)
            if not isinstance(parsed, list):
                raise ValueError(f"LLM returned non-list JSON: {type(parsed)}")
            break
        except Exception as e:
            logger.error(f"[pdf_field_drafting] License topic-split failed for batch starting at page {page_offset + 1} (attempt {attempt}/{_BATCH_MAX_ATTEMPTS}): {e}")
            parsed = None

    if parsed is None:
        return [], True

    num_local_pages = len(batch_pages_markdown)
    chunks: list[dict] = []
    for raw_topic in parsed:
        try:
            start_page = int(raw_topic["start_page"])
            end_page = int(raw_topic["end_page"])
        except (KeyError, TypeError, ValueError):
            logger.warning(f"[pdf_field_drafting] Skipping license topic with invalid page bounds: {raw_topic!r}")
            continue

        start_page = max(1, start_page)
        end_page = min(num_local_pages, end_page)
        if start_page > end_page or start_page > num_local_pages or end_page < 1:
            logger.warning(f"[pdf_field_drafting] Skipping license topic with out-of-range pages {raw_topic.get('start_page')}-{raw_topic.get('end_page')} (batch has {num_local_pages} pages)")
            continue

        chunks.append({
            "department": str(raw_topic.get("department", "")).strip(),
            "license_type": str(raw_topic.get("license_type", "")).strip(),
            "start_page": start_page + page_offset,
            "end_page": end_page + page_offset,
        })
    return chunks, False


def _identify_license_topics_batch_adaptive(batch_pages_markdown: list[str], page_offset: int, cost_accumulator: Optional[CostAccumulator] = None) -> list[dict]:
    """Wraps _identify_license_topics_batch with shrink-and-retry on a real
    failure — same rationale/mechanics as pdf_knowhow_drafting.py's
    identify_knowhow_topics adaptive wrapper (live-reproduced there: a
    same-size retry can't out-run a deterministic empty-response failure,
    but a smaller/simpler batch is more likely to actually complete).
    Bounded by construction: halves every recursion."""
    chunks, failed = _identify_license_topics_batch(batch_pages_markdown, page_offset, cost_accumulator=cost_accumulator)
    if not failed or len(batch_pages_markdown) <= 1:
        return chunks

    mid = len(batch_pages_markdown) // 2
    logger.warning(f"[pdf_field_drafting] License batch at page {page_offset + 1} ({len(batch_pages_markdown)} pages) failed outright even after retry — splitting in half and trying again")
    left = _identify_license_topics_batch_adaptive(batch_pages_markdown[:mid], page_offset, cost_accumulator=cost_accumulator)
    right = _identify_license_topics_batch_adaptive(batch_pages_markdown[mid:], page_offset + mid, cost_accumulator=cost_accumulator)
    return left + right


def identify_license_topics(pages_markdown: list[str], cost_accumulator: Optional[CostAccumulator] = None) -> list[LicenseTopicBounds]:
    """Splits a structured_license-shaped PDF into 0..N distinct license/
    procedure topics, mirroring pdf_knowhow_drafting.py's
    identify_knowhow_topics() — a single PDF is not guaranteed to be exactly
    one license (could be several combined announcements, or none at all).
    Each topic's page_ranges gets sliced (all ranges, joined) and passed to
    draft_fields_from_pages() independently by the caller (sqs_consumer.py),
    producing one fully-independent ReviewItem per topic — reuses the
    existing single-topic review/candidate-matching/decision UI unchanged
    rather than needing a new nested per-topic decision flow.

    page_ranges (plural, list of (start,end) tuples) rather than a single
    start_page/end_page — REVISED 2026-08-24: the same topic can legitimately
    appear in non-contiguous chunks (interrupted by an unrelated topic), and
    forcing that into one contiguous range would either wrongly include the
    interrupting pages' content or silently drop pages; see
    utils/page_ranges.py's merge_topic_chunks() for how same-topic chunks
    (identified by department+license_type, fuzzy-matched so an LLM wording
    slip like appending "(ต่อ)" still merges) get combined.

    Large documents (see _MAX_BATCH_CHARS) are split into sequential batches
    first — same rationale/mechanics as identify_knowhow_topics's batching.

    Returns [] (never raises) on failure OR on a genuine "0 topics" verdict —
    the caller must not assume [] always means an error; see sqs_consumer.py
    for how a truly-empty result is distinguished (falls back to treating the
    whole document as one topic only when this call raised/failed outright,
    not when the LLM legitimately found nothing license-related)."""
    total_chars = sum(len(md) for md in pages_markdown)
    if total_chars <= _MAX_BATCH_CHARS:
        batches = [(0, pages_markdown)]
    else:
        batches = _split_into_batches(pages_markdown, _MAX_BATCH_CHARS)
        logger.warning(f"[pdf_field_drafting] {len(pages_markdown)}-page document ({total_chars} chars) exceeds the single-call budget — splitting into {len(batches)} batch(es)")

    chunks: list[dict] = []
    for offset, batch in batches:
        chunks.extend(_identify_license_topics_batch_adaptive(batch, offset, cost_accumulator=cost_accumulator))

    num_pages = len(pages_markdown)
    topics: list[LicenseTopicBounds] = merge_topic_chunks(chunks, ("department", "license_type"), _fuzzy_ratio)
    topics = _resolve_page_overlaps(topics, pages_markdown, cost_accumulator=cost_accumulator)
    logger.info(f"[pdf_field_drafting] Identified {len(topics)} license topic(s) (from {len(chunks)} chunk(s) across {len(batches)} batch(es)) across {num_pages} page(s)")
    return topics


# 2026-10 security fix: this prompt embeds raw PDF page text (combined_text
# below, straight from OCR output) and was missing INJECTION_GUARD despite
# every other PDF-text-embedding prompt in this file having it — an
# oversight against this module's own stated intent (see
# utils/prompt_safety.py's docstring: "applied everywhere uniformly"). Its
# output directly controls which topic wins a disputed page, so a
# successful injection here has a real, concrete effect (page content
# silently reassigned), not just a cosmetic one — closing this gap mirrors
# the exact pattern already proven on every other call site in this file.
_OVERLAP_RESOLUTION_PROMPT = """หน้าต่อไปนี้ถูกระบบแบ่งหัวข้อจัดให้อยู่ใน 2 เรื่องพร้อมกันโดยผิดพลาด (ห้ามเกิดขึ้น
แต่เกิดขึ้นจริง) ต้องเลือกว่าเนื้อหาหน้านี้ควรอยู่เรื่องไหนมากกว่ากัน — เลือกแค่ 1 เรื่อง

""" + INJECTION_GUARD + """

เรื่อง A: หน่วยงาน "{department_a}" — ประเภทใบอนุญาต "{license_type_a}"
เรื่อง B: หน่วยงาน "{department_b}" — ประเภทใบอนุญาต "{license_type_b}"

=== เนื้อหาหน้าที่มีปัญหา (หน้า {page_nums}) ===
{combined_text}

ตอบเป็น JSON เท่านั้น: {{"winner": "A" หรือ "B"}}
"""


def _resolve_page_overlaps(
    topics: list[LicenseTopicBounds], pages_markdown: list[str], cost_accumulator: Optional[CostAccumulator] = None,
) -> list[LicenseTopicBounds]:
    """Same fix as pdf_knowhow_drafting.py's _resolve_page_overlaps (not
    imported — this checks a different topic shape: department/license_type
    instead of main_topic/sub_topic, and the two functions' prompts need
    different label text, not worth a shared abstraction over 2 small call
    sites). See that function's docstring for the full reasoning: merge_topic_
    chunks() above can still leave an overlap if the LLM's raw page_ranges
    answer claimed the same page for 2+ topics, despite the main prompt
    already saying "ห้ามข้ามหน้าหรือซ้อนทับกัน" — this was a real, reproduced
    failure mode for identify_knowhow_topics() (adjacent topics claiming
    pages 6-7 in common, 2026-08/09) and there's no reason identify_license_
    topics() is structurally immune to the identical LLM behavior, even
    though it hasn't been separately reproduced here yet.

    Resolves each contiguous run of pages overlapping between exactly 2
    topics with one small targeted follow-up call (just those pages' text +
    the 2 competing topic descriptions), falling back to first-topic-wins
    only if that follow-up call itself fails — bounded, can never hang or
    retry indefinitely."""
    def _page_set(t: LicenseTopicBounds) -> set[int]:
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

    # 2026-10 fix (PDF2-N3): `owners` used to be truncated to the first 2
    # (sorted(page_owners[p])[:2]) — a genuine 3-way claim on one page
    # still left 2 owners after "resolution" (the loser of the LLM
    # pairwise call between owners #1/#2 was removed, but owner #3 was
    # never considered at all, so its own page_ranges still included that
    # page untouched). Grouped runs now keep the FULL owner set per page.
    runs: list[tuple[list[int], tuple[int, ...]]] = []
    for p in overlapping:
        owners = tuple(sorted(page_owners[p]))
        if runs and runs[-1][1] == owners and runs[-1][0][-1] == p - 1:
            runs[-1][0].append(p)
        else:
            runs.append(([p], owners))

    losers: dict[int, set[int]] = {}
    client: OpenAI | None = None
    for pages, owners in runs:
        if len(owners) > 2:
            # 3+-way: no bespoke multi-way prompt (rare enough not to
            # warrant one) — deterministically keep the lowest-index
            # owner and drop ALL others, guaranteeing exactly one owner
            # remains for these pages (unlike the old 2-of-N truncation,
            # which could leave 2 owners standing after "resolution").
            winner_idx = min(owners)
            for loser_idx in owners:
                if loser_idx != winner_idx:
                    losers.setdefault(loser_idx, set()).update(pages)
            logger.warning(
                f"[pdf_field_drafting] {len(owners)}-way page overlap at {pages} (topics {owners}) — "
                f"no multi-way resolution prompt, deterministically assigning to topic {winner_idx}"
            )
            continue

        idx_a, idx_b = owners
        winner_idx, loser_idx = idx_a, idx_b  # safe default if the resolution call below fails
        try:
            if client is None:
                client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
            combined = "\n\n---\n\n".join(f"[หน้า {p}]\n{pages_markdown[p - 1]}" for p in pages)
            prompt = _OVERLAP_RESOLUTION_PROMPT.format(
                department_a=topics[idx_a]["department"], license_type_a=topics[idx_a]["license_type"],
                department_b=topics[idx_b]["department"], license_type_b=topics[idx_b]["license_type"],
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
            log_llm_cost(logger, "FieldDrafting/ResolveOverlap", conf.OPENROUTER_MODEL_PDF_TOPIC_SPLIT, resp, time.monotonic() - _call_start, accumulator=cost_accumulator, cost_guard=get_cost_guard())
            raw = resp.choices[0].message.content or "{}"
            winner = parse_leading_json(raw).get("winner", "A")
            # Fail-closed (2026-10 security fix): only an EXPLICIT "B" flips
            # the pre-set safe default above — the old `else` branch treated
            # any non-"A" value (malformed JSON, an injected response trying
            # to return something unexpected, anything) as an affirmative
            # "B" vote, which is the opposite of fail-closed. "A" and
            # anything else now both keep the deterministic default.
            if winner == "B":
                winner_idx, loser_idx = idx_b, idx_a
        except Exception as e:
            logger.warning(f"[pdf_field_drafting] Overlap resolution call failed for pages {pages}, defaulting to first-topic-wins: {e}")

        losers.setdefault(loser_idx, set()).update(pages)

    resolved: list[LicenseTopicBounds] = []
    for idx, t in enumerate(topics):
        if idx not in losers:
            resolved.append(t)
            continue
        remaining_pages = sorted(_page_set(t) - losers[idx])
        if not remaining_pages:
            logger.info(f"[pdf_field_drafting] Topic {t['department']!r}/{t['license_type']!r} lost all its pages to overlap resolution, dropping it")
            continue
        resolved.append({**t, "page_ranges": group_into_ranges(remaining_pages)})

    logger.info(f"[pdf_field_drafting] Resolved {len(overlapping)} overlapping page(s) across {len(runs)} dispute(s)")
    return resolved

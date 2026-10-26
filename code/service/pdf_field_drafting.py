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
from typing import Optional

from openai import OpenAI

import conf
from utils.llm_cost_logging import CostAccumulator, log_llm_cost
from utils.pdf_cost_guard import get_cost_guard
from utils.llm_json_parse import parse_leading_json
from utils.logger import get_logger
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


# 2026-10 fix (data-integrity audit): this call had NO retry/escalation at
# all, unlike its sibling identify_license_topics/identify_knowhow_topics
# in this same pipeline — both of which were bumped (1500→4000→6000,
# 2500→8000) after live-reproduced truncation on verbose Thai JSON output,
# and both now also retry. This drafts a full prose value for ALL 13
# fields per topic — strictly MORE content than a topic-boundary entry —
# so nothing structurally protected it from hitting the exact same ceiling;
# on truncation this used to just log an error and silently return every
# field blank, permanently, with no retry. finish_reason=="length" is
# checked explicitly (OpenAI/OpenRouter's own documented, authoritative
# truncation signal — a response can in principle still look like
# complete-but-wrong JSON at the exact truncation boundary, which a
# parse-error-only check would miss entirely) alongside the existing
# JSONDecodeError catch, which still covers the (much more common
# in practice) case where truncation lands mid-string/mid-object.
_DRAFT_FIELDS_MAX_TOKENS_ATTEMPTS = (3000, 6000)


def _draft_fields_single_batch(combined_text: str, cost_accumulator: Optional[CostAccumulator] = None) -> dict[str, str]:
    """One LLM call over already-combined text, returning a dict keyed by the
    REAL Sheet header text (via DRAFTABLE_FIELDS, 13 fields). Fields the LLM
    couldn't find are empty strings, not omitted. Retries once at a larger
    max_tokens if the first attempt was truncated (see
    _DRAFT_FIELDS_MAX_TOKENS_ATTEMPTS's comment) — if the retry ALSO
    truncates, returns whatever the retry parsed (empty dict if that too
    failed to parse) rather than looping forever on a document whose
    content genuinely doesn't fit."""
    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
    prompt = _PROMPT_TEMPLATE.format(combined_text=combined_text)

    drafted: dict = {}
    for attempt, max_tokens in enumerate(_DRAFT_FIELDS_MAX_TOKENS_ATTEMPTS, start=1):
        _call_start = time.monotonic()
        resp = client.chat.completions.create(
            model=conf.OPENROUTER_MODEL_PDF_DRAFTING,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        log_llm_cost(logger, "FieldDrafting/DraftFields", conf.OPENROUTER_MODEL_PDF_DRAFTING, resp, time.monotonic() - _call_start, accumulator=cost_accumulator, cost_guard=get_cost_guard())
        raw = resp.choices[0].message.content or "{}"
        truncated_by_finish_reason = getattr(resp.choices[0], "finish_reason", None) == "length"

        try:
            drafted = parse_leading_json(raw)
            parse_failed = False
        except json.JSONDecodeError as e:
            logger.error(f"[pdf_field_drafting] LLM returned invalid JSON (attempt {attempt}/{len(_DRAFT_FIELDS_MAX_TOKENS_ATTEMPTS)}, max_tokens={max_tokens}): {e} — raw: {raw[:200]}")
            drafted = {}
            parse_failed = True

        if not (truncated_by_finish_reason or parse_failed):
            break
        if attempt < len(_DRAFT_FIELDS_MAX_TOKENS_ATTEMPTS):
            logger.warning(
                f"[pdf_field_drafting] Draft looked truncated (finish_reason=length={truncated_by_finish_reason}, "
                f"parse_failed={parse_failed}) at max_tokens={max_tokens} — retrying with a larger budget"
            )

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


# Real-world evidence (2026-09): a 242-page real document's full combined
# OCR text made this same style of call fail outright — "prompt is too long:
# 227142 tokens > 200000 maximum", reproduced live. Same fix as
# pdf_knowhow_drafting.py's identify_knowhow_topics: split into sequential,
# page-aligned batches that each stay safely under the limit, run each
# through the same single-call prompt, concatenate the resulting chunks, and
# merge exactly as multiple chunks from one call already were. See that
# module for the token/char ratio this budget is based on.
_MAX_BATCH_CHARS = 100_000


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



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
import re
from typing import TypedDict

from openai import OpenAI

import conf
from utils.logger import get_logger
from utils.page_ranges import fuzzy_ratio as _fuzzy_ratio
from utils.page_ranges import merge_topic_chunks
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


def draft_fields_from_pages(pages_markdown: list[str]) -> dict[str, str]:
    """Runs one LLM call over all pages combined and returns a dict keyed by the
    REAL Sheet header text (via DRAFTABLE_FIELDS, 13 fields), ready to hand to
    sheet_write_back.py. Fields the LLM couldn't find are empty strings, not
    omitted — callers should always see all 12 draftable keys present."""
    combined_text = "\n\n---\n\n".join(pages_markdown)
    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)

    prompt = _PROMPT_TEMPLATE.format(combined_text=combined_text)
    resp = client.chat.completions.create(
        model=conf.OPENROUTER_MODEL_PRACTICAL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
    )
    raw = (resp.choices[0].message.content or "{}").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)

    try:
        drafted = json.loads(raw)
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

    logger.info(f"[pdf_field_drafting] Drafted {sum(1 for v in result.values() if v)} / {len(result)} fields")
    return result


class LicenseTopicBounds(TypedDict):
    department: str
    license_type: str
    page_ranges: list[tuple[int, int]]


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


def identify_license_topics(pages_markdown: list[str]) -> list[LicenseTopicBounds]:
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

    Returns [] (never raises) on failure OR on a genuine "0 topics" verdict —
    the caller must not assume [] always means an error; see sqs_consumer.py
    for how a truly-empty result is distinguished (falls back to treating the
    whole document as one topic only when this call raised/failed outright,
    not when the LLM legitimately found nothing license-related)."""
    numbered_pages = [f"[หน้า {i + 1}]\n{md}" for i, md in enumerate(pages_markdown)]
    combined_text = "\n\n---\n\n".join(numbered_pages)

    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
    resp = client.chat.completions.create(
        model=conf.OPENROUTER_MODEL_PDF_CLASSIFICATION,
        messages=[{
            "role": "user",
            "content": _TOPIC_SPLIT_PROMPT.format(num_pages=len(pages_markdown), combined_text=combined_text),
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
        # qwen3.7-flash defaults reasoning ON — without disabling it, the
        # model's own chain-of-thought can consume the token budget before
        # ever writing the JSON answer (message.content=None, finish_reason
        # "length"). Found live 2026-08-25 switching this call off
        # OPENROUTER_MODEL_PRACTICAL.
        extra_body={"reasoning": {"enabled": False}},
    )
    raw = (resp.choices[0].message.content or "[]").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    parsed = json.loads(raw)

    if not isinstance(parsed, list):
        raise ValueError(f"identify_license_topics: LLM returned non-list JSON: {type(parsed)}")

    num_pages = len(pages_markdown)
    chunks: list[dict] = []
    for raw_topic in parsed:
        try:
            start_page = int(raw_topic["start_page"])
            end_page = int(raw_topic["end_page"])
        except (KeyError, TypeError, ValueError):
            logger.warning(f"[pdf_field_drafting] Skipping license topic with invalid page bounds: {raw_topic!r}")
            continue

        start_page = max(1, start_page)
        end_page = min(num_pages, end_page)
        if start_page > end_page or start_page > num_pages or end_page < 1:
            logger.warning(f"[pdf_field_drafting] Skipping license topic with out-of-range pages {raw_topic.get('start_page')}-{raw_topic.get('end_page')} (doc has {num_pages} pages)")
            continue

        chunks.append({
            "department": str(raw_topic.get("department", "")).strip(),
            "license_type": str(raw_topic.get("license_type", "")).strip(),
            "start_page": start_page,
            "end_page": end_page,
        })

    topics: list[LicenseTopicBounds] = merge_topic_chunks(chunks, ("department", "license_type"), _fuzzy_ratio)
    logger.info(f"[pdf_field_drafting] Identified {len(topics)} license topic(s) (from {len(chunks)} chunk(s)) across {num_pages} page(s)")
    return topics

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

from openai import OpenAI

import conf
from utils.logger import get_logger

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

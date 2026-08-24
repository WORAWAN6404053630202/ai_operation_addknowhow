# code/service/pdf_relevance_check.py
"""LLM-based domain-relevance screening for the PDF review queue
(feature/pdf-ingestion) — judges whether an uploaded document actually
belongs in Restbiz's Thai restaurant-business regulatory knowledge base,
before a reviewer spends time reading through and drafting/approving it.

Nothing upstream of this checks relevance at all: Lambda is pure OCR, and
pdf_field_drafting.py will happily draft department/license_type/etc. fields
out of ANY document handed to it, regardless of whether it has anything to do
with restaurants — it has no concept of "in scope" vs "out of scope", only
"what does this document say". This module is the one place that asks the
in-scope question explicitly.

Advisory only, same design philosophy as pdf_candidate_matching.py: never
auto-rejects anything, just surfaces a judgment + one-line reasoning in the
admin UI so a reviewer can weigh it alongside their own read of the document.
A wrong relevance judgment costs a misleading label, not a bad decision — the
human still makes the final call for every item, no exceptions. A separate,
dedicated LLM call (not merged into pdf_field_drafting.py's own call) on
purpose: relevance and field-extraction are different judgments, and a single
prompt trying to do both risks doing each worse than two focused ones — same
one-concern-per-module pattern as pdf_extraction_validation.py /
pdf_candidate_matching.py."""

from __future__ import annotations

import json
import re
from typing import Literal, TypedDict

from openai import OpenAI

import conf
from utils.logger import get_logger
from utils.prompt_safety import INJECTION_GUARD

logger = get_logger(__name__)

RelevanceTier = Literal["relevant", "uncertain", "not_relevant"]


class RelevanceCheck(TypedDict):
    tier: RelevanceTier
    reasoning: str


_VALID_TIERS = ("relevant", "uncertain", "not_relevant")

# Fallback used whenever the LLM call or its JSON parsing fails — "uncertain"
# is the only safe default here: claiming "relevant" on a failure could let an
# out-of-scope document sail through with no warning at all, and claiming
# "not_relevant" could cause a reviewer to reflexively reject something that
# was actually fine. "uncertain" always routes to human judgment either way.
_FAILURE_FALLBACK: RelevanceCheck = {
    "tier": "uncertain",
    "reasoning": "ระบบตรวจสอบความเกี่ยวข้องทำงานผิดพลาด (LLM ไม่ตอบในรูปแบบที่คาดไว้) — กรุณาตรวจสอบเนื้อหาด้วยตนเอง",
}

_PROMPT_TEMPLATE = """คุณเป็นผู้ช่วยคัดกรองเอกสารสำหรับฐานความรู้ของ Restbiz — ระบบ AI ให้คำปรึกษาด้าน
กฎหมาย/ใบอนุญาต/ข้อบังคับสำหรับ "ธุรกิจร้านอาหารในประเทศไทย" โดยเฉพาะ

ขอบเขตของฐานความรู้นี้ครอบคลุมเรื่องที่เจ้าของร้านอาหาร/ธุรกิจอาหารในไทยต้องรู้เพื่อดำเนินธุรกิจถูกกฎหมาย เช่น:
- การจดทะเบียนธุรกิจ/นิติบุคคล/ทะเบียนพาณิชย์
- ใบอนุญาตที่เกี่ยวกับอาหารโดยตรง (สถานที่จำหน่ายอาหาร, สุขาภิบาลอาหาร, อย.)
- ใบอนุญาตขายสุรา/ยาสูบ/ไพ่ (ร้านอาหารจำนวนมากขายด้วย)
- ภาษีป้าย, ภาษีที่ดินและสิ่งปลูกสร้างของสถานประกอบการ
- กฎหมายแรงงานที่เกี่ยวกับการจ้างพนักงานร้านอาหาร
- ใบอนุญาตสิ่งแวดล้อม/บำบัดน้ำเสียของร้านอาหาร
- ระบบ/กฎระเบียบการชำระเงินอิเล็กทรอนิกส์ที่ร้านค้าต้องปฏิบัติตาม
- เรื่องอื่นใดที่เกี่ยวข้องโดยตรงกับการเปิด ดำเนิน หรือปิดกิจการร้านอาหาร/ธุรกิจอาหารในไทย

""" + INJECTION_GUARD + """

หน้าที่ของคุณ: อ่านเอกสารที่สกัดมาด้านล่าง แล้วประเมินว่าเอกสารนี้เกี่ยวข้องกับขอบเขตข้างต้นแค่ไหน

ตอบเป็น JSON เท่านั้น ไม่ต้องมีข้อความอื่น:
{{
  "tier": "relevant" | "uncertain" | "not_relevant",
  "reasoning": "เหตุผลสั้นๆ 1-2 ประโยค อธิบายว่าทำไมถึงตัดสินแบบนี้"
}}

ก่อนตัดสิน ให้ถามตัวเองก่อนว่า: **"เอกสารนี้เอ่ยถึงหรือเจาะจงเรื่องร้านอาหาร/ธุรกิจอาหารโดยตรงหรือไม่ หรือเป็นกฎหมาย/ระเบียบทั่วไปที่ใช้กับธุรกิจทุกประเภทเหมือนกันหมด (ไม่ได้เจาะจงร้านอาหารเลย)?"**

เกณฑ์การตัดสิน:
- "relevant": เอกสารเจาะจงพูดถึงร้านอาหาร/ธุรกิจอาหาร/สถานที่จำหน่ายอาหารโดยตรง หรือเป็นใบอนุญาตประเภทที่ร้านอาหารต้องขอเป็นการเฉพาะ (เช่น ใบอนุญาตขายสุรา, สุขาภิบาลอาหาร)
- "uncertain": เป็นกฎหมาย/ระเบียบ/ภาษีทั่วไปที่ใช้กับธุรกิจ**ทุกประเภทเหมือนกันหมด** ไม่ได้เจาะจงหรือเอ่ยถึงร้านอาหารเลย แม้ในทางเทคนิคร้านอาหารจะต้องปฏิบัติตามด้วยก็ตาม (เช่น ประมวลรัษฎากรทั่วไป, กฎหมายแรงงานทั่วไปที่ไม่ได้พูดถึงร้านอาหารเป็นพิเศษ, ระเบียบจดทะเบียนนิติบุคคลทั่วไป) — **อย่าตัดสินว่า "relevant" แค่เพราะร้านอาหารก็ต้องทำตามเหมือนธุรกิจอื่นๆ ทั่วไป** ให้ยกให้คนตรวจสอบพิจารณาแทน
- "not_relevant": ไม่เกี่ยวข้องกับธุรกิจร้านอาหารแม้แต่ทางอ้อม (เช่น ทะเบียนรถยนต์ส่วนบุคคล, ใบอนุญาตอุตสาหกรรมอื่นที่ไม่เกี่ยวกับอาหาร/ร้านอาหารเลย, เอกสารส่วนบุคคลทั่วไป)

ห้ามเดาเกินกว่าที่เอกสารเขียนไว้จริง — ถ้าอ่านแล้วไม่แน่ใจว่าเกี่ยวข้องหรือไม่ ให้ตอบ "uncertain" เสมอ ดีกว่าฟันธงผิด

=== เนื้อหาเอกสารทุกหน้า ===
{combined_text}
"""


def check_relevance(pages_markdown: list[str]) -> RelevanceCheck:
    """One LLM call over all pages combined. Never raises — any failure
    (network, malformed JSON, unexpected tier value) returns _FAILURE_FALLBACK
    so a hiccup here surfaces as "please check this yourself" in the UI rather
    than silently missing the whole check or crashing the caller."""
    combined_text = "\n\n---\n\n".join(pages_markdown)
    try:
        client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
        resp = client.chat.completions.create(
            model=conf.OPENROUTER_MODEL_PRACTICAL,
            messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(combined_text=combined_text)}],
            max_tokens=400,
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        parsed = json.loads(raw)
    except Exception as e:
        logger.error(f"[RelevanceCheck] LLM call/parse failed, defaulting to 'uncertain': {e}")
        return _FAILURE_FALLBACK

    tier = parsed.get("tier")
    if tier not in _VALID_TIERS:
        logger.warning(f"[RelevanceCheck] LLM returned unexpected tier {tier!r}, defaulting to 'uncertain'")
        tier = "uncertain"

    reasoning = parsed.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    logger.info(f"[RelevanceCheck] tier={tier}")
    return {"tier": tier, "reasoning": reasoning}

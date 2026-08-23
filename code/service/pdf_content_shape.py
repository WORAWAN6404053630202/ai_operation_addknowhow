# code/service/pdf_content_shape.py
"""Content-shape classification for the PDF review queue (feature/pdf-ingestion)
— decides whether a document fits the existing 13-field structured Sheet
schema (department/license_type/steps/fees/... — designed for "how do I get
license X" procedures) or is better handled as one-or-more free-form
know-how entries (advisory content, multi-topic documents, anything that
would lose most of its substance getting forced into 13 spreadsheet cells).

This is a ROUTING decision, not a relevance judgment (see pdf_relevance_check.py,
which already runs separately) — a document can be perfectly relevant to
restaurant business and still not have a single-license shape.

Advisory-only in spirit but structurally load-bearing: sqs_consumer.py uses
this to decide which drafting path to run (pdf_field_drafting.py vs
pdf_knowhow_drafting.py), so unlike relevance_check/candidate_matches this one
isn't just a UI hint a reviewer can ignore — a wrong call here means the
wrong kind of drafting ran. The failure mode is still contained though: a
reviewer can always reject and re-upload, and worst case a know-how document
drafted as "structured" just ends up with mostly-empty fields (the same
graceful-ish degradation this system already had before this module existed)."""

from __future__ import annotations

import json
import re
from typing import Literal, TypedDict

from openai import OpenAI

import conf
from utils.logger import get_logger

logger = get_logger(__name__)

ContentShape = Literal["structured_license", "know_how"]


class ShapeCheck(TypedDict):
    shape: ContentShape
    reasoning: str


_VALID_SHAPES = ("structured_license", "know_how")

# Fail toward "structured_license" — that's the ORIGINAL, longer-proven path
# (13-field drafting existed and was tested before know-how drafting did), so
# a screening failure degrades to previously-known behavior rather than
# routing into the newer, less-exercised code path.
_FAILURE_FALLBACK: ShapeCheck = {
    "shape": "structured_license",
    "reasoning": "ระบบจำแนกรูปแบบเนื้อหาทำงานผิดพลาด — ใช้เส้นทางเดิม (13 ฟิลด์) เป็นค่าเริ่มต้น กรุณาตรวจสอบด้วยตนเอง",
}

_PROMPT_TEMPLATE = """เอกสารต่อไปนี้ถูกอัปโหลดเข้าระบบฐานความรู้ของ Restbiz (ผู้ช่วย AI ด้านกฎหมาย/ใบอนุญาต/
ข้อบังคับสำหรับธุรกิจร้านอาหารในไทย) หน้าที่ของคุณคือจำแนกว่าเอกสารนี้เหมาะกับรูปแบบไหน:

**"structured_license"** — เอกสารอธิบาย**ใบอนุญาต/การจดทะเบียนเรื่องเดียวที่ชัดเจน** มีโครงสร้างครบ
(หน่วยงานที่ออก, ขั้นตอน, เอกสารที่ต้องใช้, ค่าธรรมเนียม, ระยะเวลา) — เหมาะจะสรุปลงตารางฟิลด์ตายตัวได้
โดยไม่เสียเนื้อหาสำคัญไป

**"know_how"** — เอกสารเป็นเนื้อหาให้ความรู้/คำแนะนำทั่วไป (ไม่ใช่ขั้นตอนขอใบอนุญาตเรื่องเดียว),
หรือมี**หลายหัวข้อ/หลายประเด็นปนกันในไฟล์เดียว**, หรือเนื้อหายาว/ซับซ้อนเกินกว่าจะสรุปลงฟิลด์ตายตัว
โดยไม่เสียเนื้อหาไปมาก — เหมาะเก็บเป็นรายการความรู้แยกเรื่อง

ตอบเป็น JSON เท่านั้น:
{{
  "shape": "structured_license" | "know_how",
  "reasoning": "เหตุผลสั้นๆ 1-2 ประโยค"
}}

=== เนื้อหาเอกสารทุกหน้า ===
{combined_text}
"""


def classify_content_shape(pages_markdown: list[str]) -> ShapeCheck:
    """One LLM call over all pages combined. Never raises — see
    _FAILURE_FALLBACK for why the safe default is "structured_license"."""
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
        logger.error(f"[ContentShape] LLM call/parse failed, defaulting to 'structured_license': {e}")
        return _FAILURE_FALLBACK

    shape = parsed.get("shape")
    if shape not in _VALID_SHAPES:
        logger.warning(f"[ContentShape] LLM returned unexpected shape {shape!r}, defaulting to 'structured_license'")
        shape = "structured_license"

    reasoning = parsed.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    logger.info(f"[ContentShape] shape={shape}")
    return {"shape": shape, "reasoning": reasoning}

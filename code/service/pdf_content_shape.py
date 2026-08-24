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
from utils.prompt_safety import INJECTION_GUARD

logger = get_logger(__name__)

ContentShape = Literal["structured_license", "know_how"]


class ShapeCheck(TypedDict):
    shape: ContentShape
    reasoning: str
    # REVISED 2026-08-24: a document can genuinely mix both shapes (e.g. a
    # clean single-license procedure followed by unrelated general business
    # advice) — live-tested and confirmed this happens. Forcing the whole
    # document down ONE path silently drops whichever shape lost the vote:
    # a license buried inside a know_how-classified document never gets
    # candidate-matched against the regulatory Sheet at all. secondary_pages
    # is empty in the overwhelming majority (single-shape) case; only
    # populated when a SUBSTANTIAL, clearly-structured chunk of the OTHER
    # shape exists — not just an incidental one-sentence mention. See
    # sqs_consumer.py for how these pages get run through the other
    # pipeline too, producing additional item(s) alongside the primary one.
    secondary_pages: list[tuple[int, int]]


_VALID_SHAPES = ("structured_license", "know_how")

# Fail toward "structured_license" — that's the ORIGINAL, longer-proven path
# (13-field drafting existed and was tested before know-how drafting did), so
# a screening failure degrades to previously-known behavior rather than
# routing into the newer, less-exercised code path.
_FAILURE_FALLBACK: ShapeCheck = {
    "shape": "structured_license",
    "reasoning": "ระบบจำแนกรูปแบบเนื้อหาทำงานผิดพลาด — ใช้เส้นทางเดิม (13 ฟิลด์) เป็นค่าเริ่มต้น กรุณาตรวจสอบด้วยตนเอง",
    "secondary_pages": [],
}

_PROMPT_TEMPLATE = """เอกสารต่อไปนี้ (มีทั้งหมด {num_pages} หน้า) ถูกอัปโหลดเข้าระบบฐานความรู้ของ Restbiz
(ผู้ช่วย AI ด้านกฎหมาย/ใบอนุญาต/ข้อบังคับสำหรับธุรกิจร้านอาหารในไทย)

""" + INJECTION_GUARD + """

หน้าที่ของคุณมี 2 ส่วน:

**ส่วนที่ 1 — จำแนกรูปแบบหลักของเอกสาร:**

**"structured_license"** — เอกสารอธิบาย**ใบอนุญาต/การจดทะเบียนเรื่องเดียวที่ชัดเจน** มีโครงสร้างครบ
(หน่วยงานที่ออก, ขั้นตอน, เอกสารที่ต้องใช้, ค่าธรรมเนียม, ระยะเวลา) — เหมาะจะสรุปลงตารางฟิลด์ตายตัวได้
โดยไม่เสียเนื้อหาสำคัญไป

**"know_how"** — เอกสารเป็นเนื้อหาให้ความรู้/คำแนะนำทั่วไป (ไม่ใช่ขั้นตอนขอใบอนุญาตเรื่องเดียว),
หรือมี**หลายหัวข้อ/หลายประเด็นปนกันในไฟล์เดียว**, หรือเนื้อหายาว/ซับซ้อนเกินกว่าจะสรุปลงฟิลด์ตายตัว
โดยไม่เสียเนื้อหาไปมาก — เหมาะเก็บเป็นรายการความรู้แยกเรื่อง

**ส่วนที่ 2 — เนื้อหาส่วนน้อย (secondary) ของรูปแบบตรงข้าม (ถ้ามีจริง):**
ถ้าเอกสารมีเนื้อหา**ก้อนใหญ่ชัดเจน**ของรูปแบบตรงข้ามกับที่เลือกในส่วนที่ 1 ปนอยู่ด้วย (เช่น
เอกสารหลักเป็นคู่มือ know_how ทั่วไป แต่มีช่วงหนึ่งอธิบายขั้นตอนขอใบอนุญาตเรื่องหนึ่งอย่างครบถ้วนชัดเจน)
ให้ระบุช่วงหน้าของส่วนนั้นไว้ใน "secondary_pages" — **ใช้เฉพาะเมื่อเนื้อหานั้นสมบูรณ์พอจะแยกออกมาใช้ได้จริง
เท่านั้น** ถ้าเป็นแค่การพูดถึงผ่านๆ ประโยคเดียวหรือสองประโยค **อย่า**ใส่ใน secondary_pages (จะกลายเป็น
เสียงรบกวน ไม่มีเนื้อหาพอจะแยกออกมาจริง) ถ้าเอกสารเป็นรูปแบบเดียวล้วนๆ ให้ตอบ secondary_pages เป็น [] เสมอ

ตอบเป็น JSON เท่านั้น:
{{
  "shape": "structured_license" | "know_how",
  "reasoning": "เหตุผลสั้นๆ 1-2 ประโยค",
  "secondary_pages": [[6, 10]]
}}

=== เนื้อหาเอกสารทุกหน้า (มีเลขหน้ากำกับแต่ละส่วน) ===
{combined_text}
"""


def classify_content_shape(pages_markdown: list[str]) -> ShapeCheck:
    """One LLM call over all pages combined. Never raises — see
    _FAILURE_FALLBACK for why the safe default is "structured_license"."""
    numbered_pages = [f"[หน้า {i + 1}]\n{md}" for i, md in enumerate(pages_markdown)]
    combined_text = "\n\n---\n\n".join(numbered_pages)
    num_pages = len(pages_markdown)
    try:
        client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
        resp = client.chat.completions.create(
            model=conf.OPENROUTER_MODEL_PRACTICAL,
            messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(num_pages=num_pages, combined_text=combined_text)}],
            max_tokens=500,
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

    secondary_pages: list[tuple[int, int]] = []
    for raw_range in parsed.get("secondary_pages", []) or []:
        try:
            s, e = int(raw_range[0]), int(raw_range[1])
        except (TypeError, ValueError, IndexError):
            logger.warning(f"[ContentShape] Skipping malformed secondary_pages entry: {raw_range!r}")
            continue
        s, e = max(1, s), min(num_pages, e)
        if s <= e:
            secondary_pages.append((s, e))

    logger.info(f"[ContentShape] shape={shape} secondary_pages={secondary_pages}")
    return {"shape": shape, "reasoning": reasoning, "secondary_pages": secondary_pages}

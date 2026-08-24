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

import json
import re
from typing import TypedDict

from openai import OpenAI

import conf
from utils.logger import get_logger

logger = get_logger(__name__)


class KnowhowTopicBounds(TypedDict):
    document_title: str
    source_type: str  # "book" | "document"
    main_topic: str
    sub_topic: str
    summary: str
    category: str
    start_page: int
    end_page: int


_PROMPT_TEMPLATE = """เอกสารต่อไปนี้ (มีทั้งหมด {num_pages} หน้า) ถูกจัดว่าเป็นเนื้อหา know-how/ให้ความรู้
สำหรับฐานความรู้ของ Restbiz (ผู้ช่วย AI ด้านธุรกิจร้านอาหารในไทย) ซึ่งอาจมีหลายหัวข้อปนกันอยู่ในไฟล์เดียว

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


def identify_knowhow_topics(pages_markdown: list[str]) -> list[KnowhowTopicBounds]:
    """One LLM call. Returns [] (never raises) on any failure — the caller
    treats an empty list as "could not split into topics" and should fall
    back to treating the whole document as one topic rather than losing it
    entirely; see knowhow_write_back.py. document_title/source_type are
    denormalized onto every returned topic dict (same values repeated) so
    downstream code (knowhow_write_back.py) doesn't need a separate wrapper
    shape — every topic is independently routable/writable."""
    numbered_pages = [f"[หน้า {i + 1}]\n{md}" for i, md in enumerate(pages_markdown)]
    combined_text = "\n\n---\n\n".join(numbered_pages)

    try:
        client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
        resp = client.chat.completions.create(
            model=conf.OPENROUTER_MODEL_PRACTICAL,
            messages=[{
                "role": "user",
                "content": _PROMPT_TEMPLATE.format(num_pages=len(pages_markdown), combined_text=combined_text),
            }],
            max_tokens=2500,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        parsed = json.loads(raw)
    except Exception as e:
        logger.error(f"[KnowhowDrafting] LLM call/parse failed: {e}")
        return []

    if not isinstance(parsed, dict) or not isinstance(parsed.get("topics"), list):
        logger.error(f"[KnowhowDrafting] LLM returned unexpected shape: {type(parsed)}")
        return []

    document_title = str(parsed.get("document_title", "")).strip()
    source_type = str(parsed.get("source_type", "")).strip().lower()
    if source_type not in ("book", "document"):
        source_type = "document"  # fail toward the more general tab, not a guess at "book"

    num_pages = len(pages_markdown)
    topics: list[KnowhowTopicBounds] = []
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
        end_page = min(num_pages, end_page)
        if start_page > end_page or start_page > num_pages or end_page < 1:
            logger.warning(f"[KnowhowDrafting] Skipping topic with out-of-range pages {raw_topic.get('start_page')}-{raw_topic.get('end_page')} (doc has {num_pages} pages)")
            continue

        main_topic = str(raw_topic.get("main_topic", "")).strip()
        sub_topic = str(raw_topic.get("sub_topic", "")).strip()
        if not main_topic or not sub_topic:
            logger.warning(f"[KnowhowDrafting] Skipping topic missing required main_topic/sub_topic: {raw_topic!r}")
            continue

        topics.append({
            "document_title": document_title,
            "source_type": source_type,
            "main_topic": main_topic,
            "sub_topic": sub_topic,
            "summary": str(raw_topic.get("summary", "")).strip(),
            "category": str(raw_topic.get("category", "")).strip(),
            "start_page": start_page,
            "end_page": end_page,
        })

    logger.info(
        f"[KnowhowDrafting] {document_title!r} (source_type={source_type}): "
        f"identified {len(topics)} topic(s) across {num_pages} page(s)"
    )
    return topics

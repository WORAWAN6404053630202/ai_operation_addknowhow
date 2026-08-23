# code/service/pdf_knowhow_drafting.py
"""Topic-identification for know-how-shaped PDFs (feature/pdf-ingestion) — for
documents pdf_content_shape.py routed away from the structured 13-field path
(advisory content, multi-topic documents, anything that doesn't have a single
clean license-procedure shape).

Deliberately does NOT ask the LLM to reproduce full topic text — only topic
BOUNDARIES (which pages, title, short summary, category). The actual
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
    title: str
    summary: str
    category: str
    start_page: int
    end_page: int


_PROMPT_TEMPLATE = """เอกสารต่อไปนี้ (มีทั้งหมด {num_pages} หน้า) ถูกจัดว่าเป็นเนื้อหา know-how/ให้ความรู้
สำหรับฐานความรู้ของ Restbiz (ผู้ช่วย AI ด้านธุรกิจร้านอาหารในไทย) ซึ่งอาจมีหลายหัวข้อปนกันอยู่ในไฟล์เดียว

หน้าที่ของคุณ: **แบ่งเอกสารนี้ออกเป็นหัวข้อย่อยที่แยกจากกันได้ชัดเจน** (แต่ละหัวข้อควรเป็นเรื่องเดียวกัน
ต่อเนื่องกัน ไม่จำเป็นต้องมีหลายหัวข้อเสมอไป — ถ้าเอกสารทั้งฉบับเป็นเรื่องเดียวต่อเนื่องกัน ให้ตอบมาแค่
1 หัวข้อที่ครอบคลุมทุกหน้าก็ได้) แต่ละหน้าต้องอยู่ในหัวข้อใดหัวข้อหนึ่งเท่านั้น ห้ามข้ามหน้าหรือซ้อนทับกัน

**ห้ามคัดลอกเนื้อหาเต็มมาใส่ — ระบุแค่ขอบเขตหน้าและสรุปสั้นๆ เท่านั้น** (เนื้อหาเต็มจริงจะถูกตัดมาจาก
ต้นฉบับโดยตรงทีหลัง ไม่ต้องพิมพ์ซ้ำ)

ตอบเป็น JSON array เท่านั้น ไม่ต้องมีข้อความอื่น:
[
  {{
    "title": "ชื่อหัวข้อสั้นๆ กระชับ",
    "summary": "สรุปเนื้อหาของหัวข้อนี้ 2-4 ประโยค สำหรับใช้ค้นหา",
    "category": "หมวดหมู่สั้นๆ เช่น การตลาด, การเงิน, กฎหมายแรงงาน, สุขาภิบาล ฯลฯ",
    "start_page": 1,
    "end_page": 3
  }}
]

=== เนื้อหาเอกสารทุกหน้า (มีเลขหน้ากำกับแต่ละส่วน) ===
{combined_text}
"""


def identify_knowhow_topics(pages_markdown: list[str]) -> list[KnowhowTopicBounds]:
    """One LLM call. Returns [] (never raises) on any failure — the caller
    treats an empty list as "could not split into topics" and should fall
    back to treating the whole document as one topic rather than losing it
    entirely; see knowhow_write_back.py."""
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
            max_tokens=2000,
        )
        raw = (resp.choices[0].message.content or "[]").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        parsed = json.loads(raw)
    except Exception as e:
        logger.error(f"[KnowhowDrafting] LLM call/parse failed: {e}")
        return []

    if not isinstance(parsed, list):
        logger.error(f"[KnowhowDrafting] LLM returned non-list JSON: {type(parsed)}")
        return []

    num_pages = len(pages_markdown)
    topics: list[KnowhowTopicBounds] = []
    for raw_topic in parsed:
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

        topics.append({
            "title": str(raw_topic.get("title", "")).strip() or "(ไม่มีชื่อเรื่อง)",
            "summary": str(raw_topic.get("summary", "")).strip(),
            "category": str(raw_topic.get("category", "")).strip(),
            "start_page": start_page,
            "end_page": end_page,
        })

    logger.info(f"[KnowhowDrafting] Identified {len(topics)} topic(s) across {num_pages} page(s)")
    return topics

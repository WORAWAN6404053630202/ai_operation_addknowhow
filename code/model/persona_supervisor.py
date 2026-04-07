# code/model/persona_supervisor.py
"""
Persona Supervisor (Hybrid Routing + FSM) — Option A (Supervisor owns ALL visible messages)
========================================================================================

Fixe feedback:
1) Menu choices MUST be 5 items (target exactly 5; backfill if pool small).
2) Choices must be sampled from REAL data (metadata) and diversified (not top-only).
   - Build topic_pool once per session using MULTI broad queries (more coverage than 1 query).
3) Subsequent greeting/noise/thanks MUST show choices every time
   - but intro (name/role) only ONCE (first greeting of session).
4) Greeting should NOT repeat the same 2 options over and over
   - use per-session seed + per-greeting counter to rotate randomness deterministically.
5) Keep pending_slot active across greeting turns (do not clear on greeting).
6) Priority preserved:
   confirm/switch/intake lock > pending_slot > greeting/noise > legal routing

NEW (this change request): Option A production-ready menu
7) Make menu topics look like "หัวข้อ" more, by prioritizing metadata fields that correspond to:
   - ใบอนุญาต
   - การดำเนินการตามหน่วยงาน
   - หัวข้อการดำเนินการย่อย
   and de-prioritizing pure "หน่วยงาน/department" labels unless needed as backfill.

NEW (your latest request): "quality gate + level separation + safe fallback"
- Quality gate:
  - reject obvious noise/placeholder/org-only labels aggressively (but still allow as last-resort backfill)
  - require "menu-worthy" signals for main menu (permit/procedure/tax/docs/fees/time/channel/etc.)
- Level separation:
  - keep "case-specific / detail-ish" labels OUT of main menu (put them only in drill-down / Phase 3 later)
- Safe fallback:
  - always guarantee exactly 5 menu items even if data is sparse

CURRENT BEHAVIOR (silent switch, no confirmation dialogs):
- If user asks for "ละเอียด/เชิงลึก/วิชาการ" OR hints Academic persona:
  - SWITCH IMMEDIATELY (silent, no announcement)
  - LLM is primary detector (not regex-gated)
  - Academic answers → auto-returns to Practical silently
- If user says "change/switch persona/mode" (with/without target):
  - SWITCH IMMEDIATELY to academic if that's the target, no confirmation

Professional Logging:
- ใช้ structured logging แทน plain text logs
- บันทึก request_id, session_id, performance metrics
- อ่านง่าย เข้าใจได้ทันที
- Academic resume: re-enter academic silently if user wants to continue previous topic
- No "กลับมาโหมด Practical แล้วครับ" announcements
- Academic final_answer 3-branch logic stays in AcademicPersonaService (no changes here)

BUGFIX (this request):
- pending_slot ที่เป็นตัวเลือก 2-5 ข้อ (เช่น location: กรุงเทพฯ/ต่างจังหวัด)
  ต้อง "รับคำตอบแบบข้อความ" ได้ด้วย (เช่น "กทม", "กรุงเทพ", "ต่างจังหวัด", "จังหวัด")
  โดย *ไม่ hardcode* — ใช้ LLM map user_text -> option ที่ใกล้ที่สุดแบบ deterministic (temp=0)
"""

from __future__ import annotations

from typing import Tuple, Callable, Optional, Dict, Any, List
import logging
import re
import json
import random
import hashlib

_LOG = logging.getLogger("restbiz.supervisor")

# Professional logging system
from utils.logger import get_logger, log_function_call, TimingContext
logger = get_logger(__name__)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import conf
from model.conversation_state import ConversationState
from utils.llm_call import llm_invoke, extract_llm_text
from utils.prompts_supervisor import (
    build_topic_picker_prompt,
    build_confirm_prompt,
    build_style_detect_prompt,
    build_greet_prefix_prompt,
    build_greet_kind_instructions,
    build_op_group_classifier_prompt,
    build_deduplicate_options_prompt,
    build_slot_mapper_prompt,
    build_fallback_intent_prompt,
    build_typo_check_prompt,
    build_topic_desc_prompt,
)
from utils.persona_profile import (
    normalize_persona_id,
    build_strict_profile,
    apply_persona_profile,
)

from model.persona_academic import AcademicPersonaService
from model.persona_practical import PracticalPersonaService


class PersonaSupervisor:
    """
    Central orchestrator for persona-based conversation.
    Contract: handle(state, user_input) -> (state, reply_text)
    """

    # Thai ending normalization
    _DUAL_ENDING_RE = re.compile(r"(ครับ\s*/\s*ค่ะ|ค่ะ\s*/\s*ครับ)")
    _FEMALE_ENDING_TOKEN_RE = re.compile(r"(?<![ก-๙])ค่ะ(?![ก-๙])")

    def _normalize_male(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return t
        t = self._DUAL_ENDING_RE.sub("ครับ", t)
        t = self._FEMALE_ENDING_TOKEN_RE.sub("ครับ", t)
        return t

    # Intent / Priority Matrix
    INTENT_CONFIRM_YESNO = "CONFIRM_YESNO"
    INTENT_EXPLICIT_SWITCH = "EXPLICIT_SWITCH"
    INTENT_MODE_STATUS = "MODE_STATUS"
    INTENT_ACAD_INTAKE_REPLY = "ACADEMIC_INTAKE_REPLY"
    INTENT_LEGAL_NEW = "LEGAL_NEW_QUESTION"
    INTENT_GREETING = "GREETING_SMALLTALK"
    INTENT_NOISE = "NOISE"
    INTENT_UNKNOWN = "UNKNOWN"

    # FSM states (kept for compatibility)
    S_IDLE = "S_IDLE"
    S_SWITCH_CONFIRM = "S_SWITCH_CONFIRM"
    S_PRACTICAL_ANSWER = "S_PRACTICAL_ANSWER"
    S_ACAD_INTAKE = "S_ACAD_INTAKE"
    S_ACAD_ANSWER = "S_ACAD_ANSWER"
    S_AUTO_RETURN = "S_AUTO_RETURN"

    # Deterministic detectors
    _MODE_STATUS_Q = re.compile(
        r"(ตอนนี้|ตอนนี้เรา|ตอนนี้บอท|บอทตอนนี้|อยู่|เป็น)\s*.*(โหมด|mode|persona|บุคลิก).*"
        r"|^(โหมด|mode|persona|บุคลิก)\s*(อะไร|ไหน|ไร|หยัง|ไหนอะ|ไหนครับ|ไหนคะ)?\s*\??$",
        re.IGNORECASE,
    )

    _SWITCH_VERBS = ("เปลี่ยน", "สลับ", "ปรับ", "ขอเปลี่ยน", "ขอสลับ", "ขอปรับ", "change", "switch", "ไป")
    _SWITCH_MARKERS = ("โหมด", "mode", "persona", "บุคลิก", "บอท", "bot", "ตัว")

    # Only very specific phrases that unambiguously signal "I want Academic/deep-dive mode".
    # Broad words like "ละเอียด", "มากกว่านี้", "บอกเพิ่มเติม" are intentionally excluded —
    # they are too common in general Thai speech and would cause false positives.
    # Natural phrasing that doesn't match here is caught by the LLM (primary detector).
    _TARGET_ACADEMIC_HINTS = (
        "เชิงลึก",
        "วิชาการ",
        "อ้างอิงข้อกฎหมาย",
        "อธิบายละเอียด",
        "ขอแบบละเอียด",
        "แบบละเอียด",
        "ละเอียดกว่านี้",
        "ละเอียดกว่า",
        "ละเอียดหน่อย",
        "ละเอียดขึ้น",
        "ละเอียดทั้งหมด",
        "ขอแบบละเอียดทั้งหมด",
        "เอาแบบละเอียดทั้งหมด",
        "ขยายความ",
        "ลงรายละเอียด",
    )
    # Only specific phrases that unambiguously signal "I want short/summary mode".
    # "สั้น" alone (e.g. "ข้อความสั้น") and "เร็วๆ" (e.g. "เร็วๆ นี้") are excluded — too broad.
    # LLM handles ambiguous natural phrasing as primary detector.
    _TARGET_PRACTICAL_HINTS = (
        "สั้นๆ",
        "กระชับ",
        "สรุป",
        "สรุปสั้น",
        "เอาแบบสั้น",
        "เอาแบบสรุป",
        "เช็คลิสต์",
        "เป็นข้อๆ",
    )

    _STYLE_LIKELY_RE = re.compile(
        r"(ขอ|ช่วย|รบกวน|เอา|อยากได้|ขอให้|ช่วยอธิบาย|ขยายความ|ลงรายละเอียด|ละเอียดขึ้น|เชิงลึก|สรุป|สั้นๆ|กระชับ)",
        re.IGNORECASE,
    )

    _SMALLTALK_RE = re.compile(
        r"(ทำอะไรอยู่|ทำไรอยู่|ว่างไหม|อยู่ไหม|เป็นไงบ้าง|เป็นไง|กินข้าวยัง|สบายดีไหม|สบายดีปะ|โอเคไหม|เหนื่อยไหม)",
        re.IGNORECASE,
    )
    # Personal questions directed AT the bot (greeting context but off-domain)
    _PERSONAL_Q_RE = re.compile(
        r"คุณ(จะ|ชอบ|ไป|มี|รู้สึก|เคย|อยาก|คิด|รู้|เป็น|ทำ|กิน|ดู|ฟัง|เล่น|พูด|บอก|แนะนำ|โปรด)",
        re.IGNORECASE,
    )
    _THANKS_RE = re.compile(r"(ขอบคุณ|ขอบใจ|thx|thanks)\b", re.IGNORECASE)

    _LIKELY_SELECTION_RE = re.compile(r"^\s*[\d\s,/-]+\s*$")

    _QUESTION_MARKERS_RE = re.compile(
        r"(\?|\bไหม\b|หรือไม่|หรือเปล่า|ยังไง|ทำไง|อย่างไร|ได้ไหม|ควร|ต้อง|คืออะไร"
        r"|วิธีการ|วิธี(?!ีเดียว|ีชีวิต))",
        re.IGNORECASE,
    )

    _LEGAL_SIGNAL_RE = re.compile(
        r"(ใบอนุญาต|จดทะเบียน|ทะเบียนพาณิชย์|ภาษี|vat|ภพ\.?20|สรรพากร|เทศบาล|สำนักงานเขต|สุขาภิบาล|กรม|ค่าธรรมเนียม|เอกสาร|ขั้นตอน|บทลงโทษ|ประกาศ|พ\.ร\.บ|ประกันสังคม|กองทุน|เปิดร้าน|ขึ้นทะเบียน"
        r"|qr.?pay|qr.?payment|คิวอาร์|คิวอาเพย|เพย์เมนต์|edc|รูดบัตร|merchant.?id|partner.?id|pos.?id|ระบบชำระเงิน|กสิกร|kbank|ไทยพาณิชย์|scb|ประกอบกิจการ|สุขาภิบาลอาหาร"
        # Domain entities: company/business types (จดทะเบียน/หุ้น/ห้างฯ all in-scope)
        r"|หุ้น|นิติบุคคล|บุคคลธรรมดา|บริษัท|มหาชนจำกัด|ห้างหุ้นส่วน|พาณิชยกิจ"
        # Shop/business operations (in-scope topics about running the business)
        r"|ร้านค้า|กิจการ|พนักงาน"
        # Payment / banking topics (QR, refund, mobile banking changes)
        r"|บัญชีธนาคาร|คืนเงิน|ยกเลิกบิล|ยกเลิกใช้บริการ|refund|wechat"
        # Document/meeting terms in business registration context
        r"|หนังสือนัดประชุม|บริคณห์|มอบฉันทะ|มาตรฐานร้านอาหาร|san\b"
        # Remaining generic operational topics (สุรา/ภาษี context: ชื่อและที่อยู่, เพิ่ม/ลด, สำนักงาน, เบอร์)
        r"|สำนักงาน|เบอร์มือถือ|แก้ไขเปลี่ยนแปลง|ชื่อและที่อยู่|เพิ่ม.*ลด|ลด.*เพิ่ม|ประเภทสินค้า|จำนวนชื่อ"
        # Sub-step/case phrases in registration procedures (กรณีเลือก... etc.)
        r"|กรณี"
        # Consequence/impact/risk topics — user asking about effects/penalties/risks
        r"|งบการเงิน|ผลกระทบ|ความเสี่ยง)",
        re.IGNORECASE,
    )

    _NOISE_ONLY_RE = re.compile(r"^(?:[a-z]+|[!?.]+)$", re.IGNORECASE)
    _TH_LAUGH_5_RE = re.compile(r"^\s*5{3,}\s*$")

    # Depth/detail requests — signals user wants more elaboration on current topic.
    # These must NOT be deflected by 2.2c and must be routed to Academic persona.
    _DEPTH_DETAIL_RE = re.compile(
        r"(แบ.?ละเอียด|ละเอียดกว่า|ละเอียดมากกว่า|ละเอียดหน่อย|ละเอียดขึ้น|เชิงลึก|แบบเต็ม"
        r"|ครบถ้วน|ทั้งหมดเลย|แบบวิชาการ|อธิบายละเอียด|แบบเต็มๆ"
        r"|ขอดูแบบละเอียด|ขอรายละเอียด|อธิบายเพิ่ม|อธิบายต่อ|อธิบายให้ชัดเจนขึ้น"
        r"|ขอเพิ่มเติม|ดูเพิ่มเติม|รายละเอียดเพิ่ม|แบบให้ลึกซึ้งยิ่งขึ้น|มากกว่านี้)",
        re.IGNORECASE,
    )

    # Follow-up patterns (handled before fallback_safe_return)
    _ELABORATE_RE = re.compile(
        r"(อธิบาย(มากกว่า|เพิ่ม|เพิ่มเติม|ขยาย|ให้ละเอียด|ต่อ)|ขยายความ|เพิ่มเติมอีก|รายละเอียดมากกว่า|บอกเพิ่ม|เล่าให้ฟัง|อธิบายต่อ|รายละเอียดเพิ่ม"
        r"|กลับไปเรื่องเดิม|กลับเรื่องเก่า|กลับเรื่องเดิม|กลับเรื่องที่คุย|คุยต่อเรื่องเดิม|ขอกลับไปเรื่อง|อยากคุยต่อ)",
        re.IGNORECASE,
    )

    # Patterns for resuming an Academic session after auto-return to Practical.
    # IMPORTANT: Keep only phrases that unambiguously signal "continue the SAME previous academic session".
    # Broad words like "ทั้งหมด" (could mean anything), "เรื่องเก่า" (vague),
    # "ขอกลับไป" (no qualifier), "ต้องการทราบเพิ่ม" (generic request),
    # "ขอรายละเอียดเพิ่ม", "ขอข้อมูลเพิ่มเติม" (could be new topic) are excluded.
    # Ambiguous natural phrasing falls through to LLM fallback (llm_fallback_intent_call).
    _ACADEMIC_RESUME_RE = re.compile(
        r"(ขอทั้งหมด|ดูทั้งหมด|ส่วนที่เหลือ|ขอส่วนอื่น|อยากรู้ส่วนอื่น"
        r"|อยากรู้ต่อ|อยากดูต่อ|ขอต่อจากเดิม|อยากรู้เพิ่มเรื่องนี้|ยังอยากรู้"
        r"|อยากได้ส่วน|อยากถามต่อ"
        r"|กลับไปเรื่องเดิม|กลับไปเรื่องเก่า|ต่อจากที่แล้ว)",
        re.IGNORECASE,
    )
    _NEW_TOPIC_RE = re.compile(
        r"(ขอหัวข้อ(ใหม่|อื่น)|แนะนำหัวข้อ|มีเรื่องอื่น(อีก)?|เรื่องอื่น(อีก)?|หัวข้ออื่น|เรื่องไหนอีก|อยากรู้เรื่องอื่น|ต้(อง|แ)การรู้เรื่องอื่น|มีอะไรอีก(มั้ย|ไหม)?|แนะนำเรื่องอื่น)",
        re.IGNORECASE,
    )
    _FOLLOWUP_CONTEXTUAL_RE = re.compile(
        r"^(อันไหน|แบบไหน|กรณีไหน|ของ(ฉัน|ผม|หนู)|แบบ(ฉัน|ผม|หนู)|กรณีของ|ที่เหมาะกับ|ที่ใช้ได้กับ|สำหรับ(ฉัน|ผม|หนู|กรณีนี้|ประเภทนี้))",
        re.IGNORECASE,
    )
    # Link/document request — user asking for URLs, forms, guides, or downloads
    # Used to override new_topic routing when there's an active context
    _LINK_REQUEST_RE = re.compile(
        r"(ขอลิงค์|ขอลิงก์|ขอลิ้งค์|ขอลิ้งก์"
        r"|ส่งลิงค์|ส่งลิงก์|ส่งลิ้งค์|ส่งลิ้งก์"
        r"|ลิงค์คู่มือ|ลิงก์คู่มือ|ลิ้งค์คู่มือ|ลิ้งก์คู่มือ"
        r"|ลิงค์แบบฟอร์ม|ลิงก์แบบฟอร์ม|ขอดูลิงค์|ขอดูลิงก์"
        r"|ลิงค์(ที่|ของ|ด้วย|สำหรับ)|ลิงก์(ที่|ของ|ด้วย|สำหรับ)"
        r"|ลิ้งค์(ที่|ของ|ด้วย|สำหรับ)|ลิ้งก์(ที่|ของ|ด้วย|สำหรับ)"
        r"|ต้องการลิ้งค์|ต้องการลิ้งก์|ต้องการลิงค์|ต้องการลิงก์"
        r"|URL|ดาวน์โหลด(เอกสาร|แบบฟอร์ม|คู่มือ|ไฟล์)"
        r"|ขอคู่มือ|ขอดูคู่มือ|ส่งคู่มือ|คู่มือ(การ|สำหรับ|ของ)"
        r"|ขอแบบฟอร์ม|ส่งแบบฟอร์ม|แบบฟอร์ม(ที่|ของ|ด้วย))",
        re.IGNORECASE,
    )
    # Short Thai interjections that are not legal questions → re-show menu
    _TH_INTERJECTION_RE = re.compile(
        r"^\s*(เอ้|เฮ้|เฮ|โอ้|โอ้โห|อ้าว|อ้าว|ว้าว|เออ|เอ่อ|อ่า|อ้า|อืม|อ๋อ|อ๋อ|เออนะ|เอ้าๆ|งั้นหรอ|งั้นเหรอ|จริงดิ|จริงเหรอ|ไม่ใช่เหรอ|เหรอ)\s*(ครับ|คับ|ค่ะ|คะ|นะ|นะครับ|นะคะ)?\s*$",
        re.IGNORECASE,
    )
    # Operation inference: detect from user query which operation_group is clearly implied
    _OP_INFER_NEW_RE = re.compile(
        r"จด\s*(vat|ภาษีมูลค่าเพิ่ม|ภพ\b|ทะเบียน|ใหม่)"
        r"|ต้อง\s*จด|ขอ\s*จด|ยื่นขอ\s*ใหม่|จดทะเบียน\s*ใหม่|ตั้งใหม่"
        r"|เริ่ม\s*จด|ควรจด|จะจด|จด\s*ตอนไหน|ต้องสมัคร\s*(vat|ภาษี|ภพ)",
        re.IGNORECASE,
    )
    _OP_INFER_EDIT_RE = re.compile(
        r"แก้ไข\s*(รายการ|ข้อมูล|ที่อยู่|วัตถุประสงค์|ชื่อ)|เปลี่ยนแปลง\s*(รายการ|ที่อยู่|ชื่อ)",
        re.IGNORECASE,
    )
    _OP_INFER_CANCEL_RE = re.compile(
        r"ยกเลิก\s*(การจด|ภาษี|vat|ภพ|ทะเบียน)|เลิก\s*กิจการ|ปิด\s*กิจการ|จะ\s*ยกเลิก",
        re.IGNORECASE,
    )
    # Renewal / expiry inference: "ต่ออายุ", "มีอายุกี่ปี", "หมดอายุ" etc.
    _OP_INFER_RENEW_RE = re.compile(
        r"ต่ออายุ|ต้องต่อ(?:\s*ใบ|\s*อนุญาต|\s*ทะเบียน)?|ต่อ\s*ใบอนุญาต|ต่อ\s*ทะเบียน"
        r"|(?:ใบ|อนุญาต|ทะเบียน|ภาษี).{0,20}(?:มีอายุ|อายุ.{0,15}กี่|หมดอายุ|สิ้นอายุ)"
        r"|มีอายุ(?:\s*กี่|\s*ปี|\s*เดือน|\s*วัน)?|หมดอายุ|สิ้นอายุ|วันหมดอายุ",
        re.IGNORECASE,
    )
    # Universal-fact query: answer does NOT differ by entity_type.
    # e.g. "อายุใบทะเบียนพาณิชย์ มีอายุกี่ปี" → same answer for บุคคลธรรมดา and นิติบุคคล.
    # Matched queries skip entity_type / registration_type slot questions.
    _UNIVERSAL_FACT_RE = re.compile(
        r"มีอายุ|หมดอายุ|สิ้นอายุ|วันหมดอายุ"
        r"|ต้องต่ออายุ(?:\s*ไหม|\s*หรือเปล่า|\s*หรือไม่)?"
        r"|(?:ใบ|อนุญาต|ทะเบียน|ภาษี).{0,20}(?:อายุ.{0,15}กี่|มีอายุ|หมดอายุ)",
        re.IGNORECASE,
    )
    # Entity type explicit in query text — used for auto-inference without asking
    _ENTITY_NITI_RE = re.compile(
        r"นิติบุคคล|บริษัท(?:\s*จำกัด|\s*มหาชน)?|ห้างหุ้นส่วน(?:\s*จำกัด|\s*สามัญ)?",
        re.IGNORECASE,
    )
    _ENTITY_NATURAL_RE = re.compile(
        r"บุคคลธรรมดา|ร้านค้าทั่วไป|เจ้าของ\s*คนเดียว",
        re.IGNORECASE,
    )

    # Slot question strings — defined once, reused everywhere
    _Q_OPERATION_GROUP = "ต้องการดำเนินการเรื่องใดครับ?"
    _Q_REGISTRATION_TYPE = "รูปแบบการจดทะเบียนของคุณเป็นแบบใดครับ?"

    # Multi-topic license detection: maps regex pattern → canonical license_type name in Chroma.
    # Used by _detect_license_types_from_query() to predict which licenses a query covers BEFORE retrieval.
    _MULTI_TOPIC_LICENSE_KEYWORDS: List[Tuple[str, str]] = [
        (r"vat|ภาษีมูลค่าเพิ่ม|ภพ\.?20|จด\s*ภาษี", "ใบภาษีมูลค่าเพิ่ม ภพ.20"),
        (r"ใบอนุญาตจำหน่ายสุรา|ขายสุรา|จำหน่ายสุรา|สุรา|เหล้า|ภส\.08", "ใบอนุญาตจำหน่ายสุรา"),
        (r"ภาษีป้าย|ป้ายร้าน|ภป\.1|ป้ายโฆษณา", "แบบแสดงรายการภาษีป้ายร้านอาหาร"),
        (r"ใบอนุญาตจัดตั้ง|สถานที่จำหน่ายอาหาร|bma\s*oss|อาหาร.*ใบอนุญาต|ใบอนุญาต.*อาหาร", "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร"),
        (r"ใบทะเบียนพาณิชย์|จดทะเบียนพาณิชย์|ทะเบียนพาณิชย์|dbd", "ใบทะเบียนพาณิชย์"),
        (r"ประกันสังคม|กองทุนประกันสังคม|ขึ้นทะเบียนประกัน", "การขึ้นทะเบียนกองทุนประกันสังคม"),
        (r"ใบวุฒิบัตร|อบรมผู้สัมผัสอาหาร|ผู้สัมผัสอาหาร", "ใบวุฒิบัตรผู้สัมผัสอาหาร"),
        (r"ใบรับรองแพทย์|9\s*โรค|สณ\.11", "ใบรับรองแพทย์ 9 โรค(สณ.11)"),
        (r"qr.?payment|qr.?api|พร้อมเพย์|promptpay|qr\s*พร้อมเพย์|ระบบชำระเงินออนไลน์", "ระบบชำระเงินออนไลน์"),
        (r"edc|รูดบัตร|เครื่องรูดบัตร", "เครื่องรูดบัตร EDC"),
    ]

    # --------------------------
    # Typo / garbled input detection
    # --------------------------

    # Thai characters that cannot stand alone as meaningful input:
    # - U+0E30–U+0E4E: Thai vowel signs, tone marks, sara characters
    #   (these are combining/dependent marks — alone they are garbled)
    # - "ๅ" (U+0E45 sara ae standalone, but visually looks like random char when isolated)
    # - U+0E50–U+0E59: Thai digits (alone could be a selection → handled by _LIKELY_SELECTION_RE)
    # A "standalone diacritic" pattern: 1-3 chars all from Thai combining ranges
    _STANDALONE_TH_DIACRITIC_RE = re.compile(
        r"^[\u0E30-\u0E4E\u0E47-\u0E4E]{1,3}$"
    )
    # Garbled: only punctuation/symbols, or repeated non-word chars, or random ASCII+Thai mix
    # that a real user would never intentionally type
    _ALL_PUNCTUATION_RE = re.compile(r"^[^\w\u0E00-\u0E7F]+$")
    # Looks like keyboard mash: sequence of Thai consonants ONLY (no vowel signs at all).
    # Real Thai words always include at least one vowel mark — consonants alone cannot form
    # a valid syllable (e.g., "กขค", "ภคม").  2–6 chars so single-char inputs (already caught
    # by _STANDALONE_TH_DIACRITIC_RE / single-char rule) don't double-count.
    _TH_CONSONANT_MASH_RE = re.compile(
        r"^[\u0E01-\u0E2E]{2,6}$"
    )

    # --------------------------
    # Confirmation classifier
    # --------------------------
    _YES_CORE = (
        "ใช่", "ช่าย", "ไช่", "ใข่", "ชั่ย", "ชัย", "ชัยๆ",
        "ยืนยัน", "คอนเฟิร์ม", "confirm",
        "ถูกต้อง", "ถูกแล้ว", "โอเค", "ตกลง", "ใช่เลย", "ใช่แล้ว",
        "ได้", "ได้เลย", "ต้องการ",
        "เอา", "เอาเลย", "จัดไป", "ไปเลย",
        "yes", "yeah", "yep", "yup", "ok", "okay",
        "เยส", "เยป", "เย้ป",
        "งับ", "ค้าบ", "คั้บ", "เออ", "อือ", "อืม",
        "แน่นอน", "เลย", "เปลี่ยน",
    )
    _NO_CORE = (
        "ไม่", "ไม่เอา", "ไม่ต้อง", "ยังไม่", "ยกเลิก", "ช่างมัน",
        "no", "nope", "cancel",
        "ไม่เปลี่ยน", "ไม่สลับ",
    )

    # Helpers: message append (Supervisor ONLY)
    def _add_user(self, state: ConversationState, text: str) -> None:
        state.messages = state.messages or []
        if not text:
            return
        if state.messages and state.messages[-1].get("role") == "user" and (state.messages[-1].get("content") or "").strip() == text.strip():
            return
        state.messages.append({"role": "user", "content": text})

    def _add_assistant(self, state: ConversationState, text: str) -> None:
        state.messages = state.messages or []
        if not text:
            return
        if state.messages and state.messages[-1].get("role") == "assistant" and (state.messages[-1].get("content") or "").strip() == text.strip():
            return
        state.messages.append({"role": "assistant", "content": text})
        _LOG.debug("[Supervisor] assistant_msg_len=%d", len(text))

    def _handle_deflect(self, state: ConversationState, raw_input: str) -> Tuple[ConversationState, str]:
        """
        Guardrail deflection — LLM-guided response for off-topic / personal questions.
        Used by: 2.2c early-off-topic, greeting+personal-question, unknown_intent.
        """
        _LOG.info("[Supervisor] deflect input=%r", (raw_input or "")[:60])
        try:
            _llm = ChatOpenAI(
                model=getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL),
                openai_api_key=conf.OPENROUTER_API_KEY,
                openai_api_base=conf.OPENROUTER_BASE_URL,
                temperature=0.7,
                max_tokens=200,
                request_timeout=int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30)),
            )
            _system = (
                "คุณคือ 'น้องสุดยอด' — ที่ปรึกษาธุรกิจร้านอาหารครบวงจร\n"
                "ช่วยได้ทั้ง: กฎหมาย/ใบอนุญาต/ภาษี, การตลาด/กลยุทธ์, การเปิดร้าน/เบเกอรี่/คาเฟ่\n"
                "พูดเป็นกันเอง ตรงไปตรงมา ไม่เป็นทางการ\n\n"
                "กฎเด็ดขาด:\n"
                "- ตอบเป็น plain text บรรทัดเดียว ห้ามขึ้นบรรทัดใหม่\n"
                "- ไม่เกิน 40 คำ จบประโยคสมบูรณ์ก่อนหยุด\n"
                "- ใช้ 'ผม' ลงท้าย 'ครับ'\n"
                "- ห้ามสรุปหรือพูดถึงกฎที่ได้รับมา\n"
                "- รับ vibe สั้นๆ แล้ว redirect ว่าช่วยได้เรื่องอะไร\n"
            )
            _reply = extract_llm_text(
                llm_invoke(_llm, [SystemMessage(content=_system), HumanMessage(content=raw_input)],
                           logger=_LOG, label="Supervisor/deflect")
            ).strip()
            # Hard-enforce single line — collapse any newlines the LLM sneaks in
            _reply = " ".join(_reply.splitlines()).strip()
        except Exception as _e:
            _LOG.warning("[Supervisor] deflect LLM failed: %s", _e)
            _reply = "ผมช่วยได้ทั้งเรื่องกฎหมาย/ใบอนุญาต การตลาด และการเปิดร้านครับ มีอะไรอยากถามไหมครับ? 😊"
        _reply = self._normalize_male(_reply)
        self._add_assistant(state, _reply)
        state.last_action = "unknown_deflect"
        return state, _reply

    def _normalize_for_intent(self, s: str) -> str:
        t = (s or "").strip().lower()
        t = re.sub(r"[!！?？。,，]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
        return t.strip()

    def _normalize_confirm_text(self, s: str) -> str:
        t = self._normalize_for_intent(s)
        t = re.sub(r"[^\w\u0E00-\u0E7F\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _classify_yes_no_det(self, user_text: str) -> Dict[str, Any]:
        t = self._normalize_confirm_text(user_text)
        if not t:
            return {"yes": False, "no": False, "confidence": 0.0, "method": "empty"}

        if re.fullmatch(r"1", t):
            return {"yes": True, "no": False, "confidence": 0.95, "method": "num_yes"}
        if re.fullmatch(r"2", t):
            return {"yes": False, "no": True, "confidence": 0.95, "method": "num_no"}

        # "ครับผม" = strong polite yes
        if re.search(r"ครับผม|ครับ\s*ผม", t):
            return {"yes": True, "no": False, "confidence": 0.92, "method": "det_krabphom"}

        # Extended/repeated particles like "จ้าา", "ค่ะๆ", "ครับๆ" are usually yes
        if re.fullmatch(r"(จ้า+|ครับ+ๆ*|ค่ะ+ๆ*|ใช่ๆ*)", t):
            return {"yes": True, "no": False, "confidence": 0.85, "method": "det_particle_yes"}

        # Pure filler without other signals → unclear (needs LLM)
        if re.fullmatch(r"(ครับ|คับ|ค่ะ|คะ)", t):
            return {"yes": False, "no": False, "confidence": 0.0, "method": "filler_only"}

        def _has_any(tokens) -> bool:
            for tok in tokens:
                if tok and tok in t:
                    return True
            return False

        yes = _has_any(self._YES_CORE)
        no = _has_any(self._NO_CORE)

        if yes and no:
            return {"yes": False, "no": False, "confidence": 0.0, "method": "conflict"}

        if yes:
            return {"yes": True, "no": False, "confidence": 0.86, "method": "det_contains"}
        if no:
            return {"yes": False, "no": True, "confidence": 0.86, "method": "det_contains"}

        return {"yes": False, "no": False, "confidence": 0.0, "method": "unclear"}

    # LLM helpers (confirm/style + greet prefix + topic picker + slot mapper)
    def _default_topic_picker_llm_call(self) -> Callable[[str, List[str], int, List[str]], dict]:
        """
        Pick k topics from candidates, given context hint.
        Return JSON: {"topics": ["...", ...], "confidence": 0.0-1.0}
        """
        # Use dedicated fast model + short timeout for topic_picker (non-critical, fail fast)
        topic_model = getattr(conf, "OPENROUTER_MODEL_TOPIC_PICKER", getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL))
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=topic_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=512,  # topic list JSON ต้องการพอ (5 topics + confidence + reasoning)
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(last_hint: str, candidates: List[str], k: int, banned: List[str]) -> dict:
            cand = [str(x).strip() for x in (candidates or []) if str(x).strip()]
            cand = cand[:40]
            banned2 = [str(x).strip() for x in (banned or []) if str(x).strip()]
            banned2 = banned2[:60]

            prompt = build_topic_picker_prompt(last_hint, k, banned2, cand)

            try:
                text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/topic_picker")).strip()
            except Exception as e:
                # LengthFinishReasonError: retry ไม่ช่วย — skip แล้วใช้ fallback
                if "LengthFinishReasonError" in type(e).__name__ or "LengthFinishReason" in str(e)[:80]:
                    _LOG.warning("[Supervisor/topic_picker] LengthFinishReasonError — max_tokens น้อยเกินไป, skip retry")
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_confirm_llm_call(self) -> Callable[[str], dict]:
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=180,  # Increased from 150 to accommodate yes/no + reasoning
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_confirm_prompt(user_text)
            try:
                text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/llm")).strip()
            except Exception:
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_style_llm_call(self) -> Callable[[str], dict]:
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=250,  # Increased from 200 to accommodate analysis + reasoning
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_style_detect_prompt(user_text)
            try:
                text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/llm")).strip()
            except Exception:
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_greet_prefix_llm_call(self) -> Callable[[str, str, str, bool], dict]:
        """
        LLM returns ONLY the prefix (no numbered menu).
        include_intro:
          - True  => include name/role exactly once (first greeting only)
          - False => do NOT mention name/role anymore
        Return JSON: {"prefix": "..."}
        """
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.35,
            max_tokens=200,  # Increased from 120 to accommodate context-aware greetings
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(kind: str, persona_id: str, last_topic_hint: str, include_intro: bool) -> dict:
            kind_instructions = build_greet_kind_instructions(kind, last_topic_hint)
            prompt = build_greet_prefix_prompt(kind, persona_id, last_topic_hint, include_intro, kind_instructions)
            try:
                text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/llm")).strip()
            except Exception:
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_op_group_classifier_llm_call(self) -> Callable[[str, List[str]], dict]:
        """
        LLM-based operation group classifier.
        Given a license_type and a list of raw operation_by_department values from ChromaDB,
        returns a grouped structure: {display_label: [raw_values]}.

        This replaces ALL hardcoded prefix rules — works for any new license type added to
        ChromaDB in the future without any code changes.

        Uses fast topic-picker model + JSON mode. Result is cached per (license_type, entity_type)
        so LLM is called at most once per unique key per process lifetime.

        Return JSON:
        {
          "groups": [
            {"label": "ยื่นขอใหม่ / จดทะเบียน", "raw": ["การจดทะเบียนพาณิชย์"]},
            {"label": "ต่ออายุ",                 "raw": ["อายุใบทะเบียนพาณิชย์"]},
            ...
          ]
        }
        """
        topic_model = getattr(conf, "OPENROUTER_MODEL_TOPIC_PICKER",
                              getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL))
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        llm = ChatOpenAI(
            model=topic_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=1200,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(license_type: str, raw_ops: List[str]) -> dict:
            if not raw_ops:
                return {"groups": []}
            prompt = build_op_group_classifier_prompt(license_type, raw_ops)
            try:
                text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)],
                                  logger=_LOG, label="Supervisor/op_group_classify")).strip()
                text = self._strip_code_fences(text)
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {"groups": []}
            except Exception as e:
                _LOG.warning("[Supervisor] op_group_classifier LLM failed: %s", e)
                return {}

        return _call

    def _default_deduplicate_options_llm_call(self) -> Callable[[List[str]], dict]:
        """
        LLM-based deduplication of similar/redundant options.
        Return JSON: {"unique_options": ["..."], "reasoning": "..."}
        - Groups semantically similar options
        - Prefers more specific/complete versions
        - Removes redundant entries
        """
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=600,  # เพิ่มจาก 300 เพื่อรองรับ list ที่ยาวขึ้น
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(options: List[str]) -> dict:
            if len(options) <= 1:
                return {"unique_options": options, "reasoning": "Only one option"}
            prompt = build_deduplicate_options_prompt(options)
            
            try:
                resp = llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/deduplicate")
                txt = extract_llm_text(resp).strip()
                if txt.startswith("```json"):
                    txt = txt.replace("```json", "").replace("```", "").strip()
                data = json.loads(txt)
                
                unique = data.get("unique_options", options)
                reasoning = data.get("reasoning", "")
                
                # Log แบบ structured - อ่านง่าย เห็น context ชัดเจน
                logger.log_with_data("info", "🎯 ลบตัวเลือกซ้ำสำเร็จ", {
                    "action": "deduplicate_options",
                    "before_count": len(options),
                    "after_count": len(unique),
                    "removed_count": len(options) - len(unique),
                    "reasoning": reasoning[:100],
                    "model": "claude-haiku"
                })
                return {"unique_options": unique, "reasoning": reasoning}
            except Exception as e:
                # Log error พร้อม context
                logger.log_with_data("warning", "⚠️ ลบตัวเลือกซ้ำล้มเหลว ใช้ข้อมูลเดิม", {
                    "action": "deduplicate_options",
                    "error": str(e),
                    "fallback": "using_original_options"
                })
                return {"unique_options": options, "reasoning": f"Error: {e}"}
        
        return _call

    def _default_slot_mapper_llm_call(self) -> Callable[[str, str, List[str]], dict]:
        """
        Map a free-text reply into one of pending_slot options.
        Return JSON: {"choice_index": 1..N or 0, "choice_text": "...", "confidence": 0.0-1.0}
        - deterministic (temperature=0)
        - NO hardcode: lets LLM interpret abbreviations like "กทม"
        """
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=180,  # Increased from 120 to accommodate slot mapping + confidence + reasoning
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(slot_key: str, user_text: str, options: List[str]) -> dict:
            prompt = build_slot_mapper_prompt(slot_key, user_text, options)
            try:
                text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/llm")).strip()
            except Exception:
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_fallback_intent_llm_call(self) -> Callable[[str, str, str], dict]:
        """
        LLM classifier for inputs that didn't match any deterministic rule.
        Return JSON: {"intent": "new_topic|elaborate|legal_question|greeting|unknown", "query": "", "confidence": 0.0}
        - new_topic:       user wants a new/different topic → show topic menu
        - elaborate:       user wants more detail on last answer
        - legal_question:  user is asking about restaurant business law/permit/tax
        - greeting:        greeting, thanks, farewell
        - unknown:         cannot determine → fallback to topic menu
        Uses fast topic-picker model (low latency).
        """
        topic_model = getattr(conf, "OPENROUTER_MODEL_TOPIC_PICKER", getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL))
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=topic_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=250,  # Increased from 150 to accommodate intent classification + query generation + confidence + reasoning
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str, last_query: str, persona: str) -> dict:
            prompt = build_fallback_intent_prompt(user_text, last_query, persona)
            try:
                text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/fallback_intent")).strip()
            except Exception:
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _strip_code_fences(self, text: str) -> str:
        t = (text or "").strip()
        if "```json" in t:
            return t.split("```json", 1)[1].split("```", 1)[0].strip()
        if "```" in t:
            parts = t.split("```")
            if len(parts) >= 3:
                return parts[1].strip()
        return t

    # Greeting/noise classification
    _EN_GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|yo+)\b", re.IGNORECASE)
    _EN_GOOD_TIME_RE = re.compile(r"^\s*good\s+(morning|afternoon|evening|night)\b", re.IGNORECASE)
    _TH_SAWASDEE_FUZZY_RE = re.compile(r"^\s*สว[^\s]{0,6}ดี", re.IGNORECASE)
    _TH_WATDEE_RE = re.compile(r"^\s*หวัดดี", re.IGNORECASE)
    _TH_DEE_RE = re.compile(r"^\s*ดี(?:ครับ|คับ|ค่ะ|คะ|งับ|จ้า|จ้ะ|ค่า)?", re.IGNORECASE)

    def _looks_like_greeting_or_thanks(self, s: str) -> bool:
        raw = (s or "").strip()
        if not raw:
            return True

        if self._TH_LAUGH_5_RE.match(raw):
            return True

        if self._LIKELY_SELECTION_RE.match(raw):
            return False

        t = self._normalize_for_intent(raw)

        if self._THANKS_RE.search(t):
            return True

        if len(t) <= 2:
            return True

        if self._QUESTION_MARKERS_RE.search(t):
            return False
        if self._LEGAL_SIGNAL_RE.search(t):
            return False

        if self._EN_GREETING_RE.match(t) or self._EN_GOOD_TIME_RE.match(t):
            return True
        if self._TH_WATDEE_RE.match(t) or self._TH_SAWASDEE_FUZZY_RE.match(t):
            return True
        if self._TH_DEE_RE.match(t) and not self._QUESTION_MARKERS_RE.search(t):
            return True

        if len(t) <= 14 and self._NOISE_ONLY_RE.match(t):
            return True

        return False

    def _looks_like_legal_question(self, s: str) -> bool:
        t = self._normalize_for_intent(s)
        if not t:
            return False
        if self._LEGAL_SIGNAL_RE.search(t):
            return True
        if self._QUESTION_MARKERS_RE.search(t) and len(t) >= 6:
            return True
        return False

    def _is_noise(self, s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return False
        if self._TH_LAUGH_5_RE.match(t):
            return True
        if self._LIKELY_SELECTION_RE.match(t):
            return False
        if len(t) <= 2:
            return True
        if self._NOISE_ONLY_RE.match(t.lower()) and not self._looks_like_legal_question(t):
            return True
        return False

    def _is_likely_typo_rule(self, s: str) -> bool:
        """
        Rule-based fast-path: detect obviously garbled input without LLM.
        Returns True for inputs that are clearly not intentional (standalone Thai diacritics,
        lone vowel marks, all-punctuation, etc.).
        Does NOT catch ambiguous cases — those are delegated to LLM via _should_ask_typo_llm().
        """
        t = (s or "").strip()
        if not t:
            return False

        # Numbers/selections are never typos
        if self._LIKELY_SELECTION_RE.match(t):
            return False

        # Standalone Thai combining marks / vowel signs / tone marks (e.g. "ๅ", "ิ", "็")
        if self._STANDALONE_TH_DIACRITIC_RE.match(t):
            return True

        # Single non-digit, non-Thai-consonant character
        if len(t) == 1 and not t.isdigit():
            # Only flag as typo if it's NOT a meaningful Thai consonant that could be a selection
            # Thai consonants range U+0E01–U+0E2E — a single one could be noise but not a "typo" per se
            # Single Thai vowel/diacritic is already caught above; single ASCII letter or symbol
            if not ("\u0E01" <= t <= "\u0E2E"):
                return True

        # Pure punctuation (e.g. "???", "...", "!!")
        if self._ALL_PUNCTUATION_RE.match(t):
            return True

        # Thai consonant mash with no real Thai vowel component (2–6 chars, all consonants/diacritics)
        # e.g. "กขค", "ภคม" — real Thai words need at least a vowel or sara
        if self._TH_CONSONANT_MASH_RE.match(t):
            # But make sure it's not a real common word abbreviation or a legal signal
            if not self._LEGAL_SIGNAL_RE.search(t) and not self._QUESTION_MARKERS_RE.search(t):
                return True

        return False

    def _default_typo_check_llm_call(self) -> Callable[[str, str], dict]:
        """
        LLM callable for ambiguous inputs that rule-based check couldn't classify.
        Returns {"is_typo": bool, "confidence": float, "suggested": str}.
        Uses the fast topic-picker model for low latency.
        - is_typo=True: input looks garbled/meaningless — ask user to retype
        - is_typo=False: input has apparent meaning — route normally
        - suggested: best guess at what user meant (empty string if no good guess)
        """
        topic_model = getattr(conf, "OPENROUTER_MODEL_TOPIC_PICKER", getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL))
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=topic_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=120,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str, last_topic: str) -> dict:
            prompt = build_typo_check_prompt(user_text, last_topic)
            try:
                text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/typo_check")).strip()
            except Exception:
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    # Academic intake lock only when truly active
    _ACADEMIC_LOCK_STAGES = {"awaiting_slots", "awaiting_sections"}

    def _is_academic_intake_active(self, state: ConversationState) -> bool:
        flow = (state.context or {}).get("academic_flow")
        if not isinstance(flow, dict):
            return False
        stage = str(flow.get("stage") or "").strip()
        if not stage:
            return False
        return stage in self._ACADEMIC_LOCK_STAGES

    # Intent classification helpers
    def _looks_like_mode_status_query(self, s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return False
        if ("โหมด" not in t and "mode" not in t.lower() and "persona" not in t.lower() and "บุคลิก" not in t):
            return False
        return bool(self._MODE_STATUS_Q.search(t))

    def _looks_like_switch_without_target(self, s: str) -> bool:
        t = (s or "").strip().lower()
        if not t:
            return False
        if any(v in t for v in self._SWITCH_VERBS) and any(m in t for m in self._SWITCH_MARKERS):
            if re.search(r"\b(academic|practical)\b|วิชาการ|ละเอียด|สั้น|กระชับ", t):
                return False
            return True
        if re.fullmatch(r"(เปลี่ยน|สลับ|ปรับ)\s*", t):
            return True
        return False

    def _infer_target_persona_from_text(self, s: str) -> Optional[str]:
        t = self._normalize_for_intent(s)
        if not t:
            return None

        if re.search(r"\bacademic\b", t):
            return "academic"
        if re.search(r"\bpractical\b", t):
            return "practical"

        if "วิชาการ" in t:
            return "academic"
        if any(h in t for h in self._TARGET_ACADEMIC_HINTS):
            return "academic"
        if any(h in t for h in self._TARGET_PRACTICAL_HINTS):
            return "practical"

        return None

    def _infer_user_style_request_det(self, s: str) -> Dict[str, bool]:
        t = self._normalize_for_intent(s)
        if not t:
            return {"wants_short": False, "wants_long": False}

        wants_short = any(h in t for h in self._TARGET_PRACTICAL_HINTS)
        wants_long = any(h in t for h in self._TARGET_ACADEMIC_HINTS)

        if wants_short and wants_long:
            return {"wants_short": False, "wants_long": False}

        return {"wants_short": wants_short, "wants_long": wants_long}

    def _infer_user_style_request_hybrid(self, s: str) -> Dict[str, Any]:
        text = s or ""

        # 1. Fast-path: deterministic keyword check (no LLM cost)
        det = self._infer_user_style_request_det(text)
        if det["wants_short"] or det["wants_long"]:
            return {"wants_short": det["wants_short"], "wants_long": det["wants_long"], "method": "det", "confidence": 0.9}

        # 2. Skip LLM for pure noise/number-only inputs (too short to have style intent)
        stripped = text.strip()
        if not stripped or len(stripped) <= 3 or self._LIKELY_SELECTION_RE.match(stripped):
            return {"wants_short": False, "wants_long": False, "method": "none", "confidence": 0.0}

        # 3. LLM is primary detector for all substantive inputs — catches natural phrasing
        #    that regex misses (e.g. "อธิบายแบบครบๆ", "ขอทราบทั้งหมดเลย", "แบบย่อๆ ได้ไหม")
        res: Dict[str, Any] = {}
        try:
            res = self.llm_style_call(text) or {}
        except Exception:
            res = {}

        try:
            confv = float(res.get("confidence", 0.0) or 0.0)
        except Exception:
            confv = 0.0

        wants_long = bool(res.get("wants_long", False)) if confv >= 0.70 else False
        wants_short = bool(res.get("wants_short", False)) if confv >= 0.55 else False

        if wants_long and wants_short:
            wants_long = False
            wants_short = False

        if wants_long or wants_short:
            return {"wants_short": wants_short, "wants_long": wants_long, "method": "llm", "confidence": confv}

        return {"wants_short": False, "wants_long": False, "method": "llm_low", "confidence": confv}

    def _classify_intent(self, state: ConversationState, user_input: str) -> Dict[str, Any]:
        """
        DEPRECATED — NOT called by _handle_inner.
        _handle_inner contains its own inline routing priority that supersedes this method.
        This method is kept ONLY for backward compatibility with external test fixtures that
        may call it directly.  Do NOT add new logic here — update _handle_inner instead.
        """
        state.context = state.context or {}
        text = (user_input or "")

        if self._is_academic_intake_active(state):
            return {"intent": self.INTENT_ACAD_INTAKE_REPLY, "meta": {}}

        if self._looks_like_switch_without_target(text):
            return {"intent": self.INTENT_EXPLICIT_SWITCH, "meta": {"kind": "no_target"}}

        if self._looks_like_mode_status_query(text):
            return {"intent": self.INTENT_MODE_STATUS, "meta": {}}

        if self._SMALLTALK_RE.search(text) or self._looks_like_greeting_or_thanks(text):
            return {"intent": self.INTENT_GREETING, "meta": {}}

        if self._looks_like_legal_question(text):
            return {"intent": self.INTENT_LEGAL_NEW, "meta": {}}

        if self._is_noise(text):
            return {"intent": self.INTENT_NOISE, "meta": {}}

        return {"intent": self.INTENT_UNKNOWN, "meta": {}}

    # Core router

    def __init__(
        self,
        retriever,
        llm_confirm_call: Optional[Callable[[str], dict]] = None,
        llm_style_call: Optional[Callable[[str], dict]] = None,
        llm_greet_prefix_call: Optional[Callable[[str, str, str, bool], dict]] = None,
        llm_topic_picker_call: Optional[Callable[[str, List[str], int, List[str]], dict]] = None,
        llm_slot_mapper_call: Optional[Callable[[str, str, List[str]], dict]] = None,
        llm_deduplicate_options_call: Optional[Callable[[List[str]], dict]] = None,
    ):
        self.retriever = retriever
        self._academic = AcademicPersonaService(retriever=retriever)
        self._practical = PracticalPersonaService(retriever=retriever)

        self.llm_confirm_call = llm_confirm_call or self._default_confirm_llm_call()
        self.llm_style_call = llm_style_call or self._default_style_llm_call()
        self.llm_greet_prefix_call = llm_greet_prefix_call or self._default_greet_prefix_llm_call()
        self.llm_topic_picker_call = llm_topic_picker_call or self._default_topic_picker_llm_call()

        # NEW: pending_slot free-text mapper (no hardcode)
        self.llm_slot_mapper_call = llm_slot_mapper_call or self._default_slot_mapper_llm_call()

        # NEW: LLM fallback intent classifier — replaces hardcoded error message
        self.llm_fallback_intent_call = self._default_fallback_intent_llm_call()

        # NEW: LLM-based typo/garbled input detector
        self.llm_typo_check_call = self._default_typo_check_llm_call()

        # NEW: LLM-based deduplication of similar options
        self._deduplicate_options_llm_call = llm_deduplicate_options_call or self._default_deduplicate_options_llm_call()

        # LLM-based operation group classifier (no hardcoded rules)
        # Cache: {(license_type, entity_type_normalized): (slot_options, raw_op_map)}
        # populated lazily on first call per unique key — avoids repeated LLM round-trips
        self._llm_op_group_classifier = self._default_op_group_classifier_llm_call()
        self._op_groups_cache: Dict[tuple, Tuple[List[str], Dict]] = {}

        self._rng = random.Random()

    # RNG: stable per session + moves each greeting
    def _get_session_seed(self, state: ConversationState) -> int:
        state.context = state.context or {}
        if isinstance(state.context.get("rng_seed"), int):
            return int(state.context["rng_seed"])

        sid = ""
        if hasattr(state, "session_id"):
            sid = str(getattr(state, "session_id") or "")
        if not sid:
            sid = str(state.context.get("session_id") or "")

        if not sid:
            sid = str(id(state))

        h = hashlib.sha256(sid.encode("utf-8")).hexdigest()
        seed = int(h[:8], 16)
        state.context["rng_seed"] = seed
        return seed

    def _get_rng(self, state: ConversationState) -> random.Random:
        seed = self._get_session_seed(state)
        turns = int((state.context or {}).get("greet_menu_turns") or 0)
        mixed = seed ^ ((turns + 1) * 2654435761 & 0xFFFFFFFF)
        return random.Random(mixed)

    # Retrieval gate (Practical legal MUST retrieve)
    _TOKEN_SPLIT_RE = re.compile(r"[\s/,\-–—|]+", re.UNICODE)

    def _tokenize_loose(self, s: str) -> List[str]:
        t = self._normalize_for_intent(s)
        toks = [x.strip() for x in self._TOKEN_SPLIT_RE.split(t) if x and x.strip()]
        return [x for x in toks if len(x) >= 2]

    def _topic_overlap_ratio(self, a: str, b: str) -> float:
        sa = set(self._tokenize_loose(a))
        sb = set(self._tokenize_loose(b))
        if not sa or not sb:
            return 0.0
        inter = len(sa.intersection(sb))
        union = len(sa.union(sb))
        return (inter / union) if union else 0.0

    def _should_retrieve_new_for_practical(self, state: ConversationState, user_input: str) -> bool:
        q = (user_input or "").strip()
        if not q:
            return False

        has_docs = bool(getattr(state, "current_docs", None))
        if not has_docs:
            return True

        last_q = (
            (getattr(state, "last_retrieval_query", None) or "")
            or str((state.context or {}).get("last_retrieval_query") or "")
        ).strip()

        if not last_q:
            return True

        if last_q == q:
            return False

        overlap = self._topic_overlap_ratio(last_q, q)
        return overlap < 0.22

    # Regex: at least one generic follow-up keyword present in the message.
    # Combined with length guard and no-specific-topic check below.
    _GENERIC_FOLLOWUP_KEYWORDS_RE = re.compile(
        r"(ขั้นตอน|การสมัคร|สมัคร|วิธีสมัคร|เอกสาร|ค่าธรรมเนียม|ค่าใช้จ่าย|"
        r"ระยะเวลา|ช่องทางยื่น|ช่องทาง|สถานที่ยื่น|แบบฟอร์ม"
        r"|ลิงก์|ลิงค์|ลิ้งก์|ลิ้งค์|ดาวน์โหลด|"
        r"อยากรู้|อยากทราบ|ต้องใช้|ต้องทำ|ต้องเตรียม)",
        re.IGNORECASE,
    )
    # Regex: specific topic/license keywords that indicate the user is asking about a NEW topic.
    # If present, this is NOT a generic follow-up.
    _SPECIFIC_TOPIC_RE = re.compile(
        r"(ใบอนุญาต|จดทะเบียน|ภาษีมูลค่าเพิ่ม|ภาษีป้าย|vat|ภพ\.?20|สุรา|เหล้า|ประกันสังคม|ป้ายร้าน|"
        r"ประกอบกิจการ|สถานที่จำหน่าย|ทะเบียนพาณิชย์|บริษัทจำกัด|ห้างหุ้นส่วน|qr.?pay|qr.?api|payment api|"
        r"บัญชีธนาคาร|หาพนักงาน|edc|รูดบัตร|คิวอาร์เพย์|คิวอาเพ)",
        re.IGNORECASE,
    )

    def _ensure_practical_retrieval_for_legal(self, state: ConversationState, user_input: str) -> None:
        state.context = state.context or {}
        q = (user_input or "").strip()
        if not q:
            return

        # Clear multi-topic flag: it is only valid for the single turn it was set.
        # Any new legal question (this function is called for new questions, not slot answers)
        # must start fresh — if the new query is also multi-topic the flag will be re-set below.
        state.context.pop("_multi_topic_retrieval", None)

        # 🎯 Generic follow-up anchoring: if user's message contains generic procedure/document/fee
        # keywords but NO specific topic/license name, anchor the retrieval query to the active topic
        # so we don't retrieve cross-topic docs and accidentally trigger multi-license detection.
        # Example: "อยากรู้ขั้นตอนสมัคร เอกสารที่ต้องใช้ หรือค่าธรรมเนียม" after QR Payment answer.
        _active_topic_q = (
            str(getattr(state, "last_retrieval_query", "") or "").strip()
            or str((state.context or {}).get("last_user_legal_query") or "").strip()
        )
        if (
            _active_topic_q
            and len(q) <= 80
            and self._GENERIC_FOLLOWUP_KEYWORDS_RE.search(q)
            and not self._SPECIFIC_TOPIC_RE.search(q)
        ):
            _LOG.info(
                "[Supervisor] generic follow-up detected %r — anchoring retrieval to active topic %r",
                q[:50], _active_topic_q[:50],
            )
            q = f"{_active_topic_q} {q}"

        if not self._should_retrieve_new_for_practical(state, q):
            return

        # Multi-topic predictive retrieval: if query mentions ≥2 license types, retrieve ALL
        # their docs from Chroma by license_type filter and merge — lets LLM answer all in one response.
        _detected_licenses = self._detect_license_types_from_query(q)
        if len(_detected_licenses) >= 2:
            try:
                _vstore_mt = getattr(self.retriever, "vectorstore", None)
                _coll_mt = getattr(_vstore_mt, "_collection", None) if _vstore_mt else None
                if _coll_mt is not None:
                    _doc_chars_mt = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                    _SUPERVISOR_META_WL_MT = frozenset({
                        "license_type", "operation_topic",
                        "entity_type_normalized", "registration_type", "department",
                        "fees", "operation_duration", "service_channel",
                        "operation_steps", "identification_documents", "research_reference",
                    })
                    _SUPERVISOR_FC_MT = {
                        "operation_steps": 600, "identification_documents": 1500,
                        "research_reference": 3200, "fees": 120, "service_channel": 200,
                    }
                    _merged: List[Dict] = []
                    for _lt_name in _detected_licenses:
                        _r = _coll_mt.get(
                            where={"license_type": _lt_name},
                            include=["documents", "metadatas"],
                        )
                        for _fc, _fm in zip(_r.get("documents") or [], _r.get("metadatas") or []):
                            _sm: Dict = {}
                            for _k, _v in (_fm or {}).items():
                                if _k not in _SUPERVISOR_META_WL_MT or _v in (None, "", "nan", "None"):
                                    continue
                                _vs = str(_v)
                                _cap = _SUPERVISOR_FC_MT.get(_k)
                                _sm[_k] = _vs[:_cap] if _cap and len(_vs) > _cap else _vs
                            _merged.append({"content": (_fc or "")[:_doc_chars_mt], "metadata": _sm})
                    if _merged:
                        state.current_docs = _merged
                        state.last_retrieval_query = q
                        state.context["_multi_topic_retrieval"] = True
                        _LOG.info(
                            "[Supervisor] multi-topic retrieval: %s → %d total docs merged",
                            _detected_licenses, len(_merged),
                        )
                        return
            except Exception as _e_mt:
                _LOG.warning("[Supervisor] multi-topic retrieval failed: %s — falling back", _e_mt)

        # Targeted filter retrieval for rare license types that tend to be overwhelmed
        # by ใบทะเบียนพาณิชย์ (77 docs) in embedding search.
        # When query explicitly mentions one of these keywords, fetch ALL their docs directly
        # from ChromaDB by license_type filter, bypassing the imbalanced embedding ranking.
        _RARE_LT_PATTERNS = {
            r"ประกันสังคม|กองทุนประกันสังคม|ขึ้นทะเบียนประกัน": "การขึ้นทะเบียนกองทุนประกันสังคม",
            r"ใบอนุญาตจำหน่ายสุรา|ขายสุรา|จำหน่ายสุรา|ภส\.08": "ใบอนุญาตจำหน่ายสุรา",
            r"ภาษีป้าย|ป้ายร้าน|ภป\.1": "แบบแสดงรายการภาษีป้ายร้านอาหาร",
            r"ใบวุฒิบัตร|อบรมผู้สัมผัสอาหาร": "ใบวุฒิบัตรผู้สัมผัสอาหาร",
            r"ใบรับรองแพทย์|9 โรค|สณ\.11": "ใบรับรองแพทย์ 9 โรค(สณ.11)",
        }
        for _rare_pat, _rare_lt in _RARE_LT_PATTERNS.items():
            if re.search(_rare_pat, q, re.IGNORECASE):
                try:
                    _vstore = getattr(self.retriever, "vectorstore", None)
                    _coll = getattr(_vstore, "_collection", None) if _vstore else None
                    if _coll is not None:
                        _filt_result = _coll.get(
                            where={"license_type": _rare_lt},
                            include=["documents", "metadatas"],
                        )
                        _filt_docs = _filt_result.get("documents") or []
                        _filt_mds = _filt_result.get("metadatas") or []
                        if _filt_docs:
                            _doc_chars2 = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                            _SUPERVISOR_META_WL2 = frozenset({
                                "license_type", "operation_topic", "chunk_type",
                                "entity_type_normalized", "registration_type", "department",
                                "fees", "operation_duration", "service_channel",
                                "identification_documents", "operation_steps",
                            })
                            _SUPERVISOR_FC2 = {
                                "operation_steps": 600, "identification_documents": 1500,
                                "research_reference": 3200, "fees": 120, "service_channel": 200,
                            }
                            _filtered = []
                            for _fc, _fm in zip(_filt_docs, _filt_mds):
                                _sm = {}
                                for _k, _v in (_fm or {}).items():
                                    if _k not in _SUPERVISOR_META_WL2 or _v in (None, "", "nan", "None"):
                                        continue
                                    _vs = str(_v)
                                    _cap = _SUPERVISOR_FC2.get(_k)
                                    _sm[_k] = _vs[:_cap] if _cap and len(_vs) > _cap else _vs
                                _filtered.append({"content": (_fc or "")[:_doc_chars2], "metadata": _sm})
                            _LOG.info(
                                "[Supervisor] rare-license targeted fetch: %r → %d docs (bypassing embedding imbalance)",
                                _rare_lt, len(_filtered),
                            )
                            state.current_docs = _filtered
                            state.last_retrieval_query = q
                            return
                except Exception as _e_rare:
                    _LOG.warning("[Supervisor] rare-license filter retrieval failed: %s", _e_rare)
                break  # only match first pattern

        # Query expansion: short/abbrev keywords map to full Thai terms for better embedding match
        _QUERY_EXPAND = {
            r"\bvat\b": "ภาษีมูลค่าเพิ่ม ภพ.20 จด VAT กรมสรรพากร",
            r"\bภพ\.?20\b": "ภาษีมูลค่าเพิ่ม ภพ.20 จด VAT กรมสรรพากร",
            r"ภาษีมูลค่าเพิ่ม": "ภาษีมูลค่าเพิ่ม ภพ.20 จด VAT กรมสรรพากร",
            r"สุรา|เหล้า|ขายเหล้า": "ใบอนุญาตจำหน่ายสุรา สรรพสามิต ภส.08",
            r"ประกันสังคม": "ขึ้นทะเบียนประกันสังคม นายจ้าง ลูกจ้าง",
            r"ป้ายร้าน|ภาษีป้าย": "แบบแสดงรายการภาษีป้ายร้านอาหาร ป้ายโฆษณา",
        }
        _sv_expansions: list = []
        for pattern, expansion in _QUERY_EXPAND.items():
            if re.search(pattern, q, re.IGNORECASE) and expansion not in _sv_expansions:
                _sv_expansions.append(expansion)
        q_expanded = (q + " " + " ".join(_sv_expansions)).strip() if _sv_expansions else q

        docs = self.retriever.invoke(q_expanded)
        results: List[Dict[str, Any]] = []
        top_k = int(getattr(conf, "RETRIEVAL_TOP_K", 15) or 15)
        _doc_chars = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
        # Token: cap long metadata fields before storing — prevent ×N token explosion
        _SUPERVISOR_META_WHITELIST = frozenset({
            # Source type — lets downstream LLM know doc origin
            "data_type",
            # Regulatory fields
            "license_type", "operation_topic",
            "entity_type_normalized", "registration_type", "department",
            "fees", "operation_duration", "service_channel",
            "legal_regulatory",           # บทลงโทษ ค่าปรับ ข้อกำหนดทางกฎหมาย
            "terms_and_conditions",       # หน้าที่และเงื่อนไขของผู้ประกอบการ
            "identification_documents",   # เอกสารที่ต้องใช้ — must reach LLM complete
            "operation_steps",            # ขั้นตอนการดำเนินการ
            # Marketing / business_guide fields (data_loader_general.py)
            "main_topic", "sub_topic", "answer_guideline",
        })
        _SUPERVISOR_FIELD_CAPS = {
            "operation_steps": 600, "identification_documents": 1500,
            "research_reference": 3200, "fees": 120, "service_channel": 200,
            "legal_regulatory": 2000, "terms_and_conditions": 800,
            # marketing / business_guide content fields
            "answer_guideline": 1500, "main_topic": 120, "sub_topic": 150,
        }
        for d in (docs or [])[:top_k]:
            raw_md = getattr(d, "metadata", {}) or {}
            slim_md = {}
            for k, v in raw_md.items():
                if k not in _SUPERVISOR_META_WHITELIST:
                    continue
                if v in (None, "", "nan", "None"):
                    continue
                v_str = str(v)
                cap = _SUPERVISOR_FIELD_CAPS.get(k)
                slim_md[k] = v_str[:cap] if cap and len(v_str) > cap else v_str
            results.append({"content": (getattr(d, "page_content", "") or "")[:_doc_chars], "metadata": slim_md})

        state.current_docs = results
        # FIX #5: write only to the Pydantic field — single source of truth.
        # state.get_last_retrieval_query() reads this field first, so no context mirror needed.
        state.last_retrieval_query = q

    # Informational question patterns — user wants to KNOW something, not DO something.
    # When matched (and no action pattern), skip slot queue and let LLM answer directly.
    _INFO_Q_RE = re.compile(
        r"(ประเภทใด|ประเภทไหน|อะไรบ้าง|คืออะไร|หมายถึงอะไร|"
        r"ใครบ้าง|เงื่อนไข|ข้อยกเว้น|ยกเว้น|ไม่ต้อง|ไม่จำเป็น|"
        r"ต้อง.{0,15}(ไหม|มั้ย|หรือเปล่า|หรือไม่)|"
        r"ต้องทำ(ยังไง|ไง|อย่างไร|อะไร)|ทำ(ยังไง|ไง|อย่างไร)(?!\s*(การ|ธุรกิจ|บัตร))|"
        r"แตกต่าง|เปรียบเทียบ|อธิบาย|กรณีใด|เมื่อไหร่|เมื่อไร|"
        r"จะเกิดอะไร|เกิดอะไร|จะเป็นอะไร|จะเกิดผล|ผลกระทบ|ความเสี่ยง|"
        r"ต้องโดน|จะถูก|จะมีผล|บทลงโทษ|โทษคือ|ค่าปรับ|"
        r"ชำรุด|สูญหาย|ทำหาย|ทำหล่น|เสียหาย|"
        r"ต้องการติดต่อ|ติดต่อได้ที่|ติดต่อที่ไหน|ติดต่อยังไง|ติดต่ออย่างไร|ที่อยู่|เบอร์โทร|"
        r"วิธีการ(?!จด|สมัคร|ขอ|ยื่น|ลงทะเบียน|ติดตั้ง|เปิดใช้|เริ่มต้น)|วิธี(?!การ|จด|สมัคร|ขอ|ยื่น|ลงทะเบียน|ีเดียว|ีชีวิต)|"
        r"ระยะเวลา|ช่องทาง(?!การจด|การสมัคร|การขอ|การยื่น))",
        re.IGNORECASE,
    )
    # Action patterns — user wants to perform a registration/application step.
    # Overrides _INFO_Q_RE: even if query looks informational, treat as action.
    _ACTION_Q_RE = re.compile(
        r"(อยากจด|อยากสมัคร|อยากขอ|อยากยื่น|"
        r"ต้องการจด|ต้องการสมัคร|ต้องการขอ|ต้องการยื่น|"
        r"จะจด|จะสมัคร|จะขอใบ|จะยื่น|"
        r"วิธีจด|วิธีสมัคร|ขั้นตอนการจด|ขั้นตอนการสมัคร|ขั้นตอนการขอ|"
        r"ต้องการแก้ไข|ต้องการเปลี่ยนแปลง|ต้องการยกเลิก|ต้องการต่ออายุ|"
        r"อยากแก้ไข|อยากเปลี่ยนแปลง|อยากยกเลิก|อยากต่ออายุ|"
        r"จะแก้ไข|จะเปลี่ยนแปลง|จะยกเลิก|จะต่ออายุ|"
        r"วิธีแก้ไข|วิธีเปลี่ยนแปลง|วิธียกเลิก|วิธีต่ออายุ|"
        r"ขั้นตอนการแก้ไข|ขั้นตอนการเปลี่ยนแปลง|ขั้นตอนการยกเลิก|ขั้นตอนการต่ออายุ)",
        re.IGNORECASE,
    )

    def _maybe_build_slot_queue_from_docs(self, state: ConversationState, query: str) -> None:
        """
        After doc retrieval for a direct legal question, discover slot dimensions for the
        detected license_type and set topic_slot_queue — skipping already-collected slots.
        If entity_type is already known, also re-retrieves docs with entity filter.
        """
        docs = state.current_docs or []
        if not docs:
            return

        # Multi-topic: docs span multiple licenses intentionally — skip slot queue, let LLM answer all
        if (state.context or {}).get("_multi_topic_retrieval"):
            _LOG.info("[Supervisor] multi-topic retrieval flag set — skipping slot queue build")
            return

        # Auto-update department slot whenever user explicitly mentions a known dept in ANY query type.
        # This covers follow-up queries like "ต้องการสมัครธนาคารไทยพาณิชย์" that aren't link requests.
        _cs_dept_check = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
        _cached_dept_any = (_cs_dept_check.get("department") or
                            (state.context or {}).get("slots", {}).get("department") or "").strip()
        if _cached_dept_any:
            _available_depts_any: list = []
            for _da in docs:
                if not isinstance(_da, dict): continue
                _dv_a = ((_da.get("metadata") or {}).get("department") or "").strip()
                if _dv_a and _dv_a not in _available_depts_any:
                    _available_depts_any.append(_dv_a)
            _query_dept_any = next((d for d in _available_depts_any if d and d in query), None)
            if _query_dept_any and _query_dept_any != _cached_dept_any:
                _LOG.info("[Supervisor] dept auto-update %r → %r (from query text)", _cached_dept_any, _query_dept_any)
                if hasattr(state, "save_collected_slot"):
                    state.save_collected_slot("department", _query_dept_any)
                if state.context is not None:
                    state.context.setdefault("slots", {})["department"] = _query_dept_any
                    if "collected_slots" in state.context:
                        state.context["collected_slots"]["department"] = _query_dept_any

        # Always persist entity_type if explicitly stated in query — even for informational/link
        # queries that skip slot queue — so user is never re-asked on the next topic.
        _early_et = self._infer_entity_type_from_query(query)
        if _early_et:
            _early_cs = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
            if not (_early_cs or {}).get("entity_type"):
                if hasattr(state, "save_collected_slot"):
                    state.save_collected_slot("entity_type", _early_et)
                if state.context is not None:
                    state.context.setdefault("slots", {})["entity_type"] = _early_et
                _LOG.info("[Supervisor] entity_type pre-persisted from query text: %r", _early_et)

        # Intent check: informational/link-request queries should be answered directly — no slot queue needed.
        # Action signals override: "อยากจด/วิธีจด" always builds slot queue regardless.
        _is_link_req = bool(self._LINK_REQUEST_RE.search(query)) and not self._ACTION_Q_RE.search(query)
        _is_info_q   = bool(self._INFO_Q_RE.search(query))       and not self._ACTION_Q_RE.search(query)
        if _is_info_q:
            _LOG.info("[Supervisor] informational query — skipping slot queue: %r", query[:60])
            return
        if _is_link_req:
            # For link requests: skip only if department is already known.
            # If department unknown → let slot queue build (dept-only — trimmed below).
            _cs_lr = state.get_collected_slots() or {} if hasattr(state, "get_collected_slots") else {}
            _ctx_slots_lr = (state.context or {}).get("slots") or {}
            _dept_known_lr = bool(_cs_lr.get("department") or _ctx_slots_lr.get("department"))
            if _dept_known_lr:
                # Check if query explicitly mentions a DIFFERENT department than the cached one.
                # e.g. user previously chose ธนาคารไทยพาณิชย์ but now asks about ธนาคารกสิกรไทย.
                _cached_dept = (_cs_lr.get("department") or _ctx_slots_lr.get("department") or "").strip()
                _available_depts: list = []
                for _d in docs:
                    if not isinstance(_d, dict):
                        continue
                    _dv = ((_d.get("metadata") or {}).get("department") or "").strip()
                    if _dv and _dv not in _available_depts:
                        _available_depts.append(_dv)
                _query_dept = next((d for d in _available_depts if d and d in query), None)
                if _query_dept and _query_dept != _cached_dept:
                    _LOG.info(
                        "[Supervisor] link query: dept override %r → %r (from query)",
                        _cached_dept, _query_dept,
                    )
                    if hasattr(state, "save_collected_slot"):
                        state.save_collected_slot("department", _query_dept)
                    if state.context is not None:
                        state.context.setdefault("slots", {})["department"] = _query_dept
                        if "collected_slots" in state.context:
                            state.context["collected_slots"]["department"] = _query_dept
                _LOG.info("[Supervisor] link query + dept known — skipping slot queue: %r", query[:60])
                return
            # dept unknown → fall through to build dept-only slot queue

        # Extract all distinct license_types from retrieved docs
        license_types_seen: list = []
        for d in docs:
            if not isinstance(d, dict):
                continue
            lt = ((d.get("metadata") or {}).get("license_type") or "").strip()
            if lt and lt not in license_types_seen:
                license_types_seen.append(lt)

        if not license_types_seen:
            return

        # Multi-topic question: if docs span >1 distinct license type, check for a dominant one.
        # Dominant = one license has ≥60% of docs OR ≥2× the second. If dominant → proceed as
        # single-license so slot_queue gets built from Chroma (gives correct choices).
        if len(license_types_seen) > 1:
            _license_counts: dict = {}
            for _d2 in docs:
                if not isinstance(_d2, dict):
                    continue
                _lt2 = ((_d2.get("metadata") or {}).get("license_type") or "").strip()
                if _lt2:
                    _license_counts[_lt2] = _license_counts.get(_lt2, 0) + 1
            _total = sum(_license_counts.values())
            _top = max(_license_counts, key=_license_counts.get)
            _top_count = _license_counts[_top]
            _sorted_counts = sorted(_license_counts.values(), reverse=True)
            _second_count = _sorted_counts[1] if len(_sorted_counts) > 1 else 0
            if _total > 0 and (
                _top_count / _total >= 0.6
                or (_second_count > 0 and _top_count >= _second_count * 2)
            ):
                _LOG.info(
                    "[Supervisor] multi-license → dominant: %r (%d/%d docs), proceed with slot-queue",
                    _top, _top_count, _total,
                )
                license_types_seen = [_top]
            else:
                # Priority 0: if user explicitly named a license type in the query,
                # use it directly — even when no doc-count dominant exists.
                # Checks first 8 Thai chars of the core name (after stripping ใบ/แบบ/ระบบ prefix)
                # to handle minor typos e.g. พาณิชย์ → พานิชย์.
                _q_thai = re.sub(r'[^\u0E00-\u0E7F]', '', query or "")
                _strip_pfx = re.compile(r'^(ใบ|แบบ|ระบบ|การ)')
                _explicit_lt = None
                for _lt_cand in license_types_seen:
                    _lt_core = _strip_pfx.sub('', _lt_cand).strip()
                    _match_key = _lt_core[:8]
                    if len(_match_key) >= 5 and _match_key in _q_thai:
                        _explicit_lt = _lt_cand
                        break
                if _explicit_lt:
                    _LOG.info(
                        "[Supervisor] multi-license: explicit mention %r in query → use as dominant",
                        _explicit_lt,
                    )
                    license_types_seen = [_explicit_lt]
                    # fall through to slot-queue build below (no return)
                else:
                    _LOG.info(
                        "[Supervisor] multi-license no dominant (%s) — presenting topic selector",
                        license_types_seen,
                    )
                    # No dominant license: ask user which one to focus on first.
                    # Reuses "topic" slot routing — selection triggers focused retrieval + slot discovery.
                    # NOTE: do NOT set multi_license_topics; it forces action='answer' in practical
                    # and bypasses the slot-queue ask rule, giving incomplete generic answers.
                    _n_topics = len(license_types_seen[:4])
                    _topic_list_str = " / ".join(license_types_seen[:4])
                    _multi_q = (
                        f"มีเรื่องที่เกี่ยวข้องกัน {_n_topics} เรื่องที่คุณน่าจะต้องดำเนินการ "
                        f"({_topic_list_str}) "
                        f"ต้องการเริ่มจากเรื่องใดก่อนครับ?"
                    )
                    state.context["topic_slot_queue"] = [{
                        "key": "topic",
                        "options": list(license_types_seen[:4]),
                        "question": _multi_q,
                    }]
                    return

        license_type = license_types_seen[0]

        # Discover all slot dimensions for this license_type
        all_slots = self._discover_slots_for_license(license_type)
        if not all_slots:
            return

        # Filter out already-collected slots
        # Check BOTH collected_slots (save_collected_slot) AND context["slots"] (Practical's
        # slot-queue mechanism stores answers there too) so we never re-ask slots already known.
        try:
            _cs_lq = state.get_collected_slots() or {}
            _ctx_slots_lq = (state.context or {}).get("slots") or {}
            collected_keys = set(_cs_lq.keys()) | set(_ctx_slots_lq.keys())
        except Exception:
            _cs_lq = {}
            _ctx_slots_lq = {}
            collected_keys = set()

        # Step 2: auto-infer entity_type if explicitly stated in query text
        _inferred_et_lq = self._infer_entity_type_from_query(query)
        if _inferred_et_lq and "entity_type" not in collected_keys:
            state.save_collected_slot("entity_type", _inferred_et_lq)
            state.context.setdefault("slots", {})["entity_type"] = _inferred_et_lq
            collected_keys = collected_keys | {"entity_type"}
            _LOG.info("[Supervisor] legal_q entity_type auto-inferred from query: %r", _inferred_et_lq)
        # Step 3: skip entity_type/registration_type for universal-fact queries
        elif not self._entity_slot_needed(query):
            all_slots = [s for s in all_slots if s["key"] not in ("entity_type", "registration_type")]
            _LOG.info("[Supervisor] legal_q entity_type/registration_type skipped — universal-fact query: %r", query[:60])

        remaining = [s for s in all_slots if s["key"] not in collected_keys]

        if not remaining:
            return

        # Link request: only ask department (if present) — skip entity_type/registration_type.
        # User just wants the link; knowing the bank is enough to retrieve specific links.
        if _is_link_req:
            remaining = [s for s in remaining if s["key"] == "department"]
            if not remaining:
                _LOG.info("[Supervisor] link query: dept already known — skipping slot queue")
                return

        # FIX-P-A for legal_q path: if entity_type already collected, skip asking it again
        # and apply entity-specific filtering + enriched retrieval + op_group slot
        _known_entity_lq: Optional[str] = None
        if "entity_type" in collected_keys:
            # Prefer freshly-inferred value (_inferred_et_lq) over stale snapshots
            # (_cs_lq / _ctx_slots_lq were loaded before entity_type was auto-saved at Step 2)
            _raw_et_lq = _inferred_et_lq or _cs_lq.get("entity_type") or _ctx_slots_lq.get("entity_type") or ""
            try:
                from service.data_loader import DataLoader as _DL_lq
                _known_entity_lq = _DL_lq._normalize_entity_type(_raw_et_lq)
            except Exception:
                if "นิติบุคคล" in _raw_et_lq or "บริษัท" in _raw_et_lq or "ห้างหุ้นส่วน" in _raw_et_lq:
                    _known_entity_lq = "นิติบุคคล"
                elif "บุคคลธรรมดา" in _raw_et_lq:
                    _known_entity_lq = "บุคคลธรรมดา"

        if _known_entity_lq:
            # Re-retrieve docs with entity filter for better context
            _base_q_lq = (state.context or {}).get("last_user_legal_query") or ""
            try:
                state.current_docs = self._practical._retrieve_docs(
                    f"{_base_q_lq} {_known_entity_lq}".strip(),
                    metadata_filter={"entity_type_normalized": _known_entity_lq},
                )
                state.last_retrieval_query = f"{_base_q_lq} {_known_entity_lq}".strip()
                _LOG.info(
                    "[Supervisor] legal_q entity-enriched retrieval: entity=%r docs=%d",
                    _known_entity_lq, len(state.current_docs),
                )
            except Exception as _e_lq:
                _LOG.warning("[Supervisor] legal_q entity retrieval failed: %s", _e_lq)

            # Filter registration_type options to entity-specific ones
            _filtered_lq: List[Dict] = []
            for _slot_lq in remaining:
                if _slot_lq.get("key") == "registration_type":
                    _ert_lq = self._get_registration_types_for_entity(license_type, _known_entity_lq)
                    if len(_ert_lq) >= 2:
                        _filtered_lq.append({
                            "key": "registration_type",
                            "options": _ert_lq,
                            "question": _slot_lq.get("question", self._Q_REGISTRATION_TYPE),
                        })
                        _LOG.info(
                            "[Supervisor] legal_q registration_type filtered for entity=%r → %s",
                            _known_entity_lq, _ert_lq,
                        )
                    else:
                        _LOG.info(
                            "[Supervisor] legal_q registration_type ≤1 opt for entity=%r → skip",
                            _known_entity_lq,
                        )
                elif _slot_lq.get("key") != "entity_type":
                    _filtered_lq.append(_slot_lq)
            remaining = _filtered_lq

            # Append operation_group slot (skip if already collected)
            _op_res_lq = self._get_operation_groups_for_entity(license_type, _known_entity_lq)
            _op_grps_lq, _raw_op_map_lq = (
                _op_res_lq if isinstance(_op_res_lq, tuple) else (_op_res_lq, {})
            )
            if len(_op_grps_lq) >= 2 and "operation_group" not in collected_keys:
                _inferred_lq = self._infer_operation_group_from_query(query, _op_grps_lq)
                if _inferred_lq:
                    state.save_collected_slot("operation_group", _inferred_lq)
                    state.context.setdefault("slots", {})["confirmed_operation"] = _inferred_lq
                    _LOG.info("[Supervisor] legal_q operation_group auto-inferred=%r — skipping ask", _inferred_lq)
                else:
                    remaining.append({
                        "key": "operation_group",
                        "options": _op_grps_lq,
                        "question": self._Q_OPERATION_GROUP,
                        "raw_op_map": _raw_op_map_lq,
                    })
                    _LOG.info("[Supervisor] legal_q operation_group appended → %s", _op_grps_lq)
            elif "operation_group" in collected_keys:
                _LOG.info("[Supervisor] legal_q operation_group already collected — skip")

        if not remaining:
            return

        # Step 4: cap at 2 slots max to prevent question loops
        if len(remaining) > 2:
            remaining = remaining[:2]
            _LOG.info("[Supervisor] legal_q slot_queue capped to 2: %s", [s["key"] for s in remaining])

        state.context["topic_slot_queue"] = remaining
        _LOG.info(
            "[Supervisor] legal_q slot_queue built: license=%r remaining=%s",
            license_type, [s["key"] for s in remaining],
        )

    # Pending slot routing (keep + mixed-input support)
    _RANGE_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
    _ANY_NUMBER_RE = re.compile(r"\b(\d{1,2})\b")

    def _has_pending_slot(self, state: ConversationState) -> bool:
        p = (state.context or {}).get("pending_slot")
        return isinstance(p, dict) and isinstance(p.get("options"), list) and len(p.get("options")) > 0

    def _looks_like_pending_slot_reply(self, user_input: str) -> bool:
        raw = (user_input or "").strip()
        if not raw:
            return False
        if self._TH_LAUGH_5_RE.match(raw):
            return False

        low = raw.lower()

        if self._LIKELY_SELECTION_RE.match(raw):
            return True

        if self._ANY_NUMBER_RE.search(low):
            return True

        if re.search(r"(ทั้งหมด|all\b|ทุกข้อ|ทุกอย่าง)", low):
            return True

        t = self._normalize_for_intent(raw)
        if len(t) <= 2:
            return False
        if self._looks_like_greeting_or_thanks(raw):
            return False

        return True

    def _parse_indices(self, text: str) -> List[int]:
        t = (text or "").strip()
        if not t:
            return []

        nums: List[int] = []

        for m in self._RANGE_RE.finditer(t):
            try:
                a = int(m.group(1))
                b = int(m.group(2))
                if a <= 0 or b <= 0:
                    continue
                lo, hi = (a, b) if a <= b else (b, a)
                nums.extend(list(range(lo, hi + 1)))
            except Exception:
                continue

        for m in self._ANY_NUMBER_RE.finditer(t):
            try:
                n = int(m.group(1))
                nums.append(n)
            except Exception:
                continue

        seen = set()
        out: List[int] = []
        for n in nums:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _fuzzy_match_option(self, user_input: str, options: List[Any], threshold: float = 0.30) -> Optional[str]:
        """Fuzzy match user free-text to closest slot option using character bigram overlap.
        Used as fallback when LLM slot mapper fails or returns low confidence.
        Threshold 0.30 works well for Thai text (no word boundaries)."""
        raw_norm = self._normalize_for_intent(user_input)
        if not raw_norm or len(raw_norm) < 2:
            return None

        ngrams_raw = {raw_norm[i:i + 2] for i in range(len(raw_norm) - 1)}
        if not ngrams_raw:
            return None

        best_opt: Optional[str] = None
        best_score = 0.0

        for opt in options:
            s = str(opt).strip()
            if not s:
                continue
            s_norm = self._normalize_for_intent(s)
            if not s_norm or len(s_norm) < 2:
                continue
            ngrams_opt = {s_norm[i:i + 2] for i in range(len(s_norm) - 1)}
            if not ngrams_opt:
                continue
            overlap = len(ngrams_raw & ngrams_opt)
            score = overlap / max(len(ngrams_raw), len(ngrams_opt))
            if score > best_score:
                best_score = score
                best_opt = s

        return best_opt if best_score >= threshold else None

    def _map_pending_slot_reply(self, pending: Dict[str, Any], user_input: str) -> Tuple[Optional[str], Optional[str]]:
        options = pending.get("options") or []
        allow_multi = bool(pending.get("allow_multi", False))
        key = str(pending.get("key") or "").strip()

        raw = (user_input or "").strip()
        if not raw:
            return None, None

        low = raw.lower()

        if re.search(r"(ทั้งหมด|เล่าทั้งหมด|ขอทั้งหมด|ทุกข้อ|ทุกอย่าง|all\b)", low):
            # "ทั้งหมด" might be a literal named option — treat it as selecting that option
            for opt in options:
                if str(opt).strip() == "ทั้งหมด":
                    return "ทั้งหมด", None
            if allow_multi and options:
                return ", ".join([str(x).strip() for x in options if str(x).strip()]), None
            return None, "ตัวเลือกนี้ใช้ได้เฉพาะกรณีที่เลือกได้หลายข้อครับ"

        idxs = self._parse_indices(raw)

        if not idxs and re.fullmatch(r"\d{2,}", raw) and len(options) <= 9:
            idxs = [int(ch) for ch in raw if ch.isdigit()]

        if idxs:
            valid = [i for i in idxs if 1 <= i <= len(options)]
            if not valid:
                # topic slot accepts free text — numbers embedded in queries (e.g. "ภพ.20") should not trigger error
                if key != "topic":
                    return None, "เลขที่เลือกไม่อยู่ในช่วงตัวเลือกครับ"
                # for topic: fall through to free-text handling below
            else:
                if not allow_multi:
                    chosen = options[valid[0] - 1]
                    return str(chosen).strip(), None

                texts = [str(options[i - 1]).strip() for i in valid]
                texts = [x for x in texts if x]
                if not texts:
                    return None, "เลขที่เลือกไม่อยู่ในช่วงตัวเลือกครับ"
                return ", ".join(texts), None

        # 1) exact/contains match against options (fast deterministic)
        raw_norm = self._normalize_for_intent(raw)
        for opt in options:
            s = str(opt).strip()
            if not s:
                continue
            s_norm = self._normalize_for_intent(s)
            if raw_norm == s_norm:
                return s, None
            if raw_norm and s_norm and (raw_norm in s_norm or s_norm in raw_norm):
                return s, None

        # 2) topic: accept any free text as topic
        if key == "topic":
            return raw.strip(), None

        # 3) non-topic: allow free-text mapping via LLM (no hardcode)
        # Example: location slot with options ["กรุงเทพฯ", "ต่างจังหวัด"] and user says "กทม"
        if isinstance(options, list) and len(options) >= 2 and self.llm_slot_mapper_call:
            try:
                res = self.llm_slot_mapper_call(key, raw.strip(), [str(x).strip() for x in options]) or {}
            except Exception:
                res = {}
            try:
                confv = float(res.get("confidence", 0.0) or 0.0)
            except Exception:
                confv = 0.0

            # accept only when confident
            if confv >= 0.60:
                idx = 0
                try:
                    idx = int(res.get("choice_index", 0) or 0)
                except Exception:
                    idx = 0
                if 1 <= idx <= len(options):
                    return str(options[idx - 1]).strip(), None

                choice_text = str(res.get("choice_text") or "").strip()
                if choice_text:
                    # map back to options if same/contains
                    ct = self._normalize_for_intent(choice_text)
                    for opt in options:
                        s = str(opt).strip()
                        if not s:
                            continue
                        s_norm = self._normalize_for_intent(s)
                        if ct == s_norm or (ct in s_norm) or (s_norm in ct):
                            return s, None

        # 4) Character bi-gram overlap fuzzy match (fallback when LLM fails/low-confidence)
        # Works for Thai text where word boundaries are ambiguous.
        # Helps when user types natural text (e.g. "แก้ไขหุ้นส่วน") for option "การแก้ไขข้อมูลหุ้นส่วน"
        fuzzy_match = self._fuzzy_match_option(raw, options)
        if fuzzy_match is not None:
            _LOG.info("[Supervisor] pending_slot fuzzy match: %r → %r", raw[:40], fuzzy_match[:40])
            return fuzzy_match, None

        # otherwise require numeric for non-topic pending slots
        return None, "กรุณาตอบเป็นตัวเลขตามตัวเลือกครับ"

    # Pending slot routing guards
    def _should_route_pending_slot_now(self, state: ConversationState, user_input: str) -> bool:
        if not self._has_pending_slot(state):
            return False
        if not (user_input or "").strip():
            return False
        if self._TH_LAUGH_5_RE.match((user_input or "").strip()):
            return False

        ctx = state.context or {}

        if self._is_academic_intake_active(state):
            return False
        if ctx.get("awaiting_persona_confirmation"):
            return False

        if self._looks_like_switch_without_target(user_input):
            return False
        if self._infer_target_persona_from_text(user_input) in {"academic", "practical"}:
            return False

        if self._looks_like_mode_status_query(user_input):
            return False

        if self._STYLE_LIKELY_RE.search(user_input or "") and not self._LIKELY_SELECTION_RE.match((user_input or "").strip()):
            if not self._ANY_NUMBER_RE.search(user_input or ""):
                return False

        if self._looks_like_greeting_or_thanks(user_input) or self._is_noise(user_input):
            return False

        # FIX #1: A new substantive legal question must NEVER be consumed as a slot answer.
        # This guard lets step 2.9 (_looks_like_legal_question) handle it properly.
        # Exemptions:
        #   - "topic" slot: a legal phrase IS a valid topic selection
        #   - "operation_group" slot: options contain legal keywords (จดทะเบียน, ต่ออายุ, etc.)
        #   - "registration_type" slot: options contain terms like บริษัทจำกัด that match LEGAL_SIGNAL_RE
        #   - "department" slot: user naming a dept (e.g. "กรมพัฒนาธุรกิจการค้า") triggers LEGAL_SIGNAL_RE
        #     but IS a valid slot answer and must not be treated as a new legal question
        _ps = ctx.get("pending_slot")
        pending_key = str((_ps.get("key") if isinstance(_ps, dict) else None) or "")
        _SLOT_REPLY_ALWAYS = {"topic", "operation_group", "registration_type", "department"}
        if pending_key not in _SLOT_REPLY_ALWAYS and self._looks_like_legal_question(user_input):
            _LOG.info(
                "[Supervisor] pending_slot(%r) skipped — input looks like new legal question: %r",
                pending_key, (user_input or "")[:50],
            )
            return False

        # Special case: "topic" slot accepts legal phrases as menu selections, BUT a full
        # question sentence (informational/consequential question) should bypass the menu
        # and be handled as a new question. User can still pick from menu on next turn.
        if pending_key == "topic" and self._INFO_Q_RE.search(user_input):
            _LOG.info(
                "[Supervisor] topic slot skipped — full question bypasses menu: %r",
                (user_input or "")[:60],
            )
            return False

        # topic slot: require a number OR at least one legal/business keyword.
        # Fully off-topic inputs (e.g. "กินข้าวกับอะไรดี") have neither → bypass slot
        # and fall through to normal routing (→ unknown intent → deflect).
        if pending_key == "topic":
            if not self._LIKELY_SELECTION_RE.match(user_input) and not self._LEGAL_SIGNAL_RE.search(user_input):
                _LOG.info(
                    "[Supervisor] topic slot skipped — no legal signal or number in off-topic input: %r",
                    (user_input or "")[:60],
                )
                return False

        return self._looks_like_pending_slot_reply(user_input)

    # Area-size keywords used to detect shop-area dimension in registration_type
    _AREA_KEYWORDS: List[str] = ["น้อยกว่า", "มากกว่า", "ไม่เกิน", "เกิน", "ตารางเมตร"]

    def _get_chroma_collection(self):
        """Return the raw Chroma collection object from the practical retriever, or None."""
        vectorstore = getattr(self._practical.retriever, "vectorstore", None)
        if vectorstore is None:
            return None
        return getattr(vectorstore, "_collection", None)

    def _discover_slots_for_license(self, license_type: str) -> List[Dict]:
        """
        Dynamically discover all slot dimensions that exist in Chroma for a given
        license_type. Returns an ORDERED list of slot dicts::

            [
                {"key": "entity_type",   "options": ["บุคคลธรรมดา", "นิติบุคคล"],
                 "question": "ธุรกิจของคุณเป็นรูปแบบใดครับ?"},
                {"key": "shop_area_type", "options": ["น้อยกว่า 200 ตารางเมตร", ...],
                 "question": "ร้านของคุณมีขนาดพื้นที่ประมาณเท่าไหร่ครับ?"},
                # more custom dimensions detected from registration_type …
            ]

        Detection rules (applied in order; all based on live Chroma data):
        1. entity_type_normalized field  → slot key = "entity_type"
        2. registration_type bullets containing area keywords → slot key = "shop_area_type"
        3. registration_type values that are short, clean labels (not yet captured
           by rules 1-2 and not entity/area noise) → slot key = "registration_type"

        Returns [] if license_type not found, data is empty, or on any error.
        No hardcoded option values — all come from Chroma.
        """
        try:
            coll = self._get_chroma_collection()
            if coll is None:
                return []
            result = coll.get(where={"license_type": license_type}, include=["metadatas"])
            mds = result.get("metadatas") or []
            if not mds:
                return []

            slots: List[Dict] = []
            seen_keys: set = set()

            # Rule -1: department dimension (e.g. multiple banks for QR Payment)
            # Ask FIRST — different departments have completely different procedures/docs.
            # Only add if ≥2 distinct non-empty department values AND each dept has
            # a meaningful number of docs (≥2 docs each), to avoid cases where one dept
            # covers nearly all docs and the other is just a single minor topic.
            # ALSO skip if department is fully determined by entity_type (1:1 mapping)
            # — in that case asking entity_type is sufficient (e.g. ใบทะเบียนพาณิชย์:
            #   บุคคลธรรมดา→สำนักงานเขต, นิติบุคคล→กรมพัฒนาธุรกิจ).
            dept_opts: set = set()
            _dept_doc_count: dict = {}
            # Track which entity_types appear per dept (for correlation check)
            _dept_entities: dict = {}  # dept → set of entity_type values
            for md in mds:
                dept = ((md or {}).get("department") or "").strip()
                entity = ((md or {}).get("entity_type_normalized") or "").strip()
                if dept and dept not in ("nan", "None"):
                    dept_opts.add(dept)
                    _dept_doc_count[dept] = _dept_doc_count.get(dept, 0) + 1
                    _dept_entities.setdefault(dept, set()).add(entity)
            # Each dept must have ≥2 docs to be considered a real split worth asking
            _dept_opts_qualified = {d for d in dept_opts if _dept_doc_count.get(d, 0) >= 2}
            # Check 1:1 correlation: if every qualified dept maps exclusively to
            # one entity_type (ignoring '' universal docs), dept is derivable from entity
            # → no need to ask separately.
            def _dept_determined_by_entity(dept_entities: dict, qualified: set) -> bool:
                for dept in qualified:
                    specific = {e for e in dept_entities.get(dept, set()) if e}
                    if len(specific) > 1:
                        return False  # this dept has mixed entity types → not derivable
                return True
            _dept_is_entity_derived = (
                len(_dept_opts_qualified) >= 2
                and _dept_determined_by_entity(_dept_entities, _dept_opts_qualified)
            )
            if len(_dept_opts_qualified) >= 2 and not _dept_is_entity_derived:
                # Build a generic question that fits any department type (bank, govt office, etc.)
                # Avoid hardcoding "ธนาคาร" which only fits QR Payment context.
                _all_dept_are_banks = all(
                    any(kw in d for kw in ("ธนาคาร", "Bank", "bank"))
                    for d in _dept_opts_qualified
                )
                _dept_question = (
                    "ต้องการสมัครกับธนาคารใดครับ?"
                    if _all_dept_are_banks
                    else "ร้านของคุณยื่นขอที่หน่วยงานใดครับ?"
                )
                slots.append({
                    "key": "department",
                    "options": sorted(_dept_opts_qualified),
                    "question": _dept_question,
                })
                seen_keys.add("department")
                _LOG.info("[Supervisor] discover_slots[%r]: department → %s (counts=%s)", license_type, sorted(_dept_opts_qualified), _dept_doc_count)
            else:
                _LOG.info(
                    "[Supervisor] discover_slots[%r]: department SKIPPED — qualified=%d entity_derived=%s (counts=%s)",
                    license_type, len(_dept_opts_qualified), _dept_is_entity_derived, _dept_doc_count,
                )

            # Rule 0: location dimension (กรุงเทพฯ vs ต่างจังหวัด)
            # Ask this FIRST — it determines which service channel/fee/procedure applies.
            # location is a single-value metadata field per doc — if ≥2 distinct values
            # exist across docs for this license, real split exists → ask the user.
            # Display label is derived from operation_topic of actual docs — NOT hardcoded.
            # e.g. if docs with location='กรุงเทพฯ' have topic 'กรุงเทพฯ และปริมณฑล'
            #      then show 'กรุงเทพฯ และปริมณฑล' to user, but filter key stays 'กรุงเทพฯ'.
            location_opts: set = set()
            # Map: filter_value → display_label (built from real operation_topic)
            _loc_display: dict = {}
            for md in mds:
                loc = ((md or {}).get("location") or "").strip()
                if loc and loc not in ("nan", "None"):
                    location_opts.add(loc)
                    # Extract clean label from operation_topic:
                    # topic may be "ร้านค้าตั้งอยู่ในเขต กรุงเทพฯ และปริมณฑล"
                    # → strip prefix patterns to get just "กรุงเทพฯ และปริมณฑล"
                    topic = ((md or {}).get("operation_topic") or "").strip()
                    if topic and loc in topic and topic != loc:
                        import re as _re
                        # Strip common Thai prefix phrases before the location name
                        # e.g. "ร้านค้าตั้งอยู่ในเขต กรุงเทพฯ และปริมณฑล" → "กรุงเทพฯ และปริมณฑล"
                        _clean = _re.sub(
                            r"^.*?(?:ตั้งอยู่ในเขต|ตั้งอยู่ใน|อยู่ในเขต|อยู่ใน|ในเขต|ในพื้นที่|เขต|พื้นที่|จังหวัด)\s*",
                            "", topic, flags=_re.IGNORECASE
                        ).strip()
                        # If clean result still contains the loc → use it, else fall back to loc itself
                        _loc_display[loc] = _clean if (loc in _clean and _clean != loc) else loc
            if len(location_opts) >= 2:
                _display_opts = [_loc_display.get(loc, loc) for loc in sorted(location_opts)]
                slots.append({
                    "key": "location",
                    "options": _display_opts,
                    "question": "ร้านของคุณตั้งอยู่ในพื้นที่ใดครับ?",
                })
                seen_keys.add("location")
                _LOG.info("[Supervisor] discover_slots[%r]: location → %s (display=%s)", license_type, sorted(location_opts), _display_opts)
            else:
                _LOG.info(
                    "[Supervisor] discover_slots[%r]: location SKIPPED — only %d location value(s) in docs",
                    license_type, len(location_opts),
                )

            # Rule 1: entity_type_normalized dimension
            entity_opts: set = set()
            for md in mds:
                et = ((md or {}).get("entity_type_normalized") or "").strip()
                if et and et not in ("nan", "None"):
                    entity_opts.add(et)
            if len(entity_opts) >= 2:
                slots.append({
                    "key": "entity_type",
                    "options": sorted(entity_opts),
                    "question": "ธุรกิจของคุณเป็นรูปแบบใดครับ?",
                })
                seen_keys.add("entity_type")
                _LOG.info("[Supervisor] discover_slots[%r]: entity_type → %s", license_type, sorted(entity_opts))
            elif entity_opts:
                _LOG.info(
                    "[Supervisor] discover_slots[%r]: entity_type SKIPPED — only 1 value %s (no differentiation)",
                    license_type, entity_opts,
                )

            # Rule 2: shop area dimension (area keywords in registration_type)
            # Only add this slot if there are SEPARATE docs per area option
            # (i.e. area_type actually produces different content — not just the same doc
            # that mentions both sizes in one registration_type string).
            area_opts: set = set()
            for md in mds:
                rt = ((md or {}).get("registration_type") or "").strip()
                if not rt:
                    continue
                for part in re.split(r"[•\n]|\d+\.", rt):
                    part = part.strip()
                    if part and any(kw in part for kw in self._AREA_KEYWORDS):
                        area_opts.add(part)
            if area_opts:
                # Verify: docs must be SPLIT by area — count docs whose registration_type
                # covers only ONE area option (exclusive), not both.
                # If every doc contains ALL area options in the same field → slot is useless.
                _area_splits_docs = False
                for md in mds:
                    rt2 = ((md or {}).get("registration_type") or "").strip()
                    _matched = [a for a in area_opts if a in rt2]
                    if 0 < len(_matched) < len(area_opts):
                        # This doc covers some area options but not all → real split exists
                        _area_splits_docs = True
                        break
                if _area_splits_docs:
                    slots.append({
                        "key": "shop_area_type",
                        "options": sorted(area_opts),
                        "question": "ร้านของคุณมีขนาดพื้นที่ประมาณเท่าไหร่ครับ?",
                    })
                    seen_keys.add("shop_area_type")
                    _LOG.info("[Supervisor] discover_slots[%r]: shop_area_type → %s", license_type, sorted(area_opts))
                else:
                    _LOG.info(
                        "[Supervisor] discover_slots[%r]: shop_area_type SKIPPED — all docs contain same area options (no real split)",
                        license_type,
                    )

            # Rule 2b: area_size metadata field (fallback when Rule 2 skipped)
            # Rule 2 parses registration_type text — misses cases where every doc lists
            # ALL sizes in one field.  Check the explicit area_size metadata field instead:
            # if docs have 2+ distinct area_size values, the split is real.
            if "shop_area_type" not in seen_keys and "area_size" not in seen_keys:
                area_size_opts: set = set()
                for md in mds:
                    as_val = ((md or {}).get("area_size") or "").strip()
                    if as_val and as_val not in ("nan", "None"):
                        area_size_opts.add(as_val)
                if len(area_size_opts) >= 2:
                    slots.append({
                        "key": "area_size",
                        "options": sorted(area_size_opts),
                        "question": "ร้านของคุณมีพื้นที่เท่าไหร่ครับ?",
                    })
                    seen_keys.add("area_size")
                    _LOG.info(
                        "[Supervisor] discover_slots[%r]: area_size (metadata) → %s",
                        license_type, sorted(area_size_opts),
                    )

            # Rule 3: standalone registration_type labels (no entity / area noise)
            # Collect short, clean labels that survived the two filters above.
            # Skip if the value looks like a combined paragraph (> 80 chars or has bullet •)
            rt_opts: set = set()
            known_entity = entity_opts | {"นิติบุคคล", "บุคคลธรรมดา"}
            for md in mds:
                rt = ((md or {}).get("registration_type") or "").strip()
                if not rt:
                    continue
                # Skip combined multi-line values
                if "•" in rt or "\n" in rt or len(rt) > 80:
                    continue
                # Skip if it overlaps with entity or area options already captured
                if any(kw in rt for kw in self._AREA_KEYWORDS):
                    continue
                if rt in known_entity:
                    continue
                # Skip tokens that look like prefixes/headers (e.g. "ประเภทนิติบุคคล")
                if any(rt in e or e in rt for e in known_entity):
                    continue
                # Skip operation-action sub-types (e.g. ย้าย, ปิด, เลิก) — these are scenarios
                # for changing an existing registration, not initial registration sub-types
                _OP_PREFIXES = ("ย้าย", "ปิด", "เลิก", "โอน", "หยุด", "แก้ไข", "เพิ่ม", "ลด")
                if any(rt.startswith(pfx) for pfx in _OP_PREFIXES):
                    continue
                rt_opts.add(rt)
            # ── Normalize rt_opts before building the slot ──────────────────────────
            # Step 1: Split compound "1.X 2.Y" patterns into individual items
            _expanded: set = set()
            _compound_re = re.compile(r'\d+[.)]\s*(.+?)(?=\s+\d+[.)]|$)')
            for _item in rt_opts:
                if re.search(r'\d+[.)]\s*\S', _item):
                    for _p in _compound_re.findall(_item):
                        _p = _p.strip()
                        if _p:
                            _expanded.add(_p)
                else:
                    _expanded.add(_item)

            # Step 2: Remove items that are proper substrings of other items
            # e.g. "บริษัท" ⊂ "บริษัทจำกัด" → drop "บริษัท"
            _clean = {x for x in _expanded if not any(x != y and x in y for y in _expanded)}

            # Step 3: Remove "X และ Y" supersets when both parts are covered by specific items
            # e.g. "ห้างหุ้นส่วนและบริษัท" → parts ["ห้างหุ้นส่วน","บริษัท"] both ⊂ specific options → drop
            _clean2: set = set()
            for x in _clean:
                if "และ" in x:
                    _parts = [p.strip() for p in x.split("และ") if p.strip()]
                    _others = _clean - {x}
                    if _parts and all(any(p in o for o in _others) for p in _parts):
                        _LOG.debug("[Supervisor] drop supertype option %r (all parts covered)", x)
                        continue
                _clean2.add(x)

            rt_opts = _clean2

            # Only useful when there are ≥2 distinct options to choose from
            if len(rt_opts) >= 2 and "registration_type" not in seen_keys:
                slots.append({
                    "key": "registration_type",
                    "options": sorted(rt_opts),
                    "question": self._Q_REGISTRATION_TYPE,
                })
                seen_keys.add("registration_type")
                _LOG.info("[Supervisor] discover_slots[%r]: registration_type → %s", license_type, sorted(rt_opts))

            return slots
        except Exception as e:
            _LOG.warning("[Supervisor] _discover_slots_for_license failed: %s", e)
            return []

    def _get_registration_types_for_docs(self, docs: List[Dict]) -> List[str]:
        """
        Legacy shim — kept for backward compatibility.
        Extracts license_type from docs then delegates to _discover_slots_for_license.
        Returns the options of the FIRST discovered slot (entity_type), or [].
        """
        license_type = None
        for d in (docs or []):
            lt = ((d.get("metadata") or {}).get("license_type") or "").strip()
            if lt:
                license_type = lt
                break
        if not license_type:
            return []
        slots = self._discover_slots_for_license(license_type)
        return slots[0]["options"] if slots else []

    def _get_registration_types_for_entity(self, license_type: str, entity_type_normalized: str) -> List[str]:
        """
        ดึง registration_type options จาก Chroma กรองตาม entity_type_normalized ที่ระบุ
        ใช้เพื่อสร้าง sub-slot สำหรับนิติบุคคล (บริษัทจำกัด vs ห้างหุ้นส่วนจำกัด vs ห้างหุ้นส่วนสามัญ)
        คืนค่า [] ถ้าไม่มีตัวเลือกหลายแบบหรือข้อมูลไม่พอ (ไม่มี hardcode)
        """
        try:
            coll = self._get_chroma_collection()
            if coll is None:
                return []
            result = coll.get(
                where={"$and": [
                    {"license_type": license_type},
                    {"entity_type_normalized": entity_type_normalized},
                ]},
                include=["metadatas"],
            )
            mds = result.get("metadatas") or []
            if not mds:
                return []

            rt_raw: set = set()
            for md in mds:
                rt = ((md or {}).get("registration_type") or "").strip()
                if not rt or rt in ("nan", "None"):
                    continue
                # Skip combined multi-line values
                if "•" in rt or "\n" in rt or len(rt) > 80:
                    continue
                # Skip area-type values
                if any(kw in rt for kw in self._AREA_KEYWORDS):
                    continue
                # Skip the entity-type value itself
                if rt == entity_type_normalized:
                    continue
                rt_raw.add(rt)

            if not rt_raw:
                return []

            # Normalize: split compound "1.X 2.Y" patterns 
            _expanded: set = set()
            _compound_re = re.compile(r'\d+[.)]\s*(.+?)(?=\s+\d+[.)]|$)')
            for _item in rt_raw:
                if re.search(r'\d+[.)]\s*\S', _item):
                    for _p in _compound_re.findall(_item):
                        _p = _p.strip()
                        if _p:
                            _expanded.add(_p)
                else:
                    _expanded.add(_item)

            # Remove SHORT substrings covered by a LONGER/more-specific label
            # e.g. if both 'บริษัท' and 'บริษัทจำกัด' exist → keep 'บริษัทจำกัด', drop 'บริษัท'
            _clean = {x for x in _expanded if not any(x != y and x in y for y in _expanded)}

            # Remove "X และ Y" supersets when both parts are covered by specific items
            _clean2: set = set()
            for x in _clean:
                if "และ" in x:
                    _parts = [p.strip() for p in x.split("และ") if p.strip()]
                    _others = _clean - {x}
                    if _parts and all(any(p in o for o in _others) for p in _parts):
                        continue
                _clean2.add(x)

            result_list = sorted(_clean2)
            _LOG.info(
                "[Supervisor] registration_types_for_entity: license=%r entity=%r → %s",
                license_type, entity_type_normalized, result_list,
            )
            return result_list
        except Exception as e:
            _LOG.warning("[Supervisor] _get_registration_types_for_entity failed: %s", e)
            return []

    def _detect_license_types_from_query(self, query: str) -> List[str]:
        """
        Predictive multi-topic detection: scan query for ALL license type keywords BEFORE retrieval.
        Returns list of canonical license_type names found in the query.
        """
        q = query or ""
        found: List[str] = []
        for pattern, license_name in self._MULTI_TOPIC_LICENSE_KEYWORDS:
            if re.search(pattern, q, re.IGNORECASE) and license_name not in found:
                found.append(license_name)
        return found

    def _infer_operation_group_from_query(self, query: str, op_groups: List[str]) -> Optional[str]:
        """
        If user query clearly implies a specific operation type, return matching op_group label.
        Returns None if ambiguous — must ask user.
        """
        q = query or ""

        def _find_group(keywords: list) -> Optional[str]:
            for g in op_groups:
                if any(kw in g for kw in keywords):
                    return g
            return None

        if self._OP_INFER_RENEW_RE.search(q):
            found = _find_group(["ต่ออายุ", "ต่อ"])
            if found:
                return found
        if self._OP_INFER_CANCEL_RE.search(q):
            found = _find_group(["ยกเลิก", "ปิด"])
            if found:
                return found
        if self._OP_INFER_EDIT_RE.search(q):
            found = _find_group(["แก้ไข", "เปลี่ยนแปลง"])
            if found:
                return found
        if self._OP_INFER_NEW_RE.search(q):
            found = _find_group(["ใหม่", "จดทะเบียน", "ยื่นขอ"])
            if found:
                return found
        return None

    def _infer_entity_type_from_query(self, query: str) -> Optional[str]:
        """
        Auto-detect entity type when it is explicitly stated in the query text.
        Returns 'นิติบุคคล' or 'บุคคลธรรมดา', or None if ambiguous.
        Used to skip asking entity_type slot when the answer is already in the query.
        """
        q = query or ""
        if self._ENTITY_NITI_RE.search(q):
            return "นิติบุคคล"
        if self._ENTITY_NATURAL_RE.search(q):
            return "บุคคลธรรมดา"
        return None

    def _entity_slot_needed(self, query: str) -> bool:
        """
        Returns False if the entity_type slot should be skipped for this query.
        Universal-fact queries (e.g. "มีอายุกี่ปี", "หมดอายุไหม") have answers
        that are identical across entity types — asking entity_type adds no value.
        Returns True (ask) by default; only False for matched universal-fact patterns.
        """
        return not self._UNIVERSAL_FACT_RE.search(query or "")

    def _get_operation_groups_for_entity(self, license_type: str, entity_type_normalized: str) -> Tuple[List[str], Dict]:
        """
        ดึง operation_by_department จาก Chroma แล้วจัดกลุ่มเป็นหมวดหลัก
        คืน (slot_options, raw_op_map) เสมอ:
          slot_options  = list ของ display labels ที่จะแสดงให้ user เลือก
          raw_op_map    = {display_label: [raw operation_by_department values]}
                         ใช้สร้าง enriched query ตอน retrieval
        ไม่มี hardcode — ทุกอย่างมาจากข้อมูลจริงใน Chroma
        """
        try:
            vectorstore = getattr(self._practical.retriever, "vectorstore", None)
            if vectorstore is None:
                return [], {}
            coll = getattr(vectorstore, "_collection", None)
            if coll is None:
                return [], {}

            result = coll.get(
                where={"$and": [
                    {"license_type": license_type},
                    {"entity_type_normalized": entity_type_normalized},
                ]},
                include=["metadatas"],
            )

            ops: set = set()
            for md in (result.get("metadatas") or []):
                op = ((md or {}).get("operation_by_department") or "").strip()
                if op:
                    ops.add(op)

            if not ops:
                return [], {}

            # ── Expand op set: also query without entity filter to catch shared operations ──
            # (some op types are stored without entity_type_normalized and would be missed)
            result_all = coll.get(where={"license_type": license_type}, include=["metadatas"])
            ops_all: set = set()
            for md in (result_all.get("metadatas") or []):
                op = ((md or {}).get("operation_by_department") or "").strip()
                if op:
                    ops_all.add(op)
            # Merge: entity-specific ops + license-wide ops
            combined_ops = ops | ops_all

            # ── Cache check: avoid calling LLM repeatedly for same (license, entity) ──
            _cache_key = (license_type, entity_type_normalized)
            if _cache_key in self._op_groups_cache:
                _cached = self._op_groups_cache[_cache_key]
                _LOG.info("[Supervisor] op_groups cache hit: %r → %s", license_type, _cached[0])
                return _cached

            # ── Try LLM classifier first (zero hardcode, handles future license types) ──
            def _build_from_llm(op_set: set) -> Optional[Tuple[List[str], Dict]]:
                try:
                    res = self._llm_op_group_classifier(license_type, sorted(op_set)) or {}
                    groups_raw = res.get("groups") or []
                    if not groups_raw:
                        return None
                    slot_opts: List[str] = []
                    raw_map: Dict[str, List[str]] = {}
                    for g in groups_raw:
                        label = str(g.get("label") or "").strip()
                        raw_vals = [str(v).strip() for v in (g.get("raw") or []) if str(v).strip()]
                        if label and raw_vals:
                            slot_opts.append(label)
                            raw_map[label] = raw_vals
                    if slot_opts:
                        _LOG.info("[Supervisor] op_groups LLM: %r → %s", license_type, slot_opts)
                        return slot_opts, raw_map
                except Exception as e:
                    _LOG.warning("[Supervisor] op_groups LLM classifier failed: %s", e)
                return None

            llm_result = _build_from_llm(combined_ops)
            if llm_result:
                self._op_groups_cache[_cache_key] = llm_result
                _LOG.info("[Supervisor] operation_groups: license=%r entity=%r → %s (LLM)",
                          license_type, entity_type_normalized, llm_result[0])
                return llm_result

            #Rule-based fallback (fast, no LLM cost, catches common Thai prefixes)
            # Used when LLM is unavailable or times out.
            # Covers existing data well; new license types with unknown prefixes fall to "อื่น ๆ"
            # which still works — user sees the group and can select it.
            def _ops_to_groups_fallback(op_set: set) -> List[Tuple[str, set]]:
                _NEW_PREFIXES = (
                    "การจดทะเบียน", "การจดภาษี", "การจด",
                    "การขอ", "การสมัคร", "การยื่น",
                    "การขึ้นทะเบียน", "การขออนุมัติ",
                    "การขอรับรอง", "การขอใบรับรอง",
                    "เปิดบัญชี", "จัดตั้ง", "ยื่นใหม่", "เปิดใหม่", "ขอใบรับรอง",
                )
                _RENEW_PREFIXES = ("ต่ออายุ", "การต่ออายุ")
                _new, _renew, _edit, _cancel, _move, _add, _close, _other = \
                    set(), set(), set(), set(), set(), set(), set(), set()
                for o in op_set:
                    if any(o.startswith(pfx) for pfx in _RENEW_PREFIXES) or re.match(r"^อายุ", o):
                        _renew.add(o)
                    elif o.startswith("แก้ไข") or o.startswith("เปลี่ยนแปลง") or o.startswith("การแก้ไข"):
                        _edit.add(o)
                    elif o.startswith("ยกเลิก") or o.startswith("เลิก"):
                        _cancel.add(o)
                    elif o.startswith("ย้าย"):
                        _move.add(o)
                    elif o.startswith("เพิ่ม"):
                        _add.add(o)
                    elif (o.startswith("ปิด") or o.startswith("การปิด")) and "งบการเงิน" not in o:
                        _close.add(o)
                    elif any(o.startswith(pfx) for pfx in _NEW_PREFIXES):
                        _new.add(o)
                    else:
                        _other.add(o)
                g = []
                if _new:    g.append(("ยื่นขอใบอนุญาตใหม่ / จดทะเบียน", _new))
                if _renew:  g.append(("ต่ออายุใบอนุญาต",               _renew))
                if _edit:   g.append(("แก้ไข / เปลี่ยนแปลงรายการ",    _edit))
                if _cancel: g.append(("ยกเลิกใบอนุญาต",                _cancel))
                if _move:   g.append(("ย้ายสถานประกอบการ",             _move))
                if _add:    g.append(("เพิ่มสถานประกอบการ",            _add))
                if _close:  g.append(("ปิดสถานประกอบการ",              _close))
                if _other:  g.append(("อื่น ๆ",                        _other))
                return g

            group_tuples = _ops_to_groups_fallback(combined_ops)
            slot_options = [label for label, _ in group_tuples]
            raw_op_map = {label: sorted(raw_ops) for label, raw_ops in group_tuples}
            fallback_result: Tuple[List[str], Dict] = (slot_options, raw_op_map)
            self._op_groups_cache[_cache_key] = fallback_result

            _LOG.info("[Supervisor] operation_groups: license=%r entity=%r → %s (rule-based fallback)",
                      license_type, entity_type_normalized, slot_options)
            return fallback_result
        except Exception as e:
            _LOG.warning("[Supervisor] _get_operation_groups_for_entity failed: %s", e)
            return [], {}

    def _route_pending_slot_to_persona(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        _raw_pending = (state.context or {}).get("pending_slot")
        pending = _raw_pending if isinstance(_raw_pending, dict) else {}
        mapped, err = self._map_pending_slot_reply(pending, user_input)

        if err:
            msg = self._normalize_male(err)
            self._add_assistant(state, msg)
            state.last_action = "pending_slot_invalid_reply"
            return state, msg

        if not mapped:
            msg = self._normalize_male("🙏 กรุณาตอบเป็นตัวเลขตามตัวเลือกครับ")
            self._add_assistant(state, msg)
            state.last_action = "pending_slot_invalid_reply"
            return state, msg

        state.context = state.context or {}
        if isinstance(pending, dict) and pending.get("key") == "topic":
            state.context["last_topic"] = str(mapped).strip()
            # Also set last_user_legal_query so Academic auto-replay works after mode switch
            state.context["last_user_legal_query"] = str(mapped).strip()
            # Save menu for one-turn recovery (lets user pick another topic by number)
            state.context["last_topic_menu"] = list(pending.get("options") or [])
        elif isinstance(pending, dict) and pending.get("key") and mapped:
            # Non-topic slot (e.g. registration_type=นิติบุคคล) → cross-persona slot memory
            # so Academic mode can read it via state.get_collected_slots()
            try:
                _slot_key_sv = pending["key"]
                _slot_val_sv = str(mapped)
                state.save_collected_slot(_slot_key_sv, _slot_val_sv)
                # Normalize: if key='choice' but value is entity type → also save as entity_type
                # This happens when LLM asks entity_type question but _infer_slot_key falls back to 'choice'
                if _slot_key_sv == "choice" and _slot_val_sv in ("นิติบุคคล", "บุคคลธรรมดา", "นิติ"):
                    state.save_collected_slot("entity_type", _slot_val_sv)
                    _LOG.info("[Supervisor] normalized choice→entity_type: %r", _slot_val_sv)
                elif _slot_key_sv == "choice":
                    # Value may be an entity sub-type (e.g. บริษัทจำกัด, ห้างหุ้นส่วนจำกัด)
                    # Save as both entity_type (normalized) and registration_type so slot_queue
                    # rebuild never re-asks either slot.
                    try:
                        from service.data_loader import DataLoader as _DL_cs
                        _cs_norm = _DL_cs._normalize_entity_type(_slot_val_sv)
                        if _cs_norm:
                            state.save_collected_slot("entity_type", _cs_norm)
                            state.save_collected_slot("registration_type", _slot_val_sv)
                            _LOG.info(
                                "[Supervisor] choice sub-type %r → entity_type=%r + registration_type saved",
                                _slot_val_sv, _cs_norm,
                            )
                    except Exception:
                        pass
                # Auto-fill registration_type when entity_type was answered with a specific sub-type.
                # Happens when the LLM merges entity+registration into one question and user picks
                # e.g. "บริษัทจำกัด" (not "นิติบุคคล") — the sub-type answer implicitly fills both slots.
                if _slot_key_sv == "entity_type":
                    try:
                        from service.data_loader import DataLoader as _DL_at
                        _et_norm_at = _DL_at._normalize_entity_type(_slot_val_sv)
                        if _et_norm_at and _et_norm_at != _slot_val_sv:
                            state.save_collected_slot("registration_type", _slot_val_sv)
                            _LOG.info(
                                "[Supervisor] entity_type sub-type %r → auto-fill registration_type",
                                _slot_val_sv,
                            )
                    except Exception:
                        pass
            except Exception:
                pass

        state.context.pop("pending_slot", None)

        # Pre-retrieve docs so the LLM has documents when called with _internal=True
        # (_internal=True skips the retrieval block inside persona.handle)
        if isinstance(pending, dict) and pending.get("key") == "topic" and mapped:
            # Predictive multi-topic: if user's ORIGINAL message mentions ≥2 license types,
            # retrieve ALL their docs and answer everything in one response — don't single-topic.
            _multi_from_input = self._detect_license_types_from_query(user_input or "")
            if len(_multi_from_input) >= 2:
                try:
                    _vstore_mt2 = getattr(self.retriever, "vectorstore", None)
                    _coll_mt2 = getattr(_vstore_mt2, "_collection", None) if _vstore_mt2 else None
                    if _coll_mt2 is not None:
                        _doc_chars_mt2 = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                        _SUPERVISOR_META_WL_MT2 = frozenset({
                            "license_type", "operation_topic",
                            "entity_type_normalized", "registration_type", "department",
                            "fees", "operation_duration", "service_channel",
                            "operation_steps", "identification_documents", "research_reference",
                        })
                        _SUPERVISOR_FC_MT2 = {
                            "operation_steps": 600, "identification_documents": 1500,
                            "research_reference": 3200, "fees": 120, "service_channel": 200,
                        }
                        _merged_mt2: List[Dict] = []
                        for _lt_name2 in _multi_from_input:
                            _r2 = _coll_mt2.get(
                                where={"license_type": _lt_name2},
                                include=["documents", "metadatas"],
                            )
                            for _fc2, _fm2 in zip(_r2.get("documents") or [], _r2.get("metadatas") or []):
                                _sm2: Dict = {}
                                for _k2, _v2 in (_fm2 or {}).items():
                                    if _k2 not in _SUPERVISOR_META_WL_MT2 or _v2 in (None, "", "nan", "None"):
                                        continue
                                    _vs2 = str(_v2)
                                    _cap2 = _SUPERVISOR_FC_MT2.get(_k2)
                                    _sm2[_k2] = _vs2[:_cap2] if _cap2 and len(_vs2) > _cap2 else _vs2
                                _merged_mt2.append({"content": (_fc2 or "")[:_doc_chars_mt2], "metadata": _sm2})
                        if _merged_mt2:
                            state.current_docs = _merged_mt2
                            state.last_retrieval_query = user_input or str(mapped)
                            state.context["_multi_topic_retrieval"] = True
                            state.context.pop("topic_slot_queue", None)
                            _LOG.info(
                                "[Supervisor] topic-slot multi-topic retrieval: %s → %d total docs merged",
                                _multi_from_input, len(_merged_mt2),
                            )
                            # Skip all slot building and go directly to practical answer
                            return self._practical.handle(state, user_input, _internal=True)
                except Exception as _e_mt2:
                    _LOG.warning("[Supervisor] topic-slot multi-topic retrieval failed: %s — single-topic fallback", _e_mt2)

            try:
                # Step 1: similarity search to detect license_type
                _initial_docs = self._practical._retrieve_docs(str(mapped))
                _license_detected = None
                for _d in _initial_docs:
                    _lt = ((_d.get("metadata") or {}).get("license_type") or "").strip()
                    if _lt:
                        _license_detected = _lt
                        break

                # Step 2: coll.get() — ALL docs for license (no k limit, no similarity bias)
                # Generic docs (entity='') score higher on topic-name similarity than specific docs
                # (entity='นิติบุคคล', location-specific) → top-6 misses specific docs entirely.
                # coll.get() avoids this bias by fetching every doc with the matching license_type.
                _used_full = False
                if _license_detected:
                    try:
                        _coll = self._get_chroma_collection()
                        if _coll is not None:
                            _result = _coll.get(
                                where={"license_type": _license_detected},
                                include=["documents", "metadatas"],
                            )
                            _max_chars = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                            _all_docs = [
                                {"content": (c or "")[:_max_chars], "metadata": m or {}}
                                for c, m in zip(
                                    _result.get("documents") or [],
                                    _result.get("metadatas") or [],
                                )
                            ]
                            if _all_docs:
                                state.current_docs = _all_docs
                                _used_full = True
                                _LOG.info(
                                    "[Supervisor] full-license pre-retrieved %d docs for license=%r",
                                    len(_all_docs), _license_detected,
                                )
                    except Exception as _e2:
                        _LOG.warning(
                            "[Supervisor] full-license retrieval failed: %s — fallback to similarity",
                            _e2,
                        )

                if not _used_full:
                    state.current_docs = _initial_docs
                    _LOG.info(
                        "[Supervisor] pre-retrieved %d docs (similarity) for topic=%r",
                        len(_initial_docs), str(mapped)[:40],
                    )

                state.last_retrieval_query = str(mapped)
            except Exception as e:
                _LOG.warning("[Supervisor] pre-retrieve failed for topic=%r: %s", str(mapped)[:40], e)
                state.current_docs = []

            # Multi-license check FIRST: if initial retrieval spans multiple license types,
            # skip the slot queue entirely and do per-topic retrieval so LLM can answer all topics.
            _lt_counts: dict = {}
            for _d in (state.current_docs or []):
                if not isinstance(_d, dict):
                    continue
                _lt = ((_d.get("metadata") or {}).get("license_type") or "").strip()
                if _lt:
                    _lt_counts[_lt] = _lt_counts.get(_lt, 0) + 1
            _lt_total = sum(_lt_counts.values())
            _lt_unique = list(_lt_counts.keys())
            _is_multi_license = False
            if len(_lt_unique) > 1:
                _lt_top = max(_lt_counts, key=_lt_counts.get)
                _lt_top_count = _lt_counts[_lt_top]
                _lt_sorted = sorted(_lt_counts.values(), reverse=True)
                _lt_second = _lt_sorted[1] if len(_lt_sorted) > 1 else 0
                # Not dominant → genuine multi-topic question
                if not (_lt_total > 0 and (
                    _lt_top_count / _lt_total >= 0.6
                    or (_lt_second > 0 and _lt_top_count >= _lt_second * 2)
                )):
                    _is_multi_license = True

            if _is_multi_license:
                _LOG.info(
                    "[Supervisor] topic pre-retrieve multi-license detected %s — per-topic retrieval",
                    _lt_unique,
                )
                state.context["multi_license_topics"] = _lt_unique
                _TOPIC_QUERIES_PT = {
                    "ใบภาษีมูลค่าเพิ่ม ภพ.20": "ภาษีมูลค่าเพิ่ม ภพ.20 จด VAT กรมสรรพากร",
                    "ใบอนุญาตจำหน่ายสุรา": "ใบอนุญาตจำหน่ายสุรา สรรพสามิต ภส.08",
                    "แบบแสดงรายการภาษีป้าย": "ภาษีป้าย แบบ ภป.1 ป้ายร้านอาหาร",
                    "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร": "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร BMA OSS",
                    "ใบทะเบียนพาณิชย์": "จดทะเบียนพาณิชย์ DBD",
                }
                _pt_docs: List[Dict[str, Any]] = []
                _pt_seen: set = set()
                _pt_doc_chars = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                _PT_META_WL = frozenset({
                    "license_type", "operation_topic", "chunk_type",
                    "entity_type_normalized", "registration_type", "department",
                    "fees", "operation_duration", "service_channel",
                    "operation_steps", "identification_documents", "research_reference",
                    "legal_regulatory",     # บทลงโทษ ค่าปรับ ข้อกำหนดทางกฎหมาย
                    "terms_and_conditions", # หน้าที่และเงื่อนไขของผู้ประกอบการ
                })
                _PT_FIELD_CAPS = {
                    "operation_steps": 600, "identification_documents": 1500,
                    "research_reference": 3200, "fees": 120, "service_channel": 200,
                    "legal_regulatory": 2000, "terms_and_conditions": 800,
                }
                for _lt_item in _lt_unique:
                    _tq = _TOPIC_QUERIES_PT.get(_lt_item, _lt_item)
                    try:
                        _tdocs = self.retriever.invoke(_tq)
                        _added = 0
                        for _td in (_tdocs or [])[:5]:
                            _tc = (getattr(_td, "page_content", "") or "")[:_pt_doc_chars]
                            _ck = _tc[:80]
                            if _ck in _pt_seen:
                                continue
                            _pt_seen.add(_ck)
                            _rm = getattr(_td, "metadata", {}) or {}
                            _sm: Dict[str, Any] = {}
                            for _k, _v in _rm.items():
                                if _k not in _PT_META_WL or _v in (None, "", "nan", "None"):
                                    continue
                                _vs = str(_v)
                                _cap = _PT_FIELD_CAPS.get(_k)
                                _sm[_k] = _vs[:_cap] if _cap and len(_vs) > _cap else _vs
                            _pt_docs.append({"content": _tc, "metadata": _sm})
                            _added += 1
                            if _added >= 3:
                                break
                        _LOG.info("[Supervisor] per-topic %r → %d docs", _lt_item, _added)
                    except Exception as _e_pt:
                        _LOG.warning("[Supervisor] per-topic retrieval failed %r: %s", _lt_item, _e_pt)
                if _pt_docs:
                    state.current_docs = _pt_docs
                    _LOG.info("[Supervisor] multi-license merged: %d docs total", len(_pt_docs))
                state.context.pop("topic_slot_queue", None)
                _slot_queue = []  # skip slot building below
            else:
                # Dynamic slot discovery: query Chroma to find ALL slot dimensions
                # for this license_type (entity_type, shop_area_type, registration_type, …)
                # Result is an ordered queue — practical pops one slot at a time.
                _license_type_for_slots = None
                for _d in (state.current_docs or []):
                    if not isinstance(_d, dict):
                        continue
                    _lt = ((_d.get("metadata") or {}).get("license_type") or "").strip()
                    if _lt:
                        _license_type_for_slots = _lt
                        break
                if _license_type_for_slots:
                    _slot_queue = self._discover_slots_for_license(_license_type_for_slots)
                    # Step 2: auto-infer entity_type if explicitly stated in query text
                    _q_str_main = str(mapped).strip()
                    _inferred_et_main = self._infer_entity_type_from_query(_q_str_main)
                    if _inferred_et_main:
                        _cs_main = state.get_collected_slots() or {} if hasattr(state, "get_collected_slots") else {}
                        if not _cs_main.get("entity_type"):
                            state.save_collected_slot("entity_type", _inferred_et_main)
                            state.context.setdefault("slots", {})["entity_type"] = _inferred_et_main
                            _LOG.info("[Supervisor] entity_type auto-inferred from query text: %r", _inferred_et_main)
                    # Step 3: skip entity_type/registration_type for universal-fact queries
                    elif not self._entity_slot_needed(_q_str_main):
                        _slot_queue = [s for s in _slot_queue if s["key"] not in ("entity_type", "registration_type")]
                        _LOG.info("[Supervisor] entity_type/registration_type skipped — universal-fact query: %r", _q_str_main[:60])
                else:
                    _slot_queue = []

            # Cross-topic slot memory: skip slots the user already answered
            # (e.g. entity_type answered in previous topic → don't ask again)
            if _slot_queue:
                try:
                    _prev_slots_cs = state.get_collected_slots() or {}
                    _prev_slots_ctx = (state.context or {}).get("slots") or {}
                    _prev_slots = {**_prev_slots_ctx, **_prev_slots_cs}  # collected_slots wins
                except Exception:
                    _prev_slots = {}

                if _prev_slots:
                    _new_queue: List[Dict] = []
                    _known_entity_for_new_topic: Optional[str] = None
                    for _s in _slot_queue:
                        _skey = _s.get("key")
                        if _skey in _prev_slots:
                            _LOG.info(
                                "[Supervisor] skip slot %r (already answered=%r in earlier topic)",
                                _skey, _prev_slots[_skey],
                            )
                            # If entity_type already known, remember it so we can apply
                            # FIX-P-A logic (filter registration_type + append op_group)
                            if _skey == "entity_type":
                                try:
                                    from service.data_loader import DataLoader as _DL_et
                                    _known_entity_for_new_topic = _DL_et._normalize_entity_type(_prev_slots[_skey])
                                except Exception:
                                    _raw_et = _prev_slots[_skey]
                                    if "นิติบุคคล" in _raw_et or "บริษัท" in _raw_et or "ห้างหุ้นส่วน" in _raw_et:
                                        _known_entity_for_new_topic = "นิติบุคคล"
                                    elif "บุคคลธรรมดา" in _raw_et:
                                        _known_entity_for_new_topic = "บุคคลธรรมดา"
                        else:
                            _new_queue.append(_s)

                    # If entity_type was already known, apply FIX-P-A:
                    # filter registration_type + add operation_group slot
                    if _known_entity_for_new_topic and _license_type_for_slots:
                        # Re-retrieve docs with entity filter for better context
                        _base_q = str(mapped).strip()
                        try:
                            state.current_docs = self._practical._retrieve_docs(
                                f"{_base_q} {_known_entity_for_new_topic}".strip(),
                                metadata_filter={"entity_type_normalized": _known_entity_for_new_topic},
                            )
                            state.last_retrieval_query = f"{_base_q} {_known_entity_for_new_topic}".strip()
                            _LOG.info(
                                "[Supervisor] new-topic entity-enriched retrieval: entity=%r docs=%d",
                                _known_entity_for_new_topic, len(state.current_docs),
                            )
                        except Exception as _e:
                            _LOG.warning("[Supervisor] new-topic entity retrieval failed: %s", _e)

                        _filtered_nq: List[Dict] = []
                        for _slot in _new_queue:
                            if _slot.get("key") == "registration_type":
                                _ert_opts = self._get_registration_types_for_entity(
                                    _license_type_for_slots, _known_entity_for_new_topic
                                )
                                if len(_ert_opts) >= 2:
                                    _filtered_nq.append({
                                        "key": "registration_type",
                                        "options": _ert_opts,
                                        "question": _slot.get("question", self._Q_REGISTRATION_TYPE),
                                    })
                                    _LOG.info(
                                        "[Supervisor] (new-topic) registration_type filtered for entity=%r → %s",
                                        _known_entity_for_new_topic, _ert_opts,
                                    )
                                else:
                                    _LOG.info(
                                        "[Supervisor] (new-topic) registration_type ≤1 opt for entity=%r → skip",
                                        _known_entity_for_new_topic,
                                    )
                            else:
                                _filtered_nq.append(_slot)
                        _new_queue = _filtered_nq

                        # Append operation_group slot (skip if already collected)
                        _op_res_nt = self._get_operation_groups_for_entity(
                            _license_type_for_slots, _known_entity_for_new_topic
                        )
                        _op_grps_nt, _raw_op_map_nt = (
                            _op_res_nt if isinstance(_op_res_nt, tuple) else (_op_res_nt, {})
                        )
                        if len(_op_grps_nt) >= 2 and "operation_group" not in _prev_slots:
                            _query_for_infer_nt = (state.context or {}).get("last_user_legal_query") or user_input
                            _inferred_nt = self._infer_operation_group_from_query(_query_for_infer_nt, _op_grps_nt)
                            if _inferred_nt:
                                state.save_collected_slot("operation_group", _inferred_nt)
                                state.context.setdefault("slots", {})["confirmed_operation"] = _inferred_nt
                                _LOG.info("[Supervisor] (new-topic) operation_group auto-inferred=%r — skipping ask", _inferred_nt)
                            else:
                                _new_queue.append({
                                    "key": "operation_group",
                                    "options": _op_grps_nt,
                                    "question": self._Q_OPERATION_GROUP,
                                    "raw_op_map": _raw_op_map_nt,
                                })
                                _LOG.info(
                                    "[Supervisor] (new-topic) operation_group appended → %s", _op_grps_nt
                                )
                        elif "operation_group" in _prev_slots:
                            _LOG.info("[Supervisor] (new-topic) operation_group already collected — skip")

                    _slot_queue = _new_queue

            # Step 4: cap at 2 slots max to prevent question loops
            if len(_slot_queue) > 2:
                _slot_queue = _slot_queue[:2]
                _LOG.info("[Supervisor] slot_queue capped to 2: %s", [s["key"] for s in _slot_queue])

            if _slot_queue:
                state.context["topic_slot_queue"] = _slot_queue
                _LOG.info("[Supervisor] topic_slot_queue set: %s", [s["key"] for s in _slot_queue])
            else:
                state.context.pop("topic_slot_queue", None)
            # Clear legacy keys to avoid conflicts
            state.context.pop("topic_registration_types", None)
            state.context.pop("topic_area_types", None)

        elif isinstance(pending, dict) and pending.get("key") and mapped:
            # Any non-topic slot filled — enrich retrieval if entity type can be inferred.
            # Use data_loader's _normalize_entity_type to cover all sub-types:
            # บริษัทจำกัด, ห้างหุ้นส่วน, บุคคลธรรมดา ฯลฯ → นิติบุคคล / บุคคลธรรมดา
            _raw = str(mapped).strip()
            _entity_val = None
            try:
                from service.data_loader import DataLoader as _DL
                _entity_val = _DL._normalize_entity_type(_raw)
            except Exception:
                # Fallback: simple keyword check
                if "นิติบุคคล" in _raw or "บริษัท" in _raw or "ห้างหุ้นส่วน" in _raw:
                    _entity_val = "นิติบุคคล"
                elif "บุคคลธรรมดา" in _raw:
                    _entity_val = "บุคคลธรรมดา"

            base_q = (
                getattr(state, "last_retrieval_query", None)
                or (state.context or {}).get("last_retrieval_query")
                or (state.context or {}).get("last_user_legal_query")
                or ""
            ).strip()

            # Level 3: ถ้า pending slot เป็น operation_group → กรอง docs ตาม operation ที่เลือก
            _pending_key = pending.get("key") if isinstance(pending, dict) else None
            if _pending_key == "operation_group":
                _op_group_val = str(mapped).strip()
                # ── Use raw_op_map stored in the slot (built from real ChromaDB values) ──
                # raw_op_map: {display_label: [raw operation_by_department values]}
                # This replaces the hardcoded _op_kw_map and eliminates false-positive substring matching.
                _raw_op_map = pending.get("raw_op_map") or {}
                _raw_ops_for_label = _raw_op_map.get(_op_group_val) or []
                # Build op prefix from raw values (first raw value is usually the most specific keyword)
                _op_prefix = " ".join(_raw_ops_for_label[:2]) if _raw_ops_for_label else _op_group_val

                # Recover entity_type from previously saved slots
                _saved_entity = None
                try:
                    _saved_entity = (state.get_collected_slots() or {}).get("entity_type")
                    if _saved_entity:
                        from service.data_loader import DataLoader as _DL2
                        _saved_entity = _DL2._normalize_entity_type(_saved_entity)
                except Exception:
                    pass

                # Recover license_type from last_topic (set when user chose the topic menu)
                _saved_license = (state.context or {}).get("last_topic") or ""

                if _saved_entity:
                    # Filter by BOTH entity_type AND license_type so docs from other license types
                    # (e.g. ใบทะเบียนพาณิชย์) cannot sneak in and confuse the LLM.
                    _all_slots_for_q = state.get_collected_slots() or {}
                    _slot_q_parts = [v for k, v in _all_slots_for_q.items()
                                     if k not in ("entity_type", "operation_group") and v]
                    enriched_q = " ".join(filter(None, [base_q, _op_group_val, _op_prefix or ""] + _slot_q_parts)).strip()
                    try:
                        _k_for_op = int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 6))
                        # Build combined filter: entity_type + license_type (if known)
                        _op_meta_filter: dict = {"entity_type_normalized": _saved_entity}
                        if _saved_license:
                            _op_meta_filter["license_type"] = _saved_license
                        state.current_docs = self._practical._retrieve_docs(
                            enriched_q,
                            metadata_filter=_op_meta_filter,
                            max_docs=_k_for_op,
                        )
                        # Fallback: if combined filter returns 0 docs, relax to entity-only
                        if not state.current_docs and _saved_license:
                            _LOG.info("[Supervisor] op-group combined filter 0 docs → relax to entity-only")
                            state.current_docs = self._practical._retrieve_docs(
                                enriched_q,
                                metadata_filter={"entity_type_normalized": _saved_entity},
                                max_docs=_k_for_op,
                            )
                        state.last_retrieval_query = enriched_q
                        _LOG.info(
                            "[Supervisor] op-group retrieval: entity=%r license=%r op=%r docs=%d",
                            _saved_entity, _saved_license, _op_group_val, len(state.current_docs),
                        )
                    except Exception as e:
                        _LOG.warning("[Supervisor] op-group retrieval failed: %s", e)

                # Stamp the confirmed topic+operation into slots so LLM sees it in context
                if _saved_license:
                    state.context.setdefault("slots", {})["confirmed_topic"] = _saved_license
                if _op_group_val:
                    state.context.setdefault("slots", {})["confirmed_operation"] = _op_group_val

                # BUG-A fix: operation_group slot has been consumed — clear stale queue so
                # subsequent action='ask' fallback cannot pop it and show a wrong menu.
                state.context.pop("topic_slot_queue", None)
                _LOG.info("[Supervisor] topic_slot_queue cleared after operation_group consumed")

            elif _pending_key == "department":
                # Department slot filled (e.g. user chose ธนาคารกสิกรไทย vs ธนาคารไทยพาณิชย์)
                # Re-retrieve docs filtered by chosen department + entity_type if known.
                try:
                    _dept_val = str(mapped).strip()
                    _saved_entity_dept = (state.get_collected_slots() or {}).get("entity_type")
                    _dept_filter: dict = {"department": _dept_val}
                    if _saved_entity_dept:
                        _dept_filter["entity_type_normalized"] = _saved_entity_dept
                    _dept_docs = self._practical._retrieve_docs(
                        state.last_retrieval_query or _dept_val,
                        metadata_filter=_dept_filter,
                        max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 6)),
                    )
                    if _dept_docs:
                        state.current_docs = _dept_docs
                        _LOG.info(
                            "[Supervisor] department retrieval: dept=%r entity=%r docs=%d",
                            _dept_val, _saved_entity_dept, len(_dept_docs),
                        )
                    else:
                        _LOG.info("[Supervisor] department retrieval: no docs found, keeping existing")
                except Exception as _e_dept:
                    _LOG.warning("[Supervisor] department retrieval failed: %s", _e_dept)

            elif _pending_key in ("area_size", "shop_area_type", "operation_location", "location_scope", "location"):
                # Area/location slot filled → re-retrieve docs with both entity + location/area_size
                # metadata filters so LLM gets the exact doc for this user's path.
                # New v3 schema has location + area_size as explicit metadata fields.
                _saved_entity2 = None
                try:
                    _saved_entity2 = (state.get_collected_slots() or {}).get("entity_type")
                    if _saved_entity2:
                        from service.data_loader import DataLoader as _DL3
                        _saved_entity2 = _DL3._normalize_entity_type(_saved_entity2)
                except Exception:
                    pass

                # Derive location/area_size metadata values from the user's answer
                _location_val = None
                _area_size_val = None
                _raw_lower = (_raw or "").lower()
                if "กรุงเทพ" in _raw_lower:
                    _location_val = "กรุงเทพฯ"
                elif "ต่างจังหวัด" in _raw_lower or "ต่างหวัด" in _raw_lower:
                    _location_val = "ต่างจังหวัด"
                elif "ปริมณฑล" in _raw_lower:
                    # 'ปริมณฑล' alone: resolve by checking what the slot options say.
                    # If the slot options (built from real operation_topic) contain 'กรุงเทพ'
                    # alongside 'ปริมณฑล' → this license groups them → map to 'กรุงเทพฯ'.
                    # Otherwise → ปริมณฑล is a separate province group → treat as ต่างจังหวัด.
                    _loc_slot_opts = pending.get("options") if isinstance(pending, dict) else []
                    _opts_str = " ".join(str(o) for o in (_loc_slot_opts or []))
                    if "กรุงเทพ" in _opts_str and "ปริมณฑล" in _opts_str:
                        _location_val = "กรุงเทพฯ"  # e.g. option was 'กรุงเทพฯ และปริมณฑล'
                    else:
                        _location_val = "ต่างจังหวัด"  # default: ปริมณฑล = ต่างจังหวัด
                if "มากกว่า 200" in _raw_lower or "เกิน 200" in _raw_lower or "> 200" in _raw_lower:
                    _area_size_val = "มากกว่า 200 ตารางเมตร"
                elif "น้อยกว่า 200" in _raw_lower or "ไม่เกิน 200" in _raw_lower or "< 200" in _raw_lower:
                    _area_size_val = "ไม่เกิน 200 ตารางเมตร"

                # Build enriched query from all collected slots
                _all_slots_area = state.get_collected_slots() or {}
                _area_q_parts = [v for k, v in _all_slots_area.items()
                                 if k not in ("operation_group",) and v]
                enriched_q_area = " ".join(filter(None, [base_q] + _area_q_parts + [_raw])).strip()

                # Build combined metadata filter from all known filterable slots.
                # When area_size is filled, also include location from collected_slots (and vice versa)
                # so the filter targets the exact user path (e.g. กรุงเทพฯ + >200sqm), preventing
                # ต่างจังหวัด docs from leaking in when only area_size filter is applied.
                _filter_parts_area: List[Dict] = []

                # Location: current answer takes priority, fallback to previously collected slot
                _loc_to_use = _location_val
                if not _loc_to_use:
                    _collected_loc = (_all_slots_area.get("location") or "").lower()
                    if "กรุงเทพ" in _collected_loc:
                        _loc_to_use = "กรุงเทพฯ"
                    elif "ต่างจังหวัด" in _collected_loc or "ต่างหวัด" in _collected_loc:
                        _loc_to_use = "ต่างจังหวัด"

                if _loc_to_use:
                    _filter_parts_area.append({"location": _loc_to_use})
                if _area_size_val:
                    _filter_parts_area.append({"area_size": _area_size_val})

                _meta_filter_area: dict | None = None
                if not _filter_parts_area and _saved_entity2:
                    _meta_filter_area = {"entity_type_normalized": _saved_entity2}
                elif len(_filter_parts_area) == 1:
                    _meta_filter_area = _filter_parts_area[0]
                elif len(_filter_parts_area) > 1:
                    _meta_filter_area = {"$and": _filter_parts_area}

                try:
                    _k_for_area = int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 6))
                    state.current_docs = self._practical._retrieve_docs(
                        enriched_q_area,
                        metadata_filter=_meta_filter_area,
                        max_docs=_k_for_area,
                    )
                    state.last_retrieval_query = enriched_q_area
                    _LOG.info(
                        "[Supervisor] area-slot retrieval v3: key=%r val=%r location=%r area=%r entity=%r filter=%r docs=%d",
                        _pending_key, _raw, _location_val, _area_size_val, _saved_entity2,
                        _meta_filter_area, len(state.current_docs),
                    )
                except Exception as e:
                    _LOG.warning("[Supervisor] area-slot retrieval failed: %s", e)

            elif _entity_val:
                # Re-retrieve with entity-type filter for better doc coverage.
                # Include _raw in the query when it differs from _entity_val (e.g. registration_type
                # "ห้างหุ้นส่วนจำกัด/สามัญ" → entity "นิติบุคคล") so vector similarity ranks
                # the specific sub-type docs higher.
                _pending_key2 = pending.get("key") if isinstance(pending, dict) else None
                if _pending_key2 and _pending_key2 != "entity_type" and _raw and _raw != _entity_val:
                    enriched_q = " ".join(filter(None, [base_q, _entity_val, _raw])).strip()
                else:
                    enriched_q = f"{base_q} {_entity_val}".strip() if base_q else _entity_val

                # Capture current license_type BEFORE retrieval overwrites docs.
                # Chroma entity filter returns docs from ALL licenses with that entity —
                # we post-filter to the original license to prevent cross-license slot discovery.
                _prior_license = None
                for _pd in (state.current_docs or []):
                    _plt = ((_pd.get("metadata") or {}).get("license_type") or "").strip()
                    if _plt:
                        _prior_license = _plt
                        break

                entity_filter = {"entity_type_normalized": _entity_val}
                try:
                    _entity_docs = self._practical._retrieve_docs(enriched_q, metadata_filter=entity_filter)
                    # Post-filter: keep only docs for the original license_type
                    if _prior_license and _entity_docs:
                        _license_filtered = [
                            d for d in _entity_docs
                            if ((_d_md := (d.get("metadata") or {})).get("license_type") or "").strip() == _prior_license
                        ]
                        if _license_filtered:
                            state.current_docs = _license_filtered
                            _LOG.info(
                                "[Supervisor] entity-enriched retrieval: entity=%r license=%r docs=%d (filtered from %d)",
                                _entity_val, _prior_license, len(_license_filtered), len(_entity_docs),
                            )
                        else:
                            # No docs match the original license — keep all (better than nothing)
                            state.current_docs = _entity_docs
                            _LOG.info(
                                "[Supervisor] entity-enriched retrieval: entity=%r docs=%d (no license match for %r, keeping all)",
                                _entity_val, len(_entity_docs), _prior_license,
                            )
                    else:
                        state.current_docs = _entity_docs
                        _LOG.info("[Supervisor] entity-enriched retrieval: entity=%r docs=%d", _entity_val, len(state.current_docs))
                    state.last_retrieval_query = enriched_q
                except Exception as e:
                    _LOG.warning("[Supervisor] entity-enriched retrieval failed: %s", e)

                # When registration_type was just filled, supplement with Chroma coll.get()
                # filtered by registration_type to get the exact docs (avoids pulling in
                # บริษัทมหาชนจำกัด or other sub-type docs via similarity bias).
                if _pending_key2 == "registration_type" and _raw and _prior_license:
                    try:
                        _coll_rt = self._get_chroma_collection()
                        if _coll_rt is not None:
                            _rt_result = _coll_rt.get(
                                where={"$and": [
                                    {"license_type": _prior_license},
                                    {"registration_type": _raw},
                                ]},
                                include=["documents", "metadatas"],
                            )
                            _doc_chars_rt = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                            _SUPERVISOR_META_WL_RT = frozenset({
                                "license_type", "operation_topic", "entity_type_normalized",
                                "registration_type", "department", "fees", "operation_duration",
                                "service_channel", "operation_steps", "identification_documents",
                                "research_reference",
                            })
                            _SUPERVISOR_FC_RT = {
                                "operation_steps": 600, "identification_documents": 1500,
                                "research_reference": 3200, "fees": 120, "service_channel": 200,
                            }
                            _rt_specific: List[Dict] = []
                            for _fc_rt, _fm_rt in zip(_rt_result.get("documents") or [], _rt_result.get("metadatas") or []):
                                _sm_rt: Dict = {}
                                for _k_rt, _v_rt in (_fm_rt or {}).items():
                                    if _k_rt not in _SUPERVISOR_META_WL_RT or _v_rt in (None, "", "nan", "None"):
                                        continue
                                    _vs_rt = str(_v_rt)
                                    _cap_rt = _SUPERVISOR_FC_RT.get(_k_rt)
                                    _sm_rt[_k_rt] = _vs_rt[:_cap_rt] if _cap_rt and len(_vs_rt) > _cap_rt else _vs_rt
                                _rt_specific.append({"content": (_fc_rt or "")[:_doc_chars_rt], "metadata": _sm_rt})
                            if _rt_specific:
                                # Merge: specific registration_type docs first, then remaining entity docs (dedup by content prefix)
                                _seen_rt: set = {d["content"][:60] for d in _rt_specific}
                                _rt_general = [d for d in (state.current_docs or []) if d.get("content", "")[:60] not in _seen_rt]
                                state.current_docs = _rt_specific + _rt_general
                                _LOG.info(
                                    "[Supervisor] registration_type targeted retrieval: %r → %d specific + %d general docs",
                                    _raw, len(_rt_specific), len(_rt_general),
                                )
                    except Exception as _e_rt:
                        _LOG.warning("[Supervisor] registration_type targeted retrieval failed: %s", _e_rt)

                # After entity_type collected → re-discover remaining slots
                # (area_type may now be needed; skip entity_type since it's already collected)
                # SKIP rebuild if pending_key is registration_type — operation_group is already
                # in the existing topic_slot_queue and re-building would overwrite it with a
                # fresh (potentially stale/truncated) LLM call result.
                _skip_queue_rebuild = _pending_key2 == "registration_type" and bool(
                    (state.context or {}).get("topic_slot_queue")
                )
                if _skip_queue_rebuild:
                    _LOG.info("[Supervisor] skip slot queue rebuild — registration_type answered, operation_group already queued")
                else:
                    _license_type_for_area = None
                    for d in (state.current_docs or []):
                        lt = ((d.get("metadata") or {}).get("license_type") or "").strip()
                        if lt:
                            _license_type_for_area = lt
                            break
                    if _license_type_for_area:
                        _all_slots = self._discover_slots_for_license(_license_type_for_area)
                        # Remove slots already collected (check both collected_slots and context["slots"])
                        try:
                            _cs_area = state.get_collected_slots() or {}
                            _ctx_slots_area = (state.context or {}).get("slots") or {}
                            _collected_keys = set(_cs_area.keys()) | set(_ctx_slots_area.keys())
                        except Exception:
                            _collected_keys = set()
                        _remaining = [s for s in _all_slots if s["key"] not in _collected_keys]
                        # Also remove entity_type itself (just answered)
                        _remaining = [s for s in _remaining if s["key"] != "entity_type"]

                        # ✅ FIX P-A: filter registration_type options to only show sub-types
                        # that match the entity_type the user just selected (read from real Chroma data).
                        # e.g. entity=นิติบุคคล → [บริษัทจำกัด, ห้างหุ้นส่วนจำกัด, ห้างหุ้นส่วนสามัญ]
                        # instead of showing all entity types mixed together.
                        if _entity_val:
                            _filtered_remaining: List[Dict] = []
                            for _slot in _remaining:
                                if _slot.get("key") == "registration_type":
                                    _entity_rt_opts = self._get_registration_types_for_entity(
                                        _license_type_for_area, _entity_val
                                    )
                                    if len(_entity_rt_opts) >= 2:
                                        # Replace with entity-filtered options
                                        _filtered_remaining.append({
                                            "key": "registration_type",
                                            "options": _entity_rt_opts,
                                            "question": _slot.get("question", self._Q_REGISTRATION_TYPE),
                                        })
                                        _LOG.info(
                                            "[Supervisor] registration_type filtered for entity=%r → %s",
                                            _entity_val, _entity_rt_opts,
                                        )
                                    else:
                                        # Only 0-1 options for this entity → skip asking (already unambiguous)
                                        _LOG.info(
                                            "[Supervisor] registration_type has ≤1 option for entity=%r → skip slot",
                                            _entity_val,
                                        )
                                else:
                                    _filtered_remaining.append(_slot)
                            _remaining = _filtered_remaining

                        # Level 3: append operation_group slot into the queue
                        # so it is asked in practical BEFORE switching to academic
                        _license_type_for_og = _license_type_for_area
                        if _license_type_for_og and _entity_val:
                            _op_result = self._get_operation_groups_for_entity(_license_type_for_og, _entity_val)
                            _op_groups, _raw_op_map = _op_result if isinstance(_op_result, tuple) else (_op_result, {})
                            # Only add to queue when ≥2 distinct options — single option means nothing to choose
                            if len(_op_groups) >= 2:
                                _query_for_infer_et = (state.context or {}).get("last_user_legal_query") or user_input
                                _inferred_et = self._infer_operation_group_from_query(_query_for_infer_et, _op_groups)
                                if _inferred_et:
                                    state.save_collected_slot("operation_group", _inferred_et)
                                    state.context.setdefault("slots", {})["confirmed_operation"] = _inferred_et
                                    _LOG.info("[Supervisor] operation_group auto-inferred=%r (entity path) — skipping ask", _inferred_et)
                                else:
                                    _remaining.append({
                                        "key": "operation_group",
                                        "options": _op_groups,
                                        "question": self._Q_OPERATION_GROUP,
                                        "raw_op_map": _raw_op_map,
                                    })
                                    _LOG.info("[Supervisor] operation_group appended to slot_queue → %s", _op_groups)
                            elif _op_groups:
                                _LOG.info("[Supervisor] operation_group only 1 option (%s) — skip asking", _op_groups)

                        if _remaining:
                            # Step 4: cap at 2 to prevent question loops
                            if len(_remaining) > 2:
                                _remaining = _remaining[:2]
                                _LOG.info("[Supervisor] slot_queue after entity capped to 2: %s", [s["key"] for s in _remaining])
                            state.context["topic_slot_queue"] = _remaining
                            _LOG.info("[Supervisor] topic_slot_queue after entity → remaining: %s",
                                      [s["key"] for s in _remaining])
                        else:
                            state.context.pop("topic_slot_queue", None)
                # Clear legacy keys
                state.context.pop("topic_area_types", None)
                state.context.pop("topic_registration_types", None)

                # Level 3 now handled inside slot_queue above — clear old separate key
                state.context.pop("topic_operation_groups", None)

        # BUG-E fix: record that a pending slot was successfully consumed so last_action
        # reflects the actual operation (not a stale value from a previous invalid reply).
        state.last_action = "pending_slot_filled"

        pid = normalize_persona_id(state.persona_id)
        if pid == "academic":
            st2, reply = self._academic.handle(state, mapped, _internal=True)
            st2, reply = self._post_route_academic_auto_return(st2, reply)
            reply = self._normalize_male(reply)
            self._add_assistant(st2, reply)
            return st2, reply

        st2, reply = self._practical.handle(state, mapped, _internal=True)
        reply = self._normalize_male(reply)
        self._add_assistant(st2, reply)
        return st2, reply

    # --------------------------
    # Auto-return followup
    # --------------------------
    def _build_auto_return_followup(self, state: ConversationState) -> str:
        ctx = state.context or {}
        ctx.pop("auto_return_topic_context", None)
        state.context = ctx
        return ""

    def _reply_has_closing(self, reply: str) -> bool:
        """True if LLM answer already ends with a closing/farewell phrase.
        Prevents appending a duplicate follow-up when the LLM already closed politely."""
        tail = (reply or "")[-200:]
        signals = ("สงสัยเพิ่ม", "ถามเพิ่ม", "บอกผมได้เลย", "ช่วยได้เลย", "สอบถามเพิ่มเติม")
        return any(s in tail for s in signals)

    def _post_route_academic_auto_return(self, state: ConversationState, reply: str) -> Tuple[ConversationState, str]:
        ctx = state.context or {}
        auto_supervisor = bool(ctx.get("auto_return_after_academic_done", False))
        auto_academic = bool(ctx.get("auto_return_to_practical", False))

        if not (auto_supervisor or auto_academic):
            return state, reply

        if self._is_academic_intake_active(state):
            return state, reply

        origin = normalize_persona_id(ctx.get("switch_origin_persona") or "practical")
        if origin != "practical":
            origin = "practical"

        if normalize_persona_id(getattr(state, "persona_id", "") or "") != "academic":
            return state, reply

        state.persona_id = origin
        ctx["persona_id"] = origin
        ctx.pop("auto_return_after_academic_done", None)
        ctx.pop("auto_return_to_practical", None)
        ctx.pop("switch_origin_persona", None)
        state.context = ctx

        state.last_action = "auto_return_to_practical"

        # Mark that academic session is resumable (section_catalog / academic_question / docs still in context)
        state.context["academic_resume_available"] = True

        follow_up = self._build_auto_return_followup(state)
        # Only append follow-up if LLM answer doesn't already end with a similar closing phrase
        if follow_up and not self._reply_has_closing(reply):
            combined = reply.rstrip() + "\n\n─────────────────\n" + follow_up
            return state, combined

        return state, reply

    # Greeting/Menu (unchanged from your current file)
    _MENU_SIZE = 5
    _POOL_MAX = 80
    _MENU_CANDIDATE_MAX = 30
    _LLM_PICK_MIN_CONF = 0.55

    # Loaded from conf.MENU_REQUIRE_KEYWORDS — add new domain keywords there when dataset expands
    # NOTE: org-name fragments (สรรพากร, กรม, สำนักงาน) intentionally excluded —
    # they are caught by _looks_orgish() and must NOT grant menu_worthy status.
    _MENU_REQUIRE_KEYWORDS: tuple = tuple(getattr(conf, "MENU_REQUIRE_KEYWORDS", (
        "ใบอนุญาต", "อนุญาต", "ขั้นตอน", "เอกสาร", "ค่าธรรมเนียม", "ระยะเวลา", "ช่องทาง",
        "ภาษี", "vat", "ภพ", "จดทะเบียน", "ทะเบียนพาณิชย์", "dbd",
        "ประกันสังคม", "กองทุน", "สุขาภิบาล", "เปิดร้าน", "ยื่นคำขอ", "คำขอ",
        "ใบกำกับภาษี", "ใบเสร็จ", "แบบฟอร์ม", "ฟอร์ม",
    )))
    _MENU_DETAILISH_PATTERNS = (
        r"^กรณี", r"^ถ้า", r"^สำหรับ", r"^เมื่อ", r"^หาก", r"^ในกรณี", r"^ข้อยกเว้น",
        r"^หมายเหตุ", r"^เพิ่มเติม", r"^ตัวอย่าง", r"^คำแนะนำ",
        r"ยกเว้น", r"ที่ได้รับ", r"กรณีพิเศษ", r"เฉพาะกรณี",
    )
    _MENU_REJECT_PATTERNS = (
        r"^\W+$",
        r"^https?://",
        r"@",
        r"\b(line|facebook|fb|ig|email)\b",
        r"\b\d{8,}\b",
    )

    # Loaded from conf so the list can be updated without touching this file
    _MENU_FALLBACK_TOPICS: List[str] = list(getattr(conf, "MENU_FALLBACK_TOPICS", [
        "ขอใบอนุญาตเปิดร้านอาหาร",
        "สุขาภิบาลอาหาร / อาหารสะอาด",
        "ภาษี VAT / ขอ ภพ.20",
        "จดทะเบียนพาณิชย์ / DBD",
        "เอกสารที่ต้องใช้ / เช็คลิสต์",
        "ค่าธรรมเนียม",
        "ระยะเวลาดำเนินการ",
        "ช่องทางยื่นคำขอ / หน่วยงาน",
        "ประกันสังคม (ขึ้นทะเบียนนายจ้าง)",
        "กองทุนเงินทดแทน",
    ]))

    def _format_numbered_options(self, options: List[str], max_items: int = 9) -> str:
        opts = [str(x).strip() for x in (options or []) if str(x).strip()]
        opts = opts[:max_items]
        return "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(opts)])

    def _sanitize_topic_label(self, s: str) -> str:
        raw = (s or "")
        t = raw.strip()
        if not t:
            return ""
        if "\n" in raw or "\r" in raw:
            return ""
        t = re.sub(r"\s+", " ", t).strip()

        if t in {"-", "—", "–", "N/A", "n/a", "NA", "na"}:
            return ""
        if len(t) < 3:
            return ""
        if len(t) > 64:
            return ""

        t = re.sub(r"^\s*หน่วยงาน\s*[:：]\s*", "", t).strip()
        # Collapse consecutive Thai/ASCII word repetitions (e.g. "ทะเบียนทะเบียน" → "ทะเบียน")
        t = re.sub(r"([ก-๙a-zA-Z]{2,})\1+", r"\1", t)
        return t

    # NOTE: no \b — Thai has no spaces between words so \b never fires mid-string.
    # Prefix match alone is sufficient: any label starting with these IS an org name.
    _ORGISH_RE = re.compile(
        r"^(กรม|สำนักงาน|สำนัก|เทศบาล|อบต\.?|อบจ\.?|สำนักงานเขต"
        r"|กรุงเทพมหานคร|กทม\.?)"
    )

    def _looks_orgish(self, label: str) -> bool:
        l = (label or "").strip()
        if not l:
            return False
        return bool(self._ORGISH_RE.search(l))

    def _is_detailish_label(self, label: str) -> bool:
        l = (label or "").strip()
        if not l:
            return True
        low = self._normalize_for_intent(l)
        for pat in self._MENU_DETAILISH_PATTERNS:
            if re.search(pat, low, flags=re.IGNORECASE):
                return True
        return False

    def _passes_reject_patterns(self, label: str) -> bool:
        low = self._normalize_for_intent(label)
        for pat in self._MENU_REJECT_PATTERNS:
            if re.search(pat, low, flags=re.IGNORECASE):
                return False
        return True

    def _menu_keyword_score(self, label: str) -> int:
        low = self._normalize_for_intent(label)
        score = 0
        for kw in self._MENU_REQUIRE_KEYWORDS:
            if kw and kw.lower() in low:
                score += 1
        return score

    def _is_menu_worthy(self, label: str) -> bool:
        l = self._sanitize_topic_label(label)
        if not l:
            return False
        if not self._passes_reject_patterns(l):
            return False
        if self._is_detailish_label(l):
            return False
        if self._looks_orgish(l):
            return False
        if self._menu_keyword_score(l) >= 1:
            return True
        low = self._normalize_for_intent(l)
        if re.search(r"(ขั้นตอน|เอกสาร|ค่าธรรมเนียม|ระยะเวลา|ช่องทาง|ใบอนุญาต|ภาษี|จดทะเบียน)", low):
            return True
        return False

    def _topic_kind_weight(self, label: str, source_key: str) -> int:
        k = (source_key or "").strip().lower()
        l = (label or "").strip()

        if k in {"license_type", "ใบอนุญาต"}:
            return 5
        if k in {
            "operation_topic",
            "หัวข้อการดำเนินการย่อย",
            "operation_subtopic",
            "sub_operation_topic",
            "subtopic",
        }:
            return 4
        if k in {
            "operation_by_department",
            "operation_action",
            "การดำเนินการ ตามหน่วยงาน",
            "การดำเนินการตามหน่วยงาน",
            "action_by_department",
            "operation_process",
        }:
            return 4

        if k in {"department", "หน่วยงาน"}:
            return 1 if self._ORGISH_RE.search(l) else 2

        return 2

    def _get_banned_topic_labels(self) -> List[str]:
        banned = set()
        for x in getattr(self, "_STOP_LABELS", set()) or set():
            if x:
                banned.add(str(x).strip())

        extra = {
            "หัวข้อหลัก", "หัวข้อ", "หัวข้อย่อย", "หมวด", "หมวดหมู่",
            "อื่นๆ", "อื่น ๆ", "ไม่ระบุ", "ทั่วไป",
        }
        for x in extra:
            banned.add(x)

        return sorted([b for b in banned if b])

    def _collect_topic_freq_from_docs(self, docs: List[Any]) -> Dict[str, int]:
        freq: Dict[str, int] = {}

        def _add(v: Any, source_key: str) -> None:
            s = self._sanitize_topic_label(str(v) if v is not None else "")
            if not s:
                return

            if self._is_detailish_label(s):
                return

            if not self._is_menu_worthy(s):
                if self._looks_orgish(s):
                    return
                if len(s) <= 10:
                    w = 1
                else:
                    return
            else:
                w = self._topic_kind_weight(s, source_key=source_key)

            freq[s] = freq.get(s, 0) + int(w)

        LICENSE_KEYS = ["license_type", "ใบอนุญาต"]
        OP_BY_DEPT_KEYS = ["operation_by_department", "operation_action", "action_by_department", "operation_process", "การดำเนินการ ตามหน่วยงาน", "การดำเนินการตามหน่วยงาน"]
        OP_SUB_KEYS = ["operation_subtopic", "sub_operation_topic", "subtopic", "หัวข้อการดำเนินการย่อย"]
        OP_TOPIC_KEYS = ["operation_topic"]
        DEPT_KEYS = ["department", "หน่วยงาน"]

        for d in (docs or []):
            md = getattr(d, "metadata", {}) or {}

            for k in LICENSE_KEYS:
                _add(md.get(k), source_key=k)

            for k in OP_BY_DEPT_KEYS:
                _add(md.get(k), source_key=k)

            for k in OP_SUB_KEYS:
                _add(md.get(k), source_key=k)

            for k in OP_TOPIC_KEYS:
                _add(md.get(k), source_key=k)

            for k in DEPT_KEYS:
                _add(md.get(k), source_key=k)

        return freq

    def _build_topic_pool_from_corpus(self, state: ConversationState) -> List[Tuple[str, int]]:
        # Loaded from conf.TOPIC_POOL_QUERIES — add new domain queries there when dataset expands
        queries: List[str] = list(getattr(conf, "TOPIC_POOL_QUERIES", [
            "ใบอนุญาต เปิดร้านอาหาร เทศบาล สำนักงานเขต สุขาภิบาลอาหาร",
            "ภาษี VAT ภพ.20 ใบกำกับภาษี กรมสรรพากร จด VAT",
            "จดทะเบียนพาณิชย์ นิติบุคคล DBD กรมพัฒนาธุรกิจการค้า หนังสือรับรอง",
            "ประกันสังคม ขึ้นทะเบียนนายจ้าง ลูกจ้าง กองทุนเงินทดแทน",
            "ขั้นตอนการดำเนินการ เอกสารที่ต้องใช้ ค่าธรรมเนียม ระยะเวลา ช่องทางยื่นคำขอ",
        ]))

        merged: Dict[str, int] = {}
        for q in queries:
            try:
                docs = self.retriever.invoke(q) or []
            except Exception:
                docs = []
            freq = self._collect_topic_freq_from_docs(docs)
            for k, v in freq.items():
                merged[k] = merged.get(k, 0) + int(v)

        items = sorted(merged.items(), key=lambda x: (-x[1], x[0]))
        pool = items[: self._POOL_MAX]

        if len(pool) < 12:
            existing = {k for k, _ in pool}
            for i, t in enumerate(self._MENU_FALLBACK_TOPICS):
                t2 = self._sanitize_topic_label(t)
                if t2 and t2 not in existing and self._is_menu_worthy(t2):
                    pool.append((t2, 10 - (i // 2)))

        state.context = state.context or {}
        state.context["topic_pool"] = pool
        return pool

    def _get_topic_pool(self, state: ConversationState) -> List[Tuple[str, int]]:
        state.context = state.context or {}
        cached = state.context.get("topic_pool")
        if isinstance(cached, list) and cached and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in cached):
            out: List[Tuple[str, int]] = []
            for t, w in cached:
                try:
                    ts = str(t)
                    if not self._is_menu_worthy(ts):
                        continue
                    out.append((ts, int(w)))
                except Exception:
                    continue
            if out:
                return out
        return self._build_topic_pool_from_corpus(state)

    def _get_last_topic_hint(self, state: ConversationState) -> str:
        ctx = state.context or {}

        last_legal = str(ctx.get("last_user_legal_query") or "").strip()
        if last_legal:
            return last_legal[:80]

        last = str(ctx.get("last_topic") or "").strip()
        if last:
            return last

        last_q = str(getattr(state, "last_retrieval_query", "") or "").strip()
        return last_q[:60] if last_q else ""

    def _weighted_sample_no_replace(self, pool: List[Tuple[str, int]], k: int, rng: random.Random) -> List[str]:
        if not pool or k <= 0:
            return []

        topics = [t for t, _ in pool]
        weights = [max(1, int(w)) for _, w in pool]

        max_w = max(weights) if weights else 1
        weights = [max(1, int((w / max_w) * 7) + 1) for w in weights]

        chosen: List[str] = []
        local_topics = topics[:]
        local_weights = weights[:]

        while local_topics and len(chosen) < k:
            pick = rng.choices(local_topics, weights=local_weights, k=1)[0]
            chosen.append(pick)
            idx = local_topics.index(pick)
            local_topics.pop(idx)
            local_weights.pop(idx)

        return chosen

    def _related_topics_from_last(self, state: ConversationState, need: int) -> List[str]:
        if need <= 0:
            return []
        last_q = self._get_last_topic_hint(state).strip()
        if not last_q:
            return []
        try:
            docs = self.retriever.invoke(last_q) or []
        except Exception:
            docs = []
        freq = self._collect_topic_freq_from_docs(docs[:16])
        items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        out: List[str] = []
        for k, _ in items:
            if k and k not in out:
                out.append(k)
            if len(out) >= need:
                break
        return out

    def _dedupe_semantic_loose(self, items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in items or []:
            s = self._sanitize_topic_label(x)
            if not s:
                continue
            if not self._is_menu_worthy(s):
                continue

            key = self._normalize_for_intent(s)
            key = re.sub(r"(การ|การทำ|การขอ|ขอ|ยื่น|ขึ้นทะเบียน|จดทะเบียน|ทะเบียน|ใบอนุญาต)\s*", "", key).strip()
            key = re.sub(r"\s+", " ", key)
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def _llm_pick_menu_topics(self, state: ConversationState, last_hint: str, candidates: List[str], k: int) -> List[str]:
        if not self.llm_topic_picker_call:
            return []

        banned = self._get_banned_topic_labels()
        cand = [c for c in self._dedupe_semantic_loose(candidates) if c][: self._MENU_CANDIDATE_MAX]

        if len(cand) <= k:
            return cand[:k]

        res: Dict[str, Any] = {}
        try:
            res = self.llm_topic_picker_call(last_hint or "", cand, int(k), banned) or {}
        except Exception:
            res = {}

        try:
            confv = float(res.get("confidence", 0.0) or 0.0)
        except Exception:
            confv = 0.0

        topics = res.get("topics", [])
        if not isinstance(topics, list):
            topics = []

        picked: List[str] = []
        banned_set = set([b.strip().lower() for b in banned if b and b.strip()])
        for x in topics:
            s = self._sanitize_topic_label(str(x))
            if not s:
                continue
            if s.strip().lower() in banned_set:
                continue
            if not self._is_menu_worthy(s):
                continue
            picked.append(s)

        picked = self._dedupe_semantic_loose(picked)

        non_org = [t for t in picked if not self._looks_orgish(t)]
        org = [t for t in picked if self._looks_orgish(t)]
        final = non_org[:k]
        if len(final) < k:
            final.extend([t for t in org if t not in final][: (k - len(final))])

        if confv < self._LLM_PICK_MIN_CONF:
            return []
        if len(final) < max(3, min(k, 4)):
            return []

        return final[:k]

    def _compose_menu_topics(self, state: ConversationState, size: int) -> List[str]:
        pool = self._get_topic_pool(state)
        size = max(1, int(size))

        rng = self._get_rng(state)
        last_hint = self._get_last_topic_hint(state).strip()

        related_target = 2 if size >= 5 else 1
        related = self._related_topics_from_last(state, need=related_target)

        related = [self._sanitize_topic_label(x) for x in related]
        related = [x for x in related if x and self._is_menu_worthy(x)]
        related = self._dedupe_semantic_loose(related)

        related_set = set(related)
        pool_fresh = [(t, w) for (t, w) in pool if t not in related_set]
        fresh_need = max(0, size - len(related))
        fresh = self._weighted_sample_no_replace(pool_fresh, k=max(fresh_need, size), rng=rng)

        fresh = [self._sanitize_topic_label(x) for x in fresh]
        fresh = [x for x in fresh if x and x not in related_set and self._is_menu_worthy(x)]
        fresh = self._dedupe_semantic_loose(fresh)

        candidates: List[str] = []
        for t in related:
            if t and t not in candidates:
                candidates.append(t)
        for t in fresh:
            if t and t not in candidates:
                candidates.append(t)
            if len(candidates) >= self._MENU_CANDIDATE_MAX:
                break

        if len(candidates) < min(12, self._MENU_CANDIDATE_MAX):
            for t, _ in pool:
                t2 = self._sanitize_topic_label(t)
                if t2 and t2 not in candidates and self._is_menu_worthy(t2):
                    candidates.append(t2)
                if len(candidates) >= self._MENU_CANDIDATE_MAX:
                    break

        picked = self._llm_pick_menu_topics(state, last_hint=last_hint, candidates=candidates, k=size)

        if not picked:
            combined: List[str] = []
            for t in related + fresh:
                t2 = self._sanitize_topic_label(t)
                if t2 and t2 not in combined and self._is_menu_worthy(t2):
                    combined.append(t2)
                if len(combined) >= size:
                    break

            if len(combined) < size:
                for t, _ in pool:
                    t2 = self._sanitize_topic_label(t)
                    if t2 and t2 not in combined and self._is_menu_worthy(t2):
                        combined.append(t2)
                    if len(combined) >= size:
                        break

            picked = combined[:size]

        last_menu = (state.context or {}).get("last_menu_topics")
        if isinstance(last_menu, list) and len(last_menu) >= 3:
            overlap = len(set(last_menu).intersection(set(picked)))
            if overlap >= 4 and len(pool_fresh) >= size:
                fresh2 = self._weighted_sample_no_replace(pool_fresh, k=self._MENU_CANDIDATE_MAX, rng=rng)
                fresh2 = [self._sanitize_topic_label(x) for x in fresh2]
                fresh2 = [x for x in fresh2 if x and self._is_menu_worthy(x)]
                fresh2 = self._dedupe_semantic_loose(fresh2)

                candidates2: List[str] = []
                for t in related:
                    if t and t not in candidates2:
                        candidates2.append(t)
                for t in fresh2:
                    if t and t not in candidates2:
                        candidates2.append(t)
                    if len(candidates2) >= self._MENU_CANDIDATE_MAX:
                        break

                picked2 = self._llm_pick_menu_topics(state, last_hint=last_hint, candidates=candidates2, k=size)
                if picked2 and len(set(picked2).intersection(set(last_menu))) < overlap:
                    picked = picked2

        picked = self._dedupe_semantic_loose(picked)

        if len(picked) < size:
            for t in self._MENU_FALLBACK_TOPICS:
                t2 = self._sanitize_topic_label(t)
                if t2 and t2 not in picked and self._is_menu_worthy(t2):
                    picked.append(t2)
                if len(picked) >= size:
                    break

        if len(picked) < size:
            for t, _ in pool:
                t2 = self._sanitize_topic_label(t)
                if t2 and t2 not in picked and self._is_menu_worthy(t2):
                    picked.append(t2)
                if len(picked) >= size:
                    break

        return picked[:size]

    def _get_prefix_llm(self, kind: str, state: ConversationState, include_intro: bool) -> str:
        pid = normalize_persona_id(state.persona_id)
        last_hint = self._get_last_topic_hint(state)

        res: Dict[str, Any] = {}
        try:
            res = self.llm_greet_prefix_call(kind, pid, last_hint, bool(include_intro)) or {}
        except Exception:
            res = {}

        prefix = ""
        if isinstance(res, dict):
            prefix = str(res.get("prefix") or "").strip()

        if not prefix:
            if include_intro:
                prefix = "👋 สวัสดีครับ! ผม 'น้องสุดยอด Consult Restbiz' ยินดีให้บริการครับ!"
            else:
                prefix = "ตอนนี้อยากให้ช่วยเรื่องไหนครับ"

        prefix = re.sub(r"\s+", " ", prefix).strip()
        return self._normalize_male(prefix)

    def _generate_topic_descriptions(self, topics: List[str]) -> List[str]:
        """Retrieve real docs for each topic, then summarize into a 1-sentence description."""
        if not topics:
            return []

        # Step 1: Retrieve real documents for each topic from vector store
        topic_contexts: List[str] = []
        for t in topics:
            try:
                docs = self.retriever.invoke(t)[:2]
                content = "\n".join(d.page_content[:150] for d in docs if d.page_content)
                _LOG.info(f"[topic_desc] '{t}' → {len(docs)} docs retrieved")
                topic_contexts.append(content)
            except Exception:
                _LOG.warning(f"[topic_desc] retrieval failed for '{t}'", exc_info=True)
                topic_contexts.append("")

        # Step 2: Build context block from real docs
        context_block = ""
        for i, (t, ctx) in enumerate(zip(topics, topic_contexts)):
            if ctx:
                context_block += f"=== หัวข้อ {i+1}: {t} ===\n{ctx}\n\n"

        if not context_block.strip():
            return [f"ผมจะแนะนำ{t} ตั้งแต่ต้นจนจบ พร้อมเอกสารที่ต้องใช้ ให้คุณทำตามได้ง่ายที่สุดครับ" for t in topics]

        # Step 3: LLM summarizes from real doc content
        topic_model = getattr(conf, "OPENROUTER_MODEL_TOPIC_PICKER", getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL))
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=topic_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.3,
            max_tokens=600,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        prompt = build_topic_desc_prompt(topics, context_block)
        try:
            resp = llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/topic_desc")
            text = self._strip_code_fences(extract_llm_text(resp).strip())
            obj = json.loads(text)
            descs = obj.get("descriptions") if isinstance(obj, dict) else obj
            if isinstance(descs, list) and descs:
                return [str(d).strip() for d in descs[:len(topics)]]
        except Exception:
            _LOG.warning("_generate_topic_descriptions failed, using fallback", exc_info=True)
        return [f"ผมจะแนะนำ{t} ตั้งแต่ต้นจนจบ พร้อมเอกสารที่ต้องใช้ ให้คุณทำตามได้ง่ายที่สุดครับ" for t in topics]

    # Fixed first-greeting menu — shown on every new session (no LLM call, zero token cost)
    _FIXED_GREETING_MENU: List[Tuple[str, str]] = [
        (
            "ขอใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร",
            "ผมจะแนะนำวิธีการจดใบอนุญาติจัดตั้งสถานที่จำหน่ายอาหารแบบ Step by Step "
            "พร้อมเอกสารที่ต้องใช้ให้คุณนำไปใช้ได้ง่ายที่สุดครับ",
        ),
        (
            "วิธีการลงทะเบียน QR-Payment API",
            "ผมจะแนะนำวิธีการลงทะเบียน QR-Payment API สำหรับใช้งานกับระบบของเรา"
            "ให้คุณทำตามได้แบบไม่สับสนครับ",
        ),
    ]

    def _render_greeting_with_menu(self, state: ConversationState, kind: str, menu_topics: List[str], include_intro: bool) -> str:
        if include_intro:
            intro = (
                "👋 สวัสดีครับ! ผม \"น้องสุดยอด Consult Restbiz\"\n"
                "ยินดีให้บริการครับ!\n"
                "น้องสุดยอดพร้อมเป็นที่ปรึกษาเรื่องการจัดการร้านอาหาร การจดเอกสารขอใบอนุญาติ ที่จำเป็นสำหรับร้านอาหาร\n"
                "💡 อยากให้น้องสุดยอดช่วยอะไรครับ?"
            )
            _EMOJI_NUM = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
            topic_lines = []
            for i, (t, desc) in enumerate(self._FIXED_GREETING_MENU):
                num = _EMOJI_NUM[i] if i < len(_EMOJI_NUM) else f"{i+1}."
                topic_lines.append(f"{num} {t} - {desc}")
            footer = "พิมพ์ตัวเลข หรือบอกผมได้เลยว่าต้องการข้อมูลใบอนุญาติ หรือ ทริคการจัดการร้านอาหารด้านใดสำหรับร้านของคุณครับ 😊"
            msg = intro + "\n" + "\n".join(topic_lines) + "\n" + footer
            return self._normalize_male(msg)
        prefix = self._get_prefix_llm(kind, state, include_intro=False)
        menu = self._format_numbered_options(menu_topics, max_items=9)
        msg = (prefix.rstrip() + "\n" + menu).strip()
        return self._normalize_male(msg)

    def _handle_typo_prompt(self, state: ConversationState, user_input: str, suggested: str = "") -> Tuple[ConversationState, str]:
        """
        Called when input is detected as garbled / typo.
        - Does NOT clear pending_slot or any context (user should be able to retype and continue)
        - Does NOT show the topic menu (that would confuse the flow)
        - Returns a polite ask-to-retype message
        - If LLM provided a 'suggested' guess, include it as a gentle prompt
        """
        state.context = state.context or {}
        if suggested:
            msg = self._normalize_male(
                f"ดูเหมือนพิมพ์ผิดนะครับ 😊 "
                f"หมายถึง \"{suggested}\" ใช่ไหมครับ? "
                f"ถ้าใช่ลองพิมพ์ใหม่ได้เลยครับ"
            )
        else:
            msg = self._normalize_male(
                "ดูเหมือนพิมพ์ผิดนะครับ 😊 ลองพิมพ์ใหม่ได้เลยครับ"
            )
        self._add_assistant(state, msg)
        state.last_action = "typo_prompt"
        _LOG.info("[Supervisor] typo_prompt input=%r suggested=%r", (user_input or "")[:40], suggested or "")
        return state, msg

    def _handle_greeting(self, state: ConversationState, user_input: str, show_menu: bool = True) -> Tuple[ConversationState, str]:
        state.context = state.context or {}
        raw = (user_input or "").strip()
        t = self._normalize_for_intent(raw)

        # Guardrail: greeting mixed with off-topic content → deflect, not topic menu
        # e.g. "สวัสดี ไปเที่ยวระยองกันมั้ย", "สวัสดี คุณจะไปสวัสดีใคร"
        # Strip greeting prefix → if remainder is non-empty and off-topic → deflect
        if raw:
            _greet_prefix_re = re.compile(
                r"^(สวัสดี\w*|หวัดดี|ดีครับ|ดีค่ะ|ดีจ้า|ดีนะ|hi\b|hello\b)\s*[,!]?\s*",
                re.IGNORECASE,
            )
            _remainder = _greet_prefix_re.sub("", raw).strip()
            if (
                _remainder
                and not self._LEGAL_SIGNAL_RE.search(_remainder)
                and not self._looks_like_legal_question(_remainder)
                and not self._LIKELY_SELECTION_RE.match(_remainder)
            ):
                _LOG.info("[Supervisor] greeting+offtopic → deflect remainder=%r", _remainder[:60])
                return self._handle_deflect(state, raw)

        kind = "greet"
        if not raw:
            kind = "blank"
        elif self._THANKS_RE.search(t):
            kind = "thanks"
        elif self._SMALLTALK_RE.search(raw) or self._TH_LAUGH_5_RE.match(raw):
            kind = "smalltalk"

        # Short reply path: no topic list, no pending_slot update
        if not show_menu:
            if kind == "thanks":
                msg = self._normalize_male("ยินดีครับ 😊 มีอะไรอยากถามเพิ่มเติมไหมครับ?")
            elif kind == "smalltalk":
                msg = self._normalize_male("ครับ 😊 ถ้ามีคำถามเรื่องใบอนุญาต ภาษี หรือการจดทะเบียนร้านอาหาร ถามได้เลยครับ")
            else:
                msg = self._normalize_male("สวัสดีครับ 😊 มีอะไรให้ช่วยเรื่องกฎหมายร้านอาหารไหมครับ?")
            self._add_assistant(state, msg)
            state.last_action = "greeting_short"
            _LOG.info("[Supervisor] _handle_greeting show_menu=False kind=%r input=%r", kind, raw[:40])
            return state, msg

        turns = int(state.context.get("greet_menu_turns") or 0)
        include_intro = turns == 0

        if include_intro:
            # 🎯 Token: first greeting uses fixed hardcoded menu — zero LLM/retrieval cost
            topics = [t for t, _ in self._FIXED_GREETING_MENU]
            topic_descs = [d for _, d in self._FIXED_GREETING_MENU]
        else:
            # On-demand menu: use context-relevant topics from last conversation
            topics = self._compose_menu_topics(state, size=self._MENU_SIZE)
            _cached_descs = (state.context or {}).get("last_menu_topic_descs")
            _cached_topics = (state.context or {}).get("last_menu_topics")
            if _cached_descs and _cached_topics == topics:
                topic_descs = _cached_descs
            else:
                topic_descs = self._generate_topic_descriptions(topics[:2])

        state.context["pending_slot"] = {"key": "topic", "options": topics, "allow_multi": False}
        state.context["main_menu_shown"] = True
        state.context["last_menu_topics"] = topics
        state.context["last_menu_topic_descs"] = topic_descs

        msg = self._render_greeting_with_menu(state, kind=kind, menu_topics=topics, include_intro=include_intro)
        self._add_assistant(state, msg)

        state.context["greet_menu_turns"] = turns + 1
        state.last_action = "greeting_first_menu" if include_intro else "greeting_with_menu_refresh"

        return state, msg

    # Switch helpers (unchanged)
    def _mark_auto_return_if_practical_to_academic(self, state: ConversationState, target_pid: str) -> None:
        origin = normalize_persona_id(state.persona_id)
        state.context = state.context or {}
        state.context["switch_origin_persona"] = origin

        if origin == "practical" and normalize_persona_id(target_pid) == "academic":
            state.context["auto_return_after_academic_done"] = True
        else:
            state.context.pop("auto_return_after_academic_done", None)

    def _other_persona(self, pid: str) -> str:
        pid2 = normalize_persona_id(pid)
        return "academic" if pid2 == "practical" else "practical"

    def _enter_switch_confirmation(self, state: ConversationState, target_pid: str, replay_user_input: str = "") -> Tuple[ConversationState, str]:
        state.context = state.context or {}
        state.context["pending_persona"] = normalize_persona_id(target_pid)
        state.context["pending_replay_user_input"] = replay_user_input or ""
        state.context["awaiting_persona_confirmation"] = True
        state.context["confirm_tries"] = 0

        self._mark_auto_return_if_practical_to_academic(state, target_pid)

        current = normalize_persona_id(state.persona_id)
        target = normalize_persona_id(target_pid)

        msg = self._normalize_male(f"🔄 ตอนนี้อยู่โหมด {current} ครับ ต้องการเปลี่ยนไปโหมด {target} ใช่ไหมครับ")
        self._add_assistant(state, msg)
        state.last_action = "persona_switch_confirm"
        return state, msg

    def _silent_switch_to_academic(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        """Switch to academic silently (no announcement), answer, then auto-return to practical."""
        self._mark_auto_return_if_practical_to_academic(state, "academic")
        state.persona_id = "academic"
        state.context["persona_id"] = "academic"
        state.last_action = "academic_auto_switch"

        # Reset any stale academic flow so a fresh intake starts (not stuck at stage='done'
        # from a previous session, which would cause academic to return the generic fallback).
        _old_flow = (state.context or {}).get("academic_flow") or {}
        if str(_old_flow.get("stage") or "") in ("done", ""):
            state.context.pop("academic_flow", None)
            state.context.pop("section_catalog", None)
            state.context.pop("academic_question", None)
            state.context.pop("academic_resume_available", None)

        # Clear stale Practical docs so Academic always does fresh retrieval.
        # Without this, Academic reuses docs from Practical's entity_type-filtered
        # retrieval which may include sub-types (e.g. บริษัทมหาชน) unrelated to the
        # user's actual registration_type (e.g. บริษัทจำกัด).
        state.current_docs = []
        state.last_retrieval_query = ""

        # Use user_input if it looks like a legal question; else replay last known topic
        ctx = state.context or {}
        if self._looks_like_legal_question(user_input):
            replay = user_input
        else:
            replay = ctx.get("last_user_legal_query", "").strip() or ctx.get("last_topic", "").strip() or user_input
        st2, reply = self._academic.handle(state, replay, _internal=False)
        st2, reply = self._post_route_academic_auto_return(st2, reply)
        reply = self._normalize_male(reply)
        self._add_assistant(st2, reply)
        return st2, reply

    def _propose_toggle_switch(self, state: ConversationState, reason: str, replay_user_input: str = "") -> Tuple[ConversationState, str]:
        cur = normalize_persona_id(state.persona_id)
        target = self._other_persona(cur)
        state.context = state.context or {}
        state.context["switch_reason"] = reason
        return self._enter_switch_confirmation(state, target_pid=target, replay_user_input=replay_user_input)

    def _propose_switch_to_target(self, state: ConversationState, target: str, reason: str, replay_user_input: str = "") -> Tuple[ConversationState, str]:
        cur = normalize_persona_id(state.persona_id)
        target2 = normalize_persona_id(target)

        if cur == target2:
            other = self._other_persona(cur)
            msg = self._normalize_male(f"🔄 ตอนนี้อยู่โหมด {cur} อยู่แล้วครับ ต้องการสลับไปโหมด {other} ไหมครับ")
            self._add_assistant(state, msg)
            state.last_action = "persona_already_in_mode"
            state.context = state.context or {}
            state.context["pending_persona"] = other
            state.context["pending_replay_user_input"] = replay_user_input or ""
            state.context["awaiting_persona_confirmation"] = True
            state.context["confirm_tries"] = 0
            self._mark_auto_return_if_practical_to_academic(state, other)
            return state, msg

        state.context = state.context or {}
        state.context["switch_reason"] = reason
        return self._enter_switch_confirmation(state, target_pid=target2, replay_user_input=replay_user_input)

    # Persona/Profile sync
    def _sync_persona_and_profile(self, state: ConversationState) -> None:
        state.context = state.context or {}
        state.context["supervisor_owns_menu"] = True

        raw_pid = ""
        if hasattr(state, "persona_id"):
            raw_pid = str(getattr(state, "persona_id") or "")
        if not raw_pid:
            raw_pid = str(state.context.get("persona_id") or "")
        if not raw_pid:
            raw_pid = "practical"

        pid = normalize_persona_id(raw_pid)

        if hasattr(state, "persona_id"):
            state.persona_id = pid
        state.context["persona_id"] = pid

        prof = state.context.get("persona_profile")
        prof_pid = ""
        if isinstance(prof, dict):
            prof_pid = str(prof.get("persona_id") or "")

        if not isinstance(prof, dict) or normalize_persona_id(prof_pid or pid) != pid:
            profile = build_strict_profile(pid)
            state.context["persona_profile"] = profile
        else:
            profile = prof

        try:
            apply_persona_profile(state, profile)
        except Exception:
            return

    # Main handle
    def handle(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        import time as _time
        _t0 = _time.perf_counter()
        st, reply = self._handle_inner(state, user_input)
        if hasattr(st, "trim_messages"):
            st.trim_messages(keep_last=12)  # เก็บ state 12 (ไม่ใช่ส่งไป LLM ทั้งหมด)
        # Trim large context fields that bloat the prompt (topic_pool can be 100+ items)
        if st.context and len(st.context.get("topic_pool") or []) > 10:
            st.context["topic_pool"] = st.context["topic_pool"][:10]
        _elapsed = _time.perf_counter() - _t0
        _LOG.info(
            "[Supervisor] ⏱ response time = %.2fs | user=%r | reply_len=%d",
            _elapsed,
            (user_input or "")[:60],
            len(reply or ""),
        )
        return st, reply

    def _handle_inner(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        state.context = state.context or {}
        self._sync_persona_and_profile(state)

        raw = (user_input or "")
        raw_stripped = raw.strip()

        if not state.context.get("did_greet") and not raw_stripped and not self._is_academic_intake_active(state):
            state.context["did_greet"] = True
            return self._handle_greeting(state, user_input="")

        if not state.context.get("did_greet"):
            state.context["did_greet"] = True

        if raw_stripped:
            self._add_user(state, raw)

        # 2.1) Academic intake lock
        if self._is_academic_intake_active(state) and not state.context.get("awaiting_persona_confirmation"):
            st2, reply = self._academic.handle(state, raw_stripped, _internal=False)
            st2, reply = self._post_route_academic_auto_return(st2, reply)

            reply = self._normalize_male(reply)
            self._add_assistant(st2, reply)
            st2.last_action = "academic_intake_route"
            return st2, reply

        # 2.2) Clear legacy awaiting_persona_confirmation state (confirmation dialog removed)
        if state.context.get("awaiting_persona_confirmation"):
            state.context.pop("awaiting_persona_confirmation", None)
            state.context.pop("pending_persona", None)
            state.context.pop("pending_replay_user_input", None)
            state.context.pop("confirm_tries", None)

        # 2.2b) Academic resume: user wants to continue a previous academic session (remaining sections)
        # Triggered when academic_resume_available=True AND (regex matches OR LLM detects elaborate/continue intent).
        # GUARD: If the user has since asked about a different topic (last_retrieval_query differs from
        # academic_question), do NOT resume the old session — clear the flag and fall through to fresh routing.
        # FIX #3: cache the LLM fallback-intent result here so step 2.3 (style detection)
        # can reuse it without firing a second LLM call on the same input.
        # _cached_fallback_intent is None when 2.2b was skipped or returned early.
        _cached_fallback_intent: Optional[Dict[str, Any]] = None

        if state.context.get("academic_resume_available") and raw_stripped:
            _academic_q = (state.context or {}).get("academic_question", "").strip()
            _current_q = (state.get_last_retrieval_query() or "").strip()
            _topic_changed = bool(
                _academic_q and _current_q and _academic_q[:40].lower() != _current_q[:40].lower()
            )
            if _topic_changed:
                # User moved to a new topic — old academic session is stale; clear resume availability
                _LOG.info(
                    "[Supervisor] academic_resume skipped — topic changed (was=%r now=%r)",
                    _academic_q[:40], _current_q[:40],
                )
                state.context.pop("academic_resume_available", None)
                state.context.pop("academic_flow", None)
                state.context.pop("section_catalog", None)
                # Fall through to normal routing (2.3 may trigger a fresh silent switch)
            else:
                _is_resume = bool(self._ACADEMIC_RESUME_RE.search(raw_stripped))
                if not _is_resume:
                    # Guard: new legal questions must never trigger academic resume.
                    # Only call LLM when input doesn't look like a legal/follow-up question.
                    if not self._looks_like_legal_question(raw_stripped):
                        # LLM fallback: check if user explicitly wants to elaborate on last academic topic.
                        # Store result in _cached_fallback_intent so step 2.3 can reuse it.
                        try:
                            _last_q = state.get_last_retrieval_query() or (state.context or {}).get("last_topic", "")
                            _fi = self.llm_fallback_intent_call(raw_stripped, _last_q or "", "academic") or {}
                            _cached_fallback_intent = _fi
                            _intent = str(_fi.get("intent") or "").strip().lower()
                            # Only "elaborate" (wants more detail on SAME topic) triggers resume.
                            # Raise threshold 0.75 to avoid false positives on practical follow-ups.
                            if _intent == "elaborate" and float(_fi.get("confidence") or 0.0) >= 0.75:
                                _is_resume = True
                                _LOG.info("[Supervisor] academic_resume via LLM fallback intent=%r conf=%s", _intent, _fi.get("confidence"))
                        except Exception:
                            pass
                if _is_resume:
                    _LOG.info("[Supervisor] academic_resume triggered input=%r", raw_stripped[:40])
                    state.context.pop("academic_resume_available", None)
                    state.context.pop("pending_slot", None)  # clear practical topics menu
                    state.persona_id = "academic"
                    state.context["persona_id"] = "academic"
                    # Reset FSM to awaiting_sections — section_catalog is still in context
                    flow = dict(state.context.get("academic_flow") or {})
                    flow["stage"] = "awaiting_sections"
                    state.context["academic_flow"] = flow
                    st2, reply = self._academic.handle(state, raw_stripped, _internal=False)
                    st2, reply = self._post_route_academic_auto_return(st2, reply)
                    reply = self._normalize_male(reply)
                    self._add_assistant(st2, reply)
                    st2.last_action = "academic_resume"
                    return st2, reply

        # 2.2c) Early off-topic guardrail — zero LLM calls, pure regex.
        # If input has NO legal/business signal AND is not a number AND is not a greeting,
        # do a quick Chroma similarity check before deflecting — this auto-covers any new topics
        # added to the vector store without needing hardcoded keyword updates.
        # Conditions to NOT short-circuit: legal signal present, number, question mark (might
        # be a legal question phrased as question), greeting/thanks, or depth/detail request
        # (e.g. "ขอแบบละเอียด" "อธิบายละเอียด" "เชิงลึก" — must reach 2.3 to switch to Academic).
        # BYPASS: skip entirely when pending_slot is active — user is answering a slot question,
        # not asking a new topic (e.g. "นิติ" as answer to entity_type question).
        _pending_slot_active = bool((state.context or {}).get("pending_slot"))
        if _pending_slot_active:
            pass  # fall through to practical routing below
        elif (
            raw_stripped
            and not self._LEGAL_SIGNAL_RE.search(raw_stripped)
            and not self._LIKELY_SELECTION_RE.match(raw_stripped)
            and not self._ANY_NUMBER_RE.search(raw_stripped)
            and not self._QUESTION_MARKERS_RE.search(raw_stripped)
            and not self._looks_like_greeting_or_thanks(raw_stripped)
            and not self._looks_like_legal_question(raw_stripped)
            and not self._DEPTH_DETAIL_RE.search(raw_stripped)
        ):
            # Quick Chroma check: if a relevant doc exists (sim ≥ 0.72), treat as in-domain.
            # This handles any topic in the vector store without hardcoding keywords.
            _chroma_in_domain = False
            try:
                _vstore_2c = getattr(self._practical.retriever, "vectorstore", None)
                if _vstore_2c is not None:
                    _pairs_2c = _vstore_2c.similarity_search_with_relevance_scores(raw_stripped, k=1)
                    if _pairs_2c and _pairs_2c[0][1] >= 0.72:
                        _LOG.info(
                            "[Supervisor] 2.2c Chroma hit sim=%.3f — not deflecting: %r",
                            _pairs_2c[0][1], raw_stripped[:60],
                        )
                        _chroma_in_domain = True
            except Exception as _e_2c:
                _LOG.debug("[Supervisor] 2.2c Chroma check failed: %s", _e_2c)
            if not _chroma_in_domain:
                _LOG.info("[Supervisor] 2.2c early_offtopic → deflect input=%r", raw_stripped[:60])
                return self._handle_deflect(state, raw_stripped)

        # 2.3) style request -> silent switch (no confirmation dialog)
        # FIX #2: A legal question that happens to contain words like "รายละเอียด/ทั้งหมด"
        # must NOT be intercepted here.  Style detection only applies when the input is
        # a PURE style modifier with no independent legal content.  If the input already
        # has legal signal it will be answered correctly by step 2.9 (legal routing).
        # EXCEPTION: short follow-up depth requests like "ขอเรื่อง X แบบละเอียด" or
        # "ละเอียดกว่านี้" — these are style requests even if they contain a legal keyword.
        # Heuristic: wants_long + short input (≤10 words) + explicit depth modifier present.
        _raw_is_legal = self._looks_like_legal_question(raw_stripped)
        style = self._infer_user_style_request_hybrid(raw_stripped)
        _is_short_depth_followup = (
            style.get("wants_long")
            and _raw_is_legal
            and len(raw_stripped.split()) <= 10
            and bool(self._DEPTH_DETAIL_RE.search(raw_stripped))
        )
        if style.get("wants_long") and (not _raw_is_legal or _is_short_depth_followup):
            return self._silent_switch_to_academic(state, raw_stripped)
        # wants_short or legal question: continue to practical routing below

        # 2.4) explicit target in text + switch verb -> silent switch
        explicit_target = self._infer_target_persona_from_text(raw_stripped)
        if explicit_target in {"academic", "practical"} and any(v in self._normalize_for_intent(raw_stripped) for v in self._SWITCH_VERBS):
            if explicit_target == "academic":
                return self._silent_switch_to_academic(state, raw_stripped)
            # explicit_target == "practical": already in practical, continue routing

        # 2.5) switch without target -> silent switch to academic (toggle)
        if self._looks_like_switch_without_target(raw_stripped):
            cur = normalize_persona_id(state.persona_id)
            if cur != "academic":
                return self._silent_switch_to_academic(state, raw_stripped)
            # Already in academic but intake not active (section 2.1 would've caught that) → ignore

        # 2.5.5) Number typed but no pending_slot → promote queue item OR recover from last_topic_menu
        if not self._has_pending_slot(state) and self._LIKELY_SELECTION_RE.match(raw_stripped):
            _active_queue = (state.context or {}).get("topic_slot_queue") or []
            # Sanitize: drop non-dict entries that Gemini might have smuggled in as strings
            _active_queue = [s for s in _active_queue if isinstance(s, dict)]
            if _active_queue:
                # topic_slot_queue has priority: promote next queued slot to pending_slot
                # so 2.6 can map the user's number to the correct slot options.
                # (Prevents last_topic_menu from overwriting a pending entity/area/op question.)
                _next_slot = _active_queue[0]
                _remaining_q = _active_queue[1:]
                _qkey = _next_slot.get("key", "")
                _qopts = _next_slot.get("options", [])
                if _qkey and _qopts:
                    state.context["pending_slot"] = {
                        "key": _qkey,
                        "options": list(_qopts),
                        "allow_multi": False,
                    }
                    if _remaining_q:
                        state.context["topic_slot_queue"] = _remaining_q
                    else:
                        state.context.pop("topic_slot_queue", None)
                    _LOG.info("[Supervisor] slot_queue priority: promoted key=%r to pending_slot for number input %r",
                              _qkey, raw_stripped)
            else:
                last_menu = (state.context or {}).get("last_topic_menu") or []
                if last_menu:
                    idxs = self._parse_indices(raw_stripped)
                    valid = [i for i in idxs if 1 <= i <= len(last_menu)]
                    if valid:
                        # Restore pending_slot temporarily so 2.6 can route it normally
                        state.context["pending_slot"] = {
                            "key": "topic",
                            "options": last_menu,
                            "allow_multi": len(valid) > 1,
                        }
                        _LOG.info("[Supervisor] restored last_topic_menu for number input %r", raw_stripped)

        # 2.6) pending slot route
        if self._should_route_pending_slot_now(state, raw_stripped):
            return self._route_pending_slot_to_persona(state, raw_stripped)

        # 2.6b) Typo / garbled input detection — runs BEFORE greeting/noise check so that
        #        single garbled chars (e.g. "ๅ") are caught here rather than triggering a
        #        full greeting menu refresh.
        #        Priority: rule-based fast check first, then LLM for ambiguous cases.
        #        IMPORTANT: preserve pending_slot — user should be able to retype and continue.
        if raw_stripped:
            _is_typo = False
            _typo_suggested = ""

            if self._is_likely_typo_rule(raw_stripped):
                # Obviously garbled — no need for LLM
                _is_typo = True
                _LOG.info("[Supervisor] typo_rule_fast input=%r", raw_stripped[:30])
            elif (
                not self._looks_like_greeting_or_thanks(raw_stripped)
                and not self._looks_like_legal_question(raw_stripped)
                and not self._LIKELY_SELECTION_RE.match(raw_stripped)
                and 1 <= len(raw_stripped) <= 20  # only check short ambiguous inputs
                and not self._LEGAL_SIGNAL_RE.search(raw_stripped)
                and not self._QUESTION_MARKERS_RE.search(raw_stripped)
            ):
                # Ambiguous short input — ask LLM
                try:
                    _last_topic = (state.context or {}).get("last_user_legal_query", "") or (state.context or {}).get("last_topic", "")
                    _typo_res = self.llm_typo_check_call(raw_stripped, _last_topic) or {}
                    if _typo_res.get("is_typo") and float(_typo_res.get("confidence") or 0.0) >= 0.75:
                        _is_typo = True
                        _typo_suggested = str(_typo_res.get("suggested") or "").strip()
                        _LOG.info("[Supervisor] typo_llm input=%r suggested=%r conf=%s", raw_stripped[:30], _typo_suggested, _typo_res.get("confidence"))
                except Exception as _te:
                    _LOG.debug("[Supervisor] typo_check_llm error: %s", _te)

            if _is_typo:
                return self._handle_typo_prompt(state, raw_stripped, _typo_suggested)

        # 2.7) greeting/noise → short reply only, no topic menu
        if self._looks_like_greeting_or_thanks(raw_stripped) or self._is_noise(raw_stripped) or not raw_stripped:
            return self._handle_greeting(state, user_input=raw_stripped, show_menu=False)

        # 2.8) mode status
        if self._looks_like_mode_status_query(raw_stripped):
            pid = normalize_persona_id(state.persona_id)
            msg = self._normalize_male(f"ℹ️ ตอนนี้เป็นโหมด {pid} ครับ")
            self._add_assistant(state, msg)
            state.last_action = "mode_status"
            return state, msg

        # 2.9) legal routing
        if self._looks_like_legal_question(raw_stripped):
            state.context["last_user_legal_query"] = raw_stripped

            # Legal question interrupts a pending non-topic slot — clear stale slot context.
            # Without this, practical's _consume_pending_slot_from_user() would accept the
            # legal question text as a free-text slot fill (line 966), corrupting collected_slots.
            # Exemptions:
            #   - "topic": a legal phrase IS a valid topic selection
            #   - "operation_group": options contain legal keywords (จดทะเบียน, ต่ออายุ, etc.)
            #   - "registration_type": option text triggers LEGAL_SIGNAL_RE (บริษัทจำกัด etc.)
            #   - "department": user naming a dept (e.g. "กรมพัฒนาธุรกิจการค้า") triggers LEGAL_SIGNAL_RE
            #     but is a valid slot fill — must not clear the dept slot and re-ask
            _INTERRUPT_EXEMPT = {"topic", "operation_group", "registration_type", "department"}
            _int_ps = (state.context or {}).get("pending_slot")
            if isinstance(_int_ps, dict) and _int_ps.get("key") not in _INTERRUPT_EXEMPT and _int_ps.get("key"):
                _LOG.info(
                    "[Supervisor] 2.9 legal interrupt: clearing pending_slot(%r) and topic_slot_queue",
                    _int_ps.get("key"),
                )
                state.context.pop("pending_slot", None)
                state.context.pop("topic_slot_queue", None)
            elif isinstance(_int_ps, dict) and _int_ps.get("key") in _INTERRUPT_EXEMPT:
                _exempt_key = _int_ps.get("key")
                # "topic" slot is normally exempt, BUT bypass the menu when the user asks
                # a specific question directly (informational, link request, or open-ended
                # procedure/document query like วิธีการ/ต้องการลิ้งค์/ขั้นตอน).
                _bypass_topic_slot = (
                    _exempt_key == "topic"
                    and (
                        self._INFO_Q_RE.search(raw_stripped)
                        or self._LINK_REQUEST_RE.search(raw_stripped)
                        or self._GENERIC_FOLLOWUP_KEYWORDS_RE.search(raw_stripped)
                    )
                )
                # Non-topic exempt slots (department/registration_type/operation_group):
                # bypass when user asks a full new legal question that doesn't match any option.
                _bypass_non_topic_slot = False
                if not _bypass_topic_slot and _exempt_key != "topic":
                    _opts = _int_ps.get("options") or []
                    _any_opt_match = any(str(o) and str(o) in raw_stripped for o in _opts)
                    if (
                        not _any_opt_match
                        and len(raw_stripped) >= 8
                        and self._looks_like_legal_question(raw_stripped)
                    ):
                        _bypass_non_topic_slot = True
                if _bypass_topic_slot or _bypass_non_topic_slot:
                    _LOG.info(
                        "[Supervisor] 2.9 %s slot: new legal question bypasses pending slot: %r",
                        _exempt_key, raw_stripped[:60],
                    )
                    # Clear pending_slot so practical doesn't re-evaluate it and show INVALID menu
                    state.context.pop("pending_slot", None)
                    state.context.pop("topic_slot_queue", None)
                    # fall through to legal question handler below
                else:
                    # These slots must be answered first — route back to slot handler
                    _LOG.info(
                        "[Supervisor] 2.9 legal interrupt SKIPPED — pending_slot(%r) is exempt, routing as slot reply",
                        _int_ps.get("key"),
                    )
                    return self._route_pending_slot_to_persona(state, raw_stripped)

            pid = normalize_persona_id(state.persona_id)
            if pid == "academic":
                st2, reply = self._academic.handle(state, raw_stripped, _internal=False)
                st2, reply = self._post_route_academic_auto_return(st2, reply)

                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "academic_answer"
                return st2, reply

            self._ensure_practical_retrieval_for_legal(state, raw_stripped)
            # Skip slot queue when user explicitly asks for a link/document — answer directly
            if not self._LINK_REQUEST_RE.search(raw_stripped):
                self._maybe_build_slot_queue_from_docs(state, raw_stripped)
            st2, reply = self._practical.handle(state, raw_stripped, _internal=False)
            reply = self._normalize_male(reply)
            self._add_assistant(st2, reply)
            st2.last_action = "practical_answer"
            return st2, reply

        # 3) Short interjection → short reply, no topic menu
        if self._TH_INTERJECTION_RE.match(raw_stripped):
            _LOG.info("[Supervisor] interjection→greeting persona=%s input=%r", getattr(state, "persona_id", "?"), raw_stripped[:30])
            return self._handle_greeting(state, raw_stripped, show_menu=False)

        # 3.1) "ขอหัวข้อใหม่", "มีเรื่องอื่นอีกไหม", "แนะนำหัวข้อ" → refresh menu
        if self._NEW_TOPIC_RE.search(raw_stripped):
            _LOG.info("[Supervisor] new_topic_request→greeting input=%r", raw_stripped[:40])
            return self._handle_greeting(state, raw_stripped)

        # 3.2) "อธิบายมากกว่านี้", "ขยายความ", "เพิ่มเติมอีก" → re-route with last legal query
        if self._ELABORATE_RE.search(raw_stripped):
            last_q = (state.context or {}).get("last_user_legal_query", "").strip()
            if last_q:
                _LOG.info("[Supervisor] elaborate→practical last_q=%r", last_q[:40])
                self._ensure_practical_retrieval_for_legal(state, last_q)
                elaborate_input = f"อธิบายรายละเอียดเพิ่มเติม: {last_q}"
                st2, reply = self._practical.handle(state, elaborate_input, _internal=False)
                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "practical_elaborate"
                return st2, reply

        # 3.3) "อันไหนเหมาะกับฉัน", "สำหรับกรณีฉัน" → contextual follow-up with last legal query
        if self._FOLLOWUP_CONTEXTUAL_RE.search(raw_stripped):
            last_q = (state.context or {}).get("last_user_legal_query", "").strip()
            if last_q:
                _LOG.info("[Supervisor] contextual_followup→practical input=%r last_q=%r", raw_stripped[:30], last_q[:30])
                self._ensure_practical_retrieval_for_legal(state, last_q)
                combined = f"{raw_stripped} (เกี่ยวกับ: {last_q})"
                st2, reply = self._practical.handle(state, combined, _internal=False)
                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "practical_contextual_followup"
                return st2, reply

        # 3.4) Link/document request with active context → practical directly (no LLM needed)
        # Catches "ขอลิงค์คู่มือ", "ส่งแบบฟอร์ม", "ขอดาวน์โหลด" etc. before LLM misclassifies as new_topic
        if self._LINK_REQUEST_RE.search(raw_stripped):
            _last_q_lr = (state.context or {}).get("last_user_legal_query", "").strip()
            _has_ctx_lr = bool(_last_q_lr or (state.context or {}).get("last_topic") or state.current_docs)
            if _has_ctx_lr:
                _LOG.info("[Supervisor] 3.4 link_request with active context → practical: %r", raw_stripped[:60])
                if _last_q_lr:
                    self._ensure_practical_retrieval_for_legal(state, _last_q_lr)
                st2, reply = self._practical.handle(state, raw_stripped, _internal=False)
                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "practical_link_request"
                return st2, reply

        # 4) LLM fallback intent classifier — no hardcode, no dead-end error message
        # FIX #3 (part 2): reuse the cached fallback-intent result from step 2.2b if available,
        # so we never call llm_fallback_intent_call twice on the same input in one turn.
        _LOG.info("[Supervisor] fallback_intent_llm persona=%s input=%r", getattr(state, "persona_id", "?"), raw_stripped[:60])
        intent_res: Dict[str, Any] = {}
        if _cached_fallback_intent is not None:
            intent_res = _cached_fallback_intent
            _LOG.info("[Supervisor] fallback_intent reused from step 2.2b cache")
        else:
            try:
                last_q_fb = (state.context or {}).get("last_user_legal_query", "")
                persona_fb = normalize_persona_id(getattr(state, "persona_id", "practical"))
                intent_res = self.llm_fallback_intent_call(raw_stripped, last_q_fb, persona_fb) or {}
            except Exception as _e:
                _LOG.warning("[Supervisor] fallback_intent_llm error: %s", _e)
                intent_res = {}

        fallback_intent = (intent_res.get("intent") or "unknown").strip().lower()
        _LOG.info("[Supervisor] fallback_intent=%r confidence=%s", fallback_intent, intent_res.get("confidence"))

        if fallback_intent == "new_topic":
            _LOG.info("[Supervisor] fallback_llm→new_topic input=%r", raw_stripped[:40])
            # Option B: if active context AND user didn't explicitly ask for a topic menu
            # → treat as legal_question (retrieve + practical) instead of opening menu.
            # Topic menu only when: no active context OR user explicitly says "ขอหัวข้อใหม่" etc.
            _has_active_ctx_nt = bool(
                (state.context or {}).get("last_user_legal_query")
                or (state.context or {}).get("last_topic")
                or state.current_docs
            )
            _explicitly_wants_menu = bool(self._NEW_TOPIC_RE.search(raw_stripped))
            if _has_active_ctx_nt and not _explicitly_wants_menu:
                _last_q_nt = (state.context or {}).get("last_user_legal_query", "").strip()
                _q_nt = _last_q_nt or raw_stripped
                _LOG.info("[Supervisor] new_topic → legal_question (active ctx, no explicit menu request): %r", raw_stripped[:60])
                state.context["last_user_legal_query"] = raw_stripped
                self._ensure_practical_retrieval_for_legal(state, _q_nt)
                if not self._LINK_REQUEST_RE.search(raw_stripped):
                    self._maybe_build_slot_queue_from_docs(state, raw_stripped)
                st2, reply = self._practical.handle(state, raw_stripped, _internal=False)
                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "practical_follow_up"
                return st2, reply
            return self._handle_greeting(state, raw_stripped)

        if fallback_intent == "elaborate":
            last_q_fb2 = (state.context or {}).get("last_user_legal_query", "").strip()
            if last_q_fb2:
                _LOG.info("[Supervisor] fallback_llm→elaborate last_q=%r", last_q_fb2[:40])
                self._ensure_practical_retrieval_for_legal(state, last_q_fb2)
                st2, reply = self._practical.handle(state, f"อธิบายรายละเอียดเพิ่มเติม: {last_q_fb2}", _internal=False)
                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "fallback_llm_elaborate"
                return st2, reply

        if fallback_intent in ("legal_question", "business_question"):
            q_fb = (intent_res.get("query") or raw_stripped).strip()
            _LOG.info("[Supervisor] fallback_llm→%s q=%r", fallback_intent, q_fb[:40])
            state.context["last_user_legal_query"] = q_fb
            self._ensure_practical_retrieval_for_legal(state, q_fb)
            # Slot queue only builds when retrieved docs have license_type (regulatory).
            # For marketing/business_guide docs (no license_type) it exits early — no slot-fill.
            if not self._LINK_REQUEST_RE.search(raw_stripped):
                self._maybe_build_slot_queue_from_docs(state, q_fb)
            st2, reply = self._practical.handle(state, q_fb, _internal=False)
            reply = self._normalize_male(reply)
            self._add_assistant(st2, reply)
            st2.last_action = "fallback_llm_business"
            return st2, reply

        if fallback_intent == "greeting":
            return self._handle_greeting(state, raw_stripped, show_menu=False)

        # Truly off-topic (unknown intent) → guardrail deflect
        _LOG.info("[Supervisor] unknown_intent → deflect input=%r", raw_stripped[:60])
        return self._handle_deflect(state, raw_stripped)
    
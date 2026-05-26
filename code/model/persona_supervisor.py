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
from utils.query_synonyms import SYNONYM_PATTERNS
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
    build_entity_type_detect_prompt,
    build_location_detect_prompt,
    build_operation_type_detect_prompt,
    build_area_size_detect_prompt,
    build_license_type_detect_prompt,
    build_academic_resume_detect_prompt,
    build_elaborate_detect_prompt,
    build_broad_question_detect_prompt,
    build_slot_skip_detect_prompt,
    build_followup_contextual_detect_prompt,
    build_thanks_detect_prompt,
    build_academic_stop_detect_prompt,
    build_smalltalk_detect_prompt,
    build_switch_without_target_detect_prompt,
    build_link_request_detect_prompt,
    build_mode_status_detect_prompt,
    build_greeting_detect_prompt,
    build_info_action_q_detect_prompt,
    build_legal_q_detect_prompt,
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

    # License type strings used in _SUPERVISOR_KW_OVERRIDE (keyword→license mapping).
    # Must stay in sync with actual license_type values in Chroma.
    # Validated at startup by _validate_license_types_at_startup().
    _KW_OVERRIDE_LICENSE_TYPES: List[str] = [
        'ใบรับรองมาตรฐานร้านอาหาร',
        'ระบบชำระเงินออนไลน์',
        'ใบภาษีมูลค่าเพิ่ม ภพ.20',
        'ทะเบียนผู้ประกันตน',
        'ทะเบียนนายจ้าง',
        'จัดการการเงิน',
        'ใบอนุญาตจำหน่ายสุรา',
        'แบบแสดงรายการภาษีป้ายร้านอาหาร',
    ]

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
        "เจาะลึก",
        "ลึกซึ้ง",
        "ลงลึก",
        "วิชาการ",
        "อ้างอิงข้อกฎหมาย",
        "อธิบายละเอียด",
        "ขอแบบละเอียด",
        "แบบละเอียด",
        "อย่างละเอียด",
        "ละเอียดกว่านี้",
        "ละเอียดกว่า",
        "ละเอียดหน่อย",
        "ละเอียดขึ้น",
        "ละเอียดทั้งหมด",
        "ขอแบบละเอียดทั้งหมด",
        "เอาแบบละเอียดทั้งหมด",
        # "ขยายความ" removed: แปลว่า "อธิบายเพิ่ม" ในภาษาไทยทั่วไป ไม่ใช่ signal ที่ต้องเข้า Academic
        # Academic resume จะ handle "ขยายความ" เรื่องเดิมผ่าน LLM intent detection อยู่แล้ว
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
        # Style words that stand alone — clearly indicate a style/format preference.
        r"(?:ช่วยอธิบาย|ขยายความ|ลงรายละเอียด|ละเอียดขึ้น|เชิงลึก|สรุป|สั้นๆ|กระชับ|อยากได้|ขอให้)"
        # Prefix words (ขอ/ช่วย/รบกวน/เอา) are common polite Thai prefixes used in any question.
        # Only treat them as a style request when followed by a style modifier within 8 chars.
        # e.g. "รบกวนสรุปให้" ✓, "รบกวนถามเรื่องใบอนุญาต" ✗ (no style modifier follows).
        r"|(?:ขอ(?!ง)|ช่วย|รบกวน|เอา).{0,8}(?:สั้น|กระชับ|ย่อ|สรุป|ขยายความ|ลงรายละเอียด|ละเอียดขึ้น|เชิงลึก)",
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
    # Matches any thanks expression. Removed the old negative lookahead (?![\u0E00-\u0E7F]{4,})
    # because it caused false negatives: "ขอบคุณสำหรับข้อมูล" (10 Thai chars) never matched,
    # routing legitimate thanks as legal questions. "ขอบคุณ" + Thai is always thanks.
    _THANKS_RE = re.compile(
        r"(ขอบคุณ|ขอบใจ|thx|thanks)",
        re.IGNORECASE,
    )

    _LIKELY_SELECTION_RE = re.compile(r"^\s*(?=.*\d)[\d\s,/-]+\s*$")

    _QUESTION_MARKERS_RE = re.compile(
        r"(\?|\bไหม\b|หรือไม่|หรือเปล่า|ยังไง|ทำไง|อย่างไร|ได้ไหม|ควร|ต้อง|คืออะไร"
        r"|วิธีการ|วิธี(?!ีเดียว|ีชีวิต))",
        re.IGNORECASE,
    )

    _LEGAL_SIGNAL_RE = re.compile(
        r"(ใบอนุญาต|จดทะเบียน|ทะเบียนพาณิชย์|ภาษี|vat|ภพ\.?20|สรรพากร|เทศบาล|สำนักงานเขต|สุขาภิบาล|กรม|ค่าธรรมเนียม|เอกสาร|ขั้นตอน|บทลงโทษ|ประกาศ|พ\.ร\.บ|ประกันสังคม|กองทุน|เปิดร้าน|ขึ้นทะเบียน"
        r"|เงินสมทบ|สมทบ|แยกยื่น|ยื่นรวม|ส่งเงินสมทบ|ส่งข้อมูลเงินสมทบ"
        r"|qr.?pay|qr.?payment|คิวอาร์|คิวอาเพย|เพย์เมนต์|edc|รูดบัตร|merchant.?id|partner.?id|pos.?id|ระบบชำระเงิน|กสิกร|kbank|ไทยพาณิช(?:ย์)?|ไทยพานิช|scb|ประกอบกิจการ|สุขาภิบาลอาหาร"
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
        r"|งบการเงิน|ผลกระทบ|ความเสี่ยง"
        # Reference/source queries — user asking where info comes from (Fix A)
        r"|อ้างอิง|แหล่งที่มา|แหล่งข้อมูล|แหล่งอ้างอิง|ที่มาของข้อมูล)",
        re.IGNORECASE,
    )

    _NOISE_ONLY_RE = re.compile(r"^(?:[a-z]+|[!?.]+)$", re.IGNORECASE)
    _TH_LAUGH_5_RE = re.compile(r"^\s*5{3,}\s*$")

    # Depth/detail requests — signals user wants more elaboration on current topic.
    # These must NOT be deflected by 2.2c and must be routed to Academic persona.
    _DEPTH_DETAIL_RE = re.compile(
        r"(แบ.?ละเอียด|อย่างละเอียด|ละเอียดกว่า|ละเอียดมากกว่า|ละเอียดหน่อย|ละเอียดขึ้น|เชิงลึก|แบบเต็ม"
        r"|ครบถ้วน|ทั้งหมดเลย|แบบวิชาการ|อธิบายละเอียด|แบบเต็มๆ"
        r"|ขอดูแบบละเอียด|ขอรายละเอียด|อธิบายเพิ่ม|อธิบายต่อ|อธิบายให้ชัดเจนขึ้น"
        r"|ขอเพิ่มเติม|ดูเพิ่มเติม|รายละเอียดเพิ่ม|แบบให้ลึกซึ้งยิ่งขึ้น|มากกว่านี้"
        r"|เจาะลึก|ลึกซึ้ง|เจาะจง|ลงลึก|ลึกกว่า)",
        re.IGNORECASE,
    )

    # New-question request prefix — user is asking a fresh question, not picking from a menu.
    # "ต้องการ..." is caught by _QUESTION_MARKERS_RE (ต้อง) so section 2.9 is entered,
    # but it should bypass the greeting topic slot, not consume it as a menu selection.
    _TOPIC_REQUEST_PREFIX_RE = re.compile(
        r"^(ต้องการ|อยากรู้|อยากทราบ|ขอข้อมูล|ช่วยบอก|บอกเรื่อง|สนใจเรื่อง|อธิบายเรื่อง)\S",
        re.IGNORECASE,
    )

    # Follow-up patterns (handled before fallback_safe_return)
    _ELABORATE_RE = re.compile(
        r"(อธิบาย(มากกว่า|เพิ่ม|เพิ่มเติม|ขยาย|ให้ละเอียด|ให้ครบ|ต่อ)|ขยายความ|เพิ่มเติมอีก|รายละเอียดมากกว่า|บอกเพิ่ม|เล่าให้ฟัง|อธิบายต่อ|รายละเอียดเพิ่ม"
        r"|ขอให้บอกเพิ่ม|ช่วยอธิบาย(?:เพิ่ม|ต่อ|ขยาย|เติม)|ละเอียดกว่านี้|ครบกว่านี้"
        r"|อยากได้รายละเอียดเพิ่ม|ต้องการรายละเอียดเพิ่ม|เพิ่มเติมหน่อย(?:ได้มั้ย|ได้ไหม|นะ|ครับ|ค่ะ)?"
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
        r"|กลับไปเรื่องเดิม|กลับไปเรื่องเก่า|ต่อจากที่แล้ว"
        r"|อยากอ่านต่อ|ขอดูส่วนอื่น|ยังมีข้อมูลอีกไหม|ขอต่อเลย|เอาต่อเลย"
        r"|ดูข้อมูลส่วนอื่น|ยังอยากรู้อีก|อ่านต่อได้เลย|ขอส่วนต่อไป"
        r"|ส่วนต่อไปเลย|ดูต่อเลย|ขอดูต่อ|อยากได้ข้อมูลเพิ่ม(?:เติม)?)",
        re.IGNORECASE,
    )
    # Academic stop: user wants to exit detailed/academic mode and return to practical.
    # Checked BEFORE _ACADEMIC_RESUME_RE in the academic_resume_available section.
    _ACADEMIC_STOP_RE = re.compile(
        r"จบแล้ว|พอแล้ว|ไม่เอาแล้ว|หยุดก่อน|หยุดได้แล้ว"
        r"|ออกจากโหมดนี้|กลับโหมดปกติ|กลับไปโหมดปกติ|ยกเลิกโหมดนี้"
        r"|ไม่ต้องการรายละเอียดแล้ว|หยุดแบบละเอียด|ออกโหมดนี้"
        r"|ไม่ต้องการแล้ว|พอก่อน|หยุดตรงนี้ก่อน|จบตรงนี้ก่อน",
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
    # Slot-skip: user declines to answer a slot question ("ไม่รู้", "ข้ามก่อน", etc.)
    # Only evaluated inside _route_pending_slot_to_persona where context guarantees a slot is active.
    _SLOT_SKIP_RE = re.compile(
        r"ไม่รู้|ไม่แน่ใจ|ยังไม่รู้|ยังไม่แน่ใจ|ยังไม่ตัดสินใจ|ยังไม่ได้ตัดสินใจ|ยังหาข้อมูลอยู่"
        r"|ข้ามก่อน|ข้ามไปก่อน|ข้ามได้มั้ย|ข้ามได้ไหม|ขอข้าม|ข้ามไปเลย|ข้ามเลย|ข้ามข้อนี้|ข้ามคำถามนี้|ข้ามได้เลย"
        r"|ไม่ต้องถาม|ยังไม่รู้เลย|ไม่ทราบครับ|ไม่ทราบค่ะ|ไม่ทราบนะ"
        r"|skip(?:\s+this)?",
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
        r"|ขอแบบฟอร์ม|ส่งแบบฟอร์ม|แบบฟอร์ม(ที่|ของ|ด้วย)"
        # Reference source queries — Fix B: route directly to Practical instead of LLM misclassify
        r"|อ้างอิงได้ที่|ขอแหล่งข้อมูล|ดูอ้างอิง|อ้างอิงจาก|แหล่งอ้างอิง"
        r"|แหล่งที่มา(?:ของข้อมูล)?|ที่มาของข้อมูล|ข้อมูลจากไหน|แหล่งข้อมูลเพิ่มเติม|ขอแหล่ง)",
        re.IGNORECASE,
    )
    # Short Thai interjections that are not legal questions → re-show menu
    _TH_INTERJECTION_RE = re.compile(
        r"^\s*(เอ้|เฮ้|เฮ|โอ้|โอ้โห|อ้าว|อ้าว|ว้าว|เออ|เอ่อ|อ่า|อ้า|อืม|อ๋อ|อ๋อ|เออนะ|เอ้าๆ|งั้นหรอ|งั้นเหรอ|จริงดิ|จริงเหรอ|ไม่ใช่เหรอ|เหรอ)\s*(ครับ|คับ|ค่ะ|คะ|นะ|นะครับ|นะคะ)?\s*$",
        re.IGNORECASE,
    )
    # Specificity override for broad_q detection: when these license names/codes appear in the
    # query, it is a targeted question about a specific license — not a broad overview.
    # e.g. "ขอใบภาษีมูลค่าเพิ่ม ภพ.20 ต้องใช้เอกสารอะไรบ้าง" is targeted, not broad.
    _SPECIFIC_LICENSE_INDICATOR_RE = re.compile(
        # Specific form/tax codes
        r"ภพ\.\s*\d|ภส\.\s*\d|ภป\.\s*\d|สณ\.\s*\d|บอจ\.\s*\d|VAT|ภ\.อ\."
        # Named license types
        r"|ใบภาษีมูลค่าเพิ่ม|ใบทะเบียนพาณิชย์|ทะเบียนพาณิชย์"
        r"|ใบอนุญาตจัดตั้ง|ใบอนุญาตจำหน่ายสุรา|ใบอนุญาตจำหน่าย"
        r"|ใบรับรองมาตรฐานร้านอาหาร|ใบวุฒิบัตรผู้สัมผัสอาหาร"
        r"|ใบรับรองแพทย์|แบบแสดงรายการภาษีป้าย"
        r"|ประกันสังคม|กองทุนประกันสังคม",
        re.IGNORECASE,
    )
    # Broad/overview question: triggers two-pass retrieval (pass1=user intent, pass2=regulatory).
    # Covers two categories:
    #   A) Business-opening questions: "อยากเปิดร้านต้องทำอะไรบ้าง" — spans all dimensions
    #   B) Category-overview questions: "ต้องเสียภาษีอะไรบ้าง", "ต้องขอใบอนุญาตอะไรบ้าง"
    #      — user asks for a list of items in a broad category, not a specific item
    # NOT triggered for specific questions: "VAT ต้องจดเมื่อรายได้เท่าไหร่", "จดทะเบียนพาณิชย์ต้องใช้เอกสารอะไร"
    _BROAD_Q_RE = re.compile(
        # A) Business-opening: asking what to do / prepare / know to open a business
        r"(อยากเปิด|จะเปิด|ต้องการเปิด|อยากตั้ง|จะตั้ง|อยากทำ|จะทำ)"
        r".{0,30}(ต้องทำอะไร|ต้องรู้อะไร|ต้องเตรียม|ต้องทำ\s*และ|ต้องรู้|ต้องมีอะไร|ต้องทำยังไง|ต้องทำอย่างไร|เริ่มจากตรงไหน|เริ่มต้นยังไง|ต้องจัดการอะไร|ควรรู้อะไร|ควรเตรียมอะไร|ต้องดูแลอะไร)"
        r"|(ต้องทำอะไรบ้าง|ต้องรู้อะไรบ้าง|ต้องเตรียมอะไรบ้าง|ต้องจัดการอะไรบ้าง|ควรรู้อะไรบ้าง|ควรเตรียมอะไรบ้าง)"
        r".{0,50}(ร้าน|คาเฟ่|เบเกอรี่|กิจการ|ธุรกิจ|อาหาร)"
        r"|(ร้าน|คาเฟ่|เบเกอรี่|กิจการ|ธุรกิจ).{0,30}(ต้องทำอะไรบ้าง|ต้องรู้อะไรบ้าง|ต้องเตรียมอะไรบ้าง|ต้องทำ\s*และ\s*ต้องรู้|ต้องจัดการอะไรบ้าง|ควรรู้อะไรบ้าง)"
        # B) Tax overview: asking which taxes apply (not asking about a specific tax)
        r"|(ภาษี|ค่าใช้จ่าย).{0,30}(อะไรบ้าง|มีอะไรบ้าง|ประเภทใดบ้าง|อะไรที่ต้อง)"
        r"|(ต้องเสีย|ต้องจ่าย|ต้องชำระ).{0,20}ภาษี.{0,20}(อะไรบ้าง|มีอะไรบ้าง|อะไรที่ต้อง)"
        # C) License/permit overview: asking which licenses/permits are needed
        r"|(ใบอนุญาต|อนุญาต|ใบ).{0,30}(อะไรบ้าง|มีอะไรบ้าง|ต้องขออะไรบ้าง|ที่ต้องมี)"
        r"|(ต้องขอ|ต้องมี|ต้องได้).{0,20}(ใบอนุญาต|อนุญาต).{0,20}(อะไรบ้าง|มีอะไร)"
        # D) Document/requirement overview: asking what documents/steps are needed in general
        r"|(เอกสาร|หลักฐาน).{0,30}(อะไรบ้าง|มีอะไรบ้าง|ที่ต้องใช้ทั้งหมด)"
        r"|(ขั้นตอน|กระบวนการ|ต้องทำอะไร).{0,30}(ทั้งหมด|ทุกอย่าง|ครบถ้วน|ทุกด้าน)"
        # E) Registration/compliance overview: "ต้องจดอะไรบ้าง", "ต้องขึ้นทะเบียนอะไรบ้าง"
        r"|(ต้องจด|ต้องขึ้นทะเบียน|ต้องลงทะเบียน).{0,20}(อะไรบ้าง|อะไร|อะไรบ้างครับ)"
        r"|(จดทะเบียน|ขึ้นทะเบียน|ลงทะเบียน).{0,20}(อะไรบ้าง|อะไรที่ต้อง|ทั้งหมด)",
        re.IGNORECASE,
    )
    # Operation inference: detect from user query which operation_group is clearly implied
    _OP_INFER_NEW_RE = re.compile(
        r"จด\s*(vat|ภาษีมูลค่าเพิ่ม|ภพ\b|ทะเบียน|ใหม่)"
        r"|ต้อง\s*จด|ขอ\s*จด|ยื่นขอ\s*ใหม่|จดทะเบียน\s*ใหม่|ตั้งใหม่"
        r"|เริ่ม\s*จด|ควรจด|จะจด|จด\s*ตอนไหน|ต้องสมัคร\s*(vat|ภาษี|ภพ)"
        r"|อยากเปิด|จะเปิด|ต้องการเปิด|อยากตั้ง|จะตั้ง|ต้องการตั้ง"
        r"|เปิดกิจการ|เปิดร้าน|เปิดบริษัท|ตั้งบริษัท|ตั้งกิจการ"
        r"|จะเริ่ม\s*(ทำ|ประกอบ|กิจการ)|อยากเริ่ม\s*(ทำ|ประกอบ|กิจการ)"
        r"|ขอ\s*(ใบ|อนุญาต)\s*(ใหม่|ครั้งแรก|แรก)|สมัคร\s*ครั้งแรก"
        r"|ขึ้นทะเบียน(?!\s*(?:ใหม่\s*)?(?:ต่ออายุ|แก้ไข|ยกเลิก))(?!\s*แล้ว)"
        # Bug 3: bare "ขอใบอนุญาต" / "ยื่นใบอนุญาต" implies new application (no ต่ออายุ/แก้ไข/ยกเลิก suffix)
        r"|(?:ขอ|ยื่น)\s*(?:ขอ\s*)?ใบอนุญาต(?!.{0,25}(?:ต่ออายุ|ต่อ\s*อายุ|แก้ไข|ยกเลิก|ต่อ\s*ใบ))"
        # "ถ้าจะขอ / จะขอ / ต้องการขอ / อยากขอ" — user signals intent to apply new.
        # "ขอ" here means "apply/request", not ask for info ("ขอทราบ" is excluded by negative lookahead).
        # Negative lookahead: exclude cancel/edit/renew modifiers and "ขอทราบ/ขอดู/ขอถาม".
        r"|(?:ถ้า\s*)?(?:จะ|ต้องการ|อยาก)\s*ขอ(?!\s*(?:ยกเลิก|แก้ไข|ต่ออายุ|คืน|โต้แย้ง|ทราบ|ดู|ถาม|อธิบาย|เรียน))",
        re.IGNORECASE,
    )
    _OP_INFER_EDIT_RE = re.compile(
        r"แก้ไข|เปลี่ยนแปลง\s*(รายการ|ที่อยู่|ชื่อ|ข้อมูล|วัตถุประสงค์)"
        r"|อยาก\s*แก้|ต้องการ\s*แก้|จะ\s*แก้|ต้องการ\s*เปลี่ยนแปลง",
        re.IGNORECASE,
    )
    _OP_INFER_CANCEL_RE = re.compile(
        r"ยกเลิก|เลิก\s*กิจการ|ปิด\s*กิจการ|จะ\s*ยกเลิก|ต้องการ\s*ยกเลิก|อยาก\s*ยกเลิก",
        re.IGNORECASE,
    )
    # Renewal / expiry inference: "ต่ออายุ", "มีอายุกี่ปี", "หมดอายุ" etc.
    _OP_INFER_RENEW_RE = re.compile(
        r"ต่ออายุ|ต้องต่อ(?:\s*ใบ|\s*อนุญาต|\s*ทะเบียน)?|ต่อ\s*ใบอนุญาต|ต่อ\s*ทะเบียน"
        r"|(?:ใบ|อนุญาต|ทะเบียน|ภาษี).{0,20}(?:มีอายุ|อายุ.{0,15}กี่|หมดอายุ|สิ้นอายุ)"
        r"|มีอายุ(?:\s*กี่|\s*ปี|\s*เดือน|\s*วัน)?|หมดอายุ|สิ้นอายุ|วันหมดอายุ",
        re.IGNORECASE,
    )
    # Contextual-past guard: operation keyword appears as background context, NOT direct intent.
    # e.g. "หลังจากขึ้นทะเบียนแล้ว ต้องส่งเงินสมทบ" → don't infer "new registration"
    _CONTEXTUAL_PAST_RE = re.compile(
        r"หลังจาก\s*.{0,10}(?:ขึ้นทะเบียน|จดทะเบียน|ยื่นขอ|สมัคร)",
        re.UNICODE,
    )
    # Universal-fact query: answer does NOT differ by entity_type.
    # e.g. "อายุใบทะเบียนพาณิชย์ มีอายุกี่ปี" → same answer for บุคคลธรรมดา and นิติบุคคล.
    # Matched queries skip entity_type / registration_type slot questions.
    _UNIVERSAL_FACT_RE = re.compile(
        r"มีอายุ|หมดอายุ|สิ้นอายุ|วันหมดอายุ"
        r"|ต้องต่ออายุ(?:\s*ไหม|\s*หรือเปล่า|\s*หรือไม่)?"
        r"|(?:ใบ|อนุญาต|ทะเบียน|ภาษี).{0,20}(?:อายุ.{0,15}กี่|มีอายุ|หมดอายุ)"
        # Threshold/amount queries — answer is always the same regardless of entity type
        # e.g. "VAT ต้องจดเมื่อรายได้เท่าไหร่", "รายได้เกินเท่าไหร่ถึงจด VAT"
        r"|รายได้.{0,15}(เท่าไหร่|เท่าไร)"
        r"|ยอดขาย.{0,15}(เท่าไหร่|เท่าไร)"
        r"|รายรับ.{0,15}(เท่าไหร่|เท่าไร)"
        # Require threshold context (ถึง/เกิน/ครบ/ต้องจด) after เมื่อรายได้/ยอดขาย/รายรับ
        # to avoid false match on general income questions like "เมื่อรายได้มาจากหลายแหล่ง"
        r"|เมื่อ(?:รายได้|ยอดขาย|รายรับ).{0,25}(?:ถึง|เกิน|ครบ|ต้องจด|จด\s*ภาษี|จด\s*ภพ)"
        r"|จด.{0,10}เมื่อ(?:รายได้|ยอดขาย|รายรับ)"
        r"|ต้องจด.{0,20}(เท่าไหร่|เท่าไร|เกินเท่า|ถึงเท่า)",
        re.IGNORECASE,
    )
    # Entity type explicit in query text — used for auto-inference without asking
    _ENTITY_NITI_RE = re.compile(
        r"นิติบุคคล|(?:แบบ|เป็น(?:แบบ)?|ประเภท|รูปแบบ|ถ้า(?:เป็น)?|ถ้าแบบ)\s*นิติ|บริษัท(?:\s*จำกัด|\s*มหาชน)?|ห้าง(?:หุ้นส่วน(?:\s*จำกัด|\s*สามัญ)?)?",
        re.IGNORECASE,
    )
    _ENTITY_NATURAL_RE = re.compile(
        r"บุคคลธรรมดา|บุคคล.{0,4}มดา|บุคคลทั่วไป|ร้านค้าทั่วไป|เจ้าของ\s*คนเดียว"
        r"|ทำ\s*คนเดียว|เปิด\s*เอง|ทำ\s*เอง|คนเดียว\s*(เลย|ครับ|ค่ะ|นะ)?"
        r"|ร้านส่วนตัว|กิจการเจ้าของ\s*คนเดียว"
        r"|(?:ถ้า(?:เป็น)?|ถ้าแบบ)\s*(?:บุคคล)?ธรรมดา",
        re.IGNORECASE,
    )
    # Guard: query must contain at least one entity-hint word before calling LLM fallback.
    # Prevents false positives from pronouns (ฉัน/ผม/หนู) or entity-neutral questions.
    # Catches abbreviations (บจก, หจก) and partial entity words the main regex misses.
    _ENTITY_HINT_RE = re.compile(
        r"นิติ|บุคคล|บจก\.?|หจก\.?|บมจ\.?|ห\.?พ\.?|ส่วนตัว|คนเดียว|เจ้าของ|ประเภท(?:นิติ|ธุรกิจ|กิจการ|ผู้ประกอบการ)|รูปแบบ(?:ธุรกิจ|กิจการ|นิติ)",
        re.IGNORECASE,
    )
    # Location auto-infer from query text (Bug 1)
    _LOCATION_BKK_RE = re.compile(
        r"กรุงเทพ(?:ฯ|มหานคร)?|กทม\.?|\bbangkok\b|\bBKK\b|เขต(?:พื้นที่)?กรุงเทพ|ในกรุงเทพ",
        re.IGNORECASE,
    )
    # ปริมณฑล provinces — mapped to กรุงเทพ ONLY when the available option explicitly says "ปริมณฑล"
    # (some licenses treat these as ต่างจังหวัด, so we must check the option list first)
    _LOCATION_METRO_RE = re.compile(
        r"นนทบุรี|ปทุมธานี|สมุทรปราการ|สมุทรสาคร|นครปฐม",
        re.IGNORECASE,
    )
    _LOCATION_PROVINCE_RE = re.compile(
        r"ต่างจังหวัด|ต่างหวัด|จังหวัดอื่น|ต่างพื้นที่|นอกกรุงเทพ",
        re.IGNORECASE,
    )
    # Shop area size auto-infer from query text (Bug 2)
    _AREA_SMALL_RE = re.compile(
        r"(?:พื้นที่|ขนาด|ร้าน).{0,15}(?:น้อยกว่า|ไม่เกิน|<)\s*200"
        r"|น้อยกว่า\s*200\s*ตาราง"
        r"|ไม่เกิน\s*200\s*ตาราง",
        re.IGNORECASE,
    )
    _AREA_LARGE_RE = re.compile(
        r"(?:พื้นที่|ขนาด|ร้าน).{0,15}(?:มากกว่า|เกิน|>)\s*200"
        r"|มากกว่า\s*200\s*ตาราง"
        r"|เกิน\s*200\s*ตาราง",
        re.IGNORECASE,
    )

    # Bank/department lookup — maps abbreviations to canonical Chroma "department" values.
    # Only banks WITH actual documents in Chroma (กสิกร: 8 docs, ไทยพาณิชย์: 5 docs).
    # Other banks intentionally excluded — a pattern match with 0 docs falls back silently.
    _BANK_DEPT_PATTERNS: List[Tuple] = [
        (re.compile(r"ไทยพาณิช(?:ย์)?|ไทยพานิช(?:ย์)?|scb\b", re.IGNORECASE), "ธนาคารไทยพาณิชย์"),
        (re.compile(r"กสิกร|kbank\b|kasikorn", re.IGNORECASE), "ธนาคารกสิกรไทย"),
    ]

    # Slot correction intent — user wants to change a previously given slot answer.
    _SLOT_CORRECTION_RE = re.compile(
        r"ขอแก้(?:ไข)?(?:\s*คำตอบ|\s*ข้อมูล)?|ขอเปลี่ยน(?:\s*คำตอบ)?|เปลี่ยนใหม่"
        r"|ตอบผิด|ไม่ใช่แล้ว|ขอยกเลิกคำตอบ|แก้คำตอบ|เปลี่ยนคำตอบ|ขอแก้ไขที่ตอบ",
        re.IGNORECASE,
    )

    # Rare license types overwhelmed by ใบทะเบียนพาณิชย์ in embedding search.
    # When query explicitly mentions one of these keywords, fetch docs directly by license_type filter.
    # Defined as class attribute — created once at class load, not on every function call.
    _RARE_LT_PATTERNS: Dict[str, str] = {
        r"ประกันสังคม|กองทุนประกันสังคม|ขึ้นทะเบียนประกัน": "การขึ้นทะเบียนกองทุนประกันสังคม",
        r"ใบอนุญาตจำหน่ายสุรา|ขายสุรา|จำหน่ายสุรา|ภส\.08": "ใบอนุญาตจำหน่ายสุรา",
        r"ภาษีป้าย|ป้ายร้าน|ภป(?:\.1)?": "แบบแสดงรายการภาษีป้ายร้านอาหาร",
        r"ใบวุฒิบัตร|อบรมผู้สัมผัสอาหาร": "ใบวุฒิบัตรผู้สัมผัสอาหาร",
        r"ใบรับรองแพทย์|9 โรค|สณ\.11": "ใบรับรองแพทย์ 9 โรค(สณ.11)",
    }

    # Slot question strings — defined once, reused everywhere
    _Q_OPERATION_GROUP = "ต้องการดำเนินการเรื่องใดครับ?"
    _Q_REGISTRATION_TYPE = "รูปแบบการจดทะเบียนของคุณเป็นแบบใดครับ?"

    # Multi-topic license detection: maps regex pattern → canonical license_type name in Chroma.
    # Used by _detect_license_types_from_query() to predict which licenses a query covers BEFORE retrieval.
    _MULTI_TOPIC_LICENSE_KEYWORDS: List[Tuple[str, str]] = [
        (r"vat|ภาษีมูลค่าเพิ่ม|ภพ\.?20|จด\s*ภาษี", "ใบภาษีมูลค่าเพิ่ม ภพ.20"),
        (r"ใบอนุญาตจำหน่ายสุรา|ขายสุรา|จำหน่ายสุรา|สุรา|เหล้า|ภส\.08", "ใบอนุญาตจำหน่ายสุรา"),
        (r"ภาษีป้าย|ป้ายร้าน|ภป(?:\.1)?|ป้ายโฆษณา", "แบบแสดงรายการภาษีป้ายร้านอาหาร"),
        (r"ใบอนุญาตจัดตั้ง|สถานที่จำหน่ายอาหาร|bma\s*oss|อาหาร.*ใบอนุญาต", "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร"),
        (r"ใบทะเบียนพาณ(?:ิ|ี)ชย์|จดทะเบียนพาณ(?:ิ|ี)ชย์|ทะเบียนพาณ(?:ิ|ี)ชย์|dbd", "ใบทะเบียนพาณิชย์"),
        (r"ประกันสังคม|กองทุนประกันสังคม|ขึ้นทะเบียนประกัน", "การขึ้นทะเบียนกองทุนประกันสังคม"),
        (r"ใบวุฒิบัตร|อบรมผู้สัมผัสอาหาร", "ใบวุฒิบัตรผู้สัมผัสอาหาร"),
        (r"ใบรับรองแพทย์|9\s*โรค|สณ\.11", "ใบรับรองแพทย์ 9 โรค(สณ.11)"),
        (r"qr.?payment|qr.?api|พร้อมเพย์|promptpay|qr\s*พร้อมเพย์|ระบบชำระเงินออนไลน์", "ระบบชำระเงินออนไลน์"),
        (r"edc|รูดบัตร|เครื่องรูดบัตร", "เครื่องรูดบัตร EDC"),
        (r"ใบรับรองมาตรฐานร้านอาหาร|มาตรฐานร้านอาหาร|san\b|สุขาภิบาลอาหาร.*มาตรฐาน|มาตรฐาน.*สุขาภิบาล", "ใบรับรองมาตรฐานร้านอาหาร"),
        # บัญชีและการเงิน — accounting / bookkeeping
        (r"ปิดงบ|ปิดบัญชี|ปิงบ|งบการเงิน|งบบัญชี|จัดการการเงิน|จัดการเงิน|ทำบัญชี|ตรวจสอบบัญชี|สำนักงานบัญชี|ผู้ทำบัญชี|ผู้สอบบัญชี", "จัดการการเงิน"),
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
        _LOG.debug("[Supervisor] deflect input=%r", (raw_input or "")[:60])
        try:
            _llm = ChatOpenAI(
                model=getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL),
                openai_api_key=conf.OPENROUTER_API_KEY,
                openai_api_base=conf.OPENROUTER_BASE_URL,
                temperature=0.7,
                max_tokens=200,
                request_timeout=int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30)),
            )
            # Build context string so LLM can ask a relevant clarifying question
            _ctx_parts: list = []
            _last_license = (state.context or {}).get("last_license_type") or (state.context or {}).get("last_topic") or ""
            if _last_license:
                _ctx_parts.append(f"หัวข้อล่าสุด: {_last_license}")
            _pslot = (state.context or {}).get("pending_slot")
            if isinstance(_pslot, dict) and _pslot.get("key"):
                _pkey = _pslot.get("key", "")
                _popts = [str(o) for o in (_pslot.get("options") or [])[:4]]
                _ctx_parts.append(f"รอคำตอบเรื่อง: {_pkey} (ตัวเลือก: {', '.join(_popts)})")
            _ctx_str = "\n".join(_ctx_parts) if _ctx_parts else ""
            _system = (
                "คุณคือ 'น้องสุดยอด' — ที่ปรึกษาธุรกิจร้านอาหาร\n"
                "พูดเป็นกันเอง ตรงไปตรงมา ใช้ 'ผม' ลงท้าย 'ครับ'\n\n"
                + (f"Context ปัจจุบัน:\n{_ctx_str}\n\n" if _ctx_str else "")
                + "กฎเด็ดขาด:\n"
                "- ห้ามทักทายหรือแนะนำตัวซ้ำ (ทักทายไปแล้ว)\n"
                "- ถ้ามี context ให้ถามให้ชัดขึ้นโดยอ้างอิง context นั้น\n"
                "- ถ้ามี pending slot ให้ถามซ้ำเฉพาะ slot นั้น\n"
                "- ถ้าไม่มี context ให้ถามสั้นๆ ว่าต้องการความช่วยเหลือเรื่องอะไร\n"
                "- ตอบเป็น plain text บรรทัดเดียว ไม่เกิน 30 คำ ห้ามขึ้นบรรทัดใหม่\n"
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
            except json.JSONDecodeError as _jde:
                _LOG.warning("[Supervisor/topic_picker] JSON parse failed — text=%r err=%s", text[:120], _jde)
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
            except Exception as _e:
                _LOG.warning("[Supervisor/confirm] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError as _jde:
                _LOG.warning("[Supervisor/confirm] JSON parse failed — text=%r err=%s", text[:120], _jde)
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

        def _call(user_text: str, last_query: str = "") -> dict:
            prompt = build_style_detect_prompt(user_text, last_query)
            try:
                text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/llm")).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/style] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError as _jde:
                _LOG.warning("[Supervisor/style] JSON parse failed — text=%r err=%s", text[:120], _jde)
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
                text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/greet")).strip()
            except Exception as _e:
                _LOG.debug("[Supervisor/greet] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as _je:
                _LOG.debug("[Supervisor/greet] JSON parse failed: %s", _je)
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
                logger.log_with_data("info", "ลบตัวเลือกซ้ำสำเร็จ", {
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
                logger.log_with_data("warning", "ลบตัวเลือกซ้ำล้มเหลว ใช้ข้อมูลเดิม", {
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
                text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/slot_mapper")).strip()
            except Exception as _e:
                _LOG.debug("[Supervisor/slot_mapper] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as _je:
                _LOG.debug("[Supervisor/slot_mapper] JSON parse failed: %s", _je)
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
            except Exception as _e:
                _LOG.debug("[Supervisor/fallback_intent] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as _je:
                _LOG.debug("[Supervisor/fallback_intent] JSON parse failed: %s", _je)
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

    def _looks_like_greeting_or_thanks(self, s: str, state=None) -> bool:
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

        # LLM fallback: catches formal/unusual thanks like "ขอบพระคุณ", "กราบขอบพระคุณ"
        # Only fires when text survived legal/question guards above (safe to classify as thanks).
        if state is not None and self._thanks_llm_check(t, state):
            return True

        # LLM fallback: catches polite social openers like "ยินดีที่ได้รู้จัก", "ขอโทษที่รบกวน"
        # High threshold (0.80) — false positive = ignoring a real question.
        if state is not None and self._greeting_llm_check(t, state):
            return True

        return False

    def _looks_like_legal_question(self, s: str, state=None) -> bool:
        t = self._normalize_for_intent(s)
        if not t:
            return False
        if self._LEGAL_SIGNAL_RE.search(t):
            return True
        if self._QUESTION_MARKERS_RE.search(t) and len(t) >= 6:
            return True
        # LLM fallback: catches "ขอข้อมูลค่าใช้จ่าย", "แนะนำให้ถูกกฎหมายหน่อย", "ต้องติดต่อที่ไหน"
        if state is not None and len(t) >= 4:
            return self._legal_q_llm_check(t, state)
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

    def _default_entity_type_llm_call(self) -> Callable[[str, str], dict]:
        """
        Lightweight LLM fallback for entity_type detection when regex + fuzzy both miss.
        Called only for ambiguous phrasing like "แบบนิติ", "บจก", "ทำคนเดียว".
        Returns {"entity_type": "นิติบุคคล"|"บุคคลธรรมดา"|null, "confidence": float}.
        Uses fast switch model (Haiku) — typically <400ms, ~$0.00025/call.
        """
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=300,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str, last_query: str) -> dict:
            prompt = build_entity_type_detect_prompt(user_text, last_query)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/entity_type")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/entity_type] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as _je:
                _LOG.debug("[Supervisor/entity_type] JSON parse failed: %s", _je)
                return {}

        return _call

    def _default_location_llm_call(self) -> Callable[[str, str], dict]:
        """LLM fallback for location detection (กรุงเทพฯ vs ต่างจังหวัด) when regex misses."""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=150,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str, last_query: str) -> dict:
            prompt = build_location_detect_prompt(user_text, last_query)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/location")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/location] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as _je:
                _LOG.debug("[Supervisor/location] JSON parse failed: %s", _je)
                return {}

        return _call

    def _default_operation_type_llm_call(self) -> Callable[[str, str], dict]:
        """LLM fallback for operation type detection (new/edit/cancel/renew) when regex misses."""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=300,  # was 150 — hit LengthFinishReasonError on longer inputs
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str, last_query: str) -> dict:
            prompt = build_operation_type_detect_prompt(user_text, last_query)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/operation_type")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/operation_type] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as _je:
                _LOG.debug("[Supervisor/operation_type] JSON parse failed: %s", _je)
                return {}

        return _call

    def _default_area_size_llm_call(self) -> Callable[[str], dict]:
        """LLM fallback for shop area size detection when regex + numeric parsing miss."""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=150,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_area_size_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/area_size")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/area_size] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as _je:
                _LOG.debug("[Supervisor/area_size] JSON parse failed: %s", _je)
                return {}

        return _call

    def _default_license_type_llm_call(self) -> Callable[[str, List[str]], dict]:
        """LLM fallback for license type detection — called when all keyword patterns miss."""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=150,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str, candidates: List[str]) -> dict:
            prompt = build_license_type_detect_prompt(user_text, candidates)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/license_type")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/license_type] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_academic_resume_llm_call(self) -> Callable[[str, str, int], dict]:
        """Dedicated LLM classifier for academic resume intent — faster and more precise than general fallback_intent."""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=150,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str, last_academic_query: str, section_count: int) -> dict:
            prompt = build_academic_resume_detect_prompt(user_text, last_academic_query, section_count)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/academic_resume")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/academic_resume] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_elaborate_llm_call(self) -> Callable[[str, str], dict]:
        """Lightweight classifier: does user want MORE detail on the current topic?"""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=300,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str, last_topic: str) -> dict:
            prompt = build_elaborate_detect_prompt(user_text, last_topic)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/elaborate")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/elaborate] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_broad_question_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: is the query a broad overview question?"""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=300,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_broad_question_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/broad_q")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/broad_q] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_slot_skip_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: does user want to skip the current slot question?"""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=300,  # was 150 — hit LengthFinishReasonError on longer inputs
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_slot_skip_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/slot_skip")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/slot_skip] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_followup_contextual_llm_call(self) -> Callable[[str, str], dict]:
        """Lightweight classifier: is this a contextual follow-up about user's own case?"""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=300,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str, last_topic: str) -> dict:
            prompt = build_followup_contextual_detect_prompt(user_text, last_topic)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/followup_ctx")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/followup_ctx] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_thanks_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: is the message a thanks expression?"""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=150,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_thanks_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/thanks")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/thanks] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_academic_stop_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: does user want to stop/exit academic mode?"""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=150,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_academic_stop_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/academic_stop")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/academic_stop] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_smalltalk_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: is this off-topic small talk directed at the bot?"""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=300,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_smalltalk_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/smalltalk")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/smalltalk] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_switch_without_target_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: does user want to change response style without naming a mode?"""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=300,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_switch_without_target_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/switch_no_target")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/switch_no_target] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_link_request_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: is user requesting a link, URL, form, or document download?"""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=300,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_link_request_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/link_request")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/link_request] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_mode_status_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: is user asking which mode/persona the bot is using?"""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=150,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_mode_status_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/mode_status")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/mode_status] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_greeting_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: is this a greeting or polite social opener (not thanks)?"""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=300,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_greeting_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/greeting")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/greeting] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

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
            max_tokens=300,
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

    def _default_info_action_q_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: informational query (answer directly) vs. action-oriented (collect slots)."""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=150,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_info_action_q_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/info_action_q")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/info_action_q] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_legal_q_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: is this a legal/regulatory question about Thai restaurant business?"""
        model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=150,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = build_legal_q_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/legal_q")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Supervisor/legal_q] LLM call failed: %s", _e)
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    # Academic intake lock only when truly active
    _ACADEMIC_LOCK_STAGES = {"awaiting_topic", "awaiting_slots", "awaiting_sections"}

    def _is_academic_intake_active(self, state: ConversationState) -> bool:
        flow = (state.context or {}).get("academic_flow")
        if not isinstance(flow, dict):
            return False
        stage = str(flow.get("stage") or "").strip()
        if not stage:
            return False
        return stage in self._ACADEMIC_LOCK_STAGES

    # Intent classification helpers
    def _looks_like_mode_status_query(self, s: str, state=None) -> bool:
        t = (s or "").strip()
        if not t:
            return False
        if ("โหมด" not in t and "mode" not in t.lower() and "persona" not in t.lower() and "บุคลิก" not in t):
            return False
        if self._MODE_STATUS_Q.search(t):
            return True
        # LLM fallback: catches "บอทใช้โหมดอะไรอยู่", "ตอนนี้เราอยู่โหมดไหน"
        if state is not None:
            return self._mode_status_llm_check(t, state)
        return False

    def _looks_like_switch_without_target(self, s: str, state=None) -> bool:
        t = (s or "").strip().lower()
        if not t:
            return False
        if any(v in t for v in self._SWITCH_VERBS) and any(m in t for m in self._SWITCH_MARKERS):
            if re.search(r"\b(academic|practical)\b|วิชาการ|ละเอียด|สั้น|กระชับ", t):
                return False
            return True
        if re.fullmatch(r"(เปลี่ยน|สลับ|ปรับ)\s*", t):
            return True
        # LLM fallback: catches "ขอแบบอื่น", "เปลี่ยนวิธีตอบหน่อย", "ตอบแบบอื่นได้มั้ย"
        # Guard: skip when text looks like legal content (avoids false positives on legal Q)
        if state is not None and len(t) >= 4 and not self._LEGAL_SIGNAL_RE.search(t):
            return self._switch_without_target_llm_check(s, state)
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

    def _infer_user_style_request_hybrid(self, s: str, last_query: str = "") -> Dict[str, Any]:
        text = s or ""

        # 1. Fast-path: deterministic keyword check (no LLM cost)
        det = self._infer_user_style_request_det(text)
        if det["wants_short"] or det["wants_long"]:
            return {"wants_short": det["wants_short"], "wants_long": det["wants_long"], "method": "det", "confidence": 0.9}

        # 2. Skip LLM for pure noise/number-only inputs (too short to have style intent)
        stripped = text.strip()
        if not stripped or len(stripped) <= 3 or self._LIKELY_SELECTION_RE.match(stripped):
            return {"wants_short": False, "wants_long": False, "method": "none", "confidence": 0.0}

        # 2b. "แล้วถ้าเป็น X" / "ถ้าเป็น X ล่ะ" = follow-up topic switch, NOT a style request.
        # LLM sees prior context (SAN) + "แล้วถ้าเป็น SAN PLUS" and wrongly infers wants_long.
        # These patterns always signal a NEW topic question at the same detail level.
        if re.match(r'^แล้วถ้าเป็น|^แล้วถ้า(?:ต้องการ|จะ(?:ขอ|ดู|สมัคร|ทำ))', stripped):
            return {"wants_short": False, "wants_long": False, "method": "det", "confidence": 0.9}

        # 3. LLM is primary detector for all substantive inputs — catches natural phrasing
        #    that regex misses (e.g. "อธิบายแบบครบๆ", "ขอทราบทั้งหมดเลย", "แบบย่อๆ ได้ไหม")
        #    Pass last_query as context so LLM understands follow-up phrasing correctly.
        res: Dict[str, Any] = {}
        try:
            import inspect
            _style_sig = inspect.signature(self.llm_style_call)
            if len(_style_sig.parameters) >= 2:
                res = self.llm_style_call(text, last_query) or {}
            else:
                res = self.llm_style_call(text) or {}
        except Exception:
            res = {}

        try:
            confv = float(res.get("confidence", 0.0) or 0.0)
        except Exception:
            confv = 0.0

        wants_long = bool(res.get("wants_long", False)) if confv >= 0.65 else False
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

        # Entity-type LLM fallback — runs only when regex + fuzzy both miss
        self._llm_entity_type_call = self._default_entity_type_llm_call()
        # Location / operation-type / area-size LLM fallbacks
        self._llm_location_call = self._default_location_llm_call()
        self._llm_operation_type_call = self._default_operation_type_llm_call()
        self._llm_area_size_call = self._default_area_size_llm_call()
        # License type, academic resume, elaborate, and broad-question LLM fallbacks
        self._llm_license_type_call = self._default_license_type_llm_call()
        self._llm_academic_resume_call = self._default_academic_resume_llm_call()
        self._llm_elaborate_call = self._default_elaborate_llm_call()
        self._llm_broad_q_call = self._default_broad_question_llm_call()
        self._llm_slot_skip_call = self._default_slot_skip_llm_call()
        self._llm_followup_contextual_call = self._default_followup_contextual_llm_call()
        self._llm_thanks_call = self._default_thanks_llm_call()
        self._llm_academic_stop_call = self._default_academic_stop_llm_call()
        self._llm_smalltalk_call = self._default_smalltalk_llm_call()
        self._llm_switch_without_target_call = self._default_switch_without_target_llm_call()
        self._llm_link_request_call = self._default_link_request_llm_call()
        self._llm_mode_status_call = self._default_mode_status_llm_call()
        self._llm_greeting_call = self._default_greeting_llm_call()
        self._llm_info_action_q_call = self._default_info_action_q_llm_call()
        self._llm_legal_q_call = self._default_legal_q_llm_call()

        # NEW: LLM-based deduplication of similar options
        self._deduplicate_options_llm_call = llm_deduplicate_options_call or self._default_deduplicate_options_llm_call()

        # LLM-based operation group classifier (no hardcoded rules)
        # Cache: {(license_type, entity_type_normalized): (slot_options, raw_op_map)}
        # populated lazily on first call per unique key — avoids repeated LLM round-trips
        self._llm_op_group_classifier = self._default_op_group_classifier_llm_call()
        self._op_groups_cache: Dict[tuple, Tuple[List[str], Dict]] = {}

        self._rng = random.Random()

        self._validate_license_types_at_startup()

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

        # ── Dimension-switch guards ──────────────────────────────────────────
        # High Jaccard overlap on shared generic words ("ทะเบียน", "ใบอนุญาต", "เอกสาร")
        # can falsely mark the cache as valid when the user switches a KEY dimension.
        # Each guard below detects a specific dimension change and forces fresh retrieval.

        # 1. Bank/department switch (e.g. KBANK -> SCB, same QR-API topic)
        _BANK_GROUPS = [
            ("scb",   re.compile(r"ไทยพาณิช(?:ย์)?|ไทยพานิช(?:ย์)?|scb\b", re.IGNORECASE)),
            ("bbl",   re.compile(r"ธนาคารกรุงเทพ|bbl\b", re.IGNORECASE)),
            ("ktb",   re.compile(r"กรุงไทย|ktb\b", re.IGNORECASE)),
            ("kbank", re.compile(r"กสิกร|kbank\b|kasikorn", re.IGNORECASE)),
            ("ttb",   re.compile(r"ทหารไทย|ธนชาต|ttb\b", re.IGNORECASE)),
            ("bay",   re.compile(r"กรุงศรี|bay\b", re.IGNORECASE)),
            ("gsb",   re.compile(r"ออมสิน|gsb\b", re.IGNORECASE)),
            ("uob",   re.compile(r"\buob\b", re.IGNORECASE)),
        ]
        def _get_bank(text: str) -> Optional[str]:
            for bid, pat in _BANK_GROUPS:
                if pat.search(text):
                    return bid
            return None
        _q_bank = _get_bank(q)
        if _q_bank and _q_bank != _get_bank(last_q):
            _LOG.info("[Supervisor] bank switch -> %r forcing fresh retrieval", _q_bank)
            return True

        # 2. Operation-type switch (จดทะเบียนใหม่ -> ยกเลิก/ต่ออายุ/แก้ไข)
        def _get_op(text: str) -> Optional[str]:
            if self._OP_INFER_RENEW_RE.search(text):  return "renew"
            if self._OP_INFER_CANCEL_RE.search(text): return "cancel"
            if self._OP_INFER_EDIT_RE.search(text):   return "edit"
            if self._OP_INFER_NEW_RE.search(text):    return "new"
            return None
        _q_op = _get_op(q)
        _last_op = _get_op(last_q)
        if _q_op and _last_op and _q_op != _last_op:
            _LOG.info("[Supervisor] op-type switch %r->%r forcing fresh retrieval", _last_op, _q_op)
            return True

        # 3. Entity-type explicitly stated in query (บริษัทจำกัด -> บุคคลธรรมดา)
        def _get_entity(text: str) -> Optional[str]:
            if self._ENTITY_NITI_RE.search(text):    return "niti"
            if self._ENTITY_NATURAL_RE.search(text): return "natural"
            return None
        _q_ent = _get_entity(q)
        _last_ent = _get_entity(last_q)
        if _q_ent and _last_ent and _q_ent != _last_ent:
            _LOG.info("[Supervisor] entity-type switch %r->%r forcing fresh retrieval", _last_ent, _q_ent)
            return True

        # 4. License-type switch (ใบอนุญาตอาหาร -> ใบอนุญาตสุรา)
        # shared "ใบอนุญาต จำหน่าย เอกสาร" inflates overlap to 0.5+
        _q_lts = self._detect_license_types_from_query(q)
        _last_lts = self._detect_license_types_from_query(last_q)
        if len(_q_lts) == 1 and len(_last_lts) == 1 and _q_lts[0] != _last_lts[0]:
            _LOG.info("[Supervisor] license-type switch %r->%r forcing fresh retrieval", _last_lts[0], _q_lts[0])
            return True

        overlap = self._topic_overlap_ratio(last_q, q)
        return overlap < 0.22

    # Regex: at least one generic follow-up keyword present in the message.
    # Combined with length guard and no-specific-topic check below.
    _GENERIC_FOLLOWUP_KEYWORDS_RE = re.compile(
        r"(ขั้นตอน|การสมัคร|สมัคร|วิธีสมัคร|เอกสาร|ค่าธรรมเนียม|ค่าใช้จ่าย|"
        r"ระยะเวลา|ช่องทางยื่น|ช่องทาง|สถานที่ยื่น|แบบฟอร์ม"
        r"|ลิงก์|ลิงค์|ลิ้งก์|ลิ้งค์|ดาวน์โหลด|"
        r"อยากรู้|อยากทราบ|ต้องใช้|ต้องทำ|ต้องเตรียม|"
        # How-to follow-ups: user asks HOW to do the same process already in context
        # e.g. "วิธีส่งทำไงคะ", "ทำยังไงครับ", "วิธีทำเป็นยังไง"
        r"วิธีส่ง|วิธียื่น|วิธีทำ|วิธีชำระ|วิธีดำเนินการ|"
        r"ทำไง|ทำยังไง|ทำอย่างไร|ยื่นไง|ส่งไง|ส่งยังไง|"
        # Entity-switch follow-ups: user modifies entity_type/condition of the SAME topic
        # e.g. "แล้วถ้าแบบบุคคลธรรมดาล่ะ", "ถ้าเป็นนิติบุคคลล่ะ"
        r"แล้วถ้าแบบ|ถ้าแบบ|ถ้าเป็นแบบ|แล้วถ้าเป็น|แล้วแบบ|ส่วนแบบ|"
        # Op-type follow-ups: user asks about different operation on SAME topic
        # e.g. "แล้วถ้าจะยกเลิก", "แล้วถ้าจะต่ออายุ", "แล้วถ้าจะแก้ไข"
        r"แล้วถ้าจะ)",
        re.IGNORECASE,
    )
    # Regex: specific topic/license keywords that indicate the user is asking about a NEW topic.
    # If present, this is NOT a generic follow-up.
    _SPECIFIC_TOPIC_RE = re.compile(
        r"(ใบอนุญาต|จดทะเบียน|ภาษีมูลค่าเพิ่ม|ภาษีป้าย|vat|ภพ\.?20|สุรา|เหล้า|ประกันสังคม|ป้ายร้าน|"
        r"ประกอบกิจการ|สถานที่จำหน่าย|ทะเบียนพาณิชย์|บริษัทจำกัด|ห้างหุ้นส่วน|qr.?pay|qr.?api|payment api|"
        r"บัญชีธนาคาร|หาพนักงาน|edc|รูดบัตร|คิวอาร์เพย์|คิวอาเพ|คิวอาเอพีไอ|ระบบชำระเงิน|"
        # Bank / department names: explicit bank mention means user is asking about that specific bank,
        # NOT a generic follow-up to the previous topic (even if using link/step keywords).
        r"กสิกร|kbank|ไทยพาณิช(?:ย์)?|ไทยพานิช|scb|กรุงไทย|ktb|กรุงเทพ|bbl|ออมสิน|gsb|ทหารไทย|ttb|ธ\.ก\.ส|kkp|"
        # Accounting / bookkeeping — distinct enough to always be a new topic
        r"สำนักงานบัญชี|ปิดงบ|งบบัญชี|งบการเงิน|ปิดบัญชี|ปิงบ|ตรวจสอบบัญชี|"
        r"ทำบัญชี|นักบัญชี|ผู้สอบบัญชี|ผู้ทำบัญชี|จัดทำบัญชี|"
        # HR / staffing topics — "สมัคร" in "รับสมัครงาน" must NOT anchor to prior topic
        r"พนักงาน|รับสมัครงาน|ตำแหน่งงาน|ตำแหน่งรับสมัคร|จัดหาพนักงาน|ไทยมีงานทำ|ประกาศงาน|ลูกจ้าง)",
        re.IGNORECASE,
    )
    # Channel-preference follow-up: user wants offline/in-person instead of online.
    # Matches: "ไม่อยากยื่นออนไลน์", "ไม่ต้องการออนไลน์", "ยื่นด้วยตนเอง",
    #          "แบบออฟไลน์", "ไปสำนักงาน", "ยื่นที่สำนักงาน", "ยื่นด้วยตัวเอง"
    _CHANNEL_PREF_OFFLINE_RE = re.compile(
        r"(?:ไม่.{0,10}(?:อยาก|ต้องการ|ใช้).{0,10}(?:ออนไลน์|อินเทอร์เน็ต|เน็ต))"
        r"|(?:ออฟไลน์|offline)"
        r"|(?:ยื่น.{0,8}(?:ด้วยตนเอง|ด้วยตัวเอง|ที่สำนักงาน|ที่เขต|ที่อำเภอ|ที่อบต|ที่เทศบาล))"
        r"|(?:ไป.{0,8}(?:สำนักงาน|ยื่น|สาขา|เขต|อำเภอ|เทศบาล|อบต))",
        re.IGNORECASE,
    )

    def _detect_reg_subtype_from_query(
        self, q_orig: str, state: ConversationState
    ) -> Optional[str]:
        """
        Dynamically find a registration_type value (from Chroma) that matches the user's query.
        Looks up all registration_type values for the current license_type, filters to
        business-entity sub-types only (excludes area-size / role descriptions), then returns
        the longest value that is a substring of q_orig — or None if no match.
        """
        docs = getattr(state, "current_docs", None) or []
        license_types: set = set()
        for d in docs:
            md = (d.get("metadata") or {}) if isinstance(d, dict) else (getattr(d, "metadata", {}) or {})
            lt = (md.get("license_type") or "").strip()
            if lt and lt not in ("nan", "None"):
                license_types.add(lt)
        if not license_types:
            return None

        _BIZTYPE_KW = ("บริษัท", "ห้างหุ้นส่วน", "ห้าง", "สามัญ", "จำกัด", "มหาชน")
        _BAD_KW = ("ตารางเมตร", "ผู้สัมผัส", "ผู้ประกอบ", "มากกว่า", "น้อยกว่า", "ตร.ม", "•")

        reg_candidates: set = set()
        try:
            vstore = getattr(self.retriever, "vectorstore", None)
            coll = getattr(vstore, "_collection", None) if vstore else None
            if coll is None:
                return None
            for lt in license_types:
                r = coll.get(where={"license_type": lt}, include=["metadatas"])
                for md in r.get("metadatas") or []:
                    rt = (md or {}).get("registration_type", "").strip()
                    if not rt or rt in ("nan", "None"):
                        continue
                    if any(kw in rt for kw in _BIZTYPE_KW) and not any(bad in rt for bad in _BAD_KW):
                        reg_candidates.add(rt)
        except Exception as _e:
            _LOG.debug("[Supervisor] _detect_reg_subtype failed: %s", _e)
            return None

        if not reg_candidates:
            return None

        # Longest match wins (e.g. "ห้างหุ้นส่วนจำกัด" beats "ห้างหุ้นส่วน" for the same query)
        for rt in sorted(reg_candidates, key=len, reverse=True):
            if rt in q_orig:
                # Promote short form to a better-matching longer form using priority rules:
                #   Priority 0 (best)  — numbered multi-type "1.XXX 2.YYY": these are the
                #     actual procedure docs with full form lists, steps, and URLs.
                #     e.g. "1.ห้างหุ้นส่วนจำกัด 2.ห้างหุ้นส่วนสามัญ" covers the 8-form
                #     registration procedure and is preferred over the generic naming doc.
                #   Priority 1 (mid)   — single specific type without "และ"
                #     e.g. "บริษัทจำกัด", "ห้างหุ้นส่วนจำกัด/สามัญนิติบุคคล (หสน.)"
                #   Priority 2 (worst) — catch-all "X และ Y": generic docs that cover naming
                #     rules rather than procedure steps.
                #     e.g. "ห้างหุ้นส่วนและบริษัท" → only 1 naming doc, no forms.
                # Also include multi-value RT fields that CONTAIN rt (e.g. "1.ห้างหุ้นส่วนจำกัด
                # 2.ห้างหุ้นส่วนสามัญ" contains "ห้างหุ้นส่วน" but starts with "1.") so that
                # the detailed procedure doc is not excluded by a startswith check alone.
                def _ext_priority(c: str) -> int:
                    if re.match(r"^\d+\.", c):
                        return 0
                    if "และ" in c:
                        return 2
                    return 1

                extensions = sorted(
                    [c for c in reg_candidates if c != rt and (c.startswith(rt) or rt in c)],
                    key=_ext_priority,
                )
                return extensions[0] if extensions else rt

        # Partial keyword match: user said "ห้างหุ้นส่วน" without specifying จำกัด/สามัญ.
        # practical.py link filter uses substring check (_cs_rt not in doc_rt), so storing
        # "ห้างหุ้นส่วน" correctly allows ห้างหุ้นส่วนจำกัด/สามัญ doc links while blocking
        # บริษัทจำกัด links.  Chroma exact-match filter won't hit → entity-only fallback in
        # _apply_slot_change_if_detected handles retrieval correctly.
        for _partial_kw in ("ห้างหุ้นส่วน",):
            if _partial_kw in q_orig and any(_partial_kw in c for c in reg_candidates):
                return _partial_kw
        return None

    def _apply_slot_change_if_detected(
        self, state: ConversationState, q_orig: str
    ) -> bool:
        """
        Unified slot-change handler — detects entity_type, department (bank), and
        registration_type changes from the ORIGINAL (unanchored) user query.

        When any slot changed vs collected_slots:
          1. Updates collected_slots for all changed slots
          2. Builds a combined Chroma filter (entity + dept + reg — all known values)
          3. Re-retrieves docs using the active topic query + that filter
          4. Verifies at least one doc matches a changed slot (guard against filter fallback)
          5. Sets state.current_docs and returns True

        Must be called BEFORE _should_retrieve_new_for_practical so that Jaccard
        false-positives from the anchored query (which still contains the OLD slot value)
        don't cause a premature cache-hit return.

        Returns False if no slot change detected or if filtered retrieval yields no
        matching docs (caller falls through to normal retrieval paths).
        """
        cs = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}

        old_entity = (cs.get("entity_type_normalized") or cs.get("entity_type") or "").strip()
        old_dept   = (cs.get("department") or "").strip()
        old_reg    = (cs.get("registration_type") or "").strip()

        # Fallback: infer old_dept from last_retrieval_query when collected_slots doesn't have it.
        # Handles the case where department was used in a prior retrieval but wasn't saved to slots
        # (e.g., after an entity-switch flow cleared or skipped slot persistence).
        if not old_dept:
            _lrq = str(getattr(state, "last_retrieval_query", "") or "").strip()
            for _bp, _bn in self._BANK_DEPT_PATTERNS:
                if _bp.search(_lrq):
                    old_dept = _bn
                    break

        # ── Detect entity_type ────────────────────────────────────────────────
        new_entity: Optional[str] = None
        if self._ENTITY_NATURAL_RE.search(q_orig):
            new_entity = "บุคคลธรรมดา"
        elif self._ENTITY_NITI_RE.search(q_orig):
            new_entity = "นิติบุคคล"
        else:
            # Fuzzy fallback: regex missed due to typo (e.g. "บุคคละรรมดา" — ธ→ะ).
            # Slide a window over the Thai chars in the query and compare edit-distance
            # against known entity strings. Catches any 1-2 char substitution/deletion
            # without needing to enumerate every possible typo in the regex.
            from difflib import SequenceMatcher
            _q_thai = re.sub(r"[^฀-๿]", "", q_orig)
            _ENTITY_FUZZY_TARGETS = [
                ("บุคคลธรรมดา", "บุคคลธรรมดา"),
                ("นิติบุคคล", "นิติบุคคล"),
            ]
            for _ft, _fe in _ENTITY_FUZZY_TARGETS:
                _tlen = len(_ft)
                _best = 0.0
                for _i in range(max(1, len(_q_thai) - _tlen + 2)):
                    _window = _q_thai[_i : _i + _tlen + 1]
                    _r = SequenceMatcher(None, _window, _ft).ratio()
                    if _r > _best:
                        _best = _r
                if _best >= 0.80:
                    new_entity = _fe
                    _LOG.info(
                        "[Supervisor] entity fuzzy match: %r → %r (ratio=%.2f)",
                        _q_thai, new_entity, _best,
                    )
                    break

        # ── LLM fallback (regex + fuzzy both missed) ──────────────────────────
        # Only fires when the query contains at least one entity-hint word (บจก, นิติ,
        # เจ้าของ, คนเดียว, etc.) that the main regex couldn't parse. This guard prevents
        # false positives from pronouns (ฉัน/ผม/หนู) or entity-neutral questions where the
        # LLM would over-infer บุคคลธรรมดา from a first-person pronoun.
        if new_entity is None and self._ENTITY_HINT_RE.search(q_orig):
            new_entity = self._entity_type_llm_fallback(q_orig, state)

        # ── Detect department (bank) ──────────────────────────────────────────
        new_dept: Optional[str] = None
        for _bp, _bn in self._BANK_DEPT_PATTERNS:
            if _bp.search(q_orig):
                new_dept = _bn
                break

        # ── Detect registration sub-type (dynamic — values from Chroma) ───────
        # Only relevant when effective entity is/will be นิติบุคคล
        new_reg: Optional[str] = None
        eff_entity_for_reg = new_entity if new_entity else old_entity
        if eff_entity_for_reg == "นิติบุคคล":
            new_reg = self._detect_reg_subtype_from_query(q_orig, state)

        # ── Detect location switch (กรุงเทพฯ ↔ ต่างจังหวัด) ────────────────────
        # Only in ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร but handled generically.
        old_location = (cs.get("location") or "").strip()
        new_location: Optional[str] = None
        if self._LOCATION_BKK_RE.search(q_orig):
            new_location = "กรุงเทพฯ"
        elif self._LOCATION_PROVINCE_RE.search(q_orig):
            new_location = "ต่างจังหวัด"
        # LLM fallback: catches province names, landmark mentions, etc. that regex misses
        if new_location is None and len(q_orig.strip()) >= 2:
            new_location = self._location_llm_fallback(q_orig, state)
        location_changed = bool(new_location and old_location and new_location != old_location)

        # ── Detect shop_area_type switch (<200 ↔ >200 ตรม) ───────────────────
        # Chroma area_size field is not a useful filter (all ใบอนุญาตจัดตั้ง docs share same value).
        # We only update the slot so practical's _area_hint prompt changes accordingly.
        old_area = (cs.get("shop_area_type") or "").strip()
        new_area: Optional[str] = None
        if self._AREA_SMALL_RE.search(q_orig):
            new_area = "น้อยกว่า 200 ตารางเมตร"
        elif self._AREA_LARGE_RE.search(q_orig):
            new_area = "มากกว่า 200 ตารางเมตร"
        if new_area is None:  # numeric: "234 ตรม", "8.9 ตร.ม.", "300 ตารางเมตร"
            _sqm_m = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:ตร\.?\s*ม\.?|ตารางเมตร)', q_orig)
            if _sqm_m:
                try:
                    _sqm_val = float(_sqm_m.group(1).replace(",", "."))
                    new_area = "น้อยกว่า 200 ตารางเมตร" if _sqm_val < 200 else "มากกว่า 200 ตารางเมตร"
                except ValueError:
                    pass
        # LLM fallback: catches descriptive phrasing like "ร้านเล็กมาก", "พื้นที่กว้างมาก"
        if new_area is None and old_area and len(q_orig.strip()) >= 2:
            new_area = self._area_size_llm_fallback(q_orig, state)
        area_changed = bool(new_area and old_area and new_area != old_area)

        # ── Determine which slots actually changed ────────────────────────────
        entity_changed = bool(new_entity and old_entity and new_entity != old_entity)
        # Bank: trigger on change OR first-explicit-mention after an answer already exists
        dept_changed = bool(
            new_dept and new_dept != old_dept
            and (old_dept or bool(getattr(state, "current_docs", None)))
        )
        reg_changed = bool(new_reg and old_reg and new_reg != old_reg)
        # Also trigger if entity switches to นิติบุคคล and specific sub-type was stated
        reg_first_with_entity_change = bool(new_reg and entity_changed and not old_reg)

        if not (entity_changed or dept_changed or reg_changed or reg_first_with_entity_change or location_changed):
            # area_changed alone: just update the hint slot, no retrieval needed
            # (same docs cover both area sizes — content differentiates, not separate docs)
            if area_changed and hasattr(state, "save_collected_slot"):
                state.save_collected_slot("shop_area_type", new_area)
                _LOG.info("[Supervisor] slot-change: area_type %r→%r (hint-only, no retrieval)", old_area, new_area)
            return False

        # ── Apply slot updates ────────────────────────────────────────────────
        if hasattr(state, "save_collected_slot"):
            if entity_changed:
                state.save_collected_slot("entity_type", new_entity)
                state.save_collected_slot("entity_type_normalized", new_entity)
                # Switching to บุคคลธรรมดา: clear reg sub-type (not applicable)
                if new_entity == "บุคคลธรรมดา":
                    state.save_collected_slot("registration_type", "")
            if dept_changed:
                state.save_collected_slot("department", new_dept)
            if new_reg:
                state.save_collected_slot("registration_type", new_reg)
            if location_changed:
                state.save_collected_slot("location", new_location)
            if area_changed:
                state.save_collected_slot("shop_area_type", new_area)

        # ── Build combined Chroma filter ──────────────────────────────────────
        eff_entity = new_entity if entity_changed else old_entity
        eff_dept   = new_dept  if dept_changed   else old_dept
        # Validate old_dept against current docs — drop if it's from a different topic/license.
        # e.g. dept='ธนาคารกสิกรไทย' collected during QR Payment must not leak into
        # ใบทะเบียนพาณิชย์ filter where no such dept exists → causes 0-doc fallback.
        if not dept_changed and eff_dept:
            _cur_depts = {
                (d.get("metadata") or {}).get("department", "").strip()
                for d in (getattr(state, "current_docs", None) or [])
                if isinstance(d, dict)
            }
            if eff_dept not in _cur_depts:
                _LOG.info("[Supervisor] stale dept=%r not in current docs — dropping from filter", eff_dept)
                eff_dept = ""

        # When only dept changed and entity_type was never explicitly stated (eff_entity=''),
        # infer entity_type from previous current_docs so the new retrieval stays consistent.
        # e.g. SCB returned บุคคลธรรมดา docs → user says "switch to กสิกร" → keep บุคคลธรรมดา filter.
        if dept_changed and not entity_changed and not eff_entity and getattr(state, "current_docs", None):
            _prev_entity_set: set = set()
            for _pd in state.current_docs:
                _pm = (_pd.get("metadata") or {}) if isinstance(_pd, dict) else (getattr(_pd, "metadata", {}) or {})
                _pet = _pm.get("entity_type_normalized", "").strip()
                if _pet:
                    _prev_entity_set.add(_pet)
            if len(_prev_entity_set) == 1:
                eff_entity = _prev_entity_set.pop()
        # For reg: use new_reg if any reg change detected; keep old_reg only when still นิติบุคคล
        eff_reg: Optional[str] = None
        if new_reg:
            eff_reg = new_reg
        elif old_reg and eff_entity != "บุคคลธรรมดา":
            eff_reg = old_reg

        # ── Choose filter strategy based on what changed ──────────────────────
        # Location-only switch: use ONLY location filter.
        # Chroma data has no entity-specific ต่างจังหวัด docs → combining entity+location = 0 docs.
        # For entity/dept/reg changes: include all known slot values in combined filter.
        if location_changed and not entity_changed and not dept_changed and not reg_changed and not reg_first_with_entity_change:
            chroma_filter: Dict[str, Any] = {"location": new_location}
        else:
            filter_parts: List[Dict[str, Any]] = []
            if eff_entity:
                filter_parts.append({"entity_type_normalized": eff_entity})
            if eff_dept:
                filter_parts.append({"department": eff_dept})
            if eff_reg:
                filter_parts.append({"registration_type": eff_reg})
            if location_changed:  # location + other dimension change together
                filter_parts.append({"location": new_location})
            if not filter_parts:
                return False
            chroma_filter = (
                filter_parts[0] if len(filter_parts) == 1 else {"$and": filter_parts}
            )

        # ── Re-retrieve using active topic query (not the slot-change phrase) ─
        # Normally last_retrieval_query is the anchor (last_user_legal_query is overwritten with the
        # current input before this method runs). Exception: when q_orig is itself a new topic question
        # about a *different operation* (e.g. "หากต้องการแก้ไข..." after a prior "จดทะเบียน" query),
        # using last_retrieval_query would search the OLD operation and return unrelated docs. In that
        # case prefer q_orig so the filter lands on docs for the new operation + new entity.
        _OP_CHANGE_IN_Q = bool(re.search(
            r"แก้ไข|เปลี่ยนแปลง|ยกเลิก|ปิดกิจการ|จดเลิก|ต่ออายุ|ชำรุด|สูญหาย|ใบแทน",
            q_orig or "",
        ))
        _q_is_fresh_topic = _OP_CHANGE_IN_Q and len(q_orig) > 8
        _last_rq = str(getattr(state, "last_retrieval_query", "") or "").strip()
        _last_lq = str((state.context or {}).get("last_user_legal_query") or "").strip()
        # When department changed and last_retrieval_query is anchored to the old department's
        # topic (e.g. old กสิกรไทย query still in last_retrieval_query after bank switch),
        # prefer last_user_legal_query to avoid dragging the wrong topic into the new retrieval.
        if dept_changed and old_dept and old_dept in _last_rq and _last_lq:
            base_q = _last_lq
        elif _q_is_fresh_topic or (reg_changed and not entity_changed):
            # reg switch: q_orig contains the explicit RT keyword — use it so semantic
            # search finds the correct sub-type procedure docs (e.g. ห้างหุ้นส่วน) rather than
            # the old operation's docs anchored in last_retrieval_query.
            base_q = q_orig
        else:
            base_q = _last_rq or q_orig

        docs = self._practical._retrieve_docs(base_q, metadata_filter=chroma_filter)

        # ── Fallback when RT filter returns too few docs ──────────────────────
        # After Fix 3, new_reg may be a specific 1-doc procedure RT (e.g.
        # "1.ห้างหุ้นส่วนจำกัด 2.ห้างหุ้นส่วนสามัญ"). That single doc has all 8 forms
        # and URLs, but semantic ranking benefits from more context docs alongside it.
        # Entity-only filter (71 นิติบุคคล docs) gives semantic search enough pool to
        # surface the procedure doc + related procedure/fee docs in top-4.
        _used_entity_fallback = False
        if len(docs) < 3 and (reg_changed or reg_first_with_entity_change) and eff_entity:
            _n_rt_docs = len(docs)
            _entity_only_fb = {"entity_type_normalized": eff_entity}
            _fb_docs = self._practical._retrieve_docs(base_q, metadata_filter=_entity_only_fb)
            if len(_fb_docs) > _n_rt_docs:
                docs = _fb_docs
                _used_entity_fallback = True
                _LOG.info(
                    "[Supervisor] RT filter returned only %d docs → entity-only fallback: %d docs",
                    _n_rt_docs, len(docs),
                )

        # ── Fallback when combined filter yields 0 docs ───────────────────────
        # Data gap: no entity-specific ต่างจังหวัด docs exist (only entity-neutral Doc 2).
        # Combined entity+location filter → 0 docs → verified=False → would return False.
        # Fix: retry with individual filters in priority order: entity → location → unfiltered.
        if not docs and entity_changed and location_changed:
            _fb_entity_filter: Dict[str, Any] = {"entity_type_normalized": eff_entity}
            docs = self._practical._retrieve_docs(base_q, metadata_filter=_fb_entity_filter)
            if not docs:
                docs = self._practical._retrieve_docs(base_q, metadata_filter={"location": new_location})
            if docs:
                _LOG.info(
                    "[Supervisor] entity+location combined filter returned 0 — used fallback filter, got %d docs",
                    len(docs),
                )

        if not docs:
            # Final fallback: all filtered attempts returned 0 — use unfiltered retrieval so
            # state.current_docs is refreshed (not left with stale previous-entity docs).
            docs = self._practical._retrieve_docs(base_q)
            if docs:
                _LOG.info("[Supervisor] slot-change all filters 0 — used unfiltered fallback (%d docs)", len(docs))
            else:
                return False

        # ── Verify at least one doc matches a changed slot ────────────────────
        def _meta(d: Any) -> Dict:
            return (d.get("metadata") or {}) if isinstance(d, dict) else (getattr(d, "metadata", {}) or {})

        verified = False
        if entity_changed:
            verified = any(_meta(d).get("entity_type_normalized") == new_entity for d in docs)
        elif dept_changed:
            verified = any(_meta(d).get("department") == new_dept for d in docs)
        elif reg_changed or reg_first_with_entity_change:
            verified = any(_meta(d).get("registration_type") == new_reg for d in docs)
            # Entity-only fallback: docs are now from a broader entity pool, so the specific
            # new_reg RT may not appear in top-k results even though the correct procedure doc
            # is present. Accept entity match as verification — the LLM's semantic ranking
            # will surface the right procedure doc from within the entity-filtered pool.
            if not verified and _used_entity_fallback and eff_entity:
                verified = any(_meta(d).get("entity_type_normalized") == eff_entity for d in docs)
        elif location_changed:
            verified = any(_meta(d).get("location") == new_location for d in docs)

        if not verified:
            return False

        state.current_docs = docs
        # Update retrieval anchors so subsequent calls (forced fallback, etc.) use the correct query.
        state.last_retrieval_query = base_q
        if entity_changed or dept_changed or location_changed:
            state.context["last_user_legal_query"] = base_q
        # Signal to practical.handle() that entity-switch re-retrieval was already done here —
        # prevents the practical-layer entity-switch block from doing a redundant second retrieval.
        if entity_changed and new_entity:
            state.context["_entity_switch_done"] = new_entity

        changed_desc = ", ".join(filter(None, [
            f"entity={old_entity}→{new_entity}"                if entity_changed else None,
            f"dept={'∅' if not old_dept else old_dept}→{new_dept}" if dept_changed else None,
            f"reg={old_reg or '∅'}→{new_reg}"                  if (reg_changed or reg_first_with_entity_change) else None,
            f"loc={old_location}→{new_location}"               if location_changed else None,
        ]))
        _LOG.info(
            "[Supervisor] slot-change retrieval: %s | filter=%r | query=%r | docs=%d",
            changed_desc, chroma_filter, base_q[:40], len(docs),
        )
        return True

    def _ensure_practical_retrieval_for_legal(self, state: ConversationState, user_input: str) -> None:
        state.context = state.context or {}
        q = (user_input or "").strip()
        if not q:
            return

        # Clear ephemeral flags: valid only for the single turn they were set.
        # Any new legal question must start fresh — flags will be re-set below if still relevant.
        state.context.pop("_obd_entity_override", None)
        state.context.pop("_multi_topic_retrieval", None)
        state.context.pop("_entity_switch_done", None)
        # ISSUE 4: multi_license_topics must be cleared at start of every new retrieval, not just
        # on action='answer' path — it persists through slot-fill/DC early returns otherwise.
        state.context.pop("multi_license_topics", None)
        # ISSUE 3: _broad_retrieval_docs set when switching to Academic is stale by the time
        # Practical runs again — clear to prevent unbounded context growth.
        state.context.pop("_broad_retrieval_docs", None)
        state.context.pop("_broad_retrieval_query", None)
        # Belt-and-suspenders: clear _direct_topic_match here too (also cleared in _handle_inner)
        # so that any code path calling this function directly cannot see a stale flag from a
        # prior turn's chapter retrieval hit.
        state.context.pop("_direct_topic_match", None)

        # Step 0: operation_topic chapter retrieval — if query is an exact operation_topic name,
        # bypass semantic search entirely and serve the exact doc directly.
        # Must run AFTER clearing _direct_topic_match so we re-set it cleanly.
        def _topic_segs_in_query(ot_lower: str, q_lower: str) -> bool:
            """Word-overlap match: split topic by Thai connectors; all key segments (≥5 chars)
            must appear in the query. Enables matching 'การจดทะเบียนพาณิชย์ที่ได้รับการยกเว้น'
            against 'ธุรกิจแบบไหนจะได้รับการยกเว้นในการจดทะเบียนพาณิชย์' even though word
            order differs and straight substring fails."""
            segs = re.split(r'ที่|ซึ่ง|และ|หรือ|โดย|กรณี', ot_lower)
            key_segs = [s.strip() for s in segs if len(s.strip()) >= 5]
            return len(key_segs) >= 2 and all(s in q_lower for s in key_segs)

        try:
            _ot_candidates_e = self._get_all_operation_topics_from_store()
            _q_lower_e = q.lower()
            _matched_ots_e = [
                ot for ot in _ot_candidates_e
                if len(ot) >= 8 and (
                    ot.lower() == _q_lower_e
                    or ot.lower() in _q_lower_e
                    or _q_lower_e in ot.lower()
                    or _topic_segs_in_query(ot.lower(), _q_lower_e)
                )
            ]
            if _matched_ots_e:
                # Prefer exact match over substring match — prevents "รับชำระภาษีป้าย"
                # from being overtaken by the longer "ยกเลิกรับชำระภาษีป้าย" via max(len).
                _exact_ots_e = [ot for ot in _matched_ots_e if ot.lower() == _q_lower_e]
                if _exact_ots_e:
                    _target_ot_e = _exact_ots_e[0]
                else:
                    # Group A: topic ⊆ query (topic is substring of query)
                    # Group C: word-overlap (all key segments in query, different order)
                    # Both A+C: prefer LONGEST — most specific topic that covers the query intent
                    #   e.g. query has 'การจดทะเบียนพาณิชย์' + 'ยกเว้น'
                    #   → 'การจดทะเบียนพาณิชย์ที่ได้รับการยกเว้น' (Group C, 31 chars) beats
                    #      'การจดทะเบียนพาณิชย์' (Group A, 21 chars)
                    # Group B: query ⊆ topic (topic broader than query) → prefer SHORTEST (most general)
                    _group_a = [ot for ot in _matched_ots_e if ot.lower() in _q_lower_e]
                    _group_c = [
                        ot for ot in _matched_ots_e
                        if ot.lower() not in _q_lower_e and _q_lower_e not in ot.lower()
                    ]
                    _group_ac = _group_a + _group_c
                    _target_ot_e = (
                        max(_group_ac, key=len)
                        if _group_ac
                        else min(_matched_ots_e, key=len)
                    )
                _coll_e = self._get_chroma_collection()
                if _coll_e is not None:
                    _ot_res_e = _coll_e.get(
                        where={"operation_topic": _target_ot_e},
                        include=["documents", "metadatas"],
                    )
                    _ot_max_e = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                    _ot_docs_e = [
                        {"content": (c or "")[:_ot_max_e], "metadata": m or {}}
                        for c, m in zip(
                            _ot_res_e.get("documents") or [],
                            _ot_res_e.get("metadatas") or [],
                        )
                    ]
                    if _ot_docs_e:
                        # If query also contains a license_type keyword, filter docs to that
                        # license only — prevents "แก้ไข เปลี่ยนแปลง ใบทะเบียนพาณิชย์" from
                        # returning docs from ภาษีป้าย / สรรพสามิต etc.
                        _q_lower_lt = q.lower()
                        _matched_lt_e: Optional[str] = None
                        for _lt_pat, _lt_name in self._MULTI_TOPIC_LICENSE_KEYWORDS:
                            if re.search(_lt_pat, _q_lower_lt, re.IGNORECASE):
                                _matched_lt_e = _lt_name
                                break
                        _ot_skip_return = False
                        _ot_was_group_b_match = not _exact_ots_e and not _group_a
                        if _matched_lt_e:
                            _ot_docs_filtered = [
                                d for d in _ot_docs_e
                                if str(d.get("metadata", {}).get("license_type") or "").strip() == _matched_lt_e
                            ]
                            if _ot_docs_filtered:
                                _ot_docs_e = _ot_docs_filtered
                                _LOG.info(
                                    "[Supervisor] _ensure: op_topic license-filter %r → %d docs",
                                    _matched_lt_e, len(_ot_docs_e),
                                )
                            else:
                                # license keyword found but no matching docs → fall through to
                                # normal retrieval with the full query (including license name)
                                _LOG.info(
                                    "[Supervisor] _ensure: op_topic license-filter %r → 0 docs, falling through",
                                    _matched_lt_e,
                                )
                                _ot_skip_return = True
                        # Group B match (query ⊆ topic, not topic ⊆ query) with ≤1 doc is a weak/
                        # ambiguous match. Fall through to Step 0b (OBD exact match) which may find
                        # the correct parent topic. Example: "จดทะเบียนพาณิชย์" matches the niche
                        # sub-topic "การจดทะเบียนพาณิชย์ที่ได้รับการยกเว้น" (1 doc) while the
                        # real answer is the OBD "การจดทะเบียนพาณิชย์" with many docs.
                        if not _ot_skip_return and _ot_was_group_b_match and len(_ot_docs_e) <= 1:
                            _LOG.info(
                                "[Supervisor] _ensure: op_topic group-B match %r → only %d doc(s), falling through to Step 0b",
                                _target_ot_e, len(_ot_docs_e),
                            )
                            _ot_skip_return = True
                        if not _ot_skip_return:
                            # Entity variety: same check as OBD path — inject entity_type slot
                            # when docs cover both บุคคลธรรมดา and นิติบุคคล and entity not yet known.
                            # _direct_topic_match blocks the check inside Practical (line 3742).
                            _ot_session_et = str(
                                (state.context.get("collected_slots") or {}).get("entity_type") or ""
                            ).strip()
                            if not _ot_session_et:
                                _et_in_ot = {
                                    str(d.get("metadata", {}).get("entity_type_normalized") or "").strip()
                                    for d in _ot_docs_e
                                } - {"", "nan", "None"}
                                # Fallback: combined-entity doc (entity_type_normalized empty) whose
                                # identification_documents has both types as numbered-list section headers.
                                _ot_raw_metas = _ot_res_e.get("metadatas") or []
                                _ot_combined_content = any(
                                    re.search(r"บุคคลธรรมดา\s*\d+\.", str((raw_m or {}).get("identification_documents") or ""))
                                    and re.search(r"นิติบุคคล\s*\d+\.", str((raw_m or {}).get("identification_documents") or ""))
                                    for raw_m in _ot_raw_metas
                                )
                                _ot_et_mixed = (
                                    ("นิติบุคคล" in _et_in_ot and "บุคคลธรรมดา" in _et_in_ot)
                                    or _ot_combined_content
                                )
                                if _ot_et_mixed:
                                    _sq_ot = list((state.context or {}).get("topic_slot_queue") or [])
                                    _et_already_known_ot = bool(
                                        (state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}).get("entity_type")
                                    )
                                    if not _et_already_known_ot and not any(s.get("key") == "entity_type" for s in _sq_ot):
                                        _et_sorted_ot = sorted(_et_in_ot) if len(_et_in_ot) >= 2 else ["นิติบุคคล", "บุคคลธรรมดา"]
                                        state.context["topic_slot_queue"] = [{
                                            "key": "entity_type",
                                            "options": _et_sorted_ot,
                                            "display_options": _et_sorted_ot,
                                            "question": "ร้านของคุณดำเนินการในรูปแบบใดครับ",
                                        }] + _sq_ot
                                        _LOG.info(
                                            "[Supervisor] _ensure: op_topic %r has mixed entity types %s"
                                            " — asking entity_type first",
                                            _target_ot_e, _et_sorted_ot,
                                        )
                            state.current_docs = _ot_docs_e
                            state.last_retrieval_query = q
                            state.context["last_topic"] = _target_ot_e
                            state.context["last_user_legal_query"] = q
                            state.context["_direct_topic_match"] = True
                            # Exact OBD/topic match → clear stale slot queue from prior turns
                            # so Practical doesn't ask unrelated location/area/entity slots.
                            state.context.pop("topic_slot_queue", None)
                            state.context.pop("pending_slot", None)
                            _LOG.info(
                                "[Supervisor] _ensure: op_topic chapter retrieval: %r → %d docs",
                                _target_ot_e, len(_ot_docs_e),
                            )
                            return
        except Exception as _ot_ex_e:
            _LOG.warning("[Supervisor] _ensure: op_topic retrieval failed (%s), falling through", _ot_ex_e)

        # Step 0b: operation_by_department exact-match retrieval.
        # When user types a query that exactly matches (or contains) a specific
        # operation_by_department value — e.g. "แก้ไข ตราประทับ 1 ดวง" — retrieve that
        # specific row directly, bypassing the op-type follow-up anchor which would
        # otherwise lock retrieval to the previous "แก้ไข ชื่อ" topic.
        # Only matches OBD values that are unique to a single license_type (uniqueness check
        # in the cache builder excludes cross-license duplicates like "แก้ไข เปลี่ยนแปลง").
        # Match direction: obd IN query (not query in obd) to avoid ambiguous partial matches.
        try:
            _obd_candidates_e = self._get_all_unique_operation_by_dept_from_store()
            if _obd_candidates_e:
                # Normalize whitespace so "มากกว่า  1 ดวง" (double-space) still matches
                # data values stored with single spaces.
                _q_lower_obd = re.sub(r"\s+", " ", q.lower().strip())
                _matched_obds_e = [
                    obd for obd in _obd_candidates_e
                    if (lambda _n: _n == _q_lower_obd or _n in _q_lower_obd)(
                        re.sub(r"\s+", " ", obd.lower().strip())
                    )
                ]
                if _matched_obds_e:
                    _target_obd_e = max(_matched_obds_e, key=len)
                    _coll_obd_e = self._get_chroma_collection()
                    if _coll_obd_e is not None:
                        _obd_res_e = _coll_obd_e.get(
                            where={"operation_by_department": _target_obd_e},
                            include=["documents", "metadatas"],
                        )
                        _obd_max_e = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                        _obd_docs_e = [
                            {"content": (c or "")[:_obd_max_e], "metadata": m or {}}
                            for c, m in zip(
                                _obd_res_e.get("documents") or [],
                                _obd_res_e.get("metadatas") or [],
                            )
                        ]
                        # Entity-type mismatch guard: if session has a known entity_type
                        # (บุคคลธรรมดา/นิติบุคคล) but ALL retrieved docs are for the other
                        # entity_type, try to find entity-matched docs for the same OBD.
                        # This handles: user said "บุคคลธรรมดา" earlier but now asks about a
                        # นิติบุคคล-specific operation (or changed their business type).
                        _obd_session_et = str(
                            (state.context.get("collected_slots") or {}).get("entity_type") or ""
                        ).strip()
                        if _obd_docs_e and _obd_session_et in ("นิติบุคคล", "บุคคลธรรมดา"):
                            _obd_doc_ets = {
                                str(d.get("metadata", {}).get("entity_type_normalized") or "").strip()
                                for d in _obd_docs_e
                            } - {""}
                            if _obd_doc_ets and _obd_session_et not in _obd_doc_ets:
                                # 1) Try exact OBD + entity_type filter in Chroma
                                _obd_et_res = _coll_obd_e.get(
                                    where={"$and": [
                                        {"operation_by_department": _target_obd_e},
                                        {"entity_type_normalized": _obd_session_et},
                                    ]},
                                    include=["documents", "metadatas"],
                                )
                                _obd_et_docs = [
                                    {"content": (c or "")[:_obd_max_e], "metadata": m or {}}
                                    for c, m in zip(
                                        _obd_et_res.get("documents") or [],
                                        _obd_et_res.get("metadatas") or [],
                                    )
                                ]
                                if _obd_et_docs:
                                    _obd_docs_e = _obd_et_docs
                                    _LOG.info(
                                        "[Supervisor] _ensure: OBD entity-match switch → %d %r docs",
                                        len(_obd_et_docs), _obd_session_et,
                                    )
                                else:
                                    # 2) No OBD row for this entity → the concept is entity-specific
                                    # (e.g. "ตราประทับ" only exists for นิติบุคคล).  Keep the original
                                    # OBD docs.  Set a SINGLE-TURN context override so the LLM uses the
                                    # correct entity for this answer without permanently polluting
                                    # collected_slots — subsequent queries revert to the original entity.
                                    _obd_new_et = (
                                        list(_obd_doc_ets)[0] if len(_obd_doc_ets) == 1 else None
                                    )
                                    if _obd_new_et:
                                        state.context["_obd_entity_override"] = _obd_new_et
                                    _LOG.info(
                                        "[Supervisor] _ensure: OBD entity-mismatch, no same-entity doc"
                                        " → keeping original OBD docs, single-turn entity override=%r",
                                        _obd_new_et,
                                    )
                        if _obd_docs_e:
                            # Exact OBD match → clear stale slot queue from prior turns so
                            # Practical doesn't ask unrelated location/area/entity slots.
                            state.context.pop("topic_slot_queue", None)
                            state.context.pop("pending_slot", None)
                            # Entity variety: when entity_type not yet collected and docs cover
                            # multiple entity types (บุคคลธรรมดา + นิติบุคคล), inject entity_type
                            # slot into topic_slot_queue so Practical asks before answering.
                            # _direct_topic_match blocks the same check inside Practical (line 3742)
                            # so we must do it here, before returning from _ensure.
                            if not _obd_session_et:
                                _et_in_obd = {
                                    str(d.get("metadata", {}).get("entity_type_normalized") or "").strip()
                                    for d in _obd_docs_e
                                } - {"", "nan", "None"}
                                # Fallback: combined-entity doc (entity_type_normalized empty) whose
                                # identification_documents has both types as numbered-list section headers.
                                # Format in Sheets: "บุคคลธรรมดา 1.item... นิติบุคคล 1.item..." (no newlines).
                                # Pattern: entity_type immediately followed by " N." (section header).
                                # Casual mentions like "ที่มิใช่นิติบุคคล" are NOT followed by digits.
                                _obd_raw_metas = _obd_res_e.get("metadatas") or []
                                _obd_combined_content = any(
                                    re.search(r"บุคคลธรรมดา\s*\d+\.", str((raw_m or {}).get("identification_documents") or ""))
                                    and re.search(r"นิติบุคคล\s*\d+\.", str((raw_m or {}).get("identification_documents") or ""))
                                    for raw_m in _obd_raw_metas
                                )
                                _obd_et_mixed = (
                                    ("นิติบุคคล" in _et_in_obd and "บุคคลธรรมดา" in _et_in_obd)
                                    or _obd_combined_content
                                )
                                if _obd_et_mixed:
                                    _sq_obd = list((state.context or {}).get("topic_slot_queue") or [])
                                    _et_already_known_obd = bool(
                                        (state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}).get("entity_type")
                                    )
                                    if not _et_already_known_obd and not any(s.get("key") == "entity_type" for s in _sq_obd):
                                        _et_sorted_obd = sorted(_et_in_obd) if len(_et_in_obd) >= 2 else ["นิติบุคคล", "บุคคลธรรมดา"]
                                        state.context["topic_slot_queue"] = [{
                                            "key": "entity_type",
                                            "options": _et_sorted_obd,
                                            "display_options": _et_sorted_obd,
                                            "question": "ร้านของคุณดำเนินการในรูปแบบใดครับ",
                                        }] + _sq_obd
                                        _LOG.info(
                                            "[Supervisor] _ensure: OBD %r has mixed entity types %s"
                                            " — asking entity_type first",
                                            _target_obd_e, _et_sorted_obd,
                                        )
                            state.current_docs = _obd_docs_e
                            state.last_retrieval_query = q
                            state.context["last_topic"] = _target_obd_e
                            state.context["last_user_legal_query"] = q
                            state.context["_direct_topic_match"] = True
                            _LOG.info(
                                "[Supervisor] _ensure: operation_by_dept chapter retrieval: %r → %d docs",
                                _target_obd_e, len(_obd_docs_e),
                            )
                            return
        except Exception as _obd_ex_e:
            _LOG.warning("[Supervisor] _ensure: operation_by_dept retrieval failed (%s), falling through", _obd_ex_e)

        # Save original query BEFORE anchoring — used for bank/entity detection below so we
        # don't falsely match bank names that came from the anchor (previous turn's query).
        _q_orig = q

        # ISSUE 1: track the clean anchor so we can store it as last_retrieval_query instead
        # of the compound anchored query — prevents Jaccard cache poisoning in long conversations.
        _anchored_q: Optional[str] = None

        # 🎯 Generic follow-up anchoring: if user's message contains generic procedure/document/fee
        # keywords but NO specific topic/license name, anchor the retrieval query to the active topic
        # so we don't retrieve cross-topic docs and accidentally trigger multi-license detection.
        # Example: "อยากรู้ขั้นตอนสมัคร เอกสารที่ต้องใช้ หรือค่าธรรมเนียม" after QR Payment answer.
        _last_ret_q = str(getattr(state, "last_retrieval_query", "") or "").strip()
        _last_legal_q = str((state.context or {}).get("last_user_legal_query") or "").strip()
        # If last_retrieval_query is itself a generic follow-up phrase (e.g. "ถ้าขอแค่ขั้นตอน"),
        # using it as anchor would cascade into wrong retrieval for the NEXT follow-up.
        # Prefer last_user_legal_query (the actual topic question) in that case.
        _ret_q_is_generic = bool(
            _last_ret_q
            and self._GENERIC_FOLLOWUP_KEYWORDS_RE.search(_last_ret_q)
            and not self._SPECIFIC_TOPIC_RE.search(_last_ret_q)
            and len(_last_ret_q) <= 80
        )
        _active_topic_q = (
            (_last_legal_q or _last_ret_q) if _ret_q_is_generic
            else (_last_ret_q or _last_legal_q)
        )
        if (
            _active_topic_q
            and len(q) <= 80
            and self._GENERIC_FOLLOWUP_KEYWORDS_RE.search(q)
            and not self._SPECIFIC_TOPIC_RE.search(q)
            and not self._ACTION_Q_RE.search(q)
        ):
            _LOG.info(
                "[Supervisor] generic follow-up detected %r — anchoring retrieval to active topic %r",
                q[:50], _active_topic_q[:50],
            )
            _anchored_q = _active_topic_q  # store clean anchor for last_retrieval_query
            q = f"{_active_topic_q} {q}"

        # Op-type follow-up enrichment: "จะยกเลิก/จะต่ออายุ/จะแก้ไข" on the SAME license.
        # _ACTION_Q_RE blocks the generic anchoring above, but when user does an op-switch
        # follow-up without naming the license (e.g. "แล้วถ้าจะยกเลิกละ" after แก้ไขรายการ),
        # bare retrieval returns wrong-license docs. Enrich with active topic to stay anchored.
        _force_fresh_retrieval = False
        if (
            _anchored_q is None
            and _active_topic_q
            and _active_topic_q != q  # prevent self-anchor when no prior topic exists
            and len(q) <= 80
            and not self._SPECIFIC_TOPIC_RE.search(q)
            and (
                self._OP_INFER_CANCEL_RE.search(q)
                or self._OP_INFER_EDIT_RE.search(q)
                or self._OP_INFER_RENEW_RE.search(q)
            )
        ):
            _LOG.info(
                "[Supervisor] op-type follow-up %r — anchoring retrieval to active topic %r",
                q[:50], _active_topic_q[:50],
            )
            _anchored_q = _active_topic_q
            q = f"{_active_topic_q} {q}"
            # Force fresh retrieval when the original query's op-type differs from the last.
            # The anchored q now contains the old topic (e.g. "ต่ออายุ") so _should_retrieve_new
            # mis-classifies this as same-op and returns False (Jaccard too high).
            def _infer_op(t: str) -> Optional[str]:
                if self._OP_INFER_RENEW_RE.search(t):  return "renew"
                if self._OP_INFER_CANCEL_RE.search(t): return "cancel"
                if self._OP_INFER_EDIT_RE.search(t):   return "edit"
                return None
            _orig_op = _infer_op(_q_orig)
            _prev_op = _infer_op(_last_ret_q)
            if _orig_op and _prev_op and _orig_op != _prev_op:
                _LOG.info(
                    "[Supervisor] op-type changed %r→%r — forcing fresh retrieval despite anchor",
                    _prev_op, _orig_op,
                )
                _force_fresh_retrieval = True

        # Channel-preference follow-up anchoring: "ไม่อยากยื่นออนไลน์", "ยื่นด้วยตนเอง", etc.
        # These queries contain NO specific topic keywords so they fall through all anchoring above
        # and land in broad_q — returning garbage unrelated docs.
        # Fix: when channel-negation matches AND there's an active topic, enrich with the prior
        # topic + offline keywords so retrieval finds the correct offline-channel row.
        if (
            _anchored_q is None
            and _active_topic_q
            and _active_topic_q != q
            and len(q) <= 80
            and not self._SPECIFIC_TOPIC_RE.search(q)
            and self._CHANNEL_PREF_OFFLINE_RE.search(q)
        ):
            _LOG.info(
                "[Supervisor] channel-preference follow-up %r — anchoring to active topic %r",
                q[:50], _active_topic_q[:50],
            )
            _anchored_q = _active_topic_q
            q = f"{_active_topic_q} ด้วยตนเอง สำนักงาน"
            _force_fresh_retrieval = True

        # Unified slot-change detection — must run BEFORE cache check to bypass Jaccard
        # false-positives from the anchored query still containing old slot values in its text.
        # Handles: entity_type switch, department (bank) switch, registration_type switch,
        # and any combination of the above in a single query.
        if self._apply_slot_change_if_detected(state, _q_orig):
            return

        if not _force_fresh_retrieval and not self._should_retrieve_new_for_practical(state, q):
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
                        "license_type", "operation_topic", "operation_by_department",
                        "entity_type_normalized", "registration_type", "department",
                        "fees", "operation_duration", "service_channel",
                        "operation_steps", "identification_documents", "research_reference",
                    })
                    _SUPERVISOR_FC_MT = {
                        "operation_steps": 600, "identification_documents": 4000,
                        "research_reference": 3200, "fees": 120, "service_channel": 200,
                    }
                    # Read collected_slots to filter docs by entity_type/registration_type.
                    # multi-topic path previously ignored slots → leaked docs for wrong sub-types
                    # (e.g. ห้างหุ้นส่วน forms shown even when user chose บริษัทจำกัด).
                    _cs_mt = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
                    _mt_entity = (
                        _cs_mt.get("entity_type_normalized")
                        or (_cs_mt.get("entity_type") or "").strip()
                        or (state.context or {}).get("slots", {}).get("entity_type_normalized", "")
                    ).strip()
                    _mt_reg = (
                        _cs_mt.get("registration_type")
                        or (state.context or {}).get("slots", {}).get("registration_type", "")
                    ).strip()

                    def _build_slim(fc: str, fm: dict) -> Dict:
                        _sm: Dict = {}
                        for _k, _v in (fm or {}).items():
                            if _k not in _SUPERVISOR_META_WL_MT or _v in (None, "", "nan", "None"):
                                continue
                            _vs = str(_v)
                            _cap = _SUPERVISOR_FC_MT.get(_k)
                            _sm[_k] = _vs[:_cap] if _cap and len(_vs) > _cap else _vs
                        return {"content": (fc or "")[:_doc_chars_mt], "metadata": _sm}

                    _per_lt_cap = int(getattr(conf, "LLM_DOCS_MAX_BROAD", 20))
                    _total_cap = _per_lt_cap * 2  # hard ceiling across all license types combined
                    _STRUCTURAL_KEYS = ("operation_steps", "identification_documents", "fees", "terms_and_conditions", "legal_regulatory")

                    def _structural_score(d: Dict) -> int:
                        """Higher = more operational fields present → sort first before capping."""
                        m = d.get("metadata") or {}
                        return sum(
                            1 for k in _STRUCTURAL_KEYS
                            if (m.get(k) or "").strip() not in ("", "nan", "None")
                        )

                    _merged: List[Dict] = []
                    for _lt_name in _detected_licenses:
                        if len(_merged) >= _total_cap:
                            break
                        _r = _coll_mt.get(
                            where={"license_type": _lt_name},
                            include=["documents", "metadatas"],
                        )
                        _all_lt_docs = [
                            _build_slim(_fc, _fm)
                            for _fc, _fm in zip(_r.get("documents") or [], _r.get("metadatas") or [])
                        ]

                        # Apply entity_type filter if known — prevents leaking other-entity docs
                        _filtered_lt: List[Dict] = _all_lt_docs
                        if _mt_entity:
                            _entity_match = [
                                d for d in _all_lt_docs
                                if (d.get("metadata") or {}).get("entity_type_normalized", "") in ("", _mt_entity)
                            ]
                            if _entity_match:
                                _filtered_lt = _entity_match

                        # Apply registration_type filter if known — prevents leaking other sub-type docs
                        if _mt_reg and _filtered_lt:
                            _reg_match = [
                                d for d in _filtered_lt
                                if (d.get("metadata") or {}).get("registration_type", "") in ("", _mt_reg)
                            ]
                            if _reg_match:
                                _filtered_lt = _reg_match

                        # Sort: structural docs (operation_steps/fees/documents) first so that
                        # the per-license cap keeps the most data-rich docs, not arbitrary FAQ rows.
                        _filtered_lt.sort(key=_structural_score, reverse=True)
                        _merged.extend(_filtered_lt[:_per_lt_cap])
                    _merged = _merged[:_total_cap]
                    if _merged:
                        state.current_docs = _merged
                        state.last_retrieval_query = _anchored_q if _anchored_q else q
                        state.context["_multi_topic_retrieval"] = True
                        state.context["multi_license_topics"] = list(_detected_licenses)
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
        for _rare_pat, _rare_lt in self._RARE_LT_PATTERNS.items():
            if re.search(_rare_pat, q, re.IGNORECASE):
                try:
                    _vstore = getattr(self.retriever, "vectorstore", None)
                    if _vstore is not None:
                        # Use similarity_search with license_type filter so docs are ranked
                        # by semantic relevance to the query, not by DB insertion order.
                        _top_k_rare = int(getattr(conf, "RETRIEVAL_TOP_K", 20))
                        try:
                            if getattr(conf, "HYBRID_SEARCH_ENABLED", False):
                                from utils.hybrid_retriever import hybrid_scored_search
                                _scored = hybrid_scored_search(
                                    _vstore, q, k=_top_k_rare,
                                    metadata_filter={"license_type": _rare_lt},
                                    rrf_k=int(getattr(conf, "HYBRID_RRF_K", 60)),
                                )
                            else:
                                _scored = _vstore.similarity_search_with_relevance_scores(
                                    q, k=_top_k_rare, filter={"license_type": _rare_lt}
                                )
                            _raw_docs = []
                            for _rd, _rs in _scored:
                                (getattr(_rd, "metadata", {}) or {})["_sim"] = _rs
                                _raw_docs.append(_rd)
                        except Exception as _ss_e:
                            _LOG.warning("[Supervisor] rare-license scored search failed (%s) — trying unscored", _ss_e)
                            _raw_docs = _vstore.similarity_search(
                                q, k=_top_k_rare, filter={"license_type": _rare_lt}
                            )

                        # Apply reranker if enabled — final relevance ordering
                        if getattr(conf, "RERANKER_ENABLED", False) and _raw_docs:
                            try:
                                from utils.reranker import rerank as _do_rerank
                                _rr_model = getattr(conf, "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
                                _rr_top_k = int(getattr(conf, "RERANKER_TOP_K", len(_raw_docs)))
                                _raw_docs = _do_rerank(q, _raw_docs, model_name=_rr_model, top_k=_rr_top_k)
                            except Exception as _rr_e:
                                _LOG.warning("[Supervisor] rare-license reranker failed (%s) — continuing without rerank", _rr_e)

                        if _raw_docs:
                            _doc_chars2 = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                            _SUPERVISOR_META_WL2 = frozenset({
                                "license_type", "operation_by_department", "operation_topic",
                                "entity_type_normalized", "registration_type", "department",
                                "fees", "operation_duration", "service_channel",
                                "identification_documents", "operation_steps", "research_reference",
                            })
                            _SUPERVISOR_FC2 = {
                                "operation_steps": 600, "identification_documents": 4000,
                                "research_reference": 3200, "fees": 120, "service_channel": 200,
                            }
                            _filtered = []
                            for _rd in _raw_docs:
                                _fc = getattr(_rd, "page_content", "") or ""
                                _fm = getattr(_rd, "metadata", {}) or {}
                                _sm = {}
                                for _k, _v in _fm.items():
                                    if _k not in _SUPERVISOR_META_WL2 or _v in (None, "", "nan", "None"):
                                        continue
                                    _vs = str(_v)
                                    _cap = _SUPERVISOR_FC2.get(_k)
                                    _sm[_k] = _vs[:_cap] if _cap and len(_vs) > _cap else _vs
                                _filtered.append({"content": _fc[:_doc_chars2], "metadata": _sm})
                            _LOG.info(
                                "[Supervisor] rare-license ranked fetch: %r → %d docs (%s)",
                                _rare_lt, len(_filtered),
                                "reranked" if getattr(conf, "RERANKER_ENABLED", False) else "sim-sorted",
                            )
                            state.current_docs = _filtered
                            state.last_retrieval_query = _anchored_q if _anchored_q else q
                            return
                except Exception as _e_rare:
                    _LOG.warning("[Supervisor] rare-license filter retrieval failed: %s", _e_rare)
                break  # only match first pattern

        # Query expansion: use shared SYNONYM_PATTERNS (utils/query_synonyms.py) — covers all 3 sheets
        _sv_expansions: list = []
        for _sv_pat, _sv_exp in SYNONYM_PATTERNS:
            if re.search(_sv_pat, q, re.IGNORECASE) and _sv_exp not in _sv_expansions:
                _sv_expansions.append(_sv_exp)
        q_expanded = (q + " " + " ".join(_sv_expansions)).strip() if _sv_expansions else q

        # Main-topic chapter retrieval: when user's query explicitly names a main_topic that exists
        # in Chroma (e.g. "กลยุทธ์ด้านการสื่อสาร"), retrieve ALL docs in that chapter via
        # metadata_filter instead of semantic search.  Semantic search is limited by RETRIEVAL_TOP_K=15
        # and misses docs ranked below the cutoff — main_topic filter guarantees completeness.
        # Cap at _MAIN_TOPIC_MAX_DOCS as a safety ceiling only — not a quality gate.
        # Set high enough to never truncate real chapters (largest known: 37 docs).
        # Prevents runaway token usage if a future chapter grows very large (>60 docs).
        _MAIN_TOPIC_MAX_DOCS = 60
        _mt_filter_used = False
        _main_topic_candidates = self._get_all_main_topics_from_store()
        if _main_topic_candidates:
            _q_lower_sv = q.lower()
            _matched_mts = [mt for mt in _main_topic_candidates if mt.lower() in _q_lower_sv and len(mt) >= 5]
            # When multiple main_topics match, pick the longest one (most specific).
            # E.g. query "การบริหารจัดการพนักงานภายในร้านอาหาร" matches both
            # "การฝึกอบรมพนักงาน" and "การบริหารจัดการพนักงานภายในร้านอาหาร" —
            # longest match is the correct chapter the user is asking about.
            if _matched_mts:
                _target_mt_sv = max(_matched_mts, key=len)
                try:
                    _vstore_mt_sv = getattr(self.retriever, "vectorstore", None)
                    _coll_mt_sv = getattr(_vstore_mt_sv, "_collection", None) if _vstore_mt_sv else None
                    if _coll_mt_sv is not None:
                        _mt_res = _coll_mt_sv.get(
                            where={"main_topic": _target_mt_sv},
                            include=["documents", "metadatas"],
                        )
                        _mt_docs_raw = list(zip(
                            _mt_res.get("documents") or [],
                            _mt_res.get("metadatas") or [],
                        ))
                        if _mt_docs_raw:
                            _doc_chars_mt_sv = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                            _mt_slim: List[Dict[str, Any]] = []
                            for _fc, _fm in _mt_docs_raw[:_MAIN_TOPIC_MAX_DOCS]:
                                _slim: Dict = {}
                                for _k, _v in (_fm or {}).items():
                                    if _v in (None, "", "nan", "None"):
                                        continue
                                    _slim[_k] = str(_v)
                                _mt_slim.append({"content": (_fc or "")[:_doc_chars_mt_sv], "metadata": _slim})
                            state.current_docs = _mt_slim
                            state.last_retrieval_query = _anchored_q if _anchored_q else q
                            state.context["last_user_legal_query"] = q
                            state.context["last_topic"] = _target_mt_sv
                            _mt_filter_used = True
                            _LOG.info(
                                "[Supervisor] main_topic chapter retrieval: %r → %d docs (capped at %d)",
                                _target_mt_sv, len(_mt_slim), _MAIN_TOPIC_MAX_DOCS,
                            )
                except Exception as _mt_ex:
                    _LOG.warning("[Supervisor] main_topic chapter retrieval failed (%s), falling back", _mt_ex)

        if _mt_filter_used:
            return

        # Sub-topic chapter retrieval: when user's query names a sub_topic exactly
        # (e.g. "การตลาดแบบ B2B" matches sub_topic="การตลาดแบบ B2B" in marketing docs).
        # This fires ONLY when main_topic retrieval didn't match — ensures marketing/business_guide
        # docs with no license_type are found before falling through to semantic retrieval,
        # which would rank regulatory docs (ระบบชำระเงินออนไลน์, etc.) above them via BM25 overlap.
        _st_filter_used = False
        _sub_topic_candidates = self._get_all_sub_topics_from_store()
        if _sub_topic_candidates:
            # Also check the original user message — the LLM rewriter often drops key words
            # (e.g. "การตลาดแบบ B2B" → "การตลาด B2B สำหรับร้านอาหาร", losing "แบบ").
            # state.messages[-1] (role=user) holds the raw pre-rewrite text.
            _last_raw_human = next(
                (m.get("content", "") for m in reversed(state.messages or []) if m.get("role") == "user"),
                "",
            ).lower()
            # ASCII identifier matching: catch cases where LLM drops Thai particles but keeps
            # the key ASCII token (e.g. user: "การตลาด B2B", sub_topic: "การตลาดแบบ B2B").
            # Extract significant ASCII tokens (3+ chars with at least one letter) from sub_topic
            # and check if they all appear in the combined query text.
            import re as _re_st
            _ASCII_ID_RE_ST = _re_st.compile(r'[A-Za-z][A-Za-z0-9\-]{2,}|[0-9][A-Za-z][A-Za-z0-9\-]{1,}')
            _combined_q = _q_lower_sv + " " + _last_raw_human

            def _ascii_id_match(sub_t: str) -> bool:
                tokens = _ASCII_ID_RE_ST.findall(sub_t)
                if not tokens:
                    return False
                return all(t.lower() in _combined_q for t in tokens)

            _matched_sts = [
                st for st in _sub_topic_candidates
                if len(st) >= 8 and (
                    st.lower() in _q_lower_sv
                    or st.lower() in _last_raw_human
                    or _ascii_id_match(st)
                )
            ]
            # Also check operation_topic values — rows like the exemption row have
            # operation_topic='การจดทะเบียนพาณิชย์ที่ได้รับการยกเว้น' but sub_topic=None,
            # so they're invisible to sub_topic matching. Prefer the longer match overall.
            _op_topic_candidates = self._get_all_operation_topics_from_store()
            _matched_ots = [
                ot for ot in _op_topic_candidates
                if len(ot) >= 8 and (
                    ot.lower() in _q_lower_sv
                    or ot.lower() in _last_raw_human
                    or _ascii_id_match(ot)
                )
            ]
            _target_st_sv = ""
            _use_op_topic_filter = False
            if _matched_sts or _matched_ots:
                _best_st = max(_matched_sts, key=len) if _matched_sts else ""
                _best_ot = max(_matched_ots, key=len) if _matched_ots else ""
                if len(_best_ot) > len(_best_st):
                    _target_st_sv = _best_ot
                    _use_op_topic_filter = True
                else:
                    _target_st_sv = _best_st
            if _target_st_sv:
                try:
                    _vstore_st_sv = getattr(self.retriever, "vectorstore", None)
                    _coll_st_sv = getattr(_vstore_st_sv, "_collection", None) if _vstore_st_sv else None
                    if _coll_st_sv is not None:
                        _st_filter_key = "operation_topic" if _use_op_topic_filter else "sub_topic"
                        _st_res = _coll_st_sv.get(
                            where={_st_filter_key: _target_st_sv},
                            include=["documents", "metadatas"],
                        )
                        _st_docs_raw = list(zip(
                            _st_res.get("documents") or [],
                            _st_res.get("metadatas") or [],
                        ))
                        if _st_docs_raw:
                            _doc_chars_st_sv = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                            _st_slim: List[Dict[str, Any]] = []
                            for _fc_st, _fm_st in _st_docs_raw[:_MAIN_TOPIC_MAX_DOCS]:
                                _slim_st: Dict = {}
                                for _k_st, _v_st in (_fm_st or {}).items():
                                    if _v_st in (None, "", "nan", "None"):
                                        continue
                                    _slim_st[_k_st] = str(_v_st)
                                _st_slim.append({"content": (_fc_st or "")[:_doc_chars_st_sv], "metadata": _slim_st})
                            state.current_docs = _st_slim
                            state.last_retrieval_query = _anchored_q if _anchored_q else q
                            state.context["last_user_legal_query"] = q
                            state.context["last_topic"] = _target_st_sv
                            state.context["_subtopic_chapter_used"] = True
                            if _use_op_topic_filter:
                                # Exact operation_topic match → entity-neutral direct answer
                                state.context["_direct_topic_match"] = True
                            _st_filter_used = True
                            _LOG.info(
                                "[Supervisor] %s chapter retrieval: %r → %d docs",
                                "operation_topic" if _use_op_topic_filter else "sub_topic",
                                _target_st_sv, len(_st_slim),
                            )
                except Exception as _st_ex:
                    _LOG.warning("[Supervisor] sub_topic chapter retrieval failed (%s), falling back", _st_ex)

        if _st_filter_used:
            return

        # Broad question detection: query spans multiple areas (business setup + regulatory + marketing).
        # Standard single-pass retrieval on a broad query returns mostly marketing/business_guide docs
        # because those texts semantically dominate — regulatory docs (licenses, VAT, etc.) rank low.
        # Fix: two-pass retrieval for broad questions — Pass 1 captures user intent, Pass 2 guarantees
        # regulatory coverage using a targeted legal query. Results are merged and deduped.
        _is_broad_q = self._is_broad_question(q, state)
        if _is_broad_q:
            state.context["_broad_question"] = True
            _LOG.info("[Supervisor] broad question detected — two-pass retrieval for %r", q[:60])
        else:
            state.context.pop("_broad_question", None)

        results: List[Dict[str, Any]] = []

        _sv_hybrid = getattr(conf, "HYBRID_SEARCH_ENABLED", False)
        _sv_rrf_k = int(getattr(conf, "HYBRID_RRF_K", 60))
        _sv_vstore = getattr(self.retriever, "vectorstore", None)

        def _sv_search(q: str, k: int) -> list:
            """Hybrid or Dense retrieval for Supervisor broad-retrieval paths."""
            if _sv_hybrid and _sv_vstore is not None:
                from utils.hybrid_retriever import hybrid_scored_search
                return [d for d, _ in hybrid_scored_search(_sv_vstore, q, k=k, rrf_k=_sv_rrf_k)]
            return list(self.retriever.invoke(q) or [])

        if _is_broad_q:
            # Pass 1: semantic retrieval on user's actual query — captures practical/marketing intent
            _p1_top_k = int(getattr(conf, "RETRIEVAL_TOP_K", 15) or 15)
            _p1_docs = _sv_search(q_expanded, _p1_top_k)

            # Pass 2: targeted retrieval on regulatory query — guarantees legal/license docs present
            _regulatory_q = str(
                getattr(conf, "DEFAULT_RETRIEVAL_FALLBACK_QUERY", "")
                or "กฎหมายร้านอาหาร ใบอนุญาต ภาษี VAT จดทะเบียน สุขาภิบาล ประกันสังคม"
            )
            _p2_docs = _sv_search(_regulatory_q, 8)

            # Merge: Pass 1 first (preserves intent ordering), then Pass 2 docs not already present.
            # Dedup by first 120 chars of page_content — stable fingerprint without full compare.
            _seen_fp: set = set()
            docs = []
            for _d in _p1_docs:
                _fp = (getattr(_d, "page_content", "") or "")[:120]
                if _fp not in _seen_fp:
                    _seen_fp.add(_fp)
                    docs.append(_d)
            for _d in _p2_docs:
                _fp = (getattr(_d, "page_content", "") or "")[:120]
                if _fp not in _seen_fp:
                    _seen_fp.add(_fp)
                    docs.append(_d)

            top_k = len(docs)
            _LOG.info(
                "[Supervisor] broad two-pass retrieval: pass1=%d pass2=%d → merged=%d unique docs",
                len(_p1_docs), len(_p2_docs), top_k,
            )
        else:
            docs = _sv_search(q_expanded, int(getattr(conf, "RETRIEVAL_TOP_K", 15) or 15))
            top_k = int(getattr(conf, "RETRIEVAL_TOP_K", 15) or 15)
        _doc_chars = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
        # Token: cap long metadata fields before storing — prevent ×N token explosion
        _SUPERVISOR_META_WHITELIST = frozenset({
            # Source type — lets downstream LLM know doc origin
            "data_type",
            # Regulatory fields
            "license_type", "operation_topic",
            "operation_by_department",    # needed by practical sibling-completion and dedup key
            "entity_type_normalized", "registration_type", "department",
            "fees", "operation_duration", "service_channel",
            "legal_regulatory",           # บทลงโทษ ค่าปรับ ข้อกำหนดทางกฎหมาย
            "terms_and_conditions",       # หน้าที่และเงื่อนไขของผู้ประกอบการ
            "identification_documents",   # เอกสารที่ต้องใช้ — must reach LLM complete
            "operation_steps",            # ขั้นตอนการดำเนินการ
            "research_reference",         # needed by practical link injection
            # Marketing / business_guide fields (data_loader_general.py)
            "main_topic", "sub_topic", "answer_guideline",
        })
        _SUPERVISOR_FIELD_CAPS = {
            "operation_steps": 600, "identification_documents": 4000,
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
        # ISSUE 1: store only the clean anchor query (not compound) to prevent Jaccard cache poisoning.
        state.last_retrieval_query = _anchored_q if _anchored_q else q

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
        r"ระยะเวลา|กี่วัน|กี่เดือน|กี่ปี|กี่ชั่วโมง|"
        r"เท่าไหร่|กี่บาท|ค่าธรรมเนียม(?!การจด|การสมัคร|การขอ|การยื่น)|"
        r"ช่องทาง(?!การจด|การสมัคร|การขอ|การยื่น))",
        re.IGNORECASE,
    )
    # Action patterns — user wants to perform a registration/application step.
    # Overrides _INFO_Q_RE: even if query looks informational, treat as action.
    _ACTION_Q_RE = re.compile(
        r"(อยากจด|อยากสมัคร|อยากขอ|อยากยื่น|อยากลงทะเบียน|"
        r"ต้องการจด|ต้องการสมัคร|ต้องการขอ|ต้องการยื่น|ต้องการลงทะเบียน|"
        r"จะจด|จะสมัคร|จะขอใบ|จะยื่น|จะลงทะเบียน|"
        r"วิธีจด|วิธีสมัคร|วิธีลงทะเบียน|วิธีการลงทะเบียน|"
        r"ขั้นตอนการจด|ขั้นตอนการสมัคร|ขั้นตอนการขอ|ขั้นตอนการลงทะเบียน|"
        r"ต้องการแก้ไข|ต้องการเปลี่ยนแปลง|ต้องการยกเลิก|ต้องการต่ออายุ|"
        r"อยากแก้ไข|อยากเปลี่ยนแปลง|อยากยกเลิก|อยากต่ออายุ|"
        r"จะแก้ไข|จะเปลี่ยนแปลง|จะยกเลิก|จะต่ออายุ|"
        r"วิธีแก้ไข|วิธีเปลี่ยนแปลง|วิธียกเลิก|วิธีต่ออายุ|"
        r"ขั้นตอนการแก้ไข|ขั้นตอนการเปลี่ยนแปลง|ขั้นตอนการยกเลิก|ขั้นตอนการต่ออายุ|"
        r"จะเปิด|จะทำร้าน|จะทำธุรกิจ|อยากเปิด|อยากทำร้าน|อยากทำธุรกิจ|"
        # Bare "ขอใบ(อนุญาต)" / "ยื่นขอใบ" without modal prefix — implies action, not info.
        # Negative lookahead: exclude renewal context (ต่ออายุ/หมดอายุ within 20 chars).
        r"(?:ขอ|ยื่น(?:\s*ขอ)?)\s*ใบ(?:อนุญาต)?(?!.{0,20}(?:ต่ออายุ|ต่อ\s*อายุ|หมดอายุ|สิ้นอายุ)))",
        re.IGNORECASE,
    )

    def _maybe_build_slot_queue_from_docs(self, state: ConversationState, query: str):
        """
        After doc retrieval for a direct legal question, discover slot dimensions for the
        detected license_type and set topic_slot_queue — skipping already-collected slots.
        If entity_type is already known, also re-retrieves docs with entity filter.
        """
        docs = state.current_docs or []
        if not docs:
            return

        # When supervisor found an exact operation_topic match, the doc is entity-neutral
        # (applies to all entity/registration types). Skip slot-queue building so the bot
        # answers directly without asking clarifying questions.
        if (state.context or {}).get("_direct_topic_match"):
            _LOG.debug("[Supervisor] _maybe_build_slot_queue_from_docs: skipped — _direct_topic_match set")
            return

        # Reset informational-query flag at the start of each call so it doesn't persist
        # from a previous turn into a new procedural query.
        state.context.pop("_informational_query", None)
        state.context.pop("_link_request_query", None)

        # Always persist entity_type if explicitly stated in query — must run BEFORE any early returns
        # so broad_question / multi_topic paths don't skip the update.
        _early_et = self._infer_entity_type_from_query(query)
        if _early_et:
            _early_cs = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
            _existing_et = (_early_cs or {}).get("entity_type")
            if _early_et != _existing_et:
                if hasattr(state, "save_collected_slot"):
                    state.save_collected_slot("entity_type", _early_et)
                if state.context is not None:
                    state.context.setdefault("slots", {})["entity_type"] = _early_et
                _LOG.info(
                    "[Supervisor] entity_type pre-persisted from query text: %r (was: %r)",
                    _early_et, _existing_et,
                )

        # Multi-topic: docs span multiple licenses intentionally — normally skip slot queue.
        # Exception: if entity_type=นิติบุคคล but registration_type unknown, ask first so we
        # don't dump forms for BOTH บริษัทจำกัด and ห้างหุ้นส่วน at the same time.
        if (state.context or {}).get("_multi_topic_retrieval"):
            _cs_mt_q = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
            _mt_entity_q = (
                _cs_mt_q.get("entity_type_normalized")
                or _cs_mt_q.get("entity_type")
                or (state.context or {}).get("slots", {}).get("entity_type", "")
                or ""
            ).strip()
            _mt_reg_q = (
                _cs_mt_q.get("registration_type")
                or (state.context or {}).get("slots", {}).get("registration_type", "")
                or ""
            ).strip()
            if _mt_entity_q == "นิติบุคคล" and not _mt_reg_q:
                # Collect distinct registration_type values from the merged docs.
                # IMPORTANT: only accept values that look like business entity sub-types
                # (contain keywords like บริษัท/ห้างหุ้นส่วน). The registration_type field
                # for food/health licenses contains area-size descriptions
                # (e.g. "ร้านที่มีพื้นที่มากกว่า 200 ตารางเมตร", "• ผู้สัมผัสอาหาร")
                # which are NOT entity choices and produce a nonsensical slot menu.
                _BIZTYPE_KW_MT = ("บริษัท", "ห้างหุ้นส่วน", "ห้าง", "สามัญ", "จำกัด", "มหาชน")
                _BAD_KW_MT = ("ตารางเมตร", "ผู้สัมผัส", "ผู้ประกอบ", "มากกว่า", "น้อยกว่า",
                              "ตร.ม", "•", "1.", "2.", "3.")
                _reg_opts_mt: List[str] = []
                for _d in docs:
                    if not isinstance(_d, dict):
                        continue
                    _rt = ((_d.get("metadata") or {}).get("registration_type") or "").strip()
                    if not _rt or _rt in ("nan", "None", "นิติบุคคล") or _rt in _reg_opts_mt:
                        continue
                    # Must contain a business-type keyword AND have no area/role description keywords
                    if (any(kw in _rt for kw in _BIZTYPE_KW_MT)
                            and not any(bad in _rt for bad in _BAD_KW_MT)):
                        _reg_opts_mt.append(_rt)
                if len(_reg_opts_mt) > 1:
                    _reg_q_mt = f"ธุรกิจของคุณเป็นรูปแบบใดครับ? ({' / '.join(_reg_opts_mt)})"
                    state.context["topic_slot_queue"] = [{
                        "key": "registration_type",
                        "options": _reg_opts_mt,
                        "question": _reg_q_mt,
                    }]
                    _LOG.info(
                        "[Supervisor] multi-topic: asking registration_type before answer "
                        "(entity=นิติบุคคล, options=%s)",
                        _reg_opts_mt,
                    )
                    return
            _LOG.debug("[Supervisor] multi-topic retrieval flag set — skipping slot queue build")
            return

        # Broad question: spans all business dimensions (legal + marketing + practical).
        # Two-pass retrieval already loaded both marketing and regulatory docs.
        # Skip slot queue and multi-license menu entirely — LLM answers with full doc coverage.
        if (state.context or {}).get("_broad_question"):
            _LOG.debug("[Supervisor] broad question — skipping slot queue, passing all docs to practical")
            return

        # Non-regulatory dominant: retrieved docs are mostly marketing/business_guide with no
        # license_type (e.g. B2B marketing, HR guide docs). Slot queue is only meaningful for
        # regulatory docs (which have license_type). Building it from incidental payment/bank docs
        # that ranked in alongside marketing docs causes completely wrong bank/entity questions.
        _docs_with_lt = sum(
            1 for _d in docs if isinstance(_d, dict)
            and str(((_d.get("metadata") or {}).get("license_type") or "")).strip()
            not in ("", "nan", "None")
        )
        _docs_without_lt = len(docs) - _docs_with_lt
        if _docs_without_lt > _docs_with_lt:
            _LOG.info(
                "[Supervisor] non-regulatory docs dominant (%d without license_type vs %d with) — skipping slot queue",
                _docs_without_lt, _docs_with_lt,
            )
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
            _query_dept_any = next(
                (d for d in _available_depts_any
                 if d and (d in query or (d.startswith("ธนาคาร") and d[len("ธนาคาร"):] in query))),
                None,
            )
            if _query_dept_any and _query_dept_any != _cached_dept_any:
                _LOG.info("[Supervisor] dept auto-update %r → %r (from query text)", _cached_dept_any, _query_dept_any)
                if hasattr(state, "save_collected_slot"):
                    state.save_collected_slot("department", _query_dept_any)
                if state.context is not None:
                    state.context.setdefault("slots", {})["department"] = _query_dept_any
                    if "collected_slots" in state.context:
                        state.context["collected_slots"]["department"] = _query_dept_any
                # Short-circuit: dept changed → re-retrieve with new dept + known license_type filter
                # so we skip the multi-license check (which would show a confusing topic selector).
                # The user just wants the same topic answered for the new bank/department.
                _dept_lt = (state.context or {}).get("last_topic", "").strip() if state.context else ""
                # Use last legal retrieval query for relevance — the current input is a slot
                # correction message ("เอาเป็นของไทยพาณิชย์ดีกว่า"), not a topical query.
                _dept_retrieval_q = (state.last_retrieval_query or
                                     (state.context or {}).get("last_user_legal_query") or
                                     query)
                try:
                    _dept_flt: dict = {"department": _query_dept_any}
                    if _dept_lt:
                        _dept_flt["license_type"] = _dept_lt
                    # Also filter by entity_type if already collected — prevents mixed-entity docs
                    # (e.g. user chose บุคคลธรรมดา but re-retrieve returns both entity types).
                    _cs_et_sw = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
                    _et_sw = (
                        _cs_et_sw.get("entity_type_normalized")
                        or _cs_et_sw.get("entity_type")
                        or (state.context or {}).get("slots", {}).get("entity_type", "")
                    ).strip()
                    if _et_sw:
                        _dept_flt["entity_type_normalized"] = _et_sw
                    _dept_docs = self._practical._retrieve_docs(
                        _dept_retrieval_q,
                        metadata_filter=_dept_flt,
                        max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                    )
                    if not _dept_docs and _et_sw:
                        # Relax: drop entity_type filter (new dept may not have entity differentiation)
                        _dept_flt_no_et = {k: v for k, v in _dept_flt.items() if k != "entity_type_normalized"}
                        _dept_docs = self._practical._retrieve_docs(
                            _dept_retrieval_q,
                            metadata_filter=_dept_flt_no_et,
                            max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                        )
                    if not _dept_docs and _dept_lt:
                        # Relax: dept-only filter (license might differ slightly)
                        _dept_docs = self._practical._retrieve_docs(
                            _dept_retrieval_q,
                            metadata_filter={"department": _query_dept_any},
                            max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                        )
                    if _dept_docs:
                        state.current_docs = _dept_docs
                        state.last_retrieval_query = _dept_retrieval_q
                        _LOG.info(
                            "[Supervisor] dept-switch re-retrieve: dept=%r license=%r docs=%d — skipping multi-license check",
                            _query_dept_any, _dept_lt, len(_dept_docs),
                        )
                        return  # proceed to practical.handle with updated docs
                except Exception as _e_dept_sw:
                    _LOG.warning("[Supervisor] dept-switch re-retrieve failed: %s", _e_dept_sw)

        # Intent check: informational/link-request queries should be answered directly — no slot queue needed.
        # Action signals override: "อยากจด/วิธีจด" always builds slot queue regardless.
        _is_action_q_re = bool(self._ACTION_Q_RE.search(query))
        _is_link_req = bool(self._LINK_REQUEST_RE.search(query)) and not _is_action_q_re
        _is_info_q   = bool(self._INFO_Q_RE.search(query))       and not _is_action_q_re
        # LLM fallback: when regex caught neither info nor action
        if not _is_info_q and not _is_action_q_re:
            _llm_info, _llm_action = self._info_action_q_llm_check(query, state)
            _is_info_q = _llm_info and not _llm_action
        if _is_info_q:
            _LOG.debug("[Supervisor] informational query — skipping slot queue: %r", query[:60])
            state.context["_informational_query"] = True
            # For fee/duration queries: current docs may be entity-type/format docs that lack
            # fees/duration metadata → LLM has no data to answer from and may fall back to asking.
            # Supplement with a direct Chroma fetch of docs that have the relevant metadata field.
            _FEE_Q_RE = re.compile(r"ค่าธรรมเนียม|เท่าไหร่|กี่บาท|ค่าใช้จ่าย", re.IGNORECASE)
            _DUR_Q_RE  = re.compile(r"กี่วัน|กี่เดือน|กี่ปี|กี่ชั่วโมง|ระยะเวลา", re.IGNORECASE)
            _target_field = None
            if _FEE_Q_RE.search(query):
                _target_field = "fees"
            elif _DUR_Q_RE.search(query):
                _target_field = "operation_duration"
            if _target_field:
                # Find the target license_type FIRST — must match _lt_for_fee below.
                # _has_field must only check docs of the SAME license to avoid false-positives:
                # e.g. ทะเบียนนายจ้าง has operation_duration='1 วัน' but is unrelated to
                # a ใบทะเบียนพาณิชย์ duration query — checking all docs would suppress the supplement.
                _lt_for_fee = next(
                    (((d.get("metadata") or {}).get("license_type") or "").strip()
                     for d in docs if isinstance(d, dict)
                     if ((d.get("metadata") or {}).get("license_type") or "").strip()),
                    "",
                )
                _has_field = bool(_lt_for_fee) and any(
                    str((d.get("metadata") or {}).get(_target_field) or "").strip()
                    not in ("", "nan", "None")
                    for d in docs if isinstance(d, dict)
                    if ((d.get("metadata") or {}).get("license_type") or "").strip() == _lt_for_fee
                )
                if not _has_field:
                    # Current docs lack the required field for this license — fetch from Chroma.
                    if _lt_for_fee:
                        try:
                            _coll_fee = self._get_chroma_collection()
                            if _coll_fee is not None:
                                # Filter by entity_type when already known (was pre-persisted above)
                                # to avoid mixing docs for different entity types.
                                # e.g. บุคคลธรรมดา=30วัน vs นิติบุคคล=3เดือน for ใบทะเบียนพาณิชย์.
                                _et_for_fee = (state.context or {}).get("slots", {}).get("entity_type", "").strip()
                                _fee_where: dict = (
                                    {"$and": [{"license_type": _lt_for_fee}, {"entity_type_normalized": _et_for_fee}]}
                                    if _et_for_fee else {"license_type": _lt_for_fee}
                                )
                                _fee_result = _coll_fee.get(
                                    where=_fee_where,
                                    include=["documents", "metadatas"],
                                )
                                # Fallback: if entity-filtered returns no target-field docs, retry unfiltered
                                if _et_for_fee and not any(
                                    str((_fm_fb or {}).get(_target_field) or "").strip() not in ("", "nan", "None")
                                    for _fm_fb in (_fee_result.get("metadatas") or [])
                                ):
                                    _LOG.info(
                                        "[Supervisor] info-query supplement: entity=%r → no %s docs, retry unfiltered",
                                        _et_for_fee, _target_field,
                                    )
                                    _fee_result = _coll_fee.get(
                                        where={"license_type": _lt_for_fee},
                                        include=["documents", "metadatas"],
                                    )
                                _doc_chars_fee = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                                _SUPERVISOR_META_WL_FEE = frozenset({
                                    "license_type", "operation_topic",
                                    "entity_type_normalized", "registration_type",
                                    "fees", "operation_duration",
                                })
                                _fee_docs: List[Dict] = []
                                for _fc_f, _fm_f in zip(
                                    _fee_result.get("documents") or [],
                                    _fee_result.get("metadatas") or [],
                                ):
                                    _fld_val = str((_fm_f or {}).get(_target_field) or "").strip()
                                    if not _fld_val or _fld_val in ("nan", "None"):
                                        continue
                                    _sm_f: Dict = {}
                                    for _k_f, _v_f in (_fm_f or {}).items():
                                        if _k_f not in _SUPERVISOR_META_WL_FEE or _v_f in (None, "", "nan", "None"):
                                            continue
                                        _sm_f[_k_f] = str(_v_f)[:400]
                                    _fee_docs.append({"content": (_fc_f or "")[:_doc_chars_fee], "metadata": _sm_f})
                                if _fee_docs:
                                    # Prepend fee/duration docs so LLM sees them first
                                    state.current_docs = _fee_docs + [d for d in docs if isinstance(d, dict)]
                                    _LOG.info(
                                        "[Supervisor] info-query: supplemented %d %s docs for license=%r",
                                        len(_fee_docs), _target_field, _lt_for_fee,
                                    )
                        except Exception as _e_fee:
                            _LOG.warning("[Supervisor] info-query fee doc supplement failed: %s", _e_fee)
            # Fix A: if query clearly implies one operation, infer operation_group and
            # re-retrieve focused docs so practical answers only that operation.
            _lt_infoq = next(
                (((d.get("metadata") or {}).get("license_type") or "").strip()
                 for d in docs if isinstance(d, dict)
                 if ((d.get("metadata") or {}).get("license_type") or "").strip()),
                "",
            )
            if _lt_infoq:
                try:
                    _og_res_iq = self._get_operation_groups_for_entity(_lt_infoq, "")
                    _og_labels_iq, _og_raw_iq = (
                        _og_res_iq if isinstance(_og_res_iq, tuple) else (_og_res_iq, {})
                    )
                    _cs_iq = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
                    if len(_og_labels_iq) >= 2:
                        _inferred_iq = self._infer_operation_group_from_query(query, _og_labels_iq, state)
                        if _inferred_iq:
                            _cur_og_iq = (
                                (_cs_iq or {}).get("operation_group")
                                or (state.context or {}).get("slots", {}).get("operation_group")
                                or ""
                            ).strip()
                            _og_changed_iq = (_inferred_iq != _cur_og_iq)
                            if _og_changed_iq:
                                if hasattr(state, "save_collected_slot"):
                                    state.save_collected_slot("operation_group", _inferred_iq)
                                if state.context is not None:
                                    state.context.setdefault("slots", {})["operation_group"] = _inferred_iq
                                    state.context.setdefault("slots", {})["confirmed_operation"] = _inferred_iq
                                    if "collected_slots" in state.context:
                                        state.context["collected_slots"]["operation_group"] = _inferred_iq
                                _raw_ops_iq = (_og_raw_iq or {}).get(_inferred_iq) or []
                                _op_pfx_iq = " ".join(_raw_ops_iq[:2]) if _raw_ops_iq else _inferred_iq
                                _enrich_iq = " ".join(filter(None, [query, _op_pfx_iq])).strip()
                                try:
                                    _op_docs_iq = self._practical._retrieve_docs(
                                        _enrich_iq,
                                        metadata_filter={"license_type": _lt_infoq},
                                        max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                                    )
                                    if _op_docs_iq:
                                        state.current_docs = _op_docs_iq
                                        state.last_retrieval_query = _enrich_iq
                                        _LOG.info(
                                            "[Supervisor] info-query op-inferred re-retrieve: op=%r docs=%d",
                                            _inferred_iq, len(_op_docs_iq),
                                        )
                                except Exception as _e_iq_rr:
                                    _LOG.warning("[Supervisor] info-query op re-retrieve failed: %s", _e_iq_rr)
                            _LOG.info(
                                "[Supervisor] info-query operation_group auto-inferred=%r changed=%s",
                                _inferred_iq, _og_changed_iq,
                            )
                except Exception as _e_og_iq:
                    _LOG.warning("[Supervisor] info-query operation_group inference failed: %s", _e_og_iq)
            # Exception: if shop_area_type is unknown and docs have area-size keywords in
            # registration_type, the answer differs by shop size — fall through to build
            # the slot queue so user is asked before the LLM answers.
            _cs_info_q = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
            _area_slot_known = bool(
                (_cs_info_q or {}).get("shop_area_type")
                or (_cs_info_q or {}).get("area_size")
                or (state.context or {}).get("slots", {}).get("shop_area_type")
                or (state.context or {}).get("collected_slots", {}).get("shop_area_type")
            )
            if not _area_slot_known:
                _area_in_info_docs = any(
                    any(kw in str((d.get("metadata") or {}).get("registration_type") or "")
                        for kw in self._AREA_KEYWORDS)
                    for d in docs if isinstance(d, dict)
                )
                if _area_in_info_docs:
                    _LOG.info(
                        "[Supervisor] info-query: shop_area_type unknown + area keywords in docs — falling through to slot queue"
                    )
                else:
                    return
            else:
                return
        if _is_link_req:
            # Signal to Practical not to inject entity_type into slot_queue.
            # Supervisor already trims slot queue to department-only (line ~4036).
            state.context["_link_request_query"] = True
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
                _query_dept = next(
                    (d for d in _available_depts
                     if d and (d in query or (d.startswith("ธนาคาร") and d[len("ธนาคาร"):] in query))),
                    None,
                )
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
                elif _query_dept is None and self._SPECIFIC_TOPIC_RE.search(query):
                    # Full dept name not found in query, but query contains a specific bank/topic
                    # keyword (e.g. "กสิกร" = short for ธนาคารกสิกรไทย) that current_docs don't know.
                    # Fall back to vector retrieval with the original query — the embedding model will
                    # naturally rank docs for the mentioned bank highest.
                    try:
                        _fallback_docs = self._practical._retrieve_docs(
                            query,
                            max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                        )
                        if _fallback_docs:
                            _fb_dept = ((_fallback_docs[0].get("metadata") or {}).get("department") or "").strip()
                            if _fb_dept and _fb_dept != _cached_dept:
                                _LOG.info(
                                    "[Supervisor] link query: bank alias in query → fresh retrieval dept=%r docs=%d",
                                    _fb_dept, len(_fallback_docs),
                                )
                                state.current_docs = _fallback_docs
                                state.last_retrieval_query = query
                                if hasattr(state, "save_collected_slot"):
                                    state.save_collected_slot("department", _fb_dept)
                                if state.context is not None:
                                    state.context.setdefault("slots", {})["department"] = _fb_dept
                                    if "collected_slots" in state.context:
                                        state.context["collected_slots"]["department"] = _fb_dept
                    except Exception as _fb_err:
                        _LOG.debug("[Supervisor] link bank-alias fallback retrieval failed: %s", _fb_err)
                _LOG.debug("[Supervisor] link query + dept known — skipping slot queue: %r", query[:60])
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

            # Priority 0 (checked FIRST): if user explicitly named a license type in the query,
            # use it directly — overrides doc-count dominant to prevent stale-doc contamination.
            # Checks 8 Thai chars of the core name (after stripping ใบ/แบบ/ระบบ/การ prefix).
            # Also checks English tokens (e.g. "EDC", "QR") so ASCII abbreviations are matched.
            # Example: "การขอรับรองมาตรฐาน" → core "รับรองมา" matches ใบรับรองมาตรฐานร้านอาหาร
            # Example: "EDC" → English token "EDC" matches เครื่องรูดบัตร EDC
            _q_thai = re.sub(r'[^\u0E00-\u0E7F]', '', query or "")
            _q_lower = (query or "").lower()
            _strip_pfx = re.compile(r'^(ใบ|แบบ|ระบบ|การ)')
            _explicit_lt = None
            # Pass 1: English token match (higher specificity — avoids "ชำระเงิน" false-positives
            # where a Thai substring of one license (ระบบชำระเงินออนไลน์) matches before a shorter
            # but more specific English token in another license (เครื่องรูดบัตร EDC "EDC") is checked).
            for _lt_cand in license_types_seen:
                for _eng_tok in re.findall(r'[A-Za-z0-9]{2,}', _lt_cand):
                    if _eng_tok.lower() in _q_lower:
                        _explicit_lt = _lt_cand
                        break
                if _explicit_lt:
                    break
            # Pass 2: Thai substring match (only if no English match found)
            if not _explicit_lt:
                for _lt_cand in license_types_seen:
                    _lt_core = _strip_pfx.sub('', _lt_cand).strip()
                    _match_key = _lt_core[:8]
                    if len(_match_key) >= 5 and _match_key in _q_thai:
                        _explicit_lt = _lt_cand
                        break
            # Keyword-to-license override: for licenses whose name doesn't share n-grams with
            # the user's vocabulary (e.g. "SAN PLUS" ≠ "ใบรับรองมาตรฐานร้านอาหาร", "QR" ≠
            # "ระบบชำระเงินออนไลน์"). These are evaluated ONLY when the name-based check above
            # found nothing, ensuring explicit name mentions still take priority.
            if not _explicit_lt:
                _SUPERVISOR_KW_OVERRIDE = [
                    # SAN / SAN PLUS / food quality standard
                    (re.compile(r'\bSAN\b|\bSAN\s*PLUS\b|รับรองมาตรฐาน|มาตรฐานร้านอาหาร', re.IGNORECASE),
                     'ใบรับรองมาตรฐานร้านอาหาร'),
                    # QR Payment / online payment
                    (re.compile(r'\bQR\b|QR\s*Payment|QR\s*Code|คิวอาร์', re.IGNORECASE),
                     'ระบบชำระเงินออนไลน์'),
                    # VAT / ภพ.20
                    (re.compile(r'\bVAT\b|\bภพ\.?20\b|\bภพ20\b|จด\s*ภาษีมูลค่าเพิ่ม|สมัคร\s*VAT', re.IGNORECASE),
                     'ใบภาษีมูลค่าเพิ่ม ภพ.20'),
                    # ผู้ประกันตน / เงินสมทบ (insured person, not employer)
                    (re.compile(r'ผู้ประกันตน|ทะเบียนผู้ประกัน|เงินสมทบ|แจ้งเข้าทำงาน|แจ้งออกจากงาน', re.IGNORECASE),
                     'ทะเบียนผู้ประกันตน'),
                    # ประกันสังคม alone (no นายจ้าง) → employer registration is more common entry
                    (re.compile(r'ประกันสังคม(?!.*นายจ้าง)', re.IGNORECASE),
                     'ทะเบียนนายจ้าง'),
                    # จัดการการเงิน / ปิดงบ / บัญชี (accounting, not bank account)
                    (re.compile(r'ปิดงบ|ปิดบัญชี|งบการเงิน|งบบัญชี|ทำบัญชี|จัดทำบัญชี|สำนักงานบัญชี', re.IGNORECASE),
                     'จัดการการเงิน'),
                    # สุรา / liquor license
                    (re.compile(r'สุรา|เหล้า|จำหน่ายสุรา|ขายสุรา', re.IGNORECASE),
                     'ใบอนุญาตจำหน่ายสุรา'),
                    # ภาษีป้าย (sign tax) — Supervisor name-check fails because core8="แสดงรายก"
                    (re.compile(r'ภาษีป้าย|ป้ายร้านอาหาร', re.IGNORECASE),
                     'แบบแสดงรายการภาษีป้ายร้านอาหาร'),
                    # กองทุนเงินทดแทน — distinct from ประกันสังคม; needs explicit keyword override
                    (re.compile(r'กองทุนเงินทดแทน|เงินทดแทน(?!ที่ดิน)', re.IGNORECASE),
                     'กองทุนเงินทดแทน'),
                ]
                for _kw_re, _kw_lt in _SUPERVISOR_KW_OVERRIDE:
                    if _kw_re.search(query or "") and _kw_lt in license_types_seen:
                        _explicit_lt = _kw_lt
                        _LOG.info(
                            "[Supervisor] multi-license: keyword override matched %r → dominant=%r",
                            query[:40], _explicit_lt,
                        )
                        break
            if _explicit_lt:
                _LOG.info(
                    "[Supervisor] multi-license: explicit mention %r in query → use as dominant (overrides doc-count)",
                    _explicit_lt,
                )
                license_types_seen = [_explicit_lt]
                # fall through to slot-queue build below (no return)
            else:
                # Keyword overlap check: if none of the meaningful words in the query
                # appear in any retrieved license name, the docs are from a different domain.
                # Return False to signal the caller should deflect instead of showing a menu.
                _GENERIC_LICENSE_WORDS = {
                    "ใบ", "อนุญาต", "ขอ", "การ", "จด", "แบบ", "ระบบ", "แสดง", "รายการ",
                    "ตั้ง", "ที่", "จำ", "หน่าย", "สถาน", "ประกอบ", "ธุรกิจ", "ร้าน",
                }
                _q_terms = set(re.findall(r'[\u0E00-\u0E7F]{3,}', query or ""))
                _license_key_terms: set = set()
                for _lt_check in license_types_seen:
                    for _w in re.findall(r'[\u0E00-\u0E7F]{3,}', _lt_check):
                        if _w not in _GENERIC_LICENSE_WORDS:
                            _license_key_terms.add(_w)
                # Also include department names and operation_topic terms from docs.
                # User may ask using an operation name (e.g. "แปลงมูลค่าหุ้น") that doesn't
                # appear in any license type name but IS a valid operation within that license.
                for _dept_doc in docs:
                    if not isinstance(_dept_doc, dict):
                        continue
                    _meta_ml = _dept_doc.get("metadata") or {}
                    _dept_val = (_meta_ml.get("department") or "").strip()
                    for _dw in re.findall(r'[\u0E00-\u0E7F]{3,}', _dept_val):
                        _license_key_terms.add(_dw)
                    _op_topic_val = (_meta_ml.get("operation_topic") or "").strip()
                    for _ow in re.findall(r'[\u0E00-\u0E7F]{3,}', _op_topic_val):
                        _license_key_terms.add(_ow)
                if _q_terms and _license_key_terms and not (_q_terms & _license_key_terms):
                    # Thai text has no word boundaries — the whole query is often one token.
                    # Check if any key term appears as a substring of the raw query string
                    # (e.g. "ธนาคารไทยพาณิชย์" inside "ต้องการสมัครธนาคารไทยพาณิชย์").
                    _has_substr_match = any(lkt in (query or "") for lkt in _license_key_terms)
                    if not _has_substr_match:
                        # Reverse check: the query may START WITH a word that is a
                        # suffix/infix of a key term — e.g. user says "ประกันสังคม..."
                        # and the dept name is "สำนักงานประกันสังคม".
                        # Extract the first 8 Thai-only chars of the query and see if
                        # they appear anywhere inside any key term.
                        _q_thai_pfx = re.sub(r'[^\u0E00-\u0E7F]', '', query or "")[:8]
                        if len(_q_thai_pfx) >= 5 and any(_q_thai_pfx in _lkt for _lkt in _license_key_terms):
                            _has_substr_match = True
                    if not _has_substr_match:
                        # Broad questions intentionally don't mention specific license names —
                        # they ask "what do I need to do to open a restaurant?".
                        # Two-pass retrieval already ensured regulatory docs are present,
                        # so skip the deflect and let practical answer with full doc coverage.
                        if state.context.get("_broad_question"):
                            _LOG.info(
                                "[Supervisor] multi-license mismatch skipped — broad question, regulatory docs already loaded via two-pass",
                            )
                        else:
                            _LOG.info(
                                "[Supervisor] multi-license topic mismatch — query %s no overlap with licenses %s → deflect",
                                _q_terms, _license_key_terms,
                            )
                            # Safety net: if this is a follow-up turn and we have a known
                            # retrieval anchor, re-retrieve with it instead of dumping 24+
                            # mixed docs to the LLM. Covers typos that bypass entity/bank
                            # slot-change detection (e.g. "บุคคละรรมดา" → regex miss → here).
                            # When query explicitly mentions a different operation (แก้ไข, ยกเลิก etc.),
                            # prefer it over last_retrieval_query so recovery lands on correct op docs.
                            _ml_op_change = bool(re.search(
                                r"แก้ไข|เปลี่ยนแปลง|ยกเลิก|ปิดกิจการ|จดเลิก|ต่ออายุ|ชำรุด|สูญหาย|ใบแทน",
                                query or "",
                            ))
                            _last_q = (
                                query if (_ml_op_change and len(query) > 8)
                                else str(getattr(state, "last_retrieval_query", "") or "").strip()
                            )
                            if _last_q:
                                try:
                                    _cs_ml = state.get_collected_slots() or {}
                                    _ml_et = str(_cs_ml.get("entity_type") or "").strip()
                                    _ml_dept = str(_cs_ml.get("department") or "").strip()
                                    _ml_filter: dict | None = None
                                    if _ml_et and _ml_dept:
                                        _ml_filter = {"$and": [{"entity_type_normalized": _ml_et}, {"department": _ml_dept}]}
                                    elif _ml_et:
                                        _ml_filter = {"entity_type_normalized": _ml_et}
                                    elif _ml_dept:
                                        _ml_filter = {"department": _ml_dept}
                                    _recovery_docs = self._practical._retrieve_docs(_last_q, metadata_filter=_ml_filter)
                                    if _recovery_docs:
                                        _LOG.info(
                                            "[Supervisor] multi-license mismatch recovery: re-retrieved %d docs using last_q=%r filter=%s",
                                            len(_recovery_docs), _last_q[:50], _ml_filter,
                                        )
                                        state.current_docs = _recovery_docs
                                        state.context.pop("topic_slot_queue", None)
                                        return  # don't return False — let practical answer with recovered docs
                                except Exception as _ml_exc:
                                    _LOG.warning("[Supervisor] multi-license mismatch recovery failed: %s", _ml_exc)
                            state.context.pop("topic_slot_queue", None)
                            return False
                    # Query has domain overlap — don't force topic selection.
                    # Pass all docs to LLM and let it determine what's relevant.
                    _LOG.info(
                        "[Supervisor] multi-license substring match — skipping topic selector, letting LLM answer from %s",
                        license_types_seen,
                    )
                    return

        license_type = license_types_seen[0]

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

        # Discover all slot dimensions for this license_type.
        # Pass already-known entity_type so department counting uses only entity-filtered docs
        # (prevents กรมฯ appearing as a dept option for บุคคลธรรมดา just because it has
        # many นิติบุคคล docs — those inflate the count but are irrelevant to this user).
        _discover_et = (_cs_lq.get("entity_type") or _ctx_slots_lq.get("entity_type") or "").strip()
        all_slots = self._discover_slots_for_license(license_type, known_entity_type=_discover_et or None)
        if not all_slots:
            return

        # Step 1b: auto-infer department from query text if a known department name appears.
        # Prevents asking "which bank?" when user already said "ธนาคารไทยพาณิชย์" in the query.
        # Checks against available departments from the retrieved docs — no hardcoding.
        _available_depts_lq: list = []
        for _d_lq in (state.current_docs or []):
            if not isinstance(_d_lq, dict):
                continue
            _dv_lq = ((_d_lq.get("metadata") or {}).get("department") or "").strip()
            if _dv_lq and _dv_lq not in _available_depts_lq:
                _available_depts_lq.append(_dv_lq)
        _inferred_dept_lq = next((d for d in _available_depts_lq if d and d in query), None)
        if not _inferred_dept_lq:
            # Try abbreviated bank names via regex (e.g. "กสิกรไทย" → "ธนาคารกสิกรไทย").
            for _bp_lq, _bn_lq in self._BANK_DEPT_PATTERNS:
                if _bp_lq.search(query) and _bn_lq in _available_depts_lq:
                    _inferred_dept_lq = _bn_lq
                    _LOG.info("[Supervisor] legal_q department regex-inferred from query: %r", _inferred_dept_lq)
                    break
        if not _inferred_dept_lq:
            # Scan recent user messages — last_user_legal_query is overwritten before this runs.
            _recent_user_text_lq = " ".join(
                m.get("content", "")
                for m in (getattr(state, "messages", []) or [])[-6:]
                if m.get("role") == "user"
            )
            _inferred_dept_lq = next((d for d in _available_depts_lq if d and d in _recent_user_text_lq), None)
            if not _inferred_dept_lq:
                # Try abbreviated bank names in recent messages via regex.
                for _bp_lq2, _bn_lq2 in self._BANK_DEPT_PATTERNS:
                    if _bp_lq2.search(_recent_user_text_lq) and _bn_lq2 in _available_depts_lq:
                        _inferred_dept_lq = _bn_lq2
                        break
            if _inferred_dept_lq:
                _LOG.info(
                    "[Supervisor] legal_q department auto-inferred from recent messages: %r",
                    _inferred_dept_lq,
                )
        if _inferred_dept_lq:
            state.save_collected_slot("department", _inferred_dept_lq)
            state.context.setdefault("slots", {})["department"] = _inferred_dept_lq
            if "collected_slots" in state.context:
                state.context["collected_slots"]["department"] = _inferred_dept_lq
            collected_keys = collected_keys | {"department"}
            _LOG.info("[Supervisor] legal_q department auto-inferred: %r", _inferred_dept_lq)

        # Step 2: auto-infer entity_type if explicitly stated in query text.
        # IMPORTANT: always overwrite even when entity_type was already collected from a
        # PREVIOUS topic — user may explicitly switch entity type in a new question
        # (e.g. after answering QR Payment as นิติบุคคล, now asks about ทะเบียนพาณิชย์ บุคคลธรรมดา).
        # Without this, Academic would inherit stale entity_type and retrieve wrong docs.
        _inferred_et_lq = self._infer_entity_type_from_query(query)
        if _inferred_et_lq:
            _existing_et_lq = (_cs_lq.get("entity_type") or _ctx_slots_lq.get("entity_type") or "").strip()
            _existing_normalized = ""
            if _existing_et_lq:
                try:
                    from service.data_loader import DataLoader as _DL_et_ow
                    _existing_normalized = _DL_et_ow._normalize_entity_type(_existing_et_lq)
                except Exception:
                    _existing_normalized = _existing_et_lq
            if _existing_normalized != _inferred_et_lq:
                # Overwrite only when the current license_type differs from the last session's
                # license — this prevents false overwrite when "บริษัท" appears in passing
                # (e.g. "บริษัทใดออกใบอนุญาต") on the SAME topic the user was already on.
                _cur_license = (state.context or {}).get("last_topic", "").strip()
                _prev_license = (state.context or {}).get("last_retrieval_query", "")[:60]
                _topic_changed = bool(license_type) and (license_type != _cur_license)
                # Also allow overwrite when entity keyword appears near self-referencing words
                _SELF_REF_RE = re.compile(r"ร้านของ(เรา|ผม|ฉัน|คุณ)|ธุรกิจของ|กิจการของ|เปิดในนาม|จดใน\s*นาม|รูปแบบ(บุคคล|นิติ)|แบบบุคคล|แบบนิติ")
                _has_self_ref = bool(_SELF_REF_RE.search(query))
                if _topic_changed or _has_self_ref or not _existing_normalized:
                    state.save_collected_slot("entity_type", _inferred_et_lq)
                    state.context.setdefault("slots", {})["entity_type"] = _inferred_et_lq
                    if "collected_slots" in state.context:
                        state.context["collected_slots"]["entity_type"] = _inferred_et_lq
                    _LOG.info(
                        "[Supervisor] legal_q entity_type overwrite: %r → %r (topic_changed=%s self_ref=%s)",
                        _existing_normalized, _inferred_et_lq, _topic_changed, _has_self_ref,
                    )
                else:
                    _LOG.info(
                        "[Supervisor] legal_q entity_type infer mismatch but same topic — keeping %r (query=%r)",
                        _existing_normalized, query[:60],
                    )
            else:
                _LOG.info("[Supervisor] legal_q entity_type auto-inferred from query: %r (unchanged)", _inferred_et_lq)
            collected_keys = collected_keys | {"entity_type"}
        # Step 3: skip entity_type/registration_type for universal-fact queries
        elif not self._entity_slot_needed(query):
            all_slots = [s for s in all_slots if s["key"] not in ("entity_type", "registration_type")]
            _LOG.debug("[Supervisor] legal_q entity_type/registration_type skipped — universal-fact query: %r", query[:60])

        # Step 4: auto-infer location from query text (Bug 1)
        _location_slot_lq = next((s for s in all_slots if s.get("key") == "location"), None)
        if _location_slot_lq and "location" not in collected_keys:
            _loc_opts_lq = _location_slot_lq.get("options") or []
            _inferred_loc_lq = None
            if self._LOCATION_BKK_RE.search(query):
                _inferred_loc_lq = next((o for o in _loc_opts_lq if "กรุงเทพ" in o), "กรุงเทพฯ")
            elif self._LOCATION_METRO_RE.search(query):
                # ปริมณฑล province: only map to กรุงเทพ if the option explicitly includes "ปริมณฑล"
                _metro_opt_lq = next((o for o in _loc_opts_lq if "ปริมณฑล" in o), None)
                if _metro_opt_lq:
                    _inferred_loc_lq = _metro_opt_lq
                # else: leave None → fall through to LLM slot mapper
            elif self._LOCATION_PROVINCE_RE.search(query):
                _inferred_loc_lq = next((o for o in _loc_opts_lq if "จังหวัด" in o or "ต่าง" in o), "ต่างจังหวัด")
            if _inferred_loc_lq:
                state.save_collected_slot("location", _inferred_loc_lq)
                state.context.setdefault("slots", {})["location"] = _inferred_loc_lq
                if "collected_slots" in state.context:
                    state.context["collected_slots"]["location"] = _inferred_loc_lq
                collected_keys = collected_keys | {"location"}
                _LOG.info("[Supervisor] legal_q location auto-inferred from query: %r", _inferred_loc_lq)

        # Step 5: auto-infer shop_area_type from query text (Bug 2)
        _area_slot_lq = next((s for s in all_slots if s.get("key") == "shop_area_type"), None)
        if _area_slot_lq and "shop_area_type" not in collected_keys:
            _area_opts_lq = _area_slot_lq.get("options") or []
            _inferred_area_lq = None
            if self._AREA_SMALL_RE.search(query):
                _inferred_area_lq = next((o for o in _area_opts_lq if "น้อยกว่า" in o or "ไม่เกิน" in o), None)
            elif self._AREA_LARGE_RE.search(query):
                _inferred_area_lq = next((o for o in _area_opts_lq if "มากกว่า" in o or "เกิน" in o), None)
            if _inferred_area_lq is None:  # numeric: "234 ตรม", "8.9 ตร.ม.", "300 ตารางเมตร"
                _sqm_m_lq = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:ตร\.?\s*ม\.?|ตารางเมตร)', query)
                if _sqm_m_lq:
                    try:
                        _sqm_val_lq = float(_sqm_m_lq.group(1).replace(",", "."))
                        if _sqm_val_lq < 200:
                            _inferred_area_lq = next((o for o in _area_opts_lq if "น้อยกว่า" in o or "ไม่เกิน" in o), None)
                        else:
                            _inferred_area_lq = next((o for o in _area_opts_lq if "มากกว่า" in o or "เกิน" in o), None)
                    except ValueError:
                        pass
            if _inferred_area_lq:
                state.save_collected_slot("shop_area_type", _inferred_area_lq)
                state.context.setdefault("slots", {})["shop_area_type"] = _inferred_area_lq
                if "collected_slots" in state.context:
                    state.context["collected_slots"]["shop_area_type"] = _inferred_area_lq
                collected_keys = collected_keys | {"shop_area_type"}
                _LOG.info("[Supervisor] legal_q shop_area_type auto-inferred from query: %r", _inferred_area_lq)

        # Step 6: auto-infer operation_group from query text (same pattern as Steps 2-5 above)
        # When query clearly implies one operation (e.g. "ขึ้นทะเบียน"→new, "เปลี่ยนแปลง"→edit),
        # save it and re-retrieve focused docs rather than asking the user to pick from a menu.
        # Risk-2 mitigation: also override a STALE stored operation_group when current query
        # implies a DIFFERENT operation (e.g. follow-up "ถ้าจะเปลี่ยนแปลงข้อมูลล่ะ" after
        # turn 1 stored "ยื่นขอใหม่").
        _og_slot_6 = next((s for s in all_slots if s.get("key") == "operation_group"), None)
        if _og_slot_6:
            _og_opts_6 = _og_slot_6.get("options") or []
            _inferred_og_6 = self._infer_operation_group_from_query(query, _og_opts_6, state)
            if _inferred_og_6:
                _cur_og_6 = (
                    _cs_lq.get("operation_group")
                    or _ctx_slots_lq.get("operation_group")
                    or ""
                ).strip()
                _og_changed_6 = (_inferred_og_6 != _cur_og_6)
                if _og_changed_6:
                    if hasattr(state, "save_collected_slot"):
                        state.save_collected_slot("operation_group", _inferred_og_6)
                    if state.context is not None:
                        state.context.setdefault("slots", {})["operation_group"] = _inferred_og_6
                        state.context.setdefault("slots", {})["confirmed_operation"] = _inferred_og_6
                        if "collected_slots" in state.context:
                            state.context["collected_slots"]["operation_group"] = _inferred_og_6
                    _raw_ops_6 = (_og_slot_6.get("raw_op_map") or {}).get(_inferred_og_6) or []
                    _op_pfx_6 = " ".join(_raw_ops_6[:2]) if _raw_ops_6 else _inferred_og_6
                    _enrich_q_6 = " ".join(filter(None, [query, _op_pfx_6])).strip()
                    try:
                        _op_docs_6 = self._practical._retrieve_docs(
                            _enrich_q_6,
                            metadata_filter={"license_type": license_type} if license_type else None,
                            max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                        )
                        if _op_docs_6:
                            state.current_docs = _op_docs_6
                            state.last_retrieval_query = _enrich_q_6
                            _LOG.info(
                                "[Supervisor] legal_q Step6 op-inferred re-retrieve: op=%r docs=%d",
                                _inferred_og_6, len(_op_docs_6),
                            )
                    except Exception as _e_og_6:
                        _LOG.warning("[Supervisor] legal_q Step6 op re-retrieve failed: %s", _e_og_6)
                # Always suppress the menu question when we can infer the operation
                collected_keys = collected_keys | {"operation_group"}
                _LOG.info(
                    "[Supervisor] legal_q Step6 operation_group auto-inferred=%r changed=%s — skipping ask",
                    _inferred_og_6, _og_changed_6,
                )

        # Step 7: detect operation_topic switch when the slot was already collected.
        # When user explicitly mentions a DIFFERENT operation_topic (e.g. says "SAN PLUS"
        # after previously collecting "SAN"), update the slot and re-retrieve focused docs.
        # Checks Thai string containment, English tokens, and common Thai loanwords (PLUS→พลัส).
        if "operation_topic" in collected_keys:
            _ot_slot_7 = next((s for s in all_slots if s.get("key") == "operation_topic"), None)
            if _ot_slot_7:
                _ot_opts_7 = _ot_slot_7.get("options") or []
                _cur_ot_7 = (
                    _cs_lq.get("operation_topic") or _ctx_slots_lq.get("operation_topic") or ""
                ).strip()
                _cur_eng_toks_7 = {t.lower() for t in re.findall(r"[A-Za-z]{2,}", _cur_ot_7)}
                _ENG_THAI_7 = {"PLUS": "พลัส", "SUPER": "ซุปเปอร์", "PRO": "โปร", "MINI": "มินิ"}
                _new_ot_7 = (
                    next((o for o in _ot_opts_7 if o and o != _cur_ot_7 and o in query), None)
                    or next(
                        (o for o in _ot_opts_7
                         if o and o != _cur_ot_7
                         and any(tok.lower() in query.lower()
                                 for tok in re.findall(r"[A-Za-z]{2,}", o)
                                 if tok.lower() not in _cur_eng_toks_7)),
                        None,
                    )
                    or next(
                        (o for o in _ot_opts_7
                         if o and o != _cur_ot_7
                         and any(
                             _ENG_THAI_7.get(tok.upper(), "") in query
                             for tok in re.findall(r"[A-Za-z]{2,}", o)
                             if tok.upper() not in {t.upper() for t in _cur_eng_toks_7}
                         )),
                        None,
                    )
                )
                if _new_ot_7:
                    if hasattr(state, "save_collected_slot"):
                        state.save_collected_slot("operation_topic", _new_ot_7)
                    if state.context is not None:
                        state.context.setdefault("slots", {})["operation_topic"] = _new_ot_7
                        if "collected_slots" in state.context:
                            state.context["collected_slots"]["operation_topic"] = _new_ot_7
                    try:
                        _ot_docs_7 = self._practical._retrieve_docs(
                            f"{query} {_new_ot_7}".strip(),
                            metadata_filter={"operation_topic": _new_ot_7},
                            max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                        )
                        if not _ot_docs_7 and license_type:
                            _ot_docs_7 = self._practical._retrieve_docs(
                                f"{query} {_new_ot_7}".strip(),
                                metadata_filter={"license_type": license_type},
                                max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                            )
                        if _ot_docs_7:
                            state.current_docs = _ot_docs_7
                            state.last_retrieval_query = f"{query} {_new_ot_7}".strip()
                            _LOG.info(
                                "[Supervisor] Step7 op_topic switch: %r → %r docs=%d",
                                _cur_ot_7, _new_ot_7, len(_ot_docs_7),
                            )
                    except Exception as _e_ot_7:
                        _LOG.warning("[Supervisor] Step7 op_topic re-retrieve failed: %s", _e_ot_7)

        remaining = [s for s in all_slots if s["key"] not in collected_keys]

        if not remaining:
            return

        # Link request: only ask department (if present) — skip entity_type/registration_type.
        # User just wants the link; knowing the bank is enough to retrieve specific links.
        if _is_link_req:
            remaining = [s for s in remaining if s["key"] == "department"]
            if not remaining:
                _LOG.debug("[Supervisor] link query: dept already known — skipping slot queue")
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
            # Re-retrieve docs with entity filter for better context.
            # Safety net C: detect topic_group via broad (unfiltered) pass-1 retrieve,
            # then add topic_group to the entity filter → prevents cross-topic docs from
            # contaminating slot discovery (e.g. "ปิดงบบัญชี นิติบุคคล" no longer mixes
            # ใบทะเบียนพาณิชย์ / VAT docs into the result set).
            _base_q_lq = (state.context or {}).get("last_user_legal_query") or ""
            if not _base_q_lq:
                # Single-topic flow: last_user_legal_query is often empty.
                # Fall back to last_retrieval_query (the actual topic query from the previous turn)
                # so entity-enriched retrieval uses QR Payment / prior topic context, not just entity name alone.
                _lrq_lq = (getattr(state, "last_retrieval_query", None) or "").strip()
                if _lrq_lq and not _lrq_lq.startswith("__"):
                    _base_q_lq = _lrq_lq
            _enriched_q_lq = f"{_base_q_lq} {_known_entity_lq}".strip()

            # Pass-1: detect topic_group from broad retrieve (k=5, no filter)
            _topic_group_lq: Optional[str] = None
            try:
                _vstore_lq = getattr(self._practical.retriever, "vectorstore", None)
                if _vstore_lq is not None:
                    _lq_query = _base_q_lq or _enriched_q_lq
                    if getattr(conf, "HYBRID_SEARCH_ENABLED", False):
                        from utils.hybrid_retriever import hybrid_scored_search
                        _broad_lq = [d for d, _ in hybrid_scored_search(
                            _vstore_lq, _lq_query, k=5,
                            rrf_k=int(getattr(conf, "HYBRID_RRF_K", 60)),
                        )]
                    else:
                        _tmp_broad_lq = _vstore_lq.as_retriever(search_kwargs={"k": 5})
                        _broad_lq = _tmp_broad_lq.invoke(_lq_query)
                    _grp_counts_lq: dict = {}
                    for _bd in (_broad_lq or []):
                        _tg = (getattr(_bd, "metadata", {}) or {}).get("topic_group", "").strip()
                        if _tg and _tg != "อื่นๆ":
                            _grp_counts_lq[_tg] = _grp_counts_lq.get(_tg, 0) + 1
                    if _grp_counts_lq:
                        _total_lq = sum(_grp_counts_lq.values())
                        _best_grp_lq = max(_grp_counts_lq, key=_grp_counts_lq.__getitem__)
                        if _grp_counts_lq[_best_grp_lq] / _total_lq >= 0.60:
                            _topic_group_lq = _best_grp_lq
                            _LOG.info(
                                "[Supervisor] Safety net C: detected topic_group=%r (%.0f%%) for query=%r",
                                _topic_group_lq, _grp_counts_lq[_best_grp_lq] / _total_lq * 100,
                                (_base_q_lq or _enriched_q_lq)[:60],
                            )
            except Exception as _e_broad_lq:
                _LOG.warning("[Supervisor] Safety net C broad retrieve failed: %s", _e_broad_lq)

            # Build filter: entity_type ($or with '') + optional topic_group + optional department
            # IMPORTANT: include _inferred_dept_lq so retrieval stays within the bank the user
            # is asking about (e.g. ธนาคารไทยพาณิชย์) instead of mixing docs from all banks.
            _and_parts_lq: list = [
                {"$or": [
                    {"entity_type_normalized": _known_entity_lq},
                    {"entity_type_normalized": ""},
                ]}
            ]
            if _topic_group_lq:
                _and_parts_lq.append({"topic_group": _topic_group_lq})
            if _inferred_dept_lq:
                _and_parts_lq.append({"department": _inferred_dept_lq})
            _meta_filter_lq: dict = (
                _and_parts_lq[0] if len(_and_parts_lq) == 1 else {"$and": _and_parts_lq}
            )

            try:
                state.current_docs = self._practical._retrieve_docs(
                    _enriched_q_lq,
                    metadata_filter=_meta_filter_lq,
                )
                state.last_retrieval_query = _enriched_q_lq
                _LOG.info(
                    "[Supervisor] legal_q entity-enriched retrieval: entity=%r group=%r docs=%d",
                    _known_entity_lq, _topic_group_lq, len(state.current_docs),
                )
            except Exception as _e_lq:
                _LOG.warning("[Supervisor] legal_q entity retrieval failed: %s", _e_lq)

            # Filter registration_type options to entity-specific ones
            _filtered_lq: List[Dict] = []
            for _slot_lq in remaining:
                if _slot_lq.get("key") == "registration_type":
                    _ert_lq = self._get_registration_types_for_entity(license_type, _known_entity_lq)
                    if len(_ert_lq) >= 2:
                        # Auto-infer registration_type from query text — prevents re-asking when
                        # user already stated it (e.g. "การจดทะเบียนพาณิชย์ บริษัทจำกัด").
                        # Scan both the current query and recent user messages (same pattern as
                        # department auto-infer) in case the value was stated in a prior turn.
                        _query_for_rt_lq = _base_q_lq or query
                        _recent_user_rt_lq = " ".join(
                            m.get("content", "")
                            for m in (getattr(state, "messages", []) or [])[-6:]
                            if m.get("role") == "user"
                        )
                        _inferred_rt_lq = next(
                            (o for o in _ert_lq if o and o in _query_for_rt_lq),
                            None,
                        ) or next(
                            (o for o in _ert_lq if o and o in _recent_user_rt_lq),
                            None,
                        )
                        if _inferred_rt_lq:
                            state.save_collected_slot("registration_type", _inferred_rt_lq)
                            state.context.setdefault("slots", {})["registration_type"] = _inferred_rt_lq
                            if "collected_slots" in state.context:
                                state.context["collected_slots"]["registration_type"] = _inferred_rt_lq
                            collected_keys = collected_keys | {"registration_type"}
                            _LOG.info(
                                "[Supervisor] legal_q registration_type auto-inferred from query: %r",
                                _inferred_rt_lq,
                            )
                        else:
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

        # Step: if location is already collected AND docs have location-conditional content,
        # push the known location into context["slots"] so the LLM sees it and shows only
        # the relevant timeline (e.g. only "กรุงเทพฯ: 8-14 วัน", not both).
        # This is needed when location was collected in a previous Academic flow — the value
        # is in collected_slots but NOT yet in context["slots"], so LLM doesn't see it.
        if "location" in collected_keys:
            _known_loc_val = (
                _cs_lq.get("location")
                or _cs_lq.get("location_scope")
                or _ctx_slots_lq.get("location")
                or _ctx_slots_lq.get("location_scope")
                or ""
            ).strip()
            if _known_loc_val:
                state.context.setdefault("slots", {})["location"] = _known_loc_val
                _LOG.info(
                    "[Supervisor] legal_q location already known=%r — injected into context['slots'] for LLM",
                    _known_loc_val,
                )

        # Append operation_group slot (skip if already collected).
        # NOTE: intentionally outside the `if "location"` block — operation_group applies to
        # ALL licenses regardless of whether location is known (e.g. ทะเบียนพาณิชย์, VAT,
        # ประกันสังคม have no location slot but do have multiple operation groups).
        _op_res_lq = self._get_operation_groups_for_entity(license_type, _known_entity_lq)
        _op_grps_lq, _raw_op_map_lq = (
            _op_res_lq if isinstance(_op_res_lq, tuple) else (_op_res_lq, {})
        )
        # Stale operation_group check: if already collected but not valid for new topic, clear it.
        # Example: user asked "จดทะเบียน" (op=จดทะเบียน) then asks new topic with "แก้ไข"/"ต่ออายุ"
        # ops — the stale "จดทะเบียน" suppresses the new topic's operation menu.
        if "operation_group" in collected_keys and _op_grps_lq:
            _old_og = (_cs_lq.get("operation_group") or _ctx_slots_lq.get("operation_group") or "").strip()
            _og_valid_lq = bool(_old_og) and any(
                _old_og in grp or grp in _old_og for grp in _op_grps_lq
            )
            if not _og_valid_lq:
                _LOG.info(
                    "[Supervisor] legal_q stale operation_group=%r not in new topic ops=%s — clearing",
                    _old_og, _op_grps_lq[:3],
                )
                if isinstance(state.context, dict):
                    (state.context.get("slots") or {}).pop("operation_group", None)
                    (state.context.get("slots") or {}).pop("confirmed_operation", None)
                    if isinstance(state.context.get("collected_slots"), dict):
                        state.context["collected_slots"].pop("operation_group", None)
                collected_keys = collected_keys - {"operation_group"}
        if len(_op_grps_lq) >= 2 and "operation_group" not in collected_keys:
            # Use the original legal query (last_user_legal_query) for inference —
            # not the current turn's input, which may be a slot answer like "บุคคลธรรมดา"
            # that no longer contains the operation keyword (e.g. "จดทะเบียน", "แก้ไข").
            _query_for_infer_lq = _base_q_lq or query
            _inferred_lq = self._infer_operation_group_from_query(_query_for_infer_lq, _op_grps_lq, state)
            if _inferred_lq:
                state.save_collected_slot("operation_group", _inferred_lq)
                state.context.setdefault("slots", {})["confirmed_operation"] = _inferred_lq
                _LOG.info("[Supervisor] legal_q operation_group auto-inferred=%r — skipping ask", _inferred_lq)
                # Re-retrieve with operation label in query so similarity ranks op-specific docs higher.
                # Without this, entity-filtered docs include ALL operations mixed — LLM can't find
                # the specific operation requested (e.g. "แก้ไข" vs "เปิดใหม่").
                _raw_ops_lq2 = (_raw_op_map_lq or {}).get(_inferred_lq) or []
                _op_pfx_lq2 = " ".join(_raw_ops_lq2[:2]) if _raw_ops_lq2 else _inferred_lq
                _enrich_op_q = " ".join(filter(None, [_base_q_lq, _known_entity_lq, _op_pfx_lq2])).strip()
                try:
                    _op_infer_filter: dict = {"entity_type_normalized": _known_entity_lq}
                    if license_type:
                        _op_infer_filter["license_type"] = license_type
                    _op_infer_docs = self._practical._retrieve_docs(
                        _enrich_op_q,
                        metadata_filter=_op_infer_filter,
                        max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                    )
                    if _op_infer_docs:
                        # Safety: verify returned docs are for the correct entity.
                        # If _scored_search's filter fails silently and falls back to
                        # unfiltered retrieval, we must not overwrite already-correct docs.
                        _entity_wrong = [
                            d for d in _op_infer_docs
                            if (d.get("metadata") or {}).get("entity_type_normalized", "") not in ("", _known_entity_lq)
                        ]
                        if len(_entity_wrong) == len(_op_infer_docs):
                            _LOG.warning(
                                "[Supervisor] legal_q op-inferred retrieval: all %d docs are wrong entity — keeping existing docs",
                                len(_op_infer_docs),
                            )
                        else:
                            # Level 1: post-filter by operation_topic to remove docs
                            # whose operation conflicts with the inferred operation group.
                            # e.g. inferred="สมัครใหม่" → exclude docs with operation_topic
                            # containing "ต่ออายุ|แก้ไข|เปลี่ยนแปลง|ยกเลิก".
                            # Guard: keep all if filter leaves nothing (avoids empty state.current_docs).
                            _op_excl_for_infer: Optional[str] = None
                            _inferred_lq_lower = (_inferred_lq or "").lower()
                            if any(kw in _inferred_lq_lower for kw in ("ใหม่", "จัดตั้ง", "จดทะเบียน", "ยื่นขอ", "สมัคร", "เปิด")):
                                _op_excl_for_infer = r"ต่ออายุ|แก้ไข|เปลี่ยนแปลง|ยกเลิก|ปิดกิจการ"
                            elif any(kw in _inferred_lq_lower for kw in ("แก้ไข", "เปลี่ยนแปลง")):
                                _op_excl_for_infer = r"ต่ออายุ|ยกเลิก|ปิดกิจการ|ใหม่"
                            elif any(kw in _inferred_lq_lower for kw in ("ยกเลิก", "ปิด")):
                                _op_excl_for_infer = r"ต่ออายุ|แก้ไข|เปลี่ยนแปลง|ใหม่"
                            elif any(kw in _inferred_lq_lower for kw in ("ต่ออายุ", "ต่อ")):
                                _op_excl_for_infer = r"แก้ไข|เปลี่ยนแปลง|ยกเลิก|ใหม่"
                            if _op_excl_for_infer:
                                _op_filtered = [
                                    d for d in _op_infer_docs
                                    if not re.search(_op_excl_for_infer, str((d.get("metadata") or {}).get("operation_topic") or ""))
                                ]
                                if _op_filtered:
                                    _LOG.info(
                                        "[Supervisor] op-topic post-filter: %d → %d docs (excl=%r)",
                                        len(_op_infer_docs), len(_op_filtered), _op_excl_for_infer,
                                    )
                                    _op_infer_docs = _op_filtered
                            state.current_docs = _op_infer_docs
                            state.last_retrieval_query = _enrich_op_q
                            _LOG.info(
                                "[Supervisor] legal_q op-inferred retrieval: entity=%r op=%r docs=%d",
                                _known_entity_lq, _inferred_lq, len(_op_infer_docs),
                            )
                except Exception as _e_op_infer:
                    _LOG.warning("[Supervisor] legal_q op-inferred retrieval failed: %s", _e_op_infer)
            else:
                # Guard against duplicate: _discover_slots_for_license may have already added
                # operation_group — don't add it twice or the bot asks the same question twice.
                if not any(s.get("key") == "operation_group" for s in remaining):
                    remaining.append({
                        "key": "operation_group",
                        "options": _op_grps_lq,
                        "question": self._Q_OPERATION_GROUP,
                        "raw_op_map": _raw_op_map_lq,
                    })
                    _LOG.info("[Supervisor] legal_q operation_group appended → %s", _op_grps_lq)
                else:
                    _LOG.info("[Supervisor] legal_q operation_group already in remaining — skip duplicate")
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

        # Area measurement check — must run BEFORE _parse_indices because a decimal like
        # "7.8 ตรม" gets split into [7, 8] by _ANY_NUMBER_RE (\b(\d{1,2})\b), which then
        # produces valid=[] and triggers "เลขที่เลือกไม่อยู่ในช่วงตัวเลือกครับ" before any
        # free-text mapping logic is reached.
        if key in ("shop_area_type", "area_size"):
            _sqm_early_m = re.search(
                r'(\d+(?:[.,]\d+)?)\s*(?:ตร\.?\s*ม\.?|ตารางเมตร)',
                raw,
            )
            if _sqm_early_m:
                _user_sqm_early: Optional[float] = None
                try:
                    _user_sqm_early = float(_sqm_early_m.group(1).replace(",", "."))
                except ValueError:
                    pass
                if _user_sqm_early is not None:
                    for _opt_e in options:
                        _opt_es = str(_opt_e).strip()
                        _thresh_em = re.search(
                            r'(\d+(?:[.,]\d+)?)\s*(?:ตร\.?\s*ม\.?|ตารางเมตร)',
                            _opt_es,
                        )
                        if not _thresh_em:
                            continue
                        try:
                            _thresh_e = float(_thresh_em.group(1).replace(",", "."))
                        except ValueError:
                            continue
                        if ("น้อยกว่า" in _opt_es) and _user_sqm_early < _thresh_e:
                            return _opt_es, None
                        if ("ไม่เกิน" in _opt_es) and _user_sqm_early <= _thresh_e:
                            return _opt_es, None
                        if ("มากกว่า" in _opt_es or "เกิน" in _opt_es) and _user_sqm_early > _thresh_e:
                            return _opt_es, None

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
                    # For topic slot: allow user to select multiple items even when
                    # allow_multi=False (e.g. "ทั้ง 1 และ 2", "1 และ 2").
                    # Return comma-joined selections so multi-topic path can handle them.
                    if key == "topic" and len(valid) > 1:
                        texts = [str(options[i - 1]).strip() for i in valid if 1 <= i <= len(options)]
                        texts = [x for x in texts if x]
                        if texts:
                            return ", ".join(texts), None
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

        # 1b) Stripped contains match — strip Thai politeness particles and common
        # "desire/connector" prefixes so that "ต้องการของกสิกรค่ะ" → "กสิกร" → found in
        # "ธนาคารกสิกรไทย", and "สมัครกับกสิกรไทย" → "กสิกรไทย" → found in option.
        # Only runs when stripped result is different from original (i.e. something was removed).
        _TH_TRAIL_PARTICLE_RE = re.compile(
            r"(ครับ|คับ|ค่ะ|คะ|จ้า|จ้ะ|นะ|นะคะ|นะครับ)$"
        )
        _TH_HEAD_FILLER_RE = re.compile(
            r"^(ต้องการ(?:จะ|ของ|แบบ|เป็น)?|สมัครกับ|สมัคร|อยากได้|เลือก)(ของ|กับ|แบบ)?"
        )
        _raw_stripped = _TH_TRAIL_PARTICLE_RE.sub("", raw_norm).strip()
        _raw_stripped = _TH_HEAD_FILLER_RE.sub("", _raw_stripped).strip()
        if _raw_stripped and _raw_stripped != raw_norm:
            for opt in options:
                s = str(opt).strip()
                if not s:
                    continue
                s_norm = self._normalize_for_intent(s)
                if _raw_stripped == s_norm:
                    return s, None
                if s_norm and (_raw_stripped in s_norm or s_norm in _raw_stripped):
                    return s, None

        # 1c) Numeric area comparison — for shop_area_type/area_size slots where the user states
        # a specific numeric value (e.g. "8.9 ตรม"), compare deterministically against the
        # threshold embedded in each option text. This avoids LLM hallucination on simple math
        # (e.g. "น้อยกว่า 200 ตารางเมตร" requires knowing 8.9 < 200 — LLM sometimes refuses
        # to infer this at confidence ≥0.60 and returns invalid_reply).
        if key in ("shop_area_type", "area_size"):
            _user_sqm: Optional[float] = None
            _sqm_user_m = re.search(
                r'(\d+(?:[.,]\d+)?)\s*(?:ตร\.?\s*ม\.?|ตารางเมตร)',
                raw,
            )
            if _sqm_user_m:
                try:
                    _user_sqm = float(_sqm_user_m.group(1).replace(",", "."))
                except ValueError:
                    pass
            if _user_sqm is not None:
                for _opt in options:
                    _opt_s = str(_opt).strip()
                    _thresh_m = re.search(
                        r'(\d+(?:[.,]\d+)?)\s*(?:ตร\.?\s*ม\.?|ตารางเมตร)',
                        _opt_s,
                    )
                    if not _thresh_m:
                        continue
                    try:
                        _thresh = float(_thresh_m.group(1).replace(",", "."))
                    except ValueError:
                        continue
                    _opt_l = _opt_s
                    if ("น้อยกว่า" in _opt_l) and _user_sqm < _thresh:
                        return _opt_s, None
                    if ("ไม่เกิน" in _opt_l) and _user_sqm <= _thresh:
                        return _opt_s, None
                    if ("มากกว่า" in _opt_l or "เกิน" in _opt_l) and _user_sqm > _thresh:
                        return _opt_s, None

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
                    # map back to options if same/contains; require ≥4 chars to avoid
                    # false-matches on short Thai syllables like "นิติ", "บุคคล"
                    ct = self._normalize_for_intent(choice_text)
                    for opt in options:
                        s = str(opt).strip()
                        if not s:
                            continue
                        s_norm = self._normalize_for_intent(s)
                        if (ct == s_norm
                                or (len(ct) >= 4 and ct in s_norm)
                                or (len(s_norm) >= 4 and s_norm in ct)):
                            return s, None

        # 4) Character bi-gram overlap fuzzy match (fallback when LLM fails/low-confidence)
        # Works for Thai text where word boundaries are ambiguous.
        # Helps when user types natural text (e.g. "แก้ไขหุ้นส่วน") for option "การแก้ไขข้อมูลหุ้นส่วน"
        # Threshold 0.50 (up from 0.30) to reduce false-matches on short/common Thai bigrams.
        fuzzy_match = self._fuzzy_match_option(raw, options, threshold=0.50)
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

        # Extract pending slot key early so ALL guards below can apply _SLOT_REPLY_ALWAYS exemption.
        # Slots in this set expect natural-language answers (e.g. "นิติค่ะ", "กสิกรครับ") that
        # may superficially resemble other intents — they must never be blocked by switch/persona
        # detection, greeting detection, or legal-question guards.
        _ps = ctx.get("pending_slot")
        pending_key = str((_ps.get("key") if isinstance(_ps, dict) else None) or "")
        _SLOT_REPLY_ALWAYS = {"topic", "operation_group", "operation_sub_type", "operation_topic", "registration_type", "department", "entity_type", "location", "shop_area_type", "area_size"}

        # Switch/persona guards: exempt _SLOT_REPLY_ALWAYS slots so that e.g. typing a bank name
        # ("kasikorn") while answering the department slot is never mis-classified as a persona
        # switch request. Without this guard the LLM fallback inside _looks_like_switch_without_target
        # can return True for short English words → slot routing bypassed → deflect instead of fill.
        if pending_key not in _SLOT_REPLY_ALWAYS and self._looks_like_switch_without_target(user_input, state):
            return False
        if pending_key not in _SLOT_REPLY_ALWAYS and self._infer_target_persona_from_text(user_input) in {"academic", "practical"}:
            return False

        if self._looks_like_mode_status_query(user_input, state):
            return False

        if self._STYLE_LIKELY_RE.search(user_input or "") and not self._LIKELY_SELECTION_RE.match((user_input or "").strip()):
            if not self._ANY_NUMBER_RE.search(user_input or ""):
                return False

        if pending_key not in _SLOT_REPLY_ALWAYS and (
            self._looks_like_greeting_or_thanks(user_input, state) or self._is_noise(user_input)
        ):
            return False

        # FIX #1: A new substantive legal question must NEVER be consumed as a slot answer.
        # This guard lets step 2.9 (_looks_like_legal_question) handle it properly.
        # Exemptions:
        #   - "topic" slot: a legal phrase IS a valid topic selection
        #   - "operation_group" slot: options contain legal keywords (จดทะเบียน, ต่ออายุ, etc.)
        #   - "registration_type" slot: options contain terms like บริษัทจำกัด that match LEGAL_SIGNAL_RE
        #   - "department" slot: user naming a dept (e.g. "กรมพัฒนาธุรกิจการค้า") triggers LEGAL_SIGNAL_RE
        #     but IS a valid slot answer and must not be treated as a new legal question
        #   - "entity_type" slot: answers "นิติบุคคล"/"บุคคลธรรมดา" trigger LEGAL_SIGNAL_RE
        #     but are always slot fills — treating them as new legal questions breaks license context
        if pending_key not in _SLOT_REPLY_ALWAYS and self._looks_like_legal_question(user_input, state):
            _LOG.info(
                "[Supervisor] pending_slot(%r) skipped — input looks like new legal question: %r",
                pending_key, (user_input or "")[:50],
            )
            return False

        # Secondary guard: long question-phrased inputs not caught by _looks_like_legal_question
        # (regex-based) are almost certainly new questions rather than slot answers.
        # The 8-word threshold preserves short affirmative answers such as "บุคคลธรรมดาครับ"
        # while intercepting full sentences like "แล้วถ้าจะเปิดร้านเหล้า ต้องใช้อะไรบ้าง".
        if pending_key not in _SLOT_REPLY_ALWAYS:
            _word_count = len((user_input or "").split())
            if (
                _word_count >= 8
                and not self._LIKELY_SELECTION_RE.match(user_input or "")
                and self._QUESTION_MARKERS_RE.search(user_input or "")
            ):
                _LOG.info(
                    "[Supervisor] pending_slot(%r) skipped — long question input (%d words): %r",
                    pending_key, _word_count, (user_input or "")[:60],
                )
                return False

        # Special case: "department" slot options are bank/org names (e.g. ธนาคารไทยพาณิชย์).
        # "department" is in _SLOT_REPLY_ALWAYS to prevent bank names from being misread as
        # legal questions. But if the input contains NONE of the department option texts AND
        # looks like a new legal question, the user has changed topic — clear pending_slot and
        # let it route as a new question. Without this, users are stuck in an infinite loop.
        if pending_key == "department":
            _dept_opts_esc = [str(o).strip() for o in ((_ps.get("options") if isinstance(_ps, dict) else None) or []) if o]
            _user_in_dept = (user_input or "")
            # Full match: exact option text found in user input
            _matches_dept_opt = any(opt and opt in _user_in_dept for opt in _dept_opts_esc)

            # Partial match: user abbreviated the option (e.g. "กสิกร" for "ธนาคารกสิกรไทย").
            # Strip common org-type prefix from option, then try progressively shorter prefixes
            # of the core name to check if any appear in user input.
            # Min 3 chars to avoid spurious single-syllable matches.
            if not _matches_dept_opt:
                _DEPT_ORG_PREFIX_RE = re.compile(r"^(ธนาคาร|สำนักงาน|กรม|บริษัท|ห้างหุ้นส่วน|หจก\.)\s*")
                _DEPT_MIN_CORE = 3
                for _opt_d in _dept_opts_esc:
                    _core_d = _DEPT_ORG_PREFIX_RE.sub("", _opt_d).strip()
                    if not _core_d or len(_core_d) < _DEPT_MIN_CORE:
                        continue
                    # Try longest prefix first so "กสิกรไทย" is tried before "กสิกร"
                    for _trim in range(len(_core_d), _DEPT_MIN_CORE - 1, -1):
                        if _core_d[:_trim] in _user_in_dept:
                            _matches_dept_opt = True
                            break
                    if _matches_dept_opt:
                        break

            if not _matches_dept_opt and self._looks_like_legal_question(user_input, state):
                _LOG.info(
                    "[Supervisor] department slot skipped — new legal question (no dept name match): %r",
                    (user_input or "")[:60],
                )
                # Clear stale pending_slot so new question builds its own slot queue cleanly
                ctx.pop("pending_slot", None)
                return False

        # Special case: "topic" slot accepts legal phrases as menu selections, BUT a full
        # question sentence (informational/consequential question) should bypass the menu
        # and be handled as a new question. User can still pick from menu on next turn.
        if pending_key == "topic" and (
            self._INFO_Q_RE.search(user_input)
            or (not self._ACTION_Q_RE.search(user_input) and self._info_action_q_llm_check(user_input, state)[0])
        ):
            _LOG.info(
                "[Supervisor] topic slot skipped — full question bypasses menu: %r",
                (user_input or "")[:60],
            )
            return False

        # topic slot: require a number OR at least one legal/business keyword.
        # Fully off-topic inputs (e.g. "กินข้าวกับอะไรดี") have neither → bypass slot
        # and fall through to normal routing (→ unknown intent → deflect).
        # Also accept inputs that contain numbers with Thai text (e.g. "ทั้ง 1 และ 2")
        # because _LIKELY_SELECTION_RE only matches digit-only strings.
        if pending_key == "topic":
            if (
                not self._LIKELY_SELECTION_RE.match(user_input)
                and not self._LEGAL_SIGNAL_RE.search(user_input)
                and not self._ANY_NUMBER_RE.search(user_input)
            ):
                _LOG.info(
                    "[Supervisor] topic slot skipped — no legal signal or number in off-topic input: %r",
                    (user_input or "")[:60],
                )
                return False

        return self._looks_like_pending_slot_reply(user_input)

    # Area-size keywords used to detect shop-area dimension in registration_type
    _AREA_KEYWORDS: List[str] = ["น้อยกว่า", "มากกว่า", "ไม่เกิน", "เกิน", "ตารางเมตร"]

    def _validate_license_types_at_startup(self) -> None:
        """
        Cross-check _KW_OVERRIDE_LICENSE_TYPES against Chroma at startup.
        Logs WARNING for any hardcoded string with no matching docs — catches
        stale mappings immediately after a re-ingest without blocking startup.
        """
        try:
            coll = self._get_chroma_collection()
            if coll is None:
                return
            result = coll.get(include=["metadatas"])
            mds = result.get("metadatas") or []
            chroma_lt_set = {
                str(md.get("license_type") or "").strip()
                for md in mds
                if md.get("license_type")
            }
            mismatches = []
            for lt in self._KW_OVERRIDE_LICENSE_TYPES:
                if lt not in chroma_lt_set:
                    mismatches.append(lt)
                    _LOG.warning(
                        "[Supervisor] startup: KW_OVERRIDE license_type %r not found in Chroma "
                        "— slot discovery will silently return [] for this license. "
                        "Re-check source sheet or update _KW_OVERRIDE_LICENSE_TYPES.",
                        lt,
                    )
            if not mismatches:
                _LOG.info(
                    "[Supervisor] startup: all %d KW_OVERRIDE license types verified in Chroma ✓",
                    len(self._KW_OVERRIDE_LICENSE_TYPES),
                )
        except Exception:
            _LOG.warning("[Supervisor] startup: license type validation failed", exc_info=True)

    def _get_chroma_collection(self):
        """Return the raw Chroma collection object from the practical retriever, or None."""
        vectorstore = getattr(self._practical.retriever, "vectorstore", None)
        if vectorstore is None:
            return None
        return getattr(vectorstore, "_collection", None)

    def _discover_slots_for_license(self, license_type: str, known_entity_type: Optional[str] = None) -> List[Dict]:
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
            result = coll.get(where={"license_type": license_type.strip()}, include=["metadatas", "documents"])
            mds = result.get("metadatas") or []
            docs_content = result.get("documents") or []
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
            # When entity_type is already known, count depts only within that entity's docs
            # (plus universal '' docs). Prevents กรมฯ from appearing as a department option
            # for บุคคลธรรมดา just because it has many นิติบุคคล docs.
            _mds_for_dept = (
                [md for md in mds
                 if (md or {}).get("entity_type_normalized", "").strip() in ("", known_entity_type)]
                if known_entity_type else mds
            )
            for md in _mds_for_dept:
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
                seen_entities: set = set()
                for dept in qualified:
                    specific = {e for e in dept_entities.get(dept, set()) if e}
                    if len(specific) > 1:
                        return False  # this dept has mixed entity types → not derivable
                    if specific:
                        entity = next(iter(specific))
                        if entity in seen_entities:
                            return False  # two depts share same entity → entity alone can't pick dept
                        seen_entities.add(entity)
                return True
            _dept_is_entity_derived = (
                len(_dept_opts_qualified) >= 2
                and _dept_determined_by_entity(_dept_entities, _dept_opts_qualified)
            )
            if len(_dept_opts_qualified) >= 2 and not _dept_is_entity_derived:
                _all_dept_are_banks = all(
                    any(kw in d for kw in ("ธนาคาร", "Bank", "bank"))
                    for d in _dept_opts_qualified
                )
                if _all_dept_are_banks:
                    # Bank selection is a meaningful user choice (different banks = different procedures).
                    slots.append({
                        "key": "department",
                        "options": sorted(_dept_opts_qualified),
                        "question": "ต้องการสมัครกับธนาคารใดครับ?",
                    })
                    seen_keys.add("department")
                    _LOG.info("[Supervisor] discover_slots[%r]: department → %s (counts=%s)", license_type, sorted(_dept_opts_qualified), _dept_doc_count)
                else:
                    # Government offices — not a user choice. Determined by location/entity_type.
                    # The LLM will see docs from all govt offices and explain all channels in the answer.
                    _LOG.info(
                        "[Supervisor] discover_slots[%r]: department SKIPPED — govt offices (not user-choosable), LLM covers all (counts=%s)",
                        license_type, _dept_doc_count,
                    )
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
                _raw_loc_opts = sorted(location_opts)
                _display_opts = [_loc_display.get(loc, loc) for loc in _raw_loc_opts]
                slots.append({
                    "key": "location",
                    "options": _raw_loc_opts,
                    "display_options": _display_opts,
                    "question": "ร้านของคุณตั้งอยู่ในพื้นที่ใดครับ?",
                })
                seen_keys.add("location")
                _LOG.info("[Supervisor] discover_slots[%r]: location → %s (display=%s)", license_type, _raw_loc_opts, _display_opts)
            else:
                _LOG.info(
                    "[Supervisor] discover_slots[%r]: location SKIPPED — only %d location value(s) in docs",
                    license_type, len(location_opts),
                )
                # Content-scan fallback: metadata has no location split, but document text may
                # describe location-conditional differences (e.g. "กรุงเทพฯ: 8-14 วัน / ต่างจังหวัด: 14-21 วัน").
                # Scan all doc contents for the pattern: Bangkok mention + provincial mention with DIFFERENT numeric values.
                # If found → add a synthetic location slot so the user is asked which one applies.
                if "location" not in seen_keys:
                    _LOC_BKK_RE = re.compile(r"กรุงเทพ|กทม|bkk", re.IGNORECASE)
                    _LOC_PROV_RE = re.compile(r"ต่างจังหวัด|ภูมิภาค|จังหวัดอื่น", re.IGNORECASE)
                    _NUM_RE = re.compile(r"(\d+)\s*(?:-\s*(\d+))?\s*(?:วัน|วันทำการ|ชั่วโมง)")
                    _content_has_bkk = False
                    _content_has_prov = False
                    _bkk_nums: set = set()
                    _prov_nums: set = set()
                    # Sequential within-document scan: detects location-conditional timelines
                    # in two formats:
                    #   FWD: "กทม: 8-14 วันทำการ ... ต่างจังหวัด: 14-21 วันทำการ"
                    #   REV: "8-14 วันทำการ (กทม.) ... 14-21 วันทำการ (ต่างจังหวัด)"
                    # Uses DOTALL so it spans newlines between the two mentions.
                    _SEQ_FWD = re.compile(
                        r"(?:กรุงเทพ|กทม|bkk)[^0-9]*(\d+)(?:\s*-\s*(\d+))?"
                        r"[^0-9]{0,30}(?:วัน|วันทำการ)"
                        r".{0,300}?"
                        r"(?:ต่างจังหวัด|ภูมิภาค|จังหวัดอื่น)[^0-9]*(\d+)(?:\s*-\s*(\d+))?"
                        r"[^0-9]{0,30}(?:วัน|วันทำการ)",
                        re.IGNORECASE | re.DOTALL,
                    )
                    _SEQ_REV = re.compile(
                        r"(\d+)(?:\s*-\s*(\d+))?[^0-9\n]{0,30}(?:วัน|วันทำการ)"
                        r"[^\n]{0,60}(?:กรุงเทพ|กทม|bkk)"
                        r".{0,300}?"
                        r"(\d+)(?:\s*-\s*(\d+))?[^0-9\n]{0,30}(?:วัน|วันทำการ)"
                        r"[^\n]{0,60}(?:ต่างจังหวัด|ภูมิภาค|จังหวัดอื่น)",
                        re.IGNORECASE | re.DOTALL,
                    )
                    _location_diff_found = False
                    for _doc_text in docs_content:
                        _text = str(_doc_text or "")
                        # Pass 1: line-by-line (original logic — catches multi-line structure)
                        for _line in _text.splitlines():
                            _nums_in_line = tuple(m.group() for m in _NUM_RE.finditer(_line))
                            if _LOC_BKK_RE.search(_line):
                                _content_has_bkk = True
                                _bkk_nums.update(_nums_in_line)
                            if _LOC_PROV_RE.search(_line):
                                _content_has_prov = True
                                _prov_nums.update(_nums_in_line)
                        # Pass 2: sequential within-document scan (catches same-line/compact format)
                        for _seq_re in (_SEQ_FWD, _SEQ_REV):
                            for _m in _seq_re.finditer(_text):
                                _bkk_lo = int(_m.group(1))
                                _bkk_hi = int(_m.group(2) or _m.group(1))
                                _prov_lo = int(_m.group(3))
                                _prov_hi = int(_m.group(4) or _m.group(3))
                                if (_bkk_lo, _bkk_hi) != (_prov_lo, _prov_hi):
                                    _location_diff_found = True
                                    _LOG.info(
                                        "[Supervisor] discover_slots[%r]: location (within-doc scan) BKK=%d-%d PROV=%d-%d differ",
                                        license_type, _bkk_lo, _bkk_hi, _prov_lo, _prov_hi,
                                    )
                                    break
                            if _location_diff_found:
                                break
                        if _location_diff_found:
                            break
                    # Only add location slot if BOTH locations appear AND their numeric values differ
                    _line_scan_differs = _content_has_bkk and _content_has_prov and (_bkk_nums != _prov_nums)
                    if _line_scan_differs or _location_diff_found:
                        slots.append({
                            "key": "location",
                            "options": ["กรุงเทพฯ และปริมณฑล", "ต่างจังหวัด"],
                            "question": "ร้านของคุณตั้งอยู่ในพื้นที่ใดครับ?",
                        })
                        seen_keys.add("location")
                        _LOG.info(
                            "[Supervisor] discover_slots[%r]: location (content-scan) → found BKK=%s PROV=%s nums differ",
                            license_type, _bkk_nums, _prov_nums,
                        )

            # Rule 0.5: operation_topic dimension
            # When docs for this license have 2–5 distinct non-empty operation_topic values
            # that are NOT location descriptions (already handled by Rule 0) and NOT entity
            # type labels, AND the license has no entity_type split, ask which specific topic
            # the user needs before answering.
            # Typical use: ใบรับรองมาตรฐานร้านอาหาร → "มาตรฐาน SAN" vs "มาตรฐาน SAN PLUS"
            # Skip when entity_type split exists — entity is asked first (Rule 1), and
            # operation_group is added post-entity via the existing post-entity-answer path.
            _op_topic_opts: set = set()
            _op_topic_doc_count: dict = {}
            for _md_ot in mds:
                _ot_val = ((_md_ot or {}).get("operation_topic") or "").strip()
                if _ot_val and _ot_val not in ("nan", "None"):
                    _op_topic_opts.add(_ot_val)
                    _op_topic_doc_count[_ot_val] = _op_topic_doc_count.get(_ot_val, 0) + 1
            # Skip topics that look like location descriptions (already handled by Rule 0)
            _LOC_TOPIC_KW = ("กรุงเทพ", "ต่างจังหวัด", "จังหวัด", "ปริมณฑล")
            # Skip topics that are entity type labels (already handled by Rule 1 / entity path)
            _ENTITY_TOPIC_KW = ("นิติบุคคล", "บุคคลธรรมดา")
            _op_topic_opts = {
                _ot for _ot in _op_topic_opts
                if not any(_kw in _ot for _kw in _LOC_TOPIC_KW)
                and not any(_kw in _ot for _kw in _ENTITY_TOPIC_KW)
            }
            # Check if license has entity_type split — if so, entity is asked first (skip here)
            _has_entity_split_05 = any(
                ((md or {}).get("entity_type_normalized") or "").strip() not in ("", "nan", "None")
                for md in mds
            )
            if 2 <= len(_op_topic_opts) <= 5 and "operation_topic" not in seen_keys and not _has_entity_split_05:
                slots.append({
                    "key": "operation_topic",
                    "options": sorted(_op_topic_opts),
                    "question": "ต้องการดำเนินการเรื่องใดครับ?",
                })
                seen_keys.add("operation_topic")
                _LOG.info(
                    "[Supervisor] discover_slots[%r]: operation_topic → %s (counts=%s)",
                    license_type, sorted(_op_topic_opts), _op_topic_doc_count,
                )
            else:
                _LOG.info(
                    "[Supervisor] discover_slots[%r]: operation_topic SKIPPED — topics=%d entity_split=%s",
                    license_type, len(_op_topic_opts), _has_entity_split_05,
                )

            # Rule 0.6: operation_group dimension (for licenses with no entity split)
            # When operation_topic has > 5 values or no clean split, AND operation_by_department
            # has ≥2 distinct values with ≥2 docs each, group operations into canonical buckets
            # using the existing _get_operation_groups_for_entity logic.
            # Only fires when: no operation_topic slot was added AND no entity_type slot will be added.
            # Goal: ทะเบียนผู้ประกันตน (แจ้งรับ/แก้ไข/ยกเลิก/ส่งสมทบ) without needing entity split.
            if "operation_topic" not in seen_keys and "operation_group" not in seen_keys:
                _obd_count: dict = {}
                for _md_og in mds:
                    _obd = ((_md_og or {}).get("operation_by_department") or "").strip()
                    if _obd and _obd not in ("nan", "None"):
                        _obd_count[_obd] = _obd_count.get(_obd, 0) + 1
                _obd_qualified = {_obd for _obd, _cnt in _obd_count.items() if _cnt >= 2}
                # Only add if ≥2 distinct department operations AND no entity_type split will be added.
                # (If entity_type exists, operation_group is added post entity-answer — skip here.)
                _has_entity_split = any(
                    ((md or {}).get("entity_type_normalized") or "").strip() not in ("", "nan", "None")
                    for md in mds
                )
                if len(_obd_qualified) >= 2 and not _has_entity_split:
                    try:
                        _og_labels, _og_raw_map = self._get_operation_groups_for_entity(license_type, "")
                        if len(_og_labels) >= 2:
                            slots.append({
                                "key": "operation_group",
                                "options": _og_labels,
                                "question": "ต้องการดำเนินการเรื่องใดครับ?",
                                "raw_op_map": _og_raw_map,
                            })
                            seen_keys.add("operation_group")
                            _LOG.info(
                                "[Supervisor] discover_slots[%r]: operation_group (no-entity path) → %s",
                                license_type, _og_labels,
                            )
                    except Exception as _e_og:
                        _LOG.warning(
                            "[Supervisor] discover_slots[%r]: operation_group failed: %s",
                            license_type, _e_og,
                        )

            # Rule 1: entity_type_normalized dimension
            # Only ask entity_type when docs ACTUALLY DIFFER by entity_type in content.
            # If identification_documents / operation_steps are the same across entity types,
            # asking entity_type adds no value (answer would be identical regardless).
            # Same logic as _area_splits_docs check for shop_area_type.
            entity_opts: set = set()
            _entity_doc_count: dict = {}
            for md in mds:
                et = ((md or {}).get("entity_type_normalized") or "").strip()
                if et and et not in ("nan", "None"):
                    entity_opts.add(et)
                    _entity_doc_count[et] = _entity_doc_count.get(et, 0) + 1
            # Require at least 2 entity types AND each must have ≥2 entity-specific docs.
            # If only 1 doc exists for an entity type, asking is not meaningful (too sparse
            # to be sure the answer truly differs for that type).
            # Fallback: if both types are present but one has only 1 doc (e.g. broad retrieval
            # returned 1 บุคคลธรรมดา + 2 นิติบุคคล), still qualify if total ≥ 2 entity docs.
            # This handles "ต้องการจดใบภาษีมูลค่าเพิ่ม" not matching OBD exactly → broad retrieval
            # undersamples one entity type → entity_type would be silently skipped → bad answer.
            _entity_opts_qualified = {et for et in entity_opts if _entity_doc_count.get(et, 0) >= 2}
            if len(_entity_opts_qualified) < 2 and len(entity_opts) >= 2:
                # Fallback: accept ≥1 doc per type when both types appear in retrieved docs
                _entity_opts_fallback = {et for et in entity_opts if _entity_doc_count.get(et, 0) >= 1}
                if len(_entity_opts_fallback) >= 2:
                    _entity_opts_qualified = _entity_opts_fallback
            if len(_entity_opts_qualified) >= 2:
                # Check if different entity types have meaningfully different content.
                # Compare identification_documents + operation_steps fingerprints per entity type.
                _content_sig_by_entity: dict = {}  # entity_type → set of content fingerprints
                for md in mds:
                    et2 = ((md or {}).get("entity_type_normalized") or "").strip()
                    if not et2 or et2 in ("nan", "None") or et2 not in _entity_opts_qualified:
                        continue
                    id_doc = ((md or {}).get("identification_documents") or "").strip()[:100]
                    op_steps = ((md or {}).get("operation_steps") or "").strip()[:100]
                    fees = ((md or {}).get("fees") or "").strip()[:60]
                    _sig = id_doc + "|" + op_steps + "|" + fees
                    _content_sig_by_entity.setdefault(et2, set()).add(_sig)
                _entity_splits_content = False
                _et_list = list(_content_sig_by_entity.keys())
                if len(_et_list) >= 2:
                    # Any two entity types must share NO identical signature → real content split
                    for i in range(len(_et_list)):
                        for j in range(i + 1, len(_et_list)):
                            if _content_sig_by_entity[_et_list[i]] != _content_sig_by_entity[_et_list[j]]:
                                _entity_splits_content = True
                                break
                        if _entity_splits_content:
                            break
                if _entity_splits_content:
                    slots.append({
                        "key": "entity_type",
                        "options": sorted(_entity_opts_qualified),
                        "question": "ธุรกิจของคุณเป็นรูปแบบใดครับ?",
                    })
                    seen_keys.add("entity_type")
                    _LOG.info("[Supervisor] discover_slots[%r]: entity_type → %s (content differs, counts=%s)", license_type, sorted(_entity_opts_qualified), _entity_doc_count)
                else:
                    _LOG.info(
                        "[Supervisor] discover_slots[%r]: entity_type SKIPPED — %d entity types found but content identical across types",
                        license_type, len(_entity_opts_qualified),
                    )
            elif entity_opts:
                _LOG.info(
                    "[Supervisor] discover_slots[%r]: entity_type SKIPPED — only 1 entity type in qualified set (counts=%s)",
                    license_type, _entity_doc_count,
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
                    # No doc split, but area options found → add as context-only slot.
                    # User's answer won't filter docs (same doc covers all sizes),
                    # but LLM will be told which tier to focus on for a personalised answer.
                    slots.append({
                        "key": "shop_area_type",
                        "options": sorted(area_opts),
                        "question": "ร้านของคุณมีขนาดพื้นที่ประมาณเท่าไหร่ครับ?",
                        "context_only": True,
                    })
                    seen_keys.add("shop_area_type")
                    _LOG.info(
                        "[Supervisor] discover_slots[%r]: shop_area_type → context-only (no doc split) %s",
                        license_type, sorted(area_opts),
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
            # When entity_type is already known, only look at docs for that entity
            # (plus entity-neutral '' docs) so we don't mix sub-types from different entity types.
            # e.g. when entity=นิติบุคคล, don't include "บุคคลธรรมดา กิจการเจ้าของคนเดียว" as option.
            mds_for_rt = (
                [md for md in mds
                 if (md or {}).get("entity_type_normalized", "").strip() in ("", known_entity_type)]
                if known_entity_type else mds
            )
            for md in mds_for_rt:
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
                # Skip tokens that are substrings/prefixes of entity keywords (e.g. "บุคคล" ⊂ "บุคคลธรรมดา")
                # Only check rt IN e direction — NOT e in rt, which would incorrectly drop
                # "ห้างหุ้นส่วนจำกัด/สามัญนิติบุคคล (หสน.)" just because it contains "นิติบุคคล".
                if any(rt in e for e in known_entity):
                    continue
                # Skip operation-action sub-types (e.g. ย้าย, ปิด, เลิก, เปลี่ยน) — these are scenarios
                # for changing an existing registration, not initial registration sub-types
                _OP_PREFIXES = ("ย้าย", "ปิด", "เลิก", "โอน", "หยุด", "แก้ไข", "เพิ่ม", "ลด", "เปลี่ยน")
                if any(rt.startswith(pfx) for pfx in _OP_PREFIXES):
                    continue
                # Skip submission-channel values — "สำนักงานเขตที่ตั้งร้านค้า" / "ออนไลน์" are
                # delivery channels, not business-type differentiators. Asking the user to choose
                # a channel before answering a factual question (e.g. ภาษีป้ายคำนวณยังไง) is wrong
                # because the answer (calculation formula, fee rates) is the same regardless of channel.
                _CHANNEL_KEYWORDS = ("สำนักงาน", "ออนไลน์", "ทางไปรษณีย์", "อิเล็กทรอนิกส์")
                if any(kw in rt for kw in _CHANNEL_KEYWORDS):
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

            # Guard: if none of the candidate options contain a business-type signal keyword,
            # they are operational/topic subtypes (e.g. "แยกยื่น/ยื่นรวม", "ประโยชน์ทดแทนที่ได้รับ")
            # rather than legal-entity sub-types — asking the user to choose among them as a
            # "registration type" produces meaningless slot fills and query contamination.
            _BIZTYPE_SIGNALS = ("บริษัท", "ห้างหุ้นส่วน", "ห้าง", "บุคคล", "จำกัด", "สามัญ")
            if rt_opts and not any(any(kw in opt for kw in _BIZTYPE_SIGNALS) for opt in rt_opts):
                _LOG.info(
                    "[Supervisor] _discover_slots: skip registration_type — no business-type signals in %s",
                    sorted(rt_opts),
                )
                rt_opts = set()

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

    # ── Dynamic Clarification ─────────────────────────────────────────────────

    def _maybe_dynamic_clarification(
        self, state: ConversationState, query: str
    ) -> Optional[str]:
        """Check retrieved docs for metadata divergence across collected context.
        If a diverging field is found, saves pending_dynamic_clarification and returns
        a question string. Returns None if no clarification is needed (answer directly)."""
        docs = state.current_docs or []
        if not docs:
            return None
        # ISSUE 2: skip DC when multi-license retrieval is active — choices would mix licenses
        _ctx = state.context or {}
        if _ctx.get("_multi_topic_retrieval") or _ctx.get("multi_license_topics"):
            return None

        collected = dict(state.get_collected_slots() if hasattr(state, "get_collected_slots") else {})

        # When entity_type is already known, filter docs to matching entity_type only.
        # This prevents registration_type divergence from cross-entity-type docs being detected
        # (e.g. after บุคคลธรรมดา is answered, นิติบุคคล docs leak registration_type options).
        _et_known = (
            collected.get("entity_type_normalized")
            or collected.get("entity_type")
            or (_ctx.get("slots") or {}).get("entity_type", "")
        ).strip()
        if _et_known:
            _et_filtered_docs = [
                d for d in docs
                if not str((d.get("metadata") or {}).get("entity_type_normalized") or "").strip()
                or str((d.get("metadata") or {}).get("entity_type_normalized") or "").strip()
                in (_et_known, "nan", "None")
            ]
            if _et_filtered_docs:
                docs = _et_filtered_docs

        # ISSUE 4: treat slots already queued in topic_slot_queue or pending_slot as "known"
        # so DC doesn't ask about operation_by_department when the slot queue will ask the same thing.
        # We merge pending keys into a temporary copy of collected (sentinel "__pending__") —
        # _detect_divergence sees them as known and skips those fields.  state is not modified.
        _queued_slots: List[Dict] = list(_ctx.get("topic_slot_queue") or [])
        _pending_slot = _ctx.get("pending_slot")
        if isinstance(_pending_slot, dict) and _pending_slot.get("key"):
            _queued_slots.append(_pending_slot)
        for _qs in _queued_slots:
            _qkey = str(_qs.get("key") or "").strip()
            if _qkey and _qkey not in collected:
                collected[_qkey] = "__pending__"

        diverge = self._practical._detect_divergence(docs, collected)
        if not diverge:
            return None

        # Pre-select guard: if the user's query already contains one of the diverging
        # values verbatim (or a significant sub-term), the user has already answered —
        # skip DC, save the pre-selected value to collected_slots, and filter docs.
        # Example: user says "ต่ออายุหนังสือรับรองการแจ้ง" → operation_by_department DC
        # would offer ["ต่ออายุหนังสือรับรองการแจ้ง", "การยื่นใบอนุญาต…"] — absurd to ask.
        _ps_field = diverge.get("metadata_key", "")
        _ps_vals = diverge.get("values") or []
        _ps_q = (query or "").strip()
        _ps_match: Optional[str] = None
        for _psv in _ps_vals:
            _psv_s = (_psv or "").strip()
            if not _psv_s:
                continue
            # Exact substring in either direction
            if _psv_s in _ps_q or _ps_q in _psv_s:
                _ps_match = _psv_s
                break
            # Token-level: any token of candidate (len≥5) appears in query
            _psv_toks = [t for t in re.split(r"[\s/\-,]+", _psv_s) if len(t) >= 5]
            if any(t in _ps_q for t in _psv_toks):
                _ps_match = _psv_s
                break
        if _ps_match is not None:
            _ps_ckey = diverge.get("collected_key", _ps_field)
            try:
                state.save_collected_slot(_ps_ckey, _ps_match)
            except Exception:
                pass
            # Narrow current_docs to only the pre-selected operation
            _ps_filtered = [
                d for d in (state.current_docs or [])
                if str((d.get("metadata") or {}).get(_ps_field, "")).strip() == _ps_match
            ]
            if _ps_filtered:
                state.current_docs = _ps_filtered
            _LOG.info(
                "[Supervisor] DC pre-selected from query: field=%r value=%r — skip clarification",
                _ps_field, _ps_match,
            )
            return None  # user already stated the answer — answer directly

        # Extra guard for registration_type: _detect_divergence fires when operation_steps
        # differ across sub-types.  But if the query is purely about contact/location info
        # the relevant answer field is service_channel, not steps.  When service_channel is
        # identical across all diverging registration_type values, asking the user to choose
        # a sub-type will produce the same answer regardless — so suppress DC.
        #
        # This does NOT apply to other query types (procedure, document, fee) where steps or
        # documents genuinely differ and DC remains useful.
        #
        # Data-verified (2026-05, Chroma 850 docs):
        #   SKIP: ใบทะเบียนพาณิชย์ นิติบุคคล/บริษัท/ห้างหุ้นส่วน → all DBD online (Jac=1.0)
        #   KEEP: ใบภาษีมูลค่าเพิ่ม ย้ายภายใน/ต่างท้องที่ → steps differ AND user needs to know
        #         (service_channel also same, but this query is NOT a contact query → DC kept)
        _div_field = diverge.get("metadata_key", "")
        _is_contact_q = bool(re.search(
            r"ต้องการติดต่อ|ติดต่อได้ที่|ติดต่อที่ไหน|ติดต่อยังไง|ติดต่ออย่างไร|"
            r"ที่อยู่|เบอร์โทร|เบอร์ติดต่อ|โทรหา|โทรติดต่อ",
            query, re.IGNORECASE,
        ))
        if _div_field == "registration_type" and _is_contact_q:
            _div_vals = diverge.get("values") or []
            # Build service_channel map: registration_type value → first non-empty sc
            _sc_per_rt: Dict[str, str] = {}
            for _dc_doc in docs:
                _dc_md = _dc_doc.get("metadata") or {} if isinstance(_dc_doc, dict) else (getattr(_dc_doc, "metadata", None) or {})
                _dc_rv = str(_dc_md.get("registration_type") or "").strip()
                _dc_sc = str(_dc_md.get("service_channel") or "").strip()
                if _dc_rv in _div_vals and _dc_sc and _dc_sc not in ("nan", "None") and _dc_rv not in _sc_per_rt:
                    _sc_per_rt[_dc_rv] = _dc_sc
            _sc_vals = list(_sc_per_rt.values())
            if len(_sc_vals) >= 2:
                # Jaccard similarity across all pairs of service_channel values
                _sc_toks = [frozenset(re.findall(r"[฀-๿]+|[A-Za-z0-9]+", sc[:300])) for sc in _sc_vals]
                _all_sc_sim = True
                for _si in range(len(_sc_toks)):
                    for _sj in range(_si + 1, len(_sc_toks)):
                        _s_inter = len(_sc_toks[_si] & _sc_toks[_sj])
                        _s_union = len(_sc_toks[_si] | _sc_toks[_sj])
                        if _s_union and _s_inter / _s_union < 0.5:
                            _all_sc_sim = False
                            break
                    if not _all_sc_sim:
                        break
                if _all_sc_sim:
                    _LOG.info(
                        "[Supervisor] DC suppressed: contact query + registration_type same service_channel (vals=%r)",
                        _div_vals,
                    )
                    return None  # same contact destination regardless of sub-type → answer directly

        state.context = state.context or {}
        state.context["pending_dynamic_clarification"] = {
            "field":          diverge["metadata_key"],
            "filter_key":     diverge["filter_key"],
            "collected_key":  diverge.get("collected_key", diverge["metadata_key"]),
            "filter_mode":    diverge.get("filter_mode", "chroma"),
            "values":         diverge["values"],
            "original_query": query,
        }
        question = self._practical._build_clarification_question(diverge, docs)
        _LOG.info(
            "[Supervisor] dynamic_clarification: field=%r values=%r",
            diverge["metadata_key"], diverge["values"],
        )
        return question

    def _resolve_dynamic_clarification(
        self, state: ConversationState, user_input: str
    ) -> Tuple[ConversationState, str]:
        """Process user's answer to a dynamic clarification question.
        Saves matched value to collected_slots, clears pending DC, re-retrieves
        with metadata filter, then routes to practical."""
        pending_dc = (state.context or {}).get("pending_dynamic_clarification") or {}
        field = pending_dc.get("field", "")
        filter_key = pending_dc.get("filter_key", field)
        collected_key = pending_dc.get("collected_key", field)
        values = pending_dc.get("values") or []
        original_query = pending_dc.get("original_query", "").strip() or user_input

        matched = self._practical._match_clarification_answer(user_input, values)
        filter_mode = pending_dc.get("filter_mode", "chroma")

        state.context.pop("pending_dynamic_clarification", None)

        if matched:
            try:
                state.save_collected_slot(collected_key, matched)
                # Sync entity_type ↔ entity_type_normalized so both keys are always available.
                # Academic's _re_retrieve_with_slots() looks for entity_type_normalized specifically;
                # Practical's slot context reads entity_type. Both must be present.
                if field == "entity_type_normalized":
                    state.save_collected_slot("entity_type_normalized", matched)
                    state.save_collected_slot("entity_type", matched)
                _LOG.info("[Supervisor] DC resolved: %r=%r (mode=%s) saved to collected_slots", field, matched, filter_mode)
            except Exception as _e:
                _LOG.warning("[Supervisor] DC: save_collected_slot failed: %s", _e)

            # Re-retrieve with filter appropriate to filter_mode
            try:
                if filter_mode == "query_enrich":
                    # Append matched dimension value to query; no Chroma metadata filter.
                    enriched_q = f"{original_query} {matched}".strip()
                    state.current_docs = self._practical._retrieve_docs(enriched_q)
                    state.last_retrieval_query = enriched_q
                    original_query = enriched_q  # pass enriched query to practical.handle
                    _LOG.info(
                        "[Supervisor] DC: query_enrich re-retrieved query=%r → %d docs",
                        enriched_q[:80], len(state.current_docs),
                    )
                else:
                    # chroma mode: metadata filter.
                    # For entity_type_normalized, use $or to include entity-neutral docs
                    # (entity_type_normalized="") alongside the matched entity type.
                    # Entity-neutral docs (e.g. PromptPay) have entity_type_normalized=""
                    # and would be incorrectly excluded by an exact match filter, causing
                    # the fallback to return unfiltered mixed docs.
                    if field == "entity_type_normalized":
                        metadata_filter = {"$or": [
                            {filter_key: matched},
                            {filter_key: ""},
                        ]}
                    else:
                        metadata_filter = {filter_key: matched}
                    state.current_docs = self._practical._retrieve_docs(
                        original_query, metadata_filter=metadata_filter
                    )
                    state.last_retrieval_query = original_query
                    _LOG.info(
                        "[Supervisor] DC: chroma re-retrieved filter=%r → %d docs",
                        metadata_filter, len(state.current_docs),
                    )
            except Exception as _re:
                _LOG.warning("[Supervisor] DC: re-retrieve failed: %s — using existing docs", _re)
        else:
            _LOG.info(
                "[Supervisor] DC: answer %r did not match values=%r — answering unfiltered",
                user_input[:40], values,
            )

        state.context.pop("pending_slot", None)
        st2, reply = self._practical.handle(state, original_query, _internal=False)
        reply = self._normalize_male(reply)
        self._add_assistant(st2, reply)
        st2.last_action = "dc_resolved"
        return st2, reply

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
                # Skip operation-action sub-types (เปลี่ยน, แก้ไข, ย้าย, ปิด, เลิก, etc.)
                _OP_PFXS = ("ย้าย", "ปิด", "เลิก", "โอน", "หยุด", "แก้ไข", "เพิ่ม", "ลด", "เปลี่ยน")
                if any(rt.startswith(pfx) for pfx in _OP_PFXS):
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

    def _detect_license_types_from_query(self, query: str, state=None) -> List[str]:
        """
        Predictive multi-topic detection: scan query for ALL license type keywords BEFORE retrieval.
        Returns list of canonical license_type names found in the query.
        When state is provided and regex finds nothing, falls back to LLM classification.
        """
        q = query or ""
        found: List[str] = []
        for pattern, license_name in self._MULTI_TOPIC_LICENSE_KEYWORDS:
            if re.search(pattern, q, re.IGNORECASE) and license_name not in found:
                found.append(license_name)

        # LLM fallback: only when regex found nothing AND state available for caching
        if not found and state is not None and len(q.strip()) >= 4:
            found = self._license_type_llm_fallback(q, state)

        return found

    def _infer_operation_group_from_query(self, query: str, op_groups: List[str], state=None) -> Optional[str]:
        """
        If user query clearly implies a specific operation type, return matching op_group label.
        Returns None if ambiguous — must ask user.
        state is optional; when provided, enables LLM fallback when all regex patterns miss.
        """
        q = query or ""

        # Pre-guard: operation keyword is contextual background (not direct intent) → don't infer
        if self._CONTEXTUAL_PAST_RE.search(q):
            return None

        def _find_group(keywords: list) -> Optional[str]:
            for g in op_groups:
                if any(kw in g for kw in keywords):
                    return g
            return None

        if self._OP_INFER_RENEW_RE.search(q):
            found = _find_group(["ต่ออายุ", "ต่อ"])
            if found:
                return found
        def _excl_new_group() -> Optional[str]:
            """When edit/cancel op but no group label contains the keyword,
            select by exclusion: find the new-registration group, return the other one."""
            _ng = _find_group(["ใหม่", "จดทะเบียน", "ยื่นขอ", "จัดตั้ง", "สมัคร"])
            if _ng:
                _rest = [g for g in op_groups if g != _ng]
                if len(_rest) == 1:
                    return _rest[0]
            return None

        if self._OP_INFER_CANCEL_RE.search(q):
            found = _find_group(["ยกเลิก", "ปิด"]) or _excl_new_group()
            if found:
                return found
        if self._OP_INFER_EDIT_RE.search(q):
            found = _find_group(["แก้ไข", "เปลี่ยนแปลง"]) or _excl_new_group()
            if found:
                return found
        if self._OP_INFER_NEW_RE.search(q):
            found = _find_group(["ใหม่", "จดทะเบียน", "ยื่นขอ", "จัดตั้ง"])
            if found:
                return found
            # Fallback: pick the group with the highest "new registration" signal score
            # among all op_groups — handles label variants like "ยื่นขอใบอนุญาตใหม่ (จัดตั้ง)"
            _new_signals = ["ใหม่", "จัดตั้ง", "จดทะเบียน", "ยื่นขอ", "สมัคร"]
            _best = max(
                op_groups,
                key=lambda g: sum(1 for kw in _new_signals if kw in g),
                default=None,
            )
            if _best and sum(1 for kw in _new_signals if kw in _best) > 0:
                return _best

        # ── Location sub-operation pre-checks ────────────────────────────────
        # These must fire BEFORE the LLM fallback because the LLM misclassifies:
        # - "เพิ่ม เปิดสถานประกอบการ" → 'new' (wrong; should be เพิ่มสถานประกอบการ)
        # - "ย้ายสถานประกอบการ" → unpredictable
        # "เพิ่ม/ย้าย/ปิด" + "สถาน" are branch-level ops, not new registration.
        if re.search(r"เพิ่ม.{0,8}สถาน|สถาน.{0,8}เพิ่ม", q, re.IGNORECASE):
            found = _find_group(["เพิ่มสถาน"])
            if found:
                return found
        if re.search(r"ย้าย.{0,8}สถาน|สถาน.{0,8}ย้าย", q, re.IGNORECASE):
            found = _find_group(["ย้ายสถาน"])
            if found:
                return found
        if re.search(r"ปิด.{0,8}สถาน|สถาน.{0,8}ปิด", q, re.IGNORECASE):
            found = _find_group(["ปิดสถาน"])
            if found:
                return found

        # ── Recent-message scan (all regex missed on current query) ─────────
        # User may have stated op intent in a prior turn (e.g. "ถ้าจะขอต้องทำไรบ้าง")
        # but we're now evaluating a different query (e.g. last_user_legal_query = topic only).
        # Scan the 4 most-recent user messages for an op intent signal.
        # Guard: stop scanning if a bot message with a numbered menu is found in between
        # (means a different intent was already captured in that slot session).
        if state is not None:
            _recent_msgs = (getattr(state, "messages", None) or [])[-8:]
            for _m in reversed(_recent_msgs):
                _role = (_m.get("role") or "")
                _txt = (_m.get("content") or "").strip()
                if _role == "assistant" and re.search(r"^\d\)", _txt, re.MULTILINE):
                    break  # numbered menu → prior slot session boundary, stop
                if _role != "user" or not _txt or _txt == q:
                    continue
                if self._CONTEXTUAL_PAST_RE.search(_txt):
                    continue
                _hist_code: Optional[str] = None
                if self._OP_INFER_RENEW_RE.search(_txt):  _hist_code = "renew"
                elif self._OP_INFER_CANCEL_RE.search(_txt): _hist_code = "cancel"
                elif self._OP_INFER_EDIT_RE.search(_txt):   _hist_code = "edit"
                elif self._OP_INFER_NEW_RE.search(_txt):    _hist_code = "new"
                if _hist_code:
                    _HIST_KW = {
                        "new":    ["ใหม่", "จดทะเบียน", "ยื่นขอ", "จัดตั้ง", "สมัคร"],
                        "edit":   ["แก้ไข", "เปลี่ยนแปลง"],
                        "cancel": ["ยกเลิก", "ปิด"],
                        "renew":  ["ต่ออายุ", "ต่อ"],
                    }
                    found = _find_group(_HIST_KW.get(_hist_code, []))
                    if found:
                        _LOG.info("[Supervisor/infer_op] history-scan: %r → op=%r group=%r", _txt[:50], _hist_code, found)
                        return found

        # ── LLM fallback (all regex + history scan missed) ───────────────────
        # Catches phrasing like "อยากเปลี่ยนชื่อธุรกิจ" (edit), "ใบหมดแล้วต้องทำอะไร" (renew)
        if state is not None and len(q.strip()) >= 2:
            op_code = self._operation_type_llm_fallback(q, state)
            if op_code:
                _OP_CODE_TO_KEYWORDS = {
                    "new":    ["ใหม่", "จดทะเบียน", "ยื่นขอ", "จัดตั้ง", "สมัคร"],
                    "edit":   ["แก้ไข", "เปลี่ยนแปลง"],
                    "cancel": ["ยกเลิก", "ปิด"],
                    "renew":  ["ต่ออายุ", "ต่อ"],
                }
                found = _find_group(_OP_CODE_TO_KEYWORDS.get(op_code, []))
                if not found and op_code in ("edit", "cancel"):
                    found = _excl_new_group()
                if found:
                    return found

        return None

    def _entity_type_llm_fallback(self, query: str, state, last_query: str = "") -> Optional[str]:
        """
        LLM fallback for entity_type detection — called only when regex + fuzzy both miss.
        Caches result in state.context["_et_llm_cache"] keyed by query to avoid duplicate calls.
        Returns "นิติบุคคล", "บุคคลธรรมดา", or None (unspecified / low confidence).
        """
        q = (query or "").strip()
        if not q:
            return None

        cache = (state.context or {}).setdefault("_et_llm_cache", {})
        if q in cache:
            _LOG.debug("[Supervisor/entity_type] cache hit for %r → %r", q[:40], cache[q])
            return cache[q]

        # Do NOT fallback to last_retrieval_query — it belongs to the previous topic and
        # contaminates entity_type detection when the user switches license topics.
        _lq = last_query or ""
        try:
            res = self._llm_entity_type_call(q, _lq) or {}
        except Exception as _e:
            _LOG.warning("[Supervisor/entity_type] fallback failed: %s", _e)
            cache[q] = None
            return None

        et = res.get("entity_type")
        confidence = float(res.get("confidence") or 0.0)

        if et in ("นิติบุคคล", "บุคคลธรรมดา") and confidence >= 0.70:
            _LOG.info(
                "[Supervisor/entity_type] LLM fallback: %r → %r (confidence=%.2f)",
                q[:50], et, confidence,
            )
            cache[q] = et
            return et

        _LOG.debug(
            "[Supervisor/entity_type] LLM fallback: unspecified for %r (et=%r conf=%.2f)",
            q[:50], et, confidence,
        )
        cache[q] = None
        return None

    def _location_llm_fallback(self, query: str, state) -> Optional[str]:
        """
        LLM fallback for location detection — called when regex misses.
        e.g. "ร้านอยู่เชียงใหม่" → ต่างจังหวัด, "แถวอโศก" → กรุงเทพฯ
        Caches result in state.context["_loc_llm_cache"].
        """
        q = (query or "").strip()
        if not q:
            return None
        cache = (state.context or {}).setdefault("_loc_llm_cache", {})
        if q in cache:
            return cache[q]
        # Do NOT use last_retrieval_query — it belongs to previous topic and can
        # contaminate location detection when user switches to a new topic.
        _lq = ""
        try:
            res = self._llm_location_call(q, _lq) or {}
        except Exception as _e:
            _LOG.warning("[Supervisor/location] fallback failed: %s", _e)
            cache[q] = None
            return None
        loc = res.get("location")
        confidence = float(res.get("confidence") or 0.0)
        if loc in ("กรุงเทพฯ", "ต่างจังหวัด") and confidence >= 0.70:
            _LOG.info("[Supervisor/location] LLM fallback: %r → %r (conf=%.2f)", q[:50], loc, confidence)
            cache[q] = loc
            return loc
        cache[q] = None
        return None

    def _operation_type_llm_fallback(self, query: str, state) -> Optional[str]:
        """
        LLM fallback for operation type detection — called when all _OP_INFER_* regex miss.
        Returns one of: "new", "edit", "cancel", "renew", or None.
        Caches result in state.context["_op_type_llm_cache"].
        """
        q = (query or "").strip()
        if not q:
            return None
        cache = (state.context or {}).setdefault("_op_type_llm_cache", {})
        if q in cache:
            return cache[q]
        # Do NOT use last_retrieval_query — it belongs to previous topic and can
        # contaminate operation_type detection when user switches to a new topic.
        _lq = ""
        try:
            res = self._llm_operation_type_call(q, _lq) or {}
        except Exception as _e:
            _LOG.warning("[Supervisor/operation_type] fallback failed: %s", _e)
            cache[q] = None
            return None
        op = res.get("operation")
        confidence = float(res.get("confidence") or 0.0)
        if op in ("new", "edit", "cancel", "renew") and confidence >= 0.70:
            _LOG.info("[Supervisor/operation_type] LLM fallback: %r → %r (conf=%.2f)", q[:50], op, confidence)
            cache[q] = op
            return op
        cache[q] = None
        return None

    def _area_size_llm_fallback(self, query: str, state) -> Optional[str]:
        """
        LLM fallback for shop area size — called when regex + numeric parsing both miss.
        Returns "น้อยกว่า 200 ตารางเมตร", "มากกว่า 200 ตารางเมตร", or None.
        Caches result in state.context["_area_llm_cache"].
        """
        q = (query or "").strip()
        if not q:
            return None
        cache = (state.context or {}).setdefault("_area_llm_cache", {})
        if q in cache:
            return cache[q]
        try:
            res = self._llm_area_size_call(q) or {}
        except Exception as _e:
            _LOG.warning("[Supervisor/area_size] fallback failed: %s", _e)
            cache[q] = None
            return None
        area = res.get("area_size")
        confidence = float(res.get("confidence") or 0.0)
        _VALID_AREAS = ("น้อยกว่า 200 ตารางเมตร", "มากกว่า 200 ตารางเมตร")
        if area in _VALID_AREAS and confidence >= 0.70:
            _LOG.info("[Supervisor/area_size] LLM fallback: %r → %r (conf=%.2f)", q[:50], area, confidence)
            cache[q] = area
            return area
        cache[q] = None
        return None

    def _license_type_llm_fallback(self, query: str, state) -> List[str]:
        """
        LLM fallback for license type detection — called when ALL keyword patterns return [].
        Caches result in state.context["_lt_llm_cache"] keyed by query.
        Returns list of license_type strings (empty list if unspecified or low confidence).
        """
        q = (query or "").strip()
        if not q or len(q) < 3:
            return []

        cache = (state.context or {}).setdefault("_lt_llm_cache", {})
        if q in cache:
            _LOG.debug("[Supervisor/license_type] cache hit for %r → %r", q[:40], cache[q])
            return cache[q]

        candidates = [lt for _, lt in self._MULTI_TOPIC_LICENSE_KEYWORDS]
        try:
            res = self._llm_license_type_call(q, candidates) or {}
        except Exception as _e:
            _LOG.warning("[Supervisor/license_type] fallback failed: %s", _e)
            cache[q] = []
            return []

        lts = res.get("license_types") or []
        confidence = float(res.get("confidence") or 0.0)
        valid_lt_set = set(candidates)

        if isinstance(lts, list) and lts and confidence >= 0.70:
            result = [str(lt).strip() for lt in lts if str(lt).strip() in valid_lt_set]
            if result:
                _LOG.info(
                    "[Supervisor/license_type] LLM fallback: %r → %r (conf=%.2f)",
                    q[:50], result, confidence,
                )
                cache[q] = result
                return result

        _LOG.debug("[Supervisor/license_type] LLM fallback: unspecified for %r (conf=%.2f)", q[:50], confidence)
        cache[q] = []
        return []

    def _classify_yes_no_hybrid(self, user_text: str) -> Dict[str, Any]:
        """
        Hybrid yes/no classifier: deterministic fast-path → LLM fallback when unclear.
        Returns same shape as _classify_yes_no_det: {yes, no, confidence, method}.
        Uses the already-initialized llm_confirm_call (no extra LLM instance needed).
        """
        det = self._classify_yes_no_det(user_text)
        if float(det.get("confidence") or 0.0) >= 0.70:
            return det

        # Skip LLM for empty or very short single-char inputs — too ambiguous to classify
        stripped = (user_text or "").strip()
        if not stripped or len(stripped) <= 1:
            return det

        # LLM fallback (uses the existing confirm LLM call — already set up in __init__)
        try:
            res = self.llm_confirm_call(stripped) or {}
        except Exception as _e:
            _LOG.warning("[Supervisor/yes_no_hybrid] LLM call failed: %s", _e)
            return det

        conf = float(res.get("confidence") or 0.0)
        if conf >= 0.70:
            is_yes = bool(res.get("yes"))
            is_no = bool(res.get("no"))
            if is_yes and not is_no:
                _LOG.info("[Supervisor/yes_no_hybrid] LLM: %r → yes (conf=%.2f)", stripped[:30], conf)
                return {"yes": True, "no": False, "confidence": conf, "method": "llm"}
            if is_no and not is_yes:
                _LOG.info("[Supervisor/yes_no_hybrid] LLM: %r → no (conf=%.2f)", stripped[:30], conf)
                return {"yes": False, "no": True, "confidence": conf, "method": "llm"}

        return det

    def _academic_resume_llm_check(self, user_text: str, last_academic_q: str, state) -> bool:
        """
        Dedicated LLM check for academic resume intent.
        Called when _ACADEMIC_RESUME_RE regex misses — more specific than general fallback_intent.
        Returns True if user wants to continue/resume the current Academic session.
        Caches in state.context["_ac_resume_llm_cache"].
        """
        q = (user_text or "").strip()
        if not q or len(q) < 2:
            return False

        cache = (state.context or {}).setdefault("_ac_resume_llm_cache", {})
        if q in cache:
            return cache[q]

        flow = (state.context or {}).get("academic_flow") or {}
        section_catalog = (state.context or {}).get("section_catalog") or {}
        section_count = len(section_catalog) if isinstance(section_catalog, dict) else 0

        try:
            res = self._llm_academic_resume_call(q, last_academic_q or "", section_count) or {}
        except Exception as _e:
            _LOG.warning("[Supervisor/academic_resume] LLM check failed: %s", _e)
            cache[q] = False
            return False

        conf = float(res.get("confidence") or 0.0)
        is_resume = bool(res.get("is_resume")) and conf >= 0.70

        if is_resume:
            _LOG.info(
                "[Supervisor/academic_resume] LLM detected resume: %r (conf=%.2f)",
                q[:50], conf,
            )
        cache[q] = is_resume
        return is_resume

    def _elaborate_llm_check(self, user_text: str, state) -> bool:
        """
        LLM fallback: does user want MORE detail on the current topic?
        Called when _ELABORATE_RE regex misses AND last_user_legal_query exists.
        Caches in state.context['_elab_llm_cache'].
        """
        q = (user_text or "").strip()
        if not q or len(q) < 3:
            return False
        last_topic = (state.context or {}).get("last_user_legal_query", "").strip()
        if not last_topic:
            return False
        cache = (state.context or {}).setdefault("_elab_llm_cache", {})
        cache_key = q[:60]
        if cache_key in cache:
            return cache[cache_key]
        try:
            res = self._llm_elaborate_call(q, last_topic) or {}
            conf = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_elaborate")) and conf >= 0.75
        except Exception as _e:
            _LOG.warning("[Supervisor/elaborate] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Supervisor/elaborate] LLM detected elaboration: %r (conf=%.2f)", q[:50], conf)
        cache[cache_key] = result
        return result

    def _broad_question_llm_check(self, user_text: str, state) -> bool:
        """
        LLM fallback: is this a broad overview question spanning multiple license/topic areas?
        Called when _BROAD_Q_RE regex misses AND query is substantive (≥8 chars).
        Caches in state.context['_broad_q_llm_cache'].
        """
        q = (user_text or "").strip()
        if not q or len(q) < 8:
            return False
        cache = (state.context or {}).setdefault("_broad_q_llm_cache", {})
        cache_key = q[:80]
        if cache_key in cache:
            return cache[cache_key]
        try:
            res = self._llm_broad_q_call(q) or {}
            conf_val = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_broad")) and conf_val >= 0.75
        except Exception as _e:
            _LOG.warning("[Supervisor/broad_q] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Supervisor/broad_q] LLM detected broad question: %r (conf=%.2f)", q[:50], conf_val)
        cache[cache_key] = result
        return result

    def _slot_skip_llm_check(self, user_text: str, state) -> bool:
        """
        LLM fallback: does user want to skip the current slot question?
        Called when _SLOT_SKIP_RE regex misses inside _route_pending_slot_to_persona.
        Caches in state.context['_slot_skip_llm_cache'].
        """
        q = (user_text or "").strip()
        if not q or len(q) < 2:
            return False
        cache = (state.context or {}).setdefault("_slot_skip_llm_cache", {})
        cache_key = q[:60]
        if cache_key in cache:
            return cache[cache_key]
        try:
            res = self._llm_slot_skip_call(q) or {}
            conf = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_skip")) and conf >= 0.75
        except Exception as _e:
            _LOG.warning("[Supervisor/slot_skip] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Supervisor/slot_skip] LLM detected skip: %r (conf=%.2f)", q[:50], conf)
        cache[cache_key] = result
        return result

    def _followup_contextual_llm_check(self, user_text: str, state) -> bool:
        """
        LLM fallback: is this a contextual follow-up about user's own case?
        Called when _FOLLOWUP_CONTEXTUAL_RE misses AND last_user_legal_query exists.
        Caches in state.context['_followup_ctx_llm_cache'].
        """
        q = (user_text or "").strip()
        if not q or len(q) < 4:
            return False
        last_topic = (state.context or {}).get("last_user_legal_query", "").strip()
        if not last_topic:
            return False
        cache = (state.context or {}).setdefault("_followup_ctx_llm_cache", {})
        cache_key = q[:60]
        if cache_key in cache:
            return cache[cache_key]
        try:
            res = self._llm_followup_contextual_call(q, last_topic) or {}
            conf = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_followup")) and conf >= 0.75
        except Exception as _e:
            _LOG.warning("[Supervisor/followup_ctx] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Supervisor/followup_ctx] LLM detected contextual followup: %r (conf=%.2f)", q[:50], conf)
        cache[cache_key] = result
        return result

    def _thanks_llm_check(self, user_text: str, state=None) -> bool:
        """
        LLM fallback: is this a thanks expression?
        Called from _looks_like_greeting_or_thanks when _THANKS_RE misses
        and text has no legal/question signals. Catches formal Thai thanks
        like "ขอบพระคุณ", "กราบขอบพระคุณ", "ขอขอบคุณ".
        Caches in state.context['_thanks_llm_cache'] when state is available.
        """
        q = (user_text or "").strip()
        if not q or len(q) < 3 or len(q) > 60:
            return False
        cache = (state.context or {}).setdefault("_thanks_llm_cache", {}) if state is not None else {}
        cache_key = q[:60]
        if cache_key in cache:
            return cache[cache_key]
        try:
            res = self._llm_thanks_call(q) or {}
            conf = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_thanks")) and conf >= 0.75
        except Exception as _e:
            _LOG.warning("[Supervisor/thanks] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Supervisor/thanks] LLM detected thanks: %r (conf=%.2f)", q[:50], conf)
        cache[cache_key] = result
        return result

    def _academic_stop_llm_check(self, user_text: str, state) -> bool:
        """
        LLM fallback: does user want to stop academic mode?
        Called when _ACADEMIC_STOP_RE misses. High threshold (0.80) to prevent
        accidental exits from academic mid-flow.
        Caches in state.context['_ac_stop_llm_cache'].
        """
        q = (user_text or "").strip()
        if not q or len(q) < 2:
            return False
        cache = (state.context or {}).setdefault("_ac_stop_llm_cache", {})
        cache_key = q[:60]
        if cache_key in cache:
            return cache[cache_key]
        try:
            res = self._llm_academic_stop_call(q) or {}
            conf = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_stop")) and conf >= 0.80
        except Exception as _e:
            _LOG.warning("[Supervisor/academic_stop] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Supervisor/academic_stop] LLM detected stop: %r (conf=%.2f)", q[:50], conf)
        cache[cache_key] = result
        return result

    def _smalltalk_llm_check(self, user_text: str, state) -> bool:
        """
        LLM fallback: is this off-topic small talk directed at the bot?
        Called from section 2.7 when _SMALLTALK_RE misses.
        Guards: not a legal question (LEGAL_SIGNAL_RE), text 4-80 chars.
        Caches in state.context['_smalltalk_llm_cache'].
        """
        q = (user_text or "").strip()
        if not q or len(q) < 4 or len(q) > 80:
            return False
        # Skip LLM if text already looks like legal content
        if self._LEGAL_SIGNAL_RE.search(q) or self._QUESTION_MARKERS_RE.search(q):
            return False
        cache = (state.context or {}).setdefault("_smalltalk_llm_cache", {})
        cache_key = q[:60]
        if cache_key in cache:
            return cache[cache_key]
        try:
            res = self._llm_smalltalk_call(q) or {}
            conf = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_smalltalk")) and conf >= 0.75
        except Exception as _e:
            _LOG.warning("[Supervisor/smalltalk] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Supervisor/smalltalk] LLM detected smalltalk: %r (conf=%.2f)", q[:50], conf)
        cache[cache_key] = result
        return result

    def _switch_without_target_llm_check(self, user_text: str, state) -> bool:
        """
        LLM fallback: does user want to change response style without naming academic/practical?
        Called from _looks_like_switch_without_target when regex misses.
        High threshold (0.80) — false positive = unwanted persona switch.
        Caches in state.context['_switch_no_target_llm_cache'].
        """
        q = (user_text or "").strip()
        if not q or len(q) < 4:
            return False
        cache = (state.context or {}).setdefault("_switch_no_target_llm_cache", {})
        cache_key = q[:60]
        if cache_key in cache:
            return cache[cache_key]
        try:
            res = self._llm_switch_without_target_call(q) or {}
            conf = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_switch")) and conf >= 0.80
        except Exception as _e:
            _LOG.warning("[Supervisor/switch_no_target] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Supervisor/switch_no_target] LLM detected switch: %r (conf=%.2f)", q[:50], conf)
        cache[cache_key] = result
        return result

    def _link_request_llm_check(self, user_text: str, state) -> bool:
        """
        LLM fallback: is user requesting a link, URL, form, or downloadable document?
        Called when _LINK_REQUEST_RE misses — catches "ส่งแบบฟอร์มหน่อย", "ขอ URL", "มีลิ้งค์ไหม".
        Cached per turn in state.context['_link_req_llm_cache'] — multiple calls in same turn are free.
        """
        q = (user_text or "").strip()
        if not q or len(q) < 3:
            return False
        cache = (state.context or {}).setdefault("_link_req_llm_cache", {})
        cache_key = q[:80]
        if cache_key in cache:
            return cache[cache_key]
        try:
            res = self._llm_link_request_call(q) or {}
            conf = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_link_request")) and conf >= 0.75
        except Exception as _e:
            _LOG.warning("[Supervisor/link_request] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Supervisor/link_request] LLM detected link request: %r (conf=%.2f)", q[:50], conf)
        cache[cache_key] = result
        return result

    def _mode_status_llm_check(self, user_text: str, state) -> bool:
        """
        LLM fallback: is user asking what mode/persona the bot is currently using?
        Called from _looks_like_mode_status_query after fast-fail guard (mode keywords present)
        and regex miss. Catches "บอทใช้โหมดอะไรอยู่", "ตอนนี้เราอยู่โหมดไหน".
        Caches in state.context['_mode_status_llm_cache'].
        """
        q = (user_text or "").strip()
        if not q or len(q) < 4:
            return False
        cache = (state.context or {}).setdefault("_mode_status_llm_cache", {})
        cache_key = q[:60]
        if cache_key in cache:
            return cache[cache_key]
        try:
            res = self._llm_mode_status_call(q) or {}
            conf_val = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_mode_status")) and conf_val >= 0.75
        except Exception as _e:
            _LOG.warning("[Supervisor/mode_status] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Supervisor/mode_status] LLM detected mode status query: %r (conf=%.2f)", q[:50], conf_val)
        cache[cache_key] = result
        return result

    def _greeting_llm_check(self, user_text: str, state=None) -> bool:
        """
        LLM fallback: is this a greeting or polite social opener?
        Called from _looks_like_greeting_or_thanks after thanks check misses.
        Guards already applied upstream: no legal signal, no question marker.
        High threshold (0.80) — false positive = ignoring a real question.
        Caches in state.context['_greeting_llm_cache'] when state is available.
        """
        q = (user_text or "").strip()
        if not q or len(q) < 3 or len(q) > 60:
            return False
        cache = (state.context or {}).setdefault("_greeting_llm_cache", {}) if state is not None else {}
        cache_key = q[:60]
        if cache_key in cache:
            return cache[cache_key]
        try:
            res = self._llm_greeting_call(q) or {}
            conf_val = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_greeting")) and conf_val >= 0.80
        except Exception as _e:
            _LOG.warning("[Supervisor/greeting] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Supervisor/greeting] LLM detected greeting: %r (conf=%.2f)", q[:50], conf_val)
        cache[cache_key] = result
        return result

    def _info_action_q_llm_check(self, user_text: str, state) -> tuple:
        """
        LLM fallback: classify query as informational (answer directly, skip slot queue)
        vs. action-oriented (collect entity_type/registration_type slots first).
        Called only when both _INFO_Q_RE and _ACTION_Q_RE miss.
        is_action overrides is_info when LLM returns both True.
        Returns (is_info: bool, is_action: bool).
        Caches in state.context['_info_action_llm_cache'].
        """
        q = (user_text or "").strip()
        if not q or len(q) < 4:
            return (False, False)
        # Fast-fail: if ACTION_Q_RE already matched, no LLM needed
        if self._ACTION_Q_RE.search(q):
            return (False, True)
        cache = (state.context or {}).setdefault("_info_action_llm_cache", {})
        cache_key = q[:80]
        if cache_key in cache:
            cached = cache[cache_key]
            return (cached.get("is_info", False), cached.get("is_action", False))
        conf_val = 0.0
        try:
            res = self._llm_info_action_q_call(q) or {}
            conf_val = float(res.get("confidence") or 0.0)
            if conf_val >= 0.75:
                is_action = bool(res.get("is_action"))
                is_info = bool(res.get("is_info")) and not is_action
                result = {"is_info": is_info, "is_action": is_action}
            else:
                result = {"is_info": False, "is_action": False}
        except Exception as _e:
            _LOG.warning("[Supervisor/info_action_q] LLM check failed: %s", _e)
            result = {"is_info": False, "is_action": False}
        if result["is_info"] or result["is_action"]:
            _LOG.info(
                "[Supervisor/info_action_q] LLM: is_info=%s is_action=%s q=%r (conf=%.2f)",
                result["is_info"], result["is_action"], q[:50], conf_val,
            )
        cache[cache_key] = result
        return (result["is_info"], result["is_action"])

    def _legal_q_llm_check(self, user_text: str, state) -> bool:
        """
        LLM fallback: is this a legal/regulatory question about Thai restaurant business?
        Called from _looks_like_legal_question when both _LEGAL_SIGNAL_RE and _QUESTION_MARKERS_RE miss.
        Catches unusual phrasing: "ขอข้อมูลค่าใช้จ่าย", "แนะนำให้ถูกกฎหมายหน่อย", "ต้องติดต่อที่ไหน".
        Caches in state.context['_legal_q_llm_cache'] — called up to 10x/turn, cache makes repeats free.
        """
        q = (user_text or "").strip()
        if not q or len(q) < 4:
            return False
        cache = (state.context or {}).setdefault("_legal_q_llm_cache", {})
        cache_key = q[:80]
        if cache_key in cache:
            return cache[cache_key]
        conf_val = 0.0
        try:
            res = self._llm_legal_q_call(q) or {}
            conf_val = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_legal")) and conf_val >= 0.75
        except Exception as _e:
            _LOG.warning("[Supervisor/legal_q] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Supervisor/legal_q] LLM detected legal question: %r (conf=%.2f)", q[:50], conf_val)
        cache[cache_key] = result
        return result

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

    _cached_main_topics: Optional[List[str]] = None
    _cached_sub_topics: Optional[List[str]] = None
    _cached_operation_topics: Optional[List[str]] = None
    _cached_unique_operation_by_depts: Optional[List[str]] = None

    def _get_all_operation_topics_from_store(self) -> List[str]:
        """
        Returns a deduplicated list of all operation_topic values in the Chroma collection.
        Cached after first call. Only includes values with len >= 4.
        Used for chapter retrieval when a query names an operation_topic exactly
        (e.g. "การจดทะเบียนพาณิชย์ที่ได้รับการยกเว้น" matches operation_topic of the exemption row).
        These rows have sub_topic=None so _get_all_sub_topics_from_store() misses them.
        """
        if self._cached_operation_topics is not None:
            return self._cached_operation_topics
        try:
            _vstore = getattr(self.retriever, "vectorstore", None)
            _coll = getattr(_vstore, "_collection", None) if _vstore else None
            if _coll is None:
                self._cached_operation_topics = []
                return []
            _res = _coll.get(include=["metadatas"])
            _seen: dict = {}
            for _md in (_res.get("metadatas") or []):
                _ot = str(_md.get("operation_topic") or "").strip()
                if _ot and _ot.lower() not in ("nan", "none", "") and len(_ot) >= 4:
                    _seen[_ot] = True
            self._cached_operation_topics = list(_seen.keys())
            _LOG.info("[Supervisor] cached %d distinct operation_topics from store", len(self._cached_operation_topics))
        except Exception as _ex:
            _LOG.warning("[Supervisor] _get_all_operation_topics_from_store failed: %s", _ex)
            self._cached_operation_topics = []
        return self._cached_operation_topics

    def _get_all_unique_operation_by_dept_from_store(self) -> List[str]:
        """
        Returns operation_by_department values that are unique to a single license_type.
        Excludes values shared across multiple licenses (e.g. "แก้ไข เปลี่ยนแปลง" which
        appears in ใบทะเบียนพาณิชย์, POS ID, and ทะเบียนผู้ประกันตน) to prevent
        cross-license collision when doing exact-match chapter retrieval.
        Cached after first call.
        """
        if self._cached_unique_operation_by_depts is not None:
            return self._cached_unique_operation_by_depts
        try:
            _vstore = getattr(self.retriever, "vectorstore", None)
            _coll = getattr(_vstore, "_collection", None) if _vstore else None
            if _coll is None:
                self._cached_unique_operation_by_depts = []
                return []
            _res = _coll.get(include=["metadatas"])
            # Build obd → set of license_types map to detect cross-license duplicates
            from collections import defaultdict as _defaultdict
            _obd_to_lics: dict = _defaultdict(set)
            for _md in (_res.get("metadatas") or []):
                _obd = str(_md.get("operation_by_department") or "").strip()
                _lic = str(_md.get("license_type") or "").strip()
                if _obd and _obd.lower() not in ("nan", "none", "") and len(_obd) >= 8:
                    _obd_to_lics[_obd].add(_lic)
            # Only keep OBDs that belong to exactly one license_type (no cross-license ambiguity)
            self._cached_unique_operation_by_depts = [
                obd for obd, lics in _obd_to_lics.items() if len(lics) == 1
            ]
            _LOG.info(
                "[Supervisor] cached %d unique operation_by_dept values (of %d total)",
                len(self._cached_unique_operation_by_depts), len(_obd_to_lics),
            )
        except Exception as _ex:
            _LOG.warning("[Supervisor] _get_all_unique_operation_by_dept_from_store failed: %s", _ex)
            self._cached_unique_operation_by_depts = []
        return self._cached_unique_operation_by_depts

    def _get_all_sub_topics_from_store(self) -> List[str]:
        """
        Returns a deduplicated list of all sub_topic values in the Chroma collection.
        Cached after first call. Only includes values with len >= 4.
        Used for chapter retrieval when a query exactly names a sub_topic
        (e.g. "การตลาดแบบ B2B" matches sub_topic="การตลาดแบบ B2B").
        """
        if self._cached_sub_topics is not None:
            return self._cached_sub_topics
        try:
            _vstore = getattr(self.retriever, "vectorstore", None)
            _coll = getattr(_vstore, "_collection", None) if _vstore else None
            if _coll is None:
                self._cached_sub_topics = []
                return []
            _res = _coll.get(include=["metadatas"])
            _seen: dict = {}
            for _md in (_res.get("metadatas") or []):
                _st = str(_md.get("sub_topic") or "").strip()
                if _st and _st.lower() not in ("nan", "none", "") and len(_st) >= 4:
                    _seen[_st] = True
            self._cached_sub_topics = list(_seen.keys())
            _LOG.info("[Supervisor] cached %d distinct sub_topics from store", len(self._cached_sub_topics))
        except Exception as _ex:
            _LOG.warning("[Supervisor] _get_all_sub_topics_from_store failed: %s", _ex)
            self._cached_sub_topics = []
        return self._cached_sub_topics

    def _get_all_main_topics_from_store(self) -> List[str]:
        """
        Returns a deduplicated list of all main_topic values in the Chroma collection.
        Result is cached on the instance after the first call (topics don't change at runtime).
        Only includes topics with len >= 5 to avoid empty/junk values.
        """
        if self._cached_main_topics is not None:
            return self._cached_main_topics
        try:
            _vstore = getattr(self.retriever, "vectorstore", None)
            _coll = getattr(_vstore, "_collection", None) if _vstore else None
            if _coll is None:
                self._cached_main_topics = []
                return []
            _res = _coll.get(include=["metadatas"])
            _seen: dict = {}
            for _md in (_res.get("metadatas") or []):
                _mt = str(_md.get("main_topic") or "").strip()
                if _mt and _mt.lower() not in ("nan", "none", "") and len(_mt) >= 5:
                    _seen[_mt] = True
            self._cached_main_topics = list(_seen.keys())
            _LOG.info("[Supervisor] cached %d distinct main_topics from store", len(self._cached_main_topics))
        except Exception as _ex:
            _LOG.warning("[Supervisor] _get_all_main_topics_from_store failed: %s", _ex)
            self._cached_main_topics = []
        return self._cached_main_topics

    def _is_broad_question(self, query: str, state=None) -> bool:
        """
        Returns True when the user asks a broad open-ended question about opening a business
        (e.g. "อยากเปิดร้านเบเกอรี่ ต้องทำอะไรบ้าง").
        Broad questions need more docs (across license types) to answer all dimensions.
        LLM fallback fires when regex misses and query is substantive (≥8 chars).
        """
        if self._BROAD_Q_RE.search(query or ""):
            # Specificity override: if a specific license name or form code appears in the query,
            # it is a targeted single-license question — not a broad overview — even if the phrasing
            # contains "เอกสารอะไรบ้าง" or similar. Without this override, group D of _BROAD_Q_RE
            # misclassifies "ขอใบภาษีมูลค่าเพิ่ม ภพ.20 ต้องใช้เอกสารอะไรบ้าง" as broad, causing
            # sibling-completion to pull docs from unrelated licenses and bloating the prompt to 47K tokens.
            if not self._SPECIFIC_LICENSE_INDICATOR_RE.search(query or ""):
                return True
            # Specific license named → let LLM decide (it will correctly say not-broad)
        if state is not None and len((query or "").strip()) >= 8:
            return self._broad_question_llm_check(query, state)
        return False

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

            # When entity_type_normalized is empty/None (no entity split), skip the entity
            # filter and go straight to the license-wide query.  This handles licenses like
            # ทะเบียนผู้ประกันตน where all docs have entity_type_normalized="" — Chroma may
            # not return empty-string matches reliably, and the ops set would be empty, causing
            # an early return before result_all is collected.
            ops: set = set()
            if entity_type_normalized:
                try:
                    result = coll.get(
                        where={"$and": [
                            {"license_type": license_type},
                            {"entity_type_normalized": entity_type_normalized},
                        ]},
                        include=["metadatas"],
                    )
                    for md in (result.get("metadatas") or []):
                        op = ((md or {}).get("operation_by_department") or "").strip()
                        if op:
                            ops.add(op)
                except Exception:
                    pass

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

            if not combined_ops:
                return [], {}

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

    def _group_sub_operations(self, sub_ops: List[str], license_type: str) -> Tuple[List[str], Dict]:
        """
        Groups a long list of raw sub-operation strings into ≤5 user-friendly category labels.
        Used when operation_sub_type would have >5 options (e.g. 26 แก้ไข sub-ops for
        ใบทะเบียนพาณิชย์ นิติบุคคล) — prevents the bot from dumping an overwhelming list.

        Strategy:
        1. LLM-based grouping (fast topic-picker model, JSON mode, same pattern as op_groups)
        2. Regex fallback: extract second significant word after main verb (ชื่อ/กรรมการ/ที่ตั้ง/etc.)

        Returns (category_labels, raw_sub_map) where raw_sub_map maps each label to the
        list of raw sub-op strings it covers.  When the fallback fires, labels may be raw
        keyword fragments; the caller should still present them as menu items.
        """
        from utils.prompts_supervisor import build_sub_op_group_classifier_prompt
        try:
            topic_model = getattr(conf, "OPENROUTER_MODEL_TOPIC_PICKER",
                                  getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL))
            _timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
            _sub_llm = ChatOpenAI(
                model=topic_model,
                openai_api_key=conf.OPENROUTER_API_KEY,
                openai_api_base=conf.OPENROUTER_BASE_URL,
                temperature=0.0,
                max_tokens=800,
                request_timeout=_timeout,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            _prompt = build_sub_op_group_classifier_prompt(license_type, sub_ops)
            _text = extract_llm_text(
                llm_invoke(_sub_llm, [HumanMessage(content=_prompt)],
                           logger=_LOG, label="Supervisor/sub_op_group")
            ).strip()
            _text = self._strip_code_fences(_text)
            _obj = json.loads(_text)
            _groups_raw = _obj.get("groups") or []
            _slot_opts: List[str] = []
            _raw_map: Dict[str, List[str]] = {}
            for _g in _groups_raw:
                _lbl = str(_g.get("label") or "").strip()
                _raws = [str(v).strip() for v in (_g.get("raw") or []) if str(v).strip()]
                if _lbl and _raws:
                    _slot_opts.append(_lbl)
                    _raw_map[_lbl] = _raws
            if len(_slot_opts) >= 2:
                _LOG.info(
                    "[Supervisor] sub_op_group LLM: %d ops → %d groups: %s",
                    len(sub_ops), len(_slot_opts), _slot_opts,
                )
                return _slot_opts, _raw_map
        except Exception as _e_sg:
            _LOG.warning("[Supervisor] sub_op_group LLM failed: %s", _e_sg)

        # Rule-based fallback: group by second significant word after main verb
        _SKIP_WORDS = {"กรณี", "รายการ", "ข้อมูล", "ที่", "และ", "หรือ", "ของ"}
        _groups_fb: Dict[str, List[str]] = {}
        for _op in sub_ops:
            _parts = [p for p in _op.split() if p not in _SKIP_WORDS]
            # "แก้ไข ชื่อ กรณี..." → category word = "ชื่อ" (index 1)
            _cat = _parts[1] if len(_parts) >= 2 else (_parts[0] if _parts else "อื่นๆ")
            _groups_fb.setdefault(_cat, []).append(_op)
        # Sort by count desc, cap to 5 categories
        _sorted_fb = sorted(_groups_fb.items(), key=lambda x: len(x[1]), reverse=True)
        if len(_sorted_fb) > 5:
            _top4 = _sorted_fb[:4]
            _others: List[str] = []
            for _, _ops in _sorted_fb[4:]:
                _others.extend(_ops)
            _sorted_fb = _top4 + [("อื่นๆ", _others)]
        _fb_opts = [_c for _c, _ in _sorted_fb]
        _fb_map = {_c: _ops for _c, _ops in _sorted_fb}
        _LOG.info(
            "[Supervisor] sub_op_group fallback: %d ops → %d groups: %s",
            len(sub_ops), len(_fb_opts), _fb_opts,
        )
        return _fb_opts, _fb_map

    def _route_pending_slot_to_persona(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        _raw_pending = (state.context or {}).get("pending_slot")
        pending = _raw_pending if isinstance(_raw_pending, dict) else {}

        # Slot-skip: user said "ไม่รู้" / "ข้ามก่อน" — skip this slot and proceed.
        # LLM fallback catches indirect phrasing like "ยังตัดสินใจไม่ได้", "ขอผ่านไปก่อน".
        _raw_ui = (user_input or "").strip()
        _is_slot_skip = bool(self._SLOT_SKIP_RE.search(_raw_ui))
        if not _is_slot_skip and len(_raw_ui) >= 2:
            _is_slot_skip = self._slot_skip_llm_check(_raw_ui, state)
        if _is_slot_skip:
            _LOG.info("[Supervisor] slot_skip detected for slot=%r input=%r", pending.get("key"), _raw_ui[:50])
            state.context = state.context or {}
            state.context.pop("pending_slot", None)
            _next_queue = list(state.context.get("topic_slot_queue") or [])
            if _next_queue:
                # Advance to next slot in queue
                _next_slot = _next_queue.pop(0)
                state.context["topic_slot_queue"] = _next_queue
                state.context["pending_slot"] = _next_slot
                _next_q = _next_slot.get("question", "กรุณาเลือกตัวเลือกครับ")
                _next_raw_opts = [str(o).strip() for o in (_next_slot.get("options") or []) if str(o).strip()]
                _next_disp_opts = [str(o).strip() for o in (_next_slot.get("display_options") or []) if str(o).strip()]
                _next_opts = _next_disp_opts if len(_next_disp_opts) == len(_next_raw_opts) else _next_raw_opts
                _opts_str = "\n".join(f"{i+1}) {o}" for i, o in enumerate(_next_opts))
                msg = self._normalize_male(f"ได้ครับ ข้ามข้อนั้นไปก่อน\n{_next_q}\n{_opts_str}")
                self._add_assistant(state, msg)
                state.last_action = "pending_slot_skipped_next"
                return state, msg
            else:
                # No more slots — answer with what we have so far
                state.context.pop("topic_slot_queue", None)
                _last_q = (state.context.get("last_user_legal_query") or state.context.get("last_topic") or "").strip()
                if _last_q and not _last_q.startswith("__"):
                    self._ensure_practical_retrieval_for_legal(state, _last_q)
                    st2, reply = self._practical.handle(state, _last_q, _internal=False)
                    reply = self._normalize_male(reply)
                    self._add_assistant(st2, reply)
                    st2.last_action = "pending_slot_skipped_answer"
                    return st2, reply
                msg = self._normalize_male("ได้ครับ ถ้าอยากถามเรื่องอื่นบอกได้เลยครับ 😊")
                self._add_assistant(state, msg)
                state.last_action = "pending_slot_skipped_no_context"
                return state, msg

        # New-topic escape: user changed subject while slot is pending (e.g. asked about a
        # different license while entity_type question was waiting). entity_type/registration_type
        # are in _SLOT_REPLY_ALWAYS so the guard in _should_route_pending_slot_now is bypassed —
        # short answers like "นิติบุคคล" must reach here. But a full legal question that contains
        # none of the current options should break out and be handled as a fresh query.
        _slot_opts = [str(o).strip() for o in (pending.get("options") or []) if str(o).strip()]
        _slot_disp_opts = [str(o).strip() for o in (pending.get("display_options") or []) if str(o).strip()]
        _all_slot_opts = _slot_opts + [o for o in _slot_disp_opts if o not in _slot_opts]
        _none_of_opts_in_input = not any(opt and opt in _raw_ui for opt in _all_slot_opts)
        if (
            _none_of_opts_in_input
            and len(_raw_ui) >= 6
            and (self._LEGAL_SIGNAL_RE.search(_raw_ui) or self._QUESTION_MARKERS_RE.search(_raw_ui))
            and self._looks_like_legal_question(_raw_ui, state)
        ):
            _LOG.info(
                "[Supervisor] pending_slot(%r) new-topic escape — input is a new legal question: %r",
                pending.get("key"), _raw_ui[:60],
            )
            state.context = state.context or {}
            state.context.pop("pending_slot", None)
            state.context.pop("topic_slot_queue", None)
            return self._handle_inner(state, user_input)

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
                # Normalize entity_type free-text to canonical value before persisting.
                # Prevents downstream filter mismatches when the user enters a sub-type variant
                # such as "บริษัท ลิมิเต็ด" instead of the expected canonical "นิติบุคคล".
                if _slot_key_sv == "entity_type":
                    try:
                        from service.data_loader import DataLoader as _DL_norm
                        _et_norm = _DL_norm._normalize_entity_type(_slot_val_sv)
                        if _et_norm:
                            _LOG.info(
                                "[Supervisor] entity_type input %r normalized to %r before save",
                                _slot_val_sv, _et_norm,
                            )
                            _slot_val_sv = _et_norm
                    except Exception:
                        pass
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
        # Slot fill → specific answer: clear broad-question flag so links are not suppressed.
        # _broad_question=True is set when a broad overview was shown earlier in the conversation.
        # Keeping it through slot-fill makes _suppress_links_broad=True in Practical, hiding
        # service/guide/form links even though the user now has a specific topic.
        state.context.pop("_broad_question", None)

        # Pre-retrieve docs so the LLM has documents when called with _internal=True
        # (_internal=True skips the retrieval block inside persona.handle)
        if isinstance(pending, dict) and pending.get("key") == "topic" and mapped:
            # Predictive multi-topic: if user's ORIGINAL message mentions ≥2 license types,
            # retrieve ALL their docs and answer everything in one response — don't single-topic.
            # Also check mapped value (e.g. "ภาษีป้าย, ใบรับรองมาตรฐาน") when user typed
            # number-only multi-select like "ทั้ง 1 และ 2" (user_input has no license names).
            _multi_from_input = self._detect_license_types_from_query(user_input or "")
            if len(_multi_from_input) < 2 and "," in str(mapped):
                _multi_from_input = self._detect_license_types_from_query(str(mapped))
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
                            "operation_steps": 600, "identification_documents": 4000,
                            "research_reference": 3200, "fees": 120, "service_channel": 200,
                        }
                        _per_lt_cap2 = int(getattr(conf, "LLM_DOCS_MAX_BROAD", 20))
                        _total_cap2 = _per_lt_cap2 * 2
                        _STRUCTURAL_KEYS2 = ("operation_steps", "identification_documents", "fees", "terms_and_conditions", "legal_regulatory")
                        _merged_mt2: List[Dict] = []
                        for _lt_name2 in _multi_from_input:
                            if len(_merged_mt2) >= _total_cap2:
                                break
                            _r2 = _coll_mt2.get(
                                where={"license_type": _lt_name2},
                                include=["documents", "metadatas"],
                            )
                            _lt_docs2: List[Dict] = []
                            for _fc2, _fm2 in zip(_r2.get("documents") or [], _r2.get("metadatas") or []):
                                _sm2: Dict = {}
                                for _k2, _v2 in (_fm2 or {}).items():
                                    if _k2 not in _SUPERVISOR_META_WL_MT2 or _v2 in (None, "", "nan", "None"):
                                        continue
                                    _vs2 = str(_v2)
                                    _cap2 = _SUPERVISOR_FC_MT2.get(_k2)
                                    _sm2[_k2] = _vs2[:_cap2] if _cap2 and len(_vs2) > _cap2 else _vs2
                                _lt_docs2.append({"content": (_fc2 or "")[:_doc_chars_mt2], "metadata": _sm2})
                            # Sort structural docs first so cap keeps the most data-rich rows.
                            _lt_docs2.sort(
                                key=lambda d: sum(1 for k in _STRUCTURAL_KEYS2 if (d.get("metadata") or {}).get(k, "").strip() not in ("", "nan", "None")),
                                reverse=True,
                            )
                            _merged_mt2.extend(_lt_docs2[:_per_lt_cap2])
                        _merged_mt2 = _merged_mt2[:_total_cap2]
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
                # Step 0: chapter retrieval — if query names a known main_topic (marketing/business_guide
                # chapters have no license_type so Step 2 never fires), retrieve ALL chapter docs via
                # metadata filter. This ensures sub_topics are never cut by semantic top-K.
                _fb_q_lower = str(mapped).lower()
                _fb_mt_candidates = self._get_all_main_topics_from_store()
                _fb_matched_mts = [
                    mt for mt in _fb_mt_candidates
                    if mt.lower() in _fb_q_lower and len(mt) >= 5
                ]
                if _fb_matched_mts:
                    _fb_target_mt = max(_fb_matched_mts, key=len)
                    try:
                        _fb_coll = self._get_chroma_collection()
                        if _fb_coll is not None:
                            _fb_res = _fb_coll.get(
                                where={"main_topic": _fb_target_mt},
                                include=["documents", "metadatas"],
                            )
                            _fb_max_chars = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                            _fb_chapter_docs = [
                                {"content": (c or "")[:_fb_max_chars], "metadata": m or {}}
                                for c, m in zip(
                                    _fb_res.get("documents") or [],
                                    _fb_res.get("metadatas") or [],
                                )
                            ]
                            if _fb_chapter_docs:
                                state.current_docs = _fb_chapter_docs
                                state.last_retrieval_query = str(mapped)
                                state.context["last_topic"] = _fb_target_mt
                                _LOG.info(
                                    "[Supervisor] fallback chapter retrieval: main_topic=%r → %d docs",
                                    _fb_target_mt, len(_fb_chapter_docs),
                                )
                                return self._practical.handle(state, user_input, _internal=True)
                    except Exception as _fb_mt_ex:
                        _LOG.warning("[Supervisor] fallback chapter retrieval failed (%s), falling through", _fb_mt_ex)

                # Step 0b: operation_topic chapter retrieval — if user named an exact
                # operation_topic (e.g. "การตั้งบริษัทมหาชนจำกัด"), use Chroma metadata filter
                # to retrieve that specific doc instead of semantic search, which may rank
                # related sub-topic docs above the overview doc in the reranker.
                if not _fb_matched_mts:
                    try:
                        _fb_ot_candidates = self._get_all_operation_topics_from_store()
                        _fb_mapped_lower = _fb_q_lower
                        _fb_raw_lower = _raw_ui.lower()
                        _fb_matched_ots = [
                            ot for ot in _fb_ot_candidates
                            if len(ot) >= 8 and (
                                ot.lower() == _fb_mapped_lower
                                or ot.lower() == _fb_raw_lower
                                or ot.lower() in _fb_raw_lower
                            )
                        ]
                        if _fb_matched_ots:
                            _fb_target_ot = max(_fb_matched_ots, key=len)
                            _fb_ot_coll = self._get_chroma_collection()
                            if _fb_ot_coll is not None:
                                _fb_ot_res = _fb_ot_coll.get(
                                    where={"operation_topic": _fb_target_ot},
                                    include=["documents", "metadatas"],
                                )
                                _fb_ot_max_chars = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                                _fb_ot_docs = [
                                    {"content": (c or "")[:_fb_ot_max_chars], "metadata": m or {}}
                                    for c, m in zip(
                                        _fb_ot_res.get("documents") or [],
                                        _fb_ot_res.get("metadatas") or [],
                                    )
                                ]
                                if _fb_ot_docs:
                                    state.current_docs = _fb_ot_docs
                                    state.last_retrieval_query = str(mapped)
                                    state.context["last_topic"] = _fb_target_ot
                                    state.context["last_user_legal_query"] = str(mapped)
                                    state.context["_direct_topic_match"] = True
                                    _LOG.info(
                                        "[Supervisor] topic-slot op_topic retrieval: %r → %d docs",
                                        _fb_target_ot, len(_fb_ot_docs),
                                    )
                                    return self._practical.handle(state, user_input, _internal=True)
                    except Exception as _fb_ot_ex:
                        _LOG.warning(
                            "[Supervisor] topic-slot op_topic retrieval failed (%s), falling through",
                            _fb_ot_ex,
                        )

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
                                # Save unfiltered full-license docs so academic link collection
                                # can access all docs for this license (e.g. both SAN + SAN PLUS)
                                # even after practical narrows current_docs to 1 operation topic.
                                state.context["_presaved_license_docs"] = list(_all_docs)
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
                    "แบบแสดงรายการภาษีป้ายร้านอาหาร": "ภาษีป้าย แบบ ภป.1 ป้ายร้านอาหาร",
                    "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร": "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร BMA OSS",
                    "ใบทะเบียนพาณิชย์": "จดทะเบียนพาณิชย์ DBD",
                    "ใบรับรองมาตรฐานร้านอาหาร": "ใบรับรองมาตรฐานร้านอาหาร SAN กรมอนามัย สุขาภิบาลอาหาร",
                }
                _pt_docs: List[Dict[str, Any]] = []
                _pt_seen: set = set()
                _pt_doc_chars = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                _PT_META_WL = frozenset({
                    "license_type", "operation_topic",
                    "entity_type_normalized", "registration_type", "department",
                    "fees", "operation_duration", "service_channel",
                    "operation_steps", "identification_documents", "research_reference",
                    "legal_regulatory",     # บทลงโทษ ค่าปรับ ข้อกำหนดทางกฎหมาย
                    "terms_and_conditions", # หน้าที่และเงื่อนไขของผู้ประกอบการ
                })
                _PT_FIELD_CAPS = {
                    "operation_steps": 600, "identification_documents": 4000,
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
                    # Step 2: auto-infer entity_type if explicitly stated in query text.
                    # Always overwrite stale entity_type from a previous topic.
                    # Include user_input (the full original topic-selection message) so that
                    # "ของนิติบุคคล" / "แบบบุคคลธรรมดา" in the user's text is detected here,
                    # not just the license name returned by _map_pending_slot_reply.
                    _q_str_main = (str(mapped).strip() + " " + (user_input or "")).strip()
                    _inferred_et_main = self._infer_entity_type_from_query(_q_str_main)
                    if _inferred_et_main:
                        _cs_main = state.get_collected_slots() or {} if hasattr(state, "get_collected_slots") else {}
                        _existing_et_main = (_cs_main.get("entity_type") or "").strip()
                        _existing_norm_main = ""
                        if _existing_et_main:
                            try:
                                from service.data_loader import DataLoader as _DL_et_main
                                _existing_norm_main = _DL_et_main._normalize_entity_type(_existing_et_main)
                            except Exception:
                                _existing_norm_main = _existing_et_main
                        if _existing_norm_main != _inferred_et_main:
                            _cur_lt_main = (state.context or {}).get("last_topic", "").strip()
                            _topic_changed_main = bool(_license_type_for_slots) and (_license_type_for_slots != _cur_lt_main)
                            _SELF_REF_RE2 = re.compile(r"ร้านของ(เรา|ผม|ฉัน|คุณ)|ธุรกิจของ|กิจการของ|เปิดในนาม|จดใน\s*นาม|รูปแบบ(บุคคล|นิติ)|แบบบุคคล|แบบนิติ")
                            _has_self_ref_main = bool(_SELF_REF_RE2.search(_q_str_main))
                            if _topic_changed_main or _has_self_ref_main or not _existing_norm_main:
                                state.save_collected_slot("entity_type", _inferred_et_main)
                                state.context.setdefault("slots", {})["entity_type"] = _inferred_et_main
                                if "collected_slots" in state.context:
                                    state.context["collected_slots"]["entity_type"] = _inferred_et_main
                                _LOG.info(
                                    "[Supervisor] (new-topic) entity_type overwrite: %r → %r (topic_changed=%s self_ref=%s)",
                                    _existing_norm_main, _inferred_et_main, _topic_changed_main, _has_self_ref_main,
                                )
                            else:
                                _LOG.info(
                                    "[Supervisor] (new-topic) entity_type infer mismatch but same topic — keeping %r",
                                    _existing_norm_main,
                                )
                        else:
                            _LOG.info("[Supervisor] (new-topic) entity_type auto-inferred: %r (unchanged)", _inferred_et_main)
                    # Step 3: skip entity_type/registration_type for universal-fact queries
                    elif not self._entity_slot_needed(_q_str_main):
                        _slot_queue = [s for s in _slot_queue if s["key"] not in ("entity_type", "registration_type")]
                        _LOG.info("[Supervisor] entity_type/registration_type skipped — universal-fact query: %r", _q_str_main[:60])

                    # Step 1b (topic-slot path): auto-infer department if the original query
                    # or topic label contains a recognisable department name from the retrieved docs.
                    # Must run BEFORE the cross-topic slot memory filter so "department" is in
                    # collected_slots when the filter skips it from the queue.
                    if any(s.get("key") == "department" for s in _slot_queue):
                        _available_depts_ts: list = []
                        for _d_ts in (state.current_docs or []):
                            if not isinstance(_d_ts, dict):
                                continue
                            _dv_ts = ((_d_ts.get("metadata") or {}).get("department") or "").strip()
                            if _dv_ts and _dv_ts not in _available_depts_ts:
                                _available_depts_ts.append(_dv_ts)
                        # Scan recent user messages (last 6) for department names.
                        # last_user_legal_query is already overwritten with the current short query
                        # by the time this code runs, so we read from conversation history instead.
                        _recent_user_text = " ".join(
                            m.get("content", "")
                            for m in (getattr(state, "messages", []) or [])[-6:]
                            if m.get("role") == "user"
                        )
                        _q_for_dept_ts = (str(mapped).strip() + " " + (user_input or "") + " " + _recent_user_text).strip()
                        _inferred_dept_ts = next(
                            (d for d in _available_depts_ts if d and d in _q_for_dept_ts),
                            None,
                        )
                        if not _inferred_dept_ts:
                            # Try abbreviated bank names via regex (e.g. "กสิกรไทย" → "ธนาคารกสิกรไทย").
                            for _bp_ts, _bn_ts in self._BANK_DEPT_PATTERNS:
                                if _bp_ts.search(_q_for_dept_ts) and _bn_ts in _available_depts_ts:
                                    _inferred_dept_ts = _bn_ts
                                    _LOG.info(
                                        "[Supervisor] topic-slot department regex-inferred: %r",
                                        _inferred_dept_ts,
                                    )
                                    break
                        if _inferred_dept_ts:
                            state.save_collected_slot("department", _inferred_dept_ts)
                            state.context.setdefault("slots", {})["department"] = _inferred_dept_ts
                            if "collected_slots" in state.context:
                                state.context["collected_slots"]["department"] = _inferred_dept_ts
                            _LOG.info(
                                "[Supervisor] topic-slot department auto-inferred from query: %r",
                                _inferred_dept_ts,
                            )
                else:
                    _slot_queue = []

            # Auto-infer location from query text (Bug 1, topic-slot path)
            # Must search user_input (full query) not just mapped (which is the license name only)
            # user_input = full original query (contains "ในกรุงเทพฯ", "พื้นที่น้อยกว่า 200" etc.)
            # _q_str_main = mapped (= license name only, does NOT contain location/area info)
            _q_for_infer_ts = (user_input or "")
            _location_slot_ts = next((s for s in _slot_queue if s.get("key") == "location"), None)
            if _location_slot_ts:
                _loc_opts_ts = _location_slot_ts.get("options") or []
                _inferred_loc_ts = None
                if self._LOCATION_BKK_RE.search(_q_for_infer_ts):
                    _inferred_loc_ts = next((o for o in _loc_opts_ts if "กรุงเทพ" in o), "กรุงเทพฯ")
                elif self._LOCATION_METRO_RE.search(_q_for_infer_ts):
                    # ปริมณฑล province: only map to กรุงเทพ if the option explicitly includes "ปริมณฑล"
                    _metro_opt_ts = next((o for o in _loc_opts_ts if "ปริมณฑล" in o), None)
                    if _metro_opt_ts:
                        _inferred_loc_ts = _metro_opt_ts
                    # else: leave None → fall through to LLM slot mapper
                elif self._LOCATION_PROVINCE_RE.search(_q_for_infer_ts):
                    _inferred_loc_ts = next((o for o in _loc_opts_ts if "จังหวัด" in o or "ต่าง" in o), "ต่างจังหวัด")
                if _inferred_loc_ts:
                    state.save_collected_slot("location", _inferred_loc_ts)
                    state.context.setdefault("slots", {})["location"] = _inferred_loc_ts
                    if "collected_slots" in state.context:
                        state.context["collected_slots"]["location"] = _inferred_loc_ts
                    _LOG.info("[Supervisor] topic-slot location auto-inferred from query: %r", _inferred_loc_ts)

            # Auto-infer shop_area_type from query text (Bug 2, topic-slot path)
            _area_slot_ts = next((s for s in _slot_queue if s.get("key") == "shop_area_type"), None)
            if _area_slot_ts:
                _area_opts_ts = _area_slot_ts.get("options") or []
                _inferred_area_ts = None
                if self._AREA_SMALL_RE.search(_q_for_infer_ts):
                    _inferred_area_ts = next((o for o in _area_opts_ts if "น้อยกว่า" in o or "ไม่เกิน" in o), None)
                elif self._AREA_LARGE_RE.search(_q_for_infer_ts):
                    _inferred_area_ts = next((o for o in _area_opts_ts if "มากกว่า" in o or "เกิน" in o), None)
                if _inferred_area_ts:
                    state.save_collected_slot("shop_area_type", _inferred_area_ts)
                    state.context.setdefault("slots", {})["shop_area_type"] = _inferred_area_ts
                    if "collected_slots" in state.context:
                        state.context["collected_slots"]["shop_area_type"] = _inferred_area_ts
                    _LOG.info("[Supervisor] topic-slot shop_area_type auto-inferred from query: %r", _inferred_area_ts)

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
                                    # Auto-infer registration_type from query text — prevents re-asking when
                                    # user already stated it (e.g. "การจดทะเบียนพาณิชย์ บริษัทจำกัด").
                                    _query_for_rt_nt = _base_q or user_input or ""
                                    _recent_user_rt_nt = " ".join(
                                        m.get("content", "")
                                        for m in (getattr(state, "messages", []) or [])[-6:]
                                        if m.get("role") == "user"
                                    )
                                    # Special case: query explicitly mentions "มหาชน" (บริษัทมหาชนจำกัด).
                                    # This is a specific entity type not listed in standard registration_type
                                    # options — retrieval already resolved it via topic match, so skip asking.
                                    _full_rt_query = _query_for_rt_nt + " " + _recent_user_rt_nt
                                    if "มหาชน" in _full_rt_query:
                                        _inferred_rt_nt = "บริษัทมหาชนจำกัด"
                                        _LOG.info(
                                            "[Supervisor] (new-topic) registration_type auto-inferred 'บริษัทมหาชนจำกัด' from 'มหาชน' keyword in query"
                                        )
                                    else:
                                        _inferred_rt_nt = next(
                                            (o for o in _ert_opts if o and o in _query_for_rt_nt),
                                            None,
                                        ) or next(
                                            (o for o in _ert_opts if o and o in _recent_user_rt_nt),
                                            None,
                                        )
                                    if _inferred_rt_nt:
                                        state.save_collected_slot("registration_type", _inferred_rt_nt)
                                        state.context.setdefault("slots", {})["registration_type"] = _inferred_rt_nt
                                        if "collected_slots" in state.context:
                                            state.context["collected_slots"]["registration_type"] = _inferred_rt_nt
                                        _LOG.info(
                                            "[Supervisor] (new-topic) registration_type auto-inferred from query: %r",
                                            _inferred_rt_nt,
                                        )
                                    else:
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
                            elif _slot.get("key") == "department":
                                # If entity-filtered docs have only one department, auto-infer it.
                                # e.g. นิติบุคคล → all docs from กรมพัฒนาธุรกิจการค้า → no need to ask.
                                _entity_depts: set = set()
                                for _doc_dept in (state.current_docs or []):
                                    if not isinstance(_doc_dept, dict):
                                        continue
                                    _dv = ((_doc_dept.get("metadata") or {}).get("department") or "").strip()
                                    if _dv and _dv not in ("nan", "None"):
                                        _entity_depts.add(_dv)
                                if len(_entity_depts) == 1:
                                    _auto_dept = next(iter(_entity_depts))
                                    state.save_collected_slot("department", _auto_dept)
                                    state.context.setdefault("slots", {})["department"] = _auto_dept
                                    if "collected_slots" in state.context:
                                        state.context["collected_slots"]["department"] = _auto_dept
                                    _LOG.info(
                                        "[Supervisor] (new-topic) department auto-inferred from entity=%r → %r",
                                        _known_entity_for_new_topic, _auto_dept,
                                    )
                                else:
                                    _filtered_nq.append(_slot)
                            else:
                                _filtered_nq.append(_slot)
                        _new_queue = _filtered_nq

                        # Append operation_group slot (skip if already collected for THIS topic)
                        _op_res_nt = self._get_operation_groups_for_entity(
                            _license_type_for_slots, _known_entity_for_new_topic
                        )
                        _op_grps_nt, _raw_op_map_nt = (
                            _op_res_nt if isinstance(_op_res_nt, tuple) else (_op_res_nt, {})
                        )
                        # Clear stale operation_group from prior topic if not valid for new topic ops.
                        if "operation_group" in _prev_slots and _op_grps_nt:
                            _old_og_nt = str(_prev_slots.get("operation_group") or "").strip()
                            _og_valid_nt = bool(_old_og_nt) and any(
                                _old_og_nt in grp or grp in _old_og_nt for grp in _op_grps_nt
                            )
                            if not _og_valid_nt:
                                _LOG.info(
                                    "[Supervisor] (new-topic) stale operation_group=%r not in new ops=%s — clearing",
                                    _old_og_nt, _op_grps_nt[:3],
                                )
                                if isinstance(state.context, dict):
                                    (state.context.get("slots") or {}).pop("operation_group", None)
                                    (state.context.get("slots") or {}).pop("confirmed_operation", None)
                                    if isinstance(state.context.get("collected_slots"), dict):
                                        state.context["collected_slots"].pop("operation_group", None)
                                _prev_slots = {k: v for k, v in _prev_slots.items() if k != "operation_group"}
                        if len(_op_grps_nt) >= 2 and "operation_group" not in _prev_slots:
                            _query_for_infer_nt = (state.context or {}).get("last_user_legal_query") or user_input
                            _inferred_nt = self._infer_operation_group_from_query(_query_for_infer_nt, _op_grps_nt, state)
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

            # Step 3b: Operation-specific bypass — if query terms (≥5 chars) match an
            # operation_topic or operation_by_department in retrieved docs, the retrieval
            # already resolved the exact operation; skip slot queue so the LLM can answer
            # directly without asking department/entity_type
            # (e.g. "แปลงมูลค่าหุ้น" → unique operation found in docs).
            if _slot_queue:
                _q_specific_terms = [
                    t for t in re.findall(r'[\u0E00-\u0E7F]{5,}', (user_input or str(mapped) or ""))
                    if t not in {"อยากรู้", "อยากทราบ", "ต้องการ", "ขอทราบ", "บอกผม"}
                ]
                if _q_specific_terms:
                    _op_fields_docs = []
                    for _d_bp in (state.current_docs or []):
                        if not isinstance(_d_bp, dict):
                            continue
                        _m_bp = _d_bp.get("metadata") or {}
                        for _fld in ("operation_topic", "operation_by_department", "registration_type"):
                            _v_bp = (_m_bp.get(_fld) or "").strip()
                            if _v_bp:
                                _op_fields_docs.append(_v_bp)
                                # Split "/" sub-terms: "แยกยื่น/ยื่นรวม" → "แยกยื่น", "ยื่นรวม"
                                for _sub in re.split(r"/", _v_bp):
                                    _sub = _sub.strip()
                                    if len(_sub) >= 3:
                                        _op_fields_docs.append(_sub)
                    _full_ui = user_input or ""
                    _op_topic_match = any(
                        any(term in op_t or op_t in _full_ui
                            for term in _q_specific_terms)
                        for op_t in _op_fields_docs
                        if op_t and len(op_t) >= 5
                    )
                    if _op_topic_match:
                        # Bypass operation_group only — department must still be asked when multiple
                        # banks are available, otherwise bot assumes the wrong bank.
                        # entity_type and registration_type must also still be asked.
                        _slot_queue = [
                            s for s in _slot_queue
                            if s.get("key") not in ("operation_group",)
                        ]
                        # Step 3b+: if the query DIRECTLY names an operation_topic option,
                        # bypass it too — no need to show the menu.
                        # Use longest-match-first so "SAN PLUS" beats "SAN" for query "มาตรฐาน SAN PLUS".
                        _ot_slot_3b = next((s for s in _slot_queue if s.get("key") == "operation_topic"), None)
                        if _ot_slot_3b:
                            _ot_opts_3b = sorted(_ot_slot_3b.get("options") or [], key=len, reverse=True)
                            _ENG_THAI_3B = {"PLUS": "พลัส", "SUPER": "ซุปเปอร์", "PRO": "โปร", "MINI": "มินิ"}
                            _matched_ot_3b = (
                                next((o for o in _ot_opts_3b if o and o in _full_ui), None)
                                or next(
                                    (o for o in _ot_opts_3b
                                     if o and any(tok.lower() in _full_ui.lower()
                                                  for tok in re.findall(r"[A-Za-z]{2,}", o))),
                                    None,
                                )
                                or next(
                                    (o for o in _ot_opts_3b
                                     if o and any(
                                         _ENG_THAI_3B.get(tok.upper(), "") in _full_ui
                                         for tok in re.findall(r"[A-Za-z]{2,}", o)
                                     )),
                                    None,
                                )
                            )
                            if _matched_ot_3b:
                                if hasattr(state, "save_collected_slot"):
                                    state.save_collected_slot("operation_topic", _matched_ot_3b)
                                if state.context is not None:
                                    state.context.setdefault("slots", {})["operation_topic"] = _matched_ot_3b
                                    if "collected_slots" in state.context:
                                        state.context["collected_slots"]["operation_topic"] = _matched_ot_3b
                                _slot_queue = [s for s in _slot_queue if s.get("key") != "operation_topic"]
                                if _license_type_for_slots:
                                    try:
                                        _ot_docs_3b = self._practical._retrieve_docs(
                                            f"{_full_ui} {_matched_ot_3b}".strip(),
                                            metadata_filter={"operation_topic": _matched_ot_3b},
                                            max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                                        )
                                        if not _ot_docs_3b:
                                            _ot_docs_3b = self._practical._retrieve_docs(
                                                f"{_full_ui} {_matched_ot_3b}".strip(),
                                                metadata_filter={"license_type": _license_type_for_slots},
                                                max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                                            )
                                        if _ot_docs_3b:
                                            state.current_docs = _ot_docs_3b
                                            state.last_retrieval_query = f"{_full_ui} {_matched_ot_3b}".strip()
                                    except Exception as _e_ot_3b:
                                        _LOG.warning("[Supervisor] Step3b op_topic re-retrieve failed: %s", _e_ot_3b)
                                _LOG.info(
                                    "[Supervisor] Step3b op_topic auto-collected=%r → skip ask, remaining: %s",
                                    _matched_ot_3b, [s["key"] for s in _slot_queue],
                                )
                        _LOG.info(
                            "[Supervisor] operation-specific query %r matched operation field in docs"
                            " — bypassed operation_group slot only, remaining: %s",
                            _full_ui[:60],
                            [s["key"] for s in _slot_queue],
                        )
                        # Also infer operation_group and re-retrieve focused docs so Practical
                        # answers only the matched operation (not all operations).
                        if _license_type_for_slots:
                            try:
                                _og_res_3b = self._get_operation_groups_for_entity(
                                    _license_type_for_slots, ""
                                )
                                _og_labels_3b, _og_raw_3b = (
                                    _og_res_3b if isinstance(_og_res_3b, tuple) else (_og_res_3b, {})
                                )
                                _inferred_3b = self._infer_operation_group_from_query(
                                    _full_ui, _og_labels_3b, state
                                )
                                if _inferred_3b:
                                    if hasattr(state, "save_collected_slot"):
                                        state.save_collected_slot("operation_group", _inferred_3b)
                                    if state.context is not None:
                                        state.context.setdefault("slots", {})["operation_group"] = _inferred_3b
                                        state.context.setdefault("slots", {})["confirmed_operation"] = _inferred_3b
                                        if "collected_slots" in state.context:
                                            state.context["collected_slots"]["operation_group"] = _inferred_3b
                                    _raw_ops_3b = (_og_raw_3b or {}).get(_inferred_3b) or []
                                    _op_pfx_3b = " ".join(_raw_ops_3b[:2]) if _raw_ops_3b else _inferred_3b
                                    _enrich_3b = " ".join(filter(None, [_full_ui, _op_pfx_3b])).strip()
                                    try:
                                        _op_docs_3b = self._practical._retrieve_docs(
                                            _enrich_3b,
                                            metadata_filter={"license_type": _license_type_for_slots},
                                            max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                                        )
                                        if _op_docs_3b:
                                            state.current_docs = _op_docs_3b
                                            state.last_retrieval_query = _enrich_3b
                                            _LOG.info(
                                                "[Supervisor] Step3b op-inferred re-retrieve: op=%r docs=%d",
                                                _inferred_3b, len(_op_docs_3b),
                                            )
                                    except Exception as _e_3b_rr:
                                        _LOG.warning("[Supervisor] Step3b op re-retrieve failed: %s", _e_3b_rr)
                                    _LOG.info(
                                        "[Supervisor] Step3b operation_group auto-inferred=%r",
                                        _inferred_3b,
                                    )
                            except Exception as _e_3b:
                                _LOG.warning("[Supervisor] Step3b op inference failed: %s", _e_3b)

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

                # Recover license_type — prefer current_docs (set by entity-enriched retrieval)
                # over last_topic which may be stale (e.g. set to the user's raw query text
                # when free text was typed while a topic-menu slot was pending).
                _saved_license = ""
                for _d_op in (state.current_docs or []):
                    if isinstance(_d_op, dict):
                        _lt_op = ((_d_op.get("metadata") or {}).get("license_type") or "").strip()
                        if _lt_op:
                            _saved_license = _lt_op
                            break
                if not _saved_license:
                    _saved_license = (state.context or {}).get("last_topic") or ""

                if _saved_entity:
                    # Filter by BOTH entity_type AND license_type so docs from other license types
                    # (e.g. ใบทะเบียนพาณิชย์) cannot sneak in and confuse the LLM.
                    _all_slots_for_q = state.get_collected_slots() or {}
                    # Exclude registration_type values that are NOT business-type names
                    # (e.g. "แยกยื่น/ยื่นรวม", "ประโยชน์ทดแทนที่ได้รับ" from a previous topic)
                    # to prevent stale slot contamination of the enriched query.
                    _BIZTYPE_SQ = ("บริษัท", "ห้างหุ้นส่วน", "ห้าง", "บุคคล", "จำกัด", "สามัญ")
                    _slot_q_parts = [
                        v for k, v in _all_slots_for_q.items()
                        if k not in ("entity_type", "operation_group") and v
                        and not (
                            k == "registration_type"
                            and not any(kw in str(v) for kw in _BIZTYPE_SQ)
                        )
                    ]
                    _op_enrich_parts = [base_q, _op_group_val]
                    if _op_prefix and _op_prefix != _op_group_val:
                        _op_enrich_parts.append(_op_prefix)
                    enriched_q = " ".join(filter(None, _op_enrich_parts + _slot_q_parts)).strip()
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

                # If the selected operation_group maps to ≥2 distinct raw sub-operations,
                # inject an operation_sub_type question so the user can narrow down to the exact
                # row (e.g. "แก้ไขชื่อ" vs "แก้ไขข้อบังคับ") before we answer.
                # When only 1 sub-op exists the queue is cleared and we answer immediately.
                _sub_ops = [v.strip() for v in _raw_ops_for_label if v and v.strip()]
                if len(_sub_ops) >= 2:
                    # When too many sub-ops to show as a flat list (e.g. 26 แก้ไข sub-ops for
                    # ใบทะเบียนพาณิชย์ นิติบุคคล), group them into ≤5 user-friendly categories
                    # so the user sees "ชื่อ / กรรมการ / ที่ตั้ง / ..." instead of 26 items.
                    if len(_sub_ops) > 5:
                        _grp_opts, _grp_raw = self._group_sub_operations(
                            _sub_ops, _saved_license or ""
                        )
                        if len(_grp_opts) >= 2:
                            state.context["topic_slot_queue"] = [{
                                "key": "operation_sub_type",
                                "options": _grp_opts,
                                "question": "ต้องการแก้ไขรายการใดครับ?",
                                "raw_sub_map": _grp_raw,
                                "is_grouped": True,
                            }]
                            _LOG.info(
                                "[Supervisor] operation_sub_type grouped (%d→%d): %s",
                                len(_sub_ops), len(_grp_opts), _grp_opts,
                            )
                        else:
                            # Grouper returned ≤1 group — cap flat list to 5
                            _capped = _sub_ops[:5]
                            state.context["topic_slot_queue"] = [{
                                "key": "operation_sub_type",
                                "options": _capped,
                                "question": "ต้องการดำเนินการรายการใดครับ?",
                                "raw_sub_map": {v: v for v in _capped},
                            }]
                            _LOG.info("[Supervisor] operation_sub_type capped to 5: %s", _capped)
                    else:
                        state.context["topic_slot_queue"] = [{
                            "key": "operation_sub_type",
                            "options": _sub_ops,
                            "question": "ต้องการดำเนินการรายการใดครับ?",
                            "raw_sub_map": {v: v for v in _sub_ops},
                        }]
                    _LOG.info("[Supervisor] operation_sub_type slot injected: %s", _sub_ops[:5])
                else:
                    # BUG-A fix: operation_group slot has been consumed — clear stale queue so
                    # subsequent action='ask' fallback cannot pop it and show a wrong menu.
                    state.context.pop("topic_slot_queue", None)
                    _LOG.info("[Supervisor] topic_slot_queue cleared after operation_group consumed (single sub-op)")

            elif _pending_key == "operation_sub_type":
                _sub_op_val = str(mapped).strip()
                _saved_entity_sub = None
                try:
                    _saved_entity_sub = (state.get_collected_slots() or {}).get("entity_type")
                    if _saved_entity_sub:
                        from service.data_loader import DataLoader as _DL_sub
                        _saved_entity_sub = _DL_sub._normalize_entity_type(_saved_entity_sub)
                except Exception:
                    pass
                _saved_license_sub = ""
                for _d_sub in (state.current_docs or []):
                    if isinstance(_d_sub, dict):
                        _lt_sub = ((_d_sub.get("metadata") or {}).get("license_type") or "").strip()
                        if _lt_sub:
                            _saved_license_sub = _lt_sub
                            break
                if not _saved_license_sub:
                    _saved_license_sub = (state.context or {}).get("last_topic") or ""

                # Grouped category path: raw_sub_map maps category label → list of raw sub-ops.
                # When user picked a grouped category (is_grouped=True), look up the raw sub-ops
                # so retrieval can find the actual docs (not just search for the label text).
                _is_grouped = bool(pending.get("is_grouped"))
                _raw_sub_map_pending = pending.get("raw_sub_map") or {}
                _raw_sub_ops_for_val = _raw_sub_map_pending.get(_sub_op_val, [])

                if _is_grouped and len(_raw_sub_ops_for_val) >= 2 and len(_raw_sub_ops_for_val) <= 5:
                    # Small group (2-5 raw sub-ops) — inject a second-level slot so user picks exactly
                    state.context["topic_slot_queue"] = [{
                        "key": "operation_sub_type",
                        "options": _raw_sub_ops_for_val,
                        "question": "ต้องการดำเนินการรายการใดครับ?",
                        "raw_sub_map": {v: v for v in _raw_sub_ops_for_val},
                    }]
                    _LOG.info(
                        "[Supervisor] operation_sub_type: grouped category=%r → second-level %s",
                        _sub_op_val, _raw_sub_ops_for_val,
                    )
                    state.context.setdefault("slots", {})["confirmed_sub_operation"] = _sub_op_val
                    # Skip the retrieval block below — will run when user answers second-level slot
                else:
                    # Flat sub-op path (original) OR large grouped category (answer all paths).
                    # For exact sub-op: filter by operation_by_department.
                    # For grouped category with >5 raw sub-ops: enrich query with all raw ops and
                    # use entity+license filter only — LLM covers all sub-cases in one answer.
                    if _is_grouped and _raw_sub_ops_for_val:
                        _raw_terms = " ".join(_raw_sub_ops_for_val[:3])
                        enriched_q_sub = " ".join(filter(None, [base_q, _sub_op_val, _raw_terms])).strip()
                    else:
                        enriched_q_sub = " ".join(filter(None, [base_q, _sub_op_val])).strip()

                    try:
                        _k_for_sub = int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 6))
                        if _is_grouped and _raw_sub_ops_for_val:
                            # Category selected: retrieve with entity+license only; semantic search
                            # on enriched query naturally finds docs for all sub-ops in this category
                            _sub_filter_parts_g: List[Dict] = []
                            if _saved_entity_sub:
                                _sub_filter_parts_g.append({"entity_type_normalized": _saved_entity_sub})
                            if _saved_license_sub:
                                _sub_filter_parts_g.append({"license_type": _saved_license_sub})
                            _sub_meta_filter: dict = (
                                {"$and": _sub_filter_parts_g}
                                if len(_sub_filter_parts_g) > 1
                                else (_sub_filter_parts_g[0] if _sub_filter_parts_g else {})
                            )
                            _sub_docs = self._practical._retrieve_docs(
                                enriched_q_sub,
                                metadata_filter=_sub_meta_filter or None,
                                max_docs=_k_for_sub,
                            )
                        else:
                            # Exact sub-op: filter by operation_by_department
                            _sub_filter_parts: List[Dict] = [{"operation_by_department": _sub_op_val}]
                            if _saved_entity_sub:
                                _sub_filter_parts.append({"entity_type_normalized": _saved_entity_sub})
                            if _saved_license_sub:
                                _sub_filter_parts.append({"license_type": _saved_license_sub})
                            _sub_meta_filter = (
                                {"$and": _sub_filter_parts}
                                if len(_sub_filter_parts) > 1
                                else _sub_filter_parts[0]
                            )
                            _sub_docs = self._practical._retrieve_docs(
                                enriched_q_sub,
                                metadata_filter=_sub_meta_filter,
                                max_docs=_k_for_sub,
                            )
                            # Fallback: relax to entity+license only (skip op filter)
                            if not _sub_docs:
                                _fallback_parts = [p for p in _sub_filter_parts if "operation_by_department" not in p]
                                _fallback_filter: dict = (
                                    {"$and": _fallback_parts}
                                    if len(_fallback_parts) > 1
                                    else (_fallback_parts[0] if _fallback_parts else {})
                                )
                                _LOG.info("[Supervisor] operation_sub_type exact filter 0 docs → relax filter")
                                _sub_docs = self._practical._retrieve_docs(
                                    enriched_q_sub,
                                    metadata_filter=_fallback_filter or None,
                                    max_docs=_k_for_sub,
                                )
                        if _sub_docs:
                            state.current_docs = _sub_docs
                        state.last_retrieval_query = enriched_q_sub
                        _LOG.info(
                            "[Supervisor] operation_sub_type retrieval: sub=%r grouped=%s entity=%r license=%r docs=%d",
                            _sub_op_val, _is_grouped, _saved_entity_sub, _saved_license_sub,
                            len(state.current_docs),
                        )
                    except Exception as _e_sub:
                        _LOG.warning("[Supervisor] operation_sub_type retrieval failed: %s", _e_sub)

                    if _sub_op_val:
                        state.context.setdefault("slots", {})["confirmed_sub_operation"] = _sub_op_val
                    state.context.pop("topic_slot_queue", None)

            elif _pending_key == "department":
                # Department slot filled (e.g. user chose ธนาคารกสิกรไทย vs ธนาคารไทยพาณิชย์)
                # Re-retrieve docs filtered by chosen department + entity_type if known.
                try:
                    _dept_val = str(mapped).strip()
                    # Persist so cross-topic memory skips re-asking on topic switch.
                    state.save_collected_slot("department", _dept_val)
                    state.context.setdefault("slots", {})["department"] = _dept_val
                    if "collected_slots" in state.context:
                        state.context["collected_slots"]["department"] = _dept_val
                    _saved_entity_dept = (state.get_collected_slots() or {}).get("entity_type")
                    # Use dept-only filter — adding entity_type_normalized as second key in flat dict
                    # causes Chroma to throw an exception (requires $and syntax), which falls through
                    # to completely unfiltered retrieval. Entity filtering is handled via slot_context
                    # query expansion instead (appends entity value to the embedding query).
                    _dept_filter: dict = {"department": _dept_val}
                    _dept_slot_ctx = {"entity_type": _saved_entity_dept} if _saved_entity_dept else None
                    _dept_docs = self._practical._retrieve_docs(
                        state.last_retrieval_query or _dept_val,
                        metadata_filter=_dept_filter,
                        slot_context=_dept_slot_ctx,
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

            elif _pending_key == "operation_topic":
                # User chose a specific operation topic (e.g. มาตรฐาน SAN vs มาตรฐาน SAN PLUS).
                # Re-retrieve docs filtered by the chosen operation_topic.
                try:
                    _ot_val = str(mapped).strip()
                    state.save_collected_slot("operation_topic", _ot_val)
                    state.context.setdefault("slots", {})["operation_topic"] = _ot_val
                    if "collected_slots" in state.context:
                        state.context["collected_slots"]["operation_topic"] = _ot_val
                    _ot_docs = self._practical._retrieve_docs(
                        state.last_retrieval_query or _ot_val,
                        metadata_filter={"operation_topic": _ot_val},
                        max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 6)),
                    )
                    if _ot_docs:
                        state.current_docs = _ot_docs
                        _LOG.info(
                            "[Supervisor] operation_topic retrieval: topic=%r docs=%d",
                            _ot_val, len(_ot_docs),
                        )
                    else:
                        _LOG.info(
                            "[Supervisor] operation_topic retrieval: no docs for topic=%r, keeping existing",
                            _ot_val,
                        )
                except Exception as _e_ot:
                    _LOG.warning("[Supervisor] operation_topic retrieval failed: %s", _e_ot)
                state.context.pop("topic_slot_queue", None)

            elif _pending_key in ("area_size", "shop_area_type", "operation_location", "location_scope", "location"):
                # Area/location slot filled → re-retrieve docs with both entity + location/area_size
                # metadata filters so LLM gets the exact doc for this user's path.
                # New v3 schema has location + area_size as explicit metadata fields.
                #
                # Exception: context_only slots (e.g. shop_area_type when all docs share the same
                # area options) → user's answer is stored in slots for LLM personalisation only;
                # no re-retrieval since filtering would return the same docs anyway.
                if bool((pending or {}).get("context_only")):
                    _LOG.info(
                        "[Supervisor] %r context-only slot: val=%r → store for LLM, skip re-retrieval",
                        _pending_key, _raw,
                    )
                    # If user gave a specific numeric area value (e.g. "9.1 ตรม"), extract and
                    # store it so the LLM can filter fee tiers to the exact sub-tier that applies.
                    # Without this, all sub-tiers within the category are shown (e.g. both ≤10 and
                    # >10 sqm fee bands even when the user clearly stated a value below 10).
                    if _pending_key in ("shop_area_type", "area_size"):
                        _sqm_match = re.search(
                            r'(\d+(?:[.,]\d+)?)\s*(?:ตร\.?\s*ม\.?|ตารางเมตร)',
                            (user_input or ""),
                        )
                        if _sqm_match:
                            _sqm_str = _sqm_match.group(1).replace(",", ".")
                            try:
                                _sqm_float = float(_sqm_str)
                                state.save_collected_slot("shop_area_sqm", str(_sqm_float))
                                if isinstance(state.context.get("collected_slots"), dict):
                                    state.context["collected_slots"]["shop_area_sqm"] = str(_sqm_float)
                                _LOG.info(
                                    "[Supervisor] shop_area_sqm extracted from user input: %.1f sqm",
                                    _sqm_float,
                                )
                            except ValueError:
                                pass
                else:
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
                        _loc_disp_opts = pending.get("display_options") if isinstance(pending, dict) else []
                        _opts_str = " ".join(str(o) for o in list(_loc_slot_opts or []) + list(_loc_disp_opts or []))
                        if "กรุงเทพ" in _opts_str and "ปริมณฑล" in _opts_str:
                            _location_val = "กรุงเทพฯ"  # e.g. option was 'กรุงเทพฯ และปริมณฑล'
                        else:
                            _location_val = "ต่างจังหวัด"  # default: ปริมณฑล = ต่างจังหวัด
                    if "มากกว่า 200" in _raw_lower or "เกิน 200" in _raw_lower or "> 200" in _raw_lower:
                        _area_size_val = "มากกว่า 200 ตารางเมตร"
                    elif "น้อยกว่า 200" in _raw_lower or "ไม่เกิน 200" in _raw_lower or "< 200" in _raw_lower:
                        _area_size_val = "ไม่เกิน 200 ตารางเมตร"

                    # Build enriched query from all collected slots
                    # Exclude non-business-type registration_type values (e.g. stale "แยกยื่น/ยื่นรวม")
                    _all_slots_area = state.get_collected_slots() or {}
                    _BIZTYPE_AQ = ("บริษัท", "ห้างหุ้นส่วน", "ห้าง", "บุคคล", "จำกัด", "สามัญ")
                    _area_q_parts = [
                        v for k, v in _all_slots_area.items()
                        if k not in ("operation_group",) and v
                        and not (
                            k == "registration_type"
                            and not any(kw in str(v) for kw in _BIZTYPE_AQ)
                        )
                    ]
                    enriched_q_area = " ".join(filter(None, [base_q] + _area_q_parts + [_raw])).strip()

                    # Build combined metadata filter from all known filterable slots.
                    # When area_size is filled, also include location from collected_slots (and vice versa)
                    # so the filter targets the exact user path (e.g. กรุงเทพฯ + >200sqm), preventing
                    # ต่างจังหวัด docs from leaking in when only area_size filter is applied.
                    _filter_parts_area: List[Dict] = []

                    # ALWAYS lock retrieval to the current topic's license_type.
                    # Without this, a location filter alone (e.g. {"location": "กรุงเทพฯ"}) returns docs
                    # from ANY license that has location data — causing cross-license contamination
                    # (e.g. asking about EDC but getting ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร docs).
                    #
                    # Use current_docs as PRIMARY source (set by entity-enriched retrieval with
                    # the correct license). last_topic is UNRELIABLE here — it may be stale from
                    # 2 turns ago (e.g. still pointing to ใบรับรองมาตรฐานร้านอาหาร when user is
                    # now asking about EDC) because last_topic is not updated during slot-ask turns.
                    # Count by frequency — FOCUS FILTER may not have updated state.current_docs
                    # (it only runs when _op_exclude_re is set), so the first doc may be from
                    # a different license than the dominant one in multi-license retrievals.
                    _lt_freq_al: dict = {}
                    for _d_al in (state.current_docs or []):
                        if isinstance(_d_al, dict):
                            _lt_al = ((_d_al.get("metadata") or {}).get("license_type") or "").strip()
                            if _lt_al:
                                _lt_freq_al[_lt_al] = _lt_freq_al.get(_lt_al, 0) + 1
                    _area_license_lock = max(_lt_freq_al, key=_lt_freq_al.__getitem__) if _lt_freq_al else ""
                    if not _area_license_lock:
                        _area_license_lock = (state.context or {}).get("last_topic", "").strip()
                    if _area_license_lock:
                        _filter_parts_area.append({"license_type": _area_license_lock})

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
                    # Always include entity_type_normalized when known.
                    # This ensures the location-stripped fallback in _retrieve_docs
                    # (which strips only location/area_size keys) still narrows to the
                    # correct entity — preventing wrong-entity docs and unrelated-topic
                    # docs (e.g. Refund, WeChat Pay) from leaking in via the license-only
                    # fallback when a license has no location metadata (e.g. EDC).
                    if _saved_entity2:
                        _filter_parts_area.append({"entity_type_normalized": _saved_entity2})

                    _meta_filter_area: dict | None = None
                    if len(_filter_parts_area) == 1:
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

            elif _pending_key == "registration_type" and (state.context or {}).get("_multi_topic_retrieval"):
                # Multi-topic scenario: user just answered registration_type (e.g. บริษัทจำกัด).
                # Re-merge docs for ALL detected license types, now filtered by the chosen registration_type.
                # The single-license path (elif _entity_val) would drop all but _prior_license's docs.
                _raw_rt = _raw
                _cs_rt = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
                _entity_for_mt = (
                    _cs_rt.get("entity_type_normalized")
                    or _cs_rt.get("entity_type")
                    or (state.context or {}).get("slots", {}).get("entity_type", "")
                    or _entity_val
                    or ""
                ).strip()
                _mt_lts: List[str] = []
                for _d_mt in (state.current_docs or []):
                    if not isinstance(_d_mt, dict):
                        continue
                    _lt_mt = ((_d_mt.get("metadata") or {}).get("license_type") or "").strip()
                    if _lt_mt and _lt_mt not in _mt_lts:
                        _mt_lts.append(_lt_mt)
                if _mt_lts:
                    try:
                        _coll_mt_rt = self._get_chroma_collection()
                        if _coll_mt_rt is not None:
                            _doc_chars_mt_rt = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                            _SUPERVISOR_META_WL_MT_RT = frozenset({
                                "license_type", "operation_topic", "entity_type_normalized",
                                "registration_type", "department", "fees", "operation_duration",
                                "service_channel", "operation_steps", "identification_documents",
                                "research_reference",
                            })
                            _SUPERVISOR_FC_MT_RT = {
                                "operation_steps": 600, "identification_documents": 4000,
                                "research_reference": 3200, "fees": 120, "service_channel": 200,
                            }

                            def _build_slim_mt_rt(fc: str, fm: dict) -> Dict:
                                _sm: Dict = {}
                                for _k, _v in (fm or {}).items():
                                    if _k not in _SUPERVISOR_META_WL_MT_RT or _v in (None, "", "nan", "None"):
                                        continue
                                    _vs = str(_v)
                                    _cap = _SUPERVISOR_FC_MT_RT.get(_k)
                                    _sm[_k] = _vs[:_cap] if _cap and len(_vs) > _cap else _vs
                                return {"content": (fc or "")[:_doc_chars_mt_rt], "metadata": _sm}

                            _merged_mt_rt: List[Dict] = []
                            for _lt_rt in _mt_lts:
                                _r_rt = _coll_mt_rt.get(
                                    where={"license_type": _lt_rt},
                                    include=["documents", "metadatas"],
                                )
                                _all_lt_rt = [
                                    _build_slim_mt_rt(_fc, _fm)
                                    for _fc, _fm in zip(_r_rt.get("documents") or [], _r_rt.get("metadatas") or [])
                                ]
                                # Apply entity_type filter (generic docs "" pass through)
                                if _entity_for_mt:
                                    _ent_match = [
                                        d for d in _all_lt_rt
                                        if (d.get("metadata") or {}).get("entity_type_normalized", "") in ("", _entity_for_mt)
                                    ]
                                    if _ent_match:
                                        _all_lt_rt = _ent_match
                                # Apply registration_type filter (generic docs "" pass through)
                                _reg_match = [
                                    d for d in _all_lt_rt
                                    if (d.get("metadata") or {}).get("registration_type", "") in ("", _raw_rt)
                                ]
                                if _reg_match:
                                    _all_lt_rt = _reg_match
                                _merged_mt_rt.extend(_all_lt_rt)
                            if _merged_mt_rt:
                                state.current_docs = _merged_mt_rt
                                _LOG.info(
                                    "[Supervisor] multi-topic registration_type re-merge: reg=%r entity=%r licenses=%s docs=%d",
                                    _raw_rt, _entity_for_mt, _mt_lts, len(_merged_mt_rt),
                                )
                    except Exception as _e_mt_rt:
                        _LOG.warning("[Supervisor] multi-topic registration_type re-merge failed: %s", _e_mt_rt)
                state.context.pop("topic_slot_queue", None)

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

                # Capture current license_type AND generic docs BEFORE retrieval overwrites docs.
                # Chroma entity filter returns docs from ALL licenses with that entity —
                # we post-filter to the original license to prevent cross-license slot discovery.
                # Generic docs (entity_type_normalized='') hold operation_steps, fees, channels —
                # they are excluded by the entity filter but needed for a complete LLM answer.
                _prior_license = None
                _prior_generic_docs: List[Dict] = []
                for _pd in (state.current_docs or []):
                    _plt = ((_pd.get("metadata") or {}).get("license_type") or "").strip()
                    if _plt and not _prior_license:
                        _prior_license = _plt
                    _pet = ((_pd.get("metadata") or {}).get("entity_type_normalized") or "").strip()
                    if _pet == "" and _plt:
                        _prior_generic_docs.append(_pd)

                # Bank mismatch guard: if the legal query names a bank that differs from
                # the department in current_docs, the cached docs are from the wrong bank.
                # Clear _prior_license so entity-enriched retrieval is not locked to the
                # wrong bank's license (e.g. KBANK license when user is asking about SCB).
                _slot_bank_groups = [
                    ("scb",   re.compile(r"ไทยพาณิช(?:ย์)?|ไทยพานิช(?:ย์)?|scb\b", re.IGNORECASE)),
                    ("bbl",   re.compile(r"ธนาคารกรุงเทพ|bbl\b", re.IGNORECASE)),
                    ("ktb",   re.compile(r"กรุงไทย|ktb\b", re.IGNORECASE)),
                    ("kbank", re.compile(r"กสิกร|kbank\b|kasikorn", re.IGNORECASE)),
                    ("ttb",   re.compile(r"ทหารไทย|ธนชาต|ttb\b", re.IGNORECASE)),
                    ("bay",   re.compile(r"กรุงศรี|bay\b", re.IGNORECASE)),
                    ("gsb",   re.compile(r"ออมสิน|gsb\b", re.IGNORECASE)),
                    ("uob",   re.compile(r"\buob\b", re.IGNORECASE)),
                ]
                def _slot_bank_id(text: str) -> Optional[str]:
                    for _b, _p in _slot_bank_groups:
                        if _p.search(text):
                            return _b
                    return None
                _slot_q_for_bank = (
                    (base_q or "")
                    + " "
                    + str((state.context or {}).get("last_user_legal_query") or "")
                ).strip()
                _qbank = _slot_bank_id(_slot_q_for_bank)
                if _qbank and _prior_license:
                    _doc_banks = {
                        _slot_bank_id(str((d.get("metadata") or {}).get("department") or ""))
                        for d in (state.current_docs or [])
                        if isinstance(d, dict)
                    }
                    _doc_banks.discard(None)
                    if _doc_banks and _qbank not in _doc_banks:
                        _LOG.info(
                            "[Supervisor] slot-fill bank mismatch: query_bank=%r doc_banks=%s"
                            " — clearing _prior_license lock so entity retrieval is not bank-locked",
                            _qbank, _doc_banks,
                        )
                        _prior_license = None
                        _prior_generic_docs = []

                # If department was already collected (e.g. user chose ธนาคารไทยพาณิชย์ before
                # entity_type), include it in the filter so retrieval stays within that dept.
                # Without this, entity filter returns docs from ALL departments — post-filtering
                # by license_type alone may leave docs from the wrong department (e.g. Kasikorn).
                _known_dept = None
                _cs_et = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
                _known_dept = (_cs_et.get("department") or "").strip() or None
                if not _known_dept:
                    # Also check current_docs metadata for a consistent department
                    _dept_counts: dict = {}
                    for _pd2 in (state.current_docs or []):
                        _d2 = ((_pd2.get("metadata") or {}).get("department") or "").strip()
                        if _d2:
                            _dept_counts[_d2] = _dept_counts.get(_d2, 0) + 1
                    if len(_dept_counts) == 1:
                        _known_dept = next(iter(_dept_counts))

                if _known_dept:
                    # Include entity-neutral ("") docs alongside entity-specific ones so that
                    # shared procedure/fee/channel docs aren't excluded when department is known.
                    entity_filter = {"$and": [
                        {"$or": [
                            {"entity_type_normalized": _entity_val},
                            {"entity_type_normalized": ""},
                        ]},
                        {"department": _known_dept},
                    ]}
                else:
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

                    # Supplement entity-specific docs with generic docs (entity_type_normalized='')
                    # from the location-filtered set saved before overwrite.
                    # Generic docs hold operation_steps, fees, service_channel, operation_duration
                    # which are NOT duplicated in entity-specific docs — without them the LLM answer
                    # is incomplete (missing procedure, fees, channels).
                    if _prior_generic_docs and state.current_docs is not None:
                        def _doc_dedup_key(d: dict) -> str:
                            _m = d.get("metadata") or {}
                            return (
                                str(_m.get("id") or _m.get("doc_id") or "").strip()
                                or (d.get("content") or "")[:80]
                            )
                        _entity_content_keys = {
                            _doc_dedup_key(d)
                            for d in (state.current_docs or [])
                        }
                        _new_generic = [
                            d for d in _prior_generic_docs
                            if _doc_dedup_key(d) not in _entity_content_keys
                            and ((_d_md2 := d.get("metadata") or {}).get("license_type") or "").strip() == (_prior_license or "")
                        ]
                        if _new_generic:
                            state.current_docs = list(state.current_docs) + _new_generic
                            _LOG.info(
                                "[Supervisor] entity-enriched retrieval: +%d generic docs merged → total %d",
                                len(_new_generic), len(state.current_docs),
                            )
                except Exception as e:
                    _LOG.warning("[Supervisor] entity-enriched retrieval failed: %s", e)

                # When registration_type was just filled, supplement with Chroma coll.get()
                # filtered by registration_type to get the exact docs (avoids pulling in
                # บริษัทมหาชนจำกัด or other sub-type docs via similarity bias).
                if _pending_key2 == "registration_type" and _raw and _prior_license:
                    try:
                        _coll_rt = self._get_chroma_collection()
                        if _coll_rt is not None:
                            # Fetch by license_type only; filter registration_type in Python
                            # to handle compound values like "1.ห้างหุ้นส่วนจำกัด 2.ห้างหุ้นส่วนสามัญ"
                            # that would return 0 docs with exact Chroma match.
                            _rt_all = _coll_rt.get(
                                where={"license_type": _prior_license},
                                include=["documents", "metadatas"],
                            )
                            _rt_pairs_all = list(zip(
                                _rt_all.get("documents") or [],
                                _rt_all.get("metadatas") or [],
                            ))
                            _rt_pairs_fil = [
                                (fc, fm) for fc, fm in _rt_pairs_all
                                if _raw in ((fm or {}).get("registration_type") or "")
                            ]
                            _rt_result = {
                                "documents": [fc for fc, fm in _rt_pairs_fil],
                                "metadatas": [fm for fc, fm in _rt_pairs_fil],
                            }
                            _doc_chars_rt = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                            _SUPERVISOR_META_WL_RT = frozenset({
                                "license_type", "operation_topic", "entity_type_normalized",
                                "registration_type", "department", "fees", "operation_duration",
                                "service_channel", "operation_steps", "identification_documents",
                                "research_reference",
                            })
                            _SUPERVISOR_FC_RT = {
                                "operation_steps": 600, "identification_documents": 4000,
                                "research_reference": 3200, "fees": 120, "service_channel": 200,
                            }
                            _rt_raw_docs: List[Dict] = []
                            for _fc_rt, _fm_rt in zip(_rt_result.get("documents") or [], _rt_result.get("metadatas") or []):
                                _sm_rt: Dict = {}
                                for _k_rt, _v_rt in (_fm_rt or {}).items():
                                    if _k_rt not in _SUPERVISOR_META_WL_RT or _v_rt in (None, "", "nan", "None"):
                                        continue
                                    _vs_rt = str(_v_rt)
                                    _cap_rt = _SUPERVISOR_FC_RT.get(_k_rt)
                                    _sm_rt[_k_rt] = _vs_rt[:_cap_rt] if _cap_rt and len(_vs_rt) > _cap_rt else _vs_rt
                                _rt_raw_docs.append({"content": (_fc_rt or "")[:_doc_chars_rt], "metadata": _sm_rt})
                            if _rt_raw_docs:
                                # Re-rank collection.get() results by similarity score so the most
                                # relevant registration_type doc rises to the top instead of
                                # returning in arbitrary insertion order.
                                _rt_vectorstore = getattr(self._practical.retriever, "vectorstore", None)
                                if _rt_vectorstore is not None:
                                    try:
                                        from langchain_core.documents import Document as _LCDoc
                                        _rt_lc = [
                                            _LCDoc(
                                                page_content=_d["content"],
                                                metadata=_d.get("metadata") or {},
                                            )
                                            for _d in _rt_raw_docs
                                        ]
                                        _rt_pairs = _rt_vectorstore.similarity_search_with_relevance_scores(
                                            enriched_q, k=len(_rt_lc) + 2,
                                            filter={"$and": [
                                                {"license_type": _prior_license},
                                                {"registration_type": _raw},
                                            ]},
                                        )
                                        if _rt_pairs:
                                            _rt_scored: Dict[str, float] = {
                                                _pd.page_content[:60]: _ps
                                                for _pd, _ps in _rt_pairs
                                                if _ps is not None
                                            }
                                            _rt_raw_docs.sort(
                                                key=lambda _d: _rt_scored.get(_d["content"][:60], 0.0),
                                                reverse=True,
                                            )
                                            _LOG.info(
                                                "[Supervisor] registration_type re-ranked %d docs by similarity",
                                                len(_rt_raw_docs),
                                            )
                                    except Exception as _e_rank:
                                        _LOG.debug("[Supervisor] registration_type re-rank failed (non-fatal): %s", _e_rank)

                                # Operation-context filter: if the query has a clear action verb
                                # (แก้ไข/ยกเลิก/ต่ออายุ/สมัคร), only keep rt_docs whose
                                # operation_by_department matches that verb. This prevents a
                                # wrong-operation doc (e.g. "การจดทะเบียน บริษัทจำกัด" when user
                                # asked "แก้ไขชื่อ") from being placed first and confusing the LLM.
                                _OP_ACT_RE_SUPP = re.compile(
                                    r"แก้ไข|ยกเลิก|ต่ออายุ|สมัคร", re.IGNORECASE
                                )
                                _op_acts_supp = _OP_ACT_RE_SUPP.findall(enriched_q or base_q or "")
                                if _op_acts_supp:
                                    _act_kw_supp = set(_op_acts_supp)
                                    _rt_op_matched = [
                                        _d for _d in _rt_raw_docs
                                        if any(
                                            _kw in str((_d.get("metadata") or {}).get("operation_by_department", "") or "")
                                            for _kw in _act_kw_supp
                                        )
                                    ]
                                    if not _rt_op_matched:
                                        _LOG.info(
                                            "[Supervisor] registration_type supplement: no op-match "
                                            "(acts=%s, docs=%d) — skipping, entity-enriched docs will handle",
                                            _act_kw_supp, len(_rt_raw_docs),
                                        )
                                        _rt_raw_docs = []
                                    else:
                                        _rt_raw_docs = _rt_op_matched

                                if _rt_raw_docs:
                                    _rt_specific = _rt_raw_docs
                                    # Merge: specific registration_type docs first, then remaining entity docs (dedup by content prefix)
                                    _seen_rt: set = {d["content"][:60] for d in _rt_specific}
                                    _rt_general = [d for d in (state.current_docs or []) if d.get("content", "")[:60] not in _seen_rt]
                                    state.current_docs = _rt_specific + _rt_general
                                    _LOG.info(
                                        "[Supervisor] registration_type targeted retrieval: %r → %d specific + %d general docs",
                                        _raw, len(_rt_specific), len(_rt_general),
                                    )
                                else:
                                    _LOG.info(
                                        "[Supervisor] registration_type targeted retrieval: %r → 0 op-matched docs — entity-enriched docs kept",
                                        _raw,
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
                    # Use _prior_license (captured BEFORE entity retrieval) to avoid contamination.
                    # state.current_docs may have been updated to non-target-license docs when the
                    # entity-enriched retrieval finds 0 docs for the current license and falls back
                    # to all entity-matching docs — reading license_type from current_docs then
                    # rebuilds the slot queue for the WRONG license (e.g. ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร
                    # instead of เครื่องรูดบัตร EDC).
                    _license_type_for_area = _prior_license or None
                    if not _license_type_for_area:
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
                                elif _slot.get("key") == "department":
                                    # Bank depts → keep (meaningful user choice).
                                    # Govt office depts → skip (not user-choosable; LLM covers all channels).
                                    _slot_dept_are_banks = all(
                                        any(kw in d for kw in ("ธนาคาร", "Bank", "bank"))
                                        for d in (_slot.get("options") or [])
                                    )
                                    if _slot_dept_are_banks:
                                        _filtered_remaining.append(_slot)
                                    else:
                                        _LOG.info(
                                            "[Supervisor] department slot dropped from rebuild — govt offices, LLM explains all channels",
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
                                _inferred_et = self._infer_operation_group_from_query(_query_for_infer_et, _op_groups, state)
                                if _inferred_et:
                                    state.save_collected_slot("operation_group", _inferred_et)
                                    state.context.setdefault("slots", {})["confirmed_operation"] = _inferred_et
                                    _LOG.info("[Supervisor] operation_group auto-inferred=%r (entity path) — skipping ask", _inferred_et)
                                    # Re-retrieve docs filtered to this specific operation so LLM
                                    # doesn't mix new-registration docs with edit/cancel docs.
                                    # Without this, entity-enriched retrieval returns ALL entity docs
                                    # (e.g. both จดทะเบียนใหม่ and แก้ไข for บุคคลธรรมดา) and the LLM
                                    # picks the wrong one since new-registration docs have richer content.
                                    try:
                                        _raw_ops_et = (_raw_op_map or {}).get(_inferred_et) or []
                                        _op_pfx_et = " ".join(_raw_ops_et[:2]) if _raw_ops_et else _inferred_et
                                        _enrich_q_et = " ".join(filter(None, [
                                            _query_for_infer_et, _inferred_et, _op_pfx_et,
                                        ])).strip()
                                        _meta_et: dict = {"entity_type_normalized": _entity_val}
                                        if _license_type_for_og:
                                            _meta_et["license_type"] = _license_type_for_og
                                        # If registration_type is already known, narrow the filter so
                                        # re-retrieval returns specific sub-type docs (not a mix of
                                        # บริษัทจำกัด + ห้างหุ้นส่วน + etc.) and doesn't overwrite
                                        # the registration_type-targeted docs set above.
                                        _cs_et_rt = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
                                        _known_rt_et = (_cs_et_rt.get("registration_type") or "").strip()
                                        _BIZTYPE_RT_ET = ("บริษัท", "ห้างหุ้นส่วน", "บุคคลธรรมดา", "จำกัด", "สามัญ")
                                        if _known_rt_et and any(kw in _known_rt_et for kw in _BIZTYPE_RT_ET):
                                            # Collect all actual registration_type values in current_docs
                                            # that contain _known_rt_et (handles compound values like
                                            # "1.ห้างหุ้นส่วนจำกัด 2.ห้างหุ้นส่วนสามัญ")
                                            _rt_actual: set = {_known_rt_et}
                                            for _d_crt in (state.current_docs or []):
                                                _d_rt_v = ((_d_crt.get("metadata") or {}).get("registration_type") or "").strip()
                                                if _d_rt_v and _known_rt_et in _d_rt_v:
                                                    _rt_actual.add(_d_rt_v)
                                            _rt_filter_et = (
                                                {"$in": list(_rt_actual)} if len(_rt_actual) > 1 else _known_rt_et
                                            )
                                            _meta_et_parts = [{"entity_type_normalized": _entity_val}]
                                            if _license_type_for_og:
                                                _meta_et_parts.append({"license_type": _license_type_for_og})
                                            _meta_et_parts.append({"registration_type": _rt_filter_et})
                                            _meta_et = {"$and": _meta_et_parts}
                                        _redocs_et = self._practical._retrieve_docs(
                                            _enrich_q_et,
                                            metadata_filter=_meta_et,
                                            max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                                        )
                                        # If registration_type filter returned 0 docs:
                                        # - If targeted retrieval already set rt-specific docs in current_docs, skip overwrite
                                        # - Otherwise relax to entity+license
                                        if not _redocs_et and "$and" in _meta_et:
                                            _top_rt = ((state.current_docs[0] if state.current_docs else {}).get("metadata") or {}).get("registration_type") or ""
                                            _targeted_ok = _known_rt_et and _top_rt and _known_rt_et in _top_rt
                                            if _targeted_ok:
                                                _LOG.info(
                                                    "[Supervisor] auto-infer: rt Chroma filter returned 0, "
                                                    "but targeted retrieval already set rt-specific docs — skipping overwrite",
                                                )
                                            else:
                                                _meta_et_fb: dict = {"entity_type_normalized": _entity_val}
                                                if _license_type_for_og:
                                                    _meta_et_fb["license_type"] = _license_type_for_og
                                                _redocs_et = self._practical._retrieve_docs(
                                                    _enrich_q_et,
                                                    metadata_filter=_meta_et_fb,
                                                    max_docs=int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 4)),
                                                )
                                        if _redocs_et:
                                            state.current_docs = _redocs_et
                                            state.last_retrieval_query = _enrich_q_et
                                            state.context.setdefault("slots", {})["confirmed_topic"] = _license_type_for_og
                                            _LOG.info(
                                                "[Supervisor] auto-infer op_group re-retrieval: op=%r entity=%r rt=%r docs=%d",
                                                _inferred_et, _entity_val, _known_rt_et or "any", len(_redocs_et),
                                            )
                                    except Exception as _e_og_retr:
                                        _LOG.warning("[Supervisor] auto-infer op_group re-retrieval failed: %s", _e_og_retr)
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

        # If the user was in a multi-topic Academic session, offer the remaining topics.
        remaining = list(ctx.get("academic_remaining_topics") or [])
        ctx.pop("academic_remaining_topics", None)  # consume — show only once
        state.context = ctx

        if not remaining:
            # No remaining Academic topics — still give user a gentle prompt so session doesn't end abruptly
            _last = (state.context or {}).get("last_topic") or ""
            if _last:
                return f"มีคำถามเพิ่มเติมเกี่ยวกับ{_last} หรือต้องการสอบถามเรื่องอื่นครับ? 😊"
            return "มีคำถามอะไรเพิ่มเติมไหมครับ? บอกได้เลยครับ 😊"

        if len(remaining) == 1:
            return f"ยังมีเรื่อง {remaining[0]} อีกหัวข้อหนึ่งที่เกี่ยวข้องครับ ถ้าอยากรู้รายละเอียดบอกได้เลยครับ 😊"

        shown = remaining[:3]
        topics_str = " / ".join(shown)
        more = f" และอีก {len(remaining) - 3} หัวข้อ" if len(remaining) > 3 else ""
        return (
            f"ยังมีหัวข้อที่เกี่ยวข้องอีกครับ เช่น {topics_str}{more} "
            f"อยากรู้เรื่องไหนต่อ บอกได้เลยครับ 😊"
        )

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
            _lt = (state.context or {}).get("last_topic", "").strip()
            if _lt:
                intro = (
                    "👋 ยินดีต้อนรับกลับครับ! ผม \"น้องสุดยอด Consult Restbiz\"\n"
                    f"ครั้งก่อนเราคุยเรื่อง{_lt} อยู่ครับ\n"
                    "💡 ต้องการสอบถามต่อ หรืออยากให้น้องสุดยอดช่วยเรื่องอื่นครับ?"
                )
            else:
                intro = (
                    "👋 สวัสดีครับ! ผม \"น้องสุดยอด Consult Restbiz\"\n"
                    "ช่วยได้ทั้งเรื่องใบอนุญาต/ภาษี/กฎหมายร้านอาหาร, กลยุทธ์การตลาด และคู่มือเปิดร้าน/เบเกอรี่/คาเฟ่ครับ\n"
                    "💡 อยากให้น้องสุดยอดช่วยอะไรครับ?"
                )
            _EMOJI_NUM = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
            topic_lines = []
            # Use topic_descs passed via state.context if available (returning-user path);
            # fall back to _FIXED_GREETING_MENU descriptions for brand-new users.
            _ctx_descs = (state.context or {}).get("last_menu_topic_descs") or []
            _fixed_descs = [d for _, d in self._FIXED_GREETING_MENU]
            _descs = _ctx_descs if (len(_ctx_descs) == len(menu_topics)) else _fixed_descs
            for i, t in enumerate(menu_topics):
                num = _EMOJI_NUM[i] if i < len(_EMOJI_NUM) else f"{i+1}."
                desc = _descs[i] if i < len(_descs) else ""
                topic_lines.append(f"{num} {t} - {desc}" if desc else f"{num} {t}")
            footer = "พิมพ์ตัวเลข หรือบอกผมได้เลยว่าต้องการข้อมูลใบอนุญาติ หรือ ทริคการจัดการร้านอาหารด้านใดสำหรับร้านของคุณครับ 😊"
            msg = intro + "\n" + "\n".join(topic_lines) + "\n" + footer
            return self._normalize_male(msg)
        prefix = self._get_prefix_llm(kind, state, include_intro=False)
        menu = self._format_numbered_options(menu_topics, max_items=9)
        msg = (prefix.rstrip() + "\n" + menu).strip()
        return self._normalize_male(msg)

    def _handle_slot_correction(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, Optional[str]]:
        """
        Handle "ขอแก้ไข / ตอบผิด / เปลี่ยนใหม่" when user wants to undo a slot answer.
        Returns (state, None) when no correction was actionable (caller falls through).
        """
        ctx = state.context or {}
        collected = dict(state.get_collected_slots() if hasattr(state, "get_collected_slots") else (ctx.get("collected_slots") or {}))
        pending = ctx.get("pending_slot") or {}

        # Nothing to correct
        if not collected and not pending:
            return state, None

        # Friendly labels for slot keys shown to the user
        _SLOT_LABELS: Dict[str, str] = {
            "entity_type": "รูปแบบธุรกิจ",
            "entity_type_normalized": "รูปแบบธุรกิจ",
            "registration_type": "ประเภทการจดทะเบียน",
            "department": "ธนาคาร/หน่วยงาน",
            "location_type": "ที่ตั้งร้าน",
            "operation_group": "ประเภทการดำเนินการ",
        }

        # Pop the most recently collected slot (insertion-order in Python 3.7+)
        popped_key, popped_val = None, None
        if collected:
            popped_key = list(collected.keys())[-1]
            popped_val = collected.pop(popped_key)
            state.context["collected_slots"] = collected
            _LOG.info("[Supervisor] slot_correction: popped %r=%r", popped_key, popped_val)

        # Also clear pending slot — it may depend on the value we just cleared
        if pending:
            state.context.pop("pending_slot", None)
            state.context.pop("topic_slot_queue", None)

        if popped_key:
            label = _SLOT_LABELS.get(popped_key, popped_key)
            msg = self._normalize_male(
                f"ได้เลยครับ ลบ{label} ('{popped_val}') ออกแล้วครับ "
                f"กรุณาระบุข้อมูลใหม่ได้เลยครับ"
            )
        else:
            msg = self._normalize_male(
                "ได้เลยครับ ล้างข้อมูลที่ถามค้างไว้แล้วครับ กรุณาระบุข้อมูลใหม่ได้เลยครับ"
            )

        self._add_assistant(state, msg)
        state.last_action = "slot_correction"
        return state, msg

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
            # Treat pure Thai polite particles (ครับ, ค่ะ, นะ, etc.) as empty —
            # "หวัดดีครับ" should not trigger offtopic deflect just because of ครับ.
            _POLITE_ONLY_RE = re.compile(
                r"^(ครับ|ค่ะ|คะ|นะครับ|นะค่ะ|นะ|จ้า|จ้ะ|น่ะ|คับ|ฮะ|ฮับ)$",
                re.IGNORECASE,
            )
            if _POLITE_ONLY_RE.match(_remainder):
                _remainder = ""
            if (
                _remainder
                and not self._LEGAL_SIGNAL_RE.search(_remainder)
                and not self._looks_like_legal_question(_remainder, state)
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
                msg = self._normalize_male("ครับ 😊 ถามได้เลยครับ ทั้งเรื่องใบอนุญาต ภาษี กฎหมาย การตลาด หรือคู่มือเปิดร้าน")
            else:
                _lt = (state.context or {}).get("last_topic", "").strip()
                if _lt:
                    msg = self._normalize_male(
                        f"สวัสดีครับ 😊 ครั้งก่อนเราคุยเรื่อง{_lt} อยู่ครับ "
                        "ต้องการสอบถามต่อ หรือมีเรื่องอื่นให้ช่วยไหมครับ?"
                    )
                else:
                    msg = self._normalize_male("สวัสดีครับ 😊 มีอะไรให้ช่วยไหมครับ ทั้งเรื่องใบอนุญาต ภาษี การตลาด หรือคู่มือเปิดร้าน")
            self._add_assistant(state, msg)
            state.last_action = "greeting_short"
            _LOG.debug("[Supervisor] _handle_greeting show_menu=False kind=%r input=%r", kind, raw[:40])
            return state, msg

        turns = int(state.context.get("greet_menu_turns") or 0)
        include_intro = turns == 0

        if include_intro:
            # Returning user: reuse topics from a previous session stored in context
            _prev_topics = (state.context or {}).get("last_menu_topics")
            if _prev_topics and isinstance(_prev_topics, list) and len(_prev_topics) >= 2:
                topics = _prev_topics
                _prev_descs = (state.context or {}).get("last_menu_topic_descs")
                topic_descs = _prev_descs if (_prev_descs and len(_prev_descs) == len(topics)) else self._generate_topic_descriptions(topics[:2])
            else:
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

    def _silent_switch_to_academic(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        """Switch to academic silently (no announcement), answer, then auto-return to practical."""
        self._mark_auto_return_if_practical_to_academic(state, "academic")
        state.persona_id = "academic"
        state.context["persona_id"] = "academic"
        state.last_action = "academic_auto_switch"
        # Do NOT set academic_auto_select_all here.
        # Academic Phase 2 (section menu) is a core feature — user must always pick sections
        # so the answer is scoped to their actual case, not a dump of all sections.
        # auto_select_all=True caused irrelevant sections (ยกเลิก/แก้ไข/ต่ออายุ) to appear
        # when user only asked about new registration.
        state.context.pop("academic_auto_select_all", None)

        # Reset any stale academic flow so a fresh intake starts (not stuck at stage='done'
        # from a previous session, which would cause academic to return the generic fallback).
        _old_flow = (state.context or {}).get("academic_flow") or {}
        # Clear stale flow for: done, empty, awaiting_sections (section menu from different topic),
        # and no_sections (terminal state that has no dispatch handler and would loop).
        if str(_old_flow.get("stage") or "") in ("done", "", "awaiting_sections", "no_sections"):
            state.context.pop("academic_flow", None)
            state.context.pop("section_catalog", None)
            state.context.pop("academic_question", None)
            state.context.pop("academic_resume_available", None)

        # Clear stale Practical docs so Academic always does fresh retrieval.
        # EXCEPTION: when the last Practical answer came from a broad two-pass retrieval
        # (_broad_question=True), those docs were NOT entity-filtered — they cover ALL topics
        # the user asked about (ใบอนุญาตจัดตั้ง + VAT + ภาษีป้าย + สุรา + ...).
        # Clearing them would cause Academic to re-retrieve with a single-pass k=12 query
        # and miss topics that were only found via the second retrieval pass.
        # Fix: preserve broad-retrieval docs in context so Academic's _detect_distinct_topics
        # can build a complete topic menu; clear only when docs came from an entity-filtered query.
        _is_broad = bool((state.context or {}).get("_broad_question"))
        _is_subtopic_chapter = bool((state.context or {}).get("_subtopic_chapter_used"))
        if (_is_broad or _is_subtopic_chapter) and state.current_docs:
            # Preserve docs and query for Academic intake so topic menu is built from correct docs.
            # Broad: covers all topics from two-pass retrieval.
            # Sub-topic chapter: covers only the focused chapter (e.g. B2B marketing) — must
            # NOT be cleared or Academic will retrieve with "ขอละเอียดๆ" and get random docs.
            state.context["_broad_retrieval_docs"] = list(state.current_docs)
            state.context["_broad_retrieval_query"] = state.last_retrieval_query or ""
            state.context.pop("_subtopic_chapter_used", None)  # consumed
            _LOG.info(
                "[Supervisor] Preserving %s docs (%d) for Academic topic menu",
                "sub_topic chapter" if _is_subtopic_chapter else "broad retrieval",
                len(state.current_docs),
            )
        else:
            # Before clearing, hand the full-license doc set to academic for link collection.
            # practical may have narrowed current_docs to 1 operation-topic doc; the broader
            # pre-saved set ensures academic link pool has all docs for the license (e.g. both
            # SAN + SAN PLUS), so no form/guide links are lost when switching to academic.
            if not state.context.get("_link_source_docs"):
                _presaved = state.context.get("_presaved_license_docs") or state.current_docs
                if _presaved:
                    # Convert Document objects → dicts so academic _link_doc_matches_topic
                    # can call d.get("metadata") safely (Document has no .get() method).
                    def _to_link_dict(d) -> dict:
                        if isinstance(d, dict):
                            return d
                        return {
                            "content": getattr(d, "page_content", "") or "",
                            "metadata": getattr(d, "metadata", {}) or {},
                        }
                    state.context["_link_source_docs"] = [_to_link_dict(d) for d in _presaved]
                    _LOG.info("[Supervisor] Handed %d pre-saved license docs to academic _link_source_docs", len(_presaved))
            state.current_docs = []
            state.last_retrieval_query = ""

        ctx = state.context or {}

        # Detect if user input has substantive content beyond meta-words / numbers.
        # Used for both replay selection and Bug1-fix guard below.
        _ACADEMIC_META_STRIP_RE = re.compile(
            r"(ขอแบบละเอียด|ขอละเอียด|รายละเอียดเพิ่ม|ขอรายละเอียด|ขยายความ|บอกให้ครบ"
            r"|บอกทุกอย่าง|อยากรู้ครบ|ให้ครบ|ต้องการข้อมูลเพิ่ม|เพิ่มเติม"
            r"|กว่านี้|กว่าเดิม|ละเอียดๆ|ละเอียดขึ้น|ละเอียดหน่อย)",
            re.IGNORECASE,
        )
        _ui_meta_stripped = _ACADEMIC_META_STRIP_RE.sub("", user_input).strip()
        _ui_has_substantive = len(_ui_meta_stripped) >= 5  # True when user named a new specific topic

        # Use user_input if it looks like a legal question OR has substantive new content.
        # Fall back to last known topic only for pure meta-requests ("ขอแบบละเอียด") or
        # number menu answers ("7") that have no standalone meaning.
        if self._looks_like_legal_question(user_input, state) or _ui_has_substantive:
            replay = user_input
        else:
            replay = ctx.get("last_user_legal_query", "").strip() or ctx.get("last_topic", "").strip() or user_input

        # Bug 1 fix: if Practical just answered a single-license topic, inject that topic
        # so Academic auto-selects it and skips the Phase 0 topic menu entirely.
        # ALWAYS clear stale academic_selected_license_type first — each new Academic session
        # must start clean. The old value from the previous topic must not carry over.
        state.context.pop("academic_selected_license_type", None)

        _last_lt = (ctx.get("last_answered_license_type") or "").strip()
        _detected_lts_sv = self._detect_license_types_from_query(user_input, state)
        if _last_lt:
            # Only inject when user's follow-up is about the SAME topic (depth request).
            # If user asks about a DIFFERENT license → don't inject; let Academic pick topic.
            # Exclude _last_lt itself from "other" check — "ปิดงบบัญชีอย่างละเอียด" detects
            # 'จัดการการเงิน' via _MULTI_TOPIC_LICENSE_KEYWORDS but IS the same topic.
            _other_license_mentioned = bool([lt for lt in _detected_lts_sv if lt != _last_lt])
            # Guard: if user input has substantive content that mentions no license at all
            # (e.g. a marketing question like "ขยายความส่วนประสมสินค้า"), don't inject the
            # stale license — Academic must retrieve fresh docs for the new topic instead.
            _substantive_non_license = _ui_has_substantive and not _detected_lts_sv
            if not _other_license_mentioned and not _substantive_non_license:
                state.context["academic_selected_license_type"] = _last_lt
                _LOG.info(
                    "[Supervisor] Bug1-fix: auto-injected academic_selected_license_type=%r from last Practical answer",
                    _last_lt,
                )
        elif len(_detected_lts_sv) == 1:
            # Fresh Academic session where user's query explicitly names exactly one license type.
            # Pre-inject to skip the license-type menu and go straight to slot questions/answer.
            state.context["academic_selected_license_type"] = _detected_lts_sv[0]
            _LOG.info(
                "[Supervisor] auto-injected academic_selected_license_type=%r from fresh query",
                _detected_lts_sv[0],
            )

        st2, reply = self._academic.handle(state, replay, _internal=False)
        st2, reply = self._post_route_academic_auto_return(st2, reply)
        reply = self._normalize_male(reply)
        self._add_assistant(st2, reply)
        return st2, reply

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
        # Keep 30 items: enough for menu diversity sampling without ballooning the context.
        if st.context and len(st.context.get("topic_pool") or []) > 30:
            st.context["topic_pool"] = st.context["topic_pool"][:30]
        _elapsed = _time.perf_counter() - _t0
        _LOG.info(
            "[Supervisor] response time = %.2fs | user=%r | reply_len=%d",
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

        # Clear per-turn flags that must not bleed into the next query.
        state.context.pop("_direct_topic_match", None)

        if not state.context.get("did_greet") and not raw_stripped and not self._is_academic_intake_active(state):
            state.context["did_greet"] = True
            return self._handle_greeting(state, user_input="")

        if not state.context.get("did_greet"):
            state.context["did_greet"] = True

        if raw_stripped:
            self._add_user(state, raw)

        # Proactive trim: keep history bounded before LLM calls to reduce token usage.
        # Post-call trim in llm_call.py (keep_last=8) handles further reduction after answer.
        if hasattr(state, "trim_messages"):
            state.trim_messages(keep_last=10)

        # MAX_ROUNDS enforcement: hard ceiling on conversation turns per session.
        # Checked after _add_user so the user's final message is preserved in history
        # before the limit reply is returned. Set MAX_ROUNDS=0 in env.properties to disable.
        _max_rounds = int(getattr(conf, "MAX_ROUNDS", 15) or 15)
        _cur_round = int(getattr(state, "round", 0) or 0)
        if _max_rounds > 0 and _cur_round >= _max_rounds:
            _LOG.info(
                "[Supervisor] MAX_ROUNDS=%d reached (current=%d) — returning session limit message",
                _max_rounds, _cur_round,
            )
            _limit_msg = self._normalize_male(
                "ขออภัยครับ เราได้ตอบคำถามครบ %d รอบสำหรับการสนทนานี้แล้ว "
                "กรุณาเปิดเซสชันใหม่เพื่อถามคำถามเพิ่มเติมครับ" % _max_rounds
            )
            self._add_assistant(state, _limit_msg)
            state.last_action = "max_rounds_reached"
            return state, _limit_msg

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

            # Bug 5 fix: Detect in-session Academic topic selection.
            # When Academic showed a numbered topic menu (pending_options) or user text
            # contains a known topic key from academic_topic_catalog, treat as topic
            # navigation within the same Academic session → bypass _topic_changed guard.
            _is_remaining_topic_select = False
            _pending_opts = (state.context or {}).get("pending_options") or {}
            _topic_catalog = (state.context or {}).get("academic_topic_catalog") or []
            if isinstance(_pending_opts, dict) and _pending_opts:
                _opt_nums = [int(x) for x in re.findall(r"\d{1,2}", raw_stripped) if 0 < int(x) < 100]
                if any(n in _pending_opts for n in _opt_nums):
                    _is_remaining_topic_select = True
                if not _is_remaining_topic_select:
                    for _ov in _pending_opts.values():
                        _ov_s = str(_ov or "").strip()
                        if len(_ov_s) >= 5 and _ov_s in raw_stripped:
                            _is_remaining_topic_select = True
                            break
            if not _is_remaining_topic_select and _topic_catalog:
                for _tc in _topic_catalog:
                    _tc_key = str((_tc.get("key") or "")).strip()
                    if len(_tc_key) >= 8 and _tc_key in raw_stripped:
                        _is_remaining_topic_select = True
                        break
            if _is_remaining_topic_select:
                _LOG.info("[Supervisor] academic_resume: in-session topic selection %r", raw_stripped[:50])

            _topic_changed = (not _is_remaining_topic_select) and bool(
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
                # Check academic STOP before resume — "พอแล้ว", "กลับโหมดปกติ", "ออกจากโหมดนี้"
                _is_stop = bool(self._ACADEMIC_STOP_RE.search(raw_stripped))
                if not _is_stop:
                    _is_stop = self._academic_stop_llm_check(raw_stripped, state)
                if _is_stop:
                    _LOG.info("[Supervisor] academic_stop detected: %r", raw_stripped[:50])
                    state.context.pop("academic_flow", None)
                    state.context.pop("academic_resume_available", None)
                    state.context.pop("section_catalog", None)
                    _stop_reply = self._normalize_male(
                        "ได้ครับ กลับมาที่โหมดปกติแล้ว อยากถามเรื่องอื่นสามารถบอกผมได้เลยครับ"
                    )
                    self._add_assistant(state, _stop_reply)
                    state.last_action = "academic_stop"
                    return state, _stop_reply

                # Menu topic selection: skip all regex/LLM resume checks and go directly to Academic
                if _is_remaining_topic_select:
                    _is_resume = True
                else:
                    _is_resume = bool(self._ACADEMIC_RESUME_RE.search(raw_stripped))
                if not _is_resume:
                    # Guard: new legal questions must never trigger academic resume.
                    # Only call LLM when input doesn't look like a legal/follow-up question.
                    if not self._looks_like_legal_question(raw_stripped, state):
                        # LLM path A: dedicated academic-resume classifier (faster, more precise).
                        # Knows about section navigation ("ขอดูส่วนอื่น", "ส่วนที่ยังไม่ได้อ่าน").
                        if self._academic_resume_llm_check(raw_stripped, _academic_q, state):
                            _is_resume = True
                            _LOG.info("[Supervisor] academic_resume via dedicated LLM: %r", raw_stripped[:40])

                        # LLM path B: general fallback_intent (catches "elaborate" on same topic).
                        # Store result in _cached_fallback_intent so step 2.3 can reuse it.
                        if not _is_resume:
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
                                # Topic-mismatch guard: strip elaboration verb + meta-words, then
                                # compare remaining topic content against academic_question.
                                # Bug fix: 4-char window caused false positives from shared words
                                # like "กลยุทธ์" appearing in both pricing and product queries.
                                # Solution: strip all strategy/meta vocabulary first, then compare
                                # only the domain-specific topic words that remain.
                                if _academic_q:
                                    _elab_stripped = re.sub(
                                        r'^(ขยายความ|อธิบาย\S*|เพิ่มเติม|รายละเอียด|บอกเพิ่ม|เล่าให้ฟัง|ขยาย)\s*',
                                        '', raw_stripped, count=1, flags=re.IGNORECASE
                                    ).strip()
                                    if len(_elab_stripped) >= 4:
                                        # Remove shared meta/strategy vocabulary before comparing
                                        # so that "กลยุทธ์", "ด้าน", etc. don't create false overlaps
                                        _META_STRIP_RE = re.compile(
                                            r'(กลยุทธ์|การตลาด|ต้องการ|เกี่ยวกับ|การ|ด้าน|ของ|เรื่อง|'
                                            r'หัวข้อ|วิธี|แนวทาง|แบบ|ประเภท|รูปแบบ|เชิง|สำหรับ|'
                                            r'อย่าง|ละเอียด|ขอ|ของการ)'
                                        )
                                        _acad_clean = _META_STRIP_RE.sub('', _academic_q.lower()).strip()
                                        _new_clean  = _META_STRIP_RE.sub('', _elab_stripped.lower()).strip()
                                        # Check if any 8-char window from cleaned academic_q
                                        # appears in the cleaned new topic string.
                                        # 8 chars (vs old 4) prevents false positives from shared
                                        # root words — "ทะเบียน" (7 chars) would still match
                                        # unrelated topics with window ≤7; 8 chars requires the
                                        # first unique character after the root to also match.
                                        _has_topic_overlap = bool(_acad_clean) and bool(_new_clean) and any(
                                            _acad_clean[_i:_i+8] in _new_clean
                                            for _i in range(max(0, len(_acad_clean) - 7))
                                        )
                                        if not _has_topic_overlap:
                                            _LOG.info(
                                                "[Supervisor] academic_resume cancelled — topic %r (clean=%r) differs from academic_q %r (clean=%r)",
                                                _elab_stripped[:40], _new_clean[:20],
                                                _academic_q[:40], _acad_clean[:20],
                                            )
                                            _is_resume = False
                            except Exception:
                                pass
                if _is_resume:
                    _LOG.info("[Supervisor] academic_resume triggered input=%r", raw_stripped[:40])
                    state.context.pop("academic_resume_available", None)
                    state.context.pop("pending_slot", None)  # clear practical topics menu
                    state.persona_id = "academic"
                    state.context["persona_id"] = "academic"
                    # For topic menu selection, set stage to awaiting_topic so Academic
                    # processes the input as a new topic selection from the catalog.
                    # For regular resume, reset to awaiting_sections so section_catalog is reused.
                    flow = dict(state.context.get("academic_flow") or {})
                    flow["stage"] = "awaiting_topic" if _is_remaining_topic_select else "awaiting_sections"
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
        # Also bypass 2.2c when there is an active legal conversation — short follow-ups
        # like "แล้วถ้าเป็นนิติบุคคลอ่ะ" lack standalone keywords but are clearly in-domain.
        _has_active_legal_ctx = bool(
            (state.context or {}).get("last_user_legal_query")
            or (state.context or {}).get("last_topic")
        )
        if _pending_slot_active or _has_active_legal_ctx:
            pass  # fall through to practical routing below
        elif (
            raw_stripped
            and not self._LEGAL_SIGNAL_RE.search(raw_stripped)
            and not self._LIKELY_SELECTION_RE.match(raw_stripped)
            and not self._ANY_NUMBER_RE.search(raw_stripped)
            and not self._QUESTION_MARKERS_RE.search(raw_stripped)
            and not self._looks_like_greeting_or_thanks(raw_stripped, state)
            and not self._looks_like_legal_question(raw_stripped, state)
            and not self._DEPTH_DETAIL_RE.search(raw_stripped)
            and not self._ELABORATE_RE.search(raw_stripped)
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
        _raw_is_legal = self._looks_like_legal_question(raw_stripped, state)
        _last_q_for_style = state.get_last_retrieval_query() or (state.context or {}).get("last_topic", "")
        style = self._infer_user_style_request_hybrid(raw_stripped, _last_q_for_style)
        # _is_short_depth_followup: allows academic switch even when input has legal keywords.
        # Two paths:
        # A) Regex path: explicit depth keyword (e.g. เจาะลึก, ละเอียด) — always accepted.
        # B) LLM path: no regex match but LLM detected wants_long with high confidence (≥0.80).
        #    This handles natural phrasing the regex doesn't cover (e.g. "บอกให้ครบทุกอย่าง").
        _llm_high_conf_depth = (
            style.get("method") == "llm"
            and float(style.get("confidence") or 0.0) >= 0.80
        )
        # Numbered item reference (e.g. "ขอช่องทางที่ 1", "วิธีที่ 2 อธิบายให้") — user is
        # asking about a specific item from the PREVIOUS answer, not requesting a new academic
        # deep-dive.  Practical LLM can handle this via conversation history.
        _is_numbered_item_ref = bool(re.search(r'ช่องทางที่\s*\d+|วิธีที่\s*\d+', raw_stripped))
        _is_short_depth_followup = (
            style.get("wants_long")
            and _raw_is_legal
            and not _is_numbered_item_ref
            and len(raw_stripped.split()) <= 10
            and (bool(self._DEPTH_DETAIL_RE.search(raw_stripped)) or _llm_high_conf_depth)
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
        if self._looks_like_switch_without_target(raw_stripped, state):
            cur = normalize_persona_id(state.persona_id)
            if cur != "academic":
                return self._silent_switch_to_academic(state, raw_stripped)
            # Already in academic but intake not active (section 2.1 would've caught that) → ignore

        # 2.5.5) Number typed but no pending_slot → recover from last_topic_menu
        if not self._has_pending_slot(state) and self._LIKELY_SELECTION_RE.match(raw_stripped):
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

        # 2.5.6) Dynamic clarification answer — user replied to a DC question
        if (state.context or {}).get("pending_dynamic_clarification"):
            return self._resolve_dynamic_clarification(state, raw_stripped)

        # 2.5.7) Slot correction: "ขอแก้ไข", "ตอบผิด", "เปลี่ยนใหม่"
        # Guard: only handle if correction RE matches AND it doesn't look like a legal question
        # (e.g., "ขอแก้ไขข้อมูลในทะเบียน" is a legal question, not a slot correction).
        if (
            self._SLOT_CORRECTION_RE.search(raw_stripped)
            and not self._looks_like_legal_question(raw_stripped, state)
        ):
            _corr_state, _corr_reply = self._handle_slot_correction(state, raw_stripped)
            if _corr_reply is not None:
                return _corr_state, _corr_reply

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
                _LOG.debug("[Supervisor] typo_rule_fast input=%r", raw_stripped[:30])
            elif (
                not self._looks_like_greeting_or_thanks(raw_stripped, state)
                and not self._looks_like_legal_question(raw_stripped, state)
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

        # 2.7) greeting/noise/smalltalk → short reply only, no topic menu
        # _SMALLTALK_RE must be checked here explicitly because smalltalk phrases like
        # "สบายดีไหม" contain question markers (ไหม) that cause _looks_like_legal_question()
        # to return True, bypassing this guard and routing to the expensive LLM retrieval path.
        # LLM fallback catches missed smalltalk: "เป็นไงบ้างบอท", "คิดว่ายังไง", "ว่างมั้ย".
        if (
            self._looks_like_greeting_or_thanks(raw_stripped, state)
            or self._SMALLTALK_RE.search(raw_stripped)
            or self._smalltalk_llm_check(raw_stripped, state)
            or self._is_noise(raw_stripped)
            or not raw_stripped
        ):
            # 2.7a) Yes/No guard — when short input would go to greeting handler but there's
            # an active legal conversation, check if it's actually a yes/no response.
            # e.g. "อือ" (yes, 2 chars) after bot answered a question → elaborate the last topic.
            _yn_ctx = (state.context or {}).get("last_user_legal_query", "").strip()
            if _yn_ctx and raw_stripped and len(raw_stripped) <= 10:
                try:
                    _yn = self._classify_yes_no_hybrid(raw_stripped)
                    _yn_conf = float(_yn.get("confidence") or 0.0)
                    if _yn.get("yes") and _yn_conf >= 0.78:
                        # Guard: if there's still a pending slot, "yes" confirms something related
                        # to that slot (e.g. bot suggested a bank name, user confirmed). Route back
                        # to slot handler so the slot question is re-asked cleanly rather than
                        # silently cleared and the slot lost.
                        if self._has_pending_slot(state):
                            _LOG.info(
                                "[Supervisor] 2.7a yes_hybrid: pending_slot active → re-route to slot handler instead of elaborate",
                            )
                            return self._route_pending_slot_to_persona(state, raw_stripped)
                        # User is affirming — elaborate on last legal query
                        _LOG.info(
                            "[Supervisor] 2.7a yes_hybrid: %r → elaborate last_q=%r",
                            raw_stripped[:20], _yn_ctx[:40],
                        )
                        self._ensure_practical_retrieval_for_legal(state, _yn_ctx)
                        _elab_input = f"อธิบายรายละเอียดเพิ่มเติม: {_yn_ctx}"
                        state.context.pop("pending_slot", None)
                        _st2, _reply = self._practical.handle(state, _elab_input, _internal=False)
                        _reply = self._normalize_male(_reply)
                        self._add_assistant(_st2, _reply)
                        _st2.last_action = "practical_yes_elaborate"
                        return _st2, _reply
                    if _yn.get("no") and _yn_conf >= 0.78:
                        # User is negating — show topic menu
                        _LOG.info("[Supervisor] 2.7a no_hybrid: %r → topic menu", raw_stripped[:20])
                        return self._handle_greeting(state, user_input=raw_stripped, show_menu=True)
                except Exception as _yne:
                    _LOG.debug("[Supervisor] 2.7a yes_no_hybrid error: %s", _yne)
            return self._handle_greeting(state, user_input=raw_stripped, show_menu=False)

        # 2.8) mode status
        if self._looks_like_mode_status_query(raw_stripped, state):
            pid = normalize_persona_id(state.persona_id)
            msg = self._normalize_male(f"ℹ️ ตอนนี้เป็นโหมด {pid} ครับ")
            self._add_assistant(state, msg)
            state.last_action = "mode_status"
            return state, msg

        # 2.9) legal routing
        if self._looks_like_legal_question(raw_stripped, state):
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
            #   - "entity_type": answers "นิติบุคคล"/"บุคคลธรรมดา" trigger LEGAL_SIGNAL_RE
            #     but are always slot fills — clearing them wipes license context and routes to wrong license
            _INTERRUPT_EXEMPT = {"topic", "operation_group", "operation_sub_type", "registration_type", "department", "entity_type"}
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
                        or (not self._ACTION_Q_RE.search(raw_stripped) and self._info_action_q_llm_check(raw_stripped, state)[0])
                        or self._LINK_REQUEST_RE.search(raw_stripped)
                        or self._link_request_llm_check(raw_stripped, state)
                        or self._GENERIC_FOLLOWUP_KEYWORDS_RE.search(raw_stripped)
                        # "ต้องการ..." / "อยากรู้..." are new questions, not menu selections.
                        # Guard: must have no number (otherwise "ต้องการข้อ 3" is a valid pick).
                        or (
                            bool(self._TOPIC_REQUEST_PREFIX_RE.match(raw_stripped))
                            and len(raw_stripped) >= 8
                            and not self._ANY_NUMBER_RE.search(raw_stripped)
                        )
                        # Legal question that doesn't match ANY greeting menu option → must be a new
                        # topic question, not a menu selection. Apply same logic as non-topic exempt
                        # slots: if no option text appears in input AND input is a legal question,
                        # bypass the menu so 2.9 can handle it properly.
                        # Example: "การแยกยื่นในการส่งข้อมูลสมทบ" when greeting menu shows [ใบอนุญาตจัดตั้ง, QR-Payment]
                        or (
                            _int_ps.get("options")
                            and not any(str(o) and str(o) in raw_stripped for o in (_int_ps.get("options") or []))
                            and len(raw_stripped) >= 8
                            and not self._ANY_NUMBER_RE.search(raw_stripped)
                            and self._looks_like_legal_question(raw_stripped, state)
                        )
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
                        and self._looks_like_legal_question(raw_stripped, state)
                    ):
                        _bypass_non_topic_slot = True
                if _bypass_topic_slot or _bypass_non_topic_slot:
                    _LOG.info(
                        "[Supervisor] 2.9 %s slot: new legal question bypasses pending slot: %r",
                        _exempt_key, raw_stripped[:60],
                    )
                    # Clear pending_slot so practical doesn't re-evaluate it and show INVALID menu.
                    state.context.pop("pending_slot", None)
                    # Only clear topic_slot_queue when bypassing a TOPIC slot (genuine new topic).
                    # For non-topic slots (entity_type, operation_group, etc.) keep the queue so
                    # that a false-positive legal keyword (e.g. "บริษัทจำกัด") doesn't wipe all
                    # pending slots mid-collection — the queue will be re-evaluated next turn.
                    if _bypass_topic_slot:
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
            self._maybe_build_slot_queue_from_docs(state, raw_stripped)
            if not (self._LINK_REQUEST_RE.search(raw_stripped) or self._link_request_llm_check(raw_stripped, state)) and not (state.context or {}).get("topic_slot_queue"):
                _dc_q_29 = self._maybe_dynamic_clarification(state, raw_stripped)
                if _dc_q_29:
                    self._add_assistant(state, _dc_q_29)
                    state.last_action = "dynamic_clarification_asked"
                    return state, _dc_q_29
            state.context.pop("pending_slot", None)
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
        # Guard: if user attached a new topic ("อธิบายเพิ่ม เรื่อง QR Payment"), strip elaborate
        # words first — if meaningful text remains AND it looks like a legal question, treat as
        # a new legal query rather than elaborating on the last topic.
        # LLM fallback fires when regex misses — catches indirect phrasing that doesn't match keywords.
        _is_elaborate = self._ELABORATE_RE.search(raw_stripped)
        if not _is_elaborate:
            _is_elaborate = self._elaborate_llm_check(raw_stripped, state)
        if _is_elaborate:
            _elab_strip_re = re.compile(
                r"(อธิบาย(มากกว่า|เพิ่ม|เพิ่มเติม|ขยาย|ให้ละเอียด|ต่อ)|ขยายความ|เพิ่มเติมอีก"
                r"|รายละเอียดมากกว่า|บอกเพิ่ม|เล่าให้ฟัง|อธิบายต่อ|รายละเอียดเพิ่ม)",
                re.IGNORECASE,
            )
            _elab_remainder = _elab_strip_re.sub("", raw_stripped).strip()
            _elab_is_new_topic = (
                len(_elab_remainder) >= 5
                and (
                    self._LEGAL_SIGNAL_RE.search(_elab_remainder)
                    or self._looks_like_legal_question(_elab_remainder, state)
                )
            )
            if _elab_is_new_topic:
                _LOG.info(
                    "[Supervisor] elaborate+new_topic detected — routing as legal_q remainder=%r",
                    _elab_remainder[:60],
                )
                raw_stripped = _elab_remainder  # fall through to legal routing below
            else:
                last_q = (state.context or {}).get("last_user_legal_query", "").strip()
                if last_q:
                    _slot_chgd_elab = self._apply_slot_change_if_detected(state, raw_stripped)
                    _LOG.info("[Supervisor] elaborate→practical last_q=%r", last_q[:40])
                    if not _slot_chgd_elab:
                        self._ensure_practical_retrieval_for_legal(state, last_q)
                    # When user names a specific section ("การสร้างบรรยากาศ..."), focus LLM on
                    # that section only instead of re-explaining the full topic.
                    # _elab_remainder = raw_stripped minus generic elaborate signal words.
                    # A "pure generic" request ("ขอแบบละเอียด", "รายละเอียดมากกว่านี้") should
                    # still use last_q so the LLM gives a broad deep-dive.
                    _PURE_ELAB_RE = re.compile(
                        r'^(ขอ|บอก|เล่า|ให้|ช่วย)?(แบบ)?(ละเอียด|รายละเอียด|เพิ่มเติม|มากกว่า|อีกหน่อย|ต่อ|ไหม|มั้ย)[ก-๙\s]*$',
                        re.IGNORECASE,
                    )
                    _elab_focus = (
                        _elab_remainder
                        if _elab_remainder and len(_elab_remainder) >= 6 and not _PURE_ELAB_RE.search(_elab_remainder)
                        else ""
                    )
                    if _elab_focus:
                        elaborate_input = f"อธิบายรายละเอียดเพิ่มเติมเกี่ยวกับ: {_elab_focus}"
                        _LOG.info("[Supervisor] elaborate: section-specific focus=%r", _elab_focus[:50])
                    else:
                        elaborate_input = f"อธิบายรายละเอียดเพิ่มเติม: {last_q}"
                    state.context.pop("pending_slot", None)
                    st2, reply = self._practical.handle(state, elaborate_input, _internal=False)
                    reply = self._normalize_male(reply)
                    self._add_assistant(st2, reply)
                    st2.last_action = "practical_elaborate"
                    return st2, reply

        # 3.3) "อันไหนเหมาะกับฉัน", "สำหรับกรณีฉัน" → contextual follow-up with last legal query
        # LLM fallback catches indirect phrasing: "แบบนี้ใช้ได้กับร้านผมไหม", "ของฉันต้องทำอะไร"
        _is_followup_ctx = bool(self._FOLLOWUP_CONTEXTUAL_RE.search(raw_stripped))
        if not _is_followup_ctx and len(raw_stripped) >= 4:
            _is_followup_ctx = self._followup_contextual_llm_check(raw_stripped, state)
        if _is_followup_ctx:
            last_q = (state.context or {}).get("last_user_legal_query", "").strip()
            if last_q:
                _LOG.info("[Supervisor] contextual_followup→practical input=%r last_q=%r", raw_stripped[:30], last_q[:30])
                self._ensure_practical_retrieval_for_legal(state, last_q)
                combined = f"{raw_stripped} (เกี่ยวกับ: {last_q})"
                state.context.pop("pending_slot", None)
                st2, reply = self._practical.handle(state, combined, _internal=False)
                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "practical_contextual_followup"
                return st2, reply

        # 3.4) Link/document request with active context → practical directly (no LLM needed)
        # Catches "ขอลิงค์คู่มือ", "ส่งแบบฟอร์ม", "ขอดาวน์โหลด" etc. before LLM misclassifies as new_topic
        if self._LINK_REQUEST_RE.search(raw_stripped) or self._link_request_llm_check(raw_stripped, state):
            _last_q_lr = (state.context or {}).get("last_user_legal_query", "").strip()
            _has_ctx_lr = bool(_last_q_lr or (state.context or {}).get("last_topic") or state.current_docs)
            if _has_ctx_lr:
                _LOG.info("[Supervisor] 3.4 link_request with active context → practical: %r", raw_stripped[:60])
                if _last_q_lr:
                    self._ensure_practical_retrieval_for_legal(state, _last_q_lr)
                self._maybe_build_slot_queue_from_docs(state, raw_stripped)
                state.context.pop("pending_slot", None)
                st2, reply = self._practical.handle(state, raw_stripped, _internal=False)
                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "practical_link_request"
                return st2, reply

        # 4) LLM fallback intent classifier — no hardcode, no dead-end error message
        # FIX #3 (part 2): reuse the cached fallback-intent result from step 2.2b if available,
        # so we never call llm_fallback_intent_call twice on the same input in one turn.
        _LOG.debug("[Supervisor] fallback_intent_llm persona=%s input=%r", getattr(state, "persona_id", "?"), raw_stripped[:60])
        intent_res: Dict[str, Any] = {}
        if _cached_fallback_intent is not None:
            intent_res = _cached_fallback_intent
            _LOG.debug("[Supervisor] fallback_intent reused from step 2.2b cache")
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
                self._maybe_build_slot_queue_from_docs(state, raw_stripped)
                if not (self._LINK_REQUEST_RE.search(raw_stripped) or self._link_request_llm_check(raw_stripped, state)) and not (state.context or {}).get("topic_slot_queue"):
                    _dc_q_nt = self._maybe_dynamic_clarification(state, raw_stripped)
                    if _dc_q_nt:
                        self._add_assistant(state, _dc_q_nt)
                        state.last_action = "dynamic_clarification_asked"
                        return state, _dc_q_nt
                state.context.pop("pending_slot", None)
                st2, reply = self._practical.handle(state, raw_stripped, _internal=False)
                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "practical_follow_up"
                return st2, reply
            return self._handle_greeting(state, raw_stripped)

        if fallback_intent == "elaborate":
            last_q_fb2 = (state.context or {}).get("last_user_legal_query", "").strip()
            if last_q_fb2:
                _slot_chgd_fb_elab = self._apply_slot_change_if_detected(state, raw_stripped)
                # Detect operation_topic switch in elaborate phrasing (e.g. "แบบพลัส" → SAN PLUS).
                # Checks: Thai substring, English tokens, common Thai loanwords.
                _cs_elab_ot = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
                _cur_ot_elab = (_cs_elab_ot.get("operation_topic") or "").strip()
                if _cur_ot_elab:
                    _lt_elab = next(
                        (((d.get("metadata") or {}).get("license_type") or "").strip()
                         for d in (state.current_docs or []) if isinstance(d, dict)
                         if ((d.get("metadata") or {}).get("license_type") or "").strip()),
                        "",
                    )
                    if _lt_elab:
                        try:
                            _coll_elab = self._get_chroma_collection()
                            if _coll_elab is not None:
                                _elab_mds = (_coll_elab.get(
                                    where={"license_type": _lt_elab}, include=["metadatas"]
                                ).get("metadatas") or [])
                                _ot_all_elab = sorted({
                                    ((_md or {}).get("operation_topic") or "").strip()
                                    for _md in _elab_mds
                                    if ((_md or {}).get("operation_topic") or "").strip() not in ("", "nan", "None")
                                })
                                _cur_eng_elab = {t.lower() for t in re.findall(r"[A-Za-z]{2,}", _cur_ot_elab)}
                                _ENG_THAI_ELAB = {"PLUS": "พลัส", "SUPER": "ซุปเปอร์", "PRO": "โปร", "MINI": "มินิ"}
                                _new_ot_elab = (
                                    next((o for o in _ot_all_elab if o != _cur_ot_elab and o in raw_stripped), None)
                                    or next(
                                        (o for o in _ot_all_elab
                                         if o != _cur_ot_elab
                                         and any(tok.lower() in raw_stripped.lower()
                                                 for tok in re.findall(r"[A-Za-z]{2,}", o)
                                                 if tok.lower() not in _cur_eng_elab)),
                                        None,
                                    )
                                    or next(
                                        (o for o in _ot_all_elab
                                         if o != _cur_ot_elab
                                         and any(
                                             _ENG_THAI_ELAB.get(tok.upper(), "") in raw_stripped
                                             for tok in re.findall(r"[A-Za-z]{2,}", o)
                                             if tok.upper() not in {t.upper() for t in _cur_eng_elab}
                                         )),
                                        None,
                                    )
                                )
                                if _new_ot_elab:
                                    if hasattr(state, "save_collected_slot"):
                                        state.save_collected_slot("operation_topic", _new_ot_elab)
                                    if state.context is not None:
                                        state.context.setdefault("slots", {})["operation_topic"] = _new_ot_elab
                                        if "collected_slots" in state.context:
                                            state.context["collected_slots"]["operation_topic"] = _new_ot_elab
                                    last_q_fb2 = _new_ot_elab
                                    _LOG.info(
                                        "[Supervisor] elaborate: op_topic switch %r → %r",
                                        _cur_ot_elab, _new_ot_elab,
                                    )
                        except Exception as _e_elab_ot:
                            _LOG.debug("[Supervisor] elaborate: op_topic detect failed: %s", _e_elab_ot)
                _LOG.info("[Supervisor] fallback_llm→elaborate last_q=%r", last_q_fb2[:40])
                if not _slot_chgd_fb_elab:
                    self._ensure_practical_retrieval_for_legal(state, last_q_fb2)
                state.context.pop("pending_slot", None)
                st2, reply = self._practical.handle(state, f"อธิบายรายละเอียดเพิ่มเติม: {last_q_fb2}", _internal=False)
                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "fallback_llm_elaborate"
                return st2, reply

        if fallback_intent in ("legal_question", "business_question"):
            q_fb = (intent_res.get("query") or raw_stripped).strip()
            _LOG.info("[Supervisor] fallback_llm→%s q=%r", fallback_intent, q_fb[:40])
            state.context["last_user_legal_query"] = q_fb
            # Always use raw_stripped for retrieval — LLM reformulation distorts non-regulatory
            # topic queries (e.g. adds "ขั้นตอน" to a bakery concept query), breaking exact
            # main_topic Chroma matches and triggering expensive broad two-pass retrieval.
            # q_fb (reformulated) is stored only in last_user_legal_query for tracking.
            _retrieval_q_fb = raw_stripped
            self._ensure_practical_retrieval_for_legal(state, _retrieval_q_fb)
            self._maybe_build_slot_queue_from_docs(state, raw_stripped)
            if not (self._LINK_REQUEST_RE.search(raw_stripped) or self._link_request_llm_check(raw_stripped, state)) and not (state.context or {}).get("topic_slot_queue"):
                _dc_q_fb = self._maybe_dynamic_clarification(state, _retrieval_q_fb)
                if _dc_q_fb:
                    self._add_assistant(state, _dc_q_fb)
                    state.last_action = "dynamic_clarification_asked"
                    return state, _dc_q_fb
            state.context.pop("pending_slot", None)
            st2, reply = self._practical.handle(state, _retrieval_q_fb, _internal=False)
            reply = self._normalize_male(reply)
            self._add_assistant(st2, reply)
            st2.last_action = "fallback_llm_business"
            return st2, reply

        if fallback_intent == "greeting":
            return self._handle_greeting(state, raw_stripped, show_menu=False)

        # Truly off-topic (unknown intent) → guardrail deflect
        # BUT: if Chroma has a strong hit for this input, the fallback_intent LLM may have
        # failed (LengthFinishReasonError / timeout) even though content clearly exists.
        # In that case treat as business_question so the user gets a real answer and
        # last_user_legal_query is updated (prevents stale query in follow-up "ขอแบบละเอียด").
        _LOG.info("[Supervisor] unknown_intent → deflect input=%r", raw_stripped[:60])
        try:
            _vstore_unk = getattr(self._practical.retriever, "vectorstore", None)
            if _vstore_unk is not None:
                _pairs_unk = _vstore_unk.similarity_search_with_relevance_scores(raw_stripped, k=1)
                if _pairs_unk and _pairs_unk[0][1] >= 0.72:
                    _LOG.info(
                        "[Supervisor] unknown_intent override: Chroma sim=%.3f → treating as business_question: %r",
                        _pairs_unk[0][1], raw_stripped[:60],
                    )
                    state.context["last_user_legal_query"] = raw_stripped
                    self._ensure_practical_retrieval_for_legal(state, raw_stripped)
                    self._maybe_build_slot_queue_from_docs(state, raw_stripped)
                    if not (self._LINK_REQUEST_RE.search(raw_stripped) or self._link_request_llm_check(raw_stripped, state)) and not (state.context or {}).get("topic_slot_queue"):
                        _dc_q_unk = self._maybe_dynamic_clarification(state, raw_stripped)
                        if _dc_q_unk:
                            self._add_assistant(state, _dc_q_unk)
                            state.last_action = "dynamic_clarification_asked"
                            return state, _dc_q_unk
                    state.context.pop("pending_slot", None)
                    st2, reply = self._practical.handle(state, raw_stripped, _internal=False)
                    reply = self._normalize_male(reply)
                    self._add_assistant(st2, reply)
                    st2.last_action = "practical_answer"
                    return st2, reply
        except Exception as _e_unk:
            _LOG.debug("[Supervisor] unknown_intent Chroma override check failed: %s", _e_unk)
        return self._handle_deflect(state, raw_stripped)
    
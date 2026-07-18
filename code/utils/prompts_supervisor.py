"""
Supervisor Prompt Templates
============================
All LLM prompt strings used by SupervisorPersonaService live here.
Each function takes only the dynamic variables it needs and returns a ready-to-send string.

Prompts are grouped by function:
  1. TOPIC_PICKER         — select k relevant menu topics from candidates
  2. CONFIRM              — interpret yes / no / ambiguous replies
  3. STYLE_DETECT         — detect long / short answer preference
  4. GREET_PREFIX         — generate personalized greeting text
  5. OP_GROUP_CLASSIFIER  — group raw operation values into human-readable labels
  6. DEDUPLICATE_OPTIONS  — remove semantically duplicate option entries
  7. SLOT_MAPPER          — map free-text reply to a pending-slot option
  8. FALLBACK_INTENT      — classify intent when no deterministic rule matched
  9. TYPO_CHECK           — detect garbled / accidental input
 10. TOPIC_DESC           — generate one-sentence topic menu descriptions
"""

from __future__ import annotations
from typing import List


def _safe_embed(text: str) -> str:
    """Sanitize user-supplied text before embedding in LLM prompts."""
    return str(text or "").replace('"', "'").replace("\n", " ").strip()


# 1. TOPIC PICKER

def build_topic_picker_prompt(
    last_hint: str,
    k: int,
    banned: List[str],
    candidates: List[str],
) -> str:
    """Select k menu topics from candidates based on relevance and diversity."""
    return (
        "หน้าที่: เลือกหัวข้อเมนูจำนวน k ข้อ จากรายการ candidates\n"
        "เป้าหมาย:\n"
        "1) เกี่ยวข้องกับบริบท last_topic_hint ให้มากที่สุด\n"
        "2) หลากหลาย (อย่าเลือกหัวข้อที่ความหมายซ้ำกัน)\n"
        "3) เป็น 'หัวข้อที่มีประโยชน์ต่อผู้ประกอบการ' — รวมถึงใบอนุญาต/ขั้นตอน/กฎหมาย และการตลาด/กลยุทธ์/การเปิดร้าน ไม่ใช่เคสเฉพาะหรือชื่อหน่วยงานล้วนๆ\n"
        "ข้อห้าม:\n"
        "- ห้ามเลือกคำ generic/placeholder ใน banned\n"
        "- ห้ามเลือกเคสเฉพาะ เช่น 'กรณี...', 'ถ้า...', 'สำหรับ...' (ยกเว้นจำเป็นจริงๆ)\n"
        "- ห้ามเลือกชื่อหน่วยงานล้วนๆ เช่น กรม..., สำนักงาน..., เทศบาล..., อบต., อบจ., สำนักงานเขต (ยกเว้นจำเป็นจริงๆ)\n"
        "- ห้ามเลือกซ้ำ\n"
        "ให้ตอบเป็น JSON เท่านั้น:\n"
        '{ "topics": ["..."], "confidence": 0.0 }\n'
        f"last_topic_hint: {last_hint}\n"
        f"k: {int(k)}\n"
        f"banned: {banned}\n"
        f"candidates: {candidates}\n"
    )


# 2. CONFIRM (yes / no)

def build_confirm_prompt(user_text: str) -> str:
    """Interpret whether user is confirming, rejecting, or ambiguous."""
    return (
        "หน้าที่: ตีความว่า 'ข้อความผู้ใช้' เป็นการยืนยัน (yes) หรือปฏิเสธ (no) หรือยังไม่ชัดเจน\n"
        "ให้ดูโทน/เจตนา ไม่ต้องยึดแค่คำว่า 'ใช่/ไม่'\n"
        "ตัวอย่าง yes: งับ, ได้เลย, โอเค, ถูกต้อง, ยืนยัน, เอาเลย, จัดไป, ไปเลย\n"
        "ตัวอย่าง no: ไม่เอา, ยกเลิก, ช่างมัน, ไม่ต้อง, ยังไม่\n"
        "ถ้ากำกวมจริงๆ ให้ confidence ต่ำ\n"
        "ตอบเป็น JSON เท่านั้น:\n"
        '{ "yes": true/false, "no": true/false, "confidence": 0.0 }\n'
        f"ข้อความผู้ใช้: {_safe_embed(user_text)}"
    )


# 3. STYLE DETECT (long / short)

def build_style_detect_prompt(user_text: str, last_query: str = "") -> str:
    """Detect whether user explicitly wants a long/detailed or short/concise answer."""
    context_line = f"บริบทที่คุยก่อนหน้า: {last_query}\n" if last_query else ""
    return (
        "หน้าที่: ตรวจว่า 'ข้อความผู้ใช้' บอกชัดๆ ว่าต้องการ style คำตอบแบบใด\n\n"
        f"{context_line}"
        "wants_long=true เมื่อผู้ใช้ต้องการคำตอบที่ละเอียด ครอบคลุม หรือเชิงลึก\n"
        "ตัวอย่าง wants_long=true:\n"
        "  - 'ขอละเอียด', 'อย่างละเอียด', 'แบบละเอียด', 'ละเอียดกว่านี้'\n"
        "  - 'อธิบายเชิงลึก', 'แบบวิชาการ', 'แบบครบ', 'ครบทุกประเด็น'\n"
        "  - 'ต้องการกลยุทธ์ด้านราคาอย่างละเอียด' → wants_long=true (มี 'อย่างละเอียด')\n"
        "  - 'บอกให้ครบเลย', 'อยากรู้ทั้งหมด', 'แบบเต็มๆ', 'ขยายความ'\n"
        "  - 'รายละเอียดทั้งหมด', 'ครบถ้วน', 'แบบให้ลึกซึ้ง', 'เจาะลึก'\n"
        "  - 'ขอแบบเต็ม', 'แบบให้ครบ', 'ลงรายละเอียด', 'ให้ครอบคลุม'\n\n"
        "ไม่ใช่ wants_long (คำถามธรรมดา ไม่ได้ระบุ style):\n"
        "  - 'อยากรู้', 'ต้องการทราบ', 'ถามว่า', 'คืออะไร', 'มีอะไรบ้าง'\n"
        "  - 'ค่าธรรมเนียมเท่าไหร่', 'ต้องใช้เอกสารอะไร' (แค่ถามข้อมูล ไม่ได้บอก style)\n"
        "  - 'แล้วถ้าเป็น SAN PLUS', 'แล้วถ้าเป็นใบอนุญาตจัดตั้ง' (ถามหัวข้อใหม่ ไม่ใช่ขอรายละเอียดเพิ่ม)\n"
        "  - 'แล้วถ้าต้องการ X', 'แล้วถ้าจะขอ X' (follow-up คำถามใหม่ ไม่มีคำบอก style)\n\n"
        "wants_short=true เมื่อผู้ใช้ต้องการคำตอบสั้น กระชับ\n"
        "ตัวอย่าง wants_short=true:\n"
        "  - 'แบบสั้น', 'กระชับ', 'สรุปแค่', 'พอสังเขป', 'แบบย่อๆ'\n\n"
        "ถ้าไม่ชัดหรือเป็นแค่การถามเรื่องทั่วไป → confidence ต่ำ (<0.5) และ wants_long=false\n\n"
        "ตอบเป็น JSON เท่านั้น:\n"
        '{ "wants_long": true/false, "wants_short": true/false, "confidence": 0.0 }\n\n'
        f"ข้อความผู้ใช้: {_safe_embed(user_text)}"
    )


# 4. GREET PREFIX

def build_greet_prefix_prompt(
    kind: str,
    persona_id: str,
    last_topic_hint: str,
    include_intro: bool,
    kind_instructions: str,
) -> str:
    """Generate a personalized Thai greeting/response prefix for น้องสุดยอด."""
    return (
        "หน้าที่: เขียนข้อความทักทาย/ตอบรับภาษาไทยแบบมนุษย์ ในฐานะ 'น้องสุดยอด' ที่ปรึกษาร้านอาหาร\n"
        "ข้อกำหนดร่วม:\n"
        "- 1-2 ประโยคสั้นๆ ลงท้ายด้วยคำถาม 1 ข้อ\n"
        "- ห้ามใส่รายการหัวข้อ/เลขข้อ/เมนู\n"
        "- ห้ามสั่ง user ว่า 'เลือก/พิมพ์/กด'\n"
        "- ต้องลงท้ายด้วย 'ครับ'\n"
        "- แทนตัวเองว่า 'ผม' หรือ 'น้องสุดยอด' ห้ามใช้ 'ฉัน'\n"
        "กฎ include_intro:\n"
        "- ถ้า include_intro=true: แนะนำตัวว่าน้องสุดยอดช่วยได้ทั้งเรื่องกฎหมาย/ใบอนุญาต/ภาษี และการตลาด/กลยุทธ์/การเปิดร้าน (เบเกอรี่ คาเฟ่ ร้านอาหาร)\n"
        "- ถ้า include_intro=false: ห้ามพูดชื่อบอทและห้ามบอกหน้าที่บอทซ้ำ\n"
        f"กฎเฉพาะสำหรับ kind='{kind}':\n"
        f"{kind_instructions}"
        '{"prefix": "..."}\n'
        f"kind: {kind}\n"
        f"persona: {persona_id}\n"
        f"include_intro: {str(bool(include_intro)).lower()}\n"
        f"last_topic_hint: {last_topic_hint}\n"
    )


def build_greet_kind_instructions(kind: str, last_topic_hint: str) -> str:
    """Build the kind-specific instruction block for build_greet_prefix_prompt."""
    if kind == "thanks":
        if last_topic_hint:
            return (
                f"- ผู้ใช้พึ่งขอบคุณหลังจากสอบถามเรื่อง '{last_topic_hint}'\n"
                "- ตอบรับคำขอบคุณอย่างอบอุ่น แล้วถามว่ายังมีคำถามเรื่องเดิมหรือเรื่องอื่นอีกไหม\n"
                "- ห้ามพูดแบบ generic เช่น 'มีอะไรให้ช่วยไหม' ให้เชื่อมโยงกับ topic ที่คุยมา\n"
            )
        return "- ผู้ใช้ขอบคุณ ตอบรับอย่างอบอุ่นและถามว่ามีอะไรให้ช่วยอีกไหม\n"
    if kind in ("smalltalk", "blank"):
        if last_topic_hint:
            return (
                f"- ก่อนหน้านี้คุยเรื่อง '{last_topic_hint}' ถ้าผู้ใช้อาจยังสงสัยเรื่องนั้นอยู่ ให้เชิญชวนต่อ\n"
                "- ตอบรับแบบเป็นกันเองสั้นๆ แล้วถามว่ายังมีเรื่องนั้นหรือเรื่องอื่นให้ช่วยไหม\n"
            )
        return "- ตอบรับแบบเป็นกันเองสั้นๆ และถามว่ามีอะไรให้ช่วยไหม\n"
    return "- ทักทายอย่างอบอุ่น\n"



# 5. OP GROUP CLASSIFIER

def build_op_group_classifier_prompt(license_type: str, raw_ops: List[str]) -> str:
    """Group raw operation values from ChromaDB into human-readable categories."""
    ops_str = "\n".join(f"- {o}" for o in raw_ops)
    return (
        "คุณเป็น AI ที่ช่วยจัดกลุ่มประเภทการดำเนินการสำหรับใบอนุญาตธุรกิจ\n"
        "หน้าที่: จัดกลุ่ม raw operation values ด้านล่างให้เป็นหมวดหมู่ตาม prefix ของแต่ละค่า\n"
        "กฎสำคัญ — จัดกลุ่มตาม prefix ตัวอักษรแรกของ raw value เท่านั้น ห้ามตีความความหมาย:\n"
        "1. แต่ละกลุ่มต้องมี label ภาษาไทยที่กระชับ ชัดเจน (ไม่เกิน 30 ตัวอักษร)\n"
        "2. raw op ที่ขึ้นต้นด้วย 'การจดทะเบียน', 'การจด', 'การขอ', 'การยื่น', 'ยื่นใหม่' → label 'ยื่นขอใหม่ / จดทะเบียน'\n"
        "3. raw op ที่ขึ้นต้นด้วย 'ต่ออายุ' → label 'ต่ออายุ'\n"
        "4. raw op ที่ขึ้นต้นด้วย 'แก้ไข' หรือ 'เปลี่ยนแปลง' → label 'แก้ไข / เปลี่ยนแปลงรายการ'\n"
        "5. raw op ที่ขึ้นต้นด้วย 'ยกเลิก' หรือ 'เลิก' → label 'ยกเลิก'\n"
        "6. raw op ที่ขึ้นต้นด้วย 'ย้าย' เท่านั้น → label 'ย้ายสถานประกอบการ'\n"
        "7. raw op ที่ขึ้นต้นด้วย 'เพิ่ม' เท่านั้น → label 'เพิ่มสถานประกอบการ'\n"
        "8. raw op ที่ขึ้นต้นด้วย 'ปิด' (ไม่ใช่ 'ปิดงบ') → label 'ปิดสถานประกอบการ'\n"
        "9. raw op ที่ขึ้นต้นด้วย 'ขอใบแทน', 'ทะเบียนพาณิชย์ชำรุด', 'ทะเบียนสูญหาย' → label 'ขอใบแทน / กรณีสูญหาย'\n"
        "10. raw op ที่ไม่ขึ้นต้นด้วยคำใดข้างต้น → label 'อื่น ๆ'\n"
        "11. ทุก raw value ต้องอยู่ใน group ใด group หนึ่งเสมอ — ห้ามทิ้ง raw value ไว้นอก groups\n"
        "12. ห้ามสร้าง group ที่ไม่มี raw value ใดอยู่เลย\n"
        f"license_type: {license_type}\n"
        f"raw operations:\n{ops_str}\n"
        "Return JSON only:\n"
        '{"groups": [{"label": "...", "raw": ["..."]}, ...]}'
    )



def build_sub_op_group_classifier_prompt(license_type: str, sub_ops: List[str]) -> str:
    """
    Group a long list of raw sub-operation strings (แก้ไขชื่อ/แก้ไขกรรมการ/...) into
    ≤5 user-friendly category labels. Used when operation_sub_type would have >5 options.
    """
    ops_str = "\n".join(f"- {o}" for o in sub_ops)
    return (
        "คุณเป็น AI ที่ช่วยจัดกลุ่มรายการดำเนินการย่อยสำหรับใบอนุญาตธุรกิจ\n"
        "หน้าที่: จัดกลุ่ม raw sub-operation values ด้านล่างให้เป็นหมวดหมู่ที่ user เข้าใจง่าย\n"
        "กฎ:\n"
        "1. สร้างได้ไม่เกิน 5 กลุ่ม\n"
        "2. label ต้องภาษาไทย กระชับ เข้าใจง่าย เช่น 'ชื่อ', 'กรรมการ', 'ที่ตั้งสำนักงาน', 'ทุน/หุ้น'\n"
        "3. ถ้าหลาย raw values เกี่ยวกับสิ่งเดียวกัน (เช่น แก้ไขชื่อ กรณีต่างๆ) ให้รวมเป็นกลุ่มเดียว\n"
        "4. ถ้ามีกลุ่มย่อยเกิน 5 กลุ่ม ให้รวมกลุ่มเล็กที่เหลือเป็น 'อื่นๆ'\n"
        "5. ทุก raw value ต้องอยู่ใน group ใด group หนึ่งเสมอ\n"
        "6. label ไม่เกิน 20 ตัวอักษร\n"
        f"license_type: {license_type}\n"
        f"raw sub-operations:\n{ops_str}\n"
        "Return JSON only:\n"
        '{"groups": [{"label": "...", "raw": ["..."]}, ...]}'
    )


# 6. DEDUPLICATE OPTIONS

def build_deduplicate_options_prompt(options: List[str]) -> str:
    """Remove semantically duplicate entries from a list of slot options."""
    opts_str = "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(options))
    return (
        "คุณเป็น AI ที่ช่วยจัดกลุ่มและกำจัดตัวเลือกที่ซ้ำซ้อน\n"
        "กติกา:\n"
        "1. ถ้าตัวเลือกมีความหมายเดียวกัน ให้เลือกเอาแค่ตัวที่เฉพาะเจาะจงกว่า (เช่น 'บริษัทจำกัด' ดีกว่า 'บริษัท')\n"
        "2. ถ้าตัวเลือกเป็นการรวมกันของหลายประเภท (เช่น '1.ห้างหุ้นส่วนจำกัด 2.ห้างหุ้นส่วนสามัญ') ให้แยกออกมาเป็นรายการเดี่ยว\n"
        "3. ถ้าตัวเลือกเป็นคำกว้างๆ แล้วมีคำเฉพาะเจาะจงกว่า ให้เอาเฉพาะตัวเจาะจง "
        "(เช่น มี 'ห้างหุ้นส่วนจำกัด' และ 'ห้างหุ้นส่วน' ให้เก็บทั้งคู่ถ้าต่างกัน "
        "แต่ถ้าหมายถึงสิ่งเดียวกันเอาตัวเจาะจงกว่า)\n"
        "4. รักษาคำที่มีความหมายแตกต่างกันไว้ทั้งหมด\n\n"
        f"ตัวเลือกที่มี:\n{opts_str}\n\n"
        'Return JSON: {"unique_options": [list ของตัวเลือกที่ไม่ซ้ำกัน เรียงตามความเหมาะสม], '
        '"reasoning": "อธิบายสั้นๆ ว่าทำไมถึงเลือกแบบนี้"}'
    )


# 7. SLOT MAPPER

def build_slot_mapper_prompt(slot_key: str, user_text: str, options: List[str]) -> str:
    """Map a free-text reply to the closest matching pending-slot option."""
    opts = [str(x).strip() for x in options if str(x).strip()][:20]
    # Only add ปริมณฑล hint when one of the available options explicitly says "ปริมณฑล".
    # Some licenses treat these provinces as ต่างจังหวัด — don't override in those cases.
    _has_metro_opt = slot_key == "location" and any("ปริมณฑล" in o for o in opts)
    _location_hint = (
        "หมายเหตุ location: จังหวัดที่อยู่ในปริมณฑล ได้แก่ นนทบุรี, ปทุมธานี, สมุทรปราการ, "
        "นครปฐม, สมุทรสาคร — ถือเป็น 'กรุงเทพฯ และปริมณฑล' ไม่ใช่ 'ต่างจังหวัด'\n"
    ) if _has_metro_opt else ""
    _is_area_slot = slot_key in ("shop_area_type", "area_size")
    _area_hint = (
        "หมายเหตุ area: ถ้า user_text ระบุตัวเลขพื้นที่ (เช่น '8.9 ตรม', '150 ตารางเมตร') "
        "ให้เทียบกับ threshold ที่อยู่ใน options แล้ว map ให้ตรง "
        "(เช่น 8.9 < 200 → 'น้อยกว่า 200 ตารางเมตร') — ใช้ confidence=0.95\n"
    ) if _is_area_slot else ""
    return (
        "หน้าที่: จับคู่ข้อความผู้ใช้ให้เข้ากับตัวเลือกที่ใกล้ที่สุด (เลือกได้ 1 ข้อ)\n"
        "กติกา:\n"
        "- ถ้าแมพได้ชัดเจน ให้คืน choice_index เป็นเลข 1..N และ choice_text เป็นข้อความของตัวเลือกนั้น\n"
        "- ถ้าไม่ชัดเจนจริงๆ ให้คืน choice_index=0 และ confidence ต่ำ\n"
        "- ห้ามเดาแบบสุ่ม\n"
        + _location_hint
        + _area_hint
        + "ตอบเป็น JSON เท่านั้น:\n"
        '{"choice_index": 0, "choice_text": "", "confidence": 0.0}\n'
        f"slot_key: {slot_key}\n"
        f"user_text: {_safe_embed(user_text)}\n"
        f"options: {opts}\n"
    )


# 7b. SELECT-ALL INTENT (multi-select slots only)

def build_select_all_intent_prompt(slot_key: str, user_text: str, options: List[str]) -> str:
    """Detect if user wants to select ALL available options rather than just one."""
    opts = [str(x).strip() for x in options if str(x).strip()][:10]
    return (
        "คุณเป็น classifier: user ต้องการเลือก 'ตัวเลือกทั้งหมด' หรือไม่?\n"
        "กติกา:\n"
        "- ตอบ select_all=true: ถ้า user สื่อว่าต้องการข้อมูลหรือเลือก 'ทุกตัวเลือก' / 'ทั้งหมด' / 'ทั้งคู่' / 'ทั้งนั้น' / 'หมดเลย' ฯลฯ\n"
        "- ตอบ select_all=false: ถ้า user เลือกเพียงบางตัวเลือก หรือพิมพ์ตัวเลขระบุตัวเลือกเดียว\n"
        "ตอบ JSON เท่านั้น:\n"
        '{"select_all": false, "confidence": 0.0}\n'
        f"slot_key: {slot_key}\n"
        f"user_text: {_safe_embed(user_text)}\n"
        f"options: {opts}\n"
    )


# 8. FALLBACK INTENT

def build_fallback_intent_prompt(user_text: str, last_query: str, persona: str) -> str:
    """Classify intent when no deterministic routing rule matched."""
    return (
        "คุณคือ routing classifier สำหรับ AI ที่ปรึกษาธุรกิจร้านอาหารครบวงจร\n"
        "บอทช่วยได้ทั้ง: กฎหมาย/ใบอนุญาต/ภาษี, การตลาด/กลยุทธ์, การเปิดร้าน/บริหารร้าน (เบเกอรี่, คาเฟ่, ร้านอาหาร)\n"
        "จงจำแนก intent จากข้อความผู้ใช้ด้านล่าง\n\n"
        f"user_text: {_safe_embed(user_text)}\n"
        f"last_query: {_safe_embed(last_query) or '(none)'}\n"
        f"current_persona: {persona}\n\n"
        "Intent categories:\n"
        "- new_topic: อยากดูหัวข้ออื่นหรือขอเมนูหัวข้อ (เช่น 'ขอเรื่องอื่น', 'มีหัวข้ออะไรอีก')\n"
        "  ⚠️ new_topic ต้องเป็นคำขอดูเมนู/หัวข้อ ไม่ใช่คำถามที่มีเนื้อหาชัดเจนอยู่แล้ว\n"
        "- elaborate: ขอให้ขยายความ/อธิบายเพิ่มจากคำตอบก่อนหน้า โดยยังอยู่ใน TOPIC เดิมเท่านั้น\n"
        "  ⚠️ กฎสำคัญ: elaborate ใช้ได้เฉพาะเมื่อ topic ของ user_text ตรงกับ last_query เท่านั้น\n"
        "  ⚠️ ถ้า user_text มี topic ใหม่ที่ต่างจาก last_query → ให้ใช้ business_question แทน\n"
        "  ตัวอย่าง elaborate (topic เดิม): last_query='กลยุทธ์ด้านราคา' + user_text='ขยายความราคาล่อใจ' → elaborate\n"
        "  ตัวอย่าง business_question (topic ใหม่): last_query='กลยุทธ์ด้านราคา' + user_text='ขยายความส่วนประสมสินค้า ของกลยุทธ์ด้านผลิตภัณฑ์' → business_question (เปลี่ยนจากราคา→ผลิตภัณฑ์)\n"
        "- business_question: ถามเรื่องใดๆ ที่เกี่ยวกับธุรกิจร้านอาหาร ได้แก่:\n"
        "  • กฎหมาย/ใบอนุญาต/ภาษี/จดทะเบียน\n"
        "  • การตลาด/กลยุทธ์ราคา/การโปรโมต/SOP/การจัดการร้าน\n"
        "  • การเปิดร้าน/เบเกอรี่/คาเฟ่/วัตถุดิบ/อุปกรณ์/ทุน/ทำเล\n"
        "  • พนักงาน/การจัดการต้นทุน/บัญชี\n"
        "- greeting: ทักทาย/ขอบคุณ/ปิดบทสนทนา\n"
        "- unknown: ไม่เกี่ยวกับธุรกิจร้านอาหารเลย เช่น ท่องเที่ยว ชีวิตส่วนตัว บันเทิง การเมือง\n\n"
        "ตอบ JSON เท่านั้น:\n"
        '{"intent": "business_question", "query": "", "confidence": 0.9}\n'
        "- query: ถ้า intent=business_question ให้ใส่คำถามที่ชัดเจนขึ้น, ไม่งั้นเว้นว่าง"
    )


# 9. ENTITY TYPE DETECT

def build_entity_type_detect_prompt(user_text: str, last_query: str) -> str:
    """
    Classify whether the user is referring to นิติบุคคล or บุคคลธรรมดา.
    Called only when regex + fuzzy both miss — e.g. short/abbreviated phrasing like "แบบนิติ".
    """
    return (
        "คุณคือ classifier สำหรับระบุประเภทผู้ประกอบการไทย\n"
        f"user_text: {user_text}\n"
        f"last_query: {last_query or '(none)'}\n\n"
        "นิติบุคคล = บริษัทจำกัด, บริษัทมหาชน, ห้างหุ้นส่วน, นิติบุคคล, นิติ, บจก, หจก\n"
        "บุคคลธรรมดา = เจ้าของคนเดียว, บุคคลธรรมดา, กิจการส่วนตัว, ร้านส่วนตัว, ทำคนเดียว\n\n"
        "กฎสำคัญ: ระบุ entity_type จาก user_text เท่านั้น\n"
        "last_query ใช้เฉพาะเมื่อ user_text มีคำย่อ/คำพูดที่ต้องการ context (เช่น 'แบบนิติ', 'บจก')\n"
        "ถ้า user_text ไม่มี entity signal → return null เสมอ ไม่ว่า last_query จะมีคำอะไร\n"
        "คำสรรพนาม (ฉัน/ผม/หนู/เรา/คุณ) ไม่ใช่ entity signal — ห้ามระบุ entity_type จากสรรพนาม\n"
        "entity signal ต้องเป็นคำที่ระบุโครงสร้างธุรกิจ เช่น บจก, หจก, นิติบุคคล, บริษัท, เจ้าของคนเดียว, บุคคลธรรมดา\n\n"
        'ตอบ JSON เท่านั้น: {"entity_type": "นิติบุคคล"|"บุคคลธรรมดา"|null, "confidence": 0.0}\n'
        "null = user ไม่ได้ระบุประเภทหรือไม่แน่ใจเลย (confidence < 0.70)"
    )


# 10. LOCATION DETECT

def build_location_detect_prompt(user_text: str, last_query: str) -> str:
    """Classify whether user's business is in Bangkok or province — LLM fallback when regex misses."""
    return (
        "ระบุสถานที่ตั้งร้าน/กิจการที่ user กล่าวถึง\n"
        f"user_text: {user_text}\n"
        f"last_query: {last_query or '(none)'}\n\n"
        "กรุงเทพฯ = ในกรุงเทพมหานคร, กทม, แถวสาทร/สุขุมวิท/ลาดพร้าว/บางนา ฯลฯ\n"
        "ต่างจังหวัด = จังหวัดอื่น ๆ เช่น เชียงใหม่, ขอนแก่น, ภูเก็ต, นครราชสีมา\n\n"
        'ตอบ JSON เท่านั้น: {"location": "กรุงเทพฯ"|"ต่างจังหวัด"|null, "confidence": 0.0}\n'
        "null = user ไม่ได้ระบุสถานที่หรือไม่แน่ใจ (confidence < 0.70)"
    )


# 11. OPERATION TYPE DETECT

def build_operation_type_detect_prompt(user_text: str, last_query: str) -> str:
    """Classify which operation type user wants — LLM fallback when regex misses."""
    return (
        "ระบุ operation ที่ user ต้องการทำกับใบอนุญาต/การจดทะเบียน\n"
        f"user_text: {user_text}\n"
        f"last_query: {last_query or '(none)'}\n\n"
        "new    = จดทะเบียนใหม่, ขอใบอนุญาตใหม่, เปิดกิจการ, เริ่มทำ\n"
        "edit   = แก้ไขข้อมูล, เปลี่ยนชื่อ/ที่อยู่, อัปเดตรายการ\n"
        "cancel = ยกเลิก, ปิดกิจการ, เลิกทำ, ถอนใบอนุญาต\n"
        "renew  = ต่ออายุ, ใบหมดอายุ, ขอต่อ, รีนิว\n\n"
        'ตอบ JSON เท่านั้น: {"operation": "new"|"edit"|"cancel"|"renew"|null, "confidence": 0.0}\n'
        "null = ไม่ชัดเจน หรือถามทั่วไป ไม่ได้ระบุ operation (confidence < 0.70)"
    )


# 12. AREA SIZE DETECT

def build_area_size_detect_prompt(user_text: str) -> str:
    """Classify shop area size — LLM fallback when regex + numeric parsing miss."""
    return (
        "ระบุขนาดพื้นที่ร้านของ user (เกณฑ์: 200 ตารางเมตร)\n"
        f"user_text: {user_text}\n\n"
        "น้อยกว่า 200 = ร้านเล็ก, ห้องแถว 1-2 คูหา, พื้นที่น้อย, tiny/small\n"
        "มากกว่า 200 = ร้านใหญ่, หลายคูหา, พื้นที่กว้าง, large\n\n"
        'ตอบ JSON เท่านั้น: {"area_size": "น้อยกว่า 200 ตารางเมตร"|"มากกว่า 200 ตารางเมตร"|null, "confidence": 0.0}\n'
        "null = user ไม่ได้ระบุขนาดพื้นที่ (confidence < 0.70)"
    )


# 13. REGISTRATION TYPE DETECT

def build_registration_type_detect_prompt(user_text: str, options: List[str]) -> str:
    """
    LLM fallback for registration_type inference — called when user implies a sub-type
    but doesn't use the exact option label (e.g. typo, partial name, or synonym).
    Returns the matched option string (may be a partial like "ห้างหุ้นส่วน"), or null.
    """
    opts_block = "\n".join(f"- {o}" for o in (options or []))
    return (
        "ระบุรูปแบบการจดทะเบียนที่ user กล่าวถึงจากตัวเลือกที่ให้\n"
        f"user_text: {user_text}\n\n"
        f"ตัวเลือกที่มีในระบบ:\n{opts_block}\n\n"
        "กติกา:\n"
        "- ถ้า user ระบุชัดเจนว่าเป็นประเภทใดในรายการ → ตอบตัวเลือกนั้นทั้งข้อความ\n"
        "- ถ้า user บอกแค่กลุ่มกว้างๆ เช่น 'ห้างหุ้นส่วน' (ไม่ระบุจำกัด/สามัญ) → ตอบ 'ห้างหุ้นส่วน'\n"
        "- ถ้า user บอกแค่ 'บริษัท' (ไม่ระบุรายละเอียด) → ตอบ 'บริษัทจำกัด'\n"
        "- ถ้า user ไม่ได้กล่าวถึงรูปแบบเลย หรือไม่ชัดเจน → ตอบ null\n"
        "- confidence < 0.70 → ตอบ null\n\n"
        'ตอบ JSON เท่านั้น: {"registration_type": "..."|null, "confidence": 0.0}'
    )


# 14. LICENSE TYPE DETECT

def build_license_type_detect_prompt(user_text: str, candidates: List[str]) -> str:
    """
    Classify which license types (if any) the user is asking about.
    Called only when regex keyword scan returns no match — catches variant phrasing,
    abbreviations, or conceptual references that don't hit any keyword pattern.
    """
    cand_block = "\n".join(f"- {c}" for c in (candidates or []))
    return (
        "คุณคือ classifier ระบุประเภทใบอนุญาต/การจดทะเบียนที่ user กล่าวถึง\n"
        f"user_text: {user_text}\n\n"
        "รายการประเภทที่มีในระบบ:\n"
        f"{cand_block}\n\n"
        "กติกา:\n"
        "- ถ้า user_text ระบุชัดเจนว่าเกี่ยวกับประเภทใดในรายการ → ใส่ลงใน license_types\n"
        "- อนุญาตให้เลือกได้มากกว่า 1 ประเภทถ้า user ถามหลายเรื่องพร้อมกัน\n"
        "- ถ้า user ถามกว้างมาก (เช่น 'ต้องขออะไรบ้าง') โดยไม่ระบุประเภท → ส่ง license_types=[]\n"
        "- confidence < 0.70 → ส่ง license_types=[]\n\n"
        'ตอบ JSON เท่านั้น: {"license_types": ["..."], "confidence": 0.0}\n'
        "license_types = [] หมายถึงไม่สามารถระบุได้"
    )


# 14. ACADEMIC RESUME DETECT

def build_academic_resume_detect_prompt(
    user_text: str, last_academic_query: str, section_count: int
) -> str:
    """
    Detect whether user wants to resume/continue their current Academic session
    (e.g. see remaining sections, go deeper on current topic) vs. asking something new.
    Called when _ACADEMIC_RESUME_RE regex misses — catches natural/indirect phrasing.
    """
    sec_hint = f"จำนวน section ที่ยังเหลือในหัวข้อปัจจุบัน: {section_count}" if section_count > 0 else "ยังไม่มี section ที่เลือกไว้"
    return (
        "คุณคือ classifier สำหรับระบุว่า user ต้องการ 'ดำเนินการต่อใน Academic session เดิม' หรือไม่\n"
        f"user_text: {user_text}\n"
        f"หัวข้อ Academic ปัจจุบัน: {last_academic_query or '(ไม่มี)'}\n"
        f"{sec_hint}\n\n"
        "is_resume=true เมื่อ user ต้องการ:\n"
        "- ดู section อื่น/ส่วนที่เหลือของหัวข้อเดิม (เช่น 'ขอดูส่วนอื่น', 'ส่วนที่ยังไม่ได้อ่าน')\n"
        "- กลับไปยังหัวข้อ Academic เดิม (เช่น 'ขอต่อจากที่แล้ว', 'กลับไปเรื่องนั้น')\n"
        "- รายละเอียดเพิ่มเติมของหัวข้อเดิม (เช่น 'อธิบายต่อ', 'บอกเพิ่มเรื่องนี้')\n"
        "- ยืนยันต่อเนื่อง (เช่น 'ดูต่อเลย', 'ส่วนต่อไปเลย')\n\n"
        "is_resume=false เมื่อ user:\n"
        "- ถามเรื่องใหม่ที่ไม่เกี่ยวกับหัวข้อเดิม\n"
        "- ขอเมนูหัวข้อ/เรื่องอื่น\n"
        "- ทักทาย/ขอบคุณ\n\n"
        'ตอบ JSON เท่านั้น: {"is_resume": true/false, "confidence": 0.0}\n'
        "confidence < 0.70 → is_resume=false"
    )


# 15. TYPO CHECK

def build_typo_check_prompt(user_text: str, last_topic: str) -> str:
    """Detect whether input is garbled/accidental or has genuine intent."""
    return (
        "คุณคือ typo-detector สำหรับ AI ผู้ช่วยธุรกิจร้านอาหารไทย\n"
        "วิเคราะห์ว่า user_text ด้านล่างเป็น 'การพิมพ์ผิด/ตัวอักษรสุ่ม/ไม่มีความหมาย' "
        "หรือเป็น 'ข้อความที่มีเจตนาชัดเจน'\n\n"
        f"user_text: {user_text!r}\n"
        f"บริบทล่าสุด: {last_topic or '(ยังไม่มี)'}\n\n"
        "เกณฑ์ is_typo=true:\n"
        "- อักขระสุ่มที่ไม่ก่อเป็นคำหรือประโยคได้\n"
        "- อักษรผสมกันไม่ได้ตามหลักภาษาไทย/อังกฤษ\n"
        "- ดูเหมือนกดแป้นพิมพ์โดยไม่ตั้งใจ\n"
        "เกณฑ์ is_typo=false:\n"
        "- มีคำ/ประโยคที่อ่านออกความหมายได้ แม้จะสั้น\n"
        "- เป็นชื่อ, ตัวเลข, หรือคำย่อที่ใช้บ่อย\n\n"
        "ตอบ JSON เท่านั้น:\n"
        '{"is_typo": true, "confidence": 0.92, "suggested": ""}\n'
        "- suggested: ถ้า is_typo=true แต่พอเดาได้ว่าหมายถึงอะไร ใส่คำนั้น ไม่งั้นเว้นว่าง"
    )

# 10. TOPIC DESC

def build_elaborate_detect_prompt(user_text: str, last_topic: str) -> str:
    """
    Detect whether user wants MORE detail on the CURRENT topic — not a new question.
    Called when _ELABORATE_RE regex misses; catches indirect phrasing like
    "ต้องการข้อมูลครบกว่านี้", "ช่วยขยายความ", "บอกให้ละเอียดขึ้น".
    Confidence threshold: 0.75 (higher than resume check — elaboration is unambiguous when correct).
    """
    return (
        "คุณคือ classifier ตรวจว่า user ขอให้ 'อธิบาย/บอก/ขยายความหัวข้อปัจจุบันเพิ่มเติม' หรือไม่\n"
        f"user_text: {user_text}\n"
        f"หัวข้อที่กำลังคุยอยู่: {last_topic or '(ยังไม่มี)'}\n\n"
        "is_elaborate=true เมื่อ user ต้องการ:\n"
        "- ให้อธิบาย/บอก/ขยายหัวข้อเดิมมากขึ้น (เช่น 'บอกให้ครบ', 'ต้องการรู้ให้ละเอียดขึ้น', 'ช่วยขยายความ')\n"
        "- รายละเอียดที่ครบ/ลึก/ชัดเจนกว่าที่ได้รับ\n"
        "- ยกตัวอย่างเพิ่มเกี่ยวกับเรื่องเดิม\n\n"
        "is_elaborate=false เมื่อ user:\n"
        "- ถามเรื่องใหม่หรือ topic อื่น\n"
        "- topic เปลี่ยนด้าน เช่น last_topic='กฎหมาย/ใบอนุญาต/เปิดร้าน' แต่ user_text='ทำให้ดัง/มีกระแส/การตลาด' → false เสมอ\n"
        "- ทักทาย / ขอบคุณ / ขอเมนูหัวข้อ\n"
        "- คำถามที่ไม่เกี่ยวกับ last_topic\n\n"
        'ตอบ JSON เท่านั้น: {"is_elaborate": true/false, "confidence": 0.0}\n'
        "confidence < 0.75 → is_elaborate=false"
    )


def build_meta_request_detect_prompt(user_text: str, last_topic: str) -> str:
    """
    Classify user input as meta-request and detect whether a new topic is embedded.
    Returns 3 fields in one call to avoid the need for a separate word-strip heuristic.
    Confidence threshold: 0.75.
    """
    _u = _safe_embed(user_text)
    _t = _safe_embed(last_topic)
    return (
        "คุณคือ classifier ตรวจ 3 อย่างพร้อมกัน:\n"
        f"user_text: {_u}\n"
        f"หัวข้อที่กำลังคุยอยู่: {_t or '(ยังไม่มี)'}\n\n"
        "1) is_meta_request — user ขอรายละเอียดเพิ่มของหัวข้อเดิมหรือไม่\n"
        "   TRUE: 'อธิบายละเอียดกว่านี้', 'ขอแบบครบกว่านี้', 'ขยายความ', 'เพิ่มเติมหน่อย'\n"
        "   FALSE: ถามเรื่องใหม่ที่ไม่เกี่ยวกับหัวข้อเดิม หรือทักทาย/ขอบคุณ\n\n"
        "2) has_embedded_topic — ถ้า is_meta_request=true มีหัวข้อใหม่ฝังอยู่ใน user_text ด้วยหรือไม่\n"
        "   TRUE: 'ขอรายละเอียดเพิ่มเรื่องใบอนุญาตจำหน่ายสุรา' → มีหัวข้อ 'ใบอนุญาตจำหน่ายสุรา' ฝังอยู่\n"
        "   FALSE: 'อธิบายละเอียดกว่านี้หน่อยคะ' → ไม่มีหัวข้อ — ขอรายละเอียดเรื่องเดิม\n\n"
        "3) extracted_topic — ถ้า has_embedded_topic=true ระบุหัวข้อที่แท้จริงจาก user_text (null ถ้าไม่มี)\n"
        "   เช่น 'ขอรายละเอียดค่าธรรมเนียมการจดทะเบียน' → 'ค่าธรรมเนียมการจดทะเบียน'\n\n"
        'ตอบ JSON เท่านั้น: {"is_meta_request": true/false, "confidence": 0.0, '
        '"has_embedded_topic": false, "extracted_topic": null}\n'
        "ถ้า confidence < 0.75 → is_meta_request=false, has_embedded_topic=false"
    )


def build_broad_question_detect_prompt(user_text: str) -> str:
    """
    Detect whether user is asking a BROAD overview question (multiple license types / topic areas)
    vs. a specific single-topic question.
    Called when _BROAD_Q_RE regex misses; catches variant phrasing like
    "อยากทราบว่าต้องจัดการอะไรบ้าง", "ต้องดูแลเรื่องอะไรบ้างสำหรับร้านอาหาร".
    Confidence threshold: 0.75.
    """
    return (
        "คุณคือ classifier ตรวจว่า user ถาม 'ภาพรวม' (ต้องการรายการหลายอย่างพร้อมกัน) หรือ 'เฉพาะเจาะจง' (ถามเรื่องเดียว)\n"
        f"user_text: {user_text}\n\n"
        "is_broad=true — user ถามแบบภาพรวม เช่น:\n"
        "- 'เปิดร้านอาหารต้องทำอะไรบ้าง' → ครอบคลุมหลายด้าน (ใบอนุญาต+ภาษี+จดทะเบียน)\n"
        "- 'ต้องจ่ายภาษีอะไรบ้าง' → ขอรายการ ไม่ใช่ถามภาษีใดภาษีหนึ่ง\n"
        "- 'ต้องขอใบอนุญาตอะไรบ้าง' → ขอรายการ\n"
        "- 'ต้องจดทะเบียนอะไรบ้าง' / 'ต้องดูแลอะไรบ้าง'\n\n"
        "is_broad=false — user ถามเฉพาะเจาะจง เช่น:\n"
        "- 'VAT ต้องจดยังไง' → ถาม VAT อย่างเดียว\n"
        "- 'ใบอนุญาตจัดตั้งต้องใช้เอกสารอะไร' → ถามใบอนุญาตเดียว\n"
        "- 'ขั้นตอนจดทะเบียนพาณิชย์' → เรื่องเดียว\n\n"
        'ตอบ JSON เท่านั้น: {"is_broad": true/false, "confidence": 0.0}\n'
        "confidence < 0.75 → is_broad=false"
    )


def build_new_topic_detect_prompt(user_text: str, last_topic: str) -> str:
    """
    Detect whether user wants to switch to a DIFFERENT topic / see a new topic menu,
    as opposed to asking a follow-up on the current topic.
    Called when _NEW_TOPIC_RE regex misses. Confidence threshold: 0.75.
    """
    return (
        "คุณคือ classifier ตรวจว่า user ต้องการ 'เปลี่ยนหัวข้อ / ดูรายการหัวข้อใหม่' หรือไม่\n"
        f"user_text: {user_text}\n"
        f"หัวข้อที่คุยอยู่ตอนนี้: {last_topic or '(ยังไม่มี)'}\n\n"
        "is_new_topic=TRUE — user ขอดูรายการหัวข้อหรือเปลี่ยนเรื่องอย่างชัดเจน:\n"
        "- 'มีเรื่องอื่นให้แนะนำมั้ย', 'อยากรู้เรื่องอื่นด้วย', 'มีอะไรอีกไหม'\n"
        "- 'ขอดูหัวข้ออื่น', 'ขอเปลี่ยนเรื่องได้มั้ย', 'สนใจเรื่องอื่นด้วย'\n\n"
        "is_new_topic=FALSE — user ถามต่อหรือถามคำถามกฎหมายโดยตรง:\n"
        "- 'แล้วถ้าเป็นบุคคลธรรมดาต้องทำไร' → FALSE (follow-up มุมอื่นของหัวข้อเดิม)\n"
        "- 'ถ้าจะขอใบอนุญาตต้องทำยังไง' → FALSE (คำถามกฎหมายโดยตรง ไม่ใช่ขอเมนู)\n"
        "- 'มีข้อมูลเพิ่มเติมมั้ย', 'บอกอีกหน่อย' → FALSE (ขอข้อมูลเพิ่มหัวข้อเดิม)\n"
        "- ทักทาย / ขอบคุณ → FALSE\n\n"
        "กฎสำคัญ: ถ้า user_text มีคำถามกฎหมายชัดเจน (ใบอนุญาต/จดทะเบียน/ภาษี/ขั้นตอน) → FALSE เสมอ\n\n"
        'ตอบ JSON เท่านั้น: {"is_new_topic": true/false, "confidence": 0.0}\n'
        "confidence < 0.75 → is_new_topic=false"
    )


def build_mode_status_detect_prompt(user_text: str) -> str:
    """
    Detect whether user is asking about the bot's CURRENT mode/persona status.
    Called when _MODE_STATUS_Q regex misses despite a mode keyword being present.
    Catches phrasing like "บอทใช้โหมดอะไร", "ตอนนี้เราอยู่โหมดไหน", "บอทเป็นแบบไหนอยู่".
    Confidence threshold: 0.75.
    """
    return (
        "คุณคือ classifier ตรวจว่า user ถามถึง 'โหมด/บุคลิกปัจจุบันของบอท' หรือไม่\n"
        f"user_text: {user_text}\n\n"
        "is_mode_status=true เมื่อ user ถามว่าตอนนี้บอทอยู่ในโหมดอะไร เช่น:\n"
        "- 'บอทใช้โหมดอะไรอยู่', 'ตอนนี้เราอยู่โหมดไหน', 'บอทเป็นแบบไหนอยู่'\n"
        "- 'โหมดปัจจุบันคืออะไร', 'ตอนนี้เป็น persona ไหน'\n\n"
        "is_mode_status=false เมื่อ user:\n"
        "- ต้องการเปลี่ยนโหมด ('ขอเปลี่ยนโหมด', 'สลับไปโหมดอื่น')\n"
        "- ถามเรื่องกฎหมาย/ขั้นตอนที่บังเอิญมีคำว่าโหมด\n\n"
        'ตอบ JSON เท่านั้น: {"is_mode_status": true/false, "confidence": 0.0}\n'
        "confidence < 0.75 → is_mode_status=false"
    )


def build_greeting_detect_prompt(user_text: str) -> str:
    """
    Detect whether user is greeting or making a polite intro — NOT thanks, NOT legal.
    Called from _looks_like_greeting_or_thanks when all regex patterns miss.
    Catches phrasing like "ยินดีที่ได้รู้จักครับ", "ขอโทษที่รบกวนนะ", "มาถามหน่อยนะครับ".
    Confidence threshold: 0.80 (high — false positives route legal Q to greeting handler).
    """
    return (
        "คุณคือ classifier ตรวจว่า user กำลัง 'ทักทาย/แนะนำตัว/ขอโทษ' หรือไม่\n"
        f"user_text: {user_text}\n\n"
        "is_greeting=true เมื่อ user:\n"
        "- ทักทายหรือแนะนำตัว (เช่น 'ยินดีที่ได้รู้จัก', 'มาใหม่นะครับ', 'เพิ่งเริ่มใช้')\n"
        "- ขอโทษหรือขอรบกวน (เช่น 'ขอโทษที่รบกวน', 'ขอรบกวนหน่อยนะ')\n"
        "- บอกว่าพร้อมจะเริ่ม (เช่น 'มาเริ่มต้นกันเลย', 'เริ่มกันได้เลยนะ')\n"
        "- ทักทายภาษาอังกฤษ รวมถึงที่พิมพ์ยาวหรือซ้ำตัวอักษร (เช่น 'heyyyy', 'hiiii', 'heyyyyyyyd', 'yooooo', 'hiii')\n\n"
        "is_greeting=false เมื่อ user:\n"
        "- ถามคำถามที่มีเนื้อหาจริงๆ (แม้ขึ้นต้นด้วยคำสุภาพ)\n"
        "- ต้องการข้อมูลหรือขั้นตอน\n\n"
        'ตอบ JSON เท่านั้น: {"is_greeting": true/false, "confidence": 0.0}\n'
        "confidence < 0.80 → is_greeting=false"
    )


def build_topic_group_detect_prompt(user_text: str, known_groups: List[str]) -> str:
    """
    When retrieval-based topic_group scoring is inconclusive (< 30%), ask LLM to pick the
    best matching topic group from the known list.
    Returns: {"topic_group": str | null, "confidence": 0.0-1.0}
    Threshold: 0.60 (borderline case — some signal from regex already; LLM adds directional confidence)
    """
    opts = "\n".join(f"- {g}" for g in known_groups if g and g != "อื่นๆ")
    return (
        "คุณคือ classifier หมวดหมู่หัวข้อสำหรับบอทกฎหมายร้านอาหารไทย\n"
        "งาน: จัดคำถามนี้เข้าหมวดที่ตรงที่สุดจากรายการด้านล่าง\n\n"
        f"คำถาม: {user_text}\n\n"
        "หมวดที่มี:\n"
        f"{opts}\n\n"
        "ตอบ JSON เท่านั้น:\n"
        '{"topic_group": "<ชื่อหมวด หรือ null ถ้าไม่ตรงเลย>", "confidence": 0.0}\n'
        "confidence < 0.60 → topic_group=null"
    )


def build_legal_q_detect_prompt(user_text: str) -> str:
    """
    Detect if text is a legal/regulatory question about Thai restaurant business.
    Called when _LEGAL_SIGNAL_RE and _QUESTION_MARKERS_RE both miss — catches unusual phrasing.
    Examples that miss regex: "ขอข้อมูลค่าใช้จ่าย", "แนะนำให้ทำให้ถูกกฎหมายหน่อย", "ต้องติดต่อที่ไหน"
    Returns: {"is_legal": bool, "confidence": 0.0-1.0}
    Threshold: 0.75
    """
    return (
        "คุณคือระบบจำแนกคำถามของผู้ใช้บอทกฎหมายร้านอาหารไทย\n"
        "งาน: ตัดสินว่าข้อความนี้เป็นคำถาม/คำขอเกี่ยวกับกฎหมาย กฎระเบียบ หรือการดำเนินธุรกิจร้านอาหารไทยหรือไม่\n\n"
        "is_legal=true: ถามเรื่องใบอนุญาต ทะเบียน ภาษี ค่าใช้จ่าย เอกสาร ขั้นตอน บทลงโทษ หน่วยงาน ที่อยู่/เบอร์ติดต่อ วิธีการปฏิบัติตามกฎ\n"
        "ตัวอย่าง is_legal=true:\n"
        '- "ขอข้อมูลค่าใช้จ่ายในการเปิดร้าน"\n'
        '- "แนะนำให้ทำให้ถูกกฎหมายหน่อยครับ"\n'
        '- "ต้องติดต่อที่ไหนครับ"\n'
        '- "มีเอกสารอะไรบ้างที่ต้องเตรียม"\n\n'
        "is_legal=false: ทักทาย พูดคุยทั่วไป ไม่เกี่ยวกับกฎหมาย/ธุรกิจ\n"
        "ตัวอย่าง is_legal=false:\n"
        '- "สวัสดีครับ"\n'
        '- "วันนี้อากาศดีมาก"\n\n'
        f'ข้อความผู้ใช้: "{user_text}"\n\n'
        'ตอบเป็น JSON เท่านั้น: {"is_legal": true, "confidence": 0.0}'
    )


def build_info_action_q_detect_prompt(user_text: str) -> str:
    """
    Classify user query as informational (wants to know/understand) vs. action-oriented (wants to register/apply/do).
    Called when _INFO_Q_RE and _ACTION_Q_RE both miss — catches phrasing variants regex can't cover.
    is_action overrides is_info when both true (user wants to act, not just know).
    Returns: {"is_info": bool, "is_action": bool, "confidence": 0.0-1.0}
    Threshold: 0.75
    """
    return (
        "คุณคือระบบจำแนกความตั้งใจของผู้ใช้บอทกฎหมายร้านอาหารไทย\n"
        "งาน: จำแนกว่าข้อความนี้เป็นคำถามเชิงข้อมูล (is_info) หรือต้องการดำเนินการ (is_action)\n\n"
        "is_info=true: ผู้ใช้ต้องการรับทราบข้อมูล — รวมถึงกรณีที่พิมพ์แค่ชื่อหัวข้อ/คำนามโดดๆ\n"
        "กฎสำคัญ: ถ้าข้อความเป็นแค่ชื่อหัวข้อหรือคำนามเดี่ยวๆ (ไม่มีกริยาหรือคำถาม) → is_info=true เสมอ\n"
        "ผู้ใช้กำลังบอกหัวข้อที่สนใจ ต้องการทราบข้อมูลทั่วไปเกี่ยวกับหัวข้อนั้น\n"
        "ตัวอย่าง is_info=true:\n"
        '- "ภาษีป้าย" (ชื่อหัวข้อเดี่ยว — ต้องการทราบข้อมูลทั่วไป)\n'
        '- "ทะเบียนพาณิชย์" (ชื่อหัวข้อเดี่ยว)\n'
        '- "ใบอนุญาตสุรา" (ชื่อหัวข้อเดี่ยว)\n'
        '- "ค่าธรรมเนียมเท่าไหร่ครับ"\n'
        '- "ใช้เวลากี่วันครับ"\n'
        '- "ต้องใช้เอกสารอะไรบ้าง"\n'
        '- "บทลงโทษคืออะไร"\n'
        '- "ประเภทไหนที่ไม่ต้องขออนุญาต"\n\n'
        "is_action=true: ผู้ใช้ต้องการดำเนินการจริง — จด สมัคร ยื่น ขอ เปิด แก้ไข ยกเลิก\n"
        "is_action override is_info — ถ้า action ชัดเจน ให้ is_action=true แม้จะมี info signal\n"
        "ตัวอย่าง is_action=true:\n"
        '- "อยากเปิดร้านอาหารต้องทำอะไรบ้าง"\n'
        '- "ต้องการไปยื่นเอกสารที่ไหน"\n'
        '- "จะขอใบอนุญาตต้องเริ่มต้นยังไง"\n'
        '- "อยากแก้ไขชื่อในทะเบียน"\n\n'
        f'ข้อความผู้ใช้: "{user_text}"\n\n'
        'ตอบเป็น JSON เท่านั้น: {"is_info": true, "is_action": false, "confidence": 0.0}'
    )


def build_smalltalk_detect_prompt(user_text: str) -> str:
    """
    Detect whether user is making small talk / asking personal off-topic questions to the bot
    that are NOT related to restaurant business/legal topics.
    Called when _SMALLTALK_RE regex misses — catches phrasing like
    "เป็นไงบ้างบอท", "คิดว่ายังไง", "ว่างมั้ย", "ชอบอาหารอะไร".
    Confidence threshold: 0.75.
    """
    return (
        "คุณคือ classifier ตรวจว่า user กำลัง 'คุยเรื่องทั่วไปกับบอท' หรือไม่\n"
        "บริบท: บอทนี้ช่วยเรื่องกฎหมาย/ขั้นตอนธุรกิจร้านอาหาร\n"
        f"user_text: {user_text}\n\n"
        "is_smalltalk=true เมื่อ user:\n"
        "- ถามเรื่องส่วนตัวของบอท (เช่น 'เป็นไงบ้าง', 'ว่างมั้ย', 'กินข้าวยังบอท')\n"
        "- คุยเรื่องที่ไม่เกี่ยวกับธุรกิจร้านอาหาร (เช่น 'คิดว่าหุ้นจะขึ้นไหม', 'บอทชอบอะไร')\n"
        "- แสดงอารมณ์โต้ตอบ (เช่น 'โอเคนะบอท', 'ดีเลยนะ', 'เก่งจัง')\n\n"
        "is_smalltalk=false เมื่อ user:\n"
        "- ถามเรื่องกฎหมาย/ใบอนุญาต/ภาษี/ขั้นตอนร้านอาหาร\n"
        "- ทักทาย (สวัสดี) — ทักทายไม่ใช่ smalltalk\n"
        "- ขอบคุณ\n\n"
        'ตอบ JSON เท่านั้น: {"is_smalltalk": true/false, "confidence": 0.0}\n'
        "confidence < 0.75 → is_smalltalk=false"
    )


def build_switch_without_target_detect_prompt(user_text: str) -> str:
    """
    Detect whether user wants to change the bot's response style/mode WITHOUT
    explicitly naming 'academic' or 'practical'.
    Called when _SWITCH_VERBS/_SWITCH_MARKERS regex misses — catches phrasing like
    "ขอแบบอื่น", "เปลี่ยนวิธีตอบหน่อย", "ตอบแบบอื่นได้มั้ย".
    Confidence threshold: 0.80 (high — false positive = unwanted persona switch).
    """
    return (
        "คุณคือ classifier ตรวจว่า user ต้องการให้บอท 'เปลี่ยนรูปแบบการตอบ' หรือไม่\n"
        "โดยไม่ได้ระบุว่าเป็น academic หรือ practical\n"
        f"user_text: {user_text}\n\n"
        "is_switch=true เมื่อ user ต้องการ:\n"
        "- เปลี่ยนโหมดโดยไม่บอกชื่อโหมด (เช่น 'ขอแบบอื่น', 'เปลี่ยนวิธีตอบ', 'ตอบแบบอื่นได้มั้ย')\n"
        "- ขอสลับรูปแบบการตอบ (เช่น 'ขอโหมดอื่น', 'ลองตอบแบบอื่นดู')\n\n"
        "is_switch=false เมื่อ user:\n"
        "- ระบุชื่อโหมดชัดเจน (เช่น 'ขอแบบวิชาการ', 'ขอแบบสั้น') — นั่นคือ explicit switch\n"
        "- ขอรายละเอียดเพิ่มของเรื่องเดิม (เช่น 'บอกเพิ่มเติม', 'ขยายความ')\n"
        "- ถามคำถามใหม่\n\n"
        'ตอบ JSON เท่านั้น: {"is_switch": true/false, "confidence": 0.0}\n'
        "confidence < 0.80 → is_switch=false"
    )


def build_link_request_detect_prompt(user_text: str) -> str:
    """
    Detect whether user is requesting a link, URL, form, or downloadable document.
    Called when _LINK_REQUEST_RE regex misses — catches phrasing like
    "ส่งแบบฟอร์มหน่อย", "ขอ URL", "ดาวน์โหลดคู่มือได้ที่ไหน", "มีลิ้งค์ไหม".
    Confidence threshold: 0.75.
    """
    return (
        "คุณคือ classifier ตรวจว่า user ขอ 'ลิงก์/URL/แบบฟอร์ม/เอกสารดาวน์โหลด' หรือไม่\n"
        f"user_text: {user_text}\n\n"
        "is_link_request=true เมื่อ user ต้องการ:\n"
        "- ลิงก์หรือ URL (เช่น 'ขอลิงก์', 'URL อยู่ที่ไหน', 'มีลิ้งค์ไหม', 'ส่งลิงก์ให้หน่อย')\n"
        "- แบบฟอร์ม/เอกสารดาวน์โหลด (เช่น 'ขอแบบฟอร์ม', 'ดาวน์โหลดคู่มือ', 'ส่งแบบฟอร์มหน่อย')\n"
        "- แหล่งอ้างอิง/ที่มาของข้อมูล (เช่น 'ขอแหล่งข้อมูล', 'อ้างอิงได้ที่ไหน')\n\n"
        "is_link_request=false เมื่อ user:\n"
        "- ถามเนื้อหา/ขั้นตอน/รายละเอียดของกฎหมาย (ไม่ได้ขอตัวลิงก์)\n"
        "- ถามว่า 'ต้องใช้เอกสารอะไรบ้าง' (ถามรายการ ไม่ใช่ขอดาวน์โหลด)\n\n"
        'ตอบ JSON เท่านั้น: {"is_link_request": true/false, "confidence": 0.0}\n'
        "confidence < 0.75 → is_link_request=false"
    )


def build_slot_skip_detect_prompt(user_text: str) -> str:
    """
    Detect whether user wants to skip the current slot question.
    Called when _SLOT_SKIP_RE regex misses — catches indirect phrasing like
    "ยังตัดสินใจไม่ได้", "ไม่ค่อยแน่ใจ", "ขอผ่านไปก่อน".
    Confidence threshold: 0.75.
    """
    return (
        "คุณคือ classifier ตรวจว่า user ต้องการ 'ข้ามคำถามนี้ไปก่อน' หรือไม่\n"
        "บริบท: บอทกำลังถามข้อมูลเพิ่มเติม (เช่น ประเภทนิติบุคคล, ที่ตั้งร้าน) แต่ user ยังไม่รู้หรือไม่ต้องการตอบ\n"
        f"user_text: {user_text}\n\n"
        "is_skip=true เมื่อ user:\n"
        "- บอกว่าไม่รู้/ไม่แน่ใจ/ยังหาข้อมูลอยู่ (เช่น 'ยังตัดสินใจไม่ได้', 'ไม่ค่อยแน่ใจ', 'ยังไม่รู้เลย')\n"
        "- ขอข้ามหรือผ่านไปก่อน (เช่น 'ขอผ่านไปก่อน', 'ข้ามก่อนได้ไหม', 'ไม่ต้องถามก็ได้')\n"
        "- ไม่ทราบข้อมูล (เช่น 'ไม่ทราบครับ', 'ไม่มีข้อมูล')\n\n"
        "is_skip=false เมื่อ user:\n"
        "- ตอบคำถามโดยตรง (เลือกตัวเลือก, บอกประเภท, บอกที่ตั้ง)\n"
        "- ถามคำถามใหม่\n"
        "- ทักทาย/ขอบคุณ\n\n"
        'ตอบ JSON เท่านั้น: {"is_skip": true/false, "confidence": 0.0}\n'
        "confidence < 0.75 → is_skip=false"
    )


def build_followup_contextual_detect_prompt(user_text: str, last_topic: str) -> str:
    """
    Detect whether user is asking a CONTEXTUAL follow-up about their specific case
    (not a new topic). Called when _FOLLOWUP_CONTEXTUAL_RE regex misses.
    Confidence threshold: 0.75.
    """
    return (
        "คุณคือ classifier ตรวจว่า user ถาม 'follow-up เกี่ยวกับกรณีของตัวเอง' หรือไม่\n"
        f"user_text: {user_text}\n"
        f"หัวข้อล่าสุดที่คุยกัน: {last_topic or '(ยังไม่มี)'}\n\n"
        "is_followup=true เมื่อ user ต้องการรู้ว่าข้อมูลนั้นใช้ได้กับกรณีของตัวเองไหม เช่น:\n"
        "- 'แบบนี้ใช้ได้กับร้านผมไหม'\n"
        "- 'กรณีของฉันต้องทำแบบไหน'\n"
        "- 'แบบที่บอกมาเหมาะกับผมไหม'\n"
        "- 'ของฉันต้องทำอะไรบ้าง'\n\n"
        "is_followup=false เมื่อ user ถามเรื่องใหม่ที่ไม่เกี่ยวกับหัวข้อล่าสุด\n\n"
        'ตอบ JSON เท่านั้น: {"is_followup": true/false, "confidence": 0.0}\n'
        "confidence < 0.75 → is_followup=false"
    )


def build_thanks_detect_prompt(user_text: str) -> str:
    """
    Detect whether user is expressing thanks — catches formal/unusual Thai thanks
    that _THANKS_RE regex misses (e.g., "ขอบพระคุณ", "กราบขอบพระคุณ", "ขอขอบคุณ").
    Called from _looks_like_greeting_or_thanks after legal/question guards pass.
    Confidence threshold: 0.75.
    """
    return (
        "คุณคือ classifier ตรวจว่า user กำลัง 'ขอบคุณ' หรือไม่\n"
        f"user_text: {user_text}\n\n"
        "is_thanks=true เช่น:\n"
        "- 'ขอบพระคุณมาก', 'กราบขอบพระคุณ', 'ขอขอบคุณ', 'ขอบคุณเป็นอย่างยิ่ง'\n"
        "- 'thank you so much', 'many thanks', 'cheers'\n"
        "- การแสดงความซาบซึ้งในทุกรูปแบบ\n\n"
        "is_thanks=false เช่น:\n"
        "- ถามคำถาม, บอกเล่าเรื่อง, ให้ข้อมูล\n"
        "- ทักทาย (สวัสดี)\n\n"
        'ตอบ JSON เท่านั้น: {"is_thanks": true/false, "confidence": 0.0}\n'
        "confidence < 0.75 → is_thanks=false"
    )


def build_academic_stop_detect_prompt(user_text: str) -> str:
    """
    Detect whether user wants to STOP academic mode and return to normal (practical) mode.
    Called when _ACADEMIC_STOP_RE regex misses — catches indirect phrasing.
    Confidence threshold: 0.80 (high — false positives abort academic unexpectedly).
    """
    return (
        "คุณคือ classifier ตรวจว่า user ต้องการ 'หยุด/ออกจากโหมดรายละเอียด (Academic)' หรือไม่\n"
        "บริบท: บอทกำลังให้ข้อมูลเชิงลึกแบบละเอียด (Academic mode) และ user อาจต้องการหยุด\n"
        f"user_text: {user_text}\n\n"
        "is_stop=true เมื่อ user ต้องการ:\n"
        "- หยุด/จบโหมดนี้ (เช่น 'พอแล้ว', 'ไม่ต้องการรายละเอียดแล้ว', 'จบได้แล้ว')\n"
        "- กลับโหมดปกติ (เช่น 'กลับโหมดปกติ', 'ออกจากโหมดนี้')\n"
        "- ยกเลิกการดูรายละเอียด (เช่น 'ยกเลิก', 'ไม่เอาแล้ว', 'หยุดก่อน')\n\n"
        "is_stop=false เมื่อ user:\n"
        "- ถามคำถามต่อ, ขอรายละเอียดเพิ่ม, ขอดูส่วนอื่น\n"
        "- ทักทาย/ขอบคุณ (ขอบคุณไม่ใช่การ stop)\n"
        "- ตอบคำถาม slot\n\n"
        'ตอบ JSON เท่านั้น: {"is_stop": true/false, "confidence": 0.0}\n'
        "confidence < 0.80 → is_stop=false"
    )

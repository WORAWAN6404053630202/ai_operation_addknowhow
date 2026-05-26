# code/model/persona_practical.py
import json
import logging
import re
from difflib import SequenceMatcher
from typing import Tuple, Dict, Any, List, Optional, Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import conf
from model.conversation_state import ConversationState
from utils.llm_call import llm_invoke, extract_llm_text
from utils.prompts_practical import SYSTEM_PROMPT as SYSTEM_PROMPT_PRACTICAL, build_lqs_license_detect_prompt, build_satisfaction_detect_prompt, build_dont_know_detect_prompt, build_short_followup_detect_prompt
from utils.prompts_supervisor import build_legal_q_detect_prompt
from utils.query_synonyms import SYNONYM_PATTERNS

# Import professional logging
from utils.logger import get_logger, log_function_call, TimingContext

_LOG = logging.getLogger("restbiz.practical")  # Keep for backward compatibility
logger = get_logger(__name__)  # ใช้ logger ใหม่ (มี structure + context)

# Metadata fields with no semantic value for the LLM — always hidden from docs_json
_LLM_HIDDEN_METADATA_KEYS = frozenset({"row_id", "source"})


# Module-level constants for _classify_link() — created once at import, not per call.
_CLASSIFY_LINK_GUIDE_KW: tuple = (
    "คู่มือ",
    "youtube", "youtu.be", "vdo ", " vdo",
    "facebook",
    "workflow",
    "ขั้นตอนการ",
    "ความรู้เรื่อง",
    "วิธีการ", "วิธีใช้", "การสอน",
    "tutorial", "guide",
    "info",
)
_CLASSIFY_LINK_FORM_KW: tuple = (
    "แบบฟอร์ม",
    "แบบ บอจ", "แบบ ก.", "แบบ ว.", "แบบ สปส", "แบบ สณ",
    "แบบ ภพ", "แบบ ภส", "แบบ อส", "แบบ บค", "แบบ รส",
    "ดาวน์โหลดเอกสาร", "ดาวน์โหลดแบบฟอร์ม", "ดาวน์โหลด",
    "เอกสาร",
    "แบบคำขอ", "แบบแจ้ง", "แบบแสดง", "แบบคำรับรอง",
    "คำขอจดทะเบียน", "คำขอใช้บริการ",
    # NOTE: "คำขอ" standalone intentionally excluded — it appears in process/registration
    # context ("ยื่นคำขอ", "คำขอชำระค่าธรรมเนียม") which are portals/guides, not form templates.
    "ตัวอย่างการกรอก", "ตัวอย่างการจดทะเบียน", "ตัวอย่าง",
    "ใบสมัคร",
    "หนังสือมอบอำนาจ", "หนังสือยินยอม", "หนังสือให้ความยินยอม",
    "บัญชีรายชื่อผู้ถือหุ้น",
    "กรอกข้อมูล",  # e.g. "ไฟล์สำหรับกรอกข้อมูลสถานประกอบการสาขา" — fillable data entry file
)
_CLASSIFY_LINK_REG_KW: tuple = (
    "สำหรับลงทะเบียน", "ลงทะเบียนออนไลน์", "ลงทะเบียนยื่นคำขอ", "ลงทะเบียน",
    "ยื่นออนไลน์", "ยื่นจดทะเบียนออนไลน์", "ยื่นคำขอ",
    "e-service", "e service", "eservice",
    "สมัครบริการ", "สมัครสมาชิก",
    "mobile application", "app store", "play store",
    "bma oss", "bmaoss",
)
_CLASSIFY_LINK_REF_KW: tuple = (
    "faq", "คำถามที่พบบ่อย", "ถาม-ตอบ", "ถามตอบ",
    "กฎหมาย", "พระราชบัญญัติ", "ระเบียบ", "ประกาศ",
)

# Multi-topic menu threshold: ≤ this many topics → answer all together; more → summary+menu
_MULTI_TOPIC_MENU_THRESHOLD: int = 3

# Short 1-line descriptions per license_type shown in 4+ topic summary menu
_TOPIC_DESC_MAP: dict = {
    "ใบทะเบียนพาณิชย์": "ขึ้นทะเบียนธุรกิจกับกรมพัฒนาธุรกิจการค้า (DBD)",
    "ใบภาษีมูลค่าเพิ่ม ภพ.20": "จด VAT กับกรมสรรพากร เมื่อยอดขายถึงเกณฑ์",
    "การขึ้นทะเบียนกองทุนประกันสังคม": "ขึ้นทะเบียนนายจ้าง-ลูกจ้างกับสำนักงานประกันสังคม",
    "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร": "ใบอนุญาตเปิดร้านอาหารจาก BMA/อปท.",
    "ใบอนุญาตจำหน่ายสุรา": "ใบอนุญาตขายสุราจากกรมสรรพสามิต",
    "แบบแสดงรายการภาษีป้ายร้านอาหาร": "ภาษีป้ายร้านอาหาร ยื่นกับเทศบาล/กทม.",
    "แบบแสดงรายการภาษีป้าย": "ภาษีป้ายโฆษณา ยื่นกับเทศบาล/กทม.",
    "ใบรับรองมาตรฐานร้านอาหาร": "มาตรฐานสุขาภิบาลอาหาร (SAN/SAN PLUS)",
    "ใบวุฒิบัตรผู้สัมผัสอาหาร": "อบรมผู้สัมผัสอาหารตามกฎหมาย",
    "ใบรับรองแพทย์ 9 โรค(สณ.11)": "ตรวจสุขภาพ 9 โรคต้องห้ามสำหรับผู้ประกอบการ",
    "ระบบชำระเงินออนไลน์": "สมัคร QR Payment / PromptPay สำหรับร้านค้า",
    "เครื่องรูดบัตร EDC": "สมัครเครื่อง EDC รับชำระบัตรเครดิต/เดบิต",
    "จัดการการเงิน": "ระบบบัญชีและการเงินสำหรับธุรกิจ",
}

# Groups of related topics with a shared context note (used in 4+ topic summary header)
_RELATED_TOPIC_GROUPS: list = [
    (
        frozenset({"ใบทะเบียนพาณิชย์", "การขึ้นทะเบียนกองทุนประกันสังคม", "ใบภาษีมูลค่าเพิ่ม ภพ.20"}),
        "ซึ่งเป็นเรื่องพื้นฐานที่ธุรกิจใหม่ต้องดำเนินการ",
    ),
    (
        frozenset({"ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร", "ใบอนุญาตจำหน่ายสุรา",
                   "แบบแสดงรายการภาษีป้ายร้านอาหาร", "แบบแสดงรายการภาษีป้าย",
                   "ใบรับรองมาตรฐานร้านอาหาร"}),
        "ซึ่งเป็นใบอนุญาตที่ร้านอาหารต้องขอก่อนเปิดกิจการ",
    ),
    (
        frozenset({"ใบวุฒิบัตรผู้สัมผัสอาหาร", "ใบรับรองแพทย์ 9 โรค(สณ.11)"}),
        "ซึ่งเป็นเอกสารด้านสุขลักษณะที่พนักงานต้องมี",
    ),
    (
        frozenset({"ระบบชำระเงินออนไลน์", "เครื่องรูดบัตร EDC"}),
        "ซึ่งเป็นช่องทางรับชำระเงินสำหรับร้านค้า",
    ),
]


def _classify_link(desc: str, url: str) -> str:
    """
    Classify a link based on URL first (known portals), then desc keywords.

    Categories:
      'guide'        — คู่มือ, วิดีโอ, workflow, ขั้นตอนการ
      'form'         — แบบฟอร์ม, เอกสาร, ดาวน์โหลด, แบบ บอจ etc. (NOT คำขอ standalone)
      'registration' — ลงทะเบียนออนไลน์, e-service, สมัครบริการ, mobile app
      'ref'          — FAQ, คำถามที่พบบ่อย, กฎหมาย, ระเบียบ, ประกาศ, หน้าข้อมูลทั่วไป

    Priority order (first match wins): known_portal_url → guide → form → registration → ref_kw → url_fallback → ref
    Note: eservice./bmaoss./bma-oss/first-login/oss.go.th/oss.bangkok caught at Priority 0 before
    desc checks, so their desc containing "คำขอ" won't misfire the form check.
    """
    # Priority 0: Known registration portals — URL overrides desc classification entirely.
    # These are always actionable portals regardless of how desc is worded
    # (e.g. desc "Website VDO ประกอบการอบรม" would incorrectly trigger 'guide' below,
    # but foodhandler.anamai.moph.go.th IS the training registration portal).
    # Also catches BMA OSS / eservice portals whose desc may contain "คำขอ" (process word,
    # not a form-template word) which would otherwise fire _FORM_KW before reaching _REG_KW.
    _url_pre = (url or "").lower().strip()
    if _url_pre:
        # Guard: skip Priority 0 for social-media domains (facebook.com, line.me, etc.)
        # whose URLs may contain portal keywords as part of post text
        # e.g. "facebook.com/.../bma-oss-..." → bma-oss in URL but it's a social post, not a portal.
        _is_social = bool(re.search(r"(facebook\.com|fb\.com|youtu\.?be|youtube\.com|line\.me|lin\.ee|tiktok\.com|instagram\.com)", _url_pre))
        if not _is_social and re.search(
            r"(foodhandler\.anamai\.moph\.go\.th(?!/webapp/assets)"
            r"|eservice\.|bmaoss\.|bma-oss|/first-login"
            r"|oss\.go\.th|oss\.bangkok)",
            _url_pre,
        ):
            return "registration"

    desc_l = desc.lower().strip()

    # Guide: manual, video, workflow, how-to
    if any(kw in desc_l for kw in _CLASSIFY_LINK_GUIDE_KW):
        return "guide"

    # Form: downloadable forms and documents
    if any(kw in desc_l for kw in _CLASSIFY_LINK_FORM_KW):
        return "form"

    # Registration: online portals and apps for applying
    if any(kw in desc_l for kw in _CLASSIFY_LINK_REG_KW):
        return "registration"

    # Ref: FAQ / legal documents — shown only on explicit user request
    if any(kw in desc_l for kw in _CLASSIFY_LINK_REF_KW):
        return "ref"

    # URL-based fallback — when desc gives no category signal, inspect the URL itself
    # Note: eservice./bmaoss./bma-oss/first-login/oss.go.th/oss.bangkok are already caught by Priority 0.
    url_l = (url or "").lower().strip()
    if url_l:
        # Bangkok webportal non-district paths → registration portal
        if re.search(
            r"webportal\.[^/]+/(index|register|apply|service|page/sub)",
            url_l,
        ):
            return "registration"
        # PDF files — sub-classify using desc as PRIMARY signal (URL path is secondary)
        # Rule: PDF extension alone does NOT mean it is a form.
        # Check desc first; URL path used only for guide keywords (e.g. "คู่มือ" in Thai filename).
        if re.search(r"\.pdf(\?|&|$)", url_l):
            combined_guide = url_l + " " + desc_l  # URL path may have Thai guide keywords
            if re.search(r"(คู่มือ|ขั้นตอน|วิธี|user.?file|user_file|manual|guide|tutorial)", combined_guide):
                return "guide"
            # Form: desc_l ONLY — do not use URL path to decide (path words like "อนุญาต" are unrelated)
            # Require specific form-template keywords only; "คำขอ" and "อนุญาต" are process words
            # ("ยื่นคำขอ", "คำขอชำระค่าธรรมเนียม", "ใบอนุญาต") not form templates.
            if re.search(r"(แบบคำขอ|แบบฟอร์ม|แบบแจ้ง|ฟอร์ม|form|request|template)", desc_l):
                return "form"
            # Ref: FAQ / legal PDFs — must not default to guide
            if any(kw in desc_l for kw in _CLASSIFY_LINK_REF_KW):
                return "ref"
            return "guide"  # PDF with no strong desc signal → guide by default
        # Webportal pages (non-PDF)
        if re.search(r"webportal\.[^/]+/(page|sub|index)", url_l):
            return "registration"
        # Google Sheets — always a fillable data entry file (e.g. branch-info form)
        if re.search(r"docs\.google\.com/spreadsheets", url_l):
            return "form"

    # Ref: fallback — kept for explicit reference requests, never silently dropped
    return "ref"


def _parse_link_entries(text: str) -> list:
    """Parse research_reference text into list of (desc, url) tuples.

    Handles URLs split across multiple lines (newline inside URL) by joining
    continuation lines that don't start a new entry (no bullet, not empty, not http).
    Also handles URLs concatenated without any separator (e.g. "https://a.comhttps://b.com")
    by inserting a newline before each https:// occurrence before splitting.
    Filters out truncated URLs that are clearly incomplete.
    """
    # Pre-split concatenated URLs: "https://a.comhttps://b.com" → "https://a.com\nhttps://b.com"
    text = re.sub(r"(https?://)", r"\n\1", text).lstrip("\n")
    lines = text.split("\n")
    # Step 1: re-join URL lines that were split mid-URL (no bullet prefix, not empty)
    merged: list = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            merged.append("")
            continue
        # Continuation of a URL: previous line is a URL fragment AND current line has no spaces
        # and doesn't start a new entry (no bullet/dash/asterisk/http)
        if (
            merged
            and merged[-1].startswith("http")
            and not stripped.startswith(("http", "•", "-", "*", "Website", "website"))
            and " " not in stripped  # URL fragment has no spaces
        ):
            merged[-1] = merged[-1] + stripped
        else:
            merged.append(stripped)

    entries: list = []
    i = 0
    while i < len(merged):
        stripped = merged[i]
        if not stripped:
            i += 1
            continue
        if stripped.startswith("http"):
            if entries:
                entries[-1] = (entries[-1][0], stripped)
            i += 1
            continue
        desc = stripped
        url = ""
        # Detect URL embedded in description line (e.g., "desc text  https://url")
        _emb = re.search(r'\s+(https?://\S+)$', desc)
        if _emb:
            url = _emb.group(1)
            desc = desc[:_emb.start()].strip()
            i += 1
        elif i + 1 < len(merged) and merged[i + 1].startswith("http"):
            url = merged[i + 1]
            i += 2
        else:
            i += 1
        entries.append((desc, url))

    # Step 2: filter entries with clearly truncated/incomplete URLs
    # A valid URL must end with a path character (not % or partial percent-encoding)
    import re as _re
    clean: list = []
    for desc, url in entries:
        if url and _re.search(r'%[0-9A-Fa-f]?$', url):
            # URL ends with incomplete percent-encoding → truncated, skip
            continue
        clean.append((desc, url))
    return clean



# Token: Whitelist — ส่ง metadata keys ที่ LLM ต้องการโดยตรง
# content ถูกตัดที่ LLM_DOC_CHARS_PRACTICAL (400 chars) ดังนั้น fields สำคัญต้องอยู่ใน metadata
_LLM_METADATA_WHITELIST = frozenset({
    "data_type",               # แหล่งข้อมูล: regulatory | marketing | business_guide
    "license_type",            # ประเภทใบอนุญาต
    "operation_topic",         # หัวข้อการดำเนินการ
    "main_topic",              # หัวข้อหลัก (marketing/business_guide docs)
    "sub_topic",               # หัวข้อย่อย (marketing/business_guide docs)
    "entity_type_normalized",  # ประเภทนิติบุคคล
    "registration_type",       # ประเภทการจดทะเบียน
    "department",              # หน่วยงาน
    "fees",                    # ค่าธรรมเนียม
    "operation_duration",      # ระยะเวลาดำเนินการ
    "service_channel",         # ช่องทางยื่น
    # research_reference is intentionally excluded: URLs are injected post-LLM via safety nets
    # and shown only when user explicitly requests them. Keeping research_reference
    # in docs_json causes the LLM to spontaneously copy URLs with wrong labels.
    "answer_guideline",        # แนวคำตอบ (marketing/business_guide docs)
    "operation_steps",         # ขั้นตอนการดำเนินการ (สำคัญ — content อาจถูกตัดก่อนถึงส่วนนี้)
    "identification_documents",# รายการเอกสารที่ต้องใช้ (สำคัญ — ต้องเห็นทั้งหมด)
    "legal_regulatory",        # ข้อกำหนดทางกฎหมาย บทลงโทษ ค่าปรับ
    "terms_and_conditions",    # เงื่อนไขและหน้าที่ของผู้ประกอบการ
    "restaurant_ai_document",  # เอกสาร/ฟอร์ม AI ร้านอาหาร (bypasses content truncation via metadata track)
})

# P0: practical policy "last gate"
try:
    from utils.practical_lint import enforce_practical_policy  # type: ignore
except Exception:  # pragma: no cover
    enforce_practical_policy = None  # fallback handled in _apply_practical_lint


def _repair_json_strings(text: str) -> str:
    """
    Repair common LLM JSON output issues inside string values:
    1. Literal \\n / \\r / \\t → JSON-escaped (\\\\n etc.)
    2. Unescaped " that is NOT a closing quote → escaped to \\"
       Heuristic for closing vs internal quote: if the char immediately
       following the " (after optional whitespace) is one of ,:}] or end-of-string,
       it is treated as a closing quote. Otherwise it is an internal quote.
    Character-by-character scan tracks in_string state correctly.
    """
    result: list = []
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if not in_string:
            if c == '"':
                in_string = True
            result.append(c)
            i += 1
        else:
            if c == '\\' and i + 1 < n:
                result.append(c)
                result.append(text[i + 1])
                i += 2
            elif c == '"':
                # Decide: closing quote or unescaped internal quote?
                # Peek at next non-whitespace char.
                j = i + 1
                while j < n and text[j] in ' \t\r\n':
                    j += 1
                next_c = text[j] if j < n else ''
                if next_c in (':', ',', '}', ']', ''):
                    # Closing quote — end of this string value
                    in_string = False
                    result.append(c)
                else:
                    # Internal unescaped quote — escape it
                    result.append('\\"')
                i += 1
            elif c == '\n':
                result.append('\\n')
                i += 1
            elif c == '\r':
                result.append('\\r')
                i += 1
            elif c == '\t':
                result.append('\\t')
                i += 1
            else:
                result.append(c)
                i += 1
    return ''.join(result)


class PracticalPersonaService:
    """
    Practical Persona (Agentic, fast, short)

    Supervisor-owned menu contract (IMPORTANT):
    - Practical MUST always be able to "consume choices" when pending_slot exists:
        - numeric (1,2,3), ranges (1-3), multi (1,3), digit-pack (123), exact label, and free text.
    - Practical MUST NOT hijack Supervisor's main menu/greeting:
        - If state.context["supervisor_owns_menu"] is True:
            - do NOT render greeting/topic menu
            - do NOT set/override pending_slot for topic menu
    - Pending-slot recovery remains opt-in and additionally gated by owner to prevent hijack.
    """

    persona_id = "practical"

    _EN_GREET_RE = re.compile(r"^\s*(hi+|hello+|hey+|yo+)\b", re.IGNORECASE)
    _TH_WATDEE_RE = re.compile(r"^\s*หวัด[^\s]{0,6}", re.IGNORECASE)
    _TH_SAWASDEE_RE = re.compile(r"^\s*สว[^\s]{0,8}ดี", re.IGNORECASE)
    _TH_DEE_RE = re.compile(r"^\s*ดี(?:ครับ|คับ|ค่ะ|คะ|งับ|จ้า|จ้ะ|ค่า)?", re.IGNORECASE)

    _THANKS_RE = re.compile(
        r"(ขอบคุณ|ขอบใจ|ขอบพระคุณ|ขอบคุณมาก|ขอบคุณนะ|thx|thanks|thank you)",
        re.IGNORECASE,
    )
    _OK_RE = re.compile(
        r"^\s*(โอเค|ok|okay|รับทราบ|เข้าใจแล้ว|เข้าใจ|ได้เลย|เรียบร้อย|เคลียร์|เคลียแล้ว|พอแล้ว|พอครับ|พอค่ะ|ครบแล้ว|got\s*it|clear)\s*(ครับ|คับ|ค่ะ|คะ)?\s*$",
        re.IGNORECASE,
    )

    _LEGAL_SIGNAL_RE = re.compile(
        r"(ใบอนุญาต|จดทะเบียน|ทะเบียนพาณิชย์|ภาษี|vat|ภพ\.?20|สรรพากร|เทศบาล|สำนักงานเขต|สุขาภิบาล|กรม|ค่าธรรมเนียม|เอกสาร|ขั้นตอน|บทลงโทษ|ประกาศ|พ\.ร\.บ|เปิดร้าน|ประกันสังคม|กองทุน|งบการเงิน|ผลกระทบ|ความเสี่ยง)",
        re.IGNORECASE,
    )

    _DONT_KNOW_RE = re.compile(r"^\s*(ไม่รู้|ไม่แน่ใจ|ไม่ทราบ|งง|แล้วแต่|อะไรก็ได้)\s*$")
    _ASK_TYPES_RE = re.compile(r"(มีประเภทอะไรบ้าง|ประเภทอะไรบ้าง|มีแบบไหนบ้าง|มีอะไรบ้าง)\s*$")

    _NUM_OPTION_LINE_RE = re.compile(r"^\s*(\d{1,2})\)\s*(.+?)\s*$")
    _LIKELY_SELECTION_RE = re.compile(r"^\s*[\d\s,/-]+\s*$")

    # Topic menu sanitation (STRICT)
    _TOPIC_MIN_LEN = 3
    _TOPIC_MAX_LEN = 52
    _TOPIC_REJECT_IF_HAS_NEWLINE = True

    def _sanitize_topic_label(self, s: str) -> str:
        raw = (s or "")
        t = raw.strip()
        if not t:
            return ""
        if self._TOPIC_REJECT_IF_HAS_NEWLINE and ("\n" in raw or "\r" in raw):
            return ""
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) > self._TOPIC_MAX_LEN:
            return ""
        if "ตามกฎหมาย" in t or "มีสิทธิ" in t or "ผู้ประกอบกิจการ" in t:
            return ""
        return t

    # Owner/menu guards (NEW)
    def _supervisor_owns_menu(self, state: ConversationState) -> bool:
        ctx = state.context or {}
        return bool(ctx.get("supervisor_owns_menu", False))

    def _get_last_bot_owner(self, state: ConversationState) -> str:
        ctx = state.context or {}
        owner = (ctx.get("last_bot_owner") or "").strip().lower()
        return owner

    def _set_last_bot_owner(self, state: ConversationState, owner: str) -> None:
        state.context = state.context or {}
        state.context["last_bot_owner"] = (owner or "").strip()

    # Safe append (dedupe)
    def _append_user_once(self, state: ConversationState, content: str) -> None:
        state.add_user_message_once(content)

    def _append_assistant(self, state: ConversationState, content: str) -> None:
        """
        P0: Dedupe assistant to prevent recursion/forced flows from duplicating turns.
        Also tag last_bot_owner='practical' for safe pending-slot recovery gating.
        """
        c = "" if content is None else str(content)
        if not c.strip():
            return
        if state.messages:
            last = state.messages[-1]
            if last.get("role") == "assistant" and (last.get("content") or "").strip() == c.strip():
                return
        msg = {"role": "assistant", "content": c}
        state.messages.append(msg)
        state.display_messages.append(msg)
        self._set_last_bot_owner(state, "practical")

    # Greeting prefix (Practical fallback)
    _GREET_PREFIX_FALLBACKS: Dict[str, str] = {
        "greet": "สวัสดีครับ ต้องการข้อมูลด้านใดสำหรับร้านอาหารของคุณครับ",
        "thanks": "ยินดีครับ อยากไปต่อหัวข้อไหนครับ",
        "smalltalk": "แล้วต้องการข้อมูลด้านใดสำหรับร้านของคุณครับ",
        "blank": "มีเรื่องอะไรให้ช่วยสำหรับร้านอาหารของคุณครับ",
    }

    def _default_greet_llm_call(self) -> Callable[[str, List[str], int], dict]:
        """
        Returns JSON: { "prefix": "..." }
        """
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.35,
            max_tokens=120,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(kind: str, menu: List[str], greet_streak: int) -> dict:
            menu_preview = ", ".join([str(x) for x in (menu or [])[:6]])
            prompt = (
                "หน้าที่: สร้างประโยคทักทาย/ตอบรับแบบมนุษย์ สำหรับบอทกฎหมายร้านอาหารไทย (โหมด practical)\n"
                "เงื่อนไข:\n"
                "- ห้ามสั่งผู้ใช้ว่า 'เลือก.../พิมพ์.../กด...'\n"
                "- ให้สั้น 1-2 ประโยค และถาม 1 คำถามสั้นๆ\n"
                "- โทน: practical (ตรง กระชับ มืออาชีพ)\n"
                "- ถ้า greet_streak >= 2 พยายามเปลี่ยนคำเล็กน้อยไม่ให้ซ้ำ\n"
                "- ต้องลงท้ายด้วย 'ครับ'\n"
                "ตอบเป็น JSON เท่านั้น: {\"prefix\": \"...\"}\n"
                f"kind: {kind}\n"
                f"greet_streak: {greet_streak}\n"
                f"ตัวอย่างหัวข้อในระบบ (เพื่ออ้างอิงคำ): {menu_preview}\n"
            )
            try:
                # วัดเวลาการเรียก LLM
                with TimingContext(logger, "llm_greet_call"):
                    text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Practical/greet")).strip()
                    
                # Log สำเร็จ
                logger.log_with_data("info", "สร้างคำทักทายสำเร็จ", {
                    "action": "greet_generation",
                    "kind": kind,
                    "greet_streak": greet_streak,
                    "model": switch_model,
                    "response_length": len(text)
                })
            except Exception as e:
                # Log error
                logger.log_with_data("error", "สร้างคำทักทายล้มเหลว", {
                    "action": "greet_generation",
                    "error": str(e),
                    "fallback": "empty_dict"
                })
                return {}

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as _je:
                _LOG.debug("[Practical/lqs_lt] JSON parse failed: %s", _je)
                return {}

        return _call

    def _default_lqs_license_llm_call(self) -> Callable[[str, List[str]], dict]:
        """Lightweight LLM (switch model) to identify license type from user query when LQS score=0."""
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=100,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str, candidates: List[str]) -> dict:
            prompt = build_lqs_license_detect_prompt(user_text, candidates)
            try:
                text = extract_llm_text(llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Practical/lqs_lt")).strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_satisfaction_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: did user express satisfaction/done-ness?"""
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
            prompt = build_satisfaction_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Practical/satisfaction")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Practical/satisfaction] LLM call failed: %s", _e)
                return {}
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as _je:
                _LOG.debug("[Practical/satisfaction] JSON parse failed: %s", _je)
                return {}

        return _call

    def _default_dont_know_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: user expresses uncertainty or asks to see available types."""
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
            prompt = build_dont_know_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Practical/dont_know")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Practical/dont_know] LLM call failed: %s", _e)
                return {}
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as _je:
                _LOG.debug("[Practical/dont_know] JSON parse failed: %s", _je)
                return {}

        return _call

    def _default_short_followup_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: is text a short continuation question (reuse docs) or new standalone question?"""
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
            prompt = build_short_followup_detect_prompt(user_text)
            try:
                text = extract_llm_text(
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Practical/short_followup")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Practical/short_followup] LLM call failed: %s", _e)
                return {}
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as _je:
                _LOG.debug("[Practical/short_followup] JSON parse failed: %s", _je)
                return {}

        return _call

    def _default_practical_legal_q_llm_call(self) -> Callable[[str], dict]:
        """Lightweight classifier: is this a legal question? Used in _should_retrieve_new_topic."""
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
                    llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Practical/legal_q")
                ).strip()
            except Exception as _e:
                _LOG.warning("[Practical/legal_q] LLM call failed: %s", _e)
                return {}
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as _je:
                _LOG.debug("[Practical/legal_q] JSON parse failed: %s", _je)
                return {}

        return _call

    def _pick_greet_prefix(self, kind: str, menu: List[str], greet_streak: int) -> str:
        kind2 = kind if kind in {"greet", "thanks", "smalltalk", "blank"} else "greet"
        try:
            res = self.llm_greet_call(kind2, menu, int(greet_streak or 0))
        except Exception as _e:
            _LOG.debug("[Practical/greet] LLM greet call failed: %s", _e)
            res = {}

        prefix = ""
        if isinstance(res, dict):
            prefix = str(res.get("prefix") or "").strip()

        if prefix:
            prefix = re.sub(r"\s+", " ", prefix).strip()
            if len(prefix) > 140:
                prefix = ""
            if re.search(r"(เลือก|พิมพ์|กด)", prefix) and re.search(r"(ข้อ|หัวข้อ|ด้านล่าง)", prefix):
                prefix = ""

        if not prefix:
            prefix = self._GREET_PREFIX_FALLBACKS.get(kind2, self._GREET_PREFIX_FALLBACKS["greet"])

        return prefix.strip()

    # Practical policy lint (P0 last gate)
    _MULTI_Q_SPLIT_RE = re.compile(r"[?？]\s*")
    _META_TALK_RE = re.compile(
        r"(ในฐานะ(ของ)?(บอท|ผู้ช่วย)|ฉันจะ|ผมจะ|ขออนุญาต|ขออธิบายว่า|ระบบนี้|นโยบาย|policy|ตามที่คุณขอ|ผมไม่สามารถ|ฉันไม่สามารถ|\bdocuments\b)",
        re.IGNORECASE,
    )

    # Strip standalone "documents" lines (LLM prompt bleed-through)
    _DOCUMENTS_LINE_RE = re.compile(r"(?m)^[ \t]*documents[ \t]*$", re.IGNORECASE)

    def _fallback_single_question(self, text: str, slot_key: str = "") -> str:
        """Clean and return a single question string. Never hardcode a fixed question —
        use the LLM-provided text as-is (after cleanup), or derive a sensible question
        from the slot_key context."""
        t = re.sub(r"\s+", " ", (text or "")).strip()

        t = self._META_TALK_RE.sub("", t).strip()
        t = re.sub(r"\s+", " ", t).strip()

        if "?" in t or "？" in t:
            first = re.split(r"[?？]", t, maxsplit=1)[0].strip()
            if first:
                t = first

        t = re.sub(r"(\d+\)|[-•])\s*", "", t).strip()

        # If still no question phrasing — use generic fallback
        if not re.search(r"(ไหม|หรือ|ยังไง|อย่างไร|อะไร|มั้ย|ได้ไหม|ต้องการ|อยาก|เป็นแบบ|รูปแบบ|ประเภท|ขนาด|พื้นที่|เท่าไหร่|เท่าใด|ตั้งอยู่|ใด|ดำเนินการ|แล้ว|ต่อจาก|ต่อไป|ถัดมา|ส่วน|นอกจาก|อีก|เพิ่ม)", t):
            t = "ช่วยบอกข้อมูลเพิ่มเติมหน่อยได้ไหมครับ?"
        else:
            if not t.endswith("ครับ"):
                t = t.rstrip(" .") + "ครับ"
        return t

    # ── Dynamic Clarification helpers ────────────────────────────────────────

    def _check_field_coverage(self, docs: list) -> List[str]:
        """Return Thai query terms for _COVERAGE_FIELDS truly absent from retrieved docs.

        A field counts as 'missing' (→ Round 3) only when BOTH conditions hold:
        1. No doc has a non-empty, non-nan value for that field.
        2. The field key is completely absent from every doc's metadata dict.
           Key present but value empty means the Sheet author intentionally left it
           blank (e.g. "ไม่มีค่าธรรมเนียม") — treat as covered, do NOT retry.

        Additionally: COVERAGE_FIELDS apply to regulatory content only.
        If none of the retrieved docs have data_type='regulatory', return [] so
        Round 3 never fires for marketing / business_guide queries.
        """
        # Guard: skip coverage check when no regulatory docs are present
        _has_regulatory = any(
            str((getattr(d, "metadata", None) or {}).get("data_type") or "").strip() == "regulatory"
            for d in docs
        )
        if not _has_regulatory:
            return []

        missing: List[str] = []
        for field, thai_term in self._COVERAGE_FIELDS.items():
            has_data = False
            field_key_seen = False
            for d in docs:
                md = getattr(d, "metadata", None) or {}
                if field in md:
                    field_key_seen = True
                    v = str(md.get(field) or "").strip()
                    if v and v not in ("nan", "None"):
                        has_data = True
                        break
            # Only count as missing when key is completely absent from all docs.
            # Key present but empty = intentionally blank in source → covered.
            if not has_data and not field_key_seen:
                missing.append(thai_term)
        return missing

    def _detect_divergence(
        self,
        docs: List[Dict[str, Any]],
        collected_slots: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Scan formatted docs (list of {"content":…,"metadata":…}) for diverging metadata.
        Returns the first _CLARIFICATION_FIELDS entry with ≥2 distinct values, or None."""
        collected_slots = collected_slots or {}

        # Group docs by license_type first.  Docs spanning multiple license_types each have
        # their own valid entity/registration variants — mixing them triggers false divergence
        # (e.g. entity_type options from ใบทะเบียนพาณิชย์ bleed into ประกันสังคม choices).
        # Only scan docs from the dominant license_type; let multi-license path handle the rest.
        _lt_groups: Dict[str, List] = {}
        for d in docs:
            _md = d.get("metadata") or {} if isinstance(d, dict) else (getattr(d, "metadata", None) or {})
            _lt = str(_md.get("license_type") or "").strip()
            if _lt and _lt not in ("nan", "None"):
                _lt_groups.setdefault(_lt, []).append(d)

        if len(_lt_groups) >= 2:
            # Multiple license types: focus on the dominant one (most docs), same as
            # practical's FOCUS FILTER.  This lets DC check entity_type divergence
            # within the relevant license instead of skipping DC entirely.
            _dominant_lt = max(_lt_groups, key=lambda k: len(_lt_groups[k]))
            _scan_docs = _lt_groups[_dominant_lt].copy()
        elif _lt_groups:
            _scan_docs = _lt_groups[next(iter(_lt_groups))].copy()
        else:
            _scan_docs = list(docs)

        # Track whether entity_type was found to be universal (steps similar across entity types).
        # If so, registration_type (a sub-category) is also universal → skip its DC too.
        _entity_type_universal = False

        for cfg in self._CLARIFICATION_FIELDS:
            key = cfg["metadata_key"]
            ckey = cfg.get("collected_key", key)
            # Skip if already known via primary or alt keys
            alt_keys = cfg.get("alt_keys") or []
            known = (
                collected_slots.get(ckey)
                or collected_slots.get(key)
                or any(collected_slots.get(ak) for ak in alt_keys)
            )
            if known:
                continue
            # registration_type is a sub-category of entity_type; if entity_type was found
            # to be universal (same procedure regardless of entity), registration_type is too.
            if key == "registration_type" and _entity_type_universal:
                continue
            max_len = cfg.get("max_value_len")
            vals: List[str] = []
            # op_topic → entity_val → set of operation_steps snippets
            _ot_val_steps: Dict[str, Dict[str, set]] = {}
            _ot_counts: Dict[str, int] = {}  # op_topic → doc count (to find primary)
            for d in _scan_docs:
                md = d.get("metadata") or {} if isinstance(d, dict) else (getattr(d, "metadata", None) or {})
                v = str(md.get(key) or "").strip()
                if not v or v in ("nan", "None"):
                    continue
                if max_len and len(v) > max_len:
                    continue
                if v not in vals:
                    vals.append(v)
                # Group by operation_topic: compare steps only within the primary operation
                # so that unrelated-operation step differences don't trigger false DC.
                _ot = str(md.get("operation_topic") or "").strip() or "__no_ot__"
                _steps_snippet = str(md.get("operation_steps") or "").strip()[:200]
                _ot_val_steps.setdefault(_ot, {}).setdefault(v, set()).add(_steps_snippet)
                _ot_counts[_ot] = _ot_counts.get(_ot, 0) + 1

            if len(vals) >= 2:
                # Use the #1 doc's operation_topic as the primary (most relevant to query).
                # Top-1 reliably captures the exact operation the user asked about
                # (e.g. "ชำรุด/สูญหาย" ranks #1 by BM25 for those exact keywords)
                # while averaging over top-8 would be swamped by other operations.
                _primary_ot = "__no_ot__"
                for _d0 in _scan_docs:
                    _md0 = _d0.get("metadata") or {} if isinstance(_d0, dict) else (getattr(_d0, "metadata", None) or {})
                    _ot0 = str(_md0.get("operation_topic") or "").strip()
                    if _ot0:
                        _primary_ot = _ot0
                        break
                if _primary_ot == "__no_ot__" and _ot_counts:
                    _primary_ot = max(_ot_counts, key=lambda k: _ot_counts[k])
                _primary_ev_steps = _ot_val_steps.get(_primary_ot, {})
                if len(_primary_ev_steps) < 2:
                    # Primary operation only covers one entity/registration type → skip DC
                    continue

                # Jaccard token similarity: if steps share ≥50% tokens, the procedure is
                # functionally identical regardless of entity/registration type → skip DC.
                # This handles cases where the same procedure is written slightly differently
                # per row but the answer would be the same for all groups.
                _tok_groups: List[frozenset] = []
                for _ev_ss in _primary_ev_steps.values():
                    _toks: set = set()
                    for _s in _ev_ss:
                        if _s:
                            _toks.update(re.findall(r"[฀-๿]+|[A-Za-z0-9]+", _s))
                    if _toks:
                        _tok_groups.append(frozenset(_toks))
                if len(_tok_groups) < 2:
                    continue  # fewer than 2 groups have step content → skip DC
                _any_diff = False
                for _ti in range(len(_tok_groups)):
                    for _tj in range(_ti + 1, len(_tok_groups)):
                        _inter = len(_tok_groups[_ti] & _tok_groups[_tj])
                        _union = len(_tok_groups[_ti] | _tok_groups[_tj])
                        if _union and _inter / _union < 0.5:
                            _any_diff = True
                            break
                    if _any_diff:
                        break
                if not _any_diff:
                    if key == "entity_type_normalized":
                        _entity_type_universal = True  # suppress registration_type DC too
                    # Secondary check for registration_type: even when operation_steps are
                    # similar, identification_documents can still differ between sub-types
                    # (e.g. ทะเบียนนายจ้าง: นิติบุคคล needs corporate docs, บุคคลธรรมดา
                    # needs personal ID; เครื่องรูดบัตร EDC similarly).
                    if key == "registration_type":
                        _id_ev: Dict[str, set] = {}
                        for _d2 in _scan_docs:
                            _md2 = _d2.get("metadata") or {} if isinstance(_d2, dict) else (getattr(_d2, "metadata", None) or {})
                            _v2 = str(_md2.get(key) or "").strip()
                            _id2 = str(_md2.get("identification_documents") or "").strip()
                            if _v2 in vals and _id2 and _id2 not in ("nan", "None"):
                                _id_ev.setdefault(_v2, set()).add(_id2[:300])
                        _id_filled = [v for v, ss in _id_ev.items() if any(s for s in ss)]
                        if len(_id_filled) >= 2:
                            _id_toks: List[frozenset] = []
                            for _v3 in _id_filled:
                                _t3: set = set()
                                for _s3 in _id_ev[_v3]:
                                    if _s3:
                                        _t3.update(re.findall(r"[฀-๿]+|[A-Za-z0-9]+", _s3))
                                if _t3:
                                    _id_toks.append(frozenset(_t3))
                            if len(_id_toks) >= 2:
                                for _idi in range(len(_id_toks)):
                                    for _idj in range(_idi + 1, len(_id_toks)):
                                        _id_inter = len(_id_toks[_idi] & _id_toks[_idj])
                                        _id_union = len(_id_toks[_idi] | _id_toks[_idj])
                                        if _id_union and _id_inter / _id_union < 0.5:
                                            _any_diff = True
                                            break
                                    if _any_diff:
                                        break
                    if not _any_diff:
                        continue  # no meaningful divergence in steps or id_docs → skip DC
                return {**cfg, "values": vals[:4]}
        return None

    def _build_clarification_question(
        self,
        diverge_info: Dict[str, Any],
        docs: List[Dict[str, Any]],
    ) -> str:
        """Build a friendly Thai question from divergence info."""
        values = diverge_info.get("values") or []
        if not values:
            return "ช่วยบอกข้อมูลเพิ่มเติมหน่อยได้ไหมครับ?"
        tmpl = diverge_info.get("question_tmpl", "")
        prefix = diverge_info.get("question_prefix", "")
        if tmpl and "{}" in tmpl and len(values) == 2:
            q = tmpl.format(*values)
        else:
            opts = "\n".join(f"{i + 1}) {v}" for i, v in enumerate(values))
            header = prefix or "ช่วยบอกข้อมูลเพิ่มเติมครับ"
            q = f"{header}\n{opts}"
        return q if q.endswith("ครับ") else q.rstrip(" .") + "ครับ"

    def _match_clarification_answer(
        self,
        user_text: str,
        values: List[str],
    ) -> Optional[str]:
        """Map user free-text to one of the clarification values.
        Tries exact, partial, and numeric index match. Returns None if no match."""
        raw = (user_text or "").strip()
        if not raw:
            return None
        raw_lower = raw.lower()
        for v in values:
            if v.lower() == raw_lower:
                return v
        for v in values:
            vl = v.lower()
            if vl in raw_lower or (len(raw_lower) >= 3 and raw_lower in vl):
                return v
        m = re.match(r"^(\d+)$", raw.strip())
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(values):
                return values[idx]
        return None

    def _fallback_practical_answer(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return "ตอนนี้ยังไม่พบข้อมูลที่ยืนยันได้ในเอกสารครับ"

        t = self._META_TALK_RE.sub("", t).strip()
        # Remove standalone "documents" lines (LLM prompt bleed-through)
        t = self._DOCUMENTS_LINE_RE.sub("", t).strip()
        # Remove "ทั้งหมด" menu-option lines that leaked into answer content
        t = re.sub(r"(?m)^[ \t]*\d*[.)]\s*ทั้งหมด\s*(ครับ|ค่ะ|นะ|นะครับ|นะคะ)?\s*$", "", t)
        t = re.sub(r"(?m)^[ \t]*ทั้งหมด\s*(ครับ|ค่ะ|นะ|นะครับ|นะคะ)?\s*$", "", t)
        # Remove fee section when value is zero/free — no value to show user
        _FREE_FEE = r"(?:ไม่มีค่าธรรมเนียม|ไม่เสียค่าธรรมเนียม|ไม่มี|ฟรี|0\s*บาท)"
        t = re.sub(
            rf"(?m)^\d+\)\s*ค่าธรรมเนียม[^\n]*\n(?:[ \t]*[•\-*]?\s*{_FREE_FEE}[^\n]*\n?)*",
            "",
            t,
        )

        # Strip trailing question that LLM appends after the main answer
        # (e.g. "มีคำถามอะไรเพิ่มเติมไหมครับ?" at the very end).
        # Match a short (≤100 chars) line at end-of-text that ends with ? or question markers.
        # This avoids cutting "?" from section headers or inline phrases mid-body.
        t = re.sub(
            r'(?:\n[ \t]*)+[^\n]{1,100}(?:\?|？|ไหม|หรือไม่|หรือเปล่า)\s*$',
            '',
            t,
            flags=re.IGNORECASE,
        ).strip()

        # Dedup URLs: remove from 🌐/📄/📖/📚 link sections any URL already in the body,
        # and also deduplicate URLs that appear more than once across all link sections.
        _links_hdr = re.search(r'(?m)^(?:🌐|📄|📖|📚)[^\n]*$', t)
        if _links_hdr:
            _body_part = t[:_links_hdr.start()]
            _links_part = t[_links_hdr.start():]
            _body_urls = set(re.findall(r'https?://\S+', _body_part))
            _links_lines = []
            _seen_in_sections: set = set()  # dedup within+across all link sections
            for _ln in _links_part.split('\n'):
                _found = re.search(r'https?://\S+', _ln)
                if _found:
                    _u = _found.group(0)
                    if _u in _body_urls or _u in _seen_in_sections:
                        continue  # skip duplicate
                    _seen_in_sections.add(_u)
                _links_lines.append(_ln)
            # If links section is now empty (only header left), remove it entirely
            _remaining = [l for l in _links_lines[1:] if l.strip()]
            if _remaining:
                t = _body_part + '\n'.join(_links_lines)
            else:
                t = _body_part.rstrip()

        # Move closing sentence to after all link sections if it sits before them.
        # Happens when LLM outputs: ...content... closing-sentence\n\n📄 links
        # Desired: ...content...\n\n📄 links\n\nclosing-sentence
        _lf_hdr = re.search(r'(?m)^(?:🌐|📄|📖|📚)[^\n]*$', t)
        if _lf_hdr:
            _lf_pre = t[:_lf_hdr.start()].rstrip('\n')
            _lf_links = t[_lf_hdr.start():]
            _lf_pre_lines = _lf_pre.split('\n')
            _lf_closing: list = []
            while _lf_pre_lines:
                _lf_tail = _lf_pre_lines[-1].strip()
                if not _lf_tail:
                    _lf_pre_lines.pop()
                    continue
                # A closing sentence: short, contains ครับ, not a numbered/bulleted/⚠️ line
                if (
                    len(_lf_tail) < 150
                    and 'ครับ' in _lf_tail
                    and not re.search(r'^\d+[.)]\s', _lf_tail)
                    and not _lf_tail.startswith('•')
                    and not _lf_tail.startswith('-')
                    and not _lf_tail.startswith('⚠')
                ):
                    _lf_closing.insert(0, _lf_pre_lines.pop())
                else:
                    break
            if _lf_closing:
                t = '\n'.join(_lf_pre_lines).rstrip() + '\n\n' + _lf_links.rstrip() + '\n\n' + '\n'.join(_lf_closing).strip()

        # Strip trailing emoji/spaces before checking ending (avoids "ครับ 😊ครับ")
        t_check = re.sub(r"[\U0001F300-\U0001FFFF\U00002600-\U000027BF\s]+$", "", t).strip()
        if not t_check.endswith("ครับ"):
            last_line = (t.split("\n")[-1] if "\n" in t else t).strip()
            if re.search(r"[a-zA-Z0-9/._\-]$", last_line):
                t = t + "\nครับ"
            else:
                t = t.rstrip(" .") + "ครับ"
        return t

    def _apply_practical_lint(self, text: str, kind: str) -> str:
        t = (text or "").strip()
        if not t:
            return t

        if kind in {"menu", "greet"}:
            t2 = self._META_TALK_RE.sub("", t).strip()
            return t2 or t

        if kind == "answer":
            # Answers skip strict length policy (which would replace long answers with a question).
            # Only apply meta-talk cleanup + newline normalization via _fallback_practical_answer.
            return self._fallback_practical_answer(t)

        if kind == "ask":
            # Split question text from numbered option lines BEFORE policy enforcement.
            # enforce_practical_policy must not see the option lines — they inflate line count
            # and trigger max_lines fallback, replacing the whole text with a generic question.
            lines = t.splitlines()
            opts_start = next(
                (i for i, ln in enumerate(lines) if re.match(r"^\d+[).]", ln.strip())),
                len(lines),
            )
            question_text = " ".join(ln.strip() for ln in lines[:opts_start] if ln.strip())
            opts_text = "\n".join(lines[opts_start:])

            # Apply policy only to the question part (not the options)
            if callable(enforce_practical_policy) and question_text:
                try:
                    out = enforce_practical_policy(question_text)
                    if isinstance(out, str):
                        question_text = out.strip() or question_text
                    elif isinstance(out, tuple) and len(out) == 2:
                        new_t, lint_meta = out
                        if isinstance(new_t, str) and new_t.strip():
                            old_len = len(question_text)
                            question_text = new_t.strip()
                            if old_len != len(question_text):
                                _LOG.info("[Practical/lint] Policy fallback triggered: old_len=%d new_len=%d", old_len, len(question_text))
                        if isinstance(lint_meta, dict) and lint_meta.get("ok") is False:
                            _LOG.warning("[Practical/lint] Policy validation failed: %s", lint_meta)
                    elif isinstance(out, dict):
                        new_t = out.get("text") or out.get("output") or out.get("result")
                        if isinstance(new_t, str) and new_t.strip():
                            question_text = new_t.strip()
                except Exception:
                    pass

            cleaned_q = self._fallback_single_question(question_text)
            return (cleaned_q + "\n" + opts_text).strip() if opts_text else cleaned_q

        # For non-ask: apply full practical policy enforcement
        if callable(enforce_practical_policy):
            try:
                out = enforce_practical_policy(t)
                if isinstance(out, str):
                    t = out.strip() or t
                elif isinstance(out, tuple) and len(out) == 2:
                    new_t, lint_meta = out
                    if isinstance(new_t, str) and new_t.strip():
                        old_len = len(t)
                        t = new_t.strip()
                        if old_len != len(t):
                            _LOG.info("[Practical/lint] Policy fallback triggered: old_len=%d new_len=%d", old_len, len(t))
                    if isinstance(lint_meta, dict) and lint_meta.get("ok") is False:
                        _LOG.warning("[Practical/lint] Policy validation failed: %s", lint_meta)
                        raise ValueError("practical_lint_failed")
                elif isinstance(out, dict):
                    new_t = out.get("text") or out.get("output") or out.get("result")
                    if isinstance(new_t, str) and new_t.strip():
                        t = new_t.strip()
                    ok = out.get("ok")
                    if ok is False:
                        raise ValueError("practical_lint_failed")
            except Exception:
                pass

        return t

    # Phase 3 config (unchanged)
    _PHASE3_MENU_HEADER = "ตอนนี้มีข้อมูลครบระดับหนึ่งแล้วครับ คุณอยากดูหัวข้อไหนก่อน?"
    _PHASE3_SLOT_KEY = "detail_section"
    _PHASE3_ALL = "ทั้งหมด"

    # ── Dynamic Clarification config ─────────────────────────────────────────
    # Checked in order; first diverging field triggers one clarification question.
    # filter_mode="chroma"       → re-retrieve with Chroma metadata_filter={filter_key: matched}
    # filter_mode="query_enrich" → append matched value to query string, no Chroma filter
    # alt_keys: also check these keys in collected_slots to skip if already known
    # max_value_len: skip values longer than this (noisy free-text fields)
    _CLARIFICATION_FIELDS: List[Dict[str, Any]] = [
        {
            "metadata_key":  "entity_type_normalized",
            "filter_key":    "entity_type_normalized",
            "collected_key": "entity_type",
            "filter_mode":   "chroma",
            "question_prefix": "ธุรกิจของคุณเป็นรูปแบบใดครับ?",
            "question_tmpl": "ธุรกิจของคุณเป็น{} หรือ {} ครับ?",
        },
        {
            "metadata_key":  "operation_by_department",
            "filter_key":    "operation_by_department",
            "collected_key": "operation_by_department",
            "filter_mode":   "query_enrich",
            "alt_keys":      ["operation_group", "operation_type", "operation_action"],
            "question_prefix": "ต้องการดำเนินการเรื่องใดครับ?",
            "question_tmpl": "",
            "max_value_len": 80,
        },
        {
            "metadata_key":  "registration_type",
            "filter_key":    "registration_type",
            "collected_key": "registration_type",
            "filter_mode":   "chroma",
            "question_prefix": "รูปแบบการจดทะเบียนเป็นแบบใดครับ?",
            "question_tmpl": "",
            "max_value_len": 30,
        },
        {
            "metadata_key":  "location",
            "filter_key":    "location",
            "collected_key": "location",
            "filter_mode":   "chroma",
            "question_prefix": "ร้านของคุณตั้งอยู่ในพื้นที่ใดครับ?",
            "question_tmpl": "ร้านของคุณตั้งอยู่ที่ {} หรือ {} ครับ?",
        },
        {
            "metadata_key":  "area_size",
            "filter_key":    "area_size",
            "collected_key": "shop_area_type",
            "filter_mode":   "chroma",
            "question_prefix": "ขนาดพื้นที่ร้านของคุณเป็นแบบใดครับ?",
            "question_tmpl": "พื้นที่ร้านของคุณ{} หรือ {} ครับ?",
            "max_value_len": 35,
        },
        {
            "metadata_key":  "department",
            "filter_key":    "department",
            "collected_key": "department",
            "filter_mode":   "chroma",
            "question_prefix": "ต้องการยื่นกับหน่วยงานใดครับ?",
            "question_tmpl": "",
            "max_value_len": 60,
        },
    ]

    # ── Iterative Retrieval config ────────────────────────────────────────────
    # Coverage fields checked after Round 2; missing ≥ N triggers Round 3 gap-fill.
    _COVERAGE_FIELDS: Dict[str, str] = {
        "operation_steps":          "ขั้นตอน",
        "fees":                     "ค่าธรรมเนียม",
        "identification_documents": "เอกสาร",
        "operation_duration":       "ระยะเวลา",
    }

    # Retrieval reuse/new-topic heuristic (unchanged)
    _TOKEN_SPLIT_RE = re.compile(r"[\s/,\-–—|]+", re.UNICODE)
    _FOLLOWUP_SHORT_RE = re.compile(
        r"^(แล้ว(ไง|ล่ะ)?|แล้ว(เอกสาร|ขั้นตอน|ค่าธรรมเนียม)?|ต่อไปล่ะ|มีอะไรบ้าง|ขอ(เอกสาร|ขั้นตอน|ค่าธรรมเนียม|ระยะเวลา|ช่องทาง))\s*$"
    )
    # "แค่ + generic-keyword" anywhere in a short query — use .search() not .match().
    # Catches: "ขอแค่เอกสารหน่อยค่ะ", "ถ้าขอแค่ขั้นตอนได้มั้ยคะ", "อยากทราบแค่ค่าธรรมเนียม"
    # regardless of prefix/suffix words — no tail enumeration needed.
    _SINGLE_ASPECT_RE = re.compile(
        r"แค่\s*(?:เอกสาร|ขั้นตอน|ค่าธรรมเนียม|ค่าใช้จ่าย|ระยะเวลา|ช่องทาง(?:ยื่น|ติดต่อ)?|แบบฟอร์ม|ลิงก์|ลิงค์)",
        re.IGNORECASE,
    )
    # "แล้วถ้า/แล้วเรื่อง/แล้วแค่ + keyword" continuation follow-up — use .search().
    # Catches: "แล้วถ้าเป็นเอกสารอ่ะคะ", "แล้วเรื่องค่าธรรมเนียม", "แล้วแค่ขั้นตอน"
    _THEN_ASPECT_RE = re.compile(
        r"แล้ว(?:ถ้า\s*)?(?:(?:เป็น|เรื่อง|แค่|ดู)\s*)?"
        r"(?:เอกสาร|ขั้นตอน|ค่าธรรมเนียม|ค่าใช้จ่าย|ระยะเวลา|ช่องทาง(?:ยื่น|ติดต่อ)?|แบบฟอร์ม|ลิงก์)",
        re.IGNORECASE,
    )
    # ── License-query scoring patterns (used in _license_query_score inside handle()) ──
    # Defined as class attributes so they are compiled once at class load, not per handle() call.
    _LQS_ABBREV_CHECK = [
        # User types abbreviation in query → boost the matching license_type.
        # Pattern 1: search in query; Pattern 2: search in license_type name.
        # ภป → ภาษีป้าย (bare abbreviation without form number)
        (re.compile(r"ภป\.?", re.IGNORECASE),
         re.compile(r"ภาษีป้าย", re.IGNORECASE)),
        # ภพ.20 / ภพ20 → ภาษีมูลค่าเพิ่ม (bare ภพ already in _LQS_VAT_RE for full-form queries)
        (re.compile(r"ภพ\.?20?|ภพ20", re.IGNORECASE),
         re.compile(r"ภาษีมูลค่าเพิ่ม", re.IGNORECASE)),
        # บจก → บริษัทจำกัด (correct abbreviation; old code had wrong บอจ)
        (re.compile(r"บจก\.?", re.IGNORECASE),
         re.compile(r"บริษัทจำกัด", re.IGNORECASE)),
        # หจก → ห้างหุ้นส่วนจำกัด
        (re.compile(r"หจก\.?", re.IGNORECASE),
         re.compile(r"ห้างหุ้นส่วนจำกัด", re.IGNORECASE)),
    ]
    # Negative lookbehind (?<![เโ]) prevents "เปิดบัญชี" from matching "ปิดบัญชี".
    _LQS_ACCOUNTING_RE = re.compile(
        r"ปิดงบ|(?<![เโ])ปิดบัญชี|ปิงบ|งบการเงิน|งบบัญชี|ทำบัญชี|จัดทำบัญชี|"
        r"สำนักงานบัญชี|ผู้ทำบัญชี|ผู้สอบบัญชี|ตรวจสอบบัญชี",
        re.IGNORECASE,
    )
    _LQS_BANKING_RE = re.compile(
        r"เปิดบัญชี|บัญชีธนาคาร|บัญชีออมทรัพย์|บัญชีกระแสรายวัน|บัญชีรับเงิน",
        re.IGNORECASE,
    )
    _LQS_EMPLOYER_RE = re.compile(
        r"ขึ้นทะเบียนนายจ้าง|ทะเบียนนายจ้าง|ประกันสังคม.*นายจ้าง|นายจ้าง.*ประกันสังคม",
        re.IGNORECASE,
    )
    _LQS_HIRING_RE = re.compile(
        r"จัดหาพนักงาน|รับสมัครพนักงาน|จ้างพนักงาน|หาพนักงาน|รับพนักงาน",
        re.IGNORECASE,
    )
    _LQS_SAN_RE = re.compile(
        r"\bSAN\b|\bSAN\s*PLUS\b|รับรองมาตรฐาน|มาตรฐานร้านอาหาร|สุขาภิบาลอาหาร.*SAN|SAN.*สุขาภิบาล",
        re.IGNORECASE,
    )
    _LQS_QR_KEYWORDS_RE = re.compile(
        r"\bQR\b|QR\s*Payment|QR\s*Code|คิวอาร์|ชำระ.*ออนไลน์|สมัคร.*QR|QR.*สมัคร",
        re.IGNORECASE,
    )
    _LQS_VAT_RE = re.compile(
        r"\bVAT\b|\bภพ\.?20\b|\bภพ20\b|จด\s*ภาษีมูลค่าเพิ่ม|สมัคร\s*VAT|ขึ้นทะเบียน\s*VAT",
        re.IGNORECASE,
    )
    _LQS_SOCIAL_RE = re.compile(r"ประกันสังคม", re.IGNORECASE)
    _LQS_INSURED_RE = re.compile(
        r"ผู้ประกันตน|ทะเบียนผู้ประกัน|เงินสมทบ|แจ้งเข้าทำงาน|แจ้งออกจากงาน|ลูกจ้าง",
        re.IGNORECASE,
    )
    _LQS_HEALTH_CERT_RE = re.compile(
        r"ใบรับรองแพทย์|9\s*โรค|สณ\.?11|ตรวจสุขภาพ.*ร้านอาหาร|สุขภาพ.*ผู้ประกอบ",
        re.IGNORECASE,
    )
    _LQS_LIQUOR_RE = re.compile(
        r"สุรา|เหล้า|เบียร์|แอลกอฮอล์|จำหน่ายสุรา|ขายสุรา|ใบอนุญาต.*สุรา",
        re.IGNORECASE,
    )
    _LQS_SIGN_TAX_RE = re.compile(r"ภาษีป้าย|ป้ายร้านอาหาร", re.IGNORECASE)
    _LQS_QR_PAYMENT_RE = re.compile(
        r"qr|คิวอาร์|คิวอาเพ|คิวอาเอพีไอ|payment|biller.?id|merchant.?id|"
        r"สมัคร.*ชำระ|ชำระ.*ออนไลน์|ลงทะเบียน.*ชำระ",
        re.IGNORECASE,
    )

    # ── Inline handle() patterns compiled once here ──────────────────────────────
    _TC_QUERY_RE = re.compile(r"ตัดรอบ|cut.?off|เงื่อนไขและหลักเกณฑ์|หลักเกณฑ์การให้บริการ")
    _TC_TIME_RE = re.compile(r"\d{1,2}[.:]\d{2}\s*น\.?")
    _NO_FORMS_Q_RE = re.compile(
        r"บทลงโทษ|ค่าปรับ|โทษ(?![ษ])|ผิดกฎ|ฝ่าฝืน|จะโดน|โดนปรับ|โทษปรับ"
        r"|คืออะไร|หมายความว่า|ความหมาย|นิยาม"
    )

    def __init__(self, retriever):
        self.retriever = retriever
        self._topic_menu_cache: Optional[List[str]] = None
        self.llm_greet_call = self._default_greet_llm_call()
        self._lqs_license_llm_call = self._default_lqs_license_llm_call()
        self._satisfaction_llm_call = self._default_satisfaction_llm_call()
        self._dont_know_llm_call = self._default_dont_know_llm_call()
        self._short_followup_llm_call = self._default_short_followup_llm_call()
        self._practical_legal_q_llm_call = self._default_practical_legal_q_llm_call()
        self._init_llm()

    def _init_llm(self):
        model_name = getattr(conf, "OPENROUTER_MODEL_PRACTICAL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=getattr(conf, "TEMPERATURE_PRACTICAL", 0.2),
            max_tokens=getattr(conf, "MAX_TOKENS_PRACTICAL", 4000),
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    # Normalization / detectors
    def _normalize_for_intent(self, s: str) -> str:
        t = (s or "").strip().lower()
        t = re.sub(r"[!！?？。,，]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
        return t

    def _looks_like_greeting(self, s: str) -> bool:
        raw = (s or "").strip()
        if not raw:
            return True
        t = self._normalize_for_intent(raw)
        if self._EN_GREET_RE.match(t):
            return True
        if self._TH_WATDEE_RE.match(t):
            return True
        if self._TH_SAWASDEE_RE.match(t):
            return True
        if self._TH_DEE_RE.match(t) and ("ไหม" not in t and "?" not in t):
            return True
        return False

    def _looks_like_legal_question(self, s: str, state=None) -> bool:
        t = self._normalize_for_intent(s)
        if self._LEGAL_SIGNAL_RE.search(t):
            return True
        # LLM fallback: catches "ขอข้อมูลค่าใช้จ่าย", "ต้องติดต่อที่ไหน" that _LEGAL_SIGNAL_RE misses.
        if state is not None and len(t) >= 4:
            return self._practical_legal_q_llm_check(t, state)
        return False

    def _satisfaction_llm_check(self, user_text: str, state=None) -> bool:
        """
        LLM fallback: did user express satisfaction/done-ness?
        Called when _THANKS_RE and _OK_RE both miss. _OK_RE is anchored (^...$)
        and cannot match extra modifiers like "เคลียร์มากๆ ครับ".
        High threshold (0.80) — false positive = ending conversation when user still needs help.
        Caches in state.context['_satisfaction_llm_cache'] when state is available.
        """
        q = (user_text or "").strip()
        if not q or len(q) < 3 or len(q) > 80:
            return False
        if self._LEGAL_SIGNAL_RE.search(q):
            return False
        cache = (state.context or {}).setdefault("_satisfaction_llm_cache", {}) if state is not None else {}
        cache_key = q[:60]
        if cache_key in cache:
            return cache[cache_key]
        try:
            res = self._satisfaction_llm_call(q) or {}
            conf_val = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_satisfied")) and conf_val >= 0.80
        except Exception as _e:
            _LOG.warning("[Practical/satisfaction] LLM check failed: %s", _e)
            return False  # don't cache transient errors — allow retry on next call
        if result:
            _LOG.info("[Practical/satisfaction] LLM detected satisfaction: %r (conf=%.2f)", q[:50], conf_val)
        cache[cache_key] = result
        return result

    def _dont_know_or_asking_types_llm_check(self, user_text: str, state=None) -> bool:
        """
        LLM fallback: user expresses uncertainty (ไม่รู้/ไม่แน่ใจ) or asks to see available types.
        Called when _DONT_KNOW_RE (anchored ^...$) and _ASK_TYPES_RE both miss.
        Context: only fired when last bot message contained "ประเภท".
        Caches in state.context['_dont_know_llm_cache'] when state is available.
        """
        q = (user_text or "").strip()
        if not q or len(q) < 2 or len(q) > 80:
            return False
        cache = (state.context or {}).setdefault("_dont_know_llm_cache", {}) if state is not None else {}
        cache_key = q[:60]
        if cache_key in cache:
            return cache[cache_key]
        conf_val = 0.0
        try:
            res = self._dont_know_llm_call(q) or {}
            conf_val = float(res.get("confidence") or 0.0)
            result = (bool(res.get("is_dont_know")) or bool(res.get("is_asking_types"))) and conf_val >= 0.75
        except Exception as _e:
            _LOG.warning("[Practical/dont_know] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Practical/dont_know] LLM detected dont_know/ask_types: %r (conf=%.2f)", q[:50], conf_val)
        cache[cache_key] = result
        return result

    def _short_followup_llm_check(self, user_text: str, state=None) -> bool:
        """
        LLM fallback: is this a short continuation question referring to the ongoing topic?
        Called when _FOLLOWUP_SHORT_RE, _SINGLE_ASPECT_RE, _THEN_ASPECT_RE all miss.
        High threshold (0.80) — false positive = reusing stale docs for a new topic = wrong answer.
        Guard: only fire for short text (≤ 80 chars).
        Caches in state.context['_short_followup_llm_cache'] when state is available.
        """
        q = (user_text or "").strip()
        if not q or len(q) > 80:
            return False
        cache = (state.context or {}).setdefault("_short_followup_llm_cache", {}) if state is not None else {}
        cache_key = q[:60]
        if cache_key in cache:
            return cache[cache_key]
        conf_val = 0.0
        try:
            res = self._short_followup_llm_call(q) or {}
            conf_val = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_followup")) and conf_val >= 0.80
        except Exception as _e:
            _LOG.warning("[Practical/short_followup] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Practical/short_followup] LLM detected short followup: %r (conf=%.2f)", q[:50], conf_val)
        cache[cache_key] = result
        return result

    def _practical_legal_q_llm_check(self, user_text: str, state=None) -> bool:
        """
        LLM fallback: is this a legal question? Used in _should_retrieve_new_topic.
        Called when practical.py's _LEGAL_SIGNAL_RE misses.
        Caches in state.context['_practical_legal_q_llm_cache'] when state is available.
        """
        q = (user_text or "").strip()
        if not q or len(q) < 4:
            return False
        cache = (state.context or {}).setdefault("_practical_legal_q_llm_cache", {}) if state is not None else {}
        cache_key = q[:80]
        if cache_key in cache:
            return cache[cache_key]
        conf_val = 0.0
        try:
            res = self._practical_legal_q_llm_call(q) or {}
            conf_val = float(res.get("confidence") or 0.0)
            result = bool(res.get("is_legal")) and conf_val >= 0.75
        except Exception as _e:
            _LOG.warning("[Practical/legal_q] LLM check failed: %s", _e)
            result = False
        if result:
            _LOG.info("[Practical/legal_q] LLM detected legal question: %r (conf=%.2f)", q[:50], conf_val)
        cache[cache_key] = result
        return result

    def _looks_like_satisfaction(self, s: str, state=None) -> bool:
        t = self._normalize_for_intent(s)
        if not t:
            return False
        if self._THANKS_RE.search(t) or self._OK_RE.match(t):
            return True
        # LLM fallback: catches "เคลียร์มากๆ ครับ", "เข้าใจดีมากเลย", "เพียงพอแล้วนะ"
        # _OK_RE is anchored (^...$) — misses phrases with extra modifiers.
        if len(t) <= 80 and not self._LEGAL_SIGNAL_RE.search(t):
            return self._satisfaction_llm_check(t, state)
        return False

    def _looks_like_asking_for_reference(self, s: str) -> bool:
        """Detect if user explicitly asks for research reference links."""
        t = self._normalize_for_intent(s)
        if not t:
            return False
        # Match patterns like: "อ้างอิงคืออะไร", "ขออ้างอิง", "มีอ้างอิงไหม", "reference"
        return bool(re.search(r"(อ้างอิง|reference|research|เอกสารอ้างอิง|แหล่งอ้างอิง)", t, re.IGNORECASE))

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

    def _is_short_followup(self, user_text: str, state=None) -> bool:
        t = (user_text or "").strip()
        if not t:
            return True
        n = self._normalize_for_intent(t)
        toks = self._tokenize_loose(n)
        if len(toks) <= 3 and len(n) <= 18:
            return True
        if self._FOLLOWUP_SHORT_RE.match(n):
            return True
        # "แค่ + keyword" or "แล้วถ้า/เรื่อง + keyword" anywhere in a short query.
        # Length guard (≤ 60) prevents false positives on longer topic-switch sentences.
        if len(n) <= 60 and (self._SINGLE_ASPECT_RE.search(n) or self._THEN_ASPECT_RE.search(n)):
            return True
        # LLM fallback: catches other short continuation phrasings regex misses.
        # High threshold (0.80) — false positive = reusing stale docs for a new topic.
        return self._short_followup_llm_check(n, state)

    def _should_retrieve_new_topic(self, state: ConversationState, user_text: str) -> bool:
        q = (user_text or "").strip()
        if not q:
            return False

        has_docs = bool(getattr(state, "current_docs", None))
        if not has_docs:
            return True

        last_q = (getattr(state, "last_retrieval_query", None) or "").strip()
        if not last_q:
            return True

        if self._is_short_followup(q, state):
            return False

        overlap = self._topic_overlap_ratio(last_q, q)
        if self._looks_like_legal_question(q, state) and overlap < 0.22:
            return True

        return False

    # Slot + choices helpers (unchanged)
    def _format_numbered_options(self, options: List[str], max_items: int = 9) -> str:
        opts = [str(x).strip() for x in (options or []) if str(x).strip()]
        opts = opts[:max_items]
        return "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(opts)])

    def _parse_selection_numbers(self, user_text: str, options_count: int) -> List[int]:
        t = (user_text or "").strip().lower()
        if not t:
            return []

        m = re.search(r"\b(\d+)\s*-\s*(\d+)\b", t)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            out = [x for x in range(a, b + 1) if 1 <= x <= options_count]
            seen, uniq = set(), []
            for x in out:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            return uniq

        if options_count <= 9 and re.fullmatch(r"\d{2,}", t):
            out = []
            for ch in t:
                n = int(ch)
                if 1 <= n <= options_count:
                    out.append(n)
            seen, uniq = set(), []
            for x in out:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            return uniq

        nums = re.findall(r"\d+", t)
        out = []
        for s2 in nums:
            n = int(s2)
            if 1 <= n <= options_count:
                out.append(n)
        seen, uniq = set(), []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _extract_numbered_options(self, text: str, max_items: int = 9) -> List[str]:
        if not text:
            return []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        pairs: List[Tuple[int, str]] = []
        for ln in lines:
            m = self._NUM_OPTION_LINE_RE.match(ln)
            if not m:
                continue
            idx = int(m.group(1))
            label = (m.group(2) or "").strip()
            if idx <= 0 or not label:
                continue
            pairs.append((idx, label))
        if not pairs:
            return []
        pairs.sort(key=lambda x: x[0])
        return [lbl for _, lbl in pairs][:max_items]

    def _infer_slot_key_from_question(self, question: str, options: list | None = None) -> str:
        q = self._normalize_for_intent(question)
        opts_combined = " ".join(str(o) for o in (options or []))
        # Location check FIRST — จังหวัด/เขต are strong signals for location, not area size
        if "จังหวัด" in q or ("เขต" in q and "พื้นที่" not in q) or "เทศบาล" in q:
            return "location_scope"
        if "ตารางเมตร" in q or ("พื้นที่" in q and "จังหวัด" not in q):
            return "area_size"
        # Check question text OR options for entity_type signals
        if ("บุคคลธรรมดา" in q or "นิติบุคคล" in q or "นิติ" in q
                or "บุคคลธรรมดา" in opts_combined or "นิติบุคคล" in opts_combined):
            return "entity_type"
        if "ขายสุรา" in q or "แอลกอฮอล์" in q:
            return "alcohol_business"
        if "topic" in q or "หัวข้อ" in q:
            return "topic"
        if self._PHASE3_MENU_HEADER[:10] in question:
            return self._PHASE3_SLOT_KEY
        return "choice"

    # Pending-slot recovery (OPT-IN + OWNER-GATED)
    def _maybe_recover_pending_slot_from_last_bot(self, state: ConversationState, user_text: str) -> None:
        """
        Recovery is disabled by default to prevent hijack.
        Enable only if explicitly requested:
          state.context["allow_practical_pending_recover"] = True

        PLUS owner-gate:
          - only recover if state.context["last_bot_owner"] == "practical"
        """
        ctx = state.context or {}
        if not bool(ctx.get("allow_practical_pending_recover", False)):
            return

        if (ctx.get("last_bot_owner") or "").strip().lower() != "practical":
            return

        pending = ctx.get("pending_slot")
        if isinstance(pending, dict) and pending.get("options"):
            return

        if not user_text or not self._LIKELY_SELECTION_RE.match(user_text.strip()):
            return

        last_bot = next((m.get("content", "") for m in reversed(state.messages or []) if m.get("role") == "assistant"), "")
        opts = self._extract_numbered_options(last_bot)
        if not opts:
            return

        slot_key = (
            self._PHASE3_SLOT_KEY
            if self._PHASE3_MENU_HEADER in (last_bot or "")
            else ("topic" if "เกี่ยวกับร้านอาหาร" in (last_bot or "") else self._infer_slot_key_from_question(last_bot))
        )
        allow_multi = True if slot_key in {self._PHASE3_SLOT_KEY, "choice"} else False
        ctx["pending_slot"] = {"key": slot_key, "options": opts, "allow_multi": allow_multi}
        state.context = ctx

    def _consume_pending_slot_from_user(self, state: ConversationState, user_text: str) -> Optional[str]:
        ctx = state.context or {}
        pending = ctx.get("pending_slot")
        if not isinstance(pending, dict):
            return None

        key = (pending.get("key") or "").strip()
        options = pending.get("options")
        allow_multi = bool(pending.get("allow_multi", False))

        if not key:
            ctx.pop("pending_slot", None)
            state.context = ctx
            return "FILLED"

        slots = ctx.setdefault("slots", {})
        if key in slots and slots[key] not in (None, "", [], {}):
            ctx.pop("pending_slot", None)
            state.context = ctx
            return "FILLED"

        low = self._normalize_for_intent(user_text)

        if isinstance(options, list) and options and allow_multi:
            if re.search(r"(ทั้งหมด|all\b|ทุกข้อ|ทุกอย่าง)", low):
                slots[key] = [str(x) for x in options if str(x).strip() and str(x).strip() != self._PHASE3_ALL]
                ctx.pop("pending_slot", None)
                state.context = ctx
                return "FILLED"

        if isinstance(options, list) and options:
            nums = self._parse_selection_numbers(user_text, options_count=len(options))
            chosen = [str(options[n - 1]) for n in nums if 1 <= n <= len(options)]

            if chosen:
                if self._PHASE3_ALL in chosen and key == self._PHASE3_SLOT_KEY:
                    slots[key] = [str(x) for x in options if str(x).strip() and str(x).strip() != self._PHASE3_ALL]
                else:
                    slots[key] = chosen if allow_multi else chosen[0]
                ctx.pop("pending_slot", None)
                state.context = ctx
                return "FILLED"

            matched = [str(opt) for opt in options if str(opt) and str(opt) in user_text]
            if matched:
                if self._PHASE3_ALL in matched and key == self._PHASE3_SLOT_KEY:
                    slots[key] = [str(x) for x in options if str(x).strip() and str(x).strip() != self._PHASE3_ALL]
                else:
                    slots[key] = matched if allow_multi else matched[0]
                ctx.pop("pending_slot", None)
                state.context = ctx
                return "FILLED"

            if key == "topic" and user_text.strip() and not self._LIKELY_SELECTION_RE.match(user_text.strip()):
                if self._looks_like_legal_question(user_text):
                    ctx.pop("pending_slot", None)
                    state.context = ctx
                    return "BYPASS"
                return "INVALID"

            if key == self._PHASE3_SLOT_KEY:
                return "INVALID"

            if user_text.strip() and not self._LIKELY_SELECTION_RE.match(user_text.strip()):
                slots[key] = user_text.strip()
                ctx.pop("pending_slot", None)
                state.context = ctx
                return "FILLED"

            return "INVALID"

        if user_text.strip():
            slots[key] = user_text.strip()
            ctx.pop("pending_slot", None)
            state.context = ctx
            return "FILLED"

        return "INVALID"

    # Topic menu from metadata (NO hallucination) + sanitation
    # H2: Multi-query pool for richer topic menu (4 queries = broader coverage)
    _TOPIC_POOL_QUERIES = [
        "ใบอนุญาต เปิดร้านอาหาร",
        "ภาษี VAT จดทะเบียนพาณิชย์",
        "สุขาภิบาลอาหาร ประกันสังคม",
        "เอกสาร ค่าธรรมเนียม ขั้นตอน",
    ]

    def _build_topic_menu_from_corpus(self) -> List[str]:
        freq: Dict[str, int] = {}

        def _add(v: Any) -> None:
            s = self._sanitize_topic_label(str(v) if v is not None else "")
            if len(s) < self._TOPIC_MIN_LEN:
                return
            freq[s] = freq.get(s, 0) + 1

        seen_doc_ids: set = set()
        for q in self._TOPIC_POOL_QUERIES:
            try:
                docs = self._retrieve_docs(q)
            except Exception:
                continue
            for d in docs:
                md = d.get("metadata", {}) or {}
                doc_id = md.get("doc_id") or md.get("row_id") or id(d)
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                _add(md.get("license_type"))
                _add(md.get("department"))
                _add(md.get("operation_topic"))

        items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        menu = [k for k, _ in items][:6]
        return menu

    def _get_topic_menu(self, state: ConversationState) -> List[str]:
        if self._topic_menu_cache:
            return self._topic_menu_cache

        cached = (state.context or {}).get("topic_menu")
        if isinstance(cached, list) and all(isinstance(x, str) for x in cached) and cached:
            self._topic_menu_cache = cached
            return cached

        menu = self._build_topic_menu_from_corpus()
        if not menu:
            menu = ["ใบอนุญาต/การเปิดร้าน", "ภาษี/VAT", "จดทะเบียนพาณิชย์", "สุขาภิบาลอาหาร"]

        state.context = state.context or {}
        state.context["topic_menu"] = menu
        self._topic_menu_cache = menu
        return menu

    def _build_multi_topic_summary_menu(self, state: ConversationState, topics: List[str]) -> str:
        """For 4+ topics: build a brief summary of each topic (with connection note when related)
        and a numbered menu. Sets pending_slot so the user's selection is consumed normally."""
        topic_set = set(topics)
        connection_note = ""
        best_overlap = 1
        for group_set, note in _RELATED_TOPIC_GROUPS:
            overlap = len(topic_set & group_set)
            if overlap >= 2 and overlap > best_overlap:
                best_overlap = overlap
                connection_note = note

        header = f"มี {len(topics)} เรื่องที่คุณถามถึงค่ะ"
        if connection_note:
            header += f" {connection_note}"

        summary_lines = []
        for lt in topics:
            desc = _TOPIC_DESC_MAP.get(lt, "")
            summary_lines.append(f"• {lt}" + (f" — {desc}" if desc else ""))

        state.context["pending_slot"] = {
            "key": "multi_topic_select",
            "options": topics,
            "allow_multi": False,
        }

        menu_str = self._format_numbered_options(topics)
        summary = "\n".join(summary_lines)
        return f"{header}\n\n{summary}\n\nต้องการรายละเอียดเรื่องไหนก่อนคะ?\n{menu_str}"

    def _reply_greeting_with_choices(self, state: ConversationState, kind: str = "greet") -> str:
        """
        Practical greeting/menu renderer.

        FIX: If Supervisor owns menu, do NOT show menu / do NOT set pending_slot.
        (Still returns a short prefix so the system can respond naturally.)
        """
        state.context = state.context or {}

        menu = self._get_topic_menu(state)
        streak = int(state.context.get("greet_streak", 0) or 0) + 1
        state.context["greet_streak"] = streak
        prefix = self._pick_greet_prefix(kind=kind, menu=menu, greet_streak=streak).strip()

        if self._supervisor_owns_menu(state):
            return self._apply_practical_lint(prefix, kind="greet")

        # menu once per session
        if state.context.get("main_menu_shown"):
            return self._apply_practical_lint(prefix, kind="greet")

        state.context["pending_slot"] = {"key": "topic", "options": menu, "allow_multi": False}
        state.context["main_menu_shown"] = True

        msg = (prefix.rstrip() + "\n" + self._format_numbered_options(menu)).strip()
        return self._apply_practical_lint(msg, kind="greet")

    def _reply_satisfaction(self, state: ConversationState) -> str:
        # Uses same guard in _reply_greeting_with_choices()
        return self._reply_greeting_with_choices(state, kind="thanks")

    # LLM + retrieval
    def _call_llm_json(self, prompt: str, max_retries: int = 2, state: Optional[ConversationState] = None) -> dict:
        last_err = None
        for _ in range(max_retries):
            try:
                resp = llm_invoke(self.llm, [SystemMessage(content=SYSTEM_PROMPT_PRACTICAL), HumanMessage(content=prompt)], logger=_LOG, label="Practical/json", state=state)
                text = extract_llm_text(resp).strip()

                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                try:
                    obj = json.loads(text)
                except json.JSONDecodeError as _jde:
                    # Pre-repair: escape literal newlines/tabs inside JSON string values.
                    # LLM with large system prompt sometimes writes real \n inside "answer"
                    # instead of \\n, causing json.loads to fail.
                    _repaired_ok = False
                    try:
                        obj = json.loads(_repair_json_strings(text))
                        _LOG.info("[Practical/json] JSON string repair applied — parse OK")
                        _repaired_ok = True
                    except json.JSONDecodeError:
                        pass
                    if not _repaired_ok:
                        # Three-pass rescue: extract "answer" field content from broken JSON.
                        # Pass 1 — strict: stop at first unescaped quote (standard JSON)
                        _ans_match = re.search(
                            r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)',
                            text, re.DOTALL
                        )
                        _rescued_ans = _ans_match.group(1) if _ans_match else ""
                        _start_m = re.search(r'"answer"\s*:\s*"', text)
                        if _start_m:
                            _ans_field_start = _start_m.end()
                            if _ans_field_start > _jde.pos:
                                # Pass 3 — error is in a field BEFORE "answer" (e.g. "analysis").
                                # Scan character-by-character from answer field start, tracking
                                # escape sequences, to extract the full answer string.
                                _p3_pos = _ans_field_start
                                _p3_chars: list = []
                                while _p3_pos < len(text):
                                    _c = text[_p3_pos]
                                    if _c == '\\' and _p3_pos + 1 < len(text):
                                        _p3_chars.append(text[_p3_pos:_p3_pos + 2])
                                        _p3_pos += 2
                                    elif _c == '"':
                                        break
                                    else:
                                        _p3_chars.append(_c)
                                        _p3_pos += 1
                                _p3_ans = ''.join(_p3_chars)
                                if len(_p3_ans) > len(_rescued_ans):
                                    _rescued_ans = _p3_ans
                            else:
                                # Pass 2 — error is IN or AFTER "answer": extract up to error pos.
                                _p2_threshold = max(300, len(_rescued_ans) * 3)
                                if _jde.pos > _p2_threshold:
                                    _raw = text[_ans_field_start:_jde.pos]
                                    _raw = re.sub(r'(?<!\\)"', ' ', _raw)
                                    _raw = re.sub(r'[,}\]\\]+\s*$', '', _raw).strip()
                                    if len(_raw) > len(_rescued_ans):
                                        _rescued_ans = _raw
                        if _rescued_ans:
                            _rescued_ans = _rescued_ans.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\").strip()
                            _LOG.warning(
                                "[Practical/json] JSONDecodeError at char %d — rescued answer via regex (%d chars)",
                                _jde.pos, len(_rescued_ans),
                            )
                            return {
                                "input_type": "new_question",
                                "analysis": "json_repair",
                                "action": "answer",
                                "execution": {"answer": _rescued_ans, "context_update": {}},
                            }
                        raise  # no rescue possible — fall through to retry

                # DEBUG LOG: show raw LLM JSON response before processing
                if isinstance(obj, dict):
                    action = obj.get("action", "?")
                    exec_data = obj.get("execution", {})
                    q = (exec_data.get("question") or "") if isinstance(exec_data, dict) else ""
                    _LOG.debug("[Practical/json] LLM response: action=%r question=%r", action, q[:100])

                return obj if isinstance(obj, dict) else {}
            except Exception as e:
                # ถ้า LengthFinishReasonError → retry ไม่ช่วย (input เดิม = ผลเดิม) → break ทันที
                if "LengthFinishReasonError" in type(e).__name__ or "LengthFinishReason" in str(e)[:80]:
                    _LOG.warning("[Practical/json] LengthFinishReasonError — max_tokens น้อยเกินไป, skip retry")
                    last_err = e
                    break
                import traceback as _tb
                _LOG.warning("[Practical/json] exception (attempt): %s\n%s", e, _tb.format_exc())
                last_err = e
                continue

        if last_err:
            _LOG.warning("[Practical] LLM JSON parse failed: %s", last_err)

        # BUG-A fix: returning action='ask' with empty question would pop topic_slot_queue
        # and show a wrong context menu. Use action='answer' with a safe fallback message instead.
        return {
            "input_type": "new_question",
            "analysis": "Parse error",
            "action": "answer",
            "execution": {"answer": "ขออภัยครับ ระบบประมวลผลคำถามไม่สำเร็จ กรุณาลองถามใหม่อีกครั้งครับ", "context_update": {}},
        }

    def _lqs_license_type_fallback(self, query: str, lt_candidates: List[str], state,
                                    min_confidence: float = 0.70) -> Optional[str]:
        """LLM fallback for LQS: ask Haiku which license type best matches the query.
        Returns the license_type string (must be in lt_candidates) or None.
        Caches result in state.context to avoid repeated calls for the same query.
        min_confidence: 0.70 when regex score=0 (no signal), 0.80 when score 1-4 (weak signal)."""
        if not query or not lt_candidates:
            return None
        cache = (state.context or {}).get("_lqs_lt_llm_cache") or {}
        cache_key = query.strip()[:80]
        if cache_key in cache:
            cached = cache[cache_key]
            return cached if cached in lt_candidates else None
        valid_lt_set = set(lt_candidates)
        conf_val = 0.0
        try:
            result = self._lqs_license_llm_call(query, lt_candidates)
            lt_name = (result.get("license_type") or "").strip()
            conf_val = float(result.get("confidence") or 0)
            matched = lt_name if (lt_name and lt_name in valid_lt_set and conf_val >= min_confidence) else None
        except Exception as _e:
            _LOG.warning("[Practical] LQS LLM fallback error: %s", _e)
            matched = None
        if state.context is None:
            state.context = {}
        cache[cache_key] = matched or ""
        state.context["_lqs_lt_llm_cache"] = cache
        if matched:
            _LOG.info("[Practical] LQS LLM fallback: %r → %r (conf=%.2f, min=%.2f)", query[:50], matched, conf_val, min_confidence)
        return matched

    def _retrieve_docs(self, query: str, metadata_filter: Optional[Dict[str, Any]] = None, max_docs: Optional[int] = None, slot_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        import time
        start = time.time()

        max_docs = max_docs if max_docs is not None else int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 8))
        max_chars = getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700)

        # Query expansion: Thai/English synonym bridging — patterns defined in utils/query_synonyms.py
        _expansions: list = []
        for pattern, expansion in SYNONYM_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE) and expansion not in _expansions:
                _expansions.append(expansion)
        expanded_query = (query + " " + " ".join(_expansions)).strip() if _expansions else query
        if _expansions:
            _LOG.debug("[Practical] query expanded: %r → %r", query[:50], expanded_query[:80])

        # Append collected slot values so embedding benefits from full user context.
        # Only append values not already present in the expanded query (avoids duplication).
        # Skip for calls from supervisor where enriched_q already carries slot context.
        if slot_context:
            _sc_parts = []
            for _sc_key in ("entity_type", "registration_type"):
                _sc_val = str(slot_context.get(_sc_key) or "").strip()
                if _sc_val and _sc_val not in expanded_query:
                    _sc_parts.append(_sc_val)
            if _sc_parts:
                expanded_query = (expanded_query + " " + " ".join(_sc_parts)).strip()
                _LOG.debug("[Practical] slot context appended to query: %r", expanded_query[:80])

        vectorstore = getattr(self.retriever, "vectorstore", None)

        _hybrid_enabled = getattr(conf, "HYBRID_SEARCH_ENABLED", False)
        _rrf_k = int(getattr(conf, "HYBRID_RRF_K", 60))

        def _scored_search(q: str, k: int, flt: Optional[dict] = None) -> list:
            """Dense search (+ BM25 via RRF when HYBRID_SEARCH_ENABLED=true). Attaches _sim to metadata."""
            # Entity-neutral docs (entity_type_normalized='') apply to ALL entity types.
            # Exact-match filter excludes them — widen to $in to include them.
            # Chroma requires $and when multiple keys AND any value is an operator dict.
            if flt and isinstance(flt.get("entity_type_normalized"), str) and flt["entity_type_normalized"]:
                _et_val = flt["entity_type_normalized"]
                _rest = {k: v for k, v in flt.items() if k != "entity_type_normalized"}
                _et_cond = {"entity_type_normalized": {"$in": [_et_val, ""]}}
                if _rest:
                    flt = {"$and": [_et_cond] + [{k: v} for k, v in _rest.items()]}
                else:
                    flt = _et_cond
            if vectorstore is None:
                return list(self.retriever.invoke(q))
            try:
                if _hybrid_enabled:
                    from utils.hybrid_retriever import hybrid_scored_search
                    pairs = hybrid_scored_search(vectorstore, q, k=k, metadata_filter=flt, rrf_k=_rrf_k)
                else:
                    kwargs: dict = {"k": k}
                    if flt:
                        kwargs["filter"] = flt
                    pairs = vectorstore.similarity_search_with_relevance_scores(q, **kwargs)
                result = []
                for _d, _s in pairs:
                    if _s is not None:
                        _d.metadata["_sim"] = float(_s)
                    result.append(_d)
                return result
            except Exception as _ex:
                _LOG.debug("[Practical] scored_search failed (%s), trying similarity_search with filter", _ex)
                try:
                    if flt:
                        return vectorstore.similarity_search(q, k=k, filter=flt)
                except Exception as _ex2:
                    _LOG.debug("[Practical] similarity_search with filter also failed (%s), using basic", _ex2)
                return list(self.retriever.invoke(q))

        top_k = int(getattr(conf, "RETRIEVAL_TOP_K", 8))

        if metadata_filter:
            if vectorstore is not None:
                try:
                    docs = _scored_search(query, max_docs, metadata_filter)
                    _flt_str = " | ".join(f"{k}={v!r}" for k, v in (metadata_filter or {}).items())
                    logger.log_with_data("info", f"retrieve(filtered) query={query[:50]!r} filter=[{_flt_str}] found={len(docs)}", {
                        "action": "filtered_retrieval",
                        "query": query[:60],
                        "filter": str(metadata_filter),
                        "docs_found": len(docs),
                        "persona": "practical"
                    })
                    if not docs:
                        # Try same filter with larger k before going unfiltered.
                        # This preserves entity/license filter intent — avoids mixing
                        # irrelevant entity types into the result.
                        docs = _scored_search(query, top_k, metadata_filter)
                    elif len(docs) < max(2, max_docs // 2):
                        # Got some but fewer than half requested — retry with 2× k to improve coverage
                        _extra = _scored_search(query, top_k * 2, metadata_filter)
                        _seen_h = {hash((getattr(d, "page_content", "") or "")[:80]) for d in docs}
                        for _xd in _extra:
                            _xh = hash((getattr(_xd, "page_content", "") or "")[:80])
                            if _xh not in _seen_h:
                                _seen_h.add(_xh)
                                docs.append(_xd)
                        _LOG.info("[Practical] partial-filtered retry: %d → %d docs", len(docs) - len(_extra), len(docs))
                    if not docs:
                        # Intermediate fallback: strip location/area_size from the filter,
                        # keep license_type. Handles topics where location exists only in doc
                        # content (e.g. EDC timing "8-14 วัน กทม.") not in metadata field.
                        _lt_only_filter: dict | None = None
                        if isinstance(metadata_filter, dict):
                            _and_parts = metadata_filter.get("$and") or []
                            if _and_parts:
                                _loc_keys = {"location", "area_size"}
                                _has_loc_key = any(_loc_keys & set(_p.keys()) for _p in _and_parts)
                                _lt_parts = [_p for _p in _and_parts if not (_loc_keys & set(_p.keys()))]
                                if _has_loc_key and _lt_parts:
                                    _lt_only_filter = _lt_parts[0] if len(_lt_parts) == 1 else {"$and": _lt_parts}
                        if _lt_only_filter:
                            docs = _scored_search(query, top_k, _lt_only_filter)
                            if docs:
                                _LOG.info("[Practical] location-stripped fallback: %d docs (location not in metadata for this topic)", len(docs))
                    if not docs:
                        logger.log_with_data("warning", "No documents matched metadata filter — falling back to unfiltered retrieval with expanded query", {
                            "action": "filtered_retrieval_empty_fallback",
                            "filter": str(metadata_filter),
                            "fallback_query": expanded_query[:80],
                        })
                        docs = _scored_search(expanded_query, top_k)
                except Exception as e:
                    logger.log_with_data("warning", "ค้นหาแบบกรองล้มเหลว ใช้วิธีปกติ", {
                        "action": "filtered_retrieval_failed",
                        "error": str(e),
                        "fallback": "standard_retrieval"
                    })
                    docs = _scored_search(expanded_query, top_k)
            else:
                docs = _scored_search(expanded_query, top_k)
        else:
            _LOG.debug("[Practical] retrieve(unfiltered) query=%r top_k=%d", expanded_query[:60], top_k)
            docs = _scored_search(expanded_query, top_k)

        # ── Round 2: anchor-enriched retrieval ──────────────────────────────
        # After Round 1 we know which license/topic the corpus ranks highest.
        # Round 2 re-queries with that anchor term appended, catching docs that
        # cover the same topic but use different phrasing from the user's query.
        # Skipped when metadata_filter is active — filtered queries are already targeted.
        if docs and not metadata_filter:
            _top_md = getattr(docs[0], "metadata", {}) or {}
            _r2_anchor: str = ""
            for _r2_field in ("license_type", "operation_topic", "main_topic"):
                _r2_val = str(_top_md.get(_r2_field) or "").strip()
                if _r2_val and _r2_val not in expanded_query:
                    _r2_anchor = _r2_val
                    break
            if _r2_anchor:
                _round2_q = expanded_query + " " + _r2_anchor
                _round2_docs = _scored_search(_round2_q, top_k)
                _seen_r1 = {hash((getattr(d, "page_content", "") or "")[:120]) for d in docs}
                _r2_added = 0
                for _d2 in _round2_docs:
                    _h2 = hash((getattr(_d2, "page_content", "") or "")[:120])
                    if _h2 not in _seen_r1:
                        _seen_r1.add(_h2)
                        docs.append(_d2)
                        _r2_added += 1
                if _r2_added:
                    _LOG.info(
                        "[Practical] Round 2 +%d docs (anchor=%r) total=%d",
                        _r2_added, _r2_anchor[:40], len(docs),
                    )

        # ── Round 3: gap-fill pass (Iterative Retrieval) ─────────────────────
        # After Round 2, check if key content fields are covered across docs.
        # If ≥ ITERATIVE_RETRIEVAL_MIN_MISSING_FIELDS fields are absent, re-query
        # with those Thai terms appended — fetches docs that fill the gaps.
        # Skipped when metadata_filter is active (filtered queries already targeted).
        _iter_min = int(getattr(conf, "ITERATIVE_RETRIEVAL_MIN_MISSING_FIELDS", 2))
        if docs and not metadata_filter:
            _missing_terms = self._check_field_coverage(docs)
            if len(_missing_terms) >= _iter_min:
                _round3_q = expanded_query + " " + " ".join(_missing_terms)
                _round3_docs = _scored_search(_round3_q, top_k)
                _seen_r3 = {hash((getattr(d, "page_content", "") or "")[:120]) for d in docs}
                _r3_added = 0
                for _d3 in _round3_docs:
                    _h3 = hash((getattr(_d3, "page_content", "") or "")[:120])
                    if _h3 not in _seen_r3:
                        _seen_r3.add(_h3)
                        docs.append(_d3)
                        _r3_added += 1
                if _r3_added:
                    _LOG.info(
                        "[Practical] Round 3 gap-fill +%d docs (missing=%r) total=%d",
                        _r3_added, _missing_terms, len(docs),
                    )

        retrieval_ms = (time.time() - start) * 1000

        # Token Optimization: Filter by similarity score
        # เลือกเฉพาะเอกสารที่มี similarity > threshold
        min_similarity = getattr(conf, 'RETRIEVAL_MIN_SIMILARITY', 0.6)
        filtered_docs = []
        low_quality_docs = []
        
        for d in docs:
            _md_f = (getattr(d, "metadata", {}) or {})
            # BM25-only docs (no Dense score) are trusted by RRF — bypass min_similarity
            if _md_f.get("_bm25_hit"):
                filtered_docs.append(d)
                continue
            score = _md_f.get("_sim") or getattr(d, 'score', None)
            if score is not None and score >= min_similarity:
                filtered_docs.append(d)
            elif score is not None:
                low_quality_docs.append((d, score))
        
        # Safety: ถ้ากรองจนเหลือน้อยเกิน fallback ใช้ top-N by score แทน all docs
        # docs ถูก sort โดย similarity_search แล้ว (descending) ดังนั้น [:N] = best available
        if len(filtered_docs) < 2:
            _fallback_n = int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 6))
            filtered_docs = docs[:_fallback_n]
            logger.log_with_data("warning", "Similarity filter เข้มเกิน fallback ใช้ top-N docs", {
                "filtered_count": len(filtered_docs),
                "total_docs": len(docs),
                "fallback_n": _fallback_n,
                "min_similarity": min_similarity
            })
        else:
            _LOG.debug("[Practical] sim_filter: %d → %d docs (removed %d below %.2f)", len(docs), len(filtered_docs), len(low_quality_docs), min_similarity)
        
        docs = filtered_docs

        # Dedup: remove docs with identical page_content (retriever may return duplicates
        # when multiple Chroma rows share the same embedding text)
        _seen_h: set = set()
        _deduped: list = []
        for _d in docs:
            _h = hash((getattr(_d, "page_content", "") or "")[:120])
            if _h not in _seen_h:
                _seen_h.add(_h)
                _deduped.append(_d)
        docs = _deduped

        # ── Cross-encoder reranker (Level 3) ─────────────────────────────────
        # Rerank raw retrieval results BEFORE metadata boost so the learned model
        # has the final say on relevance ordering. Metadata boost then acts only as
        # a tiebreaker on near-equal scores rather than overriding the reranker.
        if getattr(conf, "RERANKER_ENABLED", False) and docs:
            try:
                from utils.reranker import rerank as _do_rerank
                _rr_model = getattr(conf, "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
                _rr_top_k = int(getattr(conf, "RERANKER_TOP_K", len(docs)))
                # Use original query (Thai only) not expanded_query which may contain English synonyms
                docs = _do_rerank(query, docs, model_name=_rr_model, top_k=_rr_top_k)
                _LOG.debug("[Practical] reranker applied — %d docs kept", len(docs))
            except Exception as _rr_exc:
                _LOG.warning("[Practical] reranker failed (%s) — continuing without rerank", _rr_exc)

        # ── Metadata-targeted boost ───────────────────────────────────────────
        # Problem: Sheet A has 30+ rows with near-identical legal-penalty text — their
        # embeddings are almost identical, so the wrong sub-topic row can rank above the
        # correct one.  A lightweight substring check on key metadata fields (no Thai
        # tokenisation needed) surfaces the correct specific row without the cost or
        # Thai-tokenisation issues of full content re-ranking.
        #
        # Fields checked (in priority order):
        #   operation_topic / sub_topic  — exact sub-operation label  (Sheet A & B)
        #   main_topic                   — topic cluster               (Sheet B)
        #   license_type                 — license name                (Sheet A)
        #
        # blend_score = chroma_sim + BOOST_WEIGHT × metadata_hit
        # BOOST_WEIGHT = 0.25  (strong enough to re-order near-ties; won't override a
        #                        genuinely better semantic match from a different topic)
        _BOOST_WEIGHT = 0.25
        _BOOST_FIELDS = ("operation_topic", "sub_topic", "main_topic", "license_type", "book_name", "source_book")
        _q_lower_boost = expanded_query.lower()
        _boosted: list = []
        for _d in docs:
            _md_b = getattr(_d, "metadata", {}) or {}
            _sim_b = float(_md_b.get("_sim") or 0.0)
            _hit = 0.0
            for _field in _BOOST_FIELDS:
                _val = str(_md_b.get(_field) or "").strip().lower()
                if _val and len(_val) >= 3 and _val in _q_lower_boost:
                    _hit = 1.0
                    break
            _md_b["_blend"] = _sim_b + _BOOST_WEIGHT * _hit
            _boosted.append((_d, _md_b["_blend"]))
        # Sort descending by blend score; preserve original order on ties (stable sort)
        _boosted.sort(key=lambda x: x[1], reverse=True)
        docs = [_d for _d, _ in _boosted]
        # Log when boost changed the ranking
        _boost_changed = any(
            _boosted[i][1] != _boosted[i][0].metadata.get("_sim", _boosted[i][1])
            for i in range(min(3, len(_boosted)))
        )
        if _boost_changed:
            _LOG.debug("[Practical] metadata boost applied — top doc: %r blend=%.3f",
                      str((docs[0].metadata if docs else {}).get("operation_topic") or
                          (docs[0].metadata if docs else {}).get("main_topic") or "?")[:60],
                      _boosted[0][1] if _boosted else 0.0)

        # Extract similarity scores if available
        scores = []
        for d in docs:
            score = (getattr(d, "metadata", {}) or {}).get("_sim") or getattr(d, 'score', None)
            if score is not None:
                scores.append(score)

        # Extract top topics
        topics = []
        for d in docs[:3]:
            md = getattr(d, 'metadata', {})
            topic = md.get('operation_topic') or md.get('topic', '')
            if topic and topic not in topics:
                topics.append(topic)
        
        # Token: filter + cap metadata at retrieval time — prevents raw metadata accumulating in state
        _STORE_META_WHITELIST = frozenset({
            "data_type", "license_type", "operation_topic",
            "main_topic", "sub_topic", "answer_guideline",
            "entity_type_normalized", "registration_type", "department",
            "fees", "operation_duration", "service_channel",
            "research_reference", "operation_steps", "identification_documents",
            "operation_group",
            "operation_by_department",  # needed by sibling-completion in handle()
            "legal_regulatory",     # บทลงโทษ ค่าปรับ ข้อกำหนดทางกฎหมาย
            "terms_and_conditions", # หน้าที่และเงื่อนไขของผู้ประกอบการ
            "restaurant_ai_document",  # เอกสาร/ฟอร์ม AI ร้านอาหาร
        })
        _STORE_FIELD_CAPS = {
            # Must be >= the per-field caps used in the handle() prompt loop below,
            # otherwise the storage cut dominates and the prompt cap has no effect.
            "operation_steps": 1000, "identification_documents": 4000,
            "research_reference": 3100, "fees": 500, "service_channel": 500,
            "legal_regulatory": 2000, "terms_and_conditions": 800,
            "restaurant_ai_document": 800,
            # marketing / business_guide content fields
            "answer_guideline": 1500, "main_topic": 120, "sub_topic": 150,
        }
        # Business guide docs (no license_type) store all content in answer_guideline.
        # Chroma metadata has the FULL value but _STORE_FIELD_CAPS["answer_guideline"]=1500
        # truncates it, causing downstream LLM to miss later steps/sections.
        # Use a higher cap for business_guide docs so the full content reaches the LLM.
        _BG_ANSWER_GUIDELINE_CAP = int(getattr(conf, "LLM_DOC_CHARS_BUSINESS_GUIDE", 3500))
        results: List[Dict[str, Any]] = []
        for d in docs[:max_docs]:
            raw_md = getattr(d, "metadata", {}) or {}
            # Detect business_guide doc: no license_type → answer_guideline is the sole content
            _lt_val = (raw_md.get("license_type") or "").strip().lower()
            _is_bg = _lt_val in ("", "nan", "none")
            slim_md = {}
            for k, v in raw_md.items():
                if k not in _STORE_META_WHITELIST:
                    continue
                if v in (None, "", "nan", "None"):
                    continue
                v_str = str(v)
                if k == "answer_guideline" and _is_bg:
                    cap = _BG_ANSWER_GUIDELINE_CAP
                else:
                    cap = _STORE_FIELD_CAPS.get(k)
                slim_md[k] = v_str[:cap] if cap and len(v_str) > cap else v_str
            results.append(
                {"content": (getattr(d, "page_content", "") or "")[:max_chars], "metadata": slim_md}
            )

        _sim_summary = f"max={max(scores):.3f} min={min(scores):.3f} avg={sum(scores)/len(scores):.3f}" if scores else "no scores"
        _LOG.info("[Practical] retrieve done | docs=%d/%d time=%.0fms %s", len(results), max_docs, retrieval_ms, _sim_summary)

        if len(results) == 0:
            _LOG.warning("[Practical] ไม่พบเอกสาร query=%r filter=%s", query[:80], metadata_filter)
        elif scores and max(scores) < 0.5:
            _LOG.warning("[Practical] low similarity max=%.3f query=%r", max(scores), query[:60])
        
        # Log รายละเอียดแต่ละเอกสาร
        _top_topics: list = []
        _license_counts: dict = {}
        for i, r in enumerate(results):
            md = r.get("metadata", {}) or {}
            topic = md.get("operation_topic") or md.get("topic") or md.get("filename") or "?"
            etype = md.get("entity_type_normalized") or md.get("entity_type") or ""
            dept = md.get("department") or ""
            license_t = md.get("license_type") or ""
            reg_t = md.get("registration_type") or ""
            sim = scores[i] if i < len(scores) else None
            sim_str = f"{sim:.3f}" if sim is not None else "n/a"

            _parts = [f"sim={sim_str}", f"license={license_t!r}"]
            if etype: _parts.append(f"entity={etype!r}")
            if dept: _parts.append(f"dept={dept!r}")
            if reg_t: _parts.append(f"reg={reg_t!r}")

            _LOG.debug("[Practical] doc[%d/%d] topic=%r | %s", i + 1, len(results), topic, " ".join(_parts))
            _top_topics.append(topic)
            if license_t:
                _license_counts[license_t] = _license_counts.get(license_t, 0) + 1

        _LOG.debug("[Practical] retrieved_topics=%s", _top_topics)
        if _license_counts:
            _breakdown = " | ".join(f"{lt}={n}" for lt, n in sorted(_license_counts.items(), key=lambda x: -x[1]))
            _LOG.debug("[Practical] license_breakdown: %s", _breakdown)

        return results

    # Topic Registry (auto-discovery from Chroma, no hardcoding)
    def _retrieve_multi_topic(self, question: str, slot_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._retrieve_docs(question, slot_context=slot_context)

    def _debug_log(self, stage: str, query: str, docs_json: List[Dict[str, Any]]):
        if not _LOG.isEnabledFor(logging.DEBUG):
            return
        try:
            n = len(docs_json)
            top1 = docs_json[0] if n else {}
            top_meta = top1.get("metadata", {}) if isinstance(top1, dict) else {}
            top_content = (top1.get("content", "") if isinstance(top1, dict) else "")[:120]
            _LOG.debug("[DEBUG:%s] query=%r docs_count=%d", stage, query, n)
            if n:
                _LOG.debug("[DEBUG:%s] top1_metadata_keys=%s", stage, list(top_meta.keys())[:8])
                _LOG.debug("[DEBUG:%s] top1_content_120=%r", stage, top_content)
        except Exception:
            pass

    # ENTRYPOINT
    def handle(self, state: ConversationState, user_input: str, _internal: bool = False) -> Tuple[ConversationState, str]:
        state.context = state.context or {}
        state.persona_id = self.persona_id

        user_text = (user_input or "").strip()
        norm = self._normalize_for_intent(user_text)

        auto_internal_guard_key = "_auto_post_retrieve_guard"
        if not _internal:
            state.context.pop(auto_internal_guard_key, None)
        elif user_text == "__auto_post_retrieve__":
            state.context[auto_internal_guard_key] = int(state.context.get(auto_internal_guard_key, 0) or 0) + 1

        # recovery only in non-internal, owner-gated (already inside the function)
        if not _internal:
            self._maybe_recover_pending_slot_from_last_bot(state, user_text)

        filled_topic_value = None
        bypassed_menu = False

        pending_key_before = None
        if (not _internal) and isinstance((state.context or {}).get("pending_slot"), dict):
            pending_key_before = (state.context.get("pending_slot") or {}).get("key")

        # ALWAYS allow pending_slot consumption (this is the core requirement)
        if (not _internal) and user_text:
            pending_status = self._consume_pending_slot_from_user(state, user_text)

            if pending_status == "BYPASS":
                bypassed_menu = True

            if pending_status == "FILLED":
                slots = (state.context or {}).get("slots", {}) or {}

                if isinstance(slots, dict) and "topic" in slots and slots.get("topic"):
                    filled_topic_value = str(slots.get("topic")).strip()
                    state.context["topic"] = filled_topic_value

                if pending_key_before == "multi_topic_select":
                    selected_lt = (slots or {}).pop("multi_topic_select", "") or ""
                    state.context.pop("multi_license_topics", None)
                    state.context.pop("_multi_topic_retrieval", None)
                    self._append_user_once(state, user_input)
                    if selected_lt:
                        state.context["last_user_legal_query"] = selected_lt
                        state.context["last_topic"] = selected_lt
                        _sel_slot_ctx = None
                        try:
                            _sel_slot_ctx = state.get_collected_slots() or None
                        except Exception:
                            pass
                        state.current_docs = self._retrieve_docs(selected_lt, slot_context=_sel_slot_ctx)
                        state.last_retrieval_query = selected_lt
                        return self.handle(state, "__auto_post_retrieve__", _internal=True)
                    return self.handle(state, user_input, _internal=False)

                if pending_key_before == self._PHASE3_SLOT_KEY:
                    sel = slots.get(self._PHASE3_SLOT_KEY)
                    if isinstance(sel, str) and sel.strip():
                        state.context[self._PHASE3_SLOT_KEY] = [sel.strip()]
                    elif isinstance(sel, list) and sel:
                        state.context[self._PHASE3_SLOT_KEY] = [str(x).strip() for x in sel if str(x).strip()]

                    self._append_user_once(state, user_input)

                    forced = f"ขอข้อมูลเฉพาะหัวข้อ: {', '.join(state.context.get(self._PHASE3_SLOT_KEY, []))}"
                    return self.handle(state, forced, _internal=True)

            if pending_status == "INVALID":
                _ps_inv = state.context.get("pending_slot")
                pending = _ps_inv if isinstance(_ps_inv, dict) else {}
                options = pending.get("options") if isinstance(pending, dict) else None
                if isinstance(options, list) and options:
                    msg = "ตอบเป็นตัวเลขได้ครับ\n" + self._format_numbered_options(options)

                    self._append_user_once(state, user_input)
                    msg = self._apply_practical_lint(msg, kind="menu")
                    self._append_assistant(state, msg)

                    state.round = int(getattr(state, "round", 0) or 0) + 1
                    return state, msg

        # Supervisor-owned menu: do not render greeting/menu here
        if (not _internal) and self._supervisor_owns_menu(state):
            # Still allow satisfaction to be treated as normal text flow; no menu injection.
            pass
        else:
            if (not _internal) and self._looks_like_satisfaction(user_text, state):
                self._append_user_once(state, user_input)
                msg = self._reply_satisfaction(state)
                self._append_assistant(state, msg)
                state.round = int(getattr(state, "round", 0) or 0) + 1
                return state, msg

            if (not _internal) and self._looks_like_greeting(user_text) and not filled_topic_value and not bypassed_menu:
                self._append_user_once(state, user_input)
                msg = self._reply_greeting_with_choices(state, kind="greet")
                self._append_assistant(state, msg)
                state.round = int(getattr(state, "round", 0) or 0) + 1
                return state, msg

        # DEDUPE: only append once
        if not _internal:
            self._append_user_once(state, user_input)

        last_bot = next((m["content"] for m in reversed(state.messages[:-1]) if m["role"] == "assistant"), "")

        if (not _internal) and ("ประเภท" in (last_bot or "")) and (
            self._DONT_KNOW_RE.match(norm) or self._ASK_TYPES_RE.search(norm)
            or self._dont_know_or_asking_types_llm_check(norm, state)
        ):
            # If supervisor owns menu, don't inject topic menu here either.
            if not self._supervisor_owns_menu(state):
                menu = self._get_topic_menu(state)
                state.context["pending_slot"] = {"key": "topic", "options": menu, "allow_multi": False}
                state.context["main_menu_shown"] = True
                msg = self._format_numbered_options(menu)
                msg = self._apply_practical_lint(msg, kind="menu")
                self._append_assistant(state, msg)
                state.round = int(getattr(state, "round", 0) or 0) + 1
                return state, msg

        # If user picked topic -> always retrieve for that topic
        if (not _internal) and filled_topic_value:
            q = filled_topic_value
            _topic_slot_ctx = None
            try:
                _topic_slot_ctx = state.get_collected_slots() or None
            except Exception:
                pass
            state.current_docs = self._retrieve_docs(q, slot_context=_topic_slot_ctx)
            state.last_retrieval_query = q
            tmp = [
                {"content": d.get("content", "")[:120], "metadata": d.get("metadata", {})}
                for d in state.current_docs[:1]
            ]
            self._debug_log("post_retrieve(topic)", query=q, docs_json=tmp)
            return self.handle(state, "__auto_post_retrieve__", _internal=True)

        # Practical retrieval: new-topic aware (uses multi-topic registry for compound questions)
        # Skip if supervisor already built topic_slot_queue from entity-filtered docs — overwriting
        # those docs would cause the LLM to see mixed-entity docs and generate wrong choices.
        # Also skip if supervisor already did multi-topic merge — overwriting loses the merged docs.
        _has_slot_queue = bool((state.context or {}).get("topic_slot_queue"))
        _is_multi_topic_merged = bool((state.context or {}).get("_multi_topic_retrieval"))
        if (not _internal) and (not _has_slot_queue) and (not _is_multi_topic_merged) and self._looks_like_legal_question(user_text):
            if self._should_retrieve_new_topic(state, user_text):
                _mt_slot_ctx = None
                try:
                    _mt_slot_ctx = state.get_collected_slots() or None
                except Exception:
                    pass
                state.current_docs = self._retrieve_multi_topic(user_text, slot_context=_mt_slot_ctx)
                state.last_retrieval_query = user_text
                tmp = [
                    {"content": d.get("content", "")[:120], "metadata": d.get("metadata", {})}
                    for d in state.current_docs[:1]
                ]
                self._debug_log("post_retrieve", query=user_text, docs_json=tmp)
                return self.handle(state, "__auto_post_retrieve__", _internal=True)

        # Proactive trim: when session has accumulated many tokens, cut history early
        # so the CURRENT call (not just future calls) benefits from a shorter context.
        if getattr(state, "total_tokens", 0) > 20_000 and hasattr(state, "trim_messages"):
            state.trim_messages(keep_last=4)

        # Cap each assistant message in history: 300 chars is enough for context continuity.
        # 600 was generating ~7-8% of the total prompt budget per assistant turn.
        _MAX_HIST_ASST_CHARS = 300
        recent_msgs = []
        for _hm in state.messages[-4:]:
            if _hm.get("role") == "assistant":
                _hc = _hm.get("content") or ""
                if len(_hc) > _MAX_HIST_ASST_CHARS:
                    _hm = dict(_hm, content=_hc[:_MAX_HIST_ASST_CHARS] + "…")
            recent_msgs.append(_hm)

        # Broad questions (e.g. "อยากเปิดร้านเบเกอรี่ ต้องทำอะไรบ้าง") need more docs
        # to cover all license types (regulatory + business_guide + marketing).
        # Standard cap of 3 docs/license misses rare license types for these queries.
        _is_broad_q = bool((state.context or {}).get("_broad_question"))
        _prompt_max_docs = (
            int(getattr(conf, "LLM_DOCS_MAX_BROAD", 6))
            if _is_broad_q
            else int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 3))
        )
        _FIELD_CAPS = {
            "operation_steps": 900,
            "identification_documents": 4000,  # must never truncate — missing items = legal error; current max in data is 3156 chars
            "fees": 400,
            "operation_duration": 200,
            "service_channel": 400,
            "legal_regulatory": 1250,
            "terms_and_conditions": 1500,
            "restaurant_ai_document": 650,
        }
        _LONG_FIELDS_DEDUP = {"operation_steps", "legal_regulatory", "identification_documents"}  # deduped per (license_type, entity_type_normalized); terms_and_conditions excluded (cut-off time rows differ per doc)

        # Cap docs sent to LLM: _prompt_max_docs per license_type to control token usage.
        # For multi-license, each license still gets its own metadata via dedup logic below.
        _all_docs = state.current_docs or []
        _lt_order: list = []  # license_types in order of first appearance
        _lt_doc_count: dict = {}
        _mt_order: list = []  # main_topics from non-regulatory docs (no license_type)
        for _d0 in _all_docs:
            _lt0 = ((_d0.get("metadata") or {}).get("license_type") or "").strip()
            if _lt0 and _lt0.lower() not in ("nan", "none") and _lt0 not in _lt_order:
                _lt_order.append(_lt0)
            if _lt0 and _lt0.lower() not in ("nan", "none"):
                _lt_doc_count[_lt0] = _lt_doc_count.get(_lt0, 0) + 1
            elif not _lt0 or _lt0.lower() in ("nan", "none"):
                _mt0 = ((_d0.get("metadata") or {}).get("main_topic") or "").strip()
                if _mt0 and _mt0.lower() not in ("nan", "none") and _mt0 not in _mt_order:
                    _mt_order.append(_mt0)
        _is_multi_license_docs = len(_lt_order) > 1

        # ── Stale-docs guard (fix #22): entity/location mismatch without user re-mentioning them ──
        # If collected_slots has entity_type but current docs are from the WRONG entity (e.g.
        # supervisor failed to apply filter on a new topic), force entity-filtered re-retrieval.
        # Only when user is NOT mentioning entity in this turn (that case is handled by switch block).
        _cs_pre = (state.get_collected_slots() if hasattr(state, "get_collected_slots") else {})
        _cs_entity_pre = (_cs_pre.get("entity_type") or _cs_pre.get("entity_type_normalized") or "").strip()
        if _cs_entity_pre and _all_docs and not (state.context or {}).get("_entity_switch_done"):
            _wrong_entity_docs_pre = [
                d for d in _all_docs
                if str((d.get("metadata") or {}).get("entity_type_normalized") or "").strip()
                not in (_cs_entity_pre, "")
            ]
            if _wrong_entity_docs_pre and len(_wrong_entity_docs_pre) > len(_all_docs) // 2:
                _base_q_pre = (
                    str(getattr(state, "last_retrieval_query", "") or "").strip() or user_text
                )
                try:
                    _pre_docs = self._retrieve_docs(
                        _base_q_pre,
                        metadata_filter={"entity_type_normalized": _cs_entity_pre},
                        slot_context={"entity_type": _cs_entity_pre},
                    )
                    if _pre_docs:
                        _all_docs = _pre_docs
                        state.current_docs = _pre_docs
                        _LOG.info(
                            "[Practical] stale-docs guard: >50%% wrong entity (%r) — re-retrieved %d docs for %r",
                            _cs_entity_pre, len(_pre_docs), _base_q_pre[:50],
                        )
                        _lt_order = []
                        _lt_doc_count = {}
                        for _d0 in _all_docs:
                            _lt0 = ((_d0.get("metadata") or {}).get("license_type") or "").strip()
                            if _lt0 and _lt0.lower() not in ("nan", "none") and _lt0 not in _lt_order:
                                _lt_order.append(_lt0)
                            if _lt0 and _lt0.lower() not in ("nan", "none"):
                                _lt_doc_count[_lt0] = _lt_doc_count.get(_lt0, 0) + 1
                        _is_multi_license_docs = len(_lt_order) > 1
                except Exception as _sg_err:
                    _LOG.debug("[Practical] stale-docs guard re-retrieve failed: %s", _sg_err)

        # ── Entity-type switch detection ──────────────────────────────────────────────────────────
        # Must run EARLY (before focus filter, op-intent filter, doc capping) so all downstream
        # code sees the correctly-filtered docs.
        # Scenario: user says "แล้วถ้าเป็นนิติบุคคลอ่ะ" after a บุคคลธรรมดา answer.
        # If supervisor's _apply_slot_change_if_detected failed (e.g. old_entity was empty),
        # state.current_docs still has the old entity's docs. Detect the switch here and re-retrieve.
        _ENTITY_SWITCH_PATTERNS = [
            (r"นิติบุคคล|(?:แบบ|เป็น(?:แบบ)?|ประเภท|รูปแบบ)\s*นิติ|บริษัท(?:\s*จำกัด|\s*มหาชน)?|ห้าง(?:หุ้นส่วน)?", "นิติบุคคล"),
            (r"บุคคลธรรมดา|บุคคล.{0,4}มดา|บุคคลทั่วไป|เจ้าของคนเดียว|กิจการเจ้าของคนเดียว", "บุคคลธรรมดา"),
        ]
        _stored_et = (
            (state.context or {}).get("slots", {}).get("entity_type_normalized")
            or (state.get_collected_slot("entity_type") if hasattr(state, "get_collected_slot") else "")
            or ""
        ).strip()
        _query_et_override = ""
        for _epat, _eval in _ENTITY_SWITCH_PATTERNS:
            if re.search(_epat, user_text, re.IGNORECASE) and _eval != _stored_et:
                _query_et_override = _eval
                break

        # If entity switch detected AND current docs are the wrong entity type → re-retrieve now.
        # Guard: skip if supervisor already re-retrieved for this entity type this turn (_entity_switch_done).
        _entity_switch_already_done = (
            str((state.context or {}).get("_entity_switch_done") or "") == _query_et_override
        )
        if _query_et_override and _all_docs and not _entity_switch_already_done:
            # Re-retrieve if current docs have NONE of the requested entity_type.
            # Handles: (a) stored_et was wrong entity, (b) stored_et was empty (unfiltered docs).
            _correct_entity_docs = [
                d for d in _all_docs
                if str((d.get("metadata") or {}).get("entity_type_normalized") or "").strip() == _query_et_override
            ]
            if not _correct_entity_docs:  # no docs match requested entity → need fresh retrieval
                # Priority: last_retrieval_query (actual topic from prior turn) > last_user_legal_query
                # last_user_legal_query is overwritten to the follow-up phrase ("แล้วถ้าเป็นนิติบุคคลอ่ะ")
                # by supervisor line 7381 BEFORE retrieval runs — it's a bad query for vector search.
                _switch_base_q = (
                    str(getattr(state, "last_retrieval_query", "") or "").strip()
                    or str((state.context or {}).get("last_user_legal_query") or "").strip()
                    or user_text
                )
                try:
                    _dept_for_switch = str(
                        (state.context or {}).get("collected_slots", {}).get("department") or ""
                    ).strip()
                    _switch_filter: dict = (
                        {"$and": [
                            {"entity_type_normalized": _query_et_override},
                            {"department": _dept_for_switch},
                        ]}
                        if _dept_for_switch
                        else {"entity_type_normalized": _query_et_override}
                    )
                    _switch_docs = self._retrieve_docs(
                        _switch_base_q,
                        metadata_filter=_switch_filter,
                        slot_context={"entity_type": _query_et_override},
                    )
                    if _switch_docs:
                        _LOG.info(
                            "[Practical] entity-switch %r→%r: retrieved %d fresh docs for %r",
                            _stored_et, _query_et_override, len(_switch_docs), _switch_base_q[:50],
                        )
                        _all_docs = _switch_docs
                        state.current_docs = _switch_docs
                        # Rebuild _lt_order / _lt_doc_count / _is_multi_license_docs for new docs
                        _lt_order = []
                        _lt_doc_count = {}
                        for _d0 in _all_docs:
                            _lt0 = ((_d0.get("metadata") or {}).get("license_type") or "").strip()
                            if _lt0 and _lt0.lower() not in ("nan", "none") and _lt0 not in _lt_order:
                                _lt_order.append(_lt0)
                            if _lt0 and _lt0.lower() not in ("nan", "none"):
                                _lt_doc_count[_lt0] = _lt_doc_count.get(_lt0, 0) + 1
                        _is_multi_license_docs = len(_lt_order) > 1
                    else:
                        # This topic is entity-neutral (no separate docs per entity_type).
                        # Clear the override so the LLM prompt stays neutral instead of
                        # wrongly restricting the answer to "นิติบุคคล only".
                        _LOG.info(
                            "[Practical] entity-switch: no %r docs for %r — topic is entity-neutral, clearing override",
                            _query_et_override, _switch_base_q[:50],
                        )
                        _query_et_override = ""
                except Exception as _sw_err:
                    _LOG.warning("[Practical] entity-switch retrieval failed: %s", _sw_err)
                    _query_et_override = ""  # failed → fall back to neutral prompt

        # ── Location-switch fallback (practical layer) ────────────────────────
        # Mirrors entity-switch above. Handles the case where supervisor's location-filtered
        # retrieval failed or returned 0 docs, leaving state.current_docs with the wrong location.
        # Only covers ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร (the only license with location-split docs).
        _LOCATION_SWITCH_MAP = [
            (r"กรุงเทพ|กทม\.?", "กรุงเทพฯ"),
            (r"ต่างจังหวัด|ต่างหวัด|นอกกรุงเทพ", "ต่างจังหวัด"),
        ]
        _stored_loc = (
            (state.context or {}).get("slots", {}).get("location")
            or (state.get_collected_slot("location") if hasattr(state, "get_collected_slot") else "")
            or ""
        ).strip()
        _query_loc_override = ""
        for _lpat, _lval in _LOCATION_SWITCH_MAP:
            if re.search(_lpat, user_text, re.IGNORECASE) and _stored_loc:
                # "กรุงเทพฯ และปริมณฑล" and "กรุงเทพฯ" are the same location — no switch needed.
                # Check if _stored_loc already matches the same location pattern.
                if re.search(_lpat, _stored_loc, re.IGNORECASE):
                    break
                if _lval != _stored_loc:
                    _query_loc_override = _lval
                break

        if _query_loc_override and _all_docs:
            _correct_loc_docs = [
                d for d in _all_docs
                if str((d.get("metadata") or {}).get("location") or "").strip() == _query_loc_override
            ]
            if not _correct_loc_docs:
                _loc_base_q = (
                    str(getattr(state, "last_retrieval_query", "") or "").strip()
                    or str((state.context or {}).get("last_user_legal_query") or "").strip()
                    or user_text
                )
                try:
                    # Lock to current topic's license_type so a location-only filter
                    # can't pull in docs from unrelated licenses (e.g. ใบอนุญาตจัดตั้งฯ
                    # when the user is asking about EDC).
                    _loc_lt_lock = ""
                    for _d_ls in _all_docs:
                        _lt_ls = ((_d_ls.get("metadata") or {}).get("license_type") or "").strip()
                        if _lt_ls:
                            _loc_lt_lock = _lt_ls
                            break
                    _loc_meta_filter: dict = (
                        {"$and": [{"license_type": _loc_lt_lock}, {"location": _query_loc_override}]}
                        if _loc_lt_lock
                        else {"location": _query_loc_override}
                    )
                    _loc_docs = self._retrieve_docs(
                        _loc_base_q,
                        metadata_filter=_loc_meta_filter,
                        slot_context={"location": _query_loc_override},
                    )
                    if _loc_docs:
                        _LOG.info(
                            "[Practical] location-switch %r→%r: retrieved %d fresh docs",
                            _stored_loc, _query_loc_override, len(_loc_docs),
                        )
                        _all_docs = _loc_docs
                        state.current_docs = _loc_docs
                        _lt_order = []
                        _lt_doc_count = {}
                        for _d0 in _all_docs:
                            _lt0 = ((_d0.get("metadata") or {}).get("license_type") or "").strip()
                            if _lt0 and _lt0.lower() not in ("nan", "none") and _lt0 not in _lt_order:
                                _lt_order.append(_lt0)
                            if _lt0 and _lt0.lower() not in ("nan", "none"):
                                _lt_doc_count[_lt0] = _lt_doc_count.get(_lt0, 0) + 1
                        _is_multi_license_docs = len(_lt_order) > 1
                    else:
                        _query_loc_override = ""  # no location-specific docs → topic is location-neutral
                except Exception as _loc_err:
                    _LOG.warning("[Practical] location-switch retrieval failed: %s", _loc_err)
                    _query_loc_override = ""

        # Multi-topic cap: when docs span 2+ license types, halve the per-license doc limit.
        # Rationale: 2 licenses × 6 docs × large metadata fields = ~36K prompt tokens → LLM hits
        # max_tokens mid-JSON and fails. Reducing to 3 docs/license (~18K tokens) stays safe.
        # Not applied to broad questions (_is_broad_q) which intentionally send many mixed docs.
        if _is_multi_license_docs and not _is_broad_q:
            _base = int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 6))
            _prompt_max_docs = max(2, _base // 2)
            _LOG.info(
                "[Practical] multi-topic: reducing per-license doc cap %d → %d to prevent token overflow",
                _base, _prompt_max_docs,
            )

        # FOCUS FILTER: when docs span multiple license_types AND this is a single-topic query
        # (not an explicit "broad overview" request like "บอกทุกอย่าง"), keep only docs from the
        # dominant license_type — the one with the most docs and highest similarity to the query.
        # This prevents stale docs from the previous topic bleeding into the current answer.
        # Exception: if the multi-license set was intentionally built for a multi-topic query
        # (_multi_topic_retrieval flag), skip this filter.
        _is_broad_query = bool(
            re.search(r"บอกทุกอย่าง|รายละเอียดทั้งหมด|อยากรู้ครบ|ทุกอย่างที่|ทุกด้าน|ทุกเรื่อง", user_text or "")
        )
        _is_multi_topic_flag = bool((state.context or {}).get("_multi_topic_retrieval"))
        _is_broad_q_flag = bool((state.context or {}).get("_broad_question"))
        if _is_multi_license_docs and not _is_broad_query and not _is_multi_topic_flag and not _is_broad_q_flag and _lt_order:
            # Pick dominant license_type by relevance to the CURRENT user query (not doc count alone).
            # Algorithm:
            #   1. Score each license_type against user_text using:
            #      a) 8-char Thai substring match (high precision — avoids false partial-word matches)
            #      b) English/acronym token exact match with weight 5× (EDC, VAT, QR, etc.)
            #   2. If any license scores > 0, pick the highest scorer.
            #   3. Fall back to doc count only when no license matches the query at all.
            def _license_query_score(lt: str, query: str) -> float:
                s = 0.0
                q = query  # preserve case for Thai substring; lower for English
                q_lower = q.lower()
                lt_ns = re.sub(r"\s+", "", lt)
                # (a) 8-char Thai sliding window: high specificity
                for _i in range(len(lt_ns) - 7):
                    seg = lt_ns[_i:_i+8]
                    if seg in q:
                        s += 2.0
                # (a2) Short Thai names (< 10 chars): use full-name exact match instead.
                # Also try every 4-char substring to catch partial overlap with the query.
                if len(lt_ns) < 10:
                    if lt_ns in q:
                        s += 5.0
                    elif len(lt_ns) >= 4:
                        for _i in range(len(lt_ns) - 3):
                            seg4 = lt_ns[_i:_i+4]
                            if seg4 in q:
                                s += 1.0
                # (a3) Abbreviation expansion: detect when user types a short abbreviation
                # in their query (e.g. "ภป", "หจก") and map it to the correct full-name
                # license_type stored in Chroma.  Gives +8 to prevent the full-form license
                # name from scoring 0 on an abbreviation query.
                for _abbrev_pat, _full_pat in self._LQS_ABBREV_CHECK:
                    if _abbrev_pat.search(q) and _full_pat.search(lt):
                        s += 8.0
                        break
                # (b) English/numeric token exact match (case-insensitive)
                for _tok in re.split(r"\s+", lt.strip()):
                    _tok = _tok.strip()
                    if re.match(r"^[A-Za-z0-9]+$", _tok) and len(_tok) >= 2:
                        if _tok.lower() in q_lower:
                            s += 5.0
                # (c) Keyword-to-license mapping for topics whose license_type name doesn't
                # share n-grams with the user query ("ปิดงบบัญชี" ≠ "จัดการการเงิน",
                # "ขึ้นทะเบียนนายจ้าง" ≠ "การจัดหาพนักงาน", "SAN PLUS" ≠ "ใบรับรองมาตรฐานร้านอาหาร").
                #
                # IMPORTANT: when matching license name in the condition, use SPECIFIC terms only —
                # NOT generic words like "บัญชี" that appear in MULTIPLE license names
                # (e.g. "บัญชีธนาคารรับเงิน" would incorrectly score +8 for accounting queries).
                # Match "จัดการการเงิน" (the actual license name), NOT generic "บัญชี"
                # which also appears in "บัญชีธนาคารรับเงิน" → prevents false +8 tie.
                if self._LQS_ACCOUNTING_RE.search(q) and re.search(r"จัดการการเงิน|จัดการเงิน", lt, re.IGNORECASE):
                    s += 8.0
                # Banking account → "บัญชีธนาคารรับเงิน"
                if self._LQS_BANKING_RE.search(q) and re.search(r"บัญชีธนาคาร", lt, re.IGNORECASE):
                    s += 8.0
                # Employer registration → "ทะเบียนนายจ้าง" (explicit registration intent)
                # Separated from hiring (การจัดหาพนักงาน) to prevent tie when score=8 for both.
                if self._LQS_EMPLOYER_RE.search(q) and re.search(r"ทะเบียนนายจ้าง", lt, re.IGNORECASE):
                    s += 10.0  # higher than hiring (+8) so ทะเบียนนายจ้าง wins on employer queries
                # Hiring / staffing → "การจัดหาพนักงาน"
                if self._LQS_HIRING_RE.search(q) and re.search(r"จัดหาพนักงาน", lt, re.IGNORECASE):
                    s += 8.0
                # SAN / SAN PLUS / มาตรฐานร้านอาหาร → "ใบรับรองมาตรฐานร้านอาหาร"
                if self._LQS_SAN_RE.search(q) and re.search(r"รับรองมาตรฐาน|มาตรฐานร้านอาหาร", lt, re.IGNORECASE):
                    s += 8.0
                # QR Payment / คิวอาร์ → "ระบบชำระเงินออนไลน์"
                if self._LQS_QR_KEYWORDS_RE.search(q) and re.search(r"ชำระเงินออนไลน์|ระบบชำระ", lt, re.IGNORECASE):
                    s += 8.0
                # VAT / ภพ.20 / ภาษีมูลค่าเพิ่ม → "ใบภาษีมูลค่าเพิ่ม ภพ.20"
                if self._LQS_VAT_RE.search(q) and re.search(r"ภาษีมูลค่าเพิ่ม|ภพ", lt, re.IGNORECASE):
                    s += 8.0
                # ประกันสังคม (without employer context) → ทะเบียนผู้ประกันตน only.
                if (self._LQS_SOCIAL_RE.search(q)
                        and not self._LQS_EMPLOYER_RE.search(q)
                        and re.search(r"ผู้ประกันตน", lt, re.IGNORECASE)):
                    s += 4.0
                # ผู้ประกันตน / ลูกจ้าง / เงินสมทบ → "ทะเบียนผู้ประกันตน"
                if self._LQS_INSURED_RE.search(q) and re.search(r"ผู้ประกันตน", lt, re.IGNORECASE):
                    s += 8.0
                # Health certificate → "ใบรับรองแพทย์ 9 โรค(สณ.11)"
                if self._LQS_HEALTH_CERT_RE.search(q) and re.search(r"รับรองแพทย์|9\s*โรค", lt, re.IGNORECASE):
                    s += 8.0
                # สุรา / ขายสุรา → "ใบอนุญาตจำหน่ายสุรา"
                if self._LQS_LIQUOR_RE.search(q) and re.search(r"สุรา", lt, re.IGNORECASE):
                    s += 8.0
                # ภาษีป้าย → "แบบแสดงรายการภาษีป้ายร้านอาหาร"
                if self._LQS_SIGN_TAX_RE.search(q) and re.search(r"ภาษีป้าย", lt, re.IGNORECASE):
                    s += 8.0
                # QR Payment / online payment → "ระบบชำระเงินออนไลน์"
                if self._LQS_QR_PAYMENT_RE.search(q) and re.search(r"ระบบชำระเงิน", lt, re.IGNORECASE):
                    s += 8.0
                return s

            _focus_query = (user_text or "").strip()
            # When handle() is called recursively with "__auto_post_retrieve__", user_text is a
            # placeholder — use last_retrieval_query as the actual query for scoring.
            if not _focus_query or _focus_query.startswith("__"):
                _focus_query = (getattr(state, "last_retrieval_query", None) or "").strip()

            # Compute scores for ALL license types so we can detect multi-requested scenarios.
            _lt_scores: dict = {lt: _license_query_score(lt, _focus_query) for lt in _lt_order}
            _best_lt = max(_lt_scores, key=lambda lt: _lt_scores[lt]) if _lt_scores else None
            _best_score = _lt_scores.get(_best_lt, -1.0) if _best_lt else -1.0

            # When slot-answer text (e.g. "นิติบุคคล", "45 ตรม") gives score=0, retry with
            # the original legal query — user_text is a slot value, not the topic question.
            _orig_q = ""
            if _best_score <= 0:
                _orig_q = (
                    (state.context or {}).get("last_user_legal_query")
                    or (state.context or {}).get("last_topic")
                    or getattr(state, "last_retrieval_query", None)
                    or ""
                ).strip()
                if _orig_q.startswith("__"):
                    _orig_q = ""
                if _orig_q and _orig_q != _focus_query:
                    for _lt_cand in _lt_order:
                        _s2 = _license_query_score(_lt_cand, _orig_q)
                        if _s2 > _lt_scores.get(_lt_cand, 0):
                            _lt_scores[_lt_cand] = _s2
                    _best_lt = max(_lt_scores, key=lambda lt: _lt_scores[lt]) if _lt_scores else None
                    _best_score = _lt_scores.get(_best_lt, -1.0) if _best_lt else -1.0

            # LLM fallback: when regex scoring gives 0 or a weak signal (≤4 = only partial
            # 4-char window matches, no keyword hit), use Haiku to identify the license type.
            # Threshold ≤4 catches ambiguous queries where regex fired on a short substring
            # but isn't confident enough to override — LLM required confidence ≥0.80 (higher
            # than the score=0 path) so it only overrides when truly certain.
            if _best_score <= 4 and len(_lt_order) >= 2 and len(_lt_order) <= 12:
                _llm_q = (_focus_query or _orig_q).strip()
                # Require higher confidence when overriding a weak regex signal (score 1-4)
                # vs no signal at all (score ≤0) to prevent incorrect overrides.
                _min_conf = 0.80 if _best_score > 0 else 0.70
                if _llm_q and not _llm_q.startswith("__"):
                    _llm_lt = self._lqs_license_type_fallback(_llm_q, _lt_order, state, min_confidence=_min_conf)
                    if _llm_lt:
                        _best_lt = _llm_lt
                        _best_score = 8.0  # synthetic score to enable focused filtering
                        _lt_scores[_llm_lt] = max(_lt_scores.get(_llm_lt, 0), 8.0)

            # Multi-requested exception: if 2+ licenses scored > 4 (= explicit keyword match,
            # not just incidental partial substring), the user asked about multiple licenses in
            # the same query. Keep ALL explicitly-requested licenses; drop only zero-score noise.
            # Example: "ขอใบอนุญาตจัดตั้งร้านอาหาร ฉันขายเหล้าด้วย" → food(16) + liquor(8) both >4.
            _explicitly_requested = [lt for lt in _lt_order if _lt_scores.get(lt, 0) > 4]
            if len(_explicitly_requested) >= 2:
                _noise_count = len([d for d in _all_docs
                                    if (d.get("metadata") or {}).get("license_type", "").strip()
                                    not in _explicitly_requested])
                if _noise_count:
                    _all_docs = [d for d in _all_docs
                                 if (d.get("metadata") or {}).get("license_type", "").strip()
                                 in _explicitly_requested]
                    _lt_order = [lt for lt in _lt_order if lt in _explicitly_requested]
                _LOG.info(
                    "[Practical] FOCUS FILTER: multi-requested %s → keeping all (scores: %s, dropped %d noise docs)",
                    _explicitly_requested,
                    {lt: round(_lt_scores.get(lt, 0), 1) for lt in _explicitly_requested},
                    _noise_count,
                )
                # _is_multi_license_docs stays True — LLM sees docs for both licenses and will
                # cover both in its answer. We intentionally do NOT set multi_license_topics here
                # because that forces action='answer', which would prevent slot questions from
                # being asked when slots (location/area_size) haven't been filled yet.
            else:
                # Only filter when a license name actually appeared in the query (score > 0).
                # When score=0 (no license name in query), doc-count dominance is unreliable —
                # skip filtering and let LLM decide from all retrieved docs.
                _dominant_lt = _best_lt if (_best_lt and _best_score > 0) else None
                _focused_docs = [d for d in _all_docs if (d.get("metadata") or {}).get("license_type", "").strip() == _dominant_lt] if _dominant_lt else []
                if _focused_docs:
                    _LOG.info(
                        "[Practical] FOCUS FILTER: multi-license %s → keeping only %r (score=%.1f, %d docs, dropped %d)",
                        _lt_order, _dominant_lt, _best_score, len(_focused_docs), len(_all_docs) - len(_focused_docs),
                    )
                    _all_docs = _focused_docs
                    _lt_order = [_dominant_lt]
                    _is_multi_license_docs = False
                    # Update state.current_docs immediately so supervisor/_route_pending_slot_to_persona
                    # sees the focused set when reading _area_license_lock — critical for topics without
                    # op-intent filter (e.g. Refund, WeChat Pay) where state.current_docs is not
                    # updated later in the flow.
                    state.current_docs = _focused_docs
                    # op-topic Latin correction: if all surviving docs have an
                    # operation_topic whose Latin tokens are NOT all in the user query
                    # (e.g. "SAN PLUS" when query is "SAN"), re-retrieve using the
                    # best-matching operation_topic found in Chroma for this license.
                    _q_lower_ff = (user_text or "").lower()
                    if re.search(r"[A-Za-z]{2,}", user_text or ""):
                        _ot_vals_ff = [
                            str((_d.get("metadata") or {}).get("operation_topic") or "").strip()
                            for _d in _all_docs
                        ]
                        _ot_vals_ff = [v for v in _ot_vals_ff if v and v not in ("nan", "None", "-", "?")]
                        if _ot_vals_ff:
                            def _ot_lscore(ot: str, _ql: str = _q_lower_ff) -> int:
                                def _fuzzy_in(tok: str, text: str) -> bool:
                                    if tok in text:
                                        return True
                                    tlen = len(tok)
                                    return any(
                                        SequenceMatcher(None, tok, text[i:i + tlen]).ratio() >= 0.75
                                        for i in range(max(0, len(text) - tlen + 1))
                                    )
                                _toks = [t.lower() for t in re.findall(r"[A-Za-z]{2,}", ot)]
                                if not _toks:
                                    return 0
                                return (sum(1 for t in _toks if _fuzzy_in(t, _ql))
                                        - sum(1 for t in _toks if not _fuzzy_in(t, _ql)))
                            _curr_ot_sc = max(_ot_lscore(ot) for ot in _ot_vals_ff)
                            if _curr_ot_sc <= 0:
                                try:
                                    _vs_ff = getattr(self.retriever, "vectorstore", None)
                                    _coll_ff = getattr(_vs_ff, "_collection", None) if _vs_ff else None
                                    if _coll_ff is not None:
                                        _all_ot_res = _coll_ff.get(
                                            where={"license_type": _dominant_lt},
                                            include=["metadatas"],
                                        )
                                        _all_ots_ff = {
                                            str(m.get("operation_topic") or "").strip()
                                            for m in (_all_ot_res.get("metadatas") or [])
                                        } - {"", "nan", "None", "-", "?"}
                                        _best_ot_ff, _best_ot_sc_ff = "", _curr_ot_sc
                                        for _ot_cand in _all_ots_ff:
                                            _s = _ot_lscore(_ot_cand)
                                            if _s > _best_ot_sc_ff:
                                                _best_ot_sc_ff = _s
                                                _best_ot_ff = _ot_cand
                                        if _best_ot_ff:
                                            _ot_fresh = self._retrieve_docs(
                                                user_text,
                                                metadata_filter={"$and": [
                                                    {"license_type": _dominant_lt},
                                                    {"operation_topic": _best_ot_ff},
                                                ]},
                                            )
                                            if _ot_fresh:
                                                _LOG.info(
                                                    "[Practical] op-topic Latin correction: %r → %r (score %d→%d, %d docs)",
                                                    list(dict.fromkeys(_ot_vals_ff)), _best_ot_ff,
                                                    _curr_ot_sc, _best_ot_sc_ff, len(_ot_fresh),
                                                )
                                                _all_docs = _ot_fresh
                                                state.current_docs = _ot_fresh
                                except Exception as _ff_ot_err:
                                    _LOG.debug("[Practical] op-topic Latin correction failed: %s", _ff_ot_err)

        # Terms-and-conditions miss check: if query asks about cut-off times / conditions
        # but NO current doc has non-empty terms_and_conditions → stale state, force fresh
        # filtered retrieval so the service-description doc can be found.
        # When handle() is called recursively with "__auto_post_retrieve__", user_text is a placeholder.
        # Fall back to last_retrieval_query to match TC queries correctly in those recursive calls.
        _tc_query_text = user_text if not user_text.startswith("__") else (
            getattr(state, "last_retrieval_query", None) or
            next((m["content"] for m in reversed(state.messages) if m["role"] == "user" and not m["content"].startswith("__")), "") or ""
        )
        if self._TC_QUERY_RE.search(_tc_query_text):
            _dominant_for_tc = (_lt_order[0] if _lt_order else "").strip()
            if _dominant_for_tc:
                # Require terms_and_conditions with actual cut-off times (HH:MM น. format),
                # not just any terms_and_conditions (cancellation/refund docs also have terms).
                _has_tc = any(
                    self._TC_TIME_RE.search(str((d.get("metadata") or {}).get("terms_and_conditions") or ""))
                    for d in _all_docs
                )
                _LOG.info(
                    "[Practical] TC-check: query=%r dominant=%r docs=%d has_tc_times=%s | tc_vals=%s",
                    _tc_query_text[:60], _dominant_for_tc, len(_all_docs), _has_tc,
                    [str((d.get("metadata") or {}).get("terms_and_conditions") or "")[:60]
                     for d in _all_docs],
                )
                if not _has_tc:
                    try:
                        _tc_query_for_retrieve = _tc_query_text or user_text
                        _tc_fresh = self._retrieve_docs(
                            _tc_query_for_retrieve,
                            metadata_filter={"license_type": _dominant_for_tc},
                            max_docs=8,
                        )
                        if _tc_fresh:
                            # Sort: docs with actual cut-off time data (HH:MM น.) come first so
                            # they survive the _prompt_max_docs cap when sent to LLM.
                            _tc_fresh.sort(
                                key=lambda _d: 0 if self._TC_TIME_RE.search(
                                    str(((_d.get("metadata") or {}).get("terms_and_conditions") or ""))
                                ) else 1
                            )
                            _all_docs = _tc_fresh
                            _lt_order = [_dominant_for_tc]
                            _is_multi_license_docs = False
                            state.current_docs = _tc_fresh
                            _LOG.info(
                                "[Practical] TC-miss → fresh filtered retrieval: license=%r found=%d docs (tc_times_doc_first=%s)",
                                _dominant_for_tc, len(_all_docs),
                                bool(self._TC_TIME_RE.search(str((_tc_fresh[0].get("metadata") or {}).get("terms_and_conditions") or ""))),
                            )
                    except Exception as _tc_err:
                        _LOG.debug("[Practical] TC-miss retrieval failed: %s", _tc_err)

        # Sub_topic completeness: semantic search returns top-k docs; for non-regulatory topics
        # (marketing/business_guide) all docs share a sub_topic where each row is one content_type.
        # When the query is about a specific sub_topic and all retrieved docs match that sub_topic,
        # supplement with ALL remaining docs from Chroma get() so the LLM sees every item.
        # Then aggregate into a single combined doc to keep token usage under control.
        if _all_docs and not _is_multi_license_docs and not _is_broad_q:
            _pr_lt_vals = {
                str((d.get("metadata") or {}).get("license_type") or "").strip()
                for d in _all_docs
            } - {"", "nan", "None"}
            _pr_st_vals = {
                str((d.get("metadata") or {}).get("sub_topic") or "").strip()
                for d in _all_docs
            } - {"", "nan", "None"}
            _pr_dt_vals = {
                str((d.get("metadata") or {}).get("data_type") or "").strip().lower()
                for d in _all_docs
            }
            if (
                not _pr_lt_vals                       # all non-regulatory
                and len(_pr_st_vals) == 1             # single sub_topic
                and _pr_dt_vals <= {"marketing", "business_guide", ""}
            ):
                _target_pr_st = next(iter(_pr_st_vals))
                try:
                    _vs_pr = getattr(self.retriever, "vectorstore", None)
                    _coll_pr = getattr(_vs_pr, "_collection", None) if _vs_pr else None
                    if _coll_pr is not None:
                        _st_pr_res = _coll_pr.get(
                            where={"sub_topic": _target_pr_st},
                            include=["documents", "metadatas"],
                        )
                        _st_pr_pairs = list(zip(
                            _st_pr_res.get("documents") or [],
                            _st_pr_res.get("metadatas") or [],
                        ))
                        if len(_st_pr_pairs) > len(_all_docs):
                            # Build supplemented list (add rows not already present)
                            _mc_pr = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700) or 700)
                            _ag_cap_pr = 400  # per-item cap for aggregated answer_guideline
                            _seen_pr = {hash((d.get("content") or "")[:200]) for d in _all_docs}
                            for _fc_pr, _fm_pr in _st_pr_pairs:
                                _slim_pr: Dict[str, Any] = {}
                                for _k_pr, _v_pr in (_fm_pr or {}).items():
                                    if _v_pr in (None, "", "nan", "None") or str(_v_pr) in ("nan", "None", ""):
                                        continue
                                    _slim_pr[_k_pr] = str(_v_pr)[:_ag_cap_pr] if _k_pr == "answer_guideline" else str(_v_pr)
                                _nd_pr = {"content": (_fc_pr or "")[:_mc_pr], "metadata": _slim_pr}
                                _fp_pr = hash((_fc_pr or "")[:200])
                                if _fp_pr not in _seen_pr:
                                    _all_docs.append(_nd_pr)
                                    _seen_pr.add(_fp_pr)
                            # Aggregate all docs into 1 combined doc — keeps token budget safe
                            # and ensures ALL content_types are visible to the LLM.
                            _items_pr: list = []
                            _refs_pr: list = []
                            _md0_pr = (_all_docs[0].get("metadata") or {})
                            for _i_pr, _sd_pr in enumerate(_all_docs, 1):
                                _smd_pr = _sd_pr.get("metadata") or {}
                                _ct_pr = str(_smd_pr.get("content_type") or "").strip()
                                _ag_pr = str(_smd_pr.get("answer_guideline") or "").strip()[:200]
                                _items_pr.append(f"{_i_pr}. {_ct_pr}: {_ag_pr}" if _ct_pr else f"{_i_pr}. {_ag_pr}")
                                _rr_pr = str(_smd_pr.get("research_reference") or "").strip()
                                if _rr_pr and _rr_pr not in _refs_pr:
                                    _refs_pr.append(_rr_pr)
                            _comb_content_pr = (
                                f"หัวข้อย่อย: {_target_pr_st}\n"
                                f"จำนวนประเภท: {len(_all_docs)} ประเภท\n"
                                + "\n".join(_items_pr)
                            )[:3000]
                            _comb_md_pr: Dict[str, Any] = {
                                k: v for k, v in _md0_pr.items()
                                if v not in (None, "", "nan", "None") and str(v) not in ("nan", "None", "")
                            }
                            _comb_md_pr["answer_guideline"] = (
                                f"มีทั้งหมด {len(_all_docs)} ประเภท: "
                                + ", ".join(
                                    str((_sd.get("metadata") or {}).get("content_type") or "").strip()
                                    for _sd in _all_docs
                                    if str((_sd.get("metadata") or {}).get("content_type") or "").strip()
                                )
                            )[:1500]
                            if _refs_pr:
                                _comb_md_pr["research_reference"] = "\n".join(_refs_pr[:5])
                            _aggregated_pr = [{"content": _comb_content_pr, "metadata": _comb_md_pr}]
                            state.current_docs = _aggregated_pr
                            _all_docs = _aggregated_pr
                            _LOG.info(
                                "[Practical] Sub_topic completeness: aggregated %d docs → 1 combined doc for %r",
                                len(_st_pr_pairs), _target_pr_st,
                            )
                except Exception as _pr_st_e:
                    _LOG.warning("[Practical] Sub_topic supplement failed: %s", _pr_st_e)

        # Chapter overview detection: when ALL docs share the same main_topic AND
        # data_type is marketing/business_guide → bypass the per-license cap so the LLM
        # sees ALL chapter docs and can cover every sub_topic. Without this, the cap of
        # LLM_DOCS_MAX_PRACTICAL=6 would silently drop docs 7-N, causing subtopics to go missing.
        _is_chapter_overview = False
        if _all_docs and not _is_multi_license_docs:
            _co_mts = {
                ((_d.get("metadata") or {}).get("main_topic") or "").strip()
                for _d in _all_docs
            }
            _co_dts = {
                ((_d.get("metadata") or {}).get("data_type") or "").strip().lower()
                for _d in _all_docs
            }
            if len(_co_mts) == 1 and next(iter(_co_mts)) and _co_dts <= {"marketing", "business_guide", ""}:
                _is_chapter_overview = True
                _LOG.info("[Practical] Chapter overview detected — bypassing doc cap (main_topic=%r, %d docs)", next(iter(_co_mts)), len(_all_docs))

        # Topic Focus Filter: when the query contains English tokens that uniquely identify
        # one topic among several in _all_docs (e.g. "WeChat Pay" in a set that also includes
        # "Refund" and "change-request" docs), narrow _all_docs to that topic + generic docs.
        # This prevents cross-operation contamination when user asks about a specific variant.
        if _all_docs and not _is_multi_license_docs and not _is_chapter_overview:
            _lrq_tf = (getattr(state, "last_retrieval_query", None) or "").strip()
            _q_tf_texts = [user_text.lower()]
            if _lrq_tf and _lrq_tf.lower() != user_text.lower():
                _q_tf_texts.append(_lrq_tf.lower())
            _topic_docs_tf: dict = {}  # topic_str → list[doc]
            _generic_docs_tf: list = []
            for _d_tf in _all_docs:
                _m_tf = _d_tf.get("metadata") or {}
                _t_tf = (str(_m_tf.get("topic") or "").strip() or
                         str(_m_tf.get("operation_topic") or "").strip())
                if not _t_tf or _t_tf in ("?", "nan", "None", "-"):
                    _generic_docs_tf.append(_d_tf)
                else:
                    _topic_docs_tf.setdefault(_t_tf, []).append(_d_tf)
            if len(_topic_docs_tf) > 1:
                # Score each topic by how many of its English tokens appear in the query.
                # Use the highest-scoring topic; ties broken by iteration order.
                _best_tf = ""
                _best_tf_count = 0
                for _t_cand_tf in _topic_docs_tf:
                    _eng_tf = [tok.lower() for tok in re.findall(r'[A-Za-z]{2,}', _t_cand_tf)]
                    if not _eng_tf:
                        continue
                    _hit_tf = sum(1 for tok in _eng_tf if any(tok in _q_tf for _q_tf in _q_tf_texts))
                    _miss_tf = sum(1 for tok in _eng_tf if not any(tok in _q_tf for _q_tf in _q_tf_texts))
                    # Score = hit - miss: penalises options with extra Latin tokens absent
                    # from the query (e.g. "PLUS" not in "SAN" query → SAN PLUS scores lower).
                    _score_tf = _hit_tf - _miss_tf
                    if _score_tf > _best_tf_count:
                        _best_tf_count = _score_tf
                        _best_tf = _t_cand_tf
                if _best_tf:
                    _focused_tf = _topic_docs_tf[_best_tf] + _generic_docs_tf
                    if 0 < len(_focused_tf) < len(_all_docs):
                        _LOG.info(
                            "[Practical] Topic focus filter matched %r — %d → %d docs",
                            _best_tf, len(_all_docs), len(_focused_tf),
                        )
                        _all_docs = _focused_tf
                        state.current_docs = _focused_tf

        # Cap at _prompt_max_docs per license_type (prevents token explosion on multi-topic queries)
        if _is_multi_license_docs:
            _lt_counts: dict = {}
            _docs_to_process = []
            for _d0 in _all_docs:
                _lt0 = ((_d0.get("metadata") or {}).get("license_type") or "").strip()
                _lt_counts[_lt0] = _lt_counts.get(_lt0, 0) + 1
                if _lt_counts[_lt0] <= _prompt_max_docs:
                    _docs_to_process.append(_d0)
        elif bool((state.context or {}).get("_direct_topic_match")):
            # Exact OBD/operation_topic chapter match — supervisor already fetched ALL sub-case
            # docs for this operation. Never cap: each doc is a distinct sub-case and dropping any
            # would produce an incomplete answer (e.g. 4 of 5 authority-restriction scenarios).
            _docs_to_process = _all_docs
            _LOG.info("[Practical] Direct topic match — sending all %d docs (no cap)", len(_all_docs))
        elif _is_chapter_overview:
            # Supplement: the reranker cap inside _retrieve_docs may have silently dropped
            # sub_topic docs that ranked lower than LLM_DOCS_MAX_PRACTICAL.
            # Re-retrieve with a main_topic metadata filter to recover any missing sub_topics.
            _co_mt = next(iter(_co_mts))
            try:
                _suppl = self._retrieve_docs(
                    _co_mt,
                    metadata_filter={"main_topic": _co_mt},
                    max_docs=15,
                )
                _seen_co = {hash((d.get("content") or "")[:200]) for d in _all_docs}
                _added_co = [d for d in _suppl if hash((d.get("content") or "")[:200]) not in _seen_co]
                if _added_co:
                    _all_docs = _all_docs + _added_co
                    state.current_docs = _all_docs
                    _LOG.info("[Practical] Chapter overview supplement: +%d docs → total %d", len(_added_co), len(_all_docs))
            except Exception as _co_e:
                _LOG.warning("[Practical] Chapter overview supplement failed: %s", _co_e)
            # Query-supplement: some specific docs (e.g. exemption rows) are stored as
            # license-specific docs (license_type set, main_topic=None) so the main_topic
            # filter above misses them. The chapter-overview path also uses main_topic as
            # the retrieval query, so BM25 never matches content-embedded topic names
            # (e.g. "หัวข้อ: การจดทะเบียนพาณิชย์ที่ได้รับการยกเว้น").
            # Fix: retry with the original user query so BM25 can surface those docs.
            _lrq_co = (getattr(state, "last_retrieval_query", None) or "").strip()
            if _lrq_co and _lrq_co.lower() != _co_mt.lower():
                try:
                    _suppl_q = self._retrieve_docs(_lrq_co, max_docs=3)
                    _seen_co2 = {hash((d.get("content") or "")[:200]) for d in _all_docs}
                    _added_q = [d for d in _suppl_q if hash((d.get("content") or "")[:200]) not in _seen_co2]
                    if _added_q:
                        _all_docs = _added_q + _all_docs  # specific docs first so LLM prioritises them
                        state.current_docs = _all_docs
                        _LOG.info(
                            "[Practical] Chapter overview query-supplement: +%d specific docs (query=%r)",
                            len(_added_q), _lrq_co[:60],
                        )
                except Exception as _lrq_e:
                    _LOG.warning("[Practical] Chapter overview query-supplement failed: %s", _lrq_e)
            # Send all chapter docs — sub_topics like BCG Matrix may span multiple docs
            # (Stars, Cash Cows, Dogs, Question Marks each in separate rows). Deduping by
            # sub_topic would drop those sibling docs and leave content incomplete.
            # Token control is handled instead by trimming content per doc in docs_json below.
            _docs_to_process = _all_docs
            _LOG.info("[Practical] Chapter overview: sending all %d docs (no dedup)", len(_all_docs))
        else:
            _docs_to_process = _all_docs[:_prompt_max_docs]

        # Detect operation intent from query — must run BEFORE doc-filter and Pass 1.
        # When user_text is a slot answer (e.g. "ธนาคารไทยพาณิชย์"), fall back to last_retrieval_query.

        # Helper: return the field that best represents the operation type for a doc.
        # Some licenses (e.g. ใบอนุญาตจำหน่ายสุรา) use operation_by_department as a license-level
        # label ("จำหน่ายสุรา") shared across ALL operations — not an op keyword. In that case
        # Python's "or" short-circuits and never reaches operation_topic, breaking the filter.
        # This helper detects when operation_by_department has no recognizable op keyword and
        # falls through to operation_topic so the intent filter works correctly.
        _OP_DEPT_KW_RE = re.compile(
            r"ต่ออายุ|ยกเลิก|เลิกกิจการ|แก้ไข|เปลี่ยนแปลง|จดทะเบียน|ยื่น(?:ขอ|ใบ)|ขอใบ|"
            r"เปิดใหม่|สิ้นสุด|สมัคร|ลงทะเบียน|แจ้ง(?:รับ|สิ้น)|โอนกิจการ|ใบแทน"
        )
        def _eff_op(d: dict) -> str:
            _by_dept = str((d.get("metadata") or {}).get("operation_by_department") or "").strip()
            _topic   = str((d.get("metadata") or {}).get("operation_topic") or "").strip()
            _fallbk  = str((d.get("metadata") or {}).get("topic") or "").strip()
            if _by_dept and _OP_DEPT_KW_RE.search(_by_dept):
                return _by_dept
            return _topic or _by_dept or _fallbk

        _op_exclude_re: Optional[str] = None
        _op_include_re: Optional[str] = None  # positive filter: keep only docs whose op matches this
        # Extra desc-level filter for Tier4 form links (beyond _op_exclude_re).
        # Used for replacement-cert queries that need to exclude new-registration forms.
        _op_excl_t4_desc_extra: str = ""
        _op_check_texts = [user_text]
        _lrq_op = (getattr(state, "last_retrieval_query", None) or "").strip()
        if _lrq_op and _lrq_op != user_text:
            _op_check_texts.append(_lrq_op)
        # Dual-op guard: user explicitly asks about BOTH แก้ไข AND ยกเลิก in the same query
        # e.g. "แก้ไข /ยกเลิกข้อมูลตำแหน่งรับสมัครงาน".
        # Problem: the for-loop below picks the FIRST matching pattern (แก้ไข), sets
        # _op_include_re="แก้ไข", and breaks — then sibling-completion kills DOC 2
        # (the combined edit+cancel doc) because OBD="การจัดหาพนักงาน" ≠ "แก้ไข".
        # Fix: when both ops are present, allow both edit and cancel docs; only exclude
        # new-registration/renewal (truly unrelated). Positive filter must be None so
        # sibling-completion can add the missing combined doc.
        _has_edit_kw = any(re.search(r"แก้ไข|เปลี่ยนแปลง", _t) for _t in _op_check_texts)
        _has_cancel_kw = any(re.search(r"เลิก|ยกเลิก|ปิดกิจการ|จดเลิก", _t) for _t in _op_check_texts)
        _dual_op_handled = False
        if _has_edit_kw and _has_cancel_kw:
            _op_exclude_re = r"เปิดใหม่|จดใหม่|ต่ออายุ"
            _op_include_re = None
            _dual_op_handled = True

        # Each tuple: (query_pattern, exclude_pattern, include_pattern)
        # include_pattern = positive filter applied to operation_by_department after exclude filter.
        # None means no positive filter (e.g. new-registration or replacement where op name ≠ query keyword).
        if not _dual_op_handled:
            for _op_q_pat, _op_excl_pat, _op_incl_pat in [
                (r"แก้ไข|เปลี่ยนแปลง",              r"เลิก|ยกเลิก|เปิดใหม่|ต่ออายุ",  r"แก้ไข|เปลี่ยนแปลง"),
                (r"เลิก|ยกเลิก|ปิดกิจการ|จดเลิก",  r"แก้ไข|เปลี่ยนแปลง|เปิดใหม่|ต่ออายุ", r"เลิก|ยกเลิก|ปิดกิจการ|จดเลิก"),
                # Replacement certificate queries (ชำรุด/สูญหาย/ใบแทน) matched BEFORE general pattern
                # so they inherit the new-reg op_exclude (exclude cancel/edit docs), but we also
                # add an extra Tier4 desc filter to prevent new-registration forms (บอจ.) from leaking.
                # No include_pat: operation_by_department names vary (e.g. "ใบแทน" may not appear verbatim).
                (r"ชำรุด|สูญหาย|ใบแทน",            r"เลิก|ยกเลิก|แก้ไข|เปลี่ยนแปลง|ต่ออายุ", None),
                # Renewal queries — matched BEFORE new-registration so "ต่ออายุ" beats "ขอใบ" overlap.
                (r"ต่ออายุ|ต้องต่อ(?:\s*ใบ|\s*อนุญาต|\s*ทะเบียน)?|ต่อ\s*ใบอนุญาต|ต่อ\s*ทะเบียน|หมดอายุ|สิ้นอายุ",
                 r"จดทะเบียน(?!เปลี่ยน)|ขอใบ(?!แทน)|สมัคร|เปิดร้าน|เปิดกิจการ|จดใหม่|เปิดใหม่", r"ต่ออายุ"),
                # New-registration: no include_pat — "การยื่น" is the target, positive filter not needed.
                (r"จดทะเบียน(?!เปลี่ยน)|จดภาษี|ขอใบ|สมัคร|ลงทะเบียน|เปิดร้าน|เปิดกิจการ|เปิดบัญชี|จดใหม่|ขึ้นทะเบียน|ต้องใช้|ต้องทำ", r"เลิก|ยกเลิก|แก้ไข|เปลี่ยนแปลง|ต่ออายุ|(?<![เ])ปิด|สิ้นสุด|เงินสมทบ", None),
            ]:
                if any(re.search(_op_q_pat, _t) for _t in _op_check_texts):
                    _op_exclude_re = _op_excl_pat
                    _op_include_re = _op_incl_pat
                    # For replacement-cert queries: Tier4 must also exclude new-registration form descs
                    if re.search(r"ชำรุด|สูญหาย|ใบแทน", _op_q_pat):
                        _op_excl_t4_desc_extra = r"จดทะเบียนบริษัท|บอจ\.|จัดตั้งบริษัท|หนังสือบริคณห์สนธิ|บัญชีรายชื่อผู้ถือหุ้น|รายงานการประชุมตั้งบริษัท"
                    break

        # Detect if this question type needs forms/document links.
        _needs_forms: bool = not any(bool(self._NO_FORMS_Q_RE.search(_t)) for _t in _op_check_texts)

        # Operation intent doc-filter: remove docs whose operation_topic OR operation_steps
        # conflicts with detected intent. BEFORE Pass 1 and before LLM prompt.
        # e.g. user asks "สมัครใหม่" → exclude docs with operation_topic containing "แก้ไข"/"เปลี่ยนแปลง".
        # Secondary check: some edit-operation rows use generic operation_topic (e.g. "รูปแบบ นิติบุคคล")
        # shared with new-registration rows, so topic check alone misses them. When the exclusion
        # pattern targets edit operations ("แก้ไข" in _op_exclude_re), also check if operation_steps
        # starts with the edit-portal UI pattern "1. เลือก … แก้ไข/เปลี่ยนแปลง" — that prefix is
        # unambiguous and only appears in edit-flow rows.
        # Guard: only apply when filter leaves at least one doc.
        _pre_opfilter_count = len(_docs_to_process)  # snapshot before op-intent filter
        # Docs to use for link collection — may differ from _docs_to_process when sibling-completion
        # adds off-topic docs (e.g. refund docs added as siblings in a new-registration query).
        # Initialized to None; set to pre-sibling snapshot inside the sibling-completion block.
        _docs_for_links: Optional[List] = None
        if _op_exclude_re:
            _excl_edit_steps = "แก้ไข" in _op_exclude_re  # True for new-reg, cancel, replacement queries
            _op_pass_docs = [
                d for d in _docs_to_process
                if (
                    # Rescue: keep doc if registration_type contains the user's intent keyword,
                    # even when operation_by_department is labeled as a different operation.
                    # Example: POS ID stores ยกเลิก/ไม่ได้ใช้/ระงับ inside one "แก้ไข เปลี่ยนแปลง"
                    # row via registration_type — without this, the ยกเลิก intent filter removes
                    # the only doc that actually contains cancellation information.
                    _op_include_re
                    and re.search(
                        _op_include_re,
                        str((d.get("metadata") or {}).get("registration_type") or ""),
                    )
                ) or not (
                    # Check operation_by_department first (holds the actual operation label e.g. "ยกเลิก",
                    # "ต่ออายุหนังสือรับรองการแจ้ง"); fall back to operation_topic then topic for licenses
                    # where the operation name lives in a different field (e.g. ทะเบียนผู้ประกันตน).
                    re.search(_op_exclude_re, _eff_op(d))
                    or (
                        _excl_edit_steps
                        and re.search(
                            r'1\.\s*เลือก.{0,15}(แก้ไข|เปลี่ยนแปลง)',
                            str((d.get("metadata") or {}).get("operation_steps") or "")[:120],
                        )
                    )
                    or (
                        # Catch change-registration rows whose operation_topic is generic
                        # (e.g. "การจดภาษีมูลค่าเพิ่ม" shared by both initial and change rows).
                        # The first line of identification_documents uniquely identifies the form type:
                        # change rows start with "แบบแจ้งการเปลี่ยนแปลง" / "ภ.พ.09".
                        _excl_edit_steps
                        and re.search(
                            r'แบบแจ้งการเปลี่ยนแปลง|ภ\.พ\.09|แบบแจ้งเปลี่ยนแปลง',
                            str((d.get("metadata") or {}).get("identification_documents") or "")[:150],
                        )
                    )
                )
            ]
            if _op_pass_docs:
                _LOG.info(
                    "[Practical] op-intent doc-filter: %d → %d docs (excl_re=%r)",
                    len(_docs_to_process), len(_op_pass_docs), _op_exclude_re,
                )
                _docs_to_process = _op_pass_docs
                # Level 1: update state.current_docs so Academic/Supervisor sees filtered docs too.
                # Without this, Academic Phase 3 menu reads unfiltered state.current_docs
                # and shows ต่ออายุ/แก้ไข sections even when user asked about สมัครใหม่.
                state.current_docs = _op_pass_docs

                # Level 1.2: Positive filter — when _op_include_re is set, keep only docs whose
                # operation_by_department matches the target operation. This prevents "incidental"
                # operations (e.g. "การยื่นใบอนุญาต" surviving the exclude filter when user asked
                # about "แก้ไขรายการ") from contaminating links and content in the LLM prompt.
                # Safety guard: only apply if at least 1 doc passes the include filter.
                if _op_include_re:
                    _incl_pass = [
                        d for d in _op_pass_docs
                        if re.search(_op_include_re, _eff_op(d))
                    ]
                    if _incl_pass:
                        _op_pass_docs = _incl_pass
                        _docs_to_process = _op_pass_docs
                        state.current_docs = _op_pass_docs

                # Snapshot entity types before sibling-completion for Level 1.6 injection check.
                # Sibling docs may introduce entity_type diversity that doesn't reflect the user's
                # actual query (e.g. VAT exemption query → sibling VAT-registration docs have
                # entity_type splits → without snapshot, entity_type question is asked unnecessarily).
                _et_pre_sibling = {
                    str((_d.get("metadata") or {}).get("entity_type_normalized") or "").strip()
                    for _d in _op_pass_docs
                }
                _et_pre_sibling.discard("")

                # Level 1.5: Sibling completion — fetch ALL docs sharing the same
                # (license_type, operation_by_department) as each surviving doc.
                # Problem: hybrid retrieval ranks sibling sub-topic docs together; the lower-ranked
                # sibling may fall outside top-k and disappear even though it's equally relevant.
                # Example: "การขึ้นทะเบียนผู้ประกันตน" has 2 sub-topics; retrieval returns only
                # "ผู้ที่เคยมีสิทธิรักษาพยาบาลแล้ว" — sibling "แจ้งรับผู้ประกันตนเข้าทำงานใหม่" is missing.
                # Fix: after op-intent filter, for each unique (lt, op) in surviving docs, fetch ALL
                # Chroma docs with that pair directly (no embedding — pure metadata filter).
                # Skip sibling-completion when the primary doc was found via exact operation_topic match
                # (_direct_topic_match): those docs are entity-neutral (no entity/reg_type variants) so
                # sibling-completion only pollutes them with unrelated registration forms and links.
                _skip_sib = bool((state.context or {}).get("_direct_topic_match"))
                _vs_for_sib = getattr(self.retriever, "vectorstore", None)
                if _vs_for_sib is not None and not _skip_sib:
                    # Docs from _retrieve_docs use key="content" (clipped).
                    # Use 200-char hash: sibling docs share the same header prefix (~85 chars)
                    # so [:80] causes hash collisions between them — [:200] safely diverges.
                    _seen_sib_hashes = {hash((d.get("content") or "")[:200]) for d in _op_pass_docs}
                    _sib_ops_checked: set = set()
                    _sib_new: List[Dict[str, Any]] = []
                    _sib_max_chars = int(getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700))
                    for _sd in list(_op_pass_docs):
                        _sm = _sd.get("metadata") or {}
                        _slt = str(_sm.get("license_type") or "").strip()
                        _sop = str(_sm.get("operation_by_department") or "").strip()
                        if not _slt or not _sop or (_slt, _sop) in _sib_ops_checked:
                            continue
                        _sib_ops_checked.add((_slt, _sop))
                        try:
                            _sib_reg = ""
                            if hasattr(state, "get_collected_slot"):
                                _sib_reg = str(state.get_collected_slot("registration_type") or "").strip()
                            # Department filter: when primary doc belongs to a specific bank/authority,
                            # restrict siblings to the same department so cross-bank docs (e.g. กสิกรไทย
                            # siblings included in a ไทยพาณิชย์ answer) don't inflate token count or
                            # pollute links/content with wrong-bank data.
                            _sib_dept = str(_sm.get("department") or "").strip()
                            _dept_clauses = (
                                [{"department": _sib_dept}]
                                if _sib_dept and _sib_dept.lower() not in ("nan", "none")
                                else []
                            )
                            # Entity-type filter: include entity-neutral docs (entity_type_normalized='')
                            # but exclude docs belonging to the competing entity type.
                            # Without this, sibling-completion fetches ALL entity variants and floods
                            # the answer with wrong-entity forms (e.g. บริษัทจำกัด forms when user
                            # asked about บุคคลธรรมดา).
                            _sib_et = _query_et_override or str(
                                (state.context or {}).get("collected_slots", {}).get("entity_type") or ""
                            ).strip()
                            _entity_clauses = (
                                [{"$or": [
                                    {"entity_type_normalized": _sib_et},
                                    {"entity_type_normalized": ""},
                                ]}]
                                if _sib_et and _sib_et.lower() not in ("nan", "none")
                                else []
                            )
                            if _sib_reg and _sib_reg.lower() not in ("nan", "none"):
                                _sib_where: dict = {"$and": [
                                    {"license_type": _slt},
                                    {"operation_by_department": _sop},
                                    {"$or": [{"registration_type": _sib_reg}, {"registration_type": ""}]},
                                ] + _dept_clauses + _entity_clauses}
                            else:
                                _sib_where = {"$and": [
                                    {"license_type": _slt},
                                    {"operation_by_department": _sop},
                                ] + _dept_clauses + _entity_clauses}
                            _sr = _vs_for_sib._collection.get(
                                    where=_sib_where,
                                    include=["metadatas", "documents"],
                                )
                            for _spc, _smd in zip(_sr.get("documents") or [], _sr.get("metadatas") or []):
                                # Include registration_type in hash: docs with identical content
                                # but different registration_type (e.g. บริษัทจำกัด vs ห้างหุ้นส่วน
                                # sharing first 547 chars) must both be kept.
                                _sh = hash(
                                    (_spc or "")[:200]
                                    + str((_smd or {}).get("registration_type", ""))
                                )
                                if _sh not in _seen_sib_hashes:
                                    _seen_sib_hashes.add(_sh)
                                    # Apply same cleanup as _retrieve_docs: strip None/"nan" values.
                                    _clean_smd = {
                                        k: str(v) for k, v in (_smd or {}).items()
                                        if v not in (None, "", "nan", "None") and str(v) not in ("nan", "None", "")
                                    }
                                    _sib_new.append({"content": (_spc or "")[:_sib_max_chars], "metadata": _clean_smd})
                        except Exception as _e_sib:
                            _LOG.debug("[Practical] sibling-completion fetch failed for op=%r: %s", _sop, _e_sib)
                    if _sib_new:
                        # Apply op-intent exclusion to new siblings to stay consistent.
                        if _op_exclude_re:
                            _sib_new = [
                                d for d in _sib_new
                                if not re.search(_op_exclude_re, _eff_op(d))
                            ]
                        # Apply positive op-intent filter to siblings too — prevents re-adding
                        # docs from OTHER operations when operation_by_department is a generic
                        # license label (e.g. "จำหน่ายสุรา" shared by all สุรา operations).
                        # NOTE: no safety guard — if no siblings match the positive filter,
                        # _sib_new becomes [] and the outer `if _sib_new:` skips the add entirely.
                        if _op_include_re and _sib_new:
                            _sib_new = [
                                d for d in _sib_new
                                if re.search(_op_include_re, _eff_op(d))
                            ]
                        # Post-sibling filter: exclude siblings whose operation_topic
                        # is a strict superset of the primary doc's operation_topic
                        # with extra terms absent from the user's query.
                        # Example: primary='มาตรฐาน SAN', sibling='มาตรฐาน SAN PLUS'
                        # → 'PLUS' not in user query → exclude SAN PLUS sibling.
                        # Does NOT affect Thai-only operation_topics (no spaces → single token,
                        # superset check fails trivially → all siblings kept).
                        if _sib_new and _op_pass_docs:
                            _prim_ot = str((_op_pass_docs[0].get("metadata") or {}).get("operation_topic") or "").strip()
                            if _prim_ot:
                                _prim_ot_words = set(_prim_ot.lower().split())
                                _query_tok = {w.lower() for w in re.split(r'[\s()\[\].,]+', user_text) if w}
                                _sib_filtered2: List[Dict[str, Any]] = []
                                for _sib_d2 in _sib_new:
                                    _sib_ot2 = str((_sib_d2.get("metadata") or {}).get("operation_topic") or "").strip()
                                    if _sib_ot2 and _sib_ot2 != _prim_ot:
                                        _sib_ot2_words = set(_sib_ot2.lower().split())
                                        _extra2 = _sib_ot2_words - _prim_ot_words
                                        # Guard: only exclude when extra words contain Latin chars
                                        # (e.g. "PLUS", "V2"). Thai-only extras are meaningful
                                        # sub-case descriptors that should not be filtered out.
                                        _extra2_has_latin = any(
                                            any("a" <= c <= "z" for c in w) for w in _extra2
                                        )
                                        if (_prim_ot_words < _sib_ot2_words and _extra2
                                                and _extra2_has_latin
                                                and not (_extra2 & _query_tok)):
                                            _LOG.debug("[Practical] sibling-filter: excluded superset-variant op_topic=%r (primary=%r)", _sib_ot2, _prim_ot)
                                            continue
                                    _sib_filtered2.append(_sib_d2)
                                _sib_new = _sib_filtered2
                        # Entity sub-type content filter: when user selected a specific
                        # registration_type (e.g. บริษัทจำกัด), drop empty-rt siblings whose
                        # document header explicitly names a DIFFERENT entity sub-type.
                        # Root cause: many docs for การจดทะเบียนพาณิชย์ have empty
                        # registration_type metadata but their "หัวข้อ:" header says
                        # "บริษัทมหาชนจำกัด" or "ห้างหุ้นส่วน" — irrelevant to the user's case.
                        if _sib_new and _sib_reg:
                            # Keyword sets for competing entity types
                            _COMPETING_KWS: Dict[str, re.Pattern] = {
                                "มหาชน": re.compile(r"มหาชน"),
                                "ห้างหุ้นส่วน": re.compile(r"ห้างหุ้นส่วน"),
                                "บุคคลธรรมดา": re.compile(r"บุคคลธรรมดา"),
                                "บริษัทจำกัด": re.compile(r"บริษัทจำกัด"),
                            }
                            # Find which keywords should be EXCLUDED (those not in user's selection)
                            _excl_pats = [
                                pat for kw, pat in _COMPETING_KWS.items()
                                if not re.search(kw, _sib_reg, re.IGNORECASE)
                            ]
                            # Extract ONLY the "หัวข้อ:" topic name from doc header (stops at
                            # "หน่วยงาน:" which immediately follows the topic in the format).
                            # Do NOT check the full body — comparison sentences may mention
                            # other entity types legitimately (e.g. "หนังสือบริคณห์สนธิ"
                            # describes บริษัทจำกัด but references มหาชน for comparison).
                            _TOPIC_FIELD_RE = re.compile(r'หัวข้อ:\s*(.+?)(?=\s*หน่วยงาน:|$)', re.DOTALL)
                            if _excl_pats:
                                _sib_entity_filtered: List[Dict[str, Any]] = []
                                for _s_d in _sib_new:
                                    _s_rt = str((_s_d.get("metadata") or {}).get("registration_type", "")).strip()
                                    if _s_rt:
                                        # Has specific registration_type — keep only when it matches
                                        # selected type via substring: "ห้างหุ้นส่วน" matches
                                        # "1.ห้างหุ้นส่วนจำกัด 2.ห้างหุ้นส่วนสามัญ", and vice versa.
                                        if _sib_reg in _s_rt or _s_rt in _sib_reg:
                                            _sib_entity_filtered.append(_s_d)
                                        # else: competing registration_type — drop
                                    else:
                                        # Empty rt — check ONLY "หัวข้อ:" field for competing type
                                        _s_hdr = (_s_d.get("content") or "")[:400]
                                        _topic_m = _TOPIC_FIELD_RE.search(_s_hdr)
                                        _topic_txt = _topic_m.group(1).strip() if _topic_m else ""
                                        if not any(p.search(_topic_txt) for p in _excl_pats):
                                            _sib_entity_filtered.append(_s_d)
                                _dropped_entity = len(_sib_new) - len(_sib_entity_filtered)
                                if _dropped_entity:
                                    _LOG.info(
                                        "[Practical] sibling entity-filter: dropped %d wrong-entity docs "
                                        "(registration_type=%r selected)",
                                        _dropped_entity, _sib_reg,
                                    )
                                # Sort: specific registration_type first, then by content richness
                                # (ops+idd+sc+od present = richest), so LLM sees best docs early.
                                def _sib_sort_key(_d):
                                    _sm2 = _d.get("metadata") or {}
                                    _rt_match = 0 if str(_sm2.get("registration_type", "")).strip() else 1
                                    _richness = -(
                                        bool((_sm2.get("operation_steps") or "").strip())
                                        + bool((_sm2.get("identification_documents") or "").strip())
                                        + bool((_sm2.get("service_channel") or "").strip())
                                        + bool((_sm2.get("operation_duration") or "").strip())
                                    )
                                    return (_rt_match, _richness)
                                _sib_new = sorted(_sib_entity_filtered, key=_sib_sort_key)
                                # Dedup by operation_topic: keep only the richest doc per topic
                                # to prevent 10+ identical-topic docs from inflating token count.
                                _sib_seen_ot: dict = {}
                                _sib_deduped: list = []
                                for _sd2 in _sib_new:
                                    _sot2 = str((_sd2.get("metadata") or {}).get("operation_topic") or "").strip()
                                    if _sot2 and _sot2 in _sib_seen_ot:
                                        continue  # duplicate topic — first (richest) already kept
                                    _sib_deduped.append(_sd2)
                                    if _sot2:
                                        _sib_seen_ot[_sot2] = True
                                _sib_new = _sib_deduped
                                # Cap total sibling docs to avoid context overwhelm
                                _SIBLING_CAP = int(getattr(conf, "SIBLING_DOCS_CAP", 10))
                                if len(_sib_new) > _SIBLING_CAP:
                                    _LOG.info("[Practical] sibling cap: %d → %d docs", len(_sib_new), _SIBLING_CAP)
                                    _sib_new = _sib_new[:_SIBLING_CAP]
                        if _sib_new:
                            _LOG.info(
                                "[Practical] sibling-completion: added %d doc(s) for op_by_dept(s)=%r",
                                len(_sib_new), [o for _, o in _sib_ops_checked],
                            )
                            # Freeze link-collection scope.
                            # When registration_type entity-filter was applied (_sib_reg set),
                            # siblings are confirmed correct entity/RT — include them so that
                            # form/guide links from those siblings appear in the first answer.
                            # When entity_type is already known (collected slot), siblings from the
                            # same (license_type, operation_by_department) are safe to include:
                            # they share the same forms/PDFs regardless of entity variant.
                            # Without either: keep pre-sibling snapshot (conservative).
                            _et_for_sib = str(
                                (state.context.get("collected_slots") or {}).get("entity_type") or ""
                            ).strip() if state.context else ""
                            if _sib_reg or (_et_for_sib and _et_for_sib.lower() not in ("nan", "none")):
                                _docs_for_links = list(_op_pass_docs) + list(_sib_new)
                            else:
                                _docs_for_links = list(_op_pass_docs)
                            _op_pass_docs = _op_pass_docs + _sib_new
                            _docs_to_process = _op_pass_docs
                            state.current_docs = _op_pass_docs

                # Level 1.6: entity_type injection — when op-filtered docs (PRE-sibling snapshot)
                # contain both entity-specific variants (บุคคลธรรมดา and นิติบุคคล) and entity_type
                # is not yet known, prepend entity_type to topic_slot_queue so it's asked before answering.
                # Uses _et_pre_sibling (snapshot before Level 1.5) — sibling docs may introduce entity
                # diversity that belongs to a different operation (e.g. VAT exemption query: the answer
                # doc is entity-neutral, but sibling VAT-registration docs have entity splits → should
                # NOT trigger the question).
                _et_in_docs = {
                    str((_d.get("metadata") or {}).get("entity_type_normalized") or "").strip()
                    for _d in _docs_to_process
                }
                _et_in_docs.discard("")
                if len(_et_pre_sibling) >= 2 and not state.get_collected_slot("entity_type") and not (state.context or {}).get("_informational_query") and not (state.context or {}).get("_link_request_query") and not (state.context or {}).get("_broad_question") and not (state.context or {}).get("_direct_topic_match"):
                    _sq_now = list((state.context or {}).get("topic_slot_queue") or [])
                    if not any(s.get("key") == "entity_type" for s in _sq_now):
                        _et_display = {"บุคคลธรรมดา": "บุคคลธรรมดา", "นิติบุคคล": "นิติบุคคล"}
                        state.context["topic_slot_queue"] = [{
                            "key": "entity_type",
                            "options": sorted(_et_in_docs),
                            "display_options": [_et_display.get(v, v) for v in sorted(_et_in_docs)],
                            "question": "ร้านของคุณดำเนินการในรูปแบบใดครับ",
                        }] + _sq_now
                        _LOG.info(
                            "[Practical] entity_type injected into slot_queue — op docs have entity types: %s",
                            sorted(_et_in_docs),
                        )

        # Level 1.7: Prune slot_queue operation_topic options that no longer exist after op-intent
        # filter. Without this, "ยกเลิก POS ID" → filter leaves 1 doc with no operation_topic
        # matching the 3 "แก้ไข" sub-topics in the queue → bot asks meaningless menu options.
        if _op_exclude_re:
            _sq_prune = list((state.context or {}).get("topic_slot_queue") or [])
            if _sq_prune:
                _remaining_op_topics = {
                    str((d.get("metadata") or {}).get("operation_topic") or "").strip()
                    for d in _docs_to_process
                }
                _remaining_op_topics.discard("")
                _sq_pruned: list = []
                for _sq_e in _sq_prune:
                    if _sq_e.get("key") == "operation_topic":
                        _valid_opts = [o for o in (_sq_e.get("options") or []) if o in _remaining_op_topics]
                        if _valid_opts:
                            _sq_pruned.append({**_sq_e, "options": _valid_opts, "display_options": _valid_opts})
                        # else: no valid options left → drop slot, let LLM answer from remaining docs
                    else:
                        _sq_pruned.append(_sq_e)
                if len(_sq_pruned) != len(_sq_prune):
                    _LOG.info(
                        "[Practical] slot_queue pruned after op-intent: %d → %d entries (remaining topics=%r)",
                        len(_sq_prune), len(_sq_pruned), sorted(_remaining_op_topics),
                    )
                    state.context["topic_slot_queue"] = _sq_pruned

        # Level 2: when op-intent filter ACTUALLY REDUCED docs to < 2 and entity_type is known,
        # do a focused entity-filtered re-retrieval to restore context for the LLM.
        # Rationale: supervisor's broad retrieval often returns many edit/cancel docs that get
        # filtered out — leaving too little context for the LLM. Entity-filtered retrieval
        # (as the LLM would do if it chose action='retrieve') gives much better docs.
        # Guard: only fire when op-intent genuinely reduced the count (_pre_opfilter_count >
        # current count). If supervisor already retrieved a narrow correct set (e.g. 1 doc via
        # entity+dept filter) and op-intent removed nothing, the existing docs are correct and
        # refetch is not needed — triggering it would drop the dept constraint and pull in
        # unrelated docs from other departments, causing wrong-bank answers and token bloat.
        # _query_et_override and _stored_et were computed above in the entity-switch detection block.
        _refetch_et = (_query_et_override or _stored_et).strip()
        if len(_docs_to_process) < 2 and _refetch_et and _op_exclude_re and _pre_opfilter_count > len(_docs_to_process):
            # Use the stored topic query (not the slot-change phrase) so vector search
            # ranks relevant docs first — e.g. "วิธีการลงทะเบียน QR-Payment API" >> "แล้วถ้าเป็นไทยพานิชอ่ะคะ"
            _refetch_query = str(getattr(state, "last_retrieval_query", None) or "").strip() or user_text
            # Replace stale entity term in query so embeddings align with new entity
            # e.g. "วิธีลงทะเบียน QR-Payment API นิติบุคคล" → replace "นิติบุคคล" with "บุคคลธรรมดา"
            _old_et = "นิติบุคคล" if _refetch_et == "บุคคลธรรมดา" else "บุคคลธรรมดา"
            _refetch_query = re.sub(rf"\b{re.escape(_old_et)}\b", _refetch_et, _refetch_query)
            # Preserve department constraint from existing docs or collected_slots.
            # Without this, refetch drops to entity-only filter and returns docs from ALL departments —
            # causing wrong-bank answer and 36K token bloat.
            _refetch_dept = ""
            if _docs_to_process:
                _refetch_dept = str((_docs_to_process[0].get("metadata") or {}).get("department") or "").strip()
            if not _refetch_dept:
                _refetch_dept = str((state.context.get("collected_slots") or {}).get("department") or "").strip()
            if _refetch_dept:
                _refetch_filter: dict = {"$and": [{"entity_type_normalized": _refetch_et}, {"department": _refetch_dept}]}
            else:
                _refetch_filter = {"entity_type_normalized": _refetch_et}
            _LOG.info("[Practical] op-intent left %d doc — refetching with entity_type=%r dept=%r", len(_docs_to_process), _refetch_et, _refetch_dept)
            _refetch_docs = self._retrieve_docs(
                _refetch_query,
                metadata_filter=_refetch_filter,
                slot_context={"entity_type": _refetch_et},
            )
            if _refetch_docs:
                # Apply same op-intent filter to re-fetched docs
                _refetch_pass = [
                    d for d in _refetch_docs
                    if not re.search(_op_exclude_re, _eff_op(d))
                ]
                if _refetch_pass:
                    _LOG.info("[Practical] refetch → %d docs (entity=%r)", len(_refetch_pass), _refetch_et)
                    _docs_to_process = _refetch_pass
                    state.current_docs = _refetch_pass

        # Re-sort _docs_to_process: richest docs first (has ops+idd+sc+od), specific registration_type
        # before generic. Fixes "lost in the middle" — LLM pays more attention to early positions.
        # Only applies when multiple docs are present (no-op for single doc).
        if len(_docs_to_process) > 1:
            _collected_rt = (
                state.get_collected_slot("registration_type")
                if hasattr(state, "get_collected_slot") else None
            ) or ""
            def _doc_rank(_d):
                _dm = _d.get("metadata") or {}
                _rt_d = str(_dm.get("registration_type") or "").strip()
                _rt_match = 0 if (_collected_rt and (_collected_rt in _rt_d or _rt_d in _collected_rt)) else (1 if not _rt_d else 2)
                _richness = -(
                    bool((_dm.get("operation_steps") or "").strip())
                    + bool((_dm.get("identification_documents") or "").strip())
                    + bool((_dm.get("service_channel") or "").strip())
                    + bool((_dm.get("operation_duration") or "").strip())
                )
                return (_rt_match, _richness)
            _docs_to_process = sorted(_docs_to_process, key=_doc_rank)
            state.current_docs = _docs_to_process

        # Pass 1: classify research_reference + restaurant_ai_document links → SERVICE / FORM / GUIDE / REF
        # Same _classify_link logic as Academic — hybrid desc+URL classification, no URL pattern rules in prompt
        _link_service: list = []  # (desc, url) registration/portal links
        _link_form: list = []     # (desc, url) fillable form links
        _link_guide: list = []    # (desc, url) guide/manual links — shown only when user asks
        _link_ref: list = []      # (desc, url) reference/FAQ links — shown only when user asks อ้างอิง
        _link_seen: set = set()        # global URL-based dedup key
        _link_seen_desc: set = set()   # (category, cleaned_desc) dedup — one URL per unique desc
        _url_to_license: dict = {}  # url → license_type of source doc (for multi-topic tagging)
        _url_to_doc_idx: Dict[str, int] = {}  # url → 0-based index in link-source docs (for used_doc_indices filtering)

        # Registration_type known from collected_slots — used to skip links from docs that
        # explicitly cover OTHER registration types (e.g. ห้างหุ้นส่วน forms when user asked
        # about บริษัทมหาชนจำกัด).  Empty string = RT not yet known → no filtering.
        _cs_rt = str(state.get_collected_slot("registration_type") or "").strip() if hasattr(state, "get_collected_slot") else ""

        def _clean_link_desc(desc: str) -> str:
            """Strip redundant 'Website'/'เว็บไซต์' prefix from link descriptions.
            Must strip leading bullet chars first so '• Website foo' → 'foo'."""
            if not desc:
                return desc
            d = re.sub(r'^[•\-\*]\s*', '', desc.strip())
            return re.sub(r'^(?:website|เว็บไซต์)\s+', '', d, flags=re.IGNORECASE).strip()

        # entity_type already collected? Used below to suppress entity-specific forms
        # when we don't yet know which entity the user is.
        # _obd_entity_override: single-turn override set by supervisor when Step 0b retrieves an
        # entity-specific concept doc (e.g. ตราประทับ = นิติบุคคล only) without matching the session
        # entity. Overrides _cs_et for this answer only; cleared on next retrieval call.
        _cs_et = (
            str((state.context or {}).get("_obd_entity_override") or "").strip()
            or (str(state.get_collected_slot("entity_type") or "").strip() if hasattr(state, "get_collected_slot") else "")
        )

        # High-level entity-class RT values (e.g. "นิติบุคคล", "บุคคลธรรมดา") used in datasets like
        # QR Payment / EDC where registration_type stores entity class, not legal sub-type.
        # A doc whose RT is an entity-class label is COMPATIBLE with any specific sub-type
        # (e.g. "บริษัทจำกัด") — do NOT apply the subtype-conflict filter to these docs.
        _ENTITY_CLASS_RTS = {"นิติบุคคล", "บุคคลธรรมดา"}
        # Always enumerate _docs_to_process (post-sort, post-sibling) — this is exactly what
        # docs_json is built from, so LLM's used_doc_indices map to these same positions.
        # Previously used _docs_for_links (pre-sibling snapshot) which caused index drift: LLM
        # cited sibling doc at index N but _url_to_doc_idx had a different doc at that index.
        for _d1_idx, _d1 in enumerate(_docs_to_process):
            # Registration_type conflict check: if user's RT is known and this doc's RT is
            # non-empty but doesn't include the user's RT, skip all links from this doc.
            # A doc with empty RT is entity/RT-neutral — always include its links.
            # Exception: if the doc's RT is a high-level entity-class label ("นิติบุคคล" /
            # "บุคคลธรรมดา"), it is compatible with any specific sub-type and must not be
            # filtered — e.g. QR Payment docs store rt="นิติบุคคล", not "บริษัทจำกัด".
            _doc_rt_d1 = str((_d1.get("metadata") or {}).get("registration_type") or "").strip()
            if _cs_rt and _doc_rt_d1 and _doc_rt_d1 not in _ENTITY_CLASS_RTS and _cs_rt not in _doc_rt_d1:
                _LOG.debug(
                    "[Practical] link-filter: skipping links from doc rt=%r (user rt=%r)",
                    _doc_rt_d1[:60], _cs_rt,
                )
                continue
            # data_type of this doc — used to distinguish regulatory from non-regulatory links.
            _doc_data_type = str((_d1.get("metadata") or {}).get("data_type") or "").strip()
            _is_non_reg_doc = _doc_data_type in ("business_guide", "marketing")
            # entity_type_normalized of this doc — non-empty means this doc is entity-specific.
            _doc_et_d1 = str((_d1.get("metadata") or {}).get("entity_type_normalized") or "").strip()
            # entity_type not yet known AND doc is entity-specific → skip form links from this doc.
            # Prevents e.g. ห้างหุ้นส่วน forms from appearing before the user has stated their entity.
            _skip_entity_forms = bool(_doc_et_d1 and _doc_et_d1 not in ("nan", "None") and not _cs_et)
            # Collect from both research_reference and restaurant_ai_document
            _rr_raw = str((_d1.get("metadata") or {}).get("research_reference") or "").strip()
            _ai_doc = str((_d1.get("metadata") or {}).get("restaurant_ai_document") or "").strip()
            _lt_d1 = str((_d1.get("metadata") or {}).get("license_type") or "").strip()
            for _src in [_rr_raw, _ai_doc]:
                if not _src or _src in ("nan", "None"):
                    continue
                for _desc1, _url1 in _parse_link_entries(_src):
                    _key1 = (_url1 or _desc1).strip()
                    if not _key1 or _key1 in _link_seen:
                        continue
                    _link_seen.add(_key1)
                    _desc1 = _clean_link_desc(_desc1)
                    if _url1 and _lt_d1:
                        _url_to_license[_url1] = _lt_d1
                    _cat1 = _classify_link(_desc1, _url1)
                    # Dedup by (category, cleaned desc): same desc = same link, show only first URL
                    _desc_key = (_cat1, (_desc1 or "").strip().lower())
                    if _desc1 and _desc_key in _link_seen_desc:
                        continue
                    if _desc1:
                        _link_seen_desc.add(_desc_key)
                    if _cat1 == "registration":
                        _link_service.append((_desc1, _url1))
                        if _url1: _url_to_doc_idx.setdefault(_url1, _d1_idx)
                    elif _cat1 == "form":
                        # Skip form links from business_guide / marketing docs — their "forms" are
                        # retail management worksheets, not legal filings.
                        if _is_non_reg_doc:
                            continue
                        # Skip entity-specific forms when entity_type not yet collected.
                        # Prevents e.g. ห้างหุ้นส่วน PDF forms from appearing for a general
                        # "การจดทะเบียนพาณิชย์" query before entity type is known.
                        if _skip_entity_forms:
                            _LOG.debug(
                                "[Practical] link-filter: skipping entity-specific form (doc_et=%r, cs_et=''): desc=%r",
                                _doc_et_d1, (_desc1 or "")[:60],
                            )
                            continue
                        # Per-link filter: check desc text AND URL for operation-conflict signals.
                        # Needed because neutral-topic docs (e.g. "บุคคลธรรมดา กิจการเจ้าของคนเดียว")
                        # pass the doc-level filter but their research_reference lists links for ALL
                        # operations (new/edit/close). We must filter at link level too.
                        if _op_exclude_re:
                            # Check Thai desc keywords
                            if re.search(_op_exclude_re, (_desc1 or "")):
                                continue
                            # Check English URL keywords (e.g. "new_registration.pdf" vs "edit_registration.pdf")
                            # Map each excl_re to the English URL patterns that indicate excluded operations.
                            _url_excl_map = {
                                r"เลิก|ยกเลิก|เปิดใหม่":           r"close_registration|new_registration",
                                r"แก้ไข|เปลี่ยนแปลง|เปิดใหม่":     r"edit_registration|new_registration",
                                r"เลิก|ยกเลิก|แก้ไข|เปลี่ยนแปลง": r"close_registration|edit_registration",
                            }
                            _url_excl_pat = _url_excl_map.get(_op_exclude_re, "")
                            if _url_excl_pat and re.search(_url_excl_pat, (_url1 or "")):
                                continue
                        _link_form.append((_desc1, _url1))
                        if _url1: _url_to_doc_idx.setdefault(_url1, _d1_idx)
                    elif _cat1 == "guide":
                        # business_guide / marketing docs: move their guide links to ref so they
                        # only appear when the user explicitly asks for sources/references.
                        # e.g. "Makro 8 ขั้นตอนสู่ความสำเร็จ" has no place in a registration answer.
                        if _is_non_reg_doc:
                            _link_ref.append((_desc1, _url1))
                            _LOG.debug(
                                "[Practical] link→ref (non-reg doc, guide link): desc=%r",
                                (_desc1 or "")[:60],
                            )
                            continue
                        # Skip entity-specific guide links when entity_type not yet collected.
                        # e.g. "คู่มือ DBD Biz Regist จัดตั้งห้างหุ้นส่วนจำกัด" must not appear
                        # in a general overview answer before entity type is known.
                        if _skip_entity_forms:
                            _LOG.debug(
                                "[Practical] link-filter: skipping entity-specific guide (doc_et=%r, cs_et=''): desc=%r",
                                _doc_et_d1, (_desc1 or "")[:60],
                            )
                            continue
                        # Per-link filter: same operation-conflict check as form links.
                        # e.g. "คู่มือ-การต่ออายุ เปลี่ยนแปลง ยกเลิก" → excluded for new-registration query.
                        if _op_exclude_re and re.search(_op_exclude_re, (_desc1 or "")):
                            continue
                        # Per-link registration sub-type filter: skip guide links that explicitly
                        # mention a DIFFERENT sub-type when user's RT is known.
                        # e.g. "คู่มือจัดตั้งห้างหุ้นส่วนสามัญ" skipped for ห้างหุ้นส่วนจำกัด query.
                        if _cs_rt and _desc1:
                            _GUIDE_RT_SUBTYPES = (
                                "บริษัทจำกัด", "บริษัทมหาชนจำกัด",
                                "ห้างหุ้นส่วนจำกัด", "ห้างหุ้นส่วนสามัญ",
                            )
                            if any(
                                _lrt in _desc1 and _lrt not in _cs_rt and _cs_rt not in _lrt
                                for _lrt in _GUIDE_RT_SUBTYPES
                            ):
                                _LOG.debug(
                                    "[Practical] link-filter: skipping guide mismatched rt (user_rt=%r): desc=%r",
                                    _cs_rt, _desc1[:60],
                                )
                                continue
                        _link_guide.append((_desc1, _url1))
                        if _url1: _url_to_doc_idx.setdefault(_url1, _d1_idx)
                    else:  # ref — kept, injected only when user explicitly asks
                        _link_ref.append((_desc1, _url1))
                        if _url1: _url_to_doc_idx.setdefault(_url1, _d1_idx)
                        _LOG.debug(
                            "[Practical] link→ref (will show only if user asks อ้างอิง): desc=%r url=%r",
                            _desc1[:60] if _desc1 else "", _url1[:80] if _url1 else "",
                        )

        # Pass 1b: extract inline URLs from non-reference metadata fields (service_channel,
        # operation_steps). The prompt instructs the LLM to omit these URLs from the body text,
        # but we surface them here so they appear in the 🌐 links section via the safety net.
        _INLINE_URL_RE = re.compile(
            r"(https?://[^\s฀-๿]{8,}|www\.[a-zA-Z0-9][\w.\-]+(?:/[^\s฀-๿]*)?)",
            re.IGNORECASE,
        )
        _INLINE_SCAN_FIELDS = ("service_channel", "operation_steps")
        for _di in (_docs_for_links if _docs_for_links is not None else _docs_to_process):
            for _isf in _INLINE_SCAN_FIELDS:
                _isf_val = str((_di.get("metadata") or {}).get(_isf) or "").strip()
                if not _isf_val or _isf_val in ("nan", "None"):
                    continue
                for _m in _INLINE_URL_RE.finditer(_isf_val):
                    _raw_url = _m.group(1).rstrip(".,;)")
                    if _raw_url.lower().startswith("www."):
                        _raw_url = "https://" + _raw_url
                    if _raw_url in _link_seen:
                        continue
                    _link_seen.add(_raw_url)
                    _link_service.append(("", _raw_url))
                    _LOG.debug("[Practical] inline URL extracted from %r field: %r", _isf, _raw_url[:80])

        # Determine the doc pool for flag computation.
        # Problem: retrieval can return a mix of docs from different topics (e.g., a contact-info doc
        # retrieved alongside registration docs for the same department).  Computing flags from ALL
        # _docs_to_process causes false positives — the registration docs have operation_steps /
        # identification_documents that should NOT trigger link injection for the contact query.
        #
        # Fix: narrow to the primary sub_topic cluster (docs sharing the same sub_topic as the
        # top-ranked doc).  Docs with a blank sub_topic are included unconditionally (no constraint).
        # Source: use _docs_for_links (pre-sibling snapshot) when sibling-completion ran; otherwise
        # fall back to _docs_to_process so single-retrieval paths are unchanged.
        _flag_docs_src = _docs_for_links if _docs_for_links is not None else _docs_to_process
        _primary_sub_topic_flag = ""
        if _flag_docs_src:
            _primary_sub_topic_flag = str(
                (_flag_docs_src[0].get("metadata") or {}).get("sub_topic") or ""
            ).strip()
            if _primary_sub_topic_flag in ("nan", "None"):
                _primary_sub_topic_flag = ""
        _flag_docs = _flag_docs_src
        if _primary_sub_topic_flag:
            _st_cluster = [
                d for d in _flag_docs_src
                if str((d.get("metadata") or {}).get("sub_topic") or "").strip()
                in (_primary_sub_topic_flag, "", "nan", "None")
            ]
            if _st_cluster:
                _flag_docs = _st_cluster

        # Flag: any retrieved doc has identification_documents content.
        # When true, always inject FORM_LINKS regardless of query phrasing —
        # so users always get form links whenever a document list will be shown in the answer.
        _docs_have_id_docs = any(
            str((d.get("metadata") or {}).get("identification_documents") or "").strip()
            not in ("", "nan", "None")
            for d in _flag_docs
        )
        # Flag: any retrieved doc has operation_steps content.
        # When true, always inject SERVICE_LINKS — answer will include steps → user needs the portal.
        _answer_will_have_steps = any(
            str((d.get("metadata") or {}).get("operation_steps") or "").strip()
            not in ("", "nan", "None")
            for d in _flag_docs
        )

        # Pass 2: build docs_json with per-(license_type, entity_type, operation_by_department) dedup for long fields.
        # Including operation_by_department ensures different operations (แก้ไขรายการ / ยกเลิก / ต่ออายุ)
        # each send their own identification_documents even when they share the same entity_type_normalized.
        # research_reference is now injected as labeled sections outside docs_json
        _long_fields_sent_by_lt: dict = {}  # (lt, entity_type, op_by_dept) → bool
        docs_json = []
        for d in _docs_to_process:
            md = d.get("metadata", {}) or {}
            _lt2 = (md.get("license_type") or "").strip()
            _et2 = (md.get("entity_type_normalized") or "").strip()
            _op_by_dept2 = (md.get("operation_by_department") or "").strip()
            _lt_et_key = (_lt2, _et2, _op_by_dept2)
            filtered_md = {}
            for k, v in md.items():
                if k not in _LLM_METADATA_WHITELIST:
                    continue
                if v in (None, "", "nan", "None"):
                    continue
                # Per-(license,entity) dedup: skip long fields already sent for this combination
                if k in _LONG_FIELDS_DEDUP and _long_fields_sent_by_lt.get(_lt_et_key):
                    continue
                # research_reference injected as labeled sections below — skip from per-doc metadata
                if k == "research_reference":
                    continue
                v_str = str(v)
                # Flatten all internal newlines in field values to a single space.
                # Source data (Google Sheet) may contain line breaks mid-word or mid-sentence
                # (e.g. "(ภ.อ.\n11)", "ไ\nปติดแถบ"). The LLM re-formats numbered items
                # from inline markers (1., 2., 3.) so list structure is preserved.
                v_str = re.sub(r"[\r\n]+", " ", v_str)
                v_str = re.sub(r"[ \t]+", " ", v_str).strip()
                cap = _FIELD_CAPS.get(k)
                if cap and len(v_str) > cap:
                    v_str = v_str[:cap]
                filtered_md[k] = v_str
            if any(k in filtered_md for k in _LONG_FIELDS_DEDUP):
                _long_fields_sent_by_lt[_lt_et_key] = True
            # Chapter overview: trim content to 300 chars per doc — LLM only needs enough to
            # write 2-4 bullet points per sub_topic. Full 700-char content across 15+ docs
            # causes prompt > 35K tokens → JSON malformation risk on long Thai responses.
            _raw_content = (d.get("content", "") or "")
            # Strip "อ้างอิง: Website...\nhttps://..." from non-regulatory doc content.
            # Marketing/bakery docs have this appended at ingest; links shown only on explicit ask.
            if (md.get("data_type") or "").strip() in ("marketing", "business_guide"):
                _raw_content = re.sub(r'\nอ้างอิง:.*', '', _raw_content, flags=re.DOTALL).strip()
            # Chapter overview: clip to 300 chars per doc to control token budget across many docs.
            # Exception: a single aggregated doc (sub_topic completeness path) must NOT be clipped
            # — its content is the numbered list of ALL items and truncation would hide most entries.
            _content_for_llm = (
                _raw_content[:300]
                if _is_chapter_overview and len(_docs_to_process) > 1
                else _raw_content
            )
            docs_json.append(
                {
                    "metadata": filtered_md,
                    "content": _content_for_llm,
                }
            )

        # Build labeled link sections — LLM copies these directly, no URL pattern matching needed
        # When docs come from multiple licenses, tag each link with [license_type] so LLM
        # only includes links relevant to the license it is currently answering about.
        _all_link_lts = set(_url_to_license.values()) - {""}
        _use_lt_tags = len(_all_link_lts) > 1

        def _fmt_prac_link(desc: str, url: str) -> str:
            _tag = (f"[{_url_to_license[url]}] " if _use_lt_tags and url and url in _url_to_license else "")
            if desc and url:
                return f"- {_tag}{desc}\n  {url}"
            return f"- {_tag}{url or desc}"

        # Links: inject ตาม intent ของ user เท่านั้น ไม่ inject ทุก section ตลอดเวลา
        # When user fills a slot (e.g. "1"), user_text is short — fall back to last_user_legal_query
        # so that link-intent from the original question is preserved through slot collection.
        _intent_text = (user_text or "") + " " + str((state.context or {}).get("last_user_legal_query") or "")
        _user_wants_links = bool(re.search(
            r"(ขอลิงค์|ขอลิงก์|ส่งลิงค์|ส่งลิงก์|ลิงค์คู่มือ|ลิงก์คู่มือ"
            r"|ขอดูลิงค์|ขอดูลิงก์|URL|ดาวน์โหลด"
            r"|ขอคู่มือ|ขอดูคู่มือ|ส่งคู่มือ|คู่มือ(การ|สำหรับ|ของ)"
            r"|ขอแบบฟอร์ม|ส่งแบบฟอร์ม"
            r"|ขออ้างอิง|แหล่งอ้างอิง|แหล่งที่มา|แหล่งข้อมูล|อ้างอิงไหน|อ้างอิงได้ที่)",
            _intent_text, re.IGNORECASE,
        ))
        # Service/registration links: เฉพาะตอน user ถามเรื่องการสมัคร/ลงทะเบียน
        # จดภาษี covers "การจดภาษีมูลค่าเพิ่ม (VAT)" — uses "จด" + "ภาษี" not "จดทะเบียน"
        _user_wants_registration = bool(re.search(
            r"(สมัคร|ลงทะเบียน|ยื่นขอ|จดทะเบียน|จดภาษี|ขอใบ|อยากจด|ต้องการจด"
            r"|ขั้นตอน(การ|ใน)|วิธี(จด|สมัคร|ยื่น)|ต้องทำยังไง"
            r"|ลิ้งค์.{0,6}(สมัคร|ลงทะเบียน|กรอก|ไฟล์|API|form)"
            r"|ลิงค์.{0,6}(สมัคร|ลงทะเบียน|กรอก|ไฟล์|API|form)"
            r"|link.{0,6}(register|apply|form|sign.?up))",
            _intent_text, re.IGNORECASE,
        ))
        # Form links: เฉพาะตอน user ถามเรื่องเอกสาร/แบบฟอร์ม
        _user_wants_forms = bool(re.search(
            r"(แบบฟอร์ม|(?:ขอแค่|ขอ|แค่)\s*เอกสาร"
            r"|เอกสาร(?:ที่|ใช้|ต้อง|ประกอบ|สำหรับ|ของ)"
            r"|เอกสาร.{0,8}(ใช้|ที่ต้อง|ต้องใช้)|ต้องใช้.{0,8}เอกสาร"
            r"|ลิ้งค์.{0,6}(เอกสาร|ฟอร์ม|คำขอ)"
            r"|link.{0,6}(document|form|template))",
            _intent_text, re.IGNORECASE,
        ))
        # Reference links: เฉพาะตอน user ถามหาแหล่งอ้างอิง/ที่มาของข้อมูลโดยตรง
        _user_wants_reference = bool(re.search(
            r"(อ้างอิง|แหล่งที่มา|แหล่งข้อมูล|แหล่งอ้างอิง|ที่มาของข้อมูล"
            r"|อ้างอิงได้ที่|อ้างอิงจาก|ดูอ้างอิง|ขออ้างอิง|ขอแหล่ง"
            r"|reference|แหล่งข้อมูลเพิ่มเติม|ข้อมูลจากไหน)",
            _intent_text, re.IGNORECASE,
        ))

        # Link injection flags — used by safety nets post-LLM to decide which link types to append.
        _is_info_q_ctx = bool((state.context or {}).get("_informational_query"))
        _suppress_links_broad = (_is_broad_q or _is_info_q_ctx) and not _user_wants_links
        _is_link_req_ctx = bool((state.context or {}).get("_link_request_query"))
        _wants_guides_specifically = bool(re.search(
            r"ขอคู่มือ|ขอดูคู่มือ|ส่งคู่มือ|คู่มือ(การ|สำหรับ|ของ|ไหน)"
            r"|ลิ[้]?งค์คู่มือ|ลิ[้]?งก์คู่มือ",
            _intent_text, re.IGNORECASE,
        ))
        _is_contact_q = (
            bool(re.search(
                r"ต้องการติดต่อ|ขอติดต่อ|วิธีติดต่อ|ช่องทาง.{0,6}ติดต่อ"
                r"|ติดต่อ.{0,10}(กรม|หน่วย|สำนัก|ธนาคาร|สำนักงาน|องค์กร)"
                r"|ที่อยู่.{0,10}(กรม|หน่วย|สำนัก|ธนาคาร|สำนักงาน)"
                r"|เบอร์โทร.{0,10}(กรม|หน่วย|สำนัก|ธนาคาร|สำนักงาน)"
                r"|หมายเลขโทรศัพท์|โทรสาร.{0,10}(กรม|หน่วย|สำนัก)",
                _intent_text, re.IGNORECASE,
            ))
            and not _user_wants_registration
            and not _user_wants_links
        )
        _form_auto_inject = _docs_have_id_docs and not _is_link_req_ctx
        _guide_auto_ok = not _is_link_req_ctx or _wants_guides_specifically
        # True when user asked specifically for document/form links (not service portals or guides).
        # Suppresses 🌐 and 📖 safety nets and strips them from LLM output.
        _suppress_non_form_links = _user_wants_forms and not _user_wants_links and not _wants_guides_specifically

        self._debug_log("pre_llm", query=user_text, docs_json=docs_json)

        # Token: ตัด context ให้เล็กลง — เก็บเฉพาะ keys ที่ LLM ต้องการจริงๆ
        _ctx_keys_needed = {"topic", "slots", "pending_slot", "last_user_legal_query",
                            "last_topic", "topic_slot_queue", "topic_operation_groups",
                            "collected_slots", "multi_license_topics"}
        slim_context = {k: v for k, v in (state.context or {}).items()
                        if k in _ctx_keys_needed and v not in (None, {}, [], "")}

        # Strip raw_op_map from topic_slot_queue before passing to LLM.
        # raw_op_map is internal Supervisor routing data (operation_topic → group mapping)
        # that can be 1-2K tokens. The LLM only needs display options, not the raw mapping.
        if "topic_slot_queue" in slim_context and isinstance(slim_context["topic_slot_queue"], list):
            slim_context["topic_slot_queue"] = [
                {k2: v2 for k2, v2 in slot.items() if k2 != "raw_op_map"}
                for slot in slim_context["topic_slot_queue"]
                if isinstance(slot, dict)
            ]

        # Inject active topic hint so LLM never re-asks what the user already chose
        _active_topic = (
            (state.context or {}).get("last_topic")
            or (state.context or {}).get("last_user_legal_query")
            or ""
        )
        _slots_now = (state.context or {}).get("slots") or {}
        _active_op = _slots_now.get("operation_group", "") or _slots_now.get("confirmed_operation", "")
        _confirmed_topic = _slots_now.get("confirmed_topic", "") or ""
        _topic_hint = ""
        if _confirmed_topic:
            # User already picked a specific topic+operation → forbid re-asking
            _topic_hint = (
                f"\n\n⚠️ MANDATORY RULES (STRICTLY FOLLOW):\n"
                f"- User already selected topic: \"{_confirmed_topic}\"\n"
            )
            if _active_op:
                _topic_hint += f"- User already selected operation: \"{_active_op}\"\n"
            _topic_hint += (
                "- action MUST be \"answer\" — provide the actual steps/documents for this topic+operation NOW.\n"
                "- DO NOT ask user to choose a topic or license type again.\n"
                "- DO NOT generate slot_options or numbered menus asking about license selection.\n"
                "- Answer directly using the documents provided below."
            )
        elif (state.context or {}).get("_direct_topic_match"):
            # Exact operation_topic match — 1 doc retrieved by Chroma metadata filter.
            # Content may list methods/steps as informational facts (not user choices).
            # Force action='answer' so LLM presents content directly without asking sub-questions.
            _topic_hint = (
                "\n\n⚠️ MANDATORY RULES (STRICTLY FOLLOW):\n"
                "- The document is an EXACT match for the user's query.\n"
                "- Any numbered items in the document are FACTS/METHODS to present, not user choices.\n"
                "- action MUST be \"answer\" — present the document content as factual information.\n"
                "- DO NOT ask the user to choose between listed items (e.g., do not ask 'which method do you want?').\n"
                "- DO NOT generate slot_options or clarifying sub-questions about the listed content."
            )
        elif _active_topic:
            # Topic known from context but not yet confirmed via operation_group —
            # LLM may still ask clarifying questions, but should bias toward this topic
            _topic_hint = (
                f"\n\n💡 CONTEXT HINT: User is currently discussing \"{_active_topic}\"."
            )
            if _active_op:
                _topic_hint += f" Operation in focus: \"{_active_op}\"."

        # Soft multi-license hint: when docs span 2+ license types that the user explicitly
        # requested, remind LLM to cover ALL of them in its answer. Unlike multi_license_topics
        # (which forces action='answer'), this is advisory only — LLM can still ask slots first.
        if _is_multi_license_docs:
            _hint_lts = list(dict.fromkeys(
                (d.get("metadata") or {}).get("license_type", "").strip()
                for d in _all_docs
                if (d.get("metadata") or {}).get("license_type", "")
            ))
            if len(_hint_lts) >= 2:
                _topic_hint += (
                    f"\n\n💡 MULTI-LICENSE: user ถามเรื่อง {', '.join(_hint_lts)} พร้อมกัน"
                    " — เมื่อ action='answer' ต้องครอบคลุมทุก license ที่ระบุ ไม่ใช่แค่ตัวเดียว"
                    " (ยังสามารถ action='ask' เพื่อถาม slot ก่อนได้ตามปกติ)"
                )

        # Multi-choice hint: when user selected multiple sub-types (e.g. "1 2" for a choice slot)
        _choice_val = _slots_now.get("choice")
        if isinstance(_choice_val, list) and len(_choice_val) >= 2:
            _topic_hint += (
                f"\n\n💡 MULTI-CHOICE: user เลือกหลายกรณี: {', '.join(str(x) for x in _choice_val)}"
                " — action='answer' ต้องครอบคลุมทุกกรณีที่ user เลือก ไม่ใช่แค่กรณีเดียว"
                " (แยกหัวข้อย่อยชัดเจนสำหรับแต่ละกรณีที่เลือก)"
            )

        # Build special instruction when user asked about multiple license types at once
        _multi_license_topics = (state.context or {}).get("multi_license_topics") or []
        _multi_license_instruction = ""

        # 4+ topics: show summary + menu instead of answering all at once
        if len(_multi_license_topics) > _MULTI_TOPIC_MENU_THRESHOLD:
            msg = self._build_multi_topic_summary_menu(state, _multi_license_topics)
            msg = self._apply_practical_lint(msg, kind="menu")
            self._append_assistant(state, msg)
            state.round = int(getattr(state, "round", 0) or 0) + 1
            return state, msg

        if _multi_license_topics:
            _topics_str = ", ".join(_multi_license_topics)
            _multi_license_instruction = f"""

⚠️ MULTI-TOPIC INSTRUCTION (MANDATORY):
User asked about {len(_multi_license_topics)} related topics: {_topics_str}
Rules:
- action MUST be "answer" — never "ask" or "retrieve"
- Start execution.answer with a SHORT opening sentence that connects all topics together, e.g.
  "ร้านของคุณต้องดำเนินการหลายเรื่องพร้อมกัน ได้แก่ [topics] ขอสรุปทีละเรื่องเลยครับ"
  or "เนื่องจากคุณถามเรื่อง [context] จะต้องจัดการเรื่องเหล่านี้ด้วยครับ:"
  DO NOT just list section headers without context — open with 1-2 sentences that explain why these topics go together
- Then list each topic as a numbered section with header: e.g. "📌 1. ชื่อหัวข้อ" then full content
- Section headers must NOT be questions (no ไหม/อย่างไร/ตอนไหน in headers)
- Under each header, write the actual steps/requirements/documents — NOT just the header alone
- Put the most important / legally required items FIRST (e.g. ใบอนุญาตหลักก่อน ใบรับรองเสริมทีหลัง)
- If a topic has no DOCUMENTS, write EXACTLY: "ยังไม่พบข้อมูลในเอกสาร" — do NOT add any URL, website, phone number, office name, or contact info that is not in the DOCUMENTS above
- Do NOT stop after writing just one section header
- Close with a 1-sentence summary: which items are MANDATORY vs optional
"""

        # Broad-question instruction: injected directly into this call's prompt (not system prompt)
        # so it only affects broad open-ended questions, not other query types.
        # Tells LLM to treat legal/license section as mandatory named section, not a footnote.
        # Fee/cost/duration informational query instruction:
        # Injected when user asks about fees (ค่าธรรมเนียม/เท่าไหร่/กี่วัน) but has NOT asked
        # to actually register/apply (no action intent). Forces action="answer" with all fee
        # tiers combined — avoids unnecessary entity_type/operation_group asking.
        _is_fee_info_q = bool(re.search(
            r"ค่าธรรมเนียม|เท่าไหร่|กี่บาท|กี่วัน|กี่เดือน|กี่ปี|ค่าใช้จ่าย|ระยะเวลา",
            _intent_text, re.IGNORECASE,
        )) and not bool(re.search(
            r"อยากจด|ต้องการจด|จะจด|จะสมัคร|อยากสมัคร|ต้องการสมัคร|วิธีจด|ขั้นตอนการจด",
            _intent_text, re.IGNORECASE,
        ))
        _fee_info_instruction = ""
        if _is_fee_info_q:
            _fee_info_instruction = """

⚠️ FEE / DURATION INFORMATIONAL QUERY — MANDATORY RULES:
- action MUST be "answer". DO NOT set action="ask".
- DO NOT ask for entity_type, registration_type, or operation_group.
- Scan ALL DOCUMENTS for ค่าธรรมเนียม (fees) or ระยะเวลา (duration) fields.
- If values differ by entity or sub-type: list ALL tiers together in one concise answer
  (e.g. "ตั้งใหม่ 50 บาท ทุกประเภท กรณีบริษัทจำกัดมีค่าบริคณห์สนธิเพิ่ม 500 บาท").
- Keep the answer SHORT — just the fee/duration and one brief contextual note if relevant.
- Do NOT list full registration steps or document requirements unless explicitly asked.
"""

        # General informational/definitional query instruction:
        # Covers คืออะไร, หมายถึงอะไร, เงื่อนไข, ข้อยกเว้น, ใครบ้าง, เปรียบเทียบ, etc.
        # Forces action="answer" without asking entity_type/registration_type/operation_group,
        # because the answer doesn't change based on entity type for definitional questions.
        # Does NOT apply when: fee/duration already handled, or user has explicit action intent.
        _is_general_info_q = (
            not _is_fee_info_q
            and bool(re.search(
                r"คืออะไร|หมายถึงอะไร|ประเภทใด|ประเภทไหน|อะไรบ้าง|"
                r"ใครบ้าง|เงื่อนไข|ข้อยกเว้น|ยกเว้น|ไม่ต้อง|"
                r"แตกต่าง|เปรียบเทียบ|อธิบาย|กรณีใด|เมื่อไหร่|เมื่อไร|"
                r"จะเกิดอะไร|เกิดอะไร|จะเป็นอะไร|ผลกระทบ|บทลงโทษ|โทษคือ|ค่าปรับ|"
                r"ต้องโดน|จะถูก|ชำรุด|สูญหาย|ทำหาย|ทำหล่น|เสียหาย",
                _intent_text, re.IGNORECASE,
            ))
            and not bool(re.search(
                r"อยากจด|ต้องการจด|จะจด|จะสมัคร|อยากสมัคร|ต้องการสมัคร|"
                r"วิธีจด|ขั้นตอนการจด|วิธีขอ|ขั้นตอนการขอ|วิธีสมัคร|ขั้นตอนการสมัคร",
                _intent_text, re.IGNORECASE,
            ))
            and not bool(re.search(
                r"ต้องใช้|ต้องเตรียม|เอกสาร(ที่ต้อง|ประกอบ)",
                _intent_text, re.IGNORECASE,
            ))
        )
        _info_q_instruction = ""
        if _is_general_info_q:
            _info_q_instruction = """

⚠️ INFORMATIONAL / DEFINITIONAL QUERY — MANDATORY RULES:
- action MUST be "answer". DO NOT set action="ask".
- DO NOT ask for entity_type, registration_type, or operation_group.
- The answer to a definitional or conceptual question does NOT depend on the user's entity type.
- Answer directly from ALL DOCUMENTS. Cover all sub-types/cases within the same answer if they differ.
- Keep the answer concise and factual. Do NOT list registration steps unless explicitly asked.
"""

        _broad_instruction = ""
        if _is_broad_q:
            # Collect license_type names actually present in docs so the instruction is concrete
            _license_names_in_docs: list = []
            for _bd in _docs_to_process:
                _blt = ((_bd.get("metadata") or {}).get("license_type") or "").strip()
                if _blt and _blt not in _license_names_in_docs:
                    _license_names_in_docs.append(_blt)
            _license_list_str = (
                "\n".join(f"  - {lt}" for lt in _license_names_in_docs)
                if _license_names_in_docs
                else "  (ดูจาก DOCUMENTS)"
            )
            _broad_instruction = f"""

⚠️ BROAD QUESTION — LEGAL SECTION IS MANDATORY (strictly follow):
- This is a broad open-ended question about opening a business.
- Your answer MUST include a dedicated section with header "📋 ใบอนุญาตและกฎหมายที่เกี่ยวข้อง".
- This section is NOT a footnote. It must appear as a full named section alongside other sections.
- Inside this section: list EVERY license/permit found in DOCUMENTS as a numbered list.
  Each item: license name + one line explaining what it is and when it is required.
- The following license types appear in DOCUMENTS — do NOT skip any:
{_license_list_str}
- Do NOT merge multiple licenses into one bullet. Do NOT use "เช่น..." to abbreviate.
- Do NOT move legal content to the last line of the response as a footnote.
- Tone: write naturally, like a knowledgeable friend explaining what the person MUST do legally.
"""

        # Build entity filter hint — if entity_type or registration_type already known from
        # collected_slots, tell LLM to answer ONLY for that case (never split into 2 cases).
        _entity_filter_hint = ""
        _cs_for_hint = state.get_collected_slots() if hasattr(state, "get_collected_slots") else {}
        _ctx_slots_hint = (state.context or {}).get("slots") or {}
        _et_hint = (
            _cs_for_hint.get("entity_type_normalized")
            or _cs_for_hint.get("entity_type")
            or _ctx_slots_hint.get("entity_type_normalized")
            or _ctx_slots_hint.get("entity_type")
            or ""
        ).strip()
        # Single-turn OBD entity override (entity-specific concept doc): takes highest precedence.
        _obd_et_ctx = str((state.context or {}).get("_obd_entity_override") or "").strip()
        if _obd_et_ctx:
            _et_hint = _obd_et_ctx
        # If user_text explicitly switches entity_type ("แล้วถ้าเป็นนิติบุคคลอ่ะ"), override _et_hint
        # for the prompt so LLM answers for the requested entity, not the stored one.
        # _query_et_override was computed in the early entity-switch detection block above.
        if _query_et_override:
            _et_hint = _query_et_override
        _rt_hint = (
            _cs_for_hint.get("registration_type")
            or _ctx_slots_hint.get("registration_type")
            or ""
        ).strip()
        # When entity switches to บุคคลธรรมดา, registration_type sub-types (บริษัทจำกัด etc.)
        # are no longer applicable — clear stale _rt_hint to avoid contradictory prompt.
        if _query_et_override == "บุคคลธรรมดา" and _rt_hint:
            _rt_hint = ""
        _area_hint = (
            _cs_for_hint.get("shop_area_type")
            or _ctx_slots_hint.get("shop_area_type")
            or ""
        ).strip()
        if _et_hint or _rt_hint:
            _case_desc = _rt_hint or _et_hint
            _area_clause = f" ร้านมีพื้นที่ {_area_hint}" if _area_hint else ""
            _entity_filter_hint = (
                f"\n\n⚠️ USER ENTITY TYPE IS KNOWN — MANDATORY RULE:\n"
                f"- This user is: {_case_desc}{_area_clause}\n"
                f"- Answer ONLY for {_case_desc}. NEVER split answer into multiple cases.\n"
                f"- Do NOT write sections like 'สำหรับบุคคลธรรมดา' / 'สำหรับนิติบุคคล'.\n"
                f"- Write as if the user IS {_case_desc} — single unified answer."
            )
            if _area_hint:
                _entity_filter_hint += (
                    f"\n\n⚠️ SHOP AREA SIZE IS KNOWN — MANDATORY RULE:\n"
                    f"- The user's CURRENT area is: {_area_hint}\n"
                    f"- Show fee/requirement for {_area_hint} ONLY. Do NOT list other area tiers.\n"
                    f"- NEVER reclassify area values mentioned in earlier conversation turns — each turn's area is independent."
                )
        elif _area_hint:
            _entity_filter_hint = (
                f"\n\n⚠️ SHOP AREA SIZE IS KNOWN — MANDATORY RULE:\n"
                f"- The user's CURRENT area is: {_area_hint}\n"
                f"- Show fee/requirement for {_area_hint} ONLY. Do NOT list other area tiers.\n"
                f"- NEVER reclassify area values mentioned in earlier conversation turns — each turn's area is independent."
            )

        # Cut-off time query instruction: force LLM to read terms_and_conditions metadata directly.
        # Needed when user asks about ตัดรอบ / cut-off time — the data lives in terms_and_conditions
        # (not in page_content), so without an explicit directive the LLM may hallucinate.
        _tc_instruction = ""
        if self._TC_QUERY_RE.search(_tc_query_text):
            # Check if any doc in the prompt actually has cut-off time data — if so, be prescriptive
            _tc_has_data = any(
                self._TC_TIME_RE.search(str((_d.get("metadata") or {}).get("terms_and_conditions") or ""))
                for _d in _docs_to_process
            )
            if _tc_has_data:
                _tc_instruction = (
                    "\n\n⚠️ CUT-OFF TIME QUERY — MANDATORY RULES:\n"
                    "- User is asking about ตัดรอบ / cut-off time / payment settlement schedule.\n"
                    "- One of the DOCUMENTS has time values (e.g. '22.00 น.', '23:00 น.') in its 'terms_and_conditions' metadata field — that is the actual schedule data.\n"
                    "- action MUST be \"answer\".\n"
                    "- Output the terms_and_conditions data EXACTLY as-is. Preserve time tables, all rows, all columns.\n"
                    "- Do NOT paraphrase, summarize, or say 'ธนาคารจะแจ้งโดยตรง' / 'ขึ้นอยู่กับธนาคาร' — the data IS available in DOCUMENTS."
                )

        # Chapter overview: inject explicit sub_topic list as section structure guide.
        # Works together with the doc cap bypass above — LLM now sees all docs AND
        # knows exactly which sub_topics to present as separate named sections.
        _chapter_overview_instruction = ""
        if _is_chapter_overview:
            _chapter_subtopics: list = []
            for _d in _docs_to_process:
                _st = ((_d.get("metadata") or {}).get("sub_topic") or "").strip()
                if _st and _st not in _chapter_subtopics:
                    _chapter_subtopics.append(_st)
            if len(_chapter_subtopics) >= 2:
                _st_list_str = "\n".join(f"  {i+1}. {st}" for i, st in enumerate(_chapter_subtopics))
                _chapter_mt = next(
                    ((_d.get("metadata") or {}).get("main_topic") or "").strip()
                    for _d in _docs_to_process
                    if (_d.get("metadata") or {}).get("main_topic")
                )
                _chapter_overview_instruction = (
                    f"\n\n⚠️ CHAPTER OVERVIEW — MANDATORY (RULE 0.5):\n"
                    f"- All DOCUMENTS belong to chapter: \"{_chapter_mt}\"\n"
                    f"- Create ONE named section per sub_topic below. Do NOT merge sub_topics. Do NOT skip any.\n"
                    f"- Required sections ({len(_chapter_subtopics)} total):\n"
                    f"{_st_list_str}\n"
                    f"- Format: emoji header + 2-4 bullet points per section. End with short follow-up offer."
                )

        # Layer 2 of hallucination prevention: build a phone-number whitelist from retrieved docs
        # and inject it into the human message so the LLM has an explicit, per-query boundary.
        # Layer 1 = system prompt constraint. Layer 3 = post-processing strip (below, near URL validation).
        _DOC_PHONE_CTX_RE = re.compile(
            r'(?:📞|โทร(?:ศัพท์|สาร)?|สายด่วน|Tel\.?|Fax\.?|แฟกซ์)'
            r'\s*[:\s]*(\d[\d\s\-\.]{3,13})',
            re.IGNORECASE,
        )
        _doc_phones: set = set()
        for _dp in _all_docs:
            _dp_text = json.dumps(_dp, ensure_ascii=False)
            for _pm in _DOC_PHONE_CTX_RE.finditer(_dp_text):
                _norm_p = re.sub(r'[\s\-\.]', '', _pm.group(1))
                if 4 <= len(_norm_p) <= 12 and _norm_p.isdigit():
                    _doc_phones.add(_norm_p)
        if _doc_phones:
            _phones_display = ", ".join(sorted(_doc_phones))
            _phone_guard_instruction = (
                f"\n\n⛔ PHONE NUMBER GUARD: Only these phone numbers appear verbatim in DOCUMENTS: "
                f"{_phones_display}. Do NOT write any other phone number or hotline number."
            )
        else:
            _phone_guard_instruction = (
                "\n\n⛔ PHONE NUMBER GUARD: No phone numbers appear in the retrieved DOCUMENTS. "
                "Do NOT write any phone number, hotline number (e.g. 1570), or contact number. "
                "Write 'ติดต่อ [ชื่อหน่วยงาน] โดยตรง' without a number."
            )

        prompt = f"""USER INPUT:
{user_input}

LAST ASSISTANT MESSAGE:
{last_bot[:300] if last_bot else ""}

RECENT MESSAGES:
{json.dumps(recent_msgs, ensure_ascii=False)}

CONTEXT_MEMORY:
{json.dumps(slim_context, ensure_ascii=False)}

DOCUMENTS ({len(docs_json)} found):
{json.dumps(docs_json, ensure_ascii=False)}
ROUND: {int(getattr(state, "round", 0) or 0)}/{int(getattr(conf, "MAX_ROUNDS", 7) or 7)}{_entity_filter_hint}{_topic_hint}{_multi_license_instruction}{_fee_info_instruction}{_info_q_instruction}{_broad_instruction}{_tc_instruction}{_chapter_overview_instruction}{_phone_guard_instruction}

Your JSON response:
"""

        # SHORT-CIRCUIT: when topic_slot_queue has an askable slot and docs are loaded,
        # the next action is always 'ask' — skip the LLM entirely.
        # Only short-circuit when at least one slot is genuinely askable (has options AND is
        # not already collected). If all queued slots were already collected (e.g. entity_type
        # answered in a previous topic), fall through to LLM which will return action=answer.
        _pending_sq = (state.context or {}).get("topic_slot_queue") or []
        # Auto-fill operation_topic slot when user's query already contains all Latin
        # keywords of one option (e.g. "(SAN)" → auto-select 'มาตรฐาน SAN', skip asking).
        # Check most-specific option first (most Latin words) to avoid 'SAN' matching
        # 'SAN PLUS' before the full 'SAN PLUS' option is checked.
        _ot_sq = next((s for s in _pending_sq if s.get("key") == "operation_topic"), None)
        if _ot_sq:
            _ot_opts = _ot_sq.get("options") or []
            _q_lower = user_text.lower()
            def _af_fuzzy_in(tok: str, text: str) -> bool:
                if tok in text:
                    return True
                tlen = len(tok)
                return any(
                    SequenceMatcher(None, tok, text[i:i + tlen]).ratio() >= 0.75
                    for i in range(max(0, len(text) - tlen + 1))
                )
            _auto_ot: Optional[str] = None
            for _ot_opt in sorted(_ot_opts, key=lambda o: len(re.findall(r"[A-Za-z]{2,}", o)), reverse=True):
                _ot_latin = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", _ot_opt)]
                if _ot_latin and all(_af_fuzzy_in(w, _q_lower) for w in _ot_latin):
                    _auto_ot = _ot_opt
                    break
            if _auto_ot:
                state.save_collected_slot("operation_topic", _auto_ot)
                _pending_sq = [s for s in _pending_sq if s.get("key") != "operation_topic"]
                if isinstance(state.context, dict):
                    state.context["topic_slot_queue"] = _pending_sq
                _LOG.info("[Practical] auto-filled operation_topic=%r from user query", _auto_ot)
                # Filter _all_docs to docs matching the auto-filled op_topic + generic docs
                # (no operation_topic). Without this, LLM sees both SAN and SAN PLUS docs
                # and asks a clarifying question even though the user already specified.
                _prev_ot_count = len(_all_docs)
                _ot_filtered = [
                    d for d in _all_docs
                    if str((d.get("metadata") or {}).get("operation_topic") or "").strip()
                    in ("", "nan", "None", "-", "?", _auto_ot)
                ]
                if _ot_filtered:
                    _all_docs = _ot_filtered
                    state.current_docs = _ot_filtered
                    _LOG.info(
                        "[Practical] auto-fill doc-filter: %d → %d docs for op_topic=%r",
                        _prev_ot_count, len(_ot_filtered), _auto_ot,
                    )
        _SKIP_KEYS_SC = {"entity_type", "registration_type"}
        _has_askable_slot = any(
            s for s in _pending_sq
            if s.get("options")
            and not (
                s.get("key") in _SKIP_KEYS_SC
                and state.get_collected_slot(s.get("key") or "")
            )
        )
        if _has_askable_slot and state.current_docs:
            _LOG.info("[Practical] slot_queue has askable slot — skipping LLM, action=ask")
            decision = {"action": "ask", "execution": {"question": "", "slot_options": [], "answer": "", "query": "", "context_update": {}}}
        else:
            decision = self._call_llm_json(prompt, state=state)
        action = (decision.get("action") or "ask").strip()
        _exec_raw = decision.get("execution", {})
        # Gemini may return execution as a JSON string instead of a dict — parse it
        if isinstance(_exec_raw, str):
            try:
                _exec_raw = json.loads(_exec_raw)
            except Exception:
                _exec_raw = {}
        exec_ = _exec_raw if isinstance(_exec_raw, dict) else {}

        if action == "retrieve":
            # Guard: block LLM re-retrieval when docs are already loaded.
            # Two conditions — either is sufficient to block:
            #   1) topic_slot_queue non-empty AND docs exist (original guard — slot ask phase)
            #   2) _internal=True AND docs exist — supervisor already set the right docs
            #      (covers the case where queue was JUST cleared after last slot filled,
            #       but LLM ignores rule 2 and returns action='retrieve' anyway)
            _pending_queue = (state.context or {}).get("topic_slot_queue") or []
            if state.current_docs and (_pending_queue or _internal):
                # Track how many consecutive times retrieve has been blocked.
                # If LLM keeps returning 'retrieve' despite blocked (e.g. sees wrong-entity docs
                # and refuses to answer), force action='answer' after 2 blocks to break the loop.
                _rblk = int((state.context or {}).get("_retrieve_blocked_count", 0)) + 1
                state.context["_retrieve_blocked_count"] = _rblk
                _LOG.info(
                    "[Practical] action='retrieve' blocked — docs already loaded (%d), "
                    "internal=%s queue=%s (block#%d)",
                    len(state.current_docs),
                    _internal,
                    [s.get("key") for s in _pending_queue] if _pending_queue else [],
                    _rblk,
                )
                if _rblk >= 2:
                    # LLM is stuck in retrieve loop — break by forcing a non-internal call.
                    # Setting _internal=False lets handle() call the LLM without the "retrieve blocked"
                    # guard, so LLM is forced to answer with current docs on next turn.
                    # Use last_user_legal_query (the original topic) — NOT user_input, which may be
                    # a slot answer like "นิติบุคคล" that would trigger wrong retrieval.
                    _safe_query = (
                        (state.context or {}).get("last_user_legal_query")
                        or (state.context or {}).get("last_retrieval_query")
                        or user_input
                        or "__auto_post_retrieve__"
                    )
                    _force_cnt = int((state.context or {}).get("_force_answer_count", 0)) + 1
                    state.context["_force_answer_count"] = _force_cnt
                    if _force_cnt >= 2:
                        # Force-answer already tried once but LLM still loops — hard abort.
                        # Prevents infinite recursion when retrieved docs are wrong entity/dept.
                        _LOG.error(
                            "[Practical] infinite loop — force-answer attempted %d times, returning fallback (query=%r)",
                            _force_cnt,
                            (_safe_query or "")[:60],
                        )
                        state.context["_force_answer_count"] = 0
                        state.context["_retrieve_blocked_count"] = 0
                        _fb = "ขอโทษครับ ไม่พบข้อมูลที่ตรงกับคำถามนี้ในฐานข้อมูลของเรา กรุณาลองถามใหม่หรือระบุรายละเอียดเพิ่มเติมครับ"
                        state.add_assistant_message(_fb)
                        return state, _fb
                    _LOG.warning(
                        "[Practical] retrieve blocked %d times — breaking loop, forcing non-internal answer pass (query=%r)",
                        _rblk,
                        _safe_query[:60],
                    )
                    state.context["_retrieve_blocked_count"] = 0
                    return self.handle(state, _safe_query, _internal=False)
                return self.handle(state, "__auto_post_retrieve__", _internal=True)

            if action == "retrieve":
                q = exec_.get("query") or user_text or user_input
                # If entity_type is already known, retrieve with entity filter directly.
                # This avoids the embedding imbalance where "แก้ไข ทะเบียนพาณิชย์ บุคคลธรรมดา"
                # still retrieves นิติบุคคล docs because they outnumber บุคคลธรรมดา in the corpus.
                _known_ent = (state.get_collected_slots() or {}).get("entity_type", "").strip()
                if _known_ent:
                    try:
                        from service.data_loader import DataLoader as _DLR
                        _known_ent = _DLR._normalize_entity_type(_known_ent)
                    except Exception:
                        pass
                if _known_ent:
                    _retrieved_all = self._retrieve_docs(
                        q, metadata_filter={"entity_type_normalized": _known_ent}
                    )
                    # Fallback: if filtered returns nothing, try unfiltered + post-filter
                    if not _retrieved_all:
                        _retrieved_all = self._retrieve_docs(q)
                        _filtered = [
                            d for d in _retrieved_all
                            if not ((d.get("metadata") or {}).get("entity_type_normalized") or "").strip()
                            or ((d.get("metadata") or {}).get("entity_type_normalized") or "").strip() == _known_ent
                        ]
                        state.current_docs = _filtered if _filtered else _retrieved_all
                    else:
                        state.current_docs = _retrieved_all
                else:
                    state.current_docs = self._retrieve_docs(q)
                state.last_retrieval_query = q
                tmp = [
                    {"content": d.get("content", "")[:120], "metadata": d.get("metadata", {})}
                    for d in state.current_docs[:1]
                ]
                self._debug_log("post_retrieve", query=q, docs_json=tmp)
                return self.handle(state, "__auto_post_retrieve__", _internal=True)

        if action == "ask":
            question = (exec_.get("question") or "อยากให้ช่วยเรื่องอะไรเกี่ยวกับร้านอาหารครับ?").strip()

            if isinstance(exec_.get("context_update", {}), dict):
                _ctx_update = dict(exec_.get("context_update", {}))
                # Strip pending_slot and topic_slot_queue: these are managed by Supervisor/queue logic.
                # LLM must NOT overwrite them via context_update — it would corrupt slot ordering.
                _ctx_update.pop("pending_slot", None)
                _ctx_update.pop("topic_slot_queue", None)
                state.context.update(_ctx_update)
                # Sanitize: if anything wrote a non-dict pending_slot, remove it
                if not isinstance(state.context.get("pending_slot"), (dict, type(None))):
                    state.context.pop("pending_slot", None)

            pending = state.context.get("pending_slot")

            # DYNAMIC SLOT QUEUE: pop next slot from topic_slot_queue (set by Supervisor)
            # Each entry is {"key": "entity_type"|"shop_area_type"|…, "options": [...], "question": "..."}
            # This replaces all hardcoded topic_registration_types / topic_area_types logic.
            _slot_queue = (state.context or {}).get("topic_slot_queue")
            # Sanitize: drop any non-dict entries that Gemini might have smuggled in as strings
            if isinstance(_slot_queue, list):
                _slot_queue = [s for s in _slot_queue if isinstance(s, dict)]
                if _slot_queue:
                    state.context["topic_slot_queue"] = _slot_queue
                else:
                    state.context.pop("topic_slot_queue", None)
                    _slot_queue = []
            if not isinstance(pending, dict) and isinstance(_slot_queue, list) and _slot_queue:
                # Pop first slot from queue — skip identity slots already collected
                # Only entity_type / registration_type are skippable (topic-agnostic identity).
                # area_size / location_scope are topic-specific — must always be asked fresh.
                _QUEUE_SKIP_SLOTS = {"entity_type", "registration_type"}
                while _slot_queue:
                    next_slot = _slot_queue[0]
                    remaining_queue = _slot_queue[1:]
                    slot_key = next_slot.get("key", "")
                    slot_opts = next_slot.get("options", [])
                    slot_q = next_slot.get("question", "")
                    # Auto-skip identity slots already answered in cross-topic memory
                    _known_val = (
                        state.get_collected_slot(slot_key)
                        if slot_key in _QUEUE_SKIP_SLOTS else None
                    )
                    if _known_val:
                        _LOG.info(
                            "[Practical] slot_queue → skip %r (already collected=%r)",
                            slot_key, _known_val,
                        )
                        _slots = state.context.setdefault("slots", {})
                        _slots[slot_key] = _known_val
                        # Sync to collected_slots so cross-topic memory is consistent
                        state.save_collected_slot(slot_key, _known_val)
                        _slot_queue = remaining_queue
                        if _slot_queue:
                            state.context["topic_slot_queue"] = _slot_queue
                        else:
                            state.context.pop("topic_slot_queue", None)
                        continue
                    break
                if slot_key and slot_opts and not _known_val:
                    question = slot_q
                    _pslot_entry: Dict[str, Any] = {
                        "key": slot_key,
                        "options": list(slot_opts),
                        "allow_multi": False,
                    }
                    if next_slot.get("context_only"):
                        _pslot_entry["context_only"] = True
                    state.context["pending_slot"] = _pslot_entry
                    if remaining_queue:
                        state.context["topic_slot_queue"] = remaining_queue
                    else:
                        state.context.pop("topic_slot_queue", None)
                    _LOG.info("[Practical] slot_queue → popped key=%r opts=%s remaining=%d",
                              slot_key, slot_opts, len(remaining_queue))

            # BUG-F fix: re-read pending after queue pop so that if the queue just set
            # pending_slot (e.g. entity_type), the LLM opts block below is correctly skipped.
            # Without this, `pending` is the stale pre-pop value (None) and the LLM's own
            # slot options (e.g. location_type) would overwrite the queue-assigned entity_type.
            pending = state.context.get("pending_slot")

            if not isinstance(pending, dict):
                # Prefer LLM-provided slot_options over regex extraction
                llm_opts = exec_.get("slot_options")
                if isinstance(llm_opts, list):
                    llm_opts = [str(o).strip() for o in llm_opts if str(o).strip()]
                else:
                    llm_opts = []
                parsed_opts = llm_opts or self._extract_numbered_options(question)
                # Fallback: if LLM forgot to include slot_options for known slot types,
                # inject the standard options so the numbered menu is always shown.
                if not parsed_opts:
                    _inferred_key_check = self._infer_slot_key_from_question(question)
                    if _inferred_key_check == "area_size":
                        # Derive options from current_docs area_size metadata — NOT hardcoded.
                        # Only use if ≥2 distinct values exist; otherwise this question has no
                        # differentiating value and should not create a numbered menu.
                        _area_opts_from_docs: set = set()
                        for _d in (state.current_docs or []):
                            _as_val = ((_d.get("metadata") or {}).get("area_size") or "").strip()
                            if _as_val and _as_val not in ("nan", "None"):
                                _area_opts_from_docs.add(_as_val)
                        if len(_area_opts_from_docs) >= 2:
                            parsed_opts = sorted(_area_opts_from_docs)
                        # else: leave parsed_opts empty → question shown as free text or skipped
                    elif _inferred_key_check == "location_scope":
                        # Derive options from current_docs operation_topic — NOT hardcoded.
                        # This ensures the display label matches the real data
                        # (e.g. 'กรุงเทพฯ และปริมณฑล' vs plain 'กรุงเทพฯ').
                        _loc_opts_from_docs: dict = {}  # filter_val → display_label
                        for _d in (state.current_docs or []):
                            _dmeta = _d.get("metadata") or {}
                            _dloc = (_dmeta.get("location") or "").strip()
                            _dtopic = (_dmeta.get("operation_topic") or "").strip()
                            if _dloc and _dloc not in ("nan", "None"):
                                if _dtopic and _dloc in _dtopic and _dtopic != _dloc:
                                    _loc_opts_from_docs[_dloc] = _dtopic
                                else:
                                    _loc_opts_from_docs.setdefault(_dloc, _dloc)
                        if len(_loc_opts_from_docs) >= 2:
                            parsed_opts = [_loc_opts_from_docs[k] for k in sorted(_loc_opts_from_docs)]
                        # else: only 1 (or 0) location in docs → no differentiation → skip menu
                if parsed_opts:
                    slot_key = self._infer_slot_key_from_question(question, options=parsed_opts)
                    allow_multi = True if slot_key in {self._PHASE3_SLOT_KEY, "choice"} else False

                    # AUTO-FILL: if this slot was already answered in an earlier topic,
                    # skip re-asking and silently fill it from cross-topic memory.
                    # Only applies to "identity" slots (entity_type, registration_type)
                    # NOT area_size / location_scope which are topic-specific. 
                    _AUTOFILL_SLOTS = {"entity_type", "registration_type"}
                    _already_known = (
                        state.get_collected_slot(slot_key)
                        if slot_key in _AUTOFILL_SLOTS else None
                    )
                    if _already_known and not bool(state.context.get("_autofill_guard")):
                        _LOG.info(
                            "[Practical] auto-fill slot %r = %r from collected_slots (skip re-ask)",
                            slot_key, _already_known,
                        )
                        _slots = state.context.setdefault("slots", {})
                        _slots[slot_key] = _already_known
                        # Sync to collected_slots (in case it was only in context["slots"])
                        state.save_collected_slot(slot_key, _already_known)
                        # Re-invoke handle() with auto-fill guard to produce a real answer
                        state.context["_autofill_guard"] = True
                        _result = self.handle(state, _already_known, _internal=True)
                        state.context.pop("_autofill_guard", None)
                        return _result
                    # END AUTO-FILL 

                    state.context["pending_slot"] = {"key": slot_key, "options": parsed_opts, "allow_multi": allow_multi}

            pending2 = state.context.get("pending_slot")
            if isinstance(pending2, dict):
                options = pending2.get("options")
                if isinstance(options, list) and options:
                    # Sanitize question: strip inline option text the LLM may have embedded
                    # e.g. "(บริษัทจำกัด (ห้างหุ้นส่วน" or "กรุงเทพฯ หรืออยู่ต่างจังหวัด"
                    q_clean = question
                    for opt in options:
                        # Remove (opt) or (opt<no-closing-paren> patterns
                        q_clean = re.sub(r'\s*\(' + re.escape(str(opt)) + r'\)?', ' ', q_clean)
                        # Remove trailing " หรือ opt" or " หรืออยู่ opt"
                        q_clean = re.sub(
                            r'\s+หรือ(?:อยู่|ไปที่|ว่า)?\s*' + re.escape(str(opt)) + r'\s*(?:ครับ|คะ|คะ)?$',
                            'ครับ', q_clean
                        )
                    q_clean = re.sub(r'\s+', ' ', q_clean).strip()
                    # Ensure ends with ครับ
                    if q_clean and not any(q_clean.endswith(e) for e in ('ครับ', 'คะ', 'คะ', '?', 'ไหม')):
                        q_clean = q_clean.rstrip('?').rstrip() + 'ครับ'
                    question = q_clean

                    # Always append numbered menu unless already numbered (Issues 2, 5)
                    if "1)" not in question and "1." not in question:
                        _disp_opts_p = pending2.get("display_options")
                        _render_opts = _disp_opts_p if (isinstance(_disp_opts_p, list) and len(_disp_opts_p) == len(options)) else options
                        menu = self._format_numbered_options(_render_opts)
                        question = question.rstrip() + "\n" + menu

            question = self._apply_practical_lint(question, kind="ask")

            self._append_assistant(state, question)
            state.round = int(getattr(state, "round", 0) or 0) + 1
            return state, question

        if action == "answer":
            ans = (exec_.get("answer") or "").strip()
            if not ans:
                ans = "ตอนนี้ยังไม่พบข้อมูลที่ยืนยันได้ในเอกสารครับ"

            if isinstance(exec_.get("context_update", {}), dict):
                _cu = dict(exec_.get("context_update", {}))
                _cu.pop("pending_slot", None)  # never let LLM overwrite pending_slot on answer
                _cu.pop("topic_slot_queue", None)  # never let LLM overwrite the slot queue
                state.context.update(_cu)
                # Sanitize: ensure pending_slot is always dict or absent
                if not isinstance(state.context.get("pending_slot"), (dict, type(None))):
                    state.context.pop("pending_slot", None)

            # Filter link lists by used_doc_indices.
            # ref: always filter to cited docs only (reference links are topic-specific).
            # service/form/guide: normally show ALL links from retrieved docs (retrieval pipeline
            # filters by license/entity/dept so all current_docs should be relevant).
            # Exception: when LLM cited fewer than half the source docs, the retrieval was broader
            # than the question (e.g. Step 0b returns 40 OBD docs but only 1 is about exemptions).
            # In that case, also filter form/guide links to only cited docs to avoid flooding
            # the answer with unrelated registration forms.
            _raw_used_idxs = decision.get("used_doc_indices")
            # Always use _docs_to_process here — must match the list enumerated in Pass 1
            # (_url_to_doc_idx) and the list the LLM saw in docs_json. Using _docs_for_links
            # (pre-sibling snapshot) caused index mismatch: sibling doc at index N in
            # _docs_to_process maps to a different doc at index N in _docs_for_links.
            _link_src_docs = _docs_to_process
            _n_link_src = len(_link_src_docs)
            if isinstance(_raw_used_idxs, list) and _raw_used_idxs:
                _used_set = {int(i) for i in _raw_used_idxs if isinstance(i, (int, float)) and 0 <= int(i) < _n_link_src}
                if _used_set:
                    _link_ref = [(d, u) for d, u in _link_ref if _url_to_doc_idx.get(u, -1) in _used_set]
                    _narrow_answer = _n_link_src >= 3 and len(_used_set) < _n_link_src / 2
                    if _narrow_answer:
                        _link_form = [(d, u) for d, u in _link_form if _url_to_doc_idx.get(u, -1) in _used_set]
                        _link_guide = [(d, u) for d, u in _link_guide if _url_to_doc_idx.get(u, -1) in _used_set]
                        _LOG.info(
                            "[Practical] narrow answer (used=%d/%d) — filtered form/guide links to cited docs only",
                            len(_used_set), _n_link_src,
                        )
                    _LOG.info(
                        "[Practical] used_doc_indices=%s → service=%d form=%d guide=%d ref=%d links",
                        sorted(_used_set), len(_link_service), len(_link_form), len(_link_guide), len(_link_ref),
                    )

            # Restore truncated URLs: LLM sometimes shortens a full URL to just scheme://domain/
            # e.g. https://edbr.dbd.go.th/termsconditions → https://edbr.dbd.go.th/
            #
            # Build URL pool in 3 tiers (each more expensive but broader):
            #
            # Tier 1: classified link lists already built above (from _docs_to_process)
            # Tier 2: ALL retrieved docs (_all_docs) — catches URLs in docs beyond _prompt_max_docs
            # Tier 3: Chroma metadata-only query for current license_type — catches URLs in docs
            #         that were never retrieved at all (e.g. informational queries that retrieve
            #         generic docs instead of the entity-specific ones with portal URLs)
            _url_pool: set = set()

            # Tier 1
            for _, _u in (_link_service + _link_form + _link_guide):
                if _u:
                    _url_pool.add(_u)

            # Tier 2
            for _du in _all_docs:
                _rr_u = str((_du.get("metadata") or {}).get("research_reference") or "").strip()
                if _rr_u and _rr_u not in ("nan", "None"):
                    for _, _u in _parse_link_entries(_rr_u):
                        if _u:
                            _url_pool.add(_u)

            # Tier 3: Chroma metadata fetch for current license_type (no embedding — fast)
            # Only run if the answer contains a URL fragment that isn't already fully known.
            _lt_for_url = ""
            for _d_lt in _all_docs:
                _lt_for_url = ((_d_lt.get("metadata") or {}).get("license_type") or "").strip()
                if _lt_for_url:
                    break
            if not _lt_for_url:
                _lt_for_url = (state.context or {}).get("last_topic", "") or ""
            if _lt_for_url:
                try:
                    _vs = getattr(self.retriever, "vectorstore", None)
                    _col = getattr(_vs, "_collection", None) if _vs else None
                    if _col is not None:
                        _chroma_res = _col.get(
                            where={"license_type": {"$eq": _lt_for_url}},
                            include=["metadatas"],
                        )
                        for _m_c in (_chroma_res.get("metadatas") or []):
                            _rr_c = str(_m_c.get("research_reference") or "").strip()
                            if _rr_c and _rr_c not in ("nan", "None"):
                                for _, _u_c in _parse_link_entries(_rr_c):
                                    if _u_c:
                                        _url_pool.add(_u_c)
                except Exception as _e_url:
                    _LOG.debug("[Practical] URL pool Chroma fetch failed: %s", _e_url)

            # Sort longest first so more specific paths match before their domain prefixes
            _all_known_urls = sorted(_url_pool, key=len, reverse=True)
            if _all_known_urls:
                def _restore_url_match(m: re.Match) -> str:
                    _found = m.group(0)
                    # If this exact URL is already known → it's complete, no restoration needed
                    if _found in _url_pool:
                        return _found
                    # If URL ends with '/' or a file extension → treat as complete, don't expand
                    # e.g. "/webapp/" is a complete URL, not a truncated prefix of "/webapp/assets/..."
                    if _found.endswith('/') or re.search(r'\.\w{2,5}$', _found):
                        return _found
                    for _full in _all_known_urls:
                        if _full.startswith(_found) and _full != _found:
                            return _full
                    return _found
                ans = re.sub(r'https?://\S+', _restore_url_match, ans)

            # URL validation: strip hallucinated URLs from 📄 แบบฟอร์ม section.
            # LLM sometimes fabricates form URLs from training knowledge (e.g. "vat01.pdf" for ภ.พ.01)
            # despite prompt rules — especially after token-budget summarization alters context.
            # Only URLs in _url_pool (built from Chroma data) are valid; anything else is stripped.
            # Also removes the orphaned description line preceding a stripped URL.
            if "📄" in ans and _url_pool:
                _val_lines = ans.split("\n")
                _val_out: list = []
                _val_in_form = False
                _val_prev_desc = False  # True when previous line was a non-URL description line
                for _vln in _val_lines:
                    _vstripped = _vln.strip()
                    # Detect 📄 section start (header line — contains 📄 but no URL)
                    if "📄" in _vstripped and not re.search(r'https?://', _vstripped):
                        _val_in_form = True
                        _val_prev_desc = False
                        _val_out.append(_vln)
                        continue
                    # Detect next section header (emoji) → exit 📄 scope
                    if _val_in_form and re.search(r'^[📋📌🏪🌐📖📚]', _vstripped):
                        _val_in_form = False
                    if _val_in_form:
                        _vurl_m = re.search(r'https?://\S+', _vstripped)
                        if _vurl_m:
                            _vurl = _vurl_m.group(0).rstrip('.,;)')
                            if _vurl in _url_pool:
                                _val_out.append(_vln)
                            else:
                                # Hallucinated URL — drop this line and orphaned preceding desc
                                _LOG.warning("[Practical] hallucinated form URL stripped: %r", _vurl)
                                if _val_prev_desc and _val_out:
                                    _val_out.pop()
                            _val_prev_desc = False
                        else:
                            _val_out.append(_vln)
                            _val_prev_desc = bool(_vstripped)
                    else:
                        _val_out.append(_vln)
                        _val_prev_desc = False
                ans = "\n".join(_val_out)
                # Remove empty 📄 section (header line with no content following it)
                ans = re.sub(r'\n📄[^\n]*\n(\s*\n)+', '\n', ans)
                ans = re.sub(r'\n📄[^\n]*\s*$', '', ans).rstrip()

            # Post-process link header lines: strip redundant "Website"/"เว็บไซต์" prefix
            # and deduplicate entries with identical desc in 📄/📖 sections.
            if '📄' in ans or '📖' in ans:
                # Step 1: global regex strip of "Website"/"เว็บไซต์" from ALL link header lines
                # ️ is the Unicode Variation Selector-16 that LLMs sometimes append to emoji
                ans = re.sub(
                    r'(?m)^([ \t]*📄️?[ \t]*)(?:Website|เว็บไซต์)[ \t]+',
                    r'\1', ans, flags=re.IGNORECASE,
                )
                ans = re.sub(
                    r'(?m)^([ \t]*📖️?[ \t]*)(?:Website|เว็บไซต์)[ \t]+',
                    r'\1', ans, flags=re.IGNORECASE,
                )
                # Step 2: dedup entries with identical header line (keep first occurrence)
                _pp_seen: set = set()
                _pp_out2: list = []
                _pp_skip2 = False
                for _pp_ln in ans.split('\n'):
                    _pp_s = _pp_ln.strip()
                    _pp_is_hdr = _pp_s.startswith('📄') or _pp_s.startswith('📖')
                    if _pp_is_hdr:
                        # normalize variation selector so "📄️ foo" == "📄 foo" in dedup key
                        _pp_key = _pp_s.replace('️', '').lower()
                        if _pp_key in _pp_seen:
                            _pp_skip2 = True
                            continue
                        _pp_seen.add(_pp_key)
                        _pp_skip2 = False
                        _pp_out2.append(_pp_ln)
                    elif _pp_skip2 and re.match(r'^\s+https?://', _pp_ln):
                        _pp_skip2 = False
                        continue
                    else:
                        _pp_skip2 = False
                        _pp_out2.append(_pp_ln)
                ans = '\n'.join(_pp_out2)

            # Layer 3 of hallucination prevention: strip phone/hotline numbers not found in docs.
            # Uses the same _doc_phones whitelist built during prompt construction above.
            # Only strips numbers appearing in a phone-keyword context (📞, สายด่วน, โทร, etc.)
            # to avoid false positives on fee amounts, years, and form reference numbers.
            _ANS_PHONE_CTX_RE = re.compile(
                r'(?:📞|โทร(?:ศัพท์|สาร)?|สายด่วน|Tel\.?|Fax\.?|แฟกซ์)'
                r'\s*[:\s]*(\d[\d\s\-\.]{3,13})',
                re.IGNORECASE,
            )
            _phone_out: list = []
            _phone_stripped = False
            for _pl in ans.split('\n'):
                _pm_ans = _ANS_PHONE_CTX_RE.search(_pl)
                if _pm_ans:
                    _norm_ans = re.sub(r'[\s\-\.]', '', _pm_ans.group(1))
                    if _norm_ans.isdigit() and 4 <= len(_norm_ans) <= 12 and _norm_ans not in _doc_phones:
                        _LOG.warning(
                            "[Practical] hallucinated phone number stripped: %r from: %r",
                            _norm_ans, _pl[:80],
                        )
                        _phone_stripped = True
                        continue
                _phone_out.append(_pl)
            if _phone_stripped:
                ans = '\n'.join(_phone_out)

            ans = self._apply_practical_lint(ans, kind="answer")

            # Guard: skip ALL link safety nets when LLM returned the error fallback.
            # Error answers have no content — appending links creates a confusing output.
            _ans_is_error_fallback = "ขออภัยครับ ระบบประมวลผลคำถามไม่สำเร็จ" in ans

            # Safety net: LLM sometimes omits FORM_LINKS when context is large (token budget exceeded).
            # If form links were available and the answer contains a document list but no 📄 section → append directly.
            # _docs_have_id_docs covers regulatory docs with identification_documents metadata.
            # _answer_has_doc_list: answer explicitly mentions required documents or specific form codes.
            # Broader than a section-header check — also catches inline form references like "ภพ.01" or "แบบคำขอ".
            _answer_has_doc_list = bool(re.search(
                r'(?:^|\n)\s*(?:[📋📄]\s*)?(?:เอกสารที่ต้องใช้|รายการเอกสาร|ต้องใช้เอกสาร|เอกสารประกอบ|หลักฐานที่ต้องใช้|เอกสารที่ต้องเตรียม)'
                r'|ภพ\.\s*\d+|แบบ\s*ภ[./]\s*\d+|แบบคำขอ|แบบฟอร์ม|ต้องยื่น.{0,20}แบบ'
                r'|สำเนา(?:บัตร|ทะเบียน|หนังสือ)|หนังสือ(?:รับรอง|มอบอำนาจ)',
                ans, re.MULTILINE
            ))
            # _ans_has_steps: answer contains numbered procedural steps (ยื่น/คลิก/ระบุ/กรอก/แนบ/ส่ง).
            # Used to decide guide links — distinct from numbered lists in informational answers
            # (e.g. "1. กลุ่ม A..." is NOT a procedural step).
            _ans_has_steps = bool(re.search(
                r'(?:^|\n)\s*\d+\.\s+(?:เลือก|คลิก|ยื่น|ระบุ|กรอก|แนบ|ส่ง|ดาวน์โหลด|ตรวจสอบ|ลงทะเบียน|เข้าสู่ระบบ)',
                ans, re.MULTILINE
            ))

            # Tier 4: If _link_form is still empty but answer has a doc list,
            # fallback to direct Chroma lookup by license_type to find form links.
            # This handles cases where retrieved docs have empty research_reference metadata
            # but other docs of the same license_type in Chroma do have links.
            # Trigger on LLM answer content (_answer_has_doc_list) or user explicitly asked for forms —
            # NOT on _docs_have_id_docs alone (that's a doc property, not question intent).
            # e.g. "ต้องการลิ้งค์" → answer is just a URL, _answer_has_doc_list=False → skip Tier4.
            # Guard: skip for penalty/definition questions (_needs_forms=False).
            if not _link_form and _needs_forms and (_user_wants_forms or _answer_has_doc_list) and "📄" not in ans and not _ans_is_error_fallback:
                try:
                    _vs4 = getattr(self.retriever, "vectorstore", None)
                    _col4 = getattr(_vs4, "_collection", None) if _vs4 else None
                    # Collect ALL distinct license_types from docs (multi-license: iterate all)
                    _lt4_all: list = []
                    for _d4 in _all_docs:
                        _lt4 = ((_d4.get("metadata") or {}).get("license_type") or "").strip()
                        if _lt4 and _lt4 not in _lt4_all:
                            _lt4_all.append(_lt4)
                    if _col4 and _lt4_all:
                        _lt4 = _lt4_all[0]  # used in log below
                        # Collect metadatas from ALL distinct license_types (multi-license support)
                        _all_meta4: list = []
                        for _lt4x in _lt4_all:
                            try:
                                _r4x = _col4.get(
                                    where={"license_type": {"$eq": _lt4x}},
                                    include=["metadatas"],
                                )
                                _all_meta4.extend(_r4x.get("metadatas") or [])
                            except Exception:
                                pass
                        _seen4: set = set()
                        # Track last seen desc so orphan URL lines (no desc) inherit it for classification
                        _last_desc4 = ""
                        # entity_type filter: only include docs matching known entity_type (or blank).
                        # Prefer _obd_entity_override (single-turn) over collected_slots so entity-specific
                        # concept answers (e.g. ตราประทับ for นิติบุคคล) get the right form links.
                        _saved_et4 = ""
                        try:
                            _saved_et4 = (
                                str((state.context or {}).get("_obd_entity_override") or "").strip()
                                or (state.get_collected_slots() or {}).get("entity_type") or ""
                            )
                        except Exception:
                            pass
                        # When entity not in collected slots, derive it from the primary retrieved docs.
                        # OBD chapter docs (from Step 0b) carry entity_type_normalized in their metadata.
                        # Using this prevents Tier4 from returning form docs for the wrong entity type
                        # (e.g. showing 15 บอจ.1-5 บริษัทจำกัด forms for a บุคคลธรรมดา ยกเลิก query).
                        if not _saved_et4:
                            for _d4_et in (_docs_for_links if _docs_for_links is not None else _docs_to_process):
                                _et4_doc = str((_d4_et.get("metadata") or {}).get("entity_type_normalized") or "").strip()
                                if _et4_doc and _et4_doc not in ("nan", "None"):
                                    _saved_et4 = _et4_doc
                                    break
                        # registration_type filter: only include docs matching selected sub-type (or blank)
                        _saved_rt4 = ""
                        try:
                            _saved_rt4 = (state.get_collected_slots() or {}).get("registration_type") or ""
                        except Exception:
                            pass
                        # When no slot collected, derive rt from primary docs' registration_type.
                        # Only use specific sub-types (บริษัท / ห้างหุ้นส่วน), not entity-level values.
                        # Prevents Tier4 from injecting บอจ forms for a ห้างหุ้นส่วน direct query.
                        if not _saved_rt4:
                            _T4_SPECIFIC_RT_KEYS = ("บริษัท", "ห้างหุ้นส่วน")
                            for _d4_fb in (_docs_for_links if _docs_for_links is not None else _docs_to_process):
                                _rt4_fb = str((_d4_fb.get("metadata") or {}).get("registration_type") or "").strip()
                                if _rt4_fb and _rt4_fb not in ("nan", "None") and any(k in _rt4_fb for k in _T4_SPECIFIC_RT_KEYS):
                                    _saved_rt4 = _rt4_fb
                                    break
                        # sub_topic scope filter: build set of sub_topics from primary (pre-sibling) docs.
                        # Prevents Tier 4 from injecting form links from unrelated operations that share
                        # the same license_type (e.g. จดทะเบียนใหม่ forms showing for ชำรุด/สูญหาย queries
                        # where sibling-completion added many off-topic siblings to _docs_to_process).
                        # A doc with a blank sub_topic passes through (no strict match required).
                        _t4_primary_sub_topics: set = set()
                        for _d4_prim in (_docs_for_links if _docs_for_links is not None else _docs_to_process):
                            _st4_prim = str((_d4_prim.get("metadata") or {}).get("sub_topic") or "").strip()
                            if _st4_prim and _st4_prim not in ("nan", "None"):
                                _t4_primary_sub_topics.add(_st4_prim)
                        for _m4 in _all_meta4:
                            # Cap: stop after 10 form links (some licenses have many forms)
                            if len(_link_form) >= 10:
                                break
                            # Filter: skip docs whose operation_topic conflicts with detected intent
                            _op4 = str(_m4.get("operation_topic") or "").strip()
                            if _op_exclude_re and _op4 and re.search(_op_exclude_re, _op4):
                                continue
                            # Filter: skip docs whose sub_topic differs from the primary docs' scope.
                            # Applies only when primary docs have a known sub_topic and the candidate
                            # doc also has a sub_topic (blank sub_topic → unconstrained, passes through).
                            if _t4_primary_sub_topics:
                                _st4_cand = str(_m4.get("sub_topic") or "").strip()
                                if _st4_cand and _st4_cand not in ("nan", "None") and _st4_cand not in _t4_primary_sub_topics:
                                    continue
                            # Filter: skip docs for a different entity_type when user's entity is known
                            if _saved_et4:
                                _et4 = str(_m4.get("entity_type_normalized") or "").strip()
                                if _et4 and _et4 != _saved_et4:
                                    continue
                            # Filter: skip form links from different registration sub-type when known.
                            # Use substring matching: "ห้างหุ้นส่วน" matches stored
                            # "1.ห้างหุ้นส่วนจำกัด 2.ห้างหุ้นส่วนสามัญ" and vice versa.
                            if _saved_rt4:
                                _rt4m = str(_m4.get("registration_type") or "").strip()
                                if _rt4m and _rt4m not in ("nan", "None"):
                                    if not (_saved_rt4 in _rt4m or _rt4m in _saved_rt4):
                                        continue
                            _rr4 = str(_m4.get("research_reference") or "").strip()
                            if not _rr4 or _rr4 in ("nan", "None"):
                                continue
                            for _d4e, _u4e in _parse_link_entries(_rr4):
                                _k4 = (_u4e or _d4e).strip()
                                if not _k4 or _k4 in _seen4:
                                    continue
                                _seen4.add(_k4)
                                # Orphan URL lines (desc empty) inherit the last seen desc for classify
                                _eff_desc4 = _d4e if _d4e else _last_desc4
                                if _d4e:
                                    _last_desc4 = _d4e
                                if _classify_link(_eff_desc4, _u4e) == "form":
                                    # Filter by link desc (Thai keywords) and URL (English operation pattern)
                                    if _op_exclude_re:
                                        if _eff_desc4 and re.search(_op_exclude_re, _eff_desc4):
                                            continue
                                        _t4_url_excl_map = {
                                            r"เลิก|ยกเลิก|เปิดใหม่":           r"close_registration|new_registration",
                                            r"แก้ไข|เปลี่ยนแปลง|เปิดใหม่":     r"edit_registration|new_registration",
                                            r"เลิก|ยกเลิก|แก้ไข|เปลี่ยนแปลง": r"close_registration|edit_registration",
                                        }
                                        _t4_url_excl = _t4_url_excl_map.get(_op_exclude_re, "")
                                        if _t4_url_excl and re.search(_t4_url_excl, (_u4e or "")):
                                            continue
                                    # Positive filter for replacement-cert queries (ชำรุด/สูญหาย/ใบแทน):
                                    # Only include forms explicitly about ใบแทน/ทดแทน.
                                    # If no such forms exist in data, show nothing (better than wrong forms).
                                    if _op_excl_t4_desc_extra:
                                        _repl_re = r"ใบแทน|ทดแทน|สูญหาย|ชำรุด|replacement"
                                        if not (re.search(_repl_re, (_eff_desc4 or "")) or re.search(_repl_re, (_u4e or ""))):
                                            continue
                                    _link_form.append((_eff_desc4, _u4e))
                        _LOG.info(
                            "[Practical] Tier4 Chroma form-link lookup: lt=%r found=%d form links",
                            _lt4, len(_link_form),
                        )
                except Exception as _e4:
                    _LOG.debug("[Practical] Tier4 form-link lookup failed: %s", _e4)

            # Safety net: inject 🌐 service links if LLM did not include them.
            _service_urls_in_ans = any(u and u in ans for _, u in _link_service)
            if _link_service and not _suppress_links_broad and not _is_contact_q and not _service_urls_in_ans and not _ans_is_error_fallback and not _suppress_non_form_links and (_user_wants_registration or _user_wants_links or _docs_have_id_docs or _ans_has_steps):
                _service_url_lines = [f"  {_sv_url}" for _, _sv_url in _link_service if _sv_url]
                if _service_url_lines:
                    ans = ans.rstrip() + "\n\n🌐 ลิงก์สมัครบริการ\n" + "\n".join(_service_url_lines)
                    _LOG.debug("[Practical] Safety net: appended %d service link(s)", len(_service_url_lines))

            _form_urls_in_ans = any(u and u in ans for _, u in _link_form)
            if _link_form and _needs_forms and (_user_wants_forms or _answer_has_doc_list) and not _form_urls_in_ans and not _ans_is_error_fallback:
                _form_lines = []
                for _f_desc, _f_url in _link_form:
                    _f_desc_clean = re.sub(r'^[•\-\*]\s*', '', _f_desc).strip() if _f_desc else ""
                    if _f_desc_clean and _f_url:
                        _form_lines.append(f"📄 {_f_desc_clean}\n  {_f_url}")
                    elif _f_url:
                        _form_lines.append(f"📄 {_f_url}")
                    elif _f_desc_clean:
                        _form_lines.append(f"📄 {_f_desc_clean}")
                if _form_lines:
                    _t4_form_block = "\n\n" + "\n".join(_form_lines)
                    # Insert 📄 block before any closing sentence at the end of ans.
                    _t4_lines = ans.rstrip().split('\n')
                    _t4_closing: list = []
                    while _t4_lines:
                        _t4_tail = _t4_lines[-1].strip()
                        if not _t4_tail:
                            _t4_lines.pop()
                            continue
                        if (
                            len(_t4_tail) < 150
                            and 'ครับ' in _t4_tail
                            and not re.search(r'^\d+[.)]\s', _t4_tail)
                            and not _t4_tail.startswith('•')
                            and not _t4_tail.startswith('-')
                            and not _t4_tail.startswith('⚠')
                        ):
                            _t4_closing.insert(0, _t4_lines.pop())
                        else:
                            break
                    if _t4_closing:
                        ans = '\n'.join(_t4_lines).rstrip() + _t4_form_block + '\n\n' + '\n'.join(_t4_closing).strip()
                    else:
                        ans = ans.rstrip() + _t4_form_block
                    _LOG.debug("[Practical] Safety net: appended %d form link(s) omitted by LLM", len(_form_lines))

            # Safety net: inject guide links if LLM skipped them when answer includes steps.
            # Guards must mirror the main injection block (line ~3807):
            # - _guide_auto_ok: False when _is_link_req_ctx (user only wants the registration link)
            # - _is_contact_q: True when query is a contact/address query — guides are irrelevant
            #   even if retrieved docs (e.g. registration docs for the same department) have
            #   operation_steps that set _answer_will_have_steps=True.
            _guide_urls_in_ans = any(u and u in ans for _, u in _link_guide)
            if _link_guide and not _suppress_links_broad and _guide_auto_ok and not _is_contact_q and not _suppress_non_form_links and (_user_wants_links or _ans_has_steps) and not _guide_urls_in_ans and not _ans_is_error_fallback:
                _guide_lines = []
                for _g_desc, _g_url in _link_guide:
                    _g_desc_clean = re.sub(r'^[•\-\*]\s*', '', _g_desc).strip() if _g_desc else ""
                    if _g_desc_clean and _g_url:
                        _guide_lines.append(f"📖 {_g_desc_clean}\n  {_g_url}")
                    elif _g_url:
                        _guide_lines.append(f"📖 {_g_url}")
                    elif _g_desc_clean:
                        _guide_lines.append(f"📖 {_g_desc_clean}")
                if _guide_lines:
                    _guide_block = "\n\n" + "\n".join(_guide_lines)
                    ans = ans.rstrip() + _guide_block
                    _LOG.debug("[Practical] Safety net: appended %d guide link(s) omitted by LLM", len(_guide_lines))

            # Safety net: inject 📚 reference links if LLM did not include them.
            _ref_urls_in_ans = any(u and u in ans for _, u in _link_ref)
            if _link_ref and not _suppress_links_broad and (_user_wants_reference or _user_wants_links) and not _ref_urls_in_ans and not _ans_is_error_fallback:
                _ref_lines = []
                for _r_desc, _r_url in _link_ref:
                    _r_desc_clean = re.sub(r'^[•\-\*]\s*', '', _r_desc).strip() if _r_desc else ""
                    if _r_desc_clean and _r_url:
                        _ref_lines.append(f"📚 {_r_desc_clean}\n  {_r_url}")
                    elif _r_url:
                        _ref_lines.append(f"📚 {_r_url}")
                    elif _r_desc_clean:
                        _ref_lines.append(f"📚 {_r_desc_clean}")
                if _ref_lines:
                    ans = ans.rstrip() + "\n\n" + "\n".join(_ref_lines)
                    _LOG.debug("[Practical] Safety net: appended %d ref link(s) omitted by LLM", len(_ref_lines))

            # Strip 📄/📖 lines the LLM independently generated from doc content
            # (e.g. identification_documents field) when user asked for a registration link only.
            # The LLM sees the full doc JSON and may format consent forms as 📄 even when
            # no form links were in its context.  Guards: skip when user explicitly asked for
            # forms (_user_wants_forms) or guides (_wants_guides_specifically).
            if _is_link_req_ctx and not _user_wants_forms and not _wants_guides_specifically and not _ans_is_error_fallback:
                _ans_before_strip = ans
                ans = re.sub(r'(?m)^📄[^\n]*(?:\n[ \t]+https?://[^\n]*)?', '', ans)
                ans = re.sub(r'(?m)^📖[^\n]*(?:\n[ \t]+https?://[^\n]*)?', '', ans)
                ans = re.sub(r'\n{3,}', '\n\n', ans).strip()
                if ans != _ans_before_strip:
                    _LOG.debug("[Practical] Link-req strip: removed 📄/📖 lines generated by LLM from doc content")

            if _suppress_non_form_links and not _ans_is_error_fallback:
                # user asked for document/form links only — strip 🌐 and 📖 from LLM output
                ans = re.sub(r'(?m)^🌐[^\n]*(?:\n[ \t]+[^\n]*)*', '', ans)
                ans = re.sub(r'(?m)^📖[^\n]*(?:\n[ \t]+https?://[^\n]*)?', '', ans)
                ans = re.sub(r'\n{3,}', '\n\n', ans).strip()

            if _suppress_links_broad and not _ans_is_error_fallback:
                # broad/informational query — LLM may generate 🌐/📖 from doc JSON even though
                # safety nets are suppressed; strip them so "คืออะไร" answers have no portal links.
                ans = re.sub(r'(?m)^🌐[^\n]*(?:\n[ \t]+[^\n]*)*', '', ans)
                ans = re.sub(r'(?m)^📖[^\n]*(?:\n[ \t]+https?://[^\n]*)?', '', ans)
                ans = re.sub(r'\n{3,}', '\n\n', ans).strip()

            self._append_assistant(state, ans)
            state.context["phase"] = None
            state.round = 0
            # LLM answered directly — any remaining slot queue is stale, clear it.
            # Exception: on error fallback, preserve the queue so the next turn re-asks the pending slot.
            if not _ans_is_error_fallback:
                state.context.pop("topic_slot_queue", None)
            # Clear multi-license signal after it has been consumed in the answer
            state.context.pop("multi_license_topics", None)
            # Track which license_type(s) was just answered so Academic can:
            # - single topic: auto-select it, skip Phase 0 topic menu entirely
            # - multi topic: use these as menu options instead of Chroma license_types
            # For broad questions (_broad_question flag), also include main_topic values
            # from non-regulatory docs so Academic menu shows ALL answer sections,
            # not just the license_type ones.
            if _is_broad_q_flag and (_mt_order or len(_lt_order) >= 2):
                # Broad answer: merge general (main_topic) + regulatory (license_type) topics
                _combined = [mt for mt in _mt_order if mt not in _lt_order] + list(_lt_order)
                if len(_combined) >= 2:
                    state.context["last_answered_license_types"] = _combined
                    state.context.pop("last_answered_license_type", None)
                elif len(_combined) == 1:
                    state.context["last_answered_license_type"] = _combined[0]
                    state.context.pop("last_answered_license_types", None)
            elif _lt_order and len(_lt_order) == 1:
                state.context["last_answered_license_type"] = _lt_order[0]
                state.context.pop("last_answered_license_types", None)
            elif _is_multi_license_docs and len(_lt_order) >= 2:
                # Multi-topic answer — store all topics for Academic menu
                state.context["last_answered_license_types"] = list(_lt_order)
                state.context.pop("last_answered_license_type", None)
            elif not _is_multi_license_docs and _all_docs:
                _lt_snap = (((_all_docs[0].get("metadata") or {})).get("license_type") or "").strip()
                if _lt_snap:
                    state.context["last_answered_license_type"] = _lt_snap
                    state.context.pop("last_answered_license_types", None)
            return state, ans

        fallback = "ผมยังไม่เข้าใจครับ บอกหัวข้อที่อยากรู้เกี่ยวกับร้านอาหารหน่อยครับ"
        fallback = self._apply_practical_lint(fallback, kind="ask")
        self._append_assistant(state, fallback)
        return state, fallback
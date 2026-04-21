# code/model/persona_practical.py
import json
import logging
import re
from typing import Tuple, Dict, Any, List, Optional, Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import conf
from model.conversation_state import ConversationState
from utils.llm_call import llm_invoke, extract_llm_text
from utils.prompts_practical import SYSTEM_PROMPT as SYSTEM_PROMPT_PRACTICAL

# Import professional logging
from utils.logger import get_logger, log_function_call, TimingContext

_LOG = logging.getLogger("restbiz.practical")  # Keep for backward compatibility
logger = get_logger(__name__)  # ใช้ logger ใหม่ (มี structure + context)

# Metadata fields with no semantic value for the LLM — always hidden from docs_json
_LLM_HIDDEN_METADATA_KEYS = frozenset({"row_id", "source"})


def _classify_link(desc: str, url: str) -> str:  # noqa: ARG001 — url intentionally unused
    """
    Classify a link based on desc ONLY — url is ignored.

    Categories:
      'guide'        — คู่มือ, วิดีโอ, workflow, ขั้นตอนการ
      'form'         — แบบฟอร์ม, เอกสาร, ดาวน์โหลด, แบบ บอจ, คำขอ etc.
      'registration' — ลงทะเบียนออนไลน์, e-service, สมัครบริการ, mobile app
      'ref'          — fallback (กฎหมาย, FAQ, หน้าข้อมูลทั่วไป)

    Priority order (first match wins): guide → form → registration → ref
    """
    desc_l = desc.lower().strip()

    # Guide: manual, video, workflow, how-to
    _GUIDE_KW = (
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
    if any(kw in desc_l for kw in _GUIDE_KW):
        return "guide"

    # Form: downloadable forms and documents
    _FORM_KW = (
        "แบบฟอร์ม",
        "แบบ บอจ", "แบบ ก.", "แบบ ว.", "แบบ สปส", "แบบ สณ",
        "แบบ ภพ", "แบบ ภส", "แบบ อส", "แบบ บค", "แบบ รส",
        "ดาวน์โหลดเอกสาร", "ดาวน์โหลดแบบฟอร์ม", "ดาวน์โหลด",
        "เอกสาร",
        "แบบคำขอ", "แบบแจ้ง", "แบบแสดง", "แบบคำรับรอง",
        "คำขอจดทะเบียน", "คำขอใช้บริการ", "คำขอ",
        "ตัวอย่างการกรอก", "ตัวอย่างการจดทะเบียน", "ตัวอย่าง",
        "ใบสมัคร",
        "หนังสือมอบอำนาจ", "หนังสือยินยอม", "หนังสือให้ความยินยอม",
        "บัญชีรายชื่อผู้ถือหุ้น",
    )
    if any(kw in desc_l for kw in _FORM_KW):
        return "form"

    # Registration: online portals and apps for applying
    _REG_KW = (
        "สำหรับลงทะเบียน", "ลงทะเบียนออนไลน์",
        "ยื่นออนไลน์", "ยื่นจดทะเบียนออนไลน์",
        "e-service", "e service", "eservice",
        "สมัครบริการ", "สมัครสมาชิก",
        "mobile application", "app store", "play store",
    )
    if any(kw in desc_l for kw in _REG_KW):
        return "registration"

    # Ref: fallback (laws, FAQ, general info pages)
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
    "research_reference",      # ลิงก์แบบฟอร์ม / คู่มือ / เว็บไซต์ราชการ
    "answer_guideline",        # แนวคำตอบ (marketing/business_guide docs)
    "operation_steps",         # ขั้นตอนการดำเนินการ (สำคัญ — content อาจถูกตัดก่อนถึงส่วนนี้)
    "identification_documents",# รายการเอกสารที่ต้องใช้ (สำคัญ — ต้องเห็นทั้งหมด)
    "legal_regulatory",        # ข้อกำหนดทางกฎหมาย บทลงโทษ ค่าปรับ
    "terms_and_conditions",    # เงื่อนไขและหน้าที่ของผู้ประกอบการ
})

# P0: practical policy "last gate"
try:
    from utils.practical_lint import enforce_practical_policy  # type: ignore
except Exception:  # pragma: no cover
    enforce_practical_policy = None  # fallback handled in _apply_practical_lint


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
        if content is None:
            return
        c = str(content)
        if not c.strip():
            return
        if (
            state.messages
            and state.messages[-1].get("role") == "user"
            and (state.messages[-1].get("content") or "").strip() == c.strip()
        ):
            return
        state.messages.append({"role": "user", "content": c})

    def _append_assistant(self, state: ConversationState, content: str) -> None:
        """
        P0: Dedupe assistant to prevent recursion/forced flows from duplicating turns.
        Also tag last_bot_owner='practical' for safe pending-slot recovery gating.
        """
        c = "" if content is None else str(content)
        if not c.strip():
            c = c.strip()
        if state.messages:
            last = state.messages[-1]
            if last.get("role") == "assistant" and (last.get("content") or "").strip() == c.strip():
                return
        state.messages.append({"role": "assistant", "content": c})
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
            except Exception:
                return {}

        return _call

    def _pick_greet_prefix(self, kind: str, menu: List[str], greet_streak: int) -> str:
        kind2 = kind if kind in {"greet", "thanks", "smalltalk", "blank"} else "greet"
        try:
            res = self.llm_greet_call(kind2, menu, int(greet_streak or 0))
        except Exception:
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

        # If still no question phrasing — derive from slot_key context, not a fixed string
        if not re.search(r"(ไหม|หรือ|ยังไง|อย่างไร|อะไร|มั้ย|ได้ไหม|ต้องการ|อยาก|เป็นแบบ|รูปแบบ|ประเภท|ขนาด|พื้นที่|เท่าไหร่|เท่าใด|ตั้งอยู่|ใด|ดำเนินการ)", t):
            _SLOT_QUESTIONS = {
                "entity_type":         "ธุรกิจของคุณเป็นรูปแบบใดครับ?",
                "operation_location":  "ร้านของคุณตั้งอยู่ในพื้นที่ไหนครับ?",
                "shop_area_type":      "ร้านของคุณมีขนาดพื้นที่ประมาณเท่าไหร่ครับ?",
                "area_size":           "ร้านของคุณมีพื้นที่เท่าไหร่ครับ?",
                "area_type":           "ร้านของคุณมีพื้นที่เท่าไหร่ครับ?",
                "registration_type":   "การจดทะเบียนของคุณเป็นรูปแบบใดครับ?",
                "operation_group":     "ต้องการดำเนินการเรื่องใดครับ?",
                "topic":               "ต้องการข้อมูลเกี่ยวกับอะไรครับ?",
            }
            t = _SLOT_QUESTIONS.get(slot_key, "ช่วยบอกข้อมูลเพิ่มเติมหน่อยได้ไหมครับ?")
        else:
            if not t.endswith("ครับ"):
                t = t.rstrip(" .") + "ครับ"
        return t

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

        if "?" in t or "？" in t:
            # Only strip at a "?" if what follows looks like a new question sentence or menu option,
            # not if "?" appears mid-answer as part of a section header or inline phrase.
            # Safe to cut when: "?" is near the END and is followed by only whitespace/options.
            _q_pos = min(
                (t.find("?") if "?" in t else len(t)),
                (t.find("？") if "？" in t else len(t)),
            )
            _after = t[_q_pos + 1:].strip()
            # Cut only if "?" is at or near the end (trailing question), not mid-body
            if len(_after) < 120:
                t = t[:_q_pos].strip()

        # Dedup URLs: remove from links section any URL already present in the body
        _links_hdr = re.search(r'(?m)^📎[^\n]*$', t)
        if _links_hdr:
            _body_part = t[:_links_hdr.start()]
            _links_part = t[_links_hdr.start():]
            _body_urls = set(re.findall(r'https?://\S+', _body_part))
            if _body_urls:
                _links_lines = []
                for _ln in _links_part.split('\n'):
                    _found = re.search(r'https?://\S+', _ln)
                    if _found and _found.group(0) in _body_urls:
                        continue  # skip duplicate
                    _links_lines.append(_ln)
                # If links section is now empty (only header left), remove it entirely
                _remaining = [l for l in _links_lines[1:] if l.strip()]
                if _remaining:
                    t = _body_part + '\n'.join(_links_lines)
                else:
                    t = _body_part.rstrip()

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

    _PHASE3_CANONICAL = [
        "ขั้นตอนการดำเนินการ",
        "เอกสารที่ต้องใช้",
        "ค่าธรรมเนียม",
        "ระยะเวลาดำเนินการ",
        "ช่องทางยื่นคำขอ / หน่วยงาน",
        "ข้อกำหนดทางกฎหมาย และข้อบังคับ",
        "ฟอร์มเอกสารตัวจริง",
        "ทั้งหมด",
    ]

    _SECTION_SIGNALS: Dict[str, Dict[str, Any]] = {
        "ขั้นตอนการดำเนินการ": {
            "meta_keys": ["operation_steps", "operation_step", "steps", "procedure", "ขั้นตอนการดำเนินการ"],
            "content_keywords": ["ขั้นตอน", "วิธีดำเนินการ", "ลำดับ", "ยื่นคำขอ"],
        },
        "เอกสารที่ต้องใช้": {
            "meta_keys": [
                "identification_documents",
                "documents",
                "required_documents",
                "เอกสาร ยืนยันตัวตน",
                "เอกสารที่ต้องใช้",
            ],
            "content_keywords": ["เอกสาร", "สำเนา", "หลักฐาน", "แนบ", "ใบคำขอ"],
        },
        "ค่าธรรมเนียม": {
            "meta_keys": ["fees", "fee", "ค่าธรรมเนียม"],
            "content_keywords": ["ค่าธรรมเนียม", "บาท", "ชำระเงิน", "ค่าบริการ", "ราคา"],
        },
        "ระยะเวลาดำเนินการ": {
            "meta_keys": ["operation_duration", "duration", "ระยะเวลา การดำเนินการ", "ระยะเวลาดำเนินการ"],
            "content_keywords": ["ระยะเวลา", "วันทำการ", "ภายใน", "ใช้เวลา"],
        },
        "ช่องทางยื่นคำขอ / หน่วยงาน": {
            "meta_keys": ["service_channel", "department", "หน่วยงาน", "ช่องทางการ ให้บริการ", "channel"],
            "content_keywords": ["ช่องทาง", "ยื่น", "หน่วยงาน", "สำนักงานเขต", "เทศบาล", "ออนไลน์", "เว็บไซต์", "เคาน์เตอร์"],
        },
        "ข้อกำหนดทางกฎหมาย และข้อบังคับ": {
            "meta_keys": ["legal_regulatory", "law", "regulation", "ข้อกำหนดทางกฎหมาย และข้อบังคับ", "บทลงโทษ"],
            "content_keywords": ["กฎหมาย", "ข้อกำหนด", "ประกาศ", "พ.ร.บ", "บทลงโทษ", "ข้อบังคับ"],
        },
        "ฟอร์มเอกสารตัวจริง": {
            "meta_keys": [
                "restaurant_ai_document", "form", "template", "ฟอร์ม", "เอกสาร AI ร้านอาหาร",
                "research_reference",
            ],
            "content_keywords": ["แบบฟอร์ม", "ดาวน์โหลด", "ฟอร์ม", "ตัวอย่างเอกสาร", "คำขอ", "ช่องทางออนไลน์"],
        },
    }

    _PHASE3_MIN_SECTIONS = 2
    _PHASE3_ANSWER_CHAR_THRESHOLD = 520
    _PHASE3_ANSWER_LINE_THRESHOLD = 10

    # Retrieval reuse/new-topic heuristic (unchanged)
    _TOKEN_SPLIT_RE = re.compile(r"[\s/,\-–—|]+", re.UNICODE)
    _FOLLOWUP_SHORT_RE = re.compile(
        r"^(แล้ว(ไง|ล่ะ)?|แล้ว(เอกสาร|ขั้นตอน|ค่าธรรมเนียม)?|ต่อไปล่ะ|มีอะไรบ้าง|ขอ(เอกสาร|ขั้นตอน|ค่าธรรมเนียม|ระยะเวลา|ช่องทาง))\s*$"
    )
    # Continuation/follow-up questions — should always get a direct answer, never Phase 3 menu
    _CONTINUATION_RE = re.compile(
        r"(อีกไหม|อีกบ้าง|อีกมั้ย|ควรทำอะไรอีก|ต้องทำอะไรอีก|อะไรอีก|มีอะไรอีก"
        r"|แล้วต้อง|แล้วควร|แล้วล่ะ|เพิ่มเติม|ยังมีอะไร|ต้องทำด้วย|อีกด้วย"
        r"|ต่อไปต้อง|ขั้นต่อไป|หลังจากนั้น|นอกจากนี้|อื่นๆ.*ต้อง|ควรมี)",
        re.IGNORECASE,
    )

    def __init__(self, retriever):
        self.retriever = retriever
        self._topic_menu_cache: Optional[List[str]] = None
        self._topic_registry: Optional[List[str]] = None  # lazy-loaded from Chroma at first use
        self.llm_greet_call = self._default_greet_llm_call()
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

    def _looks_like_legal_question(self, s: str) -> bool:
        t = self._normalize_for_intent(s)
        return bool(self._LEGAL_SIGNAL_RE.search(t))

    def _looks_like_satisfaction(self, s: str) -> bool:
        t = self._normalize_for_intent(s)
        if not t:
            return False
        return bool(self._THANKS_RE.search(t) or self._OK_RE.match(t))

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

    def _is_short_followup(self, user_text: str) -> bool:
        t = (user_text or "").strip()
        if not t:
            return True
        n = self._normalize_for_intent(t)
        toks = self._tokenize_loose(n)
        if len(toks) <= 3 and len(n) <= 18:
            return True
        if self._FOLLOWUP_SHORT_RE.match(n):
            return True
        return False

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

        if self._is_short_followup(q):
            return False

        overlap = self._topic_overlap_ratio(last_q, q)
        if self._looks_like_legal_question(q) and overlap < 0.22:
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
        allow_multi = True if slot_key == self._PHASE3_SLOT_KEY else False
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

                obj = json.loads(text)
                
                # DEBUG LOG: show raw LLM JSON response before processing
                if isinstance(obj, dict):
                    action = obj.get("action", "?")
                    exec_data = obj.get("execution", {})
                    q = (exec_data.get("question") or "") if isinstance(exec_data, dict) else ""
                    _LOG.info("[Practical/json] LLM response: action=%r question=%r", action, q[:100])
                
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

    def _retrieve_docs(self, query: str, metadata_filter: Optional[Dict[str, Any]] = None, max_docs: Optional[int] = None, slot_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        import time
        start = time.time()

        max_docs = max_docs if max_docs is not None else int(getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 8))
        max_chars = getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 700)

        # Query expansion: short/abbrev keywords → full Thai terms for better embedding match
        # e.g. "vat" alone has low cosine similarity to "ภาษีมูลค่าเพิ่ม ภพ.20"
        _EXPAND_PATTERNS = [
            (r"\bvat\b|\bภพ\.?20\b|ภาษีมูลค่าเพิ่ม", "ภาษีมูลค่าเพิ่ม ภพ.20 จด VAT กรมสรรพากร"),
            (r"สุรา|เหล้า|ขายเหล้า", "ใบอนุญาตจำหน่ายสุรา สรรพสามิต ภส.08"),
            (r"ประกันสังคม", "ขึ้นทะเบียนประกันสังคม นายจ้าง ลูกจ้าง"),
            (r"ป้ายร้าน|ภาษีป้าย", "แบบแสดงรายการภาษีป้ายร้านอาหาร"),
            (r"ใบอนุญาต.*อาหาร|สถานที่จำหน่ายอาหาร|bma.*oss|oss.*bma", "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร สำนักงานเขต กรุงเทพมหานคร"),
            (r"ทะเบียนพาณิชย์|จดพาณิชย์|\bdbd\b", "ใบทะเบียนพาณิชย์ กรมพัฒนาธุรกิจการค้า จดทะเบียนพาณิชย์"),
            (r"qr.?pay|qr.?payment|พร้อมเพย์|promptpay", "ระบบชำระเงินออนไลน์ QR Payment พร้อมเพย์"),
            (r"\bedc\b|รูดบัตร|เครื่องรูดบัตร", "เครื่องรูดบัตร EDC ชำระเงิน"),
            (r"เปิดบัญชี.*ร้าน|บัญชีธนาคาร.*ร้าน|รับเงิน.*ร้าน", "บัญชีธนาคารรับเงิน ร้านอาหาร"),
        ]
        _expansions: list = []
        for pattern, expansion in _EXPAND_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE) and expansion not in _expansions:
                _expansions.append(expansion)
        expanded_query = (query + " " + " ".join(_expansions)).strip() if _expansions else query
        if _expansions:
            _LOG.info("[Practical] query expanded: %r → %r", query[:50], expanded_query[:80])

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
                _LOG.info("[Practical] slot context appended to query: %r", expanded_query[:80])

        vectorstore = getattr(self.retriever, "vectorstore", None)

        def _scored_search(q: str, k: int, flt: Optional[dict] = None) -> list:
            """Use similarity_search_with_relevance_scores and attach _sim to metadata."""
            if vectorstore is None:
                return list(self.retriever.invoke(q))
            kwargs: dict = {"k": k}
            if flt:
                kwargs["filter"] = flt
            try:
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
            _LOG.info("[Practical] retrieve(unfiltered) query=%r top_k=%d", expanded_query[:60], top_k)
            docs = _scored_search(expanded_query, top_k)

        retrieval_ms = (time.time() - start) * 1000
        
        # Token Optimization: Filter by similarity score
        # เลือกเฉพาะเอกสารที่มี similarity > threshold
        min_similarity = getattr(conf, 'RETRIEVAL_MIN_SIMILARITY', 0.6)
        filtered_docs = []
        low_quality_docs = []
        
        for d in docs:
            score = (getattr(d, "metadata", {}) or {}).get("_sim") or getattr(d, 'score', None)
            if score is not None and score >= min_similarity:
                filtered_docs.append(d)
            elif score is not None:
                low_quality_docs.append((d, score))
        
        # Safety: ถ้ากรองจนเหลือน้อยเกิน ให้เอาเอกสารเดิมมาใช้
        if len(filtered_docs) < 2:
            logger.log_with_data("warning", "Similarity filter เข้มเกิน fallback ใช้เอกสารทั้งหมด", {
                "filtered_count": len(filtered_docs),
                "total_docs": len(docs),
                "min_similarity": min_similarity
            })
            filtered_docs = docs
        else:
            _LOG.info("[Practical] sim_filter: %d → %d docs (removed %d below %.2f)", len(docs), len(filtered_docs), len(low_quality_docs), min_similarity)
        
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
            "legal_regulatory",     # บทลงโทษ ค่าปรับ ข้อกำหนดทางกฎหมาย
            "terms_and_conditions", # หน้าที่และเงื่อนไขของผู้ประกอบการ
        })
        _STORE_FIELD_CAPS = {
            # Must be >= the per-field caps used in the handle() prompt loop below,
            # otherwise the storage cut dominates and the prompt cap has no effect.
            "operation_steps": 1000, "identification_documents": 1500,
            "research_reference": 3100, "fees": 500, "service_channel": 500,
            "legal_regulatory": 2000, "terms_and_conditions": 800,
            # marketing / business_guide content fields
            "answer_guideline": 1500, "main_topic": 120, "sub_topic": 150,
        }
        results: List[Dict[str, Any]] = []
        for d in docs[:max_docs]:
            raw_md = getattr(d, "metadata", {}) or {}
            slim_md = {}
            for k, v in raw_md.items():
                if k not in _STORE_META_WHITELIST:
                    continue
                if v in (None, "", "nan", "None"):
                    continue
                v_str = str(v)
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

            _LOG.info("[Practical] doc[%d/%d] topic=%r | %s", i + 1, len(results), topic, " ".join(_parts))
            _top_topics.append(topic)
            if license_t:
                _license_counts[license_t] = _license_counts.get(license_t, 0) + 1

        _LOG.info("[Practical] retrieved_topics=%s", _top_topics)
        if _license_counts:
            _breakdown = " | ".join(f"{lt}={n}" for lt, n in sorted(_license_counts.items(), key=lambda x: -x[1]))
            _LOG.info("[Practical] license_breakdown: %s", _breakdown)

        return results

    # Topic Registry (auto-discovery from Chroma, no hardcoding)
    def _build_topic_registry(self) -> List[str]:
        """Read all unique operation_topic values from the Chroma collection.
        This is the source-of-truth registry — auto-updates when data is re-ingested."""
        vectorstore = getattr(self.retriever, "vectorstore", None)
        if vectorstore is None:
            _LOG.warning("[Practical] Topic registry: vectorstore not available")
            return []
        try:
            coll = getattr(vectorstore, "_collection", None)
            if coll is None:
                return []
            result = coll.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []
            topics: set = set()
            for md in metadatas:
                t = ((md or {}).get("operation_topic") or "").strip()
                if t:
                    topics.add(t)
            registry = sorted(topics)
            _LOG.info("[Practical] Topic registry loaded: %d unique topics", len(registry))
            return registry
        except Exception as e:
            _LOG.warning("[Practical] Topic registry build failed: %s", e)
            return []

    def _get_topic_registry(self) -> List[str]:
        """Lazy-load topic registry (cached for lifetime of this instance)."""
        if self._topic_registry is None:
            self._topic_registry = self._build_topic_registry()
        return self._topic_registry

    def _select_relevant_topics(self, question: str, registry: List[str]) -> List[str]:
        """Ask LLM which topics from the registry are relevant to this question.
        Returns a list of matched topic strings (may be empty if none match)."""
        if not registry:
            return []
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(registry))
        prompt = (
            "คุณเป็น AI ช่วยกรองหัวข้อกฎหมาย\n"
            "จากคำถามของ user ด้านล่าง กรุณาเลือกหัวข้อที่เกี่ยวข้องจากรายการ\n\n"
            f"คำถาม: {question}\n\n"
            f"รายการหัวข้อทั้งหมดในฐานข้อมูล:\n{numbered}\n\n"
            "ตอบเป็น JSON array ของ index (เลขที่) ที่เกี่ยวข้อง เช่น [1, 5, 12]\n"
            "ถ้าไม่มีหัวข้อที่เกี่ยวข้อง ตอบ []\n"
            "JSON:"
        )
        try:
            resp = llm_invoke(self.llm, [HumanMessage(content=prompt)], logger=_LOG, label="Practical/topic_select")
            text = extract_llm_text(resp).strip()
            m = re.search(r"\[[\d,\s]*\]", text)
            if not m:
                return []
            indices = json.loads(m.group())
            selected = []
            for idx in indices:
                try:
                    i = int(idx)
                    if 1 <= i <= len(registry):
                        selected.append(registry[i - 1])
                except Exception:
                    pass
            _LOG.info("[Practical] Topic selection: %d topics selected for question %r", len(selected), question[:60])
            return selected
        except Exception as e:
            _LOG.warning("[Practical] Topic selection failed: %s — falling back to direct retrieval", e)
            return []

    def _retrieve_multi_topic(self, question: str) -> List[Dict[str, Any]]:
        """Multi-topic retrieval — uses direct vector search (saves ~300 tokens vs LLM topic selection)."""
        return self._retrieve_docs(question)

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

    # Phase 3 helpers
    def _extract_available_phase3_sections(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Return list of canonical section labels that exist in the retrieved docs."""
        available: List[str] = []
        for label in self._PHASE3_CANONICAL:
            if label == "ทั้งหมด":
                continue
            sig = self._SECTION_SIGNALS.get(label)
            if not sig:
                continue
            meta_keys = sig.get("meta_keys") or []
            content_kws = sig.get("content_keywords") or []
            found = False
            for d in docs:
                md = d.get("metadata", {}) or {}
                for k in meta_keys:
                    v = md.get(k)
                    if v is not None and str(v).strip() and str(v).lower() != "nan":
                        found = True
                        break
                if found:
                    break
                content = d.get("content", "") or ""
                for kw in content_kws:
                    if kw and kw in content:
                        found = True
                        break
                if found:
                    break
            if found:
                available.append(label)
        return available

    def _user_requests_specific_sections(self, user_text: str) -> bool:
        """Return True if the user explicitly asked for a named section (skip phase3 menu)."""
        t = self._normalize_for_intent(user_text)
        for label in self._PHASE3_CANONICAL:
            if label == "ทั้งหมด":
                continue
            # Check shortened keywords from canonical labels
            for word in label.split():
                if len(word) >= 4 and word in t:
                    return True
        return False

    def _is_continuation_question(self, user_text: str) -> bool:
        """Return True if user is asking 'what else / what next' — should bypass Phase 3 and answer directly."""
        return bool(self._CONTINUATION_RE.search(user_text or ""))

    def _should_trigger_phase3(self, ans: str, available: List[str]) -> bool:
        """Return True if the answer is long enough and there are enough sections to offer a menu."""
        if len(available) < self._PHASE3_MIN_SECTIONS:
            return False
        if len(ans) >= self._PHASE3_ANSWER_CHAR_THRESHOLD:
            return True
        lines = [ln for ln in ans.splitlines() if ln.strip()]
        if len(lines) >= self._PHASE3_ANSWER_LINE_THRESHOLD:
            return True
        return False

    def _render_phase3_menu(self, available: List[str]) -> Tuple[str, List[str]]:
        """Render a numbered section menu. Returns (menu_text, options_list)."""
        options = list(available) + [self._PHASE3_ALL]
        lines = [self._PHASE3_MENU_HEADER]
        for i, opt in enumerate(options, 1):
            lines.append(f"{i}) {opt}")
        return "\n".join(lines), options

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
            if (not _internal) and self._looks_like_satisfaction(user_text):
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
                state.current_docs = self._retrieve_multi_topic(user_text)
                state.last_retrieval_query = user_text
                tmp = [
                    {"content": d.get("content", "")[:120], "metadata": d.get("metadata", {})}
                    for d in state.current_docs[:1]
                ]
                self._debug_log("post_retrieve", query=user_text, docs_json=tmp)
                return self.handle(state, "__auto_post_retrieve__", _internal=True)

        recent_msgs = state.messages[-6:]  # ส่งไป LLM 6 ล่าสุด

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
            "operation_steps": 1000,
            "identification_documents": 1500,
            "research_reference": 3100,
            "fees": 500,
            "operation_duration": 200,
            "service_channel": 500,
            "legal_regulatory": 2000,      # บทลงโทษ — ข้อมูลจริงยาว ~2000 chars
            "terms_and_conditions": 800,   # เงื่อนไขผู้ประกอบการ
        }
        _LONG_FIELDS_DEDUP = {"operation_steps", "legal_regulatory", "terms_and_conditions"}  # identification_documents varies by entity_type — never dedup

        # Cap docs sent to LLM: _prompt_max_docs per license_type to control token usage.
        # For multi-license, each license still gets its own metadata via dedup logic below.
        _all_docs = state.current_docs or []
        _lt_order: list = []  # license_types in order of first appearance
        for _d0 in _all_docs:
            _lt0 = ((_d0.get("metadata") or {}).get("license_type") or "").strip()
            if _lt0 and _lt0 not in _lt_order:
                _lt_order.append(_lt0)
        _is_multi_license_docs = len(_lt_order) > 1
        # Cap at _prompt_max_docs per license_type (prevents token explosion on multi-topic queries)
        if _is_multi_license_docs:
            _lt_counts: dict = {}
            _docs_to_process = []
            for _d0 in _all_docs:
                _lt0 = ((_d0.get("metadata") or {}).get("license_type") or "").strip()
                _lt_counts[_lt0] = _lt_counts.get(_lt0, 0) + 1
                if _lt_counts[_lt0] <= _prompt_max_docs:
                    _docs_to_process.append(_d0)
        else:
            _docs_to_process = _all_docs[:_prompt_max_docs]

        # Pass 1: classify research_reference links (globally deduped) → SERVICE / FORM / GUIDE
        # Same _classify_link logic as Academic — no URL pattern rules needed in the prompt
        _link_service: list = []  # (desc, url) registration/portal links — always shown
        _link_form: list = []     # (desc, url) fillable form links — always shown
        _link_guide: list = []    # (desc, url) guide/manual links — shown only when user asks
        _link_seen: set = set()   # global dedup key
        for _d1 in _docs_to_process:
            _rr_raw = str((_d1.get("metadata") or {}).get("research_reference") or "").strip()
            if not _rr_raw or _rr_raw in ("nan", "None"):
                continue
            for _desc1, _url1 in _parse_link_entries(_rr_raw):
                _key1 = (_url1 or _desc1).strip()
                if not _key1 or _key1 in _link_seen:
                    continue
                _link_seen.add(_key1)
                _cat1 = _classify_link(_desc1, _url1)
                if _cat1 == "registration":
                    _link_service.append((_desc1, _url1))
                elif _cat1 == "form":
                    _link_form.append((_desc1, _url1))
                elif _cat1 == "guide":
                    _link_guide.append((_desc1, _url1))
                # ref → dropped always

        # Flag: any retrieved doc has identification_documents content.
        # When true, always inject FORM_LINKS + SERVICE_LINKS regardless of query phrasing,
        # so users always get form links whenever a document list will be shown in the answer.
        _docs_have_id_docs = any(
            str((d.get("metadata") or {}).get("identification_documents") or "").strip()
            not in ("", "nan", "None")
            for d in _docs_to_process
        )

        # Pass 2: build docs_json with per-license dedup for long fields
        # research_reference is now injected as labeled sections outside docs_json
        _long_fields_sent_by_lt: dict = {}  # lt → bool
        docs_json = []
        for d in _docs_to_process:
            md = d.get("metadata", {}) or {}
            _lt2 = (md.get("license_type") or "").strip()
            filtered_md = {}
            for k, v in md.items():
                if k not in _LLM_METADATA_WHITELIST:
                    continue
                if v in (None, "", "nan", "None"):
                    continue
                # Per-license dedup: skip long fields already sent for this license_type
                if k in _LONG_FIELDS_DEDUP and _long_fields_sent_by_lt.get(_lt2):
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
                _long_fields_sent_by_lt[_lt2] = True
            docs_json.append(
                {
                    "metadata": filtered_md,
                    "content": (d.get("content", "") or ""),
                }
            )

        # Build labeled link sections — LLM copies these directly, no URL pattern matching needed
        def _fmt_prac_link(desc: str, url: str) -> str:
            if desc and url:
                return f"- {desc}\n  {url}"
            return f"- {url or desc}"

        # Links: inject ตาม intent ของ user เท่านั้น ไม่ inject ทุก section ตลอดเวลา
        # When user fills a slot (e.g. "1"), user_text is short — fall back to last_user_legal_query
        # so that link-intent from the original question is preserved through slot collection.
        _intent_text = (user_text or "") + " " + str((state.context or {}).get("last_user_legal_query") or "")
        _user_wants_links = bool(re.search(
            r"(ขอลิงค์|ขอลิงก์|ส่งลิงค์|ส่งลิงก์|ลิงค์คู่มือ|ลิงก์คู่มือ"
            r"|ขอดูลิงค์|ขอดูลิงก์|URL|ดาวน์โหลด"
            r"|ขอคู่มือ|ขอดูคู่มือ|ส่งคู่มือ|คู่มือ(การ|สำหรับ|ของ)"
            r"|ขอแบบฟอร์ม|ส่งแบบฟอร์ม)",
            _intent_text, re.IGNORECASE,
        ))
        # Service/registration links: เฉพาะตอน user ถามเรื่องการสมัคร/ลงทะเบียน
        _user_wants_registration = bool(re.search(
            r"(สมัคร|ลงทะเบียน|ยื่นขอ|จดทะเบียน|ขอใบ|อยากจด|ต้องการจด"
            r"|ขั้นตอน(การ|ใน)|วิธี(จด|สมัคร|ยื่น)|ต้องทำยังไง"
            r"|ลิ้งค์.{0,6}(สมัคร|ลงทะเบียน|กรอก|ไฟล์|API|form)"
            r"|ลิงค์.{0,6}(สมัคร|ลงทะเบียน|กรอก|ไฟล์|API|form)"
            r"|link.{0,6}(register|apply|form|sign.?up))",
            _intent_text, re.IGNORECASE,
        ))
        # Form links: เฉพาะตอน user ถามเรื่องเอกสาร/แบบฟอร์ม
        _user_wants_forms = bool(re.search(
            r"(แบบฟอร์ม|เอกสาร.{0,8}(ใช้|ที่ต้อง|ต้องใช้)|ต้องใช้.{0,8}เอกสาร"
            r"|ลิ้งค์.{0,6}(เอกสาร|ฟอร์ม|คำขอ)"
            r"|link.{0,6}(document|form|template))",
            _intent_text, re.IGNORECASE,
        ))

        _link_section = ""
        if _link_service and (_user_wants_registration or _docs_have_id_docs):
            _link_section += "\n🌐 SERVICE_LINKS — copy เหล่านี้ตรงๆ under section '🌐 เว็บลงทะเบียน':\n"
            _link_section += "\n".join(_fmt_prac_link(d, u) for d, u in _link_service) + "\n"
        if _link_form and (_user_wants_forms or _user_wants_registration or _docs_have_id_docs):
            _link_section += "\n📄 FORM_LINKS — copy เหล่านี้ตรงๆ under section '📄 แบบฟอร์ม':\n"
            _link_section += "\n".join(_fmt_prac_link(d, u) for d, u in _link_form) + "\n"
        if _link_guide and _user_wants_links:
            _link_section += "\n📖 GUIDE_LINKS — user ขอคู่มือ: copy เหล่านี้ตรงๆ under section '📖 คู่มือ':\n"
            _link_section += "\n".join(_fmt_prac_link(d, u) for d, u in _link_guide) + "\n"

        self._debug_log("pre_llm", query=user_text, docs_json=docs_json)

        # Token: ตัด context ให้เล็กลง — เก็บเฉพาะ keys ที่ LLM ต้องการจริงๆ
        _ctx_keys_needed = {"topic", "slots", "pending_slot", "last_user_legal_query",
                            "last_topic", "topic_slot_queue", "topic_operation_groups",
                            "collected_slots", "multi_license_topics"}
        slim_context = {k: v for k, v in (state.context or {}).items()
                        if k in _ctx_keys_needed and v not in (None, {}, [], "")}

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
        elif _active_topic:
            # Topic known from context but not yet confirmed via operation_group —
            # LLM may still ask clarifying questions, but should bias toward this topic
            _topic_hint = (
                f"\n\n💡 CONTEXT HINT: User is currently discussing \"{_active_topic}\"."
            )
            if _active_op:
                _topic_hint += f" Operation in focus: \"{_active_op}\"."

        # Build special instruction when user asked about multiple license types at once
        _multi_license_topics = (state.context or {}).get("multi_license_topics") or []
        _multi_license_instruction = ""
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
- If a topic has no DOCUMENTS, write "ยังไม่พบข้อมูลในเอกสาร แนะนำติดต่อหน่วยงานที่เกี่ยวข้องโดยตรงครับ"
- Do NOT stop after writing just one section header
- Close with a 1-sentence summary: which items are MANDATORY vs optional
"""

        # Broad-question instruction: injected directly into this call's prompt (not system prompt)
        # so it only affects broad open-ended questions, not other query types.
        # Tells LLM to treat legal/license section as mandatory named section, not a footnote.
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

        prompt = f"""USER INPUT:
{user_input}

LAST ASSISTANT MESSAGE:
{last_bot[:300] if last_bot else ""}

RECENT MESSAGES:
{json.dumps(recent_msgs, ensure_ascii=False)}

CONTEXT:
{json.dumps(slim_context, ensure_ascii=False)}

DOCUMENTS ({len(docs_json)} found):
{json.dumps(docs_json, ensure_ascii=False)}
{_link_section}
ROUND: {int(getattr(state, "round", 0) or 0)}/{int(getattr(conf, "MAX_ROUNDS", 7) or 7)}{_topic_hint}{_multi_license_instruction}{_broad_instruction}

Your JSON response:
"""

        # SHORT-CIRCUIT: when topic_slot_queue is non-empty and docs are loaded,
        # the next action is always 'ask' — skip the LLM entirely.
        # The question text and slot options come from the queue entry (not the LLM),
        # so calling the LLM here wastes tokens without changing the outcome.
        _pending_sq = (state.context or {}).get("topic_slot_queue") or []
        if _pending_sq and state.current_docs:
            _LOG.info("[Practical] slot_queue non-empty (%d) — skipping LLM, action=ask", len(_pending_sq))
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
                    _LOG.warning(
                        "[Practical] retrieve blocked %d times — breaking loop, forcing non-internal answer pass",
                        _rblk,
                    )
                    state.context["_retrieve_blocked_count"] = 0
                    return self.handle(state, user_input or "__auto_post_retrieve__", _internal=False)
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
                    state.context["pending_slot"] = {
                        "key": slot_key,
                        "options": list(slot_opts),
                        "allow_multi": False,
                    }
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
                    allow_multi = True if slot_key == self._PHASE3_SLOT_KEY else False

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
                        menu = self._format_numbered_options(options)
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
                    for _full in _all_known_urls:
                        if _full.startswith(_found) and _full != _found:
                            return _full
                    return _found
                ans = re.sub(r'https?://\S+', _restore_url_match, ans)

            ans = self._apply_practical_lint(ans, kind="answer")

            self._append_assistant(state, ans)
            state.context["phase"] = None
            state.round = 0
            # LLM answered directly — any remaining slot queue is stale, clear it
            state.context.pop("topic_slot_queue", None)
            # Clear multi-license signal after it has been consumed in the answer
            state.context.pop("multi_license_topics", None)
            return state, ans

        fallback = "ผมยังไม่เข้าใจครับ บอกหัวข้อที่อยากรู้เกี่ยวกับร้านอาหารหน่อยครับ"
        fallback = self._apply_practical_lint(fallback, kind="ask")
        self._append_assistant(state, fallback)
        return state, fallback
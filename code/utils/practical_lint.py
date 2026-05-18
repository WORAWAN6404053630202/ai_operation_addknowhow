# code/utils/practical_lint.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# Configuration

@dataclass(frozen=True)
class PracticalPolicyConfig:
    # Hard bans (content-level) — applies to NON-QUOTE lines only (see quote rules below)
    forbidden_phrases: Tuple[str, ...] = (
        # อ้อม/ขออนุญาต
        "เพื่อความถูกต้อง",
        "ขออนุญาต",
        "รบกวน",
        "ขอข้อมูลเพิ่ม",
        "ช่วยระบุให้ชัดเจน",
        "ต้องการเปลี่ยนเป็น",

        # system/meta talk ที่ practical ไม่ควรหลุด
        "จากเอกสาร",          # practical ไม่ควรขึ้นต้น/เล่าเมตา (ยังยก quote ได้)
        "จากข้อมูลในเอกสาร",
        "ในระบบของฉัน",
        "ในระบบของเรา",
        "ระบบของฉัน",
        "ระบบของเรา",
        "ฉันจะ",
        "ผมจะ",
        "ดิฉันจะ",
        "ขออธิบายว่า",
        "โมเดล",
        "LLM",
        "retrieve",
        "retriever",
        "vector",
        "chroma",
        "embedding",

        # NEW: additional meta-talk patterns LLMs commonly produce
        "ขอบอกด้วยว่า",
        "ตามที่ท่านขอ",
        "ตามที่คุณขอ",
        "กล่าวคือ",
        "ในฐานะ AI",
        "ในฐานะผู้ช่วย",
        "ในฐานะ assistant",
        "นโยบาย",
        "assistant",
        "documents",    # LLM บางตัว reproduce prompt section header ออกมา
        "Documents",
        "DOCUMENTS",

        # hedging / uncertainty phrases — bot should answer directly, never hedge
        "เท่าที่รู้",
        "เท่าที่ทราบ",
        "เท่าที่ผมรู้",
        "เท่าที่ผมทราบ",
        "ตามที่ผมทราบ",
        "ตามที่ผมรู้",
        "ตามที่รู้",
        "ข้อมูลที่ผมมี",
        "ข้อมูลที่มีอยู่",
        "ในข้อมูลที่มี",
        "จากข้อมูลที่มี",
        "ตามข้อมูลที่มี",
        "ตามที่มีอยู่",
    )

    # “อ้อม/เมตา” patterns (behavior-level) — applies to NON-QUOTE lines only
    forbidden_preface_patterns: Tuple[re.Pattern, ...] = field(default_factory=lambda: (
        # เกริ่นก่อนถามแบบอ้อม
        re.compile(r"^\s*(เพื่อ|ก่อน|ขอ|รบกวน)\b", re.IGNORECASE),
        re.compile(r"^\s*เพื่อความถูกต้อง", re.IGNORECASE),
        re.compile(r"^\s*ขออนุญาต", re.IGNORECASE),
        re.compile(r"^\s*รบกวน", re.IGNORECASE),
        re.compile(r"^\s*ขอข้อมูลเพิ่ม", re.IGNORECASE),

        # system/meta talk preface
        re.compile(r"^\s*(จากเอกสาร|จากข้อมูลในเอกสาร)\b", re.IGNORECASE),
        re.compile(r"^\s*(ในระบบของฉัน|ในระบบของเรา|ระบบของฉัน|ระบบของเรา)\b", re.IGNORECASE),
        re.compile(r"^\s*(ฉันจะ|ผมจะ|ดิฉันจะ)\b", re.IGNORECASE),
    ))

    # Structural constraints (still counts ALL lines including quotes by default)
    max_lines: int = 6
    max_bullets: int = 5
    max_chars: int = 650  # practical answer ควรสั้นอยู่แล้ว

    # Ask constraints
    require_single_question: bool = True
    question_max_chars: int = 160

    # Rewrite attempts (expensive path)
    max_rewrite_attempts: int = 2

    # If still invalid after rewrite, fallback mode:
    # - "minimal_question": return a single safe question
    # - "trim": aggressively trim content
    fallback_mode: str = "minimal_question"

    # Minimal safe question if we must fallback (domain-safe + single question)
    fallback_question: str = "ต้องการทำเรื่องอะไรเป็นหลักครับ (เช่น ใบอนุญาต/ภาษี/VAT/ประกันสังคม)?"

    # Should we enforce Thai-only? (optional, weak heuristic)
    enforce_thai_only: bool = False

    # Quote handling:
    # If True: lines starting with ">" are treated as verbatim document quotes.
    # Policy bans (forbidden phrases/preface/multi-question/english) will NOT be checked inside quote lines.
    allow_forbidden_inside_quotes: bool = True


DEFAULT_POLICY = PracticalPolicyConfig()


# Quote helpers

def _is_quote_line(line: str) -> bool:
    """
    Treat lines starting with '>' (after trimming left spaces) as verbatim quotes.
    """
    return (line or "").lstrip().startswith(">")


def _non_quote_text(text: str) -> str:
    """
    Remove quote lines from text. Used for policy scans that should ignore quotes.
    """
    lines = (text or "").split("\n")
    kept = [ln for ln in lines if ln.strip() and not _is_quote_line(ln)]
    return "\n".join(kept).strip()


# Issue detection

def _normalize(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\r\n", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def _split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").split("\n") if ln.strip()]


def _count_bullets(lines: List[str]) -> int:
    return sum(1 for ln in lines if ln.startswith(("-", "•", "*")))


# Multi-question heuristics 
_Q_ENDING_RE = re.compile(r"(ไหม|หรือไม่|หรือเปล่า)\s*$", re.IGNORECASE)
_Q_MARK_RE = re.compile(r"\?", re.IGNORECASE)
_Q_PREFIX_RE = re.compile(r"^\s*(ถาม|คำถาม)\s*[:：]", re.IGNORECASE)

# A) Question line that contains conjunctions tends to mean "A and B?" => multi-question risk
# FIX: tightened pattern — require a question marker on the SAME line to avoid false positives
# on compound nouns like "ร้านอาหารและบาร์" or phrases with "กับ" that are not questions.
# Removed "หรือ" (too broad as a question-forming particle) and "กับ" (too common in compound nouns).
# NOTE: \b word-boundary is NOT used for "และ" because Thai letters are all \w in Python 3 Unicode,
# so \b never fires between consecutive Thai characters (e.g. "ค่าธรรมเนียมและเอกสารไหม" has no
# word boundary around "และ"). Bare "และ" is safe here because the pattern already requires
# \s+ after the conjunction AND a question marker at the end.
_MULTI_Q_CONJ_RE = re.compile(
    r"(?:และ|พร้อมกับ|รวมถึง|ตลอดจน)\s+.+(?:ไหม|หรือไม่|หรือเปล่า|\?)",
    re.IGNORECASE,
)

# B) Explicit "ช่วยบอก A และ B ไหม" / "บอก A กับ B ได้ไหม" patterns
_MULTI_Q_HELP_RE = re.compile(
    r"(ช่วย|รบกวน|บอก|แจ้ง|ยืนยัน|ระบุ).*(และ|หรือ|พร้อมกับ|รวมถึง|กับ).*(ไหม|ได้ไหม|หรือไม่|หรือเปล่า|\?)",
    re.IGNORECASE,
)


def _is_questionish_line(ln: str) -> bool:
    s = (ln or "").strip()
    if not s:
        return False
    if _Q_MARK_RE.search(s):
        return True
    if _Q_ENDING_RE.search(s):
        return True
    if _Q_PREFIX_RE.search(s):
        return True
    return False


def _count_questions(text: str) -> int:
    """
    Heuristic question counter for Thai.

    Base signals:
    - count '?'
    - count line endings with "ไหม/หรือไม่/หรือเปล่า"
    - count explicit question markers like "ถาม:"

    NEW:
    - If a question line contains conjunctions that imply multiple asks (และ/หรือ/พร้อมกับ/รวมถึง/กับ),
      treat it as multi-question by adding an extra count (+1).
    - If matches stronger help-pattern ("ช่วยบอก A และ B ไหม"), count as 2.
    """
    t = (text or "")
    lines = _split_lines(t)

    q = 0

    # Base counting at line-level (more stable than raw '?' count)
    for ln in lines:
        if not _is_questionish_line(ln):
            continue

        # Strong multi-question pattern => count as 2
        if _MULTI_Q_HELP_RE.search(ln):
            q += 2
            continue

        # If contains '?' or ending markers => at least 1
        q += 1

        # Conjunction inside a question line => likely multiple asks
        # Example: "ช่วยบอก A และ B ไหม" / "ต้องทำ A หรือ B ไหม"
        if _MULTI_Q_CONJ_RE.search(ln) and (("?" in ln) or bool(_Q_ENDING_RE.search(ln))):
            q += 1

    return q


def _contains_forbidden_phrase(text: str, cfg: PracticalPolicyConfig) -> Optional[str]:
    for p in cfg.forbidden_phrases:
        if p and (p in text):
            return p
    return None


def _contains_forbidden_preface(text: str, cfg: PracticalPolicyConfig) -> bool:
    for pat in cfg.forbidden_preface_patterns:
        if pat.search(text):
            return True
    return False


def _contains_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def analyze_practical_text(text: str, cfg: PracticalPolicyConfig = DEFAULT_POLICY) -> Dict[str, object]:
    """
    Return structured issues for policy enforcement.

    Quote rule:
    - If cfg.allow_forbidden_inside_quotes=True:
      We DO NOT scan quote lines (starting with '>') for:
        - forbidden phrases
        - forbidden preface
        - multi-question
        - english (optional)
      Structural limits (max_lines/max_bullets/max_chars) still apply to the full text by default.
    """
    t = _normalize(text)
    lines = _split_lines(t)

    issues: Dict[str, object] = {"ok": True, "reasons": []}

    scan_text = t
    if cfg.allow_forbidden_inside_quotes:
        scan_text = _non_quote_text(t)

    # Hard phrase bans (scan non-quote only if enabled)
    bad = _contains_forbidden_phrase(scan_text, cfg)
    if bad:
        issues["ok"] = False
        issues["reasons"].append({"type": "forbidden_phrase", "value": bad})

    # Forbidden preface (scan non-quote only if enabled)
    if _contains_forbidden_preface(scan_text, cfg):
        issues["ok"] = False
        issues["reasons"].append({"type": "forbidden_preface"})

    # Length constraints (still count ALL lines including quotes)
    if len(lines) > cfg.max_lines:
        issues["ok"] = False
        issues["reasons"].append({"type": "too_many_lines", "value": len(lines)})

    bullets = _count_bullets(lines)
    if bullets > cfg.max_bullets:
        issues["ok"] = False
        issues["reasons"].append({"type": "too_many_bullets", "value": bullets})

    if len(t) > cfg.max_chars:
        issues["ok"] = False
        issues["reasons"].append({"type": "too_long_chars", "value": len(t)})

    # Single question policy (scan non-quote only if enabled)
    if cfg.require_single_question:
        qcount = _count_questions(scan_text)
        if qcount > 1:
            issues["ok"] = False
            issues["reasons"].append({"type": "multi_question", "value": qcount})

    # Thai-only optional check (scan non-quote only if enabled)
    if cfg.enforce_thai_only and _contains_english(scan_text):
        issues["ok"] = False
        issues["reasons"].append({"type": "contains_english"})

    return issues


# Deterministic fallback helpers
def _trim_to_policy(text: str, cfg: PracticalPolicyConfig) -> str:
    """
    Deterministic trimming to comply with max_lines/max_bullets/max_chars.
    Does NOT guarantee removing forbidden phrases (handled elsewhere).
    NOTE: Trimming applies to full text, including quote lines.
    """
    t = _normalize(text)
    lines = _split_lines(t)

    # enforce line limit
    lines = lines[: cfg.max_lines]

    # enforce bullet limit
    out: List[str] = []
    bullet_count = 0
    for ln in lines:
        if ln.startswith(("-", "•", "*")):
            bullet_count += 1
            if bullet_count > cfg.max_bullets:
                continue
        out.append(ln)

    t2 = "\n".join(out).strip()

    # enforce max chars
    if len(t2) > cfg.max_chars:
        t2 = t2[: cfg.max_chars].rstrip()

    return t2.strip()


def _minimal_safe_question(cfg: PracticalPolicyConfig) -> str:
    q = (cfg.fallback_question or "ต้องการทราบรายละเอียดเพิ่ม 1 ข้อครับ?").strip()
    if len(q) > cfg.question_max_chars:
        q = q[: cfg.question_max_chars].rstrip()
    if not (q.endswith("?") or q.endswith("ไหม") or q.endswith("หรือไม่") or q.endswith("หรือเปล่า")):
        q = q.rstrip(".") + "?"
    return q


def _hard_remove_forbidden_lines(text: str, cfg: PracticalPolicyConfig) -> str:
    """
    Stronger-than-trim: drop any NON-QUOTE line containing forbidden phrase/preface.
    Quote lines ('> ...') are preserved (verbatim docs).
    """
    t = _normalize(text)
    lines = _split_lines(t)
    kept: List[str] = []
    for ln in lines:
        if cfg.allow_forbidden_inside_quotes and _is_quote_line(ln):
            kept.append(ln)
            continue
        if _contains_forbidden_phrase(ln, cfg):
            continue
        if _contains_forbidden_preface(ln, cfg):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


# Rewrite prompt (embedded here to keep this file standalone)

def build_rewrite_prompt(text: str, cfg: PracticalPolicyConfig) -> str:
    banned = ", ".join(cfg.forbidden_phrases)
    quote_note = ""
    if cfg.allow_forbidden_inside_quotes:
        quote_note = """
- ถ้าต้องยกข้อความจากเอกสารแบบคำต่อคำ ให้ขึ้นบรรทัดใหม่ด้วย ">" และคงข้อความเดิม (verbatim)
- ข้อความบรรทัดที่ขึ้นต้นด้วย ">" ถือเป็น quote เอกสาร อาจมีคำสุภาพทางราชการได้
""".strip()

    return f"""
คุณทำหน้าที่ rewrite ข้อความให้เป็น persona “practical” สำหรับผู้ช่วยไทยสายกฎระเบียบร้านอาหาร

กติกา (ต้องทำตามทั้งหมด):
- ภาษาไทย 100%
- สุภาพแบบตรง ไม่อ้อม ไม่เล่าระบบ/กระบวนการของตัวเอง
- ห้ามมีคำ/วลีต่อไปนี้ (ยกเว้นอยู่ในบรรทัด quote ที่ขึ้นต้นด้วย ">"): {banned}
- ห้ามเกริ่นนำเชิงเมตา เช่น “จากเอกสาร.../ในระบบ.../ฉันจะ...” (ยกเว้นอยู่ในบรรทัด quote)
- ถ้าต้องถาม ให้มี “คำถามเดียว” สั้น ๆ เท่านั้น
- ห้ามทำคำถามซ้อนด้วย “และ/หรือ/พร้อมกับ/รวมถึง/กับ” ในประโยคคำถามเดียว
- ความยาว: ไม่เกิน {cfg.max_lines} บรรทัด หรือ {cfg.max_bullets} bullet
- คงสาระเดิมให้มากที่สุด แต่ทำให้กระชับและ actionable
- คืนค่าเป็น “ข้อความล้วน” เท่านั้น (ไม่ใช่ JSON, ไม่ใช่ markdown)
{quote_note}

ข้อความเดิม:
{text}

ข้อความใหม่:
""".strip()


# Main enforcement API

RewriteFn = Callable[[str], str]


def enforce_practical_policy(
    text: str,
    cfg: PracticalPolicyConfig = DEFAULT_POLICY,
    rewrite_fn: Optional[RewriteFn] = None,
) -> Tuple[str, Dict[str, object]]:
    """
    Enforce practical policy with best-effort guarantee.

    Flow:
    1) Analyze
    2) If ok -> trim-to-policy (light) -> verify
    3) If not ok and rewrite_fn provided -> rewrite attempts -> verify each
    4) If still not ok -> deterministic fallback (minimal question or hard trim)
    5) Return (final_text, meta)

    Quote rule:
    - If cfg.allow_forbidden_inside_quotes=True, quote lines starting with ">" are excluded from
      forbidden phrase/preface/multi-question/english checks.
    """
    original = _normalize(text)
    meta: Dict[str, object] = {
        "ok": True,
        "rewritten": False,
        "attempts": 0,
        "final_mode": "pass",
        "issues": {},
    }

    # Step 0: trivial empty
    if not original:
        out = _minimal_safe_question(cfg)
        meta.update({"ok": True, "final_mode": "fallback_empty"})
        return out, meta

    # Step 1: analyze
    issues1 = analyze_practical_text(original, cfg)
    meta["issues"] = issues1
    if issues1.get("ok") is True:
        out0 = _trim_to_policy(original, cfg)
        issues0 = analyze_practical_text(out0, cfg)
        if issues0.get("ok") is True:
            meta.update({"ok": True, "final_mode": "trim_ok"})
            return out0, meta

    # Step 2: rewrite with LLM (expensive path)
    if rewrite_fn is not None:
        cur = original
        for i in range(cfg.max_rewrite_attempts):
            meta["attempts"] = i + 1
            prompt = build_rewrite_prompt(cur, cfg)
            rewritten = _normalize(rewrite_fn(prompt))
            if rewritten:
                meta["rewritten"] = True
                rewritten = _trim_to_policy(rewritten, cfg)
                issues_r = analyze_practical_text(rewritten, cfg)
                if issues_r.get("ok") is True:
                    meta.update({"ok": True, "final_mode": "rewrite_ok", "issues": issues_r})
                    return rewritten, meta
                cur = rewritten

    # Step 3: deterministic strong fallback (guarantee)
    if cfg.fallback_mode == "trim":
        out = _trim_to_policy(original, cfg)
        out = _hard_remove_forbidden_lines(out, cfg)
        out = _trim_to_policy(out, cfg)
        if analyze_practical_text(out, cfg).get("ok") is True and out:
            meta.update({"ok": True, "final_mode": "fallback_trim"})
            return out, meta

    # Default fallback: minimal safe question
    out = _minimal_safe_question(cfg)
    meta.update({"ok": True, "final_mode": "fallback_minimal_question"})
    return out, meta

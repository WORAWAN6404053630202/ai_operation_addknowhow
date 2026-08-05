# code/utils/text_patterns.py
"""
Shared text-classification regex patterns used by more than one persona
(persona_supervisor.py, persona_practical.py, persona_academic.py).

Why this module exists: each persona used to define its own copy of these
patterns independently. Several copies silently diverged over time (one file
got a real-world fix or extension, the others didn't) — e.g. `_LEGAL_SIGNAL_RE`
had three different sets of trigger words across the three files, which was
the direct root cause of a topic-drift bug (persona_practical.py's narrower
copy missed a case persona_supervisor.py's copy would have caught). Giving
each shared concept exactly one place to live prevents that class of bug by
construction — there is nothing left to diverge.

See tests/test_classifier_consistency.py for regression cases pinned from the
bugs found during the 2026-07-31 audit that led to this module, plus a
generic check that catches this exact kind of drift automatically going
forward.
"""
from __future__ import annotations

import re

# ── Info-gap tag (LLM-emitted marker that triggers the feedback menu) ───────
# Shared by persona_academic.py, persona_practical.py, and persona_supervisor.py
# so the two personas can't drift onto different tags while the detection
# point in the supervisor stays a single check. Both personas' prompts
# (prompts_academic.py / prompts_practical.py) instruct the LLM to write this
# exact literal — and nothing else — in place of a section's content whenever
# that question/topic isn't covered by DOCUMENTS at all. IMPORTANT: the tag
# string is duplicated as a literal in those two prompt files (prompts are
# static strings, not built from this module) — changing INFO_GAP_TAG here
# requires updating both prompt files to match.
#
# Replaces the previous approach of scanning the visible answer text for one
# of a handful of hardcoded Thai phrases (e.g. "ไม่พบข้อมูล"), which had two
# failure modes: false positives (the phrase appears in unrelated content)
# and false negatives (the LLM phrases the gap differently and the scan
# misses it). A dedicated tag is a signal, not content: extract_info_gap_tag
# swaps it for INFO_GAP_DISPLAY_TEXT so the user still sees a normal sentence
# (including mid-answer, e.g. one section of a multi-topic reply), while
# detection upstream is an exact-string match instead of a keyword scan.
#
# NOTE: Python-level fallback strings (e.g. an empty `answer` field, or a
# retry-loop giving up) are NOT LLM output and can never carry this tag by
# construction — those call sites set the info-gap flag directly in code
# instead of relying on this tag.
INFO_GAP_TAG = "[[INFO_GAP]]"
INFO_GAP_DISPLAY_TEXT = "ยังไม่พบข้อมูลในเอกสาร"


def extract_info_gap_tag(text: str) -> tuple[str, bool]:
    """Replace any INFO_GAP_TAG occurrence(s) with INFO_GAP_DISPLAY_TEXT. Returns (clean_text, tag_found).

    Call this immediately after receiving raw LLM output, before any other
    cleanup/post-processing runs on that text — later regex-based cleanup
    (URL stripping, meta-talk removal, line-wrap fixing) wasn't written with
    this tag in mind and could otherwise mangle or swallow it.
    """
    t = text or ""
    if INFO_GAP_TAG in t:
        return t.replace(INFO_GAP_TAG, INFO_GAP_DISPLAY_TEXT).strip(), True
    return t, False


# ── Greeting fragments (byte-identical across the files that had them) ──────
TH_DEE_RE = re.compile(r"^\s*ดี(?:ครับ|คับ|ค่ะ|คะ|งับ|จ้า|จ้ะ|ค่า)?", re.IGNORECASE)
EN_GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|yo+)\b", re.IGNORECASE)
EN_GOOD_TIME_RE = re.compile(r"^\s*good\s+(morning|afternoon|evening|night)\b", re.IGNORECASE)

# ── Tokenizer helper (byte-identical between supervisor/practical) ──────────
TOKEN_SPLIT_RE = re.compile(r"[\s/,\-–—|]+", re.UNICODE)

# ── Reconciled duplicates (previously diverged — see module docstring) ──────
# "หวัดดี"-family greeting. Reconciled to practical's typo-tolerant version
# (matches "หวัด" + up to 6 trailing chars) rather than supervisor's/academic's
# exact-match-only version — a missed greeting here just falls through to the
# existing LLM greeting check elsewhere, so the wider net has no real downside.
TH_WATDEE_RE = re.compile(r"^\s*หวัด[^\s]{0,6}", re.IGNORECASE)
# Kept separate: supervisor/academic use the exact-match variant too under this
# same name in some places; TH_SAWASDEE_FUZZY_RE below is the "สวัสดี" family
# (different prefix) and was already byte-identical — no reconciliation needed.
TH_SAWASDEE_FUZZY_RE = re.compile(r"^\s*สว[^\s]{0,6}ดี", re.IGNORECASE)

# Thanks detection. Reconciled to practical's broader version (adds
# ขอบพระคุณ/ขอบคุณมาก/ขอบคุณนะ/"thank you") — supervisor's copy was missing
# these; a broader net here has no real downside (this only gates a polite
# closing response, not a data-affecting decision).
THANKS_RE = re.compile(
    r"(ขอบคุณ|ขอบใจ|ขอบพระคุณ|ขอบคุณมาก|ขอบคุณนะ|thx|thanks|thank you)",
    re.IGNORECASE,
)

# "Looks like a menu selection" (e.g. "1", "2, 3", "1-3"). Reconciled to
# supervisor's stricter version, which requires at least one digit via a
# lookahead. Practical's copy was missing that lookahead, so an
# empty/separator-only string (e.g. just "-" or a few spaces) would have
# matched as "a selection" — that looks like an unintentional bug in
# practical's copy, not a deliberate looser choice, so the stricter version
# is adopted as canonical rather than unioning the two.
LIKELY_SELECTION_RE = re.compile(r"^\s*(?=.*\d)[\d\s,/-]+\s*$")

# "Does this text look like a legal/regulatory/business question at all?"
# Reconciled to supervisor's version, which turned out — on reading the FULL
# multi-line pattern rather than a truncated grep preview — to already be a
# strict superset of both practical's and academic's term lists (it includes
# their "งบการเงิน/ผลกระทบ/ความเสี่ยง" tail plus QR Payment/bank/
# social-security-contribution/alcohol/etc. terms neither of the other two
# had). So "reconcile" here means "adopt supervisor's as canonical" rather
# than constructing a new union. This is the gate behind
# `_looks_like_legal_question` — a false negative here (missing a real legal
# question) is worse than a false positive (downstream code has further, more
# specific checks anyway), so the broader existing version is the right
# canonical choice. This was the exact classifier behind the 2026-07-31
# liquor-license topic-drift bug (see
# persona_practical_retrieve_retry_loop_bug.md in project memory) — practical's
# and academic's narrower copies missed cases supervisor's copy would have
# caught.
LEGAL_SIGNAL_RE = re.compile(
    r"(ใบอนุญาต|จดทะเบียน|ทะเบียนพาณิชย์|ภาษี|vat|ภพ\.?20|สรรพากร|เทศบาล|สำนักงานเขต|สุขาภิบาล|กรม|ค่าธรรมเนียม|เอกสาร|ขั้นตอน|บทลงโทษ|ประกาศ|พ\.ร\.บ|ประกันสังคม|กองทุน|เปิดร้าน|ขึ้นทะเบียน"
    r"|เงินสมทบ|สมทบ|แยกยื่น|ยื่นรวม|ส่งเงินสมทบ|ส่งข้อมูลเงินสมทบ"
    r"|qr\b|qr.?pay|qr.?payment|คิวอาร์|คิวอาเพย|เพย์เมนต์|edc|รูดบัตร|merchant.?id|partner.?id|pos\b|pos.?id|ระบบชำระเงิน|กสิกร|kbank|ไทยพาณิช(?:ย์)?|ไทยพานิช|scb|ประกอบกิจการ|สุขาภิบาลอาหาร"
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
    # Reference/source queries — user asking where info comes from
    r"|อ้างอิง|แหล่งที่มา|แหล่งข้อมูล|แหล่งอ้างอิง|ที่มาของข้อมูล"
    # Alcohol/tobacco/food restriction domain (common restaurant regulatory topics)
    r"|สุรา|เหล้า|แอลกอฮอล์|บุหรี่|ห้ามขาย|จำหน่ายสุรา|ขายสุรา|ขายเหล้า|วันหยุดนักขัตฤกษ์"
    # Marketing/inventory-management acronyms with real operation_topic content in the
    # corpus (data_type='marketing'/'business_guide') — same bare-acronym treatment as
    # pos/qr/edc/vat/san above, confirmed low false-positive risk (none of these are
    # common English words). Deliberately excludes generic English tokens found in the
    # same corpus scan (e.g. "of", "to", "New", "Stock", "Menu") — those are too common
    # to safely gate on without misrouting unrelated English text into the legal-question
    # pipeline; NTT is also excluded — it's department metadata inside the already-covered
    # EDC docs, not a topic a user would type on its own.
    r"|\bfifo\b|\blifo\b|\bfefo\b|\brop\b|\bsop\b|\bkol\b|\bimc\b|\bqsc\b|\bbcg\b"
    # 2026-08-04: closes the 4 WARN findings from code/scripts/license_coverage_check.py
    # --static (run against the live corpus) — these 4 license_type bare names matched
    # neither this regex nor _SUPERVISOR_KW_OVERRIDE, so recognizing them as informational
    # relied entirely on the non-deterministic LLM classifier fallback (_info_action_q_llm_
    # check / _legal_q_llm_check) rather than a guaranteed regex match, the same class of
    # risk that caused the "สุขาภิบาล" bug earlier. None of these 4 were ever observed to
    # actually misroute live — this closes the gap pre-emptively rather than reactively.
    # Checked for false-positive collision against casual Thai phrases before adding.
    r"|จัดการการเงิน|จัดการเงิน|นายจ้าง|ใบรับรองแพทย์|สณ\.?11|ผู้สัมผัสอาหาร)",
    re.IGNORECASE,
)

# ── Phase 3 (2026-07-31): entity-type detection ─────────────────────────────
# "Does this text explicitly and unambiguously state the user's business
# entity type?" These patterns feed call sites that write the result straight
# to collected_slots (or overwrite it on "switch" detection) with NO LLM
# double-check — and this project's "no slot re-ask" rule means a false
# positive here is a SILENT, PERMANENT wrong answer for the rest of the
# session (confirmed live, see qr_payment_bank_account_name_change_gap.md's
# follow-up finding / entity_false_positive_regex_tightening.md). Only terms
# with near-zero false-positive rate in ordinary Thai belong here.
#
# 2026-07-31 tightening (live-verified false positives, terms REMOVED from an
# earlier broader version): bare "คนเดียว" ("อยู่คนเดียว" = live alone,
# "มาคนเดียว" = came alone — nothing to do with sole proprietorship), "ทำเอง"/
# "เปิดเอง" (live-reproduced: "ค่าธรรมเนียมทำเองกับจ้างตัวแทน...ต่างกันไหม" —
# asking about DIY-filing vs. hiring an agent, NOT declaring entity type —
# silently set entity_type='บุคคลธรรมดา' with zero user confirmation),
# "ร้านค้าทั่วไป"/"ร้านส่วนตัว" (too generic), and bare "บริษัท"/"ห้าง" without
# their legal suffix (live-reproduced: "ร้านอยู่ในห้างครับ" — a shop merely
# *located in a shopping mall* — matched as "นิติบุคคล"; บริษัท/ห้าง are
# everyday words for "company"/"mall" outside the legal-entity sense, so the
# suffix — จำกัด/มหาชน, หุ้นส่วน — is now REQUIRED for a match).
# "เจ้าของคนเดียว"/"กิจการเจ้าของคนเดียว" (sole owner / sole proprietorship)
# are kept — the "เจ้าของ"/"กิจการ" prefix makes these deliberate, specific
# declarations, unlike bare "คนเดียว".
#
# The removed colloquial terms are NOT relocated to an LLM-fallback path in
# this pass — most (คนเดียว/ส่วนตัว/เจ้าของ) are still covered by
# ENTITY_HINT_RE below, so existing hint-gated LLM-fallback call sites
# (persona_supervisor.py's _apply_slot_change_if_detected /
# _prefill_slots_from_message) degrade gracefully to that path automatically,
# with no code change needed there. "เอง" is deliberately NOT added to
# ENTITY_HINT_RE — it is too generic on its own (confirmed above) to be a
# reliable signal even for an LLM guess. A structured, confidence-tiered
# LLM-assisted auto-fill + implicit confirmation is a deliberately separate,
# not-yet-decided follow-up (see entity_false_positive_regex_tightening.md).
ENTITY_NITI_RE = re.compile(
    r"นิติบุคคล|(?:แบบ|เป็น(?:แบบ)?|ประเภท|รูปแบบ|ถ้า(?:เป็น)?|ถ้าแบบ)\s*นิติ"
    r"|บริษัท\s*(?:จำกัด|มหาชน)|ห้างหุ้นส่วน(?:\s*จำกัด|\s*สามัญ)?",
    re.IGNORECASE,
)
ENTITY_NATURAL_RE = re.compile(
    r"บุคคลธรรมดา|บุคคล.{0,4}มดา|บุคคลทั่วไป|เจ้าของ\s*คนเดียว|กิจการเจ้าของ\s*คนเดียว"
    r"|(?:ถ้า(?:เป็น)?|ถ้าแบบ)\s*(?:บุคคล)?ธรรมดา",
    re.IGNORECASE,
)
# Guard: query must contain at least one entity-hint word before calling an LLM
# fallback for entity-type. Prevents false positives from pronouns (ฉัน/ผม/หนู)
# or entity-neutral questions. Catches abbreviations (บจก, หจก) too.
#
# Part 2 (2026-07-31, confidence-tiered auto-fill): "ทำเอง"/"เปิดเอง" added —
# these were REMOVED from ENTITY_NATURAL_RE's direct-match set in Part 1 (too
# ambiguous for a silent, no-LLM-check auto-fill: live-reproduced false
# positive on "ค่าธรรมเนียมทำเองกับจ้างตัวแทน...ต่างกันไหม", a DIY-vs-agent
# fee question with zero entity-type signal). Adding them here gives them a
# supervised second chance: instead of being silently dropped, they now route
# through the context-aware LLM fallback (_entity_type_llm_fallback, 0.70
# confidence threshold — already correctly returns null for the fee-question
# false positive, verified live) with the result tagged "inferred" so the
# answer can surface an implicit confirmation (see entity_type_source /
# _entity_type_just_inferred in persona_supervisor.py). Deliberately NOT
# adding bare "เอง" (too generic — กินเอง/จ่ายเอง/ไปเอง/ฯลฯ — would trigger the
# LLM fallback on a large fraction of unrelated messages for little value) or
# bare "บริษัท"/"ห้าง" (same reasoning — bare "บริษัท" overwhelmingly means "a
# company" in some unrelated context, e.g. a former employer; bare "ห้าง" is
# just "shopping mall" — routing THOSE to an LLM classifier on every mention
# would be a high-volume, low-value trigger, which is exactly the kind of
# blanket-LLM-call cost the confidence-tiered design is meant to avoid).
ENTITY_HINT_RE = re.compile(
    r"นิติ|บุคคล|บจก\.?|หจก\.?|บมจ\.?|ห\.?พ\.?|ส่วนตัว|คนเดียว|เจ้าของ|ประเภท(?:นิติ|ธุรกิจ|กิจการ|ผู้ประกอบการ)|รูปแบบ(?:ธุรกิจ|กิจการ|นิติ)"
    r"|ทำ\s*เอง|เปิด\s*เอง",
    re.IGNORECASE,
)

# Negation markers checked immediately before an ENTITY_NITI_RE/ENTITY_NATURAL_RE
# match by detect_entity_type_with_negation() below.
_ENTITY_NEGATION_RE = re.compile(r"ไม่ใช่|ไม่เป็น|ไม่ได้เป็น|มิใช่", re.IGNORECASE)


def detect_entity_type_with_negation(text: str) -> "str | None":
    """Detect an explicit, unambiguous entity_type ("นิติบุคคล" / "บุคคลธรรมดา")
    stated in free text, aware of simple negation.

    Plain ENTITY_NITI_RE/ENTITY_NATURAL_RE substring checks (used as-is
    elsewhere in this codebase) get this wrong for phrasing like "ไม่ใช่
    บุคคลธรรมดา เป็นนิติบุคคล" — both patterns match, and whichever the caller
    happens to check first silently wins, ignoring that the user explicitly
    negated it. Found live in persona_academic.py's C3 entity pre-fill and its
    awaiting-sections entity-change detector (2026-08-05) — added here as a
    single, reusable, negation-aware version rather than patching each call
    site's own copy of the same "which one wins" logic separately.

    Returns "นิติบุคคล" or "บุคคลธรรมดา" only when exactly one entity type is
    stated with a non-negated match. Returns None when genuinely ambiguous —
    both types mentioned with no negation (e.g. a comparison question "บุคคล
    ธรรมดากับนิติบุคคลต่างกันยังไง"), only negated forms present, or neither
    pattern matches at all — deliberately does not guess in those cases.
    """
    t = text or ""
    niti_matches = list(ENTITY_NITI_RE.finditer(t))
    natural_matches = list(ENTITY_NATURAL_RE.finditer(t))

    def _is_negated(m: "re.Match[str]") -> bool:
        window = t[max(0, m.start() - 10):m.start()]
        return bool(_ENTITY_NEGATION_RE.search(window))

    niti_positive = any(not _is_negated(m) for m in niti_matches)
    natural_positive = any(not _is_negated(m) for m in natural_matches)

    if niti_positive and not natural_positive:
        return "นิติบุคคล"
    if natural_positive and not niti_positive:
        return "บุคคลธรรมดา"
    return None

# ── Phase 3 (2026-07-31): bare generic-dimension follow-up detection ────────
# GENERIC_FOLLOWUP_KEYWORDS_RE / SPECIFIC_TOPIC_RE reconciled to
# persona_supervisor.py's versions (already the richest — persona_practical.py's
# _BARE_GENERIC_DIM_RE/_SPECIFIC_TOPIC_NAME_RE, added earlier the same day, were
# explicitly written as a trimmed "mirror" of these, intended to converge here).
GENERIC_FOLLOWUP_KEYWORDS_RE = re.compile(
    r"(ขั้นตอน|การสมัคร|สมัคร|วิธีสมัคร|เอกสาร|ค่าธรรมเนียม|ค่าใช้จ่าย|"
    r"ระยะเวลา|ช่องทางยื่น|ช่องทาง|สถานที่ยื่น|แบบฟอร์ม"
    r"|ลิงก์|ลิงค์|ลิ้งก์|ลิ้งค์|ดาวน์โหลด|"
    r"อยากรู้|อยากทราบ|ต้องใช้|ต้องทำ|ต้องเตรียม|"
    # How-to follow-ups: user asks HOW to do the same process already in context
    r"วิธีส่ง|วิธียื่น|วิธีทำ|วิธีชำระ|วิธีดำเนินการ|"
    r"ทำไง|ทำยังไง|ทำอย่างไร|ยื่นไง|ส่งไง|ส่งยังไง|"
    # Entity-switch follow-ups: user modifies entity_type/condition of the SAME topic
    r"แล้วถ้าแบบ|ถ้าแบบ|ถ้าเป็นแบบ|แล้วถ้าเป็น|แล้วแบบ|ส่วนแบบ|"
    # Op-type follow-ups: user asks about different operation on SAME topic
    r"แล้วถ้าจะ)",
    re.IGNORECASE,
)
# Specific topic/license/bank/entity keywords that indicate the user is asking
# about a NEW topic — if present, this is NOT a generic follow-up regardless of
# also containing a generic-dimension word above.
SPECIFIC_TOPIC_RE = re.compile(
    r"(ใบอนุญาต|จดทะเบียน|ภาษีมูลค่าเพิ่ม|ภาษีป้าย|vat|ภพ\.?20|สุรา|เหล้า|ประกันสังคม|ป้ายร้าน|"
    r"ประกอบกิจการ|สถานที่จำหน่าย|ทะเบียนพาณิชย์|บริษัทจำกัด|ห้างหุ้นส่วน|qr.?pay|qr.?api|payment api|"
    r"บัญชีธนาคาร|หาพนักงาน|edc|รูดบัตร|คิวอาร์เพย์|คิวอาเพ|คิวอาเอพีไอ|ระบบชำระเงิน|"
    r"กสิกร|kbank|ไทยพาณิช(?:ย์)?|ไทยพานิช|scb|กรุงไทย|ktb|กรุงเทพ|bbl|ออมสิน|gsb|ทหารไทย|ttb|ธ\.ก\.ส|kkp|"
    r"สำนักงานบัญชี|ปิดงบ|งบบัญชี|งบการเงิน|ปิดบัญชี|ปิงบ|ตรวจสอบบัญชี|"
    r"ทำบัญชี|นักบัญชี|ผู้สอบบัญชี|ผู้ทำบัญชี|จัดทำบัญชี|"
    r"พนักงาน|รับสมัครงาน|ตำแหน่งงาน|ตำแหน่งรับสมัคร|จัดหาพนักงาน|ไทยมีงานทำ|ประกาศงาน|ลูกจ้าง|"
    r"มีกระแส|ยอดนิยม|โด่งดัง|ไวรัล|viral\b|ฮิต\b|ให้ดัง|ร้านดัง|ทำให้ดัง|คาเฟ่ดัง|กาแฟดัง)",
    re.IGNORECASE,
)


def is_bare_generic_followup(text: str, max_len: int = 60) -> bool:
    """
    True when `text` is a short, generic-dimension follow-up question (e.g.
    "ต้องใช้เอกสารอะไรบ้าง", "มีค่าธรรมเนียมเท่าไหร่") that names no specific
    topic of its own — i.e. it should be treated as a continuation of whatever
    topic is already active, not a signal to start a fresh/broader retrieval.

    Root cause this closes: this exact phrase pattern was the trigger for the
    2026-07-31 liquor-license topic-drift bug in persona_practical.py — a bare
    generic follow-up shares almost no vocabulary with the specific topic that
    preceded it (by definition), so a plain word-overlap-ratio check alone
    reliably misclassifies it as "a new topic". persona_academic.py's
    `_query_changed` had (and, before this fix, still has) the identical
    vulnerability — pure overlap ratio, no keyword short-circuit at all — so it
    was equally exposed even though the original bug was only reproduced in
    Practical mode.

    Call this as an early override BEFORE any overlap-ratio/topic-continuity
    calculation, in whichever of the three personas' own retrieval-decision
    functions runs — this is deliberately a small, additive check, not a
    replacement for those functions (which differ enough in their surrounding
    retrieval side-effects that unifying them outright was judged too risky in
    the 2026-07-31 audit — see persona_practical_retrieve_retry_loop_bug.md and
    classifier_consolidation_text_patterns.md in project memory).
    """
    t = (text or "").strip()
    if not t or len(t) > max_len:
        return False
    return bool(GENERIC_FOLLOWUP_KEYWORDS_RE.search(t) and not SPECIFIC_TOPIC_RE.search(t))

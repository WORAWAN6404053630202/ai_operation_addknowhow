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

import logging
import re
from typing import List, Tuple

import conf

_LOG = logging.getLogger(__name__)

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


# ── Chapter-fetch label matching (2026-09) ───────────────────────────────────
# Used by persona_supervisor.py's main_topic/sub_topic/operation_topic
# "chapter retrieval" — when a query names a chapter label exactly, ALL its
# docs are fetched via a Chroma metadata filter instead of relying on
# semantic top-K search (which can silently drop docs ranked below the
# cutoff). Before 2026-09 this matcher was duplicated inline: sub_topic's
# copy checked both the LLM-rewritten query AND the raw pre-rewrite user
# message (the rewriter often drops key words) plus an ASCII-identifier
# fallback (e.g. "B2B"), but main_topic's copy only had the plain-substring
# check against the rewritten query — an asymmetry that made main_topic
# chapter-fetch miss more often than sub_topic for the exact same class of
# paraphrased query, contributing to reported cross-sheet/within-chapter
# incompleteness. Extracted here as one shared function so both call sites
# (and any future one) can't diverge again — same rationale as every other
# consolidation in this module (see module docstring).
_ASCII_ID_RE = re.compile(r'[A-Za-z][A-Za-z0-9\-]{2,}|[0-9][A-Za-z][A-Za-z0-9\-]{1,}')


# Thai nominalizing prefixes: "การ" turns a verb into a gerund/noun (ตั้งราคา
# "to price" -> การตั้งราคา "pricing"), "ความ" turns an adjective/state into an
# abstract noun. Sheet labels consistently use the formal nominalized form
# (การตั้งราคา), but real customer questions are often phrased as the bare verb
# (ควรตั้งราคายังไง) with no "การ" at all — a live-reproduced case (2026-09,
# "ร้านอาหารเล็กๆ ควรตั้งราคาเมนูยังไงให้แข่งขันได้") found this caused the
# exact-match chapter-fetch shortcut to silently miss, falling back to weaker
# hybrid retrieval that dropped the single most relevant sub_topic doc entirely.
_THAI_NOMINALIZER_PREFIXES = ("การ", "ความ")
# Minimum length of the prefix-stripped remainder before it's used as a match
# candidate — without this, a short/generic stem (e.g. "การผลิต" -> "ผลิต",
# "produce/production", 4 chars) would false-positive-match almost any
# food-business query. 5 chars keeps this in line with the existing
# main_topic length gate's own threshold philosophy (see docstring below).
_NOMINALIZER_STRIP_MIN_LEN = 5


def _strip_thai_nominalizer(label_lower: str) -> str:
    """Return `label_lower` with a leading การ/ความ removed, or "" if it
    doesn't start with one of those prefixes or the remainder would be too
    short/generic to safely match on its own (see _NOMINALIZER_STRIP_MIN_LEN)."""
    for prefix in _THAI_NOMINALIZER_PREFIXES:
        if label_lower.startswith(prefix):
            remainder = label_lower[len(prefix):]
            return remainder if len(remainder) >= _NOMINALIZER_STRIP_MIN_LEN else ""
    return ""


def topic_label_matches_query(label: str, query_lower: str, raw_human_lower: str = "") -> bool:
    """True if `label` (a main_topic/sub_topic/operation_topic value) should be
    treated as explicitly named by the user's current turn.

    Checks, in order: (1) `label` as a literal substring of `query_lower`
    (the query used for retrieval — possibly LLM-rewritten), (2) the same
    against `raw_human_lower` (the raw pre-rewrite user message, when
    provided), (3) if `label` starts with a Thai nominalizing prefix (การ/
    ความ) and the remainder is long enough to be specific, the same two
    checks against the prefix-stripped form — catches colloquial verb-form
    phrasing that never uses the sheet's formal nominalized label (2026-09,
    see _strip_thai_nominalizer), (4) an ASCII-identifier match — every ASCII
    token (3+ chars, e.g. "B2B") in `label` must appear somewhere in either
    text, catching cases where the LLM rewriter drops surrounding Thai
    particles but keeps the distinctive ASCII term. Callers should
    additionally gate on a minimum `len(label)` themselves (5 for main_topic,
    8 for sub_topic in the current call sites) to avoid short/generic labels
    matching too broadly — that threshold is corpus-specific, not part of
    this function's contract.

    Callers must lowercase `query_lower`/`raw_human_lower` themselves (and
    should reuse the same lowercased strings across all label checks in a
    loop, rather than re-lowercasing per call).
    """
    label_lower = label.lower()
    if label_lower in query_lower:
        return True
    if raw_human_lower and label_lower in raw_human_lower:
        return True
    stripped = _strip_thai_nominalizer(label_lower)
    if stripped:
        if stripped in query_lower:
            return True
        if raw_human_lower and stripped in raw_human_lower:
            return True
    tokens = _ASCII_ID_RE.findall(label)
    if not tokens:
        return False
    combined = query_lower + " " + raw_human_lower
    return all(t.lower() in combined for t in tokens)


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


# ── Canonical registration/application action-intent detector ──────────────
# Promoted from PersonaSupervisor._ACTION_Q_RE (2026-09) as part of the
# PDF-book truncation/retrieval fix chain — see
# pdf_chunking_truncation_and_retrieval_fixes project memory. This is the
# comprehensive list of phrasings meaning "the user wants to actually
# register/apply/modify/cancel/renew something", not just ask about it.
#
# Why promoted: while extracting persona_practical.py's separate
# fee/duration-query gate (below), its own hand-rolled negative-intent regex
# turned out to be an independently-drifted, much narrower copy of this exact
# concept — missing "อยากขอ", "ต้องการยื่น", "จะขอใบ", "วิธีลงทะเบียน", the bare
# "ขอใบ.../ยื่นขอใบ..." pattern, and the modify/cancel/renew verbs entirely.
# That gap meant a query like "อยากขอใบอนุญาตขายสุรา ค่าธรรมเนียมเท่าไหร่" (user
# has real action intent) was wrongly treated as purely informational, forcing
# a direct fee answer instead of letting the normal slot-filling flow ask for
# the details the fee actually depends on. Same class of duplicated-logic
# drift as everything else in this module's docstring — now there is exactly
# one action-intent pattern for every caller to share.
ACTION_INTENT_RE = re.compile(
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


# ── Fee/duration informational-query detector ───────────────────────────────
# Extracted from persona_practical.py (2026-09) as part of the same fix chain.
# Was previously an inline pair of regexes computed only in
# persona_practical.py; pulled out here so it can be unit-tested directly
# against a validation query set before any future tightening, following the
# same module docstring rationale (one place to live, nothing left to diverge).
FEE_DURATION_TRIGGER_RE = re.compile(
    r"ค่าธรรมเนียม|เท่าไหร่|กี่บาท|กี่วัน|กี่เดือน|กี่ปี|ค่าใช้จ่าย|ระยะเวลา",
    re.IGNORECASE,
)


def is_fee_or_duration_info_query(intent_text: str) -> bool:
    """True when `intent_text` asks about fees/cost/duration (ค่าธรรมเนียม/
    เท่าไหร่/กี่วัน/ระยะเวลา ฯลฯ) without also expressing intent to actually
    register/apply/modify/cancel/renew something (see `ACTION_INTENT_RE`).

    Used to force action="answer" with all fee tiers combined for purely
    informational fee/duration questions, instead of triggering
    entity_type/operation_group slot-asking that only makes sense when the
    user is about to act.
    """
    text = intent_text or ""
    return bool(FEE_DURATION_TRIGGER_RE.search(text)) and not bool(
        ACTION_INTENT_RE.search(text)
    )


# ── Specific-topic-conflict guard for exact-match chapter-fetch sites ───────
# Shared by persona_supervisor.py (Step 0/0b, main_topic/sub_topic chapter
# retrieval, and the duplicated fallback copies inside
# _route_pending_slot_to_persona) and persona_academic.py's _retrieve_docs.
#
# Specificity override: when these license names/codes appear in the query,
# it names a specific, more-authoritative topic that any exact-matched
# candidate must be corroborated by. e.g. "ขอใบภาษีมูลค่าเพิ่ม ภพ.20
# ต้องใช้เอกสารอะไรบ้าง" is targeted, not broad.
SPECIFIC_LICENSE_INDICATOR_RE = re.compile(
    # Specific form/tax codes
    r"ภพ\.\s*\d|ภส\.\s*\d|ภป\.\s*\d|สณ\.\s*\d|บอจ\.\s*\d|VAT|ภ\.อ\."
    # Named license types
    r"|ใบภาษีมูลค่าเพิ่ม|ใบทะเบียนพาณิชย์|ทะเบียนพาณิชย์"
    r"|ใบอนุญาตจัดตั้ง|ใบอนุญาตจำหน่ายสุรา|ใบอนุญาตจำหน่าย"
    r"|ใบรับรองมาตรฐานร้านอาหาร|ใบวุฒิบัตรผู้สัมผัสอาหาร"
    r"|ใบรับรองแพทย์|แบบแสดงรายการภาษีป้าย"
    r"|ประกันสังคม|กองทุนประกันสังคม"
    # 2026-09: added after live QA rerun found the LLM broad_q classifier
    # (_broad_question_llm_check) firing 0.95-confidence "broad" on
    # single-device questions naming EDC or QR Payment specifically (e.g.
    # "ขั้นตอนการสมัครเครื่อง EDC...เรียงลำดับ") — these terms weren't in
    # this list at all, so the specificity override never had a chance to
    # apply. license_type values confirmed in Chroma: "เครื่องรูดบัตร EDC",
    # "ระบบชำระเงินออนไลน์" (QR Payment's actual license_type).
    r"|EDC|เครื่องรูดบัตร|รูดบัตร|\bQR\b|QR\s*Payment|คิวอาร์|ระบบชำระเงินออนไลน์|ระบบชำระเงิน",
    re.IGNORECASE,
)


def filter_by_specific_topic_conflict(
    q: str, candidates: List[str], label: str = "candidate", log_prefix: str = "",
) -> List[str]:
    """Shared guard for every exact-match chapter-fetch site (OBD, operation_topic,
    main_topic, sub_topic label matching, across both persona_supervisor.py and
    persona_academic.py) — narrows or rejects a set of exact-matched candidates
    when the query ALSO names a specific, more-authoritative topic
    (`SPECIFIC_LICENSE_INDICATOR_RE`) that none of the candidates corroborate.

    2026-09: originally written inline only for the operation_by_department ("OBD")
    exact-match site in persona_supervisor.py after a live-reproduced bug — a
    short, generic OBD label ("บัญชีธนาคาร") exact-matched the query while the
    query ALSO explicitly named an unrelated, more specific topic ("QR Payment")
    that should have taken priority; the OBD match short-circuited straight to 4
    wrong docs even though plain semantic retrieval already ranked the correct QR
    Payment docs correctly on its own with no shortcut involved. A systematic
    audit then found the SAME exact-match-short-circuit shape duplicated across
    several other sites (operation_topic in Step 0, main_topic/sub_topic chapter
    retrieval, a fully separate copy-pasted duplicate of Step 0/0b inside the
    topic-slot-fill fallback, and persona_academic.py's own independent copy of
    the main_topic/sub_topic pattern) — none of which had this guard, which is
    exactly the kind of "same bug recurs via duplicated logic" risk this project
    has hit before (see classifier_consolidation_text_patterns.md-style history).
    Extracted here as ONE shared implementation, in the module already dedicated
    to preventing this exact class of cross-persona drift, so every exact-match
    site can call it instead of re-deriving (and inevitably drifting from) its
    own copy.

    Returns:
      - `candidates` unchanged if the query doesn't match any specific-indicator
        term (nothing to conflict with — the overwhelmingly common case, zero
        behavior change).
      - Only the candidates CORROBORATED by a specific-indicator term, if at
        least one exists (prefer the confirmed one(s) over a blind max(key=len),
        which live-reproduced picking the wrong candidate purely because it
        happened to be one Thai character longer).
      - [] if none of the candidates relate to the named specific topic at all —
        genuinely conflicting; the caller should treat this as "no confident
        match" and fall through to normal semantic retrieval instead.
    """
    if not candidates or not SPECIFIC_LICENSE_INDICATOR_RE.search(q):
        return candidates
    _si_matches = SPECIFIC_LICENSE_INDICATOR_RE.findall(q)
    _confirmed = [
        c for c in candidates
        if any(_si.lower() in c.lower() for _si in _si_matches)
    ]
    if _confirmed:
        if len(_confirmed) < len(candidates):
            _LOG.info(
                "%s%s matches %r narrowed to %r (corroborated by specific topic %r)",
                log_prefix, label, candidates, _confirmed, _si_matches,
            )
        return _confirmed
    _LOG.info(
        "%s%s match %r conflicts with specific topic %r in query — rejecting",
        log_prefix, label, candidates, _si_matches,
    )
    return []


# ── Regulatory-dominance guard for exact-match chapter-fetch sites ─────────
# Shared by persona_supervisor.py (operation_topic / sub_topic / main_topic
# chapter-fetch sites inside _ensure_practical_retrieval_for_legal) and
# persona_academic.py's own independent copy of main_topic chapter retrieval
# (_retrieve_docs). Moved here 2026-09/2026-10 (same reasoning as
# filter_by_specific_topic_conflict above — one shared implementation instead
# of letting independently-written duplicates drift) after persona_academic.py
# was confirmed, live-reproduced, to have the exact same bug as
# persona_supervisor.py's main_topic site: "การจดทะเบียนและขอใบอนุญาตร้านอาหาร
# ต้องทำยังไง" exact-matched a shallow 4-row business_guide chapter in BOTH
# personas' independent main_topic-matching code, discarding a confident
# (0.73 similarity) regulatory food-permit answer in both.

# Calibrated against two real, live-reproduced cases (2026-09): the commercial-
# registration case (regulatory top hit scores 0.735-0.756 — a confident,
# correct signal that a business_guide exact match should be rejected) vs. the
# SME income-tax-bracket case (the "regulatory" top hit is an unrelated
# VAT-registration row that only scores 0.55-0.57 — a weak, coincidental
# keyword overlap, NOT a genuinely better answer; rejecting business_guide
# there left NO usable content at all and broke answer generation entirely).
# 0.70 sits with comfortable margin below the confirmed-good case and above
# the confirmed-bad case.
NONREG_CHAPTER_REGULATORY_DOMINANCE_MIN_SCORE = 0.70


def nonreg_chapter_semantic_check(retriever, q: str, log_prefix: str = "") -> Tuple[str, float, str]:
    """Cheap, score-aware semantic sanity check for every non-regulatory
    (marketing/business_guide) exact-match chapter-fetch site across both
    persona_supervisor.py and persona_academic.py.

    Returns (top_data_type, top_score, top_nonreg_main_topic):
      - top_data_type: data_type of the single best-scoring doc overall (lowercased).
      - top_score: that doc's raw similarity score (0.0 if unavailable).
      - top_nonreg_main_topic: main_topic of the best-scoring doc AMONG
        marketing/business_guide results specifically (empty if none in top-5).

    Deliberately bypasses the callers' own full `_retrieve_docs()` pipelines
    (reranker, Round-2 anchor expansion, synonym/query-rewrite) — those
    pipelines don't propagate a usable score onto their returned dicts
    (reranking replaces ordering without keeping a queryable score field),
    which is exactly what this check needs. A small raw top-5 dense/hybrid
    lookup is cheap and sufficient for a sanity check; full retrieval quality
    is not what's being assessed here.
    """
    try:
        vstore = getattr(retriever, "vectorstore", None)
        if vstore is None:
            return "", 0.0, ""
        if getattr(conf, "HYBRID_SEARCH_ENABLED", False):
            from utils.hybrid_retriever import hybrid_scored_search
            pairs = hybrid_scored_search(
                vstore, q, k=5,
                rrf_k=int(getattr(conf, "HYBRID_RRF_K", 60)),
                bm25_bypass_min_ratio=float(getattr(conf, "BM25_BYPASS_MIN_RATIO", 0.35)),
            )
        else:
            pairs = vstore.similarity_search_with_relevance_scores(q, k=5)
        if not pairs:
            return "", 0.0, ""
        _top_doc, _top_score = pairs[0]
        _top_dt = str((_top_doc.metadata or {}).get("data_type") or "").strip().lower()
        _top_nonreg_mt = ""
        for _d, _s in pairs:
            _md = _d.metadata or {}
            if str(_md.get("data_type") or "").strip().lower() in ("marketing", "business_guide"):
                _top_nonreg_mt = str(_md.get("main_topic") or "").strip()
                break
        return _top_dt, float(_top_score or 0.0), _top_nonreg_mt
    except Exception as _e_nrc:
        _LOG.warning("%s_nonreg_chapter_semantic_check failed: %s", log_prefix, _e_nrc)
        return "", 0.0, ""


# ── PDF-book-aware similarity floor ─────────────────────────────────────────
# Shared by persona_practical.py's and persona_academic.py's independently
# re-implemented sim_filter steps (2026-09) — see conf.RETRIEVAL_MIN_SIMILARITY_PDF_BOOK
# for the full measurement this is based on. Extracted here, following this
# module's usual rationale, so both copies can't silently drift back apart and
# so the threshold-selection logic itself is directly unit-testable without
# needing a live retriever.
def effective_min_similarity(
    doc_metadata: dict, min_similarity: float, min_similarity_pdf_book: float,
) -> float:
    """Return the similarity floor that applies to one retrieved doc.

    PDF-book content (`ingested_via="pdf_auto"`) gets the lower,
    evidence-based `min_similarity_pdf_book` floor instead of the general
    `min_similarity` — see conf.py's RETRIEVAL_MIN_SIMILARITY_PDF_BOOK comment
    for the real score measurements (genuinely-correct PDF-book matches
    scored 0.487-0.577, all below the general 0.6 floor, while unrelated
    queries' highest PDF-doc score topped out at 0.3998 — 0.45 sits cleanly
    in that gap). Regulatory/legacy Sheet-based content is unaffected.
    """
    is_pdf_book = str((doc_metadata or {}).get("ingested_via") or "").strip().lower() == "pdf_auto"
    return min_similarity_pdf_book if is_pdf_book else min_similarity


# ── Thai-aware keyword extraction for topic/relevance scoring ───────────────
# 2026-09 bug fix, root-cause layer of the pdf_chunking_truncation_and_retrieval_fixes
# chain: persona_academic.py's _detect_distinct_topics extracted query "keywords"
# via `re.sub(r'[^฀-๿a-zA-Z0-9]', ' ', query).split()` — plain
# whitespace splitting, which does nothing useful for Thai text (no spaces
# between words in normal writing). Live-reproduced: the query "ในงบกำไรขาดทุน
# Top Line กับ Bottom Line ต่างกันยังไง" collapsed into
# ['ในงบกำไรขาดทุน', 'top', 'line', 'กับ', 'bottom', 'ต่างกันยังไง'] — the
# meaningful word "กำไร" (profit) was trapped inside one giant unsplittable
# blob and could never match the correct topic's title
# ("การวิเคราะห์รายได้และกำไร"), while the bare conjunction "กับ" ("and/with",
# meaningless on its own) survived as its own 3-character "word" and happened
# to coincidentally appear in a completely unrelated topic's content, handing
# that wrong topic the only nonzero relevance score in the whole candidate set.
#
# Fix: reuse real Thai word segmentation (pythainlp, same `word_tokenize`
# engine="newmm" already used by utils.hybrid_retriever's BM25 index, kept
# consistent on purpose rather than picking a different engine here) plus
# pythainlp's maintained Thai stopword corpus, so common function words like
# "กับ"/"ใน"/"ที่" can never masquerade as meaningful query keywords again —
# tokenizing alone would have rescued "กำไร" but NOT suppressed "กับ", both
# halves are needed to actually close this bug, not just make one query pass.
def extract_thai_keywords(text: str, min_len: int = 2) -> List[str]:
    """Extract meaningful, lowercased keywords from `text` for substring-based
    title/content relevance scoring (e.g. persona_academic.py's topic-menu
    keyword matching).

    Falls back to the old whitespace-split behavior if pythainlp is
    unavailable for any reason — same resilience pattern as
    utils.hybrid_retriever._tokenize, never a hard dependency at call time.
    """
    t = str(text or "")
    try:
        from pythainlp.corpus import thai_stopwords
        from pythainlp.tokenize import word_tokenize
        tokens = word_tokenize(t.lower(), engine="newmm", keep_whitespace=False)
        stopwords = thai_stopwords()
        return [
            tok for tok in tokens
            if tok and tok.strip() and len(tok) >= min_len and tok not in stopwords
        ]
    except Exception as _e:
        _LOG.debug("[extract_thai_keywords] PyThaiNLP unavailable (%s) — using whitespace split", type(_e).__name__)
        _clean = re.sub(r'[^฀-๿a-zA-Z0-9]', ' ', t.lower())
        return [w for w in _clean.split() if len(w) >= max(min_len, 3)]

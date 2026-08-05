# code/utils/query_rewriter.py
"""
LLM-based query rewriter for RAG retrieval.

Problem solved: embedding models struggle with informal Thai phrasing that doesn't
match formal regulatory terminology stored in Chroma documents.
  e.g. "ขายเหล้าได้ถึงกี่โมง" → vector far from "เวลาจำหน่ายเครื่องดื่มแอลกอฮอล์"
  e.g. "คิวอาเพเม้น" → vector far from "QR Payment ระบบชำระเงินออนไลน์"

Solution: use Haiku to rewrite informal queries into formal Thai legal/regulatory
vocabulary before embedding. The rewritten text is appended to (not replacing) the
original query so synonym expansion and BM25 still work on original keywords.

Usage (from any retrieval path):
    from utils.query_rewriter import enrich_query_for_retrieval

    expanded = enrich_query_for_retrieval(query)   # no-op if already formal
"""
from __future__ import annotations

import logging
import re
from typing import Optional

_LOG = logging.getLogger("restbiz.query_rewriter")

# Queries that already contain formal regulatory terms — skip LLM rewrite.
# Keeps latency low for the majority of queries that are already well-formed.
_FORMAL_RE = re.compile(
    r"ใบอนุญาต|ขั้นตอนการ(?:ดำเนิน|ยื่น|จด)|เวลาจำหน่าย|ยื่นคำขอ|จดทะเบียน(?:พาณิชย์|ห้าง|บริษัท)"
    r"|ค่าธรรมเนียม.*ใบ|หลักเกณฑ์|ข้อกำหนด|พ\.ร\.บ|ประกาศ.*(?:กรม|สำนักงาน)|ยื่นเอกสาร"
    # Common formal regulatory terms that don't benefit from LLM expansion
    r"|ภาษี|ประกันสังคม|VAT|ภพ\.?20|ทะเบียนพาณิชย์|สรรพากร|ประกอบกิจการ"
    # Formal procedure/fee/timeline vocabulary already understood by the embedding model
    r"|ขั้นตอน|ค่าธรรมเนียม|ระยะเวลา|ช่องทางการ|QR.?Pay",
    re.IGNORECASE,
)

_PROMPT = (
    "คุณเป็น query rewriter สำหรับระบบ RAG กฎหมายธุรกิจร้านอาหารไทย\n"
    "งาน: แปลงคำถามภาษาพูดเป็นคำศัพท์ทางการ/ราชการ เพื่อให้ vector search เจอเอกสารที่ถูกต้อง\n\n"
    "กฎ:\n"
    "- ตอบเฉพาะ query ที่ rewrite แล้ว บรรทัดเดียว ไม่ต้องอธิบาย ไม่ต้องขึ้นต้นด้วย 'query:' หรืออื่นๆ\n"
    "- แปลงคำพูดทั่วไปเป็นคำราชการ เช่น:\n"
    "  'กี่โมง' → 'เวลาจำหน่าย ช่วงเวลา'\n"
    "  'เหล้า/เบียร์' → 'เครื่องดื่มแอลกอฮอล์ สุรา'\n"
    "  'คิวอาเพเม้น/คิวอาร์' → 'QR Payment ระบบชำระเงินออนไลน์'\n"
    "  'ทำไมต้อง/ต้องทำอะไรบ้าง' → 'เงื่อนไข ขั้นตอน หลักเกณฑ์'\n"
    "- เพิ่ม related terms ที่น่าจะปรากฏใน document ราชการ\n"
    "- ไม่เกิน 20 คำ\n\n"
    "Query: {query}"
)

# Module-level singleton LLM (lazy init) — shared across all personas
_llm = None
# Module-level cache — persists across sessions for identical queries (correct behaviour:
# same informal phrase always maps to same formal expansion)
_module_cache: dict = {}


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    try:
        import conf
        from langchain_openai import ChatOpenAI
        from utils.llm_call import get_shared_http_client
        _llm = ChatOpenAI(
            model=getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL),
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            http_client=get_shared_http_client(),
            temperature=0.0,
            max_tokens=100,
            request_timeout=int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8)),
        )
        _LOG.debug("[QueryRewriter] LLM initialised: %s",
                   getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL))
    except Exception as _e:
        _LOG.warning("[QueryRewriter] LLM init failed: %s", _e)
    return _llm


def _needs_rewrite(query: str) -> bool:
    """Return True if the query is informal and likely to benefit from LLM rewriting."""
    q = (query or "").strip()
    if len(q) < 6:
        return False
    if _FORMAL_RE.search(q):
        return False
    return True


def enrich_query_for_retrieval(query: str, session_cache: Optional[dict] = None) -> str:
    """
    Return an enriched query string for Chroma embedding.

    If the query is already formal, returns it unchanged (no LLM call).
    Otherwise, appends an LLM-generated formal expansion to the original query
    so that: (a) informal keywords are preserved for BM25, (b) formal terms
    bridge the vector-space gap to regulatory documents.

    session_cache: optional per-session dict (pass state.context.setdefault("_qr_cache", {}))
                   to avoid repeat LLM calls within the same conversation turn.
                   Falls back to module-level cache if None.
    """
    q = (query or "").strip()
    if not q or not _needs_rewrite(q):
        return q

    _cache = session_cache if session_cache is not None else _module_cache
    if q in _cache:
        cached = _cache[q]
        _LOG.debug("[QueryRewriter] cache hit: %r → %r", q[:40], cached[:60])
        return cached

    llm = _get_llm()
    if llm is None:
        return q

    prompt = _PROMPT.format(query=q)
    try:
        from langchain_core.messages import HumanMessage
        from utils.llm_call import llm_invoke, extract_llm_text
        raw = extract_llm_text(
            llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="QueryRewriter")
        ).strip()

        # Sanity: reject if LLM returned an explanation instead of a query
        if raw and 3 < len(raw) <= 150 and raw != q:
            enriched = f"{q} {raw}"
            _LOG.info("[QueryRewriter] %r → appended %r", q[:50], raw[:70])
            if _cache is _module_cache and len(_module_cache) >= 500:
                try:
                    del _module_cache[next(iter(_module_cache))]
                except (StopIteration, KeyError):
                    pass
            _cache[q] = enriched
            return enriched
    except Exception as _e:
        _LOG.warning("[QueryRewriter] call failed: %s", _e)

    if _cache is _module_cache and len(_module_cache) >= 500:
        try:
            del _module_cache[next(iter(_module_cache))]
        except (StopIteration, KeyError):
            pass
    _cache[q] = q
    return q

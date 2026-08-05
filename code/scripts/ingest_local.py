# code/scripts/ingest_local.py
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Add parent directory to path so we can import from code/
project_root = Path(__file__).parent.parent.parent
code_dir = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from service.data_loader import DataLoader
from service.data_loader_general import GeneralDataLoader
from service.local_vector_store import ingest_documents
from langchain_core.documents import Document
import conf

# ── Sheet URLs (read from env.properties via conf) ───────────────────────────
SHEET_URL_REGULATORY = conf.SHEET_URL_REGULATORY
SHEET_URL_MARKETING  = conf.SHEET_URL_MARKETING
SHEET_URL_BAKERY     = conf.SHEET_URL_BAKERY

# Legacy alias kept for backward compatibility if anything else imports it
SHEET_URL = SHEET_URL_REGULATORY

# Data normalization (ingest-time)
_WS_RE = re.compile(r"\s+", re.UNICODE)

# Match "token token" duplicated consecutively WITH whitespace.
# - Thai token: [\u0E00-\u0E7F]+
# - Latin/num token: [A-Za-z0-9]+
_DUP_TOKEN_WS_RE = re.compile(
    r"(?:(?<=\s)|^)"
    r"("
    r"(?:[\u0E00-\u0E7F]+|[A-Za-z0-9]+)"
    r")"
    r"(?:\s+)"
    r"\1"
    r"(?=(?:\s|$))",
    re.UNICODE,
)

# Match "token" duplicated consecutively WITHOUT whitespace (typo like "เอกสารเอกสาร", "VATVAT").
# Conservative:
# - Thai token >= 6 chars: 3 was too low — "\u0e01าร" (3 chars) ends/starts compound words
#   like จัดการการเงิน (จัดการ+การเงิน) and would be incorrectly stripped.
#   6+ catches real typo-duplicates (เอกสาร=6, ทะเบียน=7) without breaking compounds.
# - Latin/num token >= 3 chars (safe: short Latin tokens rarely form meaningful compounds)
_DUP_TOKEN_NOSPACE_RE = re.compile(
    r"((?:[\u0E00-\u0E7F]{6,}|[A-Za-z0-9]{3,}))\1",
    re.UNICODE,
)

_URLISH_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_EMAILISH_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)


def _normalize_text(s: str) -> str:
    """
    Conservative normalization (ingest-time):
    - normalize whitespace
    - remove obvious consecutive duplicate tokens
        - with whitespace: "เอกสาร เอกสาร" -> "เอกสาร"
        - no whitespace: "เอกสารเอกสาร" -> "เอกสาร"
    - fix a few known repeated phrases (safe targeted)
    """
    if s is None:
        return ""
    t = str(s)

    # Remove zero-width spaces and normalize whitespace (safe for URLs too)
    t = t.replace("\u200b", "")
    t = _WS_RE.sub(" ", t).strip()

    # Skip aggressive dedup transforms when string contains a URL/email \u2014
    # duplicate-token regex can corrupt compound words adjacent to URLs
    if _URLISH_RE.search(t) or _EMAILISH_RE.search(t):
        return t
    if not t:
        return t

    # Targeted safe fixes (known typos)
    t = t.replace("การจดทะเบียนทะเบียนพาณิชย์", "การจดทะเบียนพาณิชย์")
    t = t.replace("จดทะเบียนทะเบียนพาณิชย์", "จดทะเบียนพาณิชย์")
    t = t.replace("ทะเบียนทะเบียนพาณิชย์", "ทะเบียนพาณิชย์")

    # Remove consecutive duplicate tokens (repeat until stable, but cap passes)
    # 1) whitespace duplicates: "ทะเบียน ทะเบียน พาณิชย์" -> "ทะเบียน พาณิชย์"
    # 2) no-space duplicates: "เอกสารเอกสารประเภ..." -> "เอกสารประเภ..."
    for _ in range(4):
        new_t = _DUP_TOKEN_WS_RE.sub(r"\1", t)
        new_t = _DUP_TOKEN_NOSPACE_RE.sub(r"\1", new_t)
        new_t = _WS_RE.sub(" ", new_t).strip()
        if new_t == t:
            break
        t = new_t

    return t


def _normalize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(md, dict):
        return md

    out: Dict[str, Any] = {}
    for k, v in md.items():
        if isinstance(v, str):
            out[k] = _normalize_text(v)
        else:
            # keep non-str as-is (None, numbers, lists, dicts)
            out[k] = v
    return out


_VALID_DATA_TYPES = {"regulatory", "marketing", "business_guide"}
_VALID_ENTITY_TYPES = {"นิติบุคคล", "บุคคลธรรมดา", ""}


def _validate_documents(docs: List[Document]) -> int:
    """
    Validate document metadata fields after normalization.

    Prints a warning for each document that has a metadata issue which would
    cause silent retrieval failures at query time (e.g. Chroma filter mismatches).
    Does not remove documents — all are ingested regardless of issues.

    Returns the total number of warnings emitted.
    """
    warnings_total = 0

    for i, doc in enumerate(docs or []):
        md = getattr(doc, "metadata", {}) or {}
        issues: List[str] = []

        # Validate entity_type_normalized — must be canonical value or empty.
        # A non-canonical value (e.g. "บริษัท") will never match a Chroma filter
        # and will silently return 0 results for filtered queries.
        et = str(md.get("entity_type_normalized") or "").strip()
        if et and et not in _VALID_ENTITY_TYPES:
            issues.append(
                f"entity_type_normalized={et!r} is not a canonical value "
                f"({sorted(_VALID_ENTITY_TYPES - {''})}) — Chroma filter will not match"
            )

        # Validate data_type — must be one of the known source categories.
        dt = str(md.get("data_type") or "").strip()
        if dt and dt not in _VALID_DATA_TYPES:
            issues.append(
                f"data_type={dt!r} is not a recognised category "
                f"({sorted(_VALID_DATA_TYPES)})"
            )

        # Validate page_content — very short content produces poor embeddings.
        content = str(getattr(doc, "page_content", "") or "").strip()
        if len(content) < 20:
            issues.append(
                f"page_content too short ({len(content)} chars) — embedding quality will be poor"
            )

        # Non-regulatory docs must have operation_topic for supervisor topic routing.
        if dt in ("marketing", "business_guide"):
            op_topic = str(md.get("operation_topic") or "").strip()
            if not op_topic:
                issues.append(
                    "operation_topic is empty on a non-regulatory doc — supervisor topic routing will not surface this topic"
                )

        # Regulatory docs must have a license_type to enable slot-queue discovery.
        if dt == "regulatory":
            lt = str(md.get("license_type") or "").strip()
            if not lt:
                issues.append("license_type is empty on a regulatory doc — slot discovery will be skipped")
            # topic_group must be set so 2-pass retrieval can scope by group
            tg = str(md.get("topic_group") or "").strip()
            if not tg:
                issues.append(
                    f"topic_group is empty on a regulatory doc (license_type={lt!r}) "
                    "— 2-pass retrieval will not be able to scope by topic group"
                )

        if issues:
            row_ref = md.get("license_type") or md.get("operation_topic") or f"row {i + 1}"
            for issue in issues:
                print(f"[Ingest] WARN [{row_ref}] {issue}")
                warnings_total += 1

    return warnings_total


def normalize_documents(docs: Iterable[Any]) -> None:
    """
    Mutates doc objects in-place:
      - doc.page_content (if exists)
      - doc.metadata (if exists)
    Compatible with LangChain Document-like objects.
    """
    for d in docs or []:
        # page_content
        if hasattr(d, "page_content"):
            try:
                pc = getattr(d, "page_content", "") or ""
                setattr(d, "page_content", _normalize_text(pc))
            except Exception:
                pass

        # metadata
        if hasattr(d, "metadata"):
            try:
                md = getattr(d, "metadata", {}) or {}
                setattr(d, "metadata", _normalize_metadata(md))
            except Exception:
                pass


def main():
    all_docs: List[Document] = []

    # ── Source A: Regulatory / government procedures ─────────────────────────
    print("\n[Ingest] Loading Sheet A — Regulatory (กฎหมาย/ใบอนุญาต/ราชการ)...")
    dl = DataLoader(config={})
    dl.load_from_google_sheet(SHEET_URL_REGULATORY, source_name="regulatory")
    all_docs.extend(dl.documents)
    print(f"[Ingest]   → {len(dl.documents)} regulatory docs")

    # ── Source B: Marketing strategy / SOP ──────────────────────────────────
    print("\n[Ingest] Loading Sheet B — Marketing (การตลาด/กลยุทธ์/SOP)...")
    dl_mkt = GeneralDataLoader(data_type="marketing")
    dl_mkt.load_from_google_sheet(SHEET_URL_MARKETING, source_name="marketing")
    all_docs.extend(dl_mkt.documents)
    print(f"[Ingest]   → {len(dl_mkt.documents)} marketing docs")

    # ── Source C: Bakery / coffee-shop startup guide ─────────────────────────
    print("\n[Ingest] Loading Sheet C — Business Guide (เบเกอรี่/คาเฟ่/เปิดร้าน)...")
    dl_biz = GeneralDataLoader(data_type="business_guide")
    dl_biz.load_from_google_sheet(SHEET_URL_BAKERY, source_name="business_guide")
    all_docs.extend(dl_biz.documents)
    print(f"[Ingest]   → {len(dl_biz.documents)} business_guide docs")

    # ── Normalize all docs ───────────────────────────────────────────────────
    docs = all_docs
    normalize_documents(docs)

    # ── Validate metadata fields ─────────────────────────────────────────────
    print("\n[Ingest] Validating document metadata...")
    warning_count = _validate_documents(docs)
    if warning_count == 0:
        print("[Ingest] Validation passed — no metadata issues found.")
    else:
        print(f"[Ingest] Validation complete — {warning_count} warning(s) above. "
              f"Documents will still be ingested; fix the source sheet to suppress warnings.")

    print(f"\n[Ingest] Total docs to index: {len(docs)}")
    print(f"[Ingest] Model: {conf.EMBEDDING_MODEL} (via OpenRouter)")

    # Show breakdown by data_type
    reg_docs   = [d for d in docs if d.metadata.get("data_type") != "marketing"
                                  and d.metadata.get("data_type") != "business_guide"]
    mkt_docs   = [d for d in docs if d.metadata.get("data_type") == "marketing"]
    biz_docs   = [d for d in docs if d.metadata.get("data_type") == "business_guide"]
    locs       = [d for d in docs if d.metadata.get("location")]
    entities   = [d for d in docs if d.metadata.get("entity_type_normalized")]
    print(f"[Ingest]   regulatory docs : {len(reg_docs)}")
    print(f"[Ingest]   marketing docs  : {len(mkt_docs)}")
    print(f"[Ingest]   business_guide  : {len(biz_docs)}")
    print(f"[Ingest]   location populated: {len(locs)}/{len(docs)}")
    print(f"[Ingest]   entity_type populated: {len(entities)}/{len(docs)}")

    ingest_documents(docs, reset=True)  # wipe old local chroma before rebuild

    # Invalidate BM25 cache so hybrid search rebuilds from new docs on next query
    try:
        from utils.hybrid_retriever import invalidate_bm25_cache
        invalidate_bm25_cache()
        print("[Ingest] BM25 cache invalidated — will rebuild on next search.")
    except Exception as _e:
        print(f"[Ingest] BM25 cache invalidation skipped: {_e}")

    print('[Ingest] Done! Start server: python code/app.py')


if __name__ == "__main__":
    main()
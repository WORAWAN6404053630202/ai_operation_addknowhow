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

# ── Sheet URLs ──────────────────────────────────────────────────────────────
# Sheet A: Regulatory / government procedures (original)
SHEET_URL_REGULATORY = (
    "https://docs.google.com/spreadsheets/d/1YnLKV7gJXCu7jvcH1sUL9crMlBCJKOpQfp2wtulMszE/"
    "edit?gid=657201027#gid=657201027"
)
# Sheet B: Marketing strategy, SOP, pricing, product mix
SHEET_URL_MARKETING = (
    "https://docs.google.com/spreadsheets/d/1YnLKV7gJXCu7jvcH1sUL9crMlBCJKOpQfp2wtulMszE/"
    "edit?gid=809205387#gid=809205387"
)
# Sheet C: Bakery / coffee-shop practical startup guide
SHEET_URL_BAKERY = (
    "https://docs.google.com/spreadsheets/d/1YnLKV7gJXCu7jvcH1sUL9crMlBCJKOpQfp2wtulMszE/"
    "edit?gid=610069215#gid=610069215"
)

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
# - token length >= 3 (avoid breaking short syllables)
# - Thai token OR latin/num token
_DUP_TOKEN_NOSPACE_RE = re.compile(
    r"((?:[\u0E00-\u0E7F]{3,}|[A-Za-z0-9]{3,}))\1",
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

    # Do not normalize URLs/emails aggressively (avoid breaking them)
    if _URLISH_RE.search(t) or _EMAILISH_RE.search(t):
        return t.strip()

    # Remove zero-width spaces and normalize whitespace
    t = t.replace("\u200b", "")
    t = _WS_RE.sub(" ", t).strip()
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

    print(f"\n[Ingest] Total docs to index: {len(docs)}")
    print("[Ingest] Model: multilingual-e5-large (better Thai retrieval accuracy)")

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
    print('[Ingest] ✅ Done! Start server: python code/app.py')


if __name__ == "__main__":
    main()
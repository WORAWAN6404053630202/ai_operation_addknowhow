# code/service/data_loader_general.py
"""
General Data Loader — for 6-column knowledge-base sheets.

Handles sheets with this simplified structure:
  เรื่อง/ชื่อหนังสือ | หัวข้อหลัก | หัวข้อการดำเนินการย่อย | ประเภท | แนวคำตอบ | อ้างอิง/เอกสาร

Unlike DataLoader (regulatory), these sheets carry no license/entity/location structure.
All metadata fields from the regulatory schema default to None so the slot-fill system
naturally skips slot discovery (no license_type → no queue built).

data_type values:
  "marketing"      — Sheet 1: marketing strategy, SOP, pricing, product mix
  "business_guide" — Sheet 2: bakery / coffee shop practical startup guide
"""

from __future__ import annotations

import math
import re
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd
from langchain_core.documents import Document


class GeneralDataLoader:
    """
    Flexible loader for the simplified 6-column knowledge-base sheets.

    Column mapping strategy:
    - Primary header → alias list → keyword-contains fallback (same 3-level strategy
      as DataLoader for robustness against minor header wording changes).
    - Rows with empty แนวคำตอบ AND empty หัวข้อการดำเนินการย่อย are skipped
      (no embeddable content).
    """

    # Maps logical field → (primary header, aliases, contains-any keywords)
    _COL_SPECS = {
        "source_book": (
            "เรื่อง",
            ["ชื่อหนังสือ", "ชื่อ หนังสือ", "เรื่อง/ชื่อหนังสือ"],
            ["ชื่อหนังสือ", "เรื่อง"],
        ),
        "main_topic": (
            "หัวข้อหลัก",
            ["หัวข้อ หลัก"],
            ["หัวข้อหลัก", "หัวข้อ"],
        ),
        "sub_topic": (
            "หัวข้อการดำเนินการย่อย",
            ["หัวข้อการ ดำเนินการย่อย", "หัวข้อการดำเนินการ ย่อย"],
            ["หัวข้อการดำเนินการย่อย", "ดำเนินการย่อย"],
        ),
        "content_type": (
            "ประเภท",
            [],
            ["ประเภท"],
        ),
        "answer_guideline": (
            "แนวคำตอบ",
            ["แนวคำตอบ / เนื้อหา", "แนวคำตอบ/เนื้อหา", "เนื้อหา"],
            ["แนวคำตอบ", "แนวตอบ", "เนื้อหา"],
        ),
        "research_reference": (
            "อ้างอิง",
            ["อ้างอิง/เอกสาร", "อ้างอิง / เอกสาร", "เอกสาร", "reference", "อ้างอิง (Reference)"],
            ["อ้างอิง", "reference", "เอกสาร"],
        ),
    }

    def __init__(self, data_type: str, page_content_max_chars: int = 1800):
        """
        data_type: "marketing" | "business_guide" (stored in doc metadata)
        """
        assert data_type in ("marketing", "business_guide"), \
            f"data_type must be 'marketing' or 'business_guide', got {data_type!r}"
        self.data_type = data_type
        self.page_content_max_chars = page_content_max_chars
        self.documents: List[Document] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_google_sheet(self, sheet_url: str, source_name: str = "") -> None:
        csv_url = self._build_csv_export_url(sheet_url)
        print(f"[GeneralDataLoader/{self.data_type}] Fetching CSV from: {csv_url}")
        df = pd.read_csv(csv_url, dtype=str)
        df.columns = [self._clean_header(c) for c in df.columns]
        colmap = self._build_column_map(df)
        count = self._process_dataframe(df, colmap, source_name or sheet_url)
        print(f"[GeneralDataLoader/{self.data_type}] Loaded {count} docs (skipped empty rows)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_csv_export_url(sheet_url: str) -> str:
        u = urlparse(sheet_url)
        base = f"{u.scheme}://{u.netloc}{u.path}"
        base = base.split("/edit")[0].rstrip("/")
        q = parse_qs(u.query)
        gid = (q.get("gid", [None])[0]) or (parse_qs(u.fragment).get("gid", [None])[0])
        if not gid:
            raise ValueError("Google Sheet URL missing gid.")
        return f"{base}/export?format=csv&gid={gid}"

    @staticmethod
    def _clean_header(name: str) -> str:
        if not isinstance(name, str):
            return name
        name = name.replace("\n", " ").replace("\r", " ")
        name = re.sub(r"\s+", " ", name)
        return name.strip()

    @staticmethod
    def _to_val(v) -> Optional[str]:
        if v is None:
            return None
        try:
            if isinstance(v, float) and math.isnan(v):
                return None
        except Exception:
            pass
        s = str(v).strip()
        return s if s and s.lower() not in ("nan", "none", "") else None

    def _build_column_map(self, df: pd.DataFrame) -> dict:
        """3-level column resolution: exact → alias → contains-keyword."""
        headers = list(df.columns)
        headers_lower = [h.lower() for h in headers]
        colmap = {}

        for field, (primary, aliases, keywords) in self._COL_SPECS.items():
            # Level 1: exact primary
            if primary in headers:
                colmap[field] = primary
                continue
            # Level 2: exact alias
            matched_alias = next((a for a in aliases if a in headers), None)
            if matched_alias:
                colmap[field] = matched_alias
                continue
            # Level 3: contains-keyword (case-insensitive)
            matched_kw = None
            for kw in keywords:
                kw_l = kw.lower()
                for h, h_l in zip(headers, headers_lower):
                    if kw_l in h_l:
                        matched_kw = h
                        break
                if matched_kw:
                    break
            if matched_kw:
                colmap[field] = matched_kw

        return colmap

    def _get(self, row, colmap: dict, field: str) -> Optional[str]:
        col = colmap.get(field)
        if col is None:
            return None
        return self._to_val(row.get(col))

    def _build_page_content(
        self,
        source_book: Optional[str],
        main_topic: Optional[str],
        sub_topic: Optional[str],
        content_type: Optional[str],
        answer_guideline: Optional[str],
        research_reference: Optional[str],
    ) -> str:
        """
        Build embedding text optimised for multilingual-e5-large.

        Lead with category signals (หัวข้อหลัก, sub_topic) so similarity search
        surfaces the right doc when user asks about the same topic.
        Body is the answer guideline — primary retrieval payload.
        """
        parts = []

        # Header: topic signals
        if main_topic:
            parts.append(f"หัวข้อหลัก: {main_topic}")
        if sub_topic:
            parts.append(f"หัวข้อย่อย: {sub_topic}")
        if content_type:
            parts.append(f"ประเภท: {content_type}")
        if source_book:
            parts.append(f"แหล่งข้อมูล: {source_book}")

        # Body: the actual answer content
        if answer_guideline:
            parts.append(f"แนวคำตอบ:\n{answer_guideline}")

        if research_reference:
            parts.append(f"อ้างอิง: {research_reference}")

        text = "\n".join(p.strip() for p in parts if p.strip())

        max_c = self.page_content_max_chars or 1800
        if len(text) > max_c:
            text = text[:max_c].rstrip()

        return text.strip()

    def _process_dataframe(self, df: pd.DataFrame, colmap: dict, source: str) -> int:
        added = 0
        for idx, row in df.iterrows():
            source_book = self._get(row, colmap, "source_book")
            main_topic = self._get(row, colmap, "main_topic")
            sub_topic = self._get(row, colmap, "sub_topic")
            content_type = self._get(row, colmap, "content_type")
            answer_guideline = self._get(row, colmap, "answer_guideline")
            research_reference = self._get(row, colmap, "research_reference")

            page_content = self._build_page_content(
                source_book, main_topic, sub_topic,
                content_type, answer_guideline, research_reference,
            )

            # Skip rows with no meaningful content — thin embeddings degrade search
            if not page_content or len(page_content) < 30:
                continue

            # topic_group for retrieval scoping — maps data_type to broad category
            _topic_group_map = {
                "marketing": "การตลาด",
                "business_guide": "คู่มือเปิดร้าน",
            }

            # Use main_topic as operation_topic analogue so supervisor topic-picker works
            metadata = {
                "row_id": int(idx),
                "data_type": self.data_type,
                "topic_group": _topic_group_map.get(self.data_type, "อื่นๆ"),
                "source": source,
                # Topic fields (used by topic-picker and retrieval display)
                "operation_topic": sub_topic or main_topic,
                "main_topic": main_topic,
                "sub_topic": sub_topic,
                "source_book": source_book,
                "book_name": source_book,   # explicit alias — used by Chroma filter & boost
                "content_type": content_type,
                # Primary answer content
                "answer_guideline": answer_guideline,
                "research_reference": research_reference,
                # Regulatory fields: all None — prevents slot-fill from triggering
                "license_type": None,
                "entity_type_normalized": None,
                "location": None,
                "area_size": None,
                "registration_type": None,
                "department": None,
                "operation_steps": None,
                "identification_documents": None,
                "fees": None,
                "operation_duration": None,
                "service_channel": None,
                "terms_and_conditions": None,
                "legal_regulatory": None,
            }

            self.documents.append(Document(page_content=page_content, metadata=metadata))
            added += 1

        return added

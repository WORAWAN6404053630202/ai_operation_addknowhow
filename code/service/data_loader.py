# code/service/data_loader.py
"""
Data Loader Adapter
Loads data from Google Sheets and converts to document format

PRODUCTION (RAG behavior):
- Reduce page_content noise to improve embedding/retrieval precision.
- Put high-signal procedural fields in page_content (steps/docs/fees/channel/legal/duration/conditions/topic)
- Keep everything else in metadata (still queryable downstream if needed).
"""

import re
import math
import pandas as pd
from langchain_core.documents import Document
from typing import List, Optional, Dict
from urllib.parse import urlparse, parse_qs


class DataLoader:
    """
    Flexible Data Loader
    - Load Google Sheet via CSV export URL
    - Normalize multi-line headers
    - Convert NaN/None → None
    - Build Document + metadata compatible with pipeline

    PATCH (production):
    - Robust header alias mapping for Thai government sheets where column names change often.
    - Supports:
        * exact header
        * alias list
        * "contains keyword" fallback

    PATCH (RAG quality):
    - page_content: only include high-signal procedural/legal fields to reduce embedding drift.
    - other columns remain in metadata.
    """

    def __init__(self, config):
        self.config = config
        self.documents = []
        self.departments_found = set()

        # RAG content tuning knobs (safe defaults)
        self.page_content_max_chars = int(getattr(config, "PAGE_CONTENT_MAX_CHARS", 1800) or 1800)
        self.include_research_reference_in_content = bool(
            getattr(config, "INCLUDE_RESEARCH_REF_IN_CONTENT", False)
        )

    @staticmethod
    def clean_header(name: str) -> str:
        if not isinstance(name, str):
            return name
        name = name.replace("\n", " ").replace("\r", " ")
        name = re.sub(r"\s+", " ", name)
        return name.strip()

    @staticmethod
    def _build_csv_export_url(sheet_url: str) -> str:
        """
        Build Google Sheets CSV export URL robustly.
        - Accepts URLs like:
          /edit?gid=657...#gid=657...
          /edit#gid=657...
          /edit?usp=sharing&gid=657...
        - Extracts spreadsheet base and gid safely.
        """
        u = urlparse(sheet_url)

        # Base path: https://docs.google.com/spreadsheets/d/<SPREADSHEET_ID>
        base = f"{u.scheme}://{u.netloc}{u.path}"
        base = base.split("/edit")[0].rstrip("/")

        # gid can be in query (?gid=) or fragment (#gid=)
        q = parse_qs(u.query)
        gid = (q.get("gid", [None])[0]) or (parse_qs(u.fragment).get("gid", [None])[0])

        if not gid:
            raise ValueError(
                "Google Sheet URL missing gid. "
                "Please provide a link that includes gid=... (pointing to the content sheet tab)."
            )
        print("[DataLoader] Using gid:", gid)
        return f"{base}/export?format=csv&gid={gid}"

    @staticmethod
    def to_json_safe(v):
        if v is None:
            return None

        try:
            if isinstance(v, float) and math.isnan(v):
                return None
        except Exception:
            pass

        s = str(v).strip()
        return s if s and s.lower() != "nan" else None

    def load_from_google_sheet(self, sheet_url: str, source_name: str = None):
        print("\nLoading data from Google Sheet...")
        csv_url = self._build_csv_export_url(sheet_url)
        print("CSV export URL:", csv_url)
        df = pd.read_csv(csv_url)
        df.rename(columns={c: self.clean_header(c) for c in df.columns}, inplace=True)
        # NEW: schema validation (fail fast)
        self._validate_sheet_schema(df, sheet_url)

        print("Columns detected:")
        for c in df.columns:
            print(" •", repr(c))

        print(f"\nLoaded {len(df)} rows")
        docs_added = self._process_dataframe(df, source=source_name or csv_url)

        print(f"Added {docs_added} documents")
        return docs_added

    # Header resolution 
    def _resolve_col(
        self,
        df: pd.DataFrame,
        *,
        primary: str,
        aliases: Optional[List[str]] = None,
        contains_any: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Resolve dataframe column name robustly.
        Priority:
          1) exact match: primary
          2) exact match: any alias
          3) contains-match: any token in contains_any (case-insensitive)
        Returns actual column name in df or None.
        """
        cols = list(df.columns)
        if primary in cols:
            return primary

        for a in (aliases or []):
            if a in cols:
                return a

        if contains_any:
            low_cols = [(c, str(c).lower()) for c in cols]
            for token in contains_any:
                t = str(token).lower().strip()
                if not t:
                    continue
                for orig, low in low_cols:
                    if t in low:
                        return orig

        return None

    def _build_column_map(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Build mapping from logical field -> actual df column name.
        This keeps ingestion stable even if sheet headers change slightly.
        """
        return {
            "department": self._resolve_col(
                df,
                primary="หน่วยงาน",
                aliases=[],
                contains_any=["หน่วยงาน"],
            ),
            "license_type": self._resolve_col(
                df,
                primary="ใบอนุญาต",
                aliases=[],
                contains_any=["ใบอนุญาต"],
            ),
            "operation_by_department": self._resolve_col(
                df,
                primary="การดำเนินการ ตามหน่วยงาน",
                aliases=["การดำเนินการตามหน่วยงาน", "การดำเนินการ  ตามหน่วยงาน"],
                contains_any=["การดำเนินการ", "ตามหน่วยงาน"],
            ),
            "operation_topic": self._resolve_col(
                df,
                primary="หัวข้อการดำเนินการ",
                aliases=["หัวข้อการดำเนินการย่อย", "หัวข้อการดำเนินการ (ย่อย)"],
                contains_any=["หัวข้อการดำเนินการ"],
            ),
            "registration_type": self._resolve_col(
                df,
                primary="ประเภทการจดทะเบียน",
                aliases=[],
                contains_any=["ประเภทการจดทะเบียน", "ประเภท", "จดทะเบียน"],
            ),
            "terms_and_conditions": self._resolve_col(
                df,
                primary="เงื่อนไขและหลักเกณฑ์",
                aliases=["เงื่อนไข", "หลักเกณฑ์"],
                contains_any=["เงื่อนไข", "หลักเกณฑ์"],
            ),
            "service_channel": self._resolve_col(
                df,
                primary="ช่องทางการ ให้บริการ",
                aliases=["ช่องทางการให้บริการ", "ช่องทางให้บริการ", "ช่องทาง"],
                contains_any=["ช่องทาง", "ให้บริการ"],
            ),
            "operation_steps": self._resolve_col(
                df,
                primary="ขั้นตอนการดำเนินการ",
                aliases=["ขั้นตอน"],
                contains_any=["ขั้นตอนการดำเนินการ", "ขั้นตอน"],
            ),
            "identification_documents": self._resolve_col(
                df,
                primary="เอกสาร ยืนยันตัวตน",
                aliases=["เอกสารยืนยันตัวตน", "เอกสาร ยืนยันตัวตน", "เอกสารยืนยัน"],
                contains_any=["เอกสาร", "ยืนยันตัวตน"],
            ),
            "operation_duration": self._resolve_col(
                df,
                primary="ระยะเวลา การดำเนินการ",
                aliases=["ระยะเวลา การดำเนินการ", "ระยะเวลา", "ระยะเวลาการดำเนินการ"],
                contains_any=["ระยะเวลา"],
            ),
            "fees": self._resolve_col(
                df,
                primary="ค่าธรรมเนียม",
                aliases=[],
                contains_any=["ค่าธรรมเนียม", "ค่าใช้จ่าย", "ค่าบริการ"],
            ),
            "legal_regulatory": self._resolve_col(
                df,
                primary="ข้อกำหนดทางกฎหมาย และข้อบังคับ",
                aliases=["ข้อกำหนดทางกฎหมายและข้อบังคับ", "ข้อกำหนดทางกฎหมาย", "ข้อบังคับ"],
                contains_any=["ข้อกำหนด", "กฎหมาย", "ข้อบังคับ"],
            ),
            "restaurant_ai_document": self._resolve_col(
                df,
                primary="เอกสาร AI ร้านอาหาร",
                aliases=["เอกสาร AI ร้านอาหาร", "เอกสาร AI", "เอกสาร (AI)"],
                contains_any=["เอกสาร ai", "เอกสาร (ai)"],
            ),
            "research_reference": self._resolve_col(
                df,
                primary="อ้างอิง Research",
                aliases=[
                    "อ้างอิง (Research) เอกสาร (Document)",
                    "อ้างอิง (Research)",
                    "อ้างอิง (Research) เอกสาร",
                    "อ้างอิง (Research) เอกสาร(Document)",
                ],
                contains_any=["อ้างอิง", "research", "document"],
            ),
            "answer_guideline": self._resolve_col(
                df,
                primary="แนวคำตอบ",
                aliases=["แนวทางคำตอบ", "แนวตอบ"],
                contains_any=["แนวคำตอบ", "แนวตอบ"],
            ),
        }

    def _validate_sheet_schema(self, df: pd.DataFrame, sheet_url: str) -> None:
        """
        Fail fast if the loaded sheet doesn't look like our content table.
        This prevents silently ingesting the wrong tab (wrong gid).
        """
        required_tokens = ["หน่วยงาน", "ใบอนุญาต", "ขั้นตอน", "ค่าธรรมเนียม", "ข้อกำหนด", "อ้างอิง"]

        cols = [str(c) for c in df.columns]
        joined = " | ".join(cols)

        hits = sum(1 for t in required_tokens if t in joined)

        if hits < 3:
            raise RuntimeError(
                "Loaded Google Sheet tab does not match expected content schema "
                f"(header tokens matched={hits}/{len(required_tokens)}). "
                "Likely the URL points to the wrong tab (wrong gid) or the sheet format changed.\n"
                f"Given URL: {sheet_url}\n"
                f"Detected columns: {cols}"
            )

    def _get_row_value(self, row: pd.Series, col_name: Optional[str]) -> Optional[str]:
        if not col_name:
            return None
        return self.to_json_safe(row.get(col_name))

    @staticmethod
    def _normalize_entity_type(registration_type: Optional[str]) -> Optional[str]:
        """Map raw registration_type values to normalized category (บุคคลธรรมดา / นิติบุคคล)."""
        if not registration_type:
            return None
        t = registration_type.strip()
        _JURISTIC = {
            "นิติบุคคล", "บริษัท", "บริษัทจำกัด", "บริษัทมหาชน", "บริษัทมหาชนจำกัด",
            "ห้างหุ้นส่วน", "ห้างหุ้นส่วนจำกัด", "ห้างหุ้นส่วนสามัญ",
            "ห้างหุ้นส่วนสามัญนิติบุคคล",
        }
        _INDIVIDUAL = {
            "บุคคลธรรมดา", "บุคคลธรรมดา (คนเดียว)", "บุคคลธรรมดา กิจการเจ้าของคนเดียว",
            "ประเภทบุคคลธรรมดา",
        }
        if t in _JURISTIC:
            return "นิติบุคคล"
        if t in _INDIVIDUAL:
            return "บุคคลธรรมดา"
        # Fuzzy: contains keyword
        if any(kw in t for kw in ("บริษัท", "ห้างหุ้นส่วน", "นิติบุคคล")):
            return "นิติบุคคล"
        if "บุคคลธรรมดา" in t:
            return "บุคคลธรรมดา"
        return None

    @staticmethod
    def _extract_location(operation_topic: Optional[str], registration_type: Optional[str],
                          operation_by_dept: Optional[str]) -> Optional[str]:
        """
        Extract location context from text fields.
        Returns 'กรุงเทพฯ' | 'ต่างจังหวัด' | None.

        Sources (checked in order):
          1. operation_topic (most reliable — e.g. "ร้านค้าตั้งอยู่ในเขต กรุงเทพฯ")
          2. registration_type
          3. operation_by_dept (fallback)
        """
        sources = [
            operation_topic or "",
            registration_type or "",
            operation_by_dept or "",
        ]
        combined = " ".join(sources)
        if "กรุงเทพ" in combined:
            return "กรุงเทพฯ"
        if "ต่างหวัด" in combined or "ต่างจังหวัด" in combined or "ต่างจังหวัด" in combined:
            return "ต่างจังหวัด"
        return None

    @staticmethod
    def _extract_area_size(registration_type: Optional[str],
                           operation_topic: Optional[str]) -> Optional[str]:
        """
        Extract shop area size from registration_type / operation_topic text.
        Returns 'มากกว่า 200 ตารางเมตร' | 'ไม่เกิน 200 ตารางเมตร' | None.
        """
        combined = " ".join(filter(None, [registration_type or "", operation_topic or ""]))
        if "มากกว่า 200" in combined or "เกิน 200" in combined:
            return "มากกว่า 200 ตารางเมตร"
        if "น้อยกว่า 200" in combined or "ไม่เกิน 200" in combined or "ต่ำกว่า 200" in combined:
            return "ไม่เกิน 200 ตารางเมตร"
        return None

    @staticmethod
    def _extract_entity_from_topic(operation_topic: Optional[str],
                                   registration_type: Optional[str]) -> Optional[str]:
        """
        Additional entity-type extraction from topic/reg fields
        to supplement the registration_type-based extraction.
        Covers cases like topic='บุคคลธรรมดา' or topic='นิติบุคคล'.
        """
        combined = " ".join(filter(None, [operation_topic or "", registration_type or ""]))
        if any(kw in combined for kw in ("บริษัท", "ห้างหุ้นส่วน", "นิติบุคคล")):
            return "นิติบุคคล"
        if "บุคคลธรรมดา" in combined:
            return "บุคคลธรรมดา"
        return None

    # RAG: content shaping
    def _join_nonempty(self, parts: List[str]) -> str:
        parts2 = [p.strip() for p in (parts or []) if p and str(p).strip()]
        return "\n".join(parts2).strip()

    def _build_page_content(self, md: Dict[str, Optional[str]]) -> str:
        """
        Build high-signal embedding text optimised for multilingual-e5-large.

        Design principles (senior RAG):
        1. Lead with the most disambiguating signals: license, operation, location, area, entity
        2. Include all actionable answer fields so similarity search surfaces the right doc
        3. Include answer_guideline + conditions for FAQ/knowledge queries
        4. Cap total length to stay within model token budget (~512 tokens ≈ 1800 Thai chars)

        For e5 models the text is prepended with "passage: " by the embedding layer,
        so we do NOT add it here.
        """
        # Context header: high-signal disambiguators
        head_parts = []
        if md.get("license_type"):
            head_parts.append(f"ใบอนุญาต/ทะเบียน: {md['license_type']}")
        if md.get("operation_by_department"):
            head_parts.append(f"การดำเนินการ: {md['operation_by_department']}")
        if md.get("location"):
            head_parts.append(f"พื้นที่: {md['location']}")
        if md.get("area_size"):
            head_parts.append(f"ขนาดพื้นที่ร้าน: {md['area_size']}")
        if md.get("entity_type_normalized"):
            head_parts.append(f"ประเภทผู้ประกอบการ: {md['entity_type_normalized']}")
        elif md.get("registration_type"):
            head_parts.append(f"ประเภทการจดทะเบียน: {md['registration_type']}")
        if md.get("operation_topic"):
            head_parts.append(f"หัวข้อ: {md['operation_topic']}")
        if md.get("department"):
            head_parts.append(f"หน่วยงาน: {md['department']}")

        head = self._join_nonempty(head_parts)

        # Answer body: all actionable fields
        # NOTE: legal_regulatory is placed early (position 2) to ensure penalty/law keywords
        # are always embedded in the vector — they are short but critical for retrieval.
        # identification_documents is long and can tolerate being partially truncated.
        body_parts = []
        # NOTE: All three long fields are capped for embedding budget only.
        # Full values are always available in metadata for the LLM to read.
        # Caps ensure every field's keywords appear in the vector regardless of doc length.
        if md.get("operation_steps"):
            body_parts.append(f"ขั้นตอนการดำเนินการ:\n{md['operation_steps'][:600]}")
        if md.get("legal_regulatory"):
            body_parts.append(f"ข้อกำหนดกฎหมาย/บทลงโทษ:\n{md['legal_regulatory'][:300]}")
        if md.get("identification_documents"):
            body_parts.append(f"เอกสารที่ต้องใช้:\n{md['identification_documents'][:500]}")
        if md.get("fees"):
            body_parts.append(f"ค่าธรรมเนียม:\n{md['fees']}")
        if md.get("operation_duration"):
            body_parts.append(f"ระยะเวลาดำเนินการ: {md['operation_duration']}")
        if md.get("service_channel"):
            body_parts.append(f"ช่องทางยื่นคำขอ/ติดต่อ:\n{md['service_channel'][:150]}")
        if md.get("terms_and_conditions"):
            body_parts.append(f"เงื่อนไขและหลักเกณฑ์:\n{md['terms_and_conditions']}")
        if md.get("answer_guideline"):
            body_parts.append(f"แนวคำตอบ:\n{md['answer_guideline']}")
        if md.get("restaurant_ai_document"):
            body_parts.append(f"เอกสาร/ฟอร์ม: {md['restaurant_ai_document']}")

        body = self._join_nonempty(body_parts)

        extra = ""
        if self.include_research_reference_in_content and md.get("research_reference"):
            extra = f"อ้างอิง: {md.get('research_reference')}"

        text = self._join_nonempty([head, body, extra])

        # Final clamp to avoid huge embeddings (e5-large: 512 tokens ≈ ~2000 Thai chars)
        max_chars = self.page_content_max_chars or 2000
        if len(text) > max_chars:
            text = text[:max_chars].rstrip()

        return text.strip()

    def _process_dataframe(self, df: pd.DataFrame, source: str) -> int:
        docs_before = len(self.documents)

        colmap = self._build_column_map(df)

        for idx, row in df.iterrows():
            dept = self._get_row_value(row, colmap.get("department"))
            if dept:
                self.departments_found.add(dept)

            reg_type = self._get_row_value(row, colmap.get("registration_type"))
            op_topic = self._get_row_value(row, colmap.get("operation_topic"))
            op_by_dept = self._get_row_value(row, colmap.get("operation_by_department"))

            # Split compound registration_type values joined with " / "
            # e.g. "ห้างหุ้นส่วนจำกัด / ห้างหุ้นส่วนสามัญ" → two separate docs so that
            # Chroma metadata filters on exact registration_type values work correctly.
            if reg_type and " / " in reg_type:
                reg_type_values = [v.strip() for v in reg_type.split(" / ") if v.strip()]
            else:
                reg_type_values = [reg_type]

            for reg_type_single in reg_type_values:
                # Derived metadata: entity, location, area_size
                entity_from_reg = self._normalize_entity_type(reg_type_single)
                entity_from_topic = self._extract_entity_from_topic(op_topic, reg_type_single)
                # Prefer reg-based entity; fill with topic-based if missing
                entity_normalized = entity_from_reg or entity_from_topic

                location = self._extract_location(op_topic, reg_type_single, op_by_dept)
                area_size = self._extract_area_size(reg_type_single, op_topic)

                metadata = {
                    "row_id": int(idx),
                    "department": dept,
                    "license_type": self._get_row_value(row, colmap.get("license_type")),
                    "operation_by_department": op_by_dept,
                    "operation_topic": op_topic,
                    "registration_type": reg_type_single,
                    "entity_type_normalized": entity_normalized,
                    # Derived metadata: entity, location, area_size
                    "location": location,          # 'กรุงเทพฯ' | 'ต่างจังหวัด' | None
                    "area_size": area_size,        # 'มากกว่า 200 ตารางเมตร' | 'ไม่เกิน 200 ตารางเมตร' | None
                    # Answer fields
                    "terms_and_conditions": self._get_row_value(row, colmap.get("terms_and_conditions")),
                    "service_channel": self._get_row_value(row, colmap.get("service_channel")),
                    "operation_steps": self._get_row_value(row, colmap.get("operation_steps")),
                    "identification_documents": self._get_row_value(row, colmap.get("identification_documents")),
                    "operation_duration": self._get_row_value(row, colmap.get("operation_duration")),
                    "fees": self._get_row_value(row, colmap.get("fees")),
                    "legal_regulatory": self._get_row_value(row, colmap.get("legal_regulatory")),
                    "restaurant_ai_document": self._get_row_value(row, colmap.get("restaurant_ai_document")),
                    "research_reference": self._get_row_value(row, colmap.get("research_reference")),
                    "answer_guideline": self._get_row_value(row, colmap.get("answer_guideline")),
                    "source": source,
                }

                # Build page_content (high-signal embedding text)
                page_content = self._build_page_content(metadata)

                # Skip rows with no procedural content — indexing boilerplate degrades search quality
                if not page_content:
                    topic = metadata.get("operation_topic") or metadata.get("license_type") or f"row {idx}"
                    print(f"[DataLoader] WARNING: Skipping empty row (no content): {topic}")
                    continue

                self.documents.append(Document(page_content=page_content, metadata=metadata))

        return len(self.documents) - docs_before

    def get_statistics(self):
        print("\n--- Data Statistics ---")
        print(f"Total documents: {len(self.documents)}")
        print(f"Departments: {len(self.departments_found)}")

        for dept in sorted(self.departments_found):
            count = sum(1 for d in self.documents if d.metadata.get("department") == dept)
            print(f" • {dept}: {count} docs")

        return {
            "total_docs": len(self.documents),
            "departments": list(self.departments_found),
        }
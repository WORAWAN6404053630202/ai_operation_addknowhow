# code/service/pdf_direct_ingest.py
"""Converts PDF-pipeline topic output (identify_knowhow_topics /
identify_license_topics, optionally draft_fields_from_pages) directly into
LangChain Documents ready for local_vector_store.add_documents() — the
no-human-review path (feature/pdf-ingestion, 2026-09) that replaces the
review-queue → candidate-matching → Sheet-write-back flow for PDF content.

Every document built here carries 3 fields no Sheet-sourced document has,
so PDF-derived content is always identifiable and auditable after the fact:
  - ingested_via="pdf_auto"
  - source_file=<original PDF filename>
  - ingested_at=<unix timestamp>
querying `where={"ingested_via": "pdf_auto"}` on the Chroma collection is
the whole "audit log" this path relies on instead of a review UI.

Schema choices below intentionally MIRROR the real Sheet-loader schemas
(service/data_loader.py's DataLoader, service/data_loader_general.py's
GeneralDataLoader) field-for-field rather than inventing a new shape —
persona_supervisor.py's retrieval logic (exact-match chapter-fetch, slot
discovery, entity-injection, 2-pass retrieval) keys off those exact field
names and values, so a PDF chunk that doesn't match the shape a human
reviewer would have produced by hand simply won't participate in that
logic. know_how-shaped content reuses topic_group/data_type="business_guide"
so it lands in the SAME code paths the 2 real know-how tabs already use
(see _merge_cross_chapter_docs_for_nonreg and friends — they check
data_type in ("marketing", "business_guide") specifically, not "know_how")."""

from __future__ import annotations

import time
from typing import Any, Optional

from langchain_core.documents import Document

import conf
from service.data_loader import DataLoader
from utils.pdf_text_chunking import split_into_subchunks

_dl = DataLoader(config={})  # cheap, stateless enough to reuse across calls — only used for its helper methods/_build_page_content


def _pdf_tags(source_file: str) -> dict:
    return {
        "ingested_via": "pdf_auto",
        "source_file": source_file,
        "ingested_at": time.time(),
    }


def _build_pdf_knowhow_page_content(
    document_title: str, main_topic: str, sub_topic: str, body: str,
    chunk_index: int, chunk_total: int,
) -> str:
    """Untruncated page_content builder for PDF book content — deliberately
    NOT reusing DataLoader._build_page_content(), whose ~2,000-2,500 char
    clamp is sized for short Sheet-row content and was the root cause of the
    truncation bug this module fixes (see module docstring). Repeats a
    context header on every sub-chunk (book/chapter/sub-topic + part N/M) so
    each chunk stays independently interpretable if retrieved on its own."""
    header_lines = [ln for ln in [
        f"หนังสือ: {document_title}" if document_title else "",
        f"หมวด: {main_topic}" if main_topic else "",
        f"หัวข้อย่อย: {sub_topic}" if sub_topic else "",
        f"ส่วนที่ {chunk_index}/{chunk_total}" if chunk_total > 1 else "",
    ] if ln]
    header = "\n".join(header_lines)
    return f"{header}\n\n{body}".strip() if header else body


def build_knowhow_documents(topics: list[dict[str, Any]], source_file: str) -> list[Document]:
    """topics: KnowhowTopicBounds-shaped dicts (main_topic/sub_topic/summary/
    category/full_text/document_title/source_type) — output of
    pdf_knowhow_drafting.identify_knowhow_topics() + full_text sliced by the
    caller. Tags data_type="business_guide" to match the real, live know-how
    tabs' schema (see data_loader_general.py's _process_dataframe) — NOT
    "know_how", which the currently-inactive Sheet-based know-how loader
    uses but no retrieval logic actually checks for.

    2026-09 sub-chunking fix: a topic's full_text is split (via
    utils.pdf_text_chunking.split_into_subchunks) into one Document per
    sub-chunk whenever it exceeds conf.PDF_KNOWHOW_CHUNK_MAX_CHARS, instead
    of being silently truncated. All sub-chunks of one topic share
    main_topic/sub_topic (so persona_supervisor.py's chapter-fetch/
    cross-chapter-merge groups them together automatically) and carry
    chunk_index/chunk_total metadata identifying their place in the topic.
    The common case (topic already fits under the cap) still produces
    exactly one Document, same as before."""
    docs: list[Document] = []
    max_chars = int(getattr(conf, "PDF_KNOWHOW_CHUNK_MAX_CHARS", 24000) or 24000)
    for t in topics:
        main_topic = (t.get("main_topic") or "").strip()
        sub_topic = (t.get("sub_topic") or "").strip()
        full_text = (t.get("full_text") or "").strip()
        if not full_text:
            continue
        document_title = t.get("document_title") or source_file
        page_range = (t.get("page_range") or "").strip()
        research_reference = f"{source_file} (หน้า {page_range})" if page_range else f"เอกสาร: {source_file}"

        subchunks = split_into_subchunks(full_text, max_chars)
        chunk_total = len(subchunks)
        for idx, chunk_text in enumerate(subchunks, start=1):
            page_content = _build_pdf_knowhow_page_content(
                document_title, main_topic, sub_topic, chunk_text, idx, chunk_total,
            )
            metadata = {
                "data_type": "business_guide",
                "topic_group": "คู่มือเปิดร้าน",
                "source": "pdf_knowhow",
                "operation_topic": sub_topic or main_topic,
                "main_topic": main_topic,
                "sub_topic": sub_topic,
                "source_book": document_title,
                "book_name": document_title,
                "content_type": t.get("category") or "",
                # answer_guideline holds THIS sub-chunk's own content (not
                # the original full full_text) — page_content and the LLM
                # context field now always match exactly, unlike the old
                # code where metadata held the full text but page_content
                # (used for embedding) was silently truncated.
                "answer_guideline": chunk_text,
                "chunk_index": idx,
                "chunk_total": chunk_total,
                "research_reference": research_reference,
                # Regulatory fields explicitly None — prevents slot-fill from
                # mistaking this for a regulatory row (matches GeneralDataLoader).
                "license_type": None, "entity_type_normalized": None, "location": None,
                "area_size": None, "registration_type": None, "department": None,
                "operation_steps": None, "identification_documents": None, "fees": None,
                "operation_duration": None, "service_channel": None, "terms_and_conditions": None,
                "legal_regulatory": None,
                **_pdf_tags(source_file),
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


# draft_fields_from_pages() returns a dict keyed by the REAL Sheet header
# text (e.g. "หน่วยงาน", "ใบอนุญาต" — see DRAFTABLE_FIELDS in
# pdf_field_drafting.py), not these internal snake_case names — this maps
# one to the other so callers (sqs_consumer.py) can pass its raw output
# straight through instead of remapping it themselves at every call site.
_DRAFTED_FIELD_KEY_MAP = {
    "department": "หน่วยงาน",
    "license_type": "ใบอนุญาต",
    "operation_by_department": "การดำเนินการ ตามหน่วยงาน",
    "operation_topic": "หัวข้อการดำเนินการ",
    "registration_type": "ประเภทการจดทะเบียน",
    "terms_and_conditions": "เงื่อนไขและหลักเกณฑ์",
    "service_channel": "ช่องทางการ ให้บริการ",
    "operation_steps": "ขั้นตอนการดำเนินการ",
    "identification_documents": "เอกสาร ยืนยันตัวตน",
    "operation_duration": "ระยะเวลา การดำเนินการ",
    "fees": "ค่าธรรมเนียม",
    "legal_regulatory": "ข้อกำหนดทางกฎหมาย และข้อบังคับ",
    "answer_guideline": "แนวคำตอบ",
}


def merge_drafted_fields(topic: dict[str, Any], drafted: dict[str, Any]) -> dict[str, Any]:
    """topic: a LicenseTopicBounds dict (department/license_type/page_ranges).
    drafted: draft_fields_from_pages()'s raw, Thai-header-keyed output for
    that same topic's page range. Returns one dict in the internal-key shape
    build_regulatory_documents_full() expects — department/license_type come
    from `topic` (already known, no need to re-derive), everything else from
    `drafted` via _DRAFTED_FIELD_KEY_MAP."""
    merged = {"department": topic.get("department", ""), "license_type": topic.get("license_type", "")}
    for internal_key, thai_header in _DRAFTED_FIELD_KEY_MAP.items():
        if internal_key in ("department", "license_type"):
            continue  # already set from topic above — draft_fields_from_pages doesn't draft these anyway
        merged[internal_key] = drafted.get(thai_header, "")
    return merged


def build_regulatory_documents_full(
    topics_with_fields: list[dict[str, Any]], source_file: str,
) -> list[Document]:
    """"Schema A" for the ขั้น 3 test — full 13-field regulatory shape.
    topics_with_fields: each dict = a LicenseTopicBounds (department/
    license_type) MERGED with draft_fields_from_pages()'s 13-field output
    for that topic's page range (caller's responsibility to merge — this
    function only shapes the result into a Document). Reuses
    DataLoader._assign_topic_group/_normalize_entity_type/_build_page_content
    so this is byte-for-byte the same derivation a human-reviewed Sheet row
    would get at ingest time."""
    docs: list[Document] = []
    for t in topics_with_fields:
        department = (t.get("department") or "").strip()
        license_type = (t.get("license_type") or "").strip()
        registration_type = (t.get("registration_type") or "").strip()
        entity_type_normalized = DataLoader._normalize_entity_type(registration_type)
        topic_group = DataLoader._assign_topic_group(license_type)

        # location/area_size (2026-10): previously hardcoded None — DataLoader's
        # extractors are pure text-sniffing functions with no Sheet dependency,
        # reused here exactly as _normalize_entity_type/_assign_topic_group
        # already are. Sniffed against the drafted operation_topic PLUS the raw
        # source markdown (_raw_text, threaded in by sqs_consumer.py) when
        # available — the LLM-drafted operation_topic may paraphrase away a
        # literal "กรุงเทพ"/"มากกว่า 200 ตารางเมตร" that the source PDF states
        # verbatim, so the raw text gives the sniffer a second chance to match.
        operation_topic = t.get("operation_topic") or ""
        location_text = " ".join(filter(None, [operation_topic, t.get("_raw_text") or ""]))
        location = DataLoader._extract_location(location_text, registration_type, t.get("operation_by_department"))
        area_size = DataLoader._extract_area_size(registration_type, location_text)

        page_range = (t.get("page_range") or "").strip()
        research_reference = f"{source_file} (หน้า {page_range})" if page_range else f"เอกสาร: {source_file}"

        metadata = {
            "data_type": "regulatory",
            "department": department or None,
            "license_type": license_type or None,
            "topic_group": topic_group,
            "operation_by_department": t.get("operation_by_department") or None,
            "operation_topic": t.get("operation_topic") or None,
            "registration_type": registration_type or None,
            "entity_type_normalized": entity_type_normalized,
            "location": location,
            "area_size": area_size,
            "terms_and_conditions": t.get("terms_and_conditions") or None,
            "service_channel": t.get("service_channel") or None,
            "operation_steps": t.get("operation_steps") or None,
            "identification_documents": t.get("identification_documents") or None,
            "operation_duration": t.get("operation_duration") or None,
            "fees": t.get("fees") or None,
            "legal_regulatory": t.get("legal_regulatory") or None,
            "restaurant_ai_document": None,
            "research_reference": research_reference,
            "answer_guideline": t.get("answer_guideline") or None,
            "source": "pdf_regulatory",
            **_pdf_tags(source_file),
        }
        page_content = _dl._build_page_content(metadata)
        if not page_content or len(page_content) < 20:
            continue
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def build_regulatory_documents_simple(
    topics: list[dict[str, Any]], source_file: str,
) -> list[Document]:
    """"Schema B" for the ขั้น 3 test — simple main_topic/sub_topic shape
    (department -> main_topic, license_type -> sub_topic), NO 13-field
    drafting, NO entity_type_normalized/topic_group. Tagged data_type=
    "business_guide" like know-how content, so it's a fair comparison of
    "same simple shape, does it work for regulatory content too" rather
    than a shape with no real-world precedent at all."""
    docs: list[Document] = []
    for t in topics:
        department = (t.get("department") or "").strip()
        license_type = (t.get("license_type") or "").strip()
        full_text = (t.get("full_text") or "").strip()
        if not full_text:
            continue
        metadata = {
            "data_type": "business_guide",
            "topic_group": "คู่มือเปิดร้าน",
            "source": "pdf_regulatory_simple",
            "operation_topic": license_type or department,
            "main_topic": department,
            "sub_topic": license_type,
            "source_book": source_file,
            "book_name": source_file,
            "content_type": "",
            "answer_guideline": full_text,
            "research_reference": f"เอกสาร: {source_file}",
            "license_type": None, "entity_type_normalized": None, "location": None,
            "area_size": None, "registration_type": None, "department": None,
            "operation_steps": None, "identification_documents": None, "fees": None,
            "operation_duration": None, "service_channel": None, "terms_and_conditions": None,
            "legal_regulatory": None,
            **_pdf_tags(source_file),
        }
        page_content = f"{department} — {license_type}\n{full_text}"[:2000]
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs

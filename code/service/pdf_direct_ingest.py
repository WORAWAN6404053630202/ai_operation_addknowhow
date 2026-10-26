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
    """topic: identify_knowhow_topics()'s output for one topic (main_topic/
    sub_topic/summary/category/document_title/source_type/page_ranges) —
    the universal topic-splitter now used for every document (2026-10, PDF
    ingestion schema unification), which never produces department/
    license_type itself (unlike the old, now-removed identify_license_topics).
    drafted: draft_fields_from_pages()'s raw, Thai-header-keyed output for
    that same topic's page range, now run for EVERY topic regardless of
    shape — department/license_type come from `drafted` like every other
    field now (via _DRAFTED_FIELD_KEY_MAP), not pre-supplied by topic-
    splitting the way the old LicenseTopicBounds shape did.

    Returns one dict carrying both the topic-split identity AND the
    drafted structured fields — ready for build_merged_documents, which
    derives data_type from whether department/license_type ended up
    non-empty after drafting."""
    merged = {
        "main_topic": topic.get("main_topic", ""),
        "sub_topic": topic.get("sub_topic", ""),
        "summary": topic.get("summary", ""),
        "category": topic.get("category", ""),
        "document_title": topic.get("document_title", ""),
        "source_type": topic.get("source_type", ""),
    }
    for internal_key, thai_header in _DRAFTED_FIELD_KEY_MAP.items():
        merged[internal_key] = drafted.get(thai_header, "")
    return merged


# 2026-10 (PDF ingestion schema unification): per-field embedding-text caps,
# ported verbatim from DataLoader._build_page_content's own (live-measured,
# see that method's comment for the real-corpus-distribution audit behind
# each number) — reused here rather than calling that method directly,
# because its OWN final clamp (self.page_content_max_chars or 2000) is sized
# for terse Sheet rows and is the exact truncation
# _build_pdf_merged_page_content exists to avoid for long PDF-book content.
_MERGED_FIELD_CAPS: dict[str, int] = {
    "operation_steps": 1800,
    "legal_regulatory": 2000,
    "terms_and_conditions": 2300,
    "identification_documents": 2100,
    "fees": 900,
    "operation_duration": 300,
    "service_channel": 600,
}
_MERGED_FIELD_LABELS: dict[str, str] = {
    "operation_steps": "ขั้นตอนการดำเนินการ",
    "legal_regulatory": "ข้อกำหนดกฎหมาย/บทลงโทษ",
    "terms_and_conditions": "เงื่อนไขและหลักเกณฑ์",
    "identification_documents": "เอกสารที่ต้องใช้",
    "fees": "ค่าธรรมเนียม",
    "operation_duration": "ระยะเวลาดำเนินการ",
    "service_channel": "ช่องทางยื่นคำขอ/ติดต่อ",
}

# 2026-10 fix (live-reproduced on "ยื่น_เพิ่ม_ชำระ_ยกเลิก ภาษีป้าย.pdf"): the
# per-field caps above (summing to ~9,800 chars worst case) plus the body
# cap (conf.PDF_KNOWHOW_CHUNK_MAX_CHARS, 24,000 by default) were each sized
# to stay under baai/bge-m3's 8192-token ceiling on their OWN, but nothing
# capped their SUM — a topic with every field populated near its cap AND a
# near-max body pushed one real document's assembled page_content to ~34,000
# chars, which OpenRouter's embed call rejected outright ("maximum context
# length is 8192 tokens... prompt contains at least 8193 input tokens"),
# losing the whole batch write (add_documents embeds in batches of
# conf's chunk_size, so one oversized document fails every document sharing
# its batch, not just itself).
#
# local_vector_store.py's own cost-estimate comment flags "~4 chars/token,
# not exact for Thai" as a rough heuristic — this incident's first fix
# (since corrected) trusted that heuristic and picked a 21,000-char cap,
# assuming ~3 chars/token with margin. Still live-reproduced the EXACT same
# "8193 input tokens" failure afterward, on a SINGLE already-capped
# document (confirmed via a direct per-document length dump right before
# the embed call — every document measured ≤21,000 chars, yet the write
# still 400'd). A direct tiktoken (cl100k_base) measurement on
# representative Thai legal/technical prose from this exact document
# showed ~1.0-1.2 chars/token, not ~4 — confirming the "~4 chars/token"
# heuristic, while reasonable for English, is badly wrong for Thai, whose
# script isn't well-covered by most BPE vocabularies and falls back to
# much shorter sub-word/byte-level tokens. bge-m3's own real tokenizer
# isn't necessarily cl100k_base, but the live failures (~2.5 chars/token
# back-calculated from the real 400 errors) confirm the same order-of-
# magnitude problem, not a cl100k-specific quirk.
#
# 7,000 chars (not 7,000 TOKENS this time) — conservative even against the
# worst measured ratio (~1 char/token would still land near ~7,000 tokens,
# a real ~15% margin below the 8,192 ceiling; the live system's own better
# ~2.5 chars/token ratio leaves far more).
#
# Only the body gets trimmed when the total runs over (see
# _build_pdf_merged_page_content below) — header/fields_block stay intact
# since they're short and carry the highest information density per
# character. This clamp affects ONLY the embedded page_content; the full,
# untrimmed chunk text is still stored verbatim in this same Document's
# `answer_guideline` metadata value (set from the same chunk_text by
# build_merged_documents), so nothing a human or the LLM-facing context
# builder reads is actually lost — only what the embedding VECTOR
# "sees" is bounded. (local_vector_store.py's add_documents also retries
# once with chunk_size=1 on this same error class, for the SEPARATE
# failure mode where several individually-safe documents land in one
# embedding batch and their SUM exceeds the ceiling — the two fixes guard
# different layers of the same underlying constraint.)
_MAX_MERGED_PAGE_CONTENT_CHARS = 7000


def _build_pdf_merged_page_content(
    document_title: str, main_topic: str, sub_topic: str, department: str, license_type: str,
    structured_fields: dict[str, Any], body: str, chunk_index: int, chunk_total: int,
) -> str:
    """page_content builder for the merged pipeline — header (book/
    department-license/topic identity + part N/M) + a compact structured-
    fields block (each field capped individually via _MERGED_FIELD_CAPS,
    same reasoning as DataLoader._build_page_content: every field having
    its OWN cap, not just an overall clamp, is what guarantees each one's
    keywords have a chance to appear in the embedding vector regardless of
    how much content precedes it) + this chunk's own body text — NOT
    passed through DataLoader._build_page_content's tight overall
    ~2000-char clamp, but IS passed through _MAX_MERGED_PAGE_CONTENT_CHARS
    (see that constant's comment) as a last-resort safety net: only the
    body is trimmed if the assembled total would exceed it, since
    header+fields_block are short and carry the highest information
    density per character. This never loses data the LLM-facing side
    reads — build_merged_documents stores the full, untrimmed chunk text
    separately in this same Document's answer_guideline metadata."""
    header_lines = [ln for ln in [
        f"หนังสือ: {document_title}" if document_title else "",
        f"หน่วยงาน: {department}" if department else "",
        f"ใบอนุญาต/ทะเบียน: {license_type}" if license_type else "",
        f"หมวด: {main_topic}" if main_topic else "",
        f"หัวข้อย่อย: {sub_topic}" if sub_topic else "",
        f"ส่วนที่ {chunk_index}/{chunk_total}" if chunk_total > 1 else "",
    ] if ln]
    header = "\n".join(header_lines)

    field_blocks = []
    for key, cap in _MERGED_FIELD_CAPS.items():
        value = str(structured_fields.get(key) or "").strip()
        if value:
            field_blocks.append(f"{_MERGED_FIELD_LABELS[key]}:\n{value[:cap]}")
    fields_block = "\n\n".join(field_blocks)

    prefix = "\n\n".join(p for p in [header, fields_block] if p)
    if len(prefix) >= _MAX_MERGED_PAGE_CONTENT_CHARS:
        # Last-resort branch, expected to be rare: the structured-fields
        # block alone (every field populated near its own _MERGED_FIELD_CAPS
        # ceiling at once) already meets or exceeds the overall budget —
        # trimming only the body (the normal path below) would still leave
        # the assembled total over the limit. Trim the prefix itself instead
        # so the guarantee ("page_content never exceeds
        # _MAX_MERGED_PAGE_CONTENT_CHARS") holds unconditionally, not just
        # in the common case.
        prefix = prefix[:_MAX_MERGED_PAGE_CONTENT_CHARS]
        trimmed_body = ""
    else:
        separator_len = 2 if prefix and body else 0  # the "\n\n" that will join prefix to body below
        body_budget = max(0, _MAX_MERGED_PAGE_CONTENT_CHARS - len(prefix) - separator_len)
        trimmed_body = body[:body_budget]

    parts = [p for p in [prefix, trimmed_body] if p]
    return "\n\n".join(parts).strip()


def build_merged_documents(topics_with_fields: list[dict[str, Any]], source_file: str) -> list[Document]:
    """Replaces build_knowhow_documents + build_regulatory_documents_full
    (2026-10, PDF ingestion schema unification) — every topic now goes
    through this ONE builder regardless of whether it ended up carrying
    department/license_type or not.

    topics_with_fields: each dict = topic-split output (main_topic/
    sub_topic/summary/category/document_title/source_type/page_range/
    full_text, from identify_knowhow_topics — now the universal splitter)
    MERGED with draft_fields_from_pages()'s output for that topic (via
    merge_drafted_fields — department/license_type/operation_by_department/
    registration_type/terms_and_conditions/service_channel/operation_steps/
    identification_documents/operation_duration/fees/legal_regulatory/
    answer_guideline). Caller's responsibility to merge; this function only
    shapes the result into Document(s).

    data_type is DERIVED, not pre-classified: "regulatory" if department OR
    license_type drafted non-empty, else "business_guide" — this is what
    fixes the misclassification bug class this whole merge exists for (a
    law with real penalties but no application-procedure phrasing, e.g.
    "กำหนดเวลาห้ามขายเครื่องดื่มแอลกอฮอล์.pdf", now correctly resolves to
    regulatory once its department/legal_regulatory get drafted — nothing
    about it needs to have been pre-classified as a "procedure").

    Chunking (2026-10 decision): every topic's full_text is split via
    split_into_subchunks exactly as build_knowhow_documents already did —
    regulatory-shaped topics are no longer guaranteed to be exactly one
    Document. The full structured-field set (department, fees,
    legal_regulatory, etc.) is attached to EVERY sub-chunk, not just the
    first — a user asking "ค่าธรรมเนียมเท่าไหร่" must find the fee
    regardless of which chunk's embedding wins the similarity search; these
    are LLM-drafted summaries (bounded by the drafting prompt's own token
    budget), not raw OCR, so replication is cheap. `answer_guideline`
    metadata keeps meaning exactly what it already means for know-how
    content — THIS sub-chunk's own raw text (needed byte-for-byte by the
    chunk_total/chunk_index reassembly logic in persona_practical.py/
    persona_academic.py) — rather than draft_fields_from_pages()'s own
    drafted "answer_guideline" prose summary, which is dropped: once every
    topic carries its real full_text in chunked form, a short LLM summary
    of the same content is strictly redundant, and keeping both under one
    metadata key name would silently break reassembly for regulatory-shaped
    topics (today's build_regulatory_documents_full never chunked, so this
    collision never existed before)."""
    docs: list[Document] = []
    max_chars = int(getattr(conf, "PDF_KNOWHOW_CHUNK_MAX_CHARS", 24000) or 24000)
    for t in topics_with_fields:
        main_topic = (t.get("main_topic") or "").strip()
        sub_topic = (t.get("sub_topic") or "").strip()
        full_text = (t.get("full_text") or "").strip()
        if not full_text:
            continue

        department = (t.get("department") or "").strip()
        license_type = (t.get("license_type") or "").strip()
        is_regulatory = bool(department or license_type)
        data_type = "regulatory" if is_regulatory else "business_guide"

        registration_type = (t.get("registration_type") or "").strip()
        # entity_type_normalized/location/area_size/topic_group: pure text-
        # sniffing helpers with no Sheet dependency, called unconditionally
        # (cheap; each already tolerates empty input and returns None/""
        # rather than raising) — naturally come back empty for non-
        # regulatory topics with no department/registration_type to sniff.
        entity_type_normalized = DataLoader._normalize_entity_type(registration_type)
        topic_group = DataLoader._assign_topic_group(license_type) or "คู่มือเปิดร้าน"
        operation_topic = t.get("operation_topic") or ""
        # Sniffed against the drafted operation_topic PLUS the raw source
        # markdown (_raw_text, threaded in by sqs_consumer.py) when
        # available — the LLM-drafted operation_topic may paraphrase away a
        # literal "กรุงเทพ"/"มากกว่า 200 ตารางเมตร" the source PDF states
        # verbatim, so the raw text gives the sniffer a second chance.
        location_text = " ".join(filter(None, [operation_topic, t.get("_raw_text") or ""]))
        location = DataLoader._extract_location(location_text, registration_type, t.get("operation_by_department"))
        area_size = DataLoader._extract_area_size(registration_type, location_text)

        document_title = t.get("document_title") or source_file
        page_range = (t.get("page_range") or "").strip()
        research_reference = f"{source_file} (หน้า {page_range})" if page_range else f"เอกสาร: {source_file}"

        subchunks = split_into_subchunks(full_text, max_chars)
        chunk_total = len(subchunks)
        for idx, chunk_text in enumerate(subchunks, start=1):
            page_content = _build_pdf_merged_page_content(
                document_title=document_title, main_topic=main_topic, sub_topic=sub_topic,
                department=department, license_type=license_type,
                structured_fields=t, body=chunk_text, chunk_index=idx, chunk_total=chunk_total,
            )
            metadata = {
                "data_type": data_type,
                "topic_group": topic_group,
                "source": "pdf_regulatory" if is_regulatory else "pdf_knowhow",
                "operation_topic": operation_topic or sub_topic or main_topic or None,
                "main_topic": main_topic or None,
                "sub_topic": sub_topic or None,
                "summary": t.get("summary") or None,
                "category": t.get("category") or None,
                "source_book": document_title,
                "book_name": document_title,
                "content_type": t.get("category") or "",
                "department": department or None,
                "license_type": license_type or None,
                "operation_by_department": t.get("operation_by_department") or None,
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
                # This sub-chunk's own raw text — see this function's
                # docstring on why the drafted "answer_guideline" prose
                # summary is dropped rather than stored alongside it.
                "answer_guideline": chunk_text,
                "chunk_index": idx,
                "chunk_total": chunk_total,
                "research_reference": research_reference,
                **_pdf_tags(source_file),
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs

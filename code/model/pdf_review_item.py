# code/model/pdf_review_item.py
"""Schema for one PDF review-queue item (feature/pdf-ingestion) — one record per
uploaded PDF, holding the per-page dual-model extraction + validation results
from pdf_dual_extraction.py, plus the human review decision once made.

Fields for later pipeline stages (candidate_matches, llm_drafted_fields,
regex_draft_ref) are included now so the schema doesn't need to change shape
when those stages are built — they just stay None until then. Everything here
follows the additive-only design agreed for the whole feature: this model
records a decision, it never deletes or overwrites anything in the real
Google Sheet data itself."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class PageExtractionRecord(BaseModel):
    page_num: int
    typhoon_markdown: str
    claude_markdown: str
    # Serialized ValidationFlag dicts (category/severity/message/details) — kept as
    # plain dicts rather than importing the ValidationFlag dataclass here, so this
    # model doesn't need to know about pdf_extraction_validation.py's internals.
    flags: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return len(self.flags) > 0

    @property
    def high_severity_flag_count(self) -> int:
        return sum(1 for f in self.flags if f.get("severity") == "high")


DecisionType = Literal["new", "duplicate", "update", "new_category"]
ReviewStatus = Literal["pending", "approved", "rejected"]


class ReviewItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    filename: str
    s3_raw_pdf_path: str = ""
    uploaded_at: float = Field(default_factory=time.time)
    extraction_completed_at: Optional[float] = None

    # Added 2026-09 for the admin UI's per-document cost/time display.
    # extraction_started_at is captured where extraction actually begins
    # (Lambda invocation start, or extract_full_document()'s start on the
    # EC2 handoff path) — NOT when this ReviewItem object is constructed,
    # so processing_duration_seconds below reflects real wall-clock time
    # from "file appeared" to "review item ready", not just the cheap
    # build phase. None on items processed before this field existed.
    extraction_started_at: Optional[float] = None
    # Sum of every log_llm_cost() call's returned cost across the WHOLE
    # pipeline for this document (extraction/vision-verify + content-shape
    # classification + topic drafting + candidate matching) — see
    # utils/llm_cost_logging.py's CostAccumulator. If a document splits into
    # multiple ReviewItems (mixed-shape or multi-topic), the same
    # document-level total is attached to every resulting item — the cost
    # isn't meaningfully separable per sub-item since e.g. the one
    # classification call covers the whole document. None on items
    # processed before this field existed.
    total_cost_usd: Optional[float] = None

    pages: list[PageExtractionRecord] = Field(default_factory=list)

    # --- Human review decision (unset until a reviewer acts) ---
    review_status: ReviewStatus = "pending"
    decision_type: Optional[DecisionType] = None
    reviewer_id: Optional[str] = None
    reviewer_notes: Optional[str] = None
    reviewed_at: Optional[float] = None

    # decision_type == "update": which existing Sheet row this supersedes
    # (gets an additive "superseded by" flag, never auto-deleted — see design).
    old_row_ref: Optional[str] = None

    # decision_type == "new_category": path/ref to the LLM-drafted regex diff for
    # a dev to review before merging — not built yet.
    regex_draft_ref: Optional[str] = None

    # Candidate rows from the existing Sheet that might be the same regulatory
    # topic as this new document — union of 3 independent signals (embedding
    # similarity, department+license_type fuzzy match, LLM batch-scan), each
    # candidate tagged with which signal(s) flagged it plus a field-by-field
    # diff against the new document's drafted values. No fixed top-N cutoff —
    # see service/pdf_candidate_matching.py for why.
    candidate_matches: Optional[list[dict[str, Any]]] = None

    # decision_type == "new_category" support signal: {"fits_known_category":
    # bool, "matched_topic": str|None, "reasoning": str} — LLM judgment of
    # whether this document's topic fits any operation_topic already present
    # in the Sheet. A suggestion surfaced in the admin UI, never auto-applied —
    # the reviewer still has to click ④ themselves. See
    # service/pdf_candidate_matching.py's check_category_fit().
    category_fit_check: Optional[dict[str, Any]] = None

    # Legacy field — the LLM relevance-scope check that used to populate this
    # (service/pdf_relevance_check.py, plus a Lambda pre-screen for oversized
    # docs) was removed 2026-09: every document entering this pipeline is
    # already curated as relevant by whoever uploaded it, and the fixed
    # topic-scope list both checks judged against kept being wrong/too
    # narrow (e.g. general tax/labor-law documents wrongly flagged, when
    # Restbiz's real scope is broader than food-specific licenses). Kept on
    # the model, defaulting to None, only so already-processed items from
    # before this change keep displaying correctly in the admin UI.
    relevance_check: Optional[dict[str, str]] = None

    # {"shape": "structured_license"|"know_how", "reasoning": str,
    # "secondary_pages": list[[start,end]]} — routes which drafting path ran
    # (see service/pdf_content_shape.py); secondary_pages (REVISED
    # 2026-08-24) flags a substantial chunk of the OTHER shape mixed into the
    # same document, which gets run through that other pipeline too,
    # producing additional sibling ReviewItem(s) for this same upload rather
    # than silently losing whichever shape didn't win the primary vote. None
    # means the classifier itself never ran (old item, or it failed before
    # reaching this step) — treat the same as "structured_license" for
    # display, since that's the fallback the classifier itself uses too.
    content_shape: Optional[dict[str, Any]] = None

    # Populated instead of llm_drafted_fields when content_shape == "know_how"
    # — one dict per identified topic: {title, summary, full_text, category,
    # page_range}. A single PDF can produce several of these, unlike the
    # structured path's one-PDF-one-row model. See
    # service/pdf_knowhow_drafting.py + service/knowhow_write_back.py.
    knowhow_topics: Optional[list[dict[str, Any]]] = None

    # Not built yet — LLM-drafted structured fields (main_topic, sub_topic,
    # entity_type, ขั้นตอน, เอกสาร, ค่าธรรมเนียม, ระยะเวลา, เงื่อนไข) for the reviewer
    # to edit before approval, per the original staging/review design.
    llm_drafted_fields: Optional[dict[str, Any]] = None

    @property
    def needs_review(self) -> bool:
        return any(p.needs_review for p in self.pages)

    @property
    def total_flag_count(self) -> int:
        return sum(len(p.flags) for p in self.pages)

    @property
    def high_severity_flag_count(self) -> int:
        return sum(p.high_severity_flag_count for p in self.pages)

    @property
    def processing_duration_seconds(self) -> Optional[float]:
        """Wall-clock time from extraction_started_at to extraction_completed_at
        — NOT the sum of every call's elapsed_seconds, since pages extract
        concurrently (summing would overcount). None if either timestamp is
        missing (e.g. an item processed before these fields existed)."""
        if self.extraction_started_at is None or self.extraction_completed_at is None:
            return None
        return self.extraction_completed_at - self.extraction_started_at

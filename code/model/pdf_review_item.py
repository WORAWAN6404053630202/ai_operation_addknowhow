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

    # LLM judgment of whether this document belongs in the restaurant-business
    # regulatory knowledge base at all — {"tier": "relevant"|"uncertain"|
    # "not_relevant", "reasoning": str}. See service/pdf_relevance_check.py.
    relevance_check: Optional[dict[str, str]] = None

    # {"shape": "structured_license"|"know_how", "reasoning": str} — routes
    # which drafting path ran (see service/pdf_content_shape.py). None means
    # the classifier itself never ran (old item, or it failed before reaching
    # this step) — treat the same as "structured_license" for display, since
    # that's the fallback the classifier itself uses too.
    content_shape: Optional[dict[str, str]] = None

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

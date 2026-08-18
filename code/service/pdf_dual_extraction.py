# code/service/pdf_dual_extraction.py
"""Automatic dual-model PDF page extraction (feature/pdf-ingestion) — calls both
Typhoon OCR and Claude vision (via OpenRouter) on the same page and runs the
automatic validation checks (pdf_extraction_validation.py) on the pair.

This is the missing piece the manual comparison tests (test_typhoon_ocr.py /
test_claude_vision_ocr.py) didn't cover: those required copy-pasting output
between two separate script runs to compare by hand. This module does both
calls and the comparison in one function call.

Why two models instead of one: AWS Textract was ruled out (no Thai support at
all). Typhoon alone and Claude alone were each tested against the real source
PDF and neither is reliable enough on its own — they have different, uncorrelated
failure patterns (Typhoon: occasional fabricated details; Claude: recurring
misreads on words that should be easy). Running both and flagging disagreements
catches real errors that a single model, however good, would silently miss —
this mirrors the "model-assisted, human-refined" pattern used in published
document-extraction pipelines, not a workaround unique to this project."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import asdict, dataclass

from openai import OpenAI
from pdf2image import convert_from_path
from typhoon_ocr import ocr_document

import conf
from model.pdf_review_item import PageExtractionRecord, ReviewItem
from service.pdf_extraction_validation import ValidationFlag, validate_extraction
from utils.logger import get_logger

logger = get_logger(__name__)

# TRAINING WHEEL — this feature is still under active dev/test, not yet a reviewed
# production path. conf.py defaults to loading the REAL env.properties unless
# RESTBIZ_ENV_FILE is explicitly set (see conf.py) — that default exists so
# production never needs any extra env var, but it means a developer who simply
# forgets to `export RESTBIZ_ENV_FILE=env.dev.properties` before running this
# module would silently pull real production config instead of the isolated dev
# one. Refuse to run at all in that case, loudly, rather than trust anyone to
# remember. Remove this guard deliberately once this module is an intentional,
# reviewed part of the production pipeline — don't just delete it because it's
# in the way; that decision should be made on purpose, not by accident.
def _assert_dev_environment() -> None:
    if conf.ENV_FILE_NAME == "env.properties":
        raise RuntimeError(
            "pdf_dual_extraction refused to run: conf.py loaded the REAL env.properties "
            "(RESTBIZ_ENV_FILE was not set). This module is still dev/test-only — set "
            "RESTBIZ_ENV_FILE=env.dev.properties before running it, to make sure nothing "
            "here can touch production config by accident."
        )

_CLAUDE_PROMPT = (
    "ถอดข้อความภาษาไทย (และอังกฤษถ้ามี) ทั้งหมดในภาพเอกสารนี้ให้ครบถ้วนและถูกต้องที่สุด "
    "รักษาโครงสร้างเดิมไว้เป็น markdown (หัวข้อ, ตาราง, รายการเลข/bullet) "
    "กฎสำคัญ: ห้ามเดาหรือแต่งเติมข้อความที่อ่านไม่ชัดเจนเด็ดขาด "
    "ถ้าตัวอักษร/ตัวเลขจุดไหนอ่านไม่ออกจริงๆ ให้ใส่ [อ่านไม่ชัด] แทนตรงจุดนั้น "
    "แทนที่จะเดาคำที่ดูสมเหตุสมผล"
)


@dataclass
class DualExtractionResult:
    page_num: int
    typhoon_markdown: str
    claude_markdown: str
    flags: list[ValidationFlag]

    @property
    def needs_review(self) -> bool:
        return len(self.flags) > 0

    @property
    def high_severity_flags(self) -> list[ValidationFlag]:
        return [f for f in self.flags if f.severity == "high"]


def _run_typhoon(pdf_path: str, page_num: int, image_dim: int) -> str:
    return ocr_document(
        pdf_path,
        model="typhoon-ocr",
        figure_language="Thai",
        task_type="v1.5",
        page_num=page_num,
        target_image_dim=image_dim,
    )


def _run_claude(pdf_path: str, page_num: int) -> str:
    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
    image = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=300)[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    resp = client.chat.completions.create(
        model=conf.OPENROUTER_MODEL_PRACTICAL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _CLAUDE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=4000,
    )
    return resp.choices[0].message.content or "(no text returned)"


def extract_page_dual(
    pdf_path: str, page_num: int, image_dim: int = 1800, use_llm_comparison: bool = False
) -> DualExtractionResult:
    """Runs Typhoon and Claude on the same page and validates the pair.
    Both calls happen sequentially (simple, and each call already takes several
    seconds — parallelizing is a later optimization, not needed for the review
    queue's throughput at this stage). `use_llm_comparison` additionally runs the
    general-purpose LLM catch-all check (extra API call — off by default)."""
    _assert_dev_environment()
    logger.info(f"[PDFDualExtraction] page {page_num}: running Typhoon...")
    typhoon_markdown = _run_typhoon(pdf_path, page_num, image_dim)

    logger.info(f"[PDFDualExtraction] page {page_num}: running Claude...")
    claude_markdown = _run_claude(pdf_path, page_num)

    flags = validate_extraction(
        typhoon_markdown,
        compare_with=claude_markdown,
        compare_label="claude",
        use_llm_comparison=use_llm_comparison,
    )
    logger.info(f"[PDFDualExtraction] page {page_num}: {len(flags)} flag(s) raised")

    return DualExtractionResult(
        page_num=page_num,
        typhoon_markdown=typhoon_markdown,
        claude_markdown=claude_markdown,
        flags=flags,
    )


def build_review_item(
    filename: str, page_results: list[DualExtractionResult], s3_raw_pdf_path: str = "", draft_fields: bool = True
) -> ReviewItem:
    """Bridges this module's per-page extraction results into a ReviewItem ready
    to hand to PdfReviewQueueManager.save(). Kept separate from extract_page_dual
    so callers can extract several pages, inspect them, THEN decide to queue —
    not forced to persist on every single-page call. `draft_fields=True` (default)
    also runs the LLM field-drafting step (extra API call) so a reviewer sees
    suggested Sheet-column values immediately instead of an empty form."""
    pages = [
        PageExtractionRecord(
            page_num=r.page_num,
            typhoon_markdown=r.typhoon_markdown,
            claude_markdown=r.claude_markdown,
            flags=[asdict(f) for f in r.flags],
        )
        for r in page_results
    ]

    llm_drafted_fields = None
    if draft_fields:
        from service.pdf_field_drafting import draft_fields_from_pages

        logger.info(f"[PDFDualExtraction] Drafting Sheet fields for {filename}...")
        llm_drafted_fields = draft_fields_from_pages([r.typhoon_markdown for r in page_results])

    return ReviewItem(
        filename=filename,
        s3_raw_pdf_path=s3_raw_pdf_path,
        extraction_completed_at=time.time(),
        llm_drafted_fields=llm_drafted_fields,
        pages=pages,
    )

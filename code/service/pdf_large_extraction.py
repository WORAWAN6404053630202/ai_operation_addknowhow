# code/service/pdf_large_extraction.py
"""EC2-side full-document OCR extraction for PDFs too large for Lambda's hard
15-minute execution cap (feature/pdf-ingestion) — the counterpart to
lambda/pdf_extraction/handler.py's MAX_PAGES_FOR_LAMBDA handoff.

Real test evidence (2026-08): a 66-page PDF hit Lambda's 15-minute ceiling
around page ~55-60 even with 4-way page concurrency — and that ceiling is a
hard AWS platform limit, not something more memory/timeout config can raise.
Rather than building fan-out/Step-Functions machinery to split one document
across many Lambda invocations, oversized documents hand off to this module
instead: it runs as part of the EC2 consumer process, which is already
running 24/7 for other reasons, so a long extraction here has no hard time
limit and adds zero marginal infrastructure cost or complexity beyond what's
already deployed.

Deliberately duplicates (does not import) lambda/pdf_extraction/handler.py's
OCR functions — that file is a standalone zip-deployable Lambda with no
access to code/ at all, so sharing a module isn't an option; keeping the
extraction logic here as a close, clearly-labeled mirror is the pragmatic
choice over maintaining a shared package two different deployment targets
would both need to vendor anyway."""

from __future__ import annotations

import base64
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import boto3
import pymupdf
from openai import OpenAI
from typhoon_ocr import ocr_document

import conf
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_WORKERS = 4  # same concurrency cap as the Lambda extractor, for the same reason

_CLAUDE_PROMPT = (
    "ถอดข้อความภาษาไทย (และอังกฤษถ้ามี) ทั้งหมดในภาพเอกสารนี้ให้ครบถ้วนและถูกต้องที่สุด "
    "รักษาโครงสร้างเดิมไว้เป็น markdown (หัวข้อ, ตาราง, รายการเลข/bullet) "
    "กฎสำคัญ: ห้ามเดาหรือแต่งเติมข้อความที่อ่านไม่ชัดเจนเด็ดขาด "
    "ถ้าตัวอักษร/ตัวเลขจุดไหนอ่านไม่ออกจริงๆ ให้ใส่ [อ่านไม่ชัด] แทนตรงจุดนั้น "
    "แทนที่จะเดาคำที่ดูสมเหตุสมผล"
)


def _render_page_to_png_bytes(pdf_bytes: bytes, page_num: int, dpi: int = 200) -> bytes:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def _run_typhoon(png_bytes: bytes, page_num: int) -> str:
    tmp_path = f"{tempfile.gettempdir()}/restbiz_large_page_{page_num}.png"
    with open(tmp_path, "wb") as f:
        f.write(png_bytes)
    try:
        return ocr_document(
            tmp_path,
            model="typhoon-ocr",
            figure_language="Thai",
            task_type="v1.5",
            api_key=conf.TYPHOON_OCR_API_KEY,
        )
    finally:
        os.remove(tmp_path)


def _run_claude(png_bytes: bytes) -> str:
    client = OpenAI(api_key=conf.OPENROUTER_API_KEY, base_url=conf.OPENROUTER_BASE_URL)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    resp = client.chat.completions.create(
        model=conf.OPENROUTER_MODEL,
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


def _extract_one_page(pdf_bytes: bytes, page_num: int) -> dict:
    png_bytes = _render_page_to_png_bytes(pdf_bytes, page_num)
    typhoon_markdown = _run_typhoon(png_bytes, page_num)
    claude_markdown = _run_claude(png_bytes)
    logger.info(f"[LargeExtraction] Page {page_num}: extracted (Typhoon {len(typhoon_markdown)} chars, Claude {len(claude_markdown)} chars)")
    return {"page_num": page_num, "typhoon_markdown": typhoon_markdown, "claude_markdown": claude_markdown}


def extract_full_document(
    bucket: str, raw_pdf_key: str, on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
    """Downloads the original PDF from S3 and OCRs every page with no time
    pressure (unlike the Lambda path, this can safely take as long as it
    needs to). Returns the same page-record shape process_extraction_result()
    already expects, so downstream validation/drafting/candidate-matching
    code needs zero changes to handle large-document results.

    on_progress(pages_done, pages_total), if given, is called after every
    page completes — from this function's own thread (the as_completed loop
    below), never from a worker thread, so the callback doesn't need to be
    thread-safe itself. Whether/how often that actually turns into an S3
    write is the caller's call (see sqs_consumer.py) — this function just
    reports every single page, no throttling logic here."""
    s3 = boto3.client("s3", region_name=conf.AWS_REGION or None)
    logger.info(f"[LargeExtraction] Downloading s3://{bucket}/{raw_pdf_key}")
    obj = s3.get_object(Bucket=bucket, Key=raw_pdf_key)
    pdf_bytes = obj["Body"].read()

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    num_pages = len(doc)
    doc.close()
    logger.info(f"[LargeExtraction] {raw_pdf_key}: {num_pages} page(s), starting full extraction (no time limit on EC2)")

    pages: list[dict | None] = [None] * num_pages
    pages_done = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_extract_one_page, pdf_bytes, p): p for p in range(1, num_pages + 1)}
        for future in as_completed(futures):
            page_num = futures[future]
            pages[page_num - 1] = future.result()
            pages_done += 1
            if on_progress is not None:
                on_progress(pages_done, num_pages)

    logger.info(f"[LargeExtraction] {raw_pdf_key}: all {num_pages} page(s) extracted successfully")
    return pages

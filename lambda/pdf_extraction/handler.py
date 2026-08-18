# lambda/pdf_extraction/handler.py
"""AWS Lambda handler v2: PDF extraction ONLY (Typhoon OCR + Claude vision), no
validation/queueing/Sheet logic here — per the original architecture diagram,
Lambda's job stops at "extract and save the result to S3"; everything downstream
(validation, field-drafting, review queue) runs on the EC2 app, reusing the
already-tested code in code/service/pdf_extraction_validation.py etc., not
duplicated here.

v1 of this file used AWS Textract — ruled out, no Thai support at all (see
project history: avg_confidence=60.58, fully garbled output on a real test doc).

STANDALONE + zip-deployable (not a container image) — do not import from code/
(the FastAPI app). Renders PDF pages to PNG via PyMuPDF (pure Python wheel, NO
system Poppler dependency), then hands the PNG directly to typhoon_ocr — its
own source confirms passing a non-.pdf path skips its internal Poppler-based
render entirely (see typhoon_ocr.ocr_utils.prepare_ocr_messages: the `is_image`
branch uses PIL.Image.open() directly). This is what keeps this Lambda a plain
zip upload instead of needing a container image with Poppler installed at the
OS level.

Requires these Lambda environment variables (set in the console, NOT read from
this repo's conf.py/env.properties — this function has no access to those):
  TYPHOON_OCR_API_KEY, OPENROUTER_API_KEY
Optional:
  OPENROUTER_BASE_URL   (default: https://openrouter.ai/api/v1)
  OPENROUTER_MODEL      (default: anthropic/claude-sonnet-4-5)
  OUTPUT_S3_PREFIX      (default: restbiz/extracted/)

KNOWN LIMITATION: pages are processed concurrently (ThreadPoolExecutor) to fit
within Lambda's execution time limit, but a very large PDF (100+ pages) could
still exceed even the 15-minute Lambda maximum. Not solved here — would need
splitting into one invocation per page (e.g. via Step Functions) if that case
becomes real. Today's real test document (66 pages) was sized with this in mind."""

import base64
import json
import logging
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pymupdf
from openai import OpenAI
from typhoon_ocr import ocr_document

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

TYPHOON_OCR_API_KEY = os.environ.get("TYPHOON_OCR_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
OUTPUT_S3_PREFIX = os.environ.get("OUTPUT_S3_PREFIX", "restbiz/extracted/")

_MAX_WORKERS = 4  # concurrent pages — keep modest, each page = 2 API calls already

_CLAUDE_PROMPT = (
    "ถอดข้อความภาษาไทย (และอังกฤษถ้ามี) ทั้งหมดในภาพเอกสารนี้ให้ครบถ้วนและถูกต้องที่สุด "
    "รักษาโครงสร้างเดิมไว้เป็น markdown (หัวข้อ, ตาราง, รายการเลข/bullet) "
    "กฎสำคัญ: ห้ามเดาหรือแต่งเติมข้อความที่อ่านไม่ชัดเจนเด็ดขาด "
    "ถ้าตัวอักษร/ตัวเลขจุดไหนอ่านไม่ออกจริงๆ ให้ใส่ [อ่านไม่ชัด] แทนตรงจุดนั้น "
    "แทนที่จะเดาคำที่ดูสมเหตุสมผล"
)


def _render_page_to_png_bytes(pdf_bytes: bytes, page_num: int, dpi: int = 200) -> bytes:
    """page_num is 1-indexed to match the rest of this codebase's convention."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def _run_typhoon(png_bytes: bytes, page_num: int) -> str:
    # ocr_document takes a file path, not bytes — /tmp is Lambda's writable scratch space.
    tmp_path = f"/tmp/page_{page_num}.png"
    with open(tmp_path, "wb") as f:
        f.write(png_bytes)
    try:
        return ocr_document(
            tmp_path,
            model="typhoon-ocr",
            figure_language="Thai",
            task_type="v1.5",
            api_key=TYPHOON_OCR_API_KEY,
        )
    finally:
        os.remove(tmp_path)


def _run_claude(png_bytes: bytes) -> str:
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
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
    logger.info(f"Page {page_num}: extracted (Typhoon {len(typhoon_markdown)} chars, Claude {len(claude_markdown)} chars)")
    return {"page_num": page_num, "typhoon_markdown": typhoon_markdown, "claude_markdown": claude_markdown}


def lambda_handler(event, context):
    if not TYPHOON_OCR_API_KEY or not OPENROUTER_API_KEY:
        raise RuntimeError("TYPHOON_OCR_API_KEY and OPENROUTER_API_KEY must be set as Lambda environment variables.")

    results = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        filename = key.rsplit("/", 1)[-1]

        logger.info(f"Processing s3://{bucket}/{key}")
        obj = s3.get_object(Bucket=bucket, Key=key)
        pdf_bytes = obj["Body"].read()

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(doc)
        doc.close()
        logger.info(f"{filename}: {num_pages} page(s)")

        pages = [None] * num_pages
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {pool.submit(_extract_one_page, pdf_bytes, p): p for p in range(1, num_pages + 1)}
            for future in as_completed(futures):
                page_num = futures[future]
                pages[page_num - 1] = future.result()

        output = {"filename": filename, "s3_raw_pdf_path": key, "pages": pages}
        output_key = f"{OUTPUT_S3_PREFIX}{filename}.json"
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=json.dumps(output, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"Wrote extraction result to s3://{bucket}/{output_key}")
        results.append({"input_key": key, "output_key": output_key, "page_count": num_pages})

    return {"statusCode": 200, "body": json.dumps(results, ensure_ascii=False)}

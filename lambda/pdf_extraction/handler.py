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
  TYPHOON_OCR_API_KEY, OPENROUTER_API_KEY, SQS_QUEUE_URL
Optional:
  OPENROUTER_BASE_URL     (default: https://openrouter.ai/api/v1)
  OPENROUTER_MODEL        (default: anthropic/claude-sonnet-4-5)
  OUTPUT_S3_PREFIX        (default: restbiz/extracted/)
  HANDOFF_S3_PREFIX       (default: restbiz/pending_large/)
  MAX_PAGES_FOR_LAMBDA    (default: 35)

SQS DELIVERY (2026-08-24): originally relied on S3's own bucket notification
configuration (prefix/suffix rule -> SQS) to tell the EC2 consumer a result
was ready. Live-tested and reproduced twice: Lambda wrote the output to S3
successfully both times, but the S3->SQS notification never fired — no
message ever arrived (queue confirmed at 0 messages via GetQueueAttributes
each time), despite the notification config and the queue's access policy
both checking out correct via the console and API. Rather than keep
debugging an opaque AWS-side gap, this Lambda now sends the SQS message
itself directly (_notify_sqs below) right after each S3 write — the same
Records[]-shaped body a real S3 event notification would have sent, so
sqs_consumer.py's process_sqs_message() needed zero changes. This removes
the dependency on that notification path entirely; the bucket's
extracted/ and pending_large/ -> SQS notification rules are no longer
needed and were removed (the restbiz/ -> Lambda trigger notification is
unrelated and stays)."""

"""LARGE-DOCUMENT HANDLING (was "KNOWN LIMITATION: not solved here" before
2026-08-22): a real test (66-page PDF) confirmed Lambda's 15-minute hard cap
gets hit around ~55-60 pages even with 4-way page concurrency, and that cap
cannot be raised — it's a platform limit, not a config knob. Rather than
building the fan-out/Step-Functions machinery to split one document across
many Lambda invocations, large documents instead hand off to the EC2 instance
(already running 24/7 for the SQS consumer, so this adds zero marginal
infrastructure cost, and a long-lived process has no 15-minute ceiling at
all): this Lambda does a CHEAP screen — OCR + relevance-check just the first
2 pages — and either (a) writes a "not worth the spend" skip marker if that
partial content is clearly out of scope, saving the cost of the remaining
~30+ pages entirely, or (b) writes a small handoff marker (not the extraction
itself) to HANDOFF_S3_PREFIX for service/pdf_large_extraction.py on the EC2
side to pick up and fully process with no time pressure. Small/typical
documents (the common case) are completely unaffected — same one Lambda
invocation, same output shape, same downstream SQS path as before."""

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
sqs = boto3.client("sqs")

TYPHOON_OCR_API_KEY = os.environ.get("TYPHOON_OCR_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
OUTPUT_S3_PREFIX = os.environ.get("OUTPUT_S3_PREFIX", "restbiz/extracted/")
HANDOFF_S3_PREFIX = os.environ.get("HANDOFF_S3_PREFIX", "restbiz/pending_large/")
MAX_PAGES_FOR_LAMBDA = int(os.environ.get("MAX_PAGES_FOR_LAMBDA", "35"))
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "")

_MAX_WORKERS = 4  # concurrent pages — keep modest, each page = 2 API calls already
_SCREEN_PAGE_COUNT = 2  # pages OCR'd for the cheap pre-screen on oversized docs

_CLAUDE_PROMPT = (
    "ถอดข้อความภาษาไทย (และอังกฤษถ้ามี) ทั้งหมดในภาพเอกสารนี้ให้ครบถ้วนและถูกต้องที่สุด "
    "รักษาโครงสร้างเดิมไว้เป็น markdown (หัวข้อ, ตาราง, รายการเลข/bullet) "
    "กฎสำคัญ: ห้ามเดาหรือแต่งเติมข้อความที่อ่านไม่ชัดเจนเด็ดขาด "
    "ถ้าตัวอักษร/ตัวเลขจุดไหนอ่านไม่ออกจริงๆ ให้ใส่ [อ่านไม่ชัด] แทนตรงจุดนั้น "
    "แทนที่จะเดาคำที่ดูสมเหตุสมผล"
)

# Deliberately a plain yes/no triage prompt, NOT the full 3-tier relevant/
# uncertain/not_relevant judgment code/service/pdf_relevance_check.py already
# does properly on the complete document later in the EC2 pipeline — this one
# only exists to answer "worth spending ~30 more pages of OCR on?", so it's
# biased to say yes when unsure (a wrongly-skipped real document is a much
# worse outcome than a few extra dollars of OCR on a false positive).
_SCREEN_PROMPT_TEMPLATE = """นี่คือ 2 หน้าแรกของเอกสารขนาดใหญ่ (หลายสิบหน้า) ที่อัปโหลดเข้าระบบฐานความรู้
ของ Restbiz ซึ่งเป็นระบบ AI ให้คำปรึกษาด้านกฎหมาย/ใบอนุญาต/ข้อบังคับสำหรับธุรกิจร้านอาหารในไทย

จากแค่ 2 หน้านี้ ประเมินคร่าวๆ ว่าเอกสารทั้งฉบับ**น่าจะ**เกี่ยวข้องกับธุรกิจร้านอาหาร/อาหารในไทยหรือไม่
(ใบอนุญาต, กฎหมาย, know-how ที่เกี่ยวข้อง) — ถ้าไม่แน่ใจ ให้ตอบว่าเกี่ยวข้องไว้ก่อนเสมอ (ป้องกันการข้าม
เอกสารที่จริงๆ มีประโยชน์ไปอย่างผิดพลาด)

ตอบเป็น JSON เท่านั้น: {{"worth_processing": true หรือ false, "reason": "เหตุผลสั้นๆ 1 ประโยค"}}

=== เนื้อหา 2 หน้าแรก ===
{combined_text}
"""


def _notify_sqs(bucket: str, key: str) -> None:
    """Sends the same Records[]-shaped body a real S3 event notification would
    have sent — sqs_consumer.py's process_sqs_message() parses this exact
    shape and doesn't know or care whether S3 or this function sent it. Logs
    and re-raises on failure rather than swallowing it: if this fails, the
    S3 object this Lambda just wrote would otherwise sit there forever with
    nothing to ever pick it up, which is worse than a visible Lambda error
    (visible in CloudWatch, and S3 will retry the whole invocation)."""
    if not SQS_QUEUE_URL:
        raise RuntimeError("SQS_QUEUE_URL is not set — cannot notify the consumer that s3://{}/{} is ready".format(bucket, key))
    message_body = json.dumps({"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]}, ensure_ascii=False)
    sqs.send_message(QueueUrl=SQS_QUEUE_URL, MessageBody=message_body)
    logger.info(f"Notified SQS for s3://{bucket}/{key}")


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


def _screen_worth_processing(pdf_bytes: bytes, num_pages: int) -> dict:
    """OCRs just the first _SCREEN_PAGE_COUNT pages and asks a cheap yes/no
    triage question — the whole point is spending a little to potentially
    avoid spending a lot (OCR on the remaining 30+ pages of something
    out-of-scope). Never raises: any failure here defaults to "process it
    anyway" (see caller), since a broken screen must not silently drop a real
    document."""
    screen_pages = min(_SCREEN_PAGE_COUNT, num_pages)
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_extract_one_page, pdf_bytes, p): p for p in range(1, screen_pages + 1)}
        results = {futures[f]: f.result() for f in as_completed(futures)}
    ordered = [results[p] for p in sorted(results)]

    combined_text = "\n\n---\n\n".join(p["typhoon_markdown"] for p in ordered)
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": _SCREEN_PROMPT_TEMPLATE.format(combined_text=combined_text)}],
        max_tokens=300,
    )
    raw = (resp.choices[0].message.content or "{}").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(raw)
    return {"worth_processing": bool(parsed.get("worth_processing", True)), "reason": str(parsed.get("reason", ""))}


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

        if num_pages > MAX_PAGES_FOR_LAMBDA:
            logger.info(f"{filename}: {num_pages} pages exceeds MAX_PAGES_FOR_LAMBDA={MAX_PAGES_FOR_LAMBDA}, screening first")
            try:
                screen = _screen_worth_processing(pdf_bytes, num_pages)
            except Exception as e:
                logger.error(f"Screening failed, defaulting to worth_processing=True (fail open, not closed): {e}")
                screen = {"worth_processing": True, "reason": f"screen error, defaulted to processing: {e}"}

            if not screen["worth_processing"]:
                skip_output = {
                    "filename": filename,
                    "s3_raw_pdf_path": key,
                    "skipped": True,
                    "skip_reason": screen["reason"],
                    "page_count": num_pages,
                }
                output_key = f"{OUTPUT_S3_PREFIX}{filename}.json"
                s3.put_object(
                    Bucket=bucket, Key=output_key,
                    Body=json.dumps(skip_output, ensure_ascii=False).encode("utf-8"),
                    ContentType="application/json",
                )
                logger.info(f"{filename}: skipped after cheap screen ({screen['reason']}) — wrote {output_key}, saved OCR on remaining {num_pages - _SCREEN_PAGE_COUNT} page(s)")
                _notify_sqs(bucket, output_key)
                results.append({"input_key": key, "output_key": output_key, "page_count": num_pages, "skipped": True})
                continue

            handoff = {
                "filename": filename,
                "s3_raw_pdf_path": key,
                "bucket": bucket,
                "page_count": num_pages,
                "screen_reason": screen["reason"],
            }
            handoff_key = f"{HANDOFF_S3_PREFIX}{filename}.json"
            s3.put_object(
                Bucket=bucket, Key=handoff_key,
                Body=json.dumps(handoff, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json",
            )
            logger.info(f"{filename}: {num_pages} pages, screen passed ({screen['reason']}) — handed off to EC2 via {handoff_key}, not processed in Lambda")
            _notify_sqs(bucket, handoff_key)
            results.append({"input_key": key, "handoff_key": handoff_key, "page_count": num_pages, "handed_off": True})
            continue

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
        _notify_sqs(bucket, output_key)
        results.append({"input_key": key, "output_key": output_key, "page_count": num_pages})

    return {"statusCode": 200, "body": json.dumps(results, ensure_ascii=False)}

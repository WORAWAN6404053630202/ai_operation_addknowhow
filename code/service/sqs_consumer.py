# code/service/sqs_consumer.py
"""EC2-side consumer for the PDF extraction pipeline (feature/pdf-ingestion) —
the "rest of the work" half of the Lambda(extract-only) → S3 → SQS → here
architecture agreed for this feature. Reuses every already-tested piece from
today's work (pdf_extraction_validation.validate_extraction,
pdf_field_drafting.draft_fields_from_pages, PdfReviewQueueManager) rather than
re-implementing any of it — the split exists specifically so this logic lives
in ONE place, not duplicated between Lambda and EC2.

Processed-tracking is SQS's own delete-on-success semantics, not a separate
mechanism: a message is only deleted after successfully building and saving a
ReviewItem. If processing raises, the message is left alone — SQS's visibility
timeout expires and it becomes available for a retry automatically. This means
process_extraction_result() below must be safe to re-run on the same S3 key
(it is: PdfReviewQueueManager.save() is an upsert keyed by ReviewItem.id, and
a fresh id is generated per call — a retried message produces a second queue
item rather than corrupting a partial one; acceptable for now, a follow-up
could dedupe on s3_raw_pdf_path if that becomes a real problem)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Optional

import boto3

import conf
from model.pdf_review_item import PageExtractionRecord, ReviewItem
from model.pdf_review_queue_manager import PdfReviewQueueManager
from service.pdf_candidate_matching import find_candidate_matches
from service.pdf_content_shape import classify_content_shape
from service.pdf_extraction_validation import validate_extraction
from service.pdf_field_drafting import draft_fields_from_pages
from service.pdf_knowhow_drafting import identify_knowhow_topics
from service.pdf_large_extraction import extract_full_document
from service.pdf_relevance_check import check_relevance
from utils.logger import get_logger

logger = get_logger(__name__)

_queue_manager = PdfReviewQueueManager()


def _assert_dev_environment() -> None:
    """Same training-wheel as pdf_dual_extraction.py — this module still isn't
    a reviewed production path. Remove deliberately once it is."""
    if conf.ENV_FILE_NAME == "env.properties":
        raise RuntimeError(
            "sqs_consumer refused to run: conf.py loaded the REAL env.properties "
            "(RESTBIZ_ENV_FILE was not set). Set RESTBIZ_ENV_FILE=env.dev.properties first."
        )


def _build_and_save_review_item(
    filename: str, s3_raw_pdf_path: str, raw_pages: list[dict], use_llm_comparison: bool = False
) -> ReviewItem:
    """Shared tail end of the pipeline (validate → draft fields → relevance
    check → candidate matching → save) — identical regardless of whether
    raw_pages came from Lambda's normal-size path or
    pdf_large_extraction.extract_full_document()'s oversized-document path.
    Keeping this as one function is what makes the handoff path "free" from
    downstream code's perspective: nothing past this point needs to know or
    care which extractor produced the pages."""
    pages: list[PageExtractionRecord] = []
    for p in raw_pages:
        flags = validate_extraction(
            p["typhoon_markdown"],
            compare_with=p["claude_markdown"],
            compare_label="claude",
            use_llm_comparison=use_llm_comparison,
        )
        pages.append(
            PageExtractionRecord(
                page_num=p["page_num"],
                typhoon_markdown=p["typhoon_markdown"],
                claude_markdown=p["claude_markdown"],
                flags=[asdict(f) for f in flags],
            )
        )
        logger.info(f"[SQSConsumer] {filename} page {p['page_num']}: {len(flags)} flag(s)")

    page_markdowns = [p["typhoon_markdown"] for p in raw_pages]

    item = ReviewItem(
        filename=filename,
        s3_raw_pdf_path=s3_raw_pdf_path,
        extraction_completed_at=time.time(),
        pages=pages,
    )

    # Both best-effort: a Sheet/embedding/LLM hiccup here must not lose the
    # extraction work already done above — the item still saves with the
    # relevant field left None, reviewer just won't see that particular hint
    # for this one item (check_relevance itself already never raises; the
    # try/except here is defense-in-depth against an unexpected bug, not the
    # primary safety net).
    try:
        item.relevance_check = check_relevance(page_markdowns)
    except Exception as e:
        logger.error(f"[SQSConsumer] Relevance check failed for {filename}, continuing without it: {e}")

    # Routing decision: does this document fit the structured 13-field Sheet
    # schema, or is it know-how/multi-topic content that would lose most of
    # its substance forced into that shape? classify_content_shape() itself
    # never raises (fails toward "structured_license", the longer-proven
    # path) — no try/except needed here, unlike the calls above/below that
    # wrap genuinely-fallible external services.
    shape_check = classify_content_shape(page_markdowns)
    item.content_shape = shape_check

    if shape_check["shape"] == "know_how":
        try:
            topic_bounds = identify_knowhow_topics(page_markdowns)
            item.knowhow_topics = [
                {
                    "title": t["title"],
                    "summary": t["summary"],
                    "category": t["category"],
                    "page_range": f"{t['start_page']}-{t['end_page']}" if t["start_page"] != t["end_page"] else str(t["start_page"]),
                    "full_text": "\n\n---\n\n".join(page_markdowns[t["start_page"] - 1 : t["end_page"]]),
                }
                for t in topic_bounds
            ]
            logger.info(f"[SQSConsumer] {filename}: know-how path, {len(item.knowhow_topics)} topic(s) identified")
        except Exception as e:
            logger.error(f"[SQSConsumer] Know-how topic-splitting failed for {filename}, item saved with no topics for manual handling: {e}")
    else:
        llm_drafted_fields = draft_fields_from_pages(page_markdowns)
        item.llm_drafted_fields = llm_drafted_fields
        try:
            item.candidate_matches = find_candidate_matches(item)
        except Exception as e:
            logger.error(f"[SQSConsumer] Candidate matching failed for {filename}, continuing without it: {e}")

    _queue_manager.save(item)
    logger.info(f"[SQSConsumer] Saved review item {item.id} for {filename} ({len(pages)} pages, shape={shape_check['shape']})")
    return item


def process_extraction_result(bucket: str, key: str, use_llm_comparison: bool = False) -> ReviewItem:
    """Downloads one Lambda-produced S3 object and routes it to the right
    handling, based on what Lambda actually did with this document (see
    lambda/pdf_extraction/handler.py's MAX_PAGES_FOR_LAMBDA branch):

    1. Normal completed extraction (the common case, unchanged since this
       feature's original build) — pages already OCR'd by Lambda, just run
       the validate/draft/review pipeline on them.
    2. Skip marker ({"skipped": true, ...}) — Lambda's cheap pre-screen on an
       oversized document judged it not worth the remaining OCR spend. Saves
       a lightweight, zero-page ReviewItem carrying the skip reason in
       relevance_check (reusing the same admin-UI banner candidate items
       use) so it's still visible/overridable, not silently dropped.
    3. Handoff marker (key under conf.PDF_HANDOFF_S3_PREFIX) — Lambda's
       screen passed but the document is too large for Lambda's 15-minute
       cap to process itself. Runs the full OCR here on EC2 (no such time
       limit), then continues through the identical pipeline as case 1.

    Safe to call directly (bypassing SQS) for local testing against a known
    S3 key — this is the actual processing logic; poll_and_process_forever()
    below is just the SQS event loop wrapped around this function."""
    _assert_dev_environment()

    s3 = boto3.client("s3", region_name=conf.AWS_REGION or None)
    logger.info(f"[SQSConsumer] Fetching s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = json.loads(obj["Body"].read().decode("utf-8"))

    if key.startswith(conf.PDF_HANDOFF_S3_PREFIX):
        filename = data["filename"]
        s3_raw_pdf_path = data["s3_raw_pdf_path"]
        logger.info(
            f"[SQSConsumer] {filename}: handoff marker ({data.get('page_count')} pages, "
            f"screen_reason={data.get('screen_reason')!r}) — running full extraction on EC2"
        )
        raw_pages = extract_full_document(bucket, s3_raw_pdf_path)
        return _build_and_save_review_item(filename, s3_raw_pdf_path, raw_pages, use_llm_comparison)

    filename = data["filename"]
    s3_raw_pdf_path = data.get("s3_raw_pdf_path", "")

    if data.get("skipped"):
        reason = data.get("skip_reason", "")
        logger.info(f"[SQSConsumer] {filename}: skip marker from Lambda's cheap screen — {reason!r}")
        item = ReviewItem(
            filename=filename,
            s3_raw_pdf_path=s3_raw_pdf_path,
            extraction_completed_at=time.time(),
            pages=[],
            relevance_check={
                "tier": "not_relevant",
                "reasoning": f"ข้ามการประมวลผลอัตโนมัติ (เอกสารมี {data.get('page_count', '?')} หน้า, "
                f"เช็คแค่ 2 หน้าแรกแล้วพบว่าไม่น่าเกี่ยวข้อง): {reason}",
            },
        )
        _queue_manager.save(item)
        logger.info(f"[SQSConsumer] Saved skip-marker review item {item.id} for {filename}")
        return item

    raw_pages = data["pages"]
    return _build_and_save_review_item(filename, s3_raw_pdf_path, raw_pages, use_llm_comparison)


def process_sqs_message(message_body: str) -> Optional[ReviewItem]:
    """Parses one SQS message body (the raw S3 event notification JSON S3
    sends directly to SQS — same Records[].s3.bucket/object shape as the
    Lambda trigger event) and processes it. Returns None if the message
    doesn't contain an S3 record (e.g. a malformed/foreign message — logged,
    not raised, so the poll loop doesn't crash on one bad message)."""
    try:
        body = json.loads(message_body)
    except json.JSONDecodeError:
        logger.error(f"[SQSConsumer] Message body is not valid JSON: {message_body[:200]}")
        return None

    records = body.get("Records", [])
    if not records:
        logger.warning(f"[SQSConsumer] Message has no S3 Records, skipping: {message_body[:200]}")
        return None

    record = records[0]
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]
    return process_extraction_result(bucket, key)


def poll_and_process_forever(queue_url: str, use_llm_comparison: bool = False) -> None:
    """The real deployment entrypoint (not exercised in today's local testing —
    needs SQS-reachable AWS credentials, deferred per plan). Long-polls SQS
    (up to 20s per call — near-instant pickup without a tight/wasteful loop),
    processes each message, and deletes it ONLY after process_sqs_message
    succeeds. A message that fails is simply left alone; SQS's own visibility
    timeout makes it reappear for an automatic retry."""
    _assert_dev_environment()
    sqs = boto3.client("sqs", region_name=conf.AWS_REGION or None)
    logger.info(f"[SQSConsumer] Long-polling {queue_url} ...")

    while True:
        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )
        for message in resp.get("Messages", []):
            receipt_handle = message["ReceiptHandle"]
            try:
                process_sqs_message(message["Body"])
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
                logger.info("[SQSConsumer] Message processed and deleted.")
            except Exception as e:
                logger.error(f"[SQSConsumer] Processing failed, leaving message for retry: {e}")

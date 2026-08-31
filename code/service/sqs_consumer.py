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
from service.pdf_candidate_matching import check_category_fit, find_candidate_matches
from service.pdf_content_shape import classify_content_shape
from service.pdf_extraction_validation import validate_extraction
from service.pdf_field_drafting import draft_fields_from_pages, identify_license_topics
from service.pdf_knowhow_drafting import identify_knowhow_topics
from service.pdf_large_extraction import extract_full_document
from service.pdf_relevance_check import check_relevance
from service import pdf_status_tracker
from utils.logger import get_logger
from utils.page_ranges import format_page_ranges, group_into_ranges

logger = get_logger(__name__)

_queue_manager = PdfReviewQueueManager()

# Hard cap on the EC2 large-document handoff path (see process_extraction_result's
# handoff branch below) — each attempt re-OCRs and re-runs Claude Vision over
# EVERY page of the document from scratch (no per-page checkpoint), so an
# uncapped retry loop here is the most expensive possible version of the
# 2026-08-27/08-28 field-drafting incident this same sweep fixed. 3 attempts
# gives real transient errors (a momentary 500, a network blip) room to
# resolve on their own without letting a permanently-broken document
# (e.g. out-of-credits, a malformed PDF) retry indefinitely.
_MAX_HANDOFF_ATTEMPTS = 3


def _assert_dev_environment() -> None:
    """Same training-wheel as pdf_dual_extraction.py — this module still isn't
    a reviewed production path. Remove deliberately once it is."""
    if conf.ENV_FILE_NAME == "env.properties":
        raise RuntimeError(
            "sqs_consumer refused to run: conf.py loaded the REAL env.properties "
            "(RESTBIZ_ENV_FILE was not set). Set RESTBIZ_ENV_FILE=env.dev.properties first."
        )


def _slice_by_ranges(
    pages: list[PageExtractionRecord], page_markdowns: list[str], ranges: list[tuple[int, int]]
) -> tuple[list[PageExtractionRecord], list[str]]:
    """ranges are LOCAL 1-based indices into the given pages/page_markdowns
    lists — NOT necessarily the original document's page numbers, since this
    gets called on secondary sub-slices too (see _build_and_save_review_item).
    Joins multiple ranges in order, e.g. [(1,1),(3,3)] for a topic that
    resumed after an interruption."""
    sliced_pages: list[PageExtractionRecord] = []
    sliced_markdowns: list[str] = []
    for s, e in ranges:
        sliced_pages.extend(pages[s - 1 : e])
        sliced_markdowns.extend(page_markdowns[s - 1 : e])
    return sliced_pages, sliced_markdowns


def _build_knowhow_items(
    filename: str, s3_raw_pdf_path: str, pages: list[PageExtractionRecord], page_markdowns: list[str],
    relevance_check: Optional[dict], content_shape: dict,
) -> list[ReviewItem]:
    """Builds ONE bundled ReviewItem (multiple knowhow_topics inside) from
    this pages/page_markdowns slice. Safe to call on the full document OR a
    secondary sub-slice of a mixed-shape document (see
    _build_and_save_review_item) — identify_knowhow_topics()'s returned
    page_ranges are always local to whatever page_markdowns was passed in,
    and _slice_by_ranges resolves them against the SAME pages/page_markdowns
    given here, so nesting composes correctly regardless of depth."""
    item = ReviewItem(
        filename=filename, s3_raw_pdf_path=s3_raw_pdf_path, extraction_completed_at=time.time(),
        pages=pages, relevance_check=relevance_check, content_shape=content_shape,
    )
    try:
        topic_bounds = identify_knowhow_topics(page_markdowns)
        knowhow_topics = []
        covered_local: set[int] = set()
        for t in topic_bounds:
            topic_pages, topic_markdowns = _slice_by_ranges(pages, page_markdowns, t["page_ranges"])
            for s, e in t["page_ranges"]:
                covered_local.update(range(s, e + 1))
            knowhow_topics.append({
                "document_title": t["document_title"],
                "source_type": t["source_type"],
                "main_topic": t["main_topic"],
                "sub_topic": t["sub_topic"],
                "summary": t["summary"],
                "category": t["category"],
                "page_range": format_page_ranges([p.page_num for p in topic_pages]),
                "full_text": "\n\n---\n\n".join(topic_markdowns),
            })

        # Defensive: the prompt REQUIRES every page land in some topic, and
        # live-testing confirms it reliably does — but an LLM not perfectly
        # following an instruction is exactly the kind of thing that must not
        # silently lose content if it ever happens (see the sibling gap this
        # same sweep found in _build_license_items, where exclusion is
        # actually the INTENDED behavior). No second LLM call here — just a
        # deterministic catch-all topic over whatever local page_markdowns
        # index wasn't claimed by anything, verbatim, so nothing vanishes.
        uncovered_local = sorted(set(range(1, len(page_markdowns) + 1)) - covered_local)
        if uncovered_local:
            logger.warning(f"[SQSConsumer] {filename}: know-how splitting left {len(uncovered_local)} page(s) uncovered ({uncovered_local}), adding a catch-all topic")
            uncovered_ranges = group_into_ranges(uncovered_local)
            leftover_pages, leftover_markdowns = _slice_by_ranges(pages, page_markdowns, uncovered_ranges)
            fallback_doc_title = topic_bounds[0]["document_title"] if topic_bounds else filename
            fallback_source_type = topic_bounds[0]["source_type"] if topic_bounds else "document"
            knowhow_topics.append({
                "document_title": fallback_doc_title,
                "source_type": fallback_source_type,
                "main_topic": "เนื้อหาที่เหลือ (ไม่ถูกจัดหมวดหมู่โดยอัตโนมัติ)",
                "sub_topic": "ตรวจสอบและจัดหมวดหมู่ด้วยตนเอง",
                "summary": "",
                "category": "",
                "page_range": format_page_ranges([p.page_num for p in leftover_pages]),
                "full_text": "\n\n---\n\n".join(leftover_markdowns),
            })

        item.knowhow_topics = knowhow_topics
        logger.info(f"[SQSConsumer] {filename}: know-how path, {len(knowhow_topics)} topic(s) identified")
    except Exception as e:
        logger.error(f"[SQSConsumer] Know-how topic-splitting failed for {filename}, item saved with no topics for manual handling: {e}")

    _queue_manager.save(item)
    logger.info(f"[SQSConsumer] Saved review item {item.id} for {filename} ({len(pages)} pages, shape=know_how)")
    return [item]


def _build_license_items(
    filename: str, s3_raw_pdf_path: str, pages: list[PageExtractionRecord], page_markdowns: list[str],
    relevance_check: Optional[dict], content_shape: dict,
) -> list[ReviewItem]:
    """Builds 0..N independent ReviewItems (own candidate-matching, own
    decision each) from this pages/page_markdowns slice — a single slice is
    not guaranteed to be exactly one regulatory topic (could be several
    combined announcements, or none at all). Safe to call on the full
    document OR a secondary sub-slice of a mixed-shape document; see
    _build_knowhow_items for why nesting composes correctly."""
    try:
        topic_bounds = identify_license_topics(page_markdowns)
    except Exception as e:
        logger.error(
            f"[SQSConsumer] License topic-splitting failed for {filename}, falling back to treating "
            f"this slice as one topic (old single-topic behavior): {e}"
        )
        topic_bounds = [{"department": "", "license_type": "", "page_ranges": [(1, len(page_markdowns))]}]

    if not topic_bounds:
        # Genuinely 0 license topics found (not a failure) — save the whole-
        # slice item anyway with no drafted fields, so it's still visible in
        # the queue for a human to look at/reject rather than vanishing
        # silently. Distinguishing this from the exception fallback above
        # matters: an LLM failure must not look identical to "confirmed
        # nothing here".
        logger.info(f"[SQSConsumer] {filename}: no license topics identified in this slice — saving unlabeled item for manual review")
        item = ReviewItem(
            filename=filename, s3_raw_pdf_path=s3_raw_pdf_path, extraction_completed_at=time.time(),
            pages=pages, relevance_check=relevance_check, content_shape=content_shape,
        )
        _queue_manager.save(item)
        logger.info(f"[SQSConsumer] Saved review item {item.id} for {filename} ({len(pages)} pages, shape=structured_license, 0 topics)")
        return [item]

    items: list[ReviewItem] = []
    covered_local: set[int] = set()
    for topic in topic_bounds:
        topic_pages, topic_markdowns = _slice_by_ranges(pages, page_markdowns, topic["page_ranges"])
        for s, e in topic["page_ranges"]:
            covered_local.update(range(s, e + 1))

        topic_item = ReviewItem(
            filename=filename, s3_raw_pdf_path=s3_raw_pdf_path, extraction_completed_at=time.time(),
            pages=topic_pages, relevance_check=relevance_check, content_shape=content_shape,
        )
        # Was unwrapped until 2026-08-28 — the one call in this function
        # without error handling, unlike every sibling call around it. When
        # draft_fields_from_pages() escalates to Sonnet for an oversized
        # field (max_tokens=6000, the largest in the PDF pipeline) and that
        # call fails, the exception used to propagate all the way up through
        # process_extraction_result(), so the SQS message was never deleted
        # and SQS redelivered it for a full-cost retry from page 1 — 47
        # times over ~24h on one document, ~$78-80 in OpenRouter spend
        # before anyone noticed. Matching the candidate_matches/
        # category_fit_check pattern below: log and leave the item
        # (llm_drafted_fields stays None) for manual review instead of
        # losing the whole item to one field-drafting failure.
        try:
            topic_item.llm_drafted_fields = draft_fields_from_pages(topic_markdowns)
        except Exception as e:
            logger.error(f"[SQSConsumer] Field drafting failed for {filename} ({topic['department']!r}/{topic['license_type']!r}), continuing without it: {e}")
        page_str = format_page_ranges([p.page_num for p in topic_pages])
        try:
            topic_item.candidate_matches = find_candidate_matches(topic_item)
        except Exception as e:
            logger.error(f"[SQSConsumer] Candidate matching failed for {filename} ({topic['department']!r}/{topic['license_type']!r}), continuing without it: {e}")
        try:
            topic_item.category_fit_check = check_category_fit(topic_item)
        except Exception as e:
            logger.error(f"[SQSConsumer] Category-fit check failed for {filename} ({topic['department']!r}/{topic['license_type']!r}), continuing without it: {e}")

        _queue_manager.save(topic_item)
        logger.info(
            f"[SQSConsumer] Saved review item {topic_item.id} for {filename} "
            f"(pages {page_str}, dept={topic['department']!r}, license_type={topic['license_type']!r})"
        )
        items.append(topic_item)

    # Live-testing found a real gap here: identify_license_topics() correctly
    # (by design) excludes pages that don't describe a license procedure —
    # but those pages must not just vanish. Route them through the know-how
    # pipeline instead (one hop, not recursive — _build_knowhow_items has its
    # own deterministic no-LLM fallback for ITS uncovered pages, so this
    # can't chain indefinitely even in a pathological case).
    uncovered_local = sorted(set(range(1, len(page_markdowns) + 1)) - covered_local)
    if uncovered_local:
        logger.info(f"[SQSConsumer] {filename}: pages {uncovered_local} weren't part of any license topic, routing through know-how instead")
        uncovered_ranges = group_into_ranges(uncovered_local)
        leftover_pages, leftover_markdowns = _slice_by_ranges(pages, page_markdowns, uncovered_ranges)
        items.extend(_build_knowhow_items(filename, s3_raw_pdf_path, leftover_pages, leftover_markdowns, relevance_check, content_shape))

    return items


def _build_and_save_review_item(
    filename: str, s3_raw_pdf_path: str, raw_pages: list[dict], use_llm_comparison: bool = False
) -> list[ReviewItem]:
    """Shared tail end of the pipeline (validate → classify → draft/split →
    candidate matching → save) — identical regardless of whether raw_pages
    came from Lambda's normal-size path or
    pdf_large_extraction.extract_full_document()'s oversized-document path.
    Keeping this as one function is what makes the handoff path "free" from
    downstream code's perspective: nothing past this point needs to know or
    care which extractor produced the pages.

    Returns a LIST of ReviewItems, not one:
      - structured_license content splits into 0..N independent topics
        (identify_license_topics) — a PDF isn't guaranteed to be exactly one
        license, could be several combined announcements or none at all.
      - REVISED 2026-08-24: a document can also genuinely MIX both shapes
        (live-tested and confirmed — e.g. a clean license procedure followed
        by unrelated general business advice). classify_content_shape()'s
        secondary_pages flags a substantial chunk of the OTHER shape when
        present; that chunk is run through the OTHER pipeline too, producing
        additional sibling item(s) alongside the primary one, instead of
        silently dropping whichever shape didn't win the primary classification."""
    pages: list[PageExtractionRecord] = []
    for p in raw_pages:
        # validate_extraction() only calls an LLM when use_llm_comparison=True
        # (off by default in the normal SQS flow, so this wasn't part of the
        # 2026-08-27/08-28 incident) — but when it IS on (manual/test runs
        # pass it through), this was the one call in this loop without error
        # handling. Same fix pattern as the rest of this sweep: log and keep
        # going with no flags for this page rather than losing the whole
        # document to one page's comparison call failing.
        try:
            flags = validate_extraction(
                p["typhoon_markdown"],
                compare_with=p["claude_markdown"],
                compare_label="claude",
                use_llm_comparison=use_llm_comparison,
            )
        except Exception as e:
            logger.error(f"[SQSConsumer] Extraction validation failed for {filename} page {p['page_num']}, continuing without flags: {e}")
            flags = []
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

    # Best-effort: a Sheet/embedding/LLM hiccup here must not lose the
    # extraction work already done above — every resulting item just gets
    # relevance_check=None (check_relevance itself already never raises; this
    # try/except is defense-in-depth against an unexpected bug, not the
    # primary safety net).
    relevance_check = None
    try:
        relevance_check = check_relevance(page_markdowns)
    except Exception as e:
        logger.error(f"[SQSConsumer] Relevance check failed for {filename}, continuing without it: {e}")

    # Routing decision: does this document fit the structured 13-field Sheet
    # schema, or is it know-how/multi-topic content that would lose most of
    # its substance forced into that shape? classify_content_shape() itself
    # never raises (fails toward "structured_license", the longer-proven
    # path) — no try/except needed here, unlike the calls above/below that
    # wrap genuinely-fallible external services.
    shape_check = classify_content_shape(page_markdowns)

    if shape_check["shape"] == "know_how":
        primary_builder, secondary_builder = _build_knowhow_items, _build_license_items
    else:
        primary_builder, secondary_builder = _build_license_items, _build_knowhow_items

    secondary_pages = shape_check.get("secondary_pages", [])

    # Exclude secondary-shape pages from the PRIMARY pass — without this, a
    # license procedure classified as "secondary" would ALSO get re-picked-up
    # as its own topic inside the primary know_how bundle (confirmed live:
    # this happened before this exclusion was added), redundantly describing
    # the same content twice across 2 different tabs/sheets — a reviewer
    # approving both would write near-duplicate data into 2 places. Each
    # page is handled by exactly one builder.
    excluded_page_nums = {n for start, end in secondary_pages for n in range(start, end + 1)}
    primary_indices = [i for i, p in enumerate(pages) if p.page_num not in excluded_page_nums]
    primary_pages = [pages[i] for i in primary_indices]
    primary_markdowns = [page_markdowns[i] for i in primary_indices]

    items: list[ReviewItem] = []
    if primary_pages:
        items.extend(primary_builder(filename, s3_raw_pdf_path, primary_pages, primary_markdowns, relevance_check, shape_check))
    else:
        logger.info(f"[SQSConsumer] {filename}: entire document was flagged as secondary-shape content, nothing left for the primary pass")

    for start, end in secondary_pages:
        secondary_shape = "structured_license" if shape_check["shape"] == "know_how" else "know_how"
        logger.info(f"[SQSConsumer] {filename}: processing secondary {secondary_shape} content at pages {start}-{end}")
        sub_pages, sub_markdowns = pages[start - 1 : end], page_markdowns[start - 1 : end]
        items.extend(secondary_builder(filename, s3_raw_pdf_path, sub_pages, sub_markdowns, relevance_check, shape_check))

    return items


def process_extraction_result(bucket: str, key: str, use_llm_comparison: bool = False) -> list[ReviewItem]:
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
        page_count = data.get("page_count")
        logger.info(
            f"[SQSConsumer] {filename}: handoff marker ({page_count} pages, "
            f"screen_reason={data.get('screen_reason')!r}) — running full extraction on EC2"
        )

        # Visibility for this specific long-running path (see
        # service/pdf_status_tracker.py's docstring for why: a page-1 restart
        # on every retry previously left zero trace anywhere a human could
        # see until either success or a manual journalctl dig). attempt is
        # read from any status object a PRIOR failed attempt left behind, so
        # the admin UI shows a real retry count instead of always "1".
        attempt = pdf_status_tracker.next_attempt_number(bucket, filename)
        pdf_status_tracker.write_status(
            bucket, filename, stage="processing", pages_done=0, pages_total=page_count, attempt=attempt,
        )

        # Throttled to every 20 pages (or the final page) — this is purely
        # about S3 request/log volume, NOT OCR spend: the real cost (Typhoon
        # + Claude Vision per page) happens regardless of how often progress
        # gets reported, an S3 PUT here and there is negligible either way.
        _PROGRESS_WRITE_EVERY_N_PAGES = 20

        def _on_progress(pages_done: int, pages_total: int) -> None:
            if pages_done == pages_total or pages_done % _PROGRESS_WRITE_EVERY_N_PAGES == 0:
                pdf_status_tracker.write_status(
                    bucket, filename, stage="processing",
                    pages_done=pages_done, pages_total=pages_total, attempt=attempt,
                )

        try:
            raw_pages = extract_full_document(bucket, s3_raw_pdf_path, on_progress=_on_progress)
        except Exception as e:
            pdf_status_tracker.write_status(
                bucket, filename, stage="failed", pages_total=page_count, attempt=attempt, error=str(e),
            )
            # Added 2026-08-28, same incident that motivated pdf_status_tracker.py
            # itself (see its docstring: a 634-page document retried for 13+
            # hours with no cap). Visibility into failed attempts alone doesn't
            # stop them from repeating — extract_full_document() has no
            # per-page checkpoint, so every SQS redelivery here re-runs OCR +
            # Claude Vision on ALL pages_total pages again, not just the ones
            # that failed. For a 634-page document that's ~634 fresh Claude
            # calls per retry, worse than the field-drafting bug this same
            # sweep fixed. Past _MAX_HANDOFF_ATTEMPTS, stop re-raising (so SQS
            # deletes the message instead of redelivering it forever) and save
            # a placeholder item flagged for manual review instead.
            if attempt >= _MAX_HANDOFF_ATTEMPTS:
                logger.error(
                    f"[SQSConsumer] {filename}: giving up after {attempt} failed attempts "
                    f"(handoff extraction), saving for manual review instead of retrying again: {e}"
                )
                item = ReviewItem(
                    filename=filename, s3_raw_pdf_path=s3_raw_pdf_path, extraction_completed_at=time.time(),
                    pages=[],
                    relevance_check={
                        "tier": "extraction_failed",
                        "reasoning": f"หยุดลองใหม่หลังล้มเหลว {attempt} ครั้งติดต่อกัน (เอกสาร {page_count} หน้า) — ข้อผิดพลาดล่าสุด: {e}",
                    },
                )
                _queue_manager.save(item)
                pdf_status_tracker.clear(bucket, filename)
                return [item]
            raise

        items = _build_and_save_review_item(filename, s3_raw_pdf_path, raw_pages, use_llm_comparison)
        pdf_status_tracker.clear(bucket, filename)
        return items

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
        return [item]

    raw_pages = data["pages"]
    items = _build_and_save_review_item(filename, s3_raw_pdf_path, raw_pages, use_llm_comparison)
    # Clears the status Lambda's normal (≤MAX_PAGES_FOR_LAMBDA) path wrote —
    # see lambda/pdf_extraction/handler.py's _write_status. Mirrors the
    # handoff branch above: covers the full gap (Lambda OCR + this
    # validation/drafting step), not just the Lambda portion.
    pdf_status_tracker.clear(bucket, filename)
    return items


def process_sqs_message(message_body: str) -> Optional[list[ReviewItem]]:
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

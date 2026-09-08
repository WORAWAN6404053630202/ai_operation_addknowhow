"""
Admin Router — PDF review queue only.

Split off from the old router/admin.py (2026-09) into its own file, served by
a separate app/process/port (see app_pdf.py) from the production chat bot —
so a PDF-pipeline dependency crash (see the boto3 incident, 2026-09-06/07)
can never take the bot down again. The bot's own session-monitoring routes
live in router/admin_bot.py.

static/admin.html (the combined dashboard shell, still served by the BOT
service) points its PDF-related fetch calls at this service's own base URL —
see admin.html's PDF_API_BASE constant — so the one page still shows both
Sessions and PDF Queue tabs even though two different backends serve them.

Each endpoint below carries its own dependencies=[_auth] (X-Admin-Key), same
as the pattern in admin_bot.py — there is no router-level auth dependency to
inherit from (the old router-level require_admin_basic_auth was removed in
favor of this per-route mechanism; merging that removal with main's new
per-route auth left these endpoints open until fixed).
"""

import json
import time
from typing import Optional

try:
    import conf as _conf
except Exception:
    _conf = None

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model.pdf_review_queue_manager import PdfReviewQueueManager
from service import pdf_status_tracker
from utils.admin_auth import require_admin_key
from utils.logger import get_logger
from utils.page_ranges import format_page_ranges

_LOG = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin-pdf"])
_auth = Depends(require_admin_key)

_pdf_queue_manager = PdfReviewQueueManager()


class PdfReviewDecision(BaseModel):
    review_status: str  # "approved" | "rejected"
    decision_type: Optional[str] = None  # "new" | "duplicate" | "update" | "new_category" — required when approving
    reviewer_id: Optional[str] = None
    reviewer_notes: Optional[str] = None
    old_row_ref: Optional[str] = None  # decision_type == "update": which existing Sheet row this supersedes
    edited_fields: Optional[dict[str, str]] = None  # reviewer's corrections to llm_drafted_fields, if any —
    # what actually gets written to the Sheet is these values (merged over the original draft),
    # never the raw LLM draft blindly — the whole point of the review step.


def _page_range_str(item) -> str:
    """Same gap-aware logic as sheet_write_back.py's _format_page_range —
    surfaced in the queue list so a structured_license PDF that split into
    several independent review items (see sqs_consumer.py, 2026-08-24)
    doesn't just show N identical filenames with no way to tell them apart
    at a glance. Uses format_page_ranges rather than plain min/max since an
    item's pages aren't guaranteed contiguous (a topic can resume after an
    interruption, see utils/page_ranges.py)."""
    return format_page_ranges([p.page_num for p in item.pages])


@router.get("/api/pdf-queue", dependencies=[_auth])
async def pdf_queue_list():
    """Summary list for the left-hand panel — newest upload first.

    "in_flight" (2026-08-27) surfaces documents still being worked on by the
    EC2 large-document OCR path (or stuck retrying against a failure) that
    have no ReviewItem yet — see service/pdf_status_tracker.py's docstring
    for the incident that motivated this: a 634-page document silently
    retried for 13+ hours with nothing visible here beforehand."""
    items = _pdf_queue_manager.list_all()
    items.sort(key=lambda i: i.uploaded_at, reverse=True)

    in_flight = []
    bucket = getattr(_conf, "PDF_INGESTION_S3_BUCKET", "") if _conf else ""
    if bucket:
        try:
            in_flight = pdf_status_tracker.list_in_flight(bucket)
        except Exception as e:
            _LOG.warning(f"[pdf_queue_list] Failed to list in-flight status: {e}")

    return JSONResponse({
        "items": [
            {
                "id": i.id,
                "filename": i.filename,
                "uploaded_at": i.uploaded_at,
                "extraction_completed_at": i.extraction_completed_at,
                "review_status": i.review_status,
                "decision_type": i.decision_type,
                "page_count": len(i.pages),
                "page_range": _page_range_str(i),
                "department": (i.llm_drafted_fields or {}).get("หน่วยงาน", ""),
                "license_type": (i.llm_drafted_fields or {}).get("ใบอนุญาต", ""),
                "total_flag_count": i.total_flag_count,
                "high_severity_flag_count": i.high_severity_flag_count,
                "needs_review": i.needs_review,
            }
            for i in items
        ],
        "total": len(items),
        "in_flight": in_flight,
    })


@router.get("/api/pdf-queue/{item_id}", dependencies=[_auth])
async def pdf_queue_detail(item_id: str):
    """Full record — every page's Typhoon/Claude text and flags, for the review UI."""
    item = _pdf_queue_manager.load(item_id)
    if item is None:
        return JSONResponse({"error": "Item not found"}, status_code=404)
    return JSONResponse(item.model_dump())


@router.post("/api/pdf-queue/{item_id}/decision", dependencies=[_auth])
async def pdf_queue_decide(item_id: str, decision: PdfReviewDecision):
    """Records a human review decision. For review_status="approved" with
    decision_type in (new, update, new_category), this WRITES a new row to the
    real (additive-only, never edited/deleted) Google Sheet before saving the
    decision — if that write fails, the item is left "pending" and the error is
    returned, rather than silently recording an "approved" status that never
    actually reached the Sheet. decision_type="duplicate" and "rejected" never
    touch the Sheet at all — nothing new to write."""
    item = _pdf_queue_manager.load(item_id)
    if item is None:
        return JSONResponse({"error": "Item not found"}, status_code=404)

    if decision.review_status not in ("approved", "rejected"):
        return JSONResponse({"error": "review_status must be 'approved' or 'rejected'"}, status_code=400)
    if decision.review_status == "approved" and not decision.decision_type:
        return JSONResponse({"error": "decision_type is required when approving"}, status_code=400)
    if decision.decision_type not in (None, "new", "duplicate", "update", "new_category"):
        return JSONResponse({"error": "invalid decision_type"}, status_code=400)

    item.review_status = decision.review_status
    item.decision_type = decision.decision_type
    item.reviewer_id = decision.reviewer_id
    item.reviewer_notes = decision.reviewer_notes
    item.old_row_ref = decision.old_row_ref
    item.reviewed_at = time.time()

    # Reviewer's corrections win over the raw LLM draft — this is the actual
    # review step, not a formality. Merge (not replace) so any field the
    # reviewer left untouched in the UI still has its drafted value.
    if decision.edited_fields:
        item.llm_drafted_fields = {**(item.llm_drafted_fields or {}), **decision.edited_fields}

    sheet_result = None
    if decision.review_status == "approved" and decision.decision_type != "duplicate":
        try:
            if item.knowhow_topics:
                # know_how items skip the structured Sheet entirely — no
                # duplicate-detection built for the know_how tab yet, so
                # decision_type is effectively always "new" here regardless
                # of what the UI sent (see admin.html: it always submits
                # "new" for this path).
                from service.knowhow_write_back import append_knowhow_topics

                sheet_result = append_knowhow_topics(item.knowhow_topics, item.filename)
            else:
                from service.sheet_write_back import append_review_item_to_sheet

                sheet_result = append_review_item_to_sheet(item)
        except Exception as e:
            _LOG.error(f"[pdf_queue_decide] Sheet write-back failed for item {item_id}: {e}")
            return JSONResponse(
                {"error": f"Sheet write-back failed, decision NOT saved: {e}"}, status_code=502
            )

    _pdf_queue_manager.save(item)

    if decision.decision_type == "new_category":
        # Deliberately NOT auto-generating routing/regex changes here — the
        # original design note ("LLM-drafted regex diff for a dev to review")
        # would mean an AI-authored change to the supervisor's core topic-
        # routing logic going out with only sheet_write_back's per-row safety
        # net, not the review this actually needs. This is the safe, scoped-
        # down version: the row is written (same as "new" — the content isn't
        # lost) and flagged loudly so a developer notices it needs manual
        # topic-routing/classifier work, instead of silently looking like any
        # other approved row. Search server logs for "NEW_CATEGORY" or check
        # the admin UI's queue list for the ④ badge (item.decision_type ==
        # "new_category") to find these.
        _LOG.warning(
            f"[pdf_queue_decide] NEW_CATEGORY flagged for dev attention: "
            f"item={item.id} filename={item.filename!r} — content was written to the Sheet "
            f"as a new row, but likely needs manual topic-routing/classifier updates "
            f"(see persona_supervisor.py's topic classification) that this review flow "
            f"does not attempt automatically."
        )

    return JSONResponse({"ok": True, "item": item.model_dump(), "sheet_result": sheet_result})


@router.post("/api/pdf-queue/{item_id}/reprocess", dependencies=[_auth])
async def pdf_queue_reprocess(item_id: str):
    """Manual safety net for a document Lambda's cheap oversized-document
    pre-screen (lambda/pdf_extraction/handler.py's _screen_worth_processing,
    REMOVED 2026-09 — see that file and sqs_consumer.py for why) decided to
    skip back when that screen still existed — it only sampled a handful of
    pages, so it could misjudge a large document whose relevant content
    happened to be outside its sample (real case, 2026-08/09: a 258-page tax
    guide was skipped because its first 2 pages read as generic
    front-matter — this endpoint is how that specific item was recovered).
    Kept even though nothing can produce a new skip-marker item anymore, so
    any already-skipped item from before the removal stays recoverable:
    writes a fresh handoff marker pointing at the SAME original PDF and
    re-notifies SQS, so it goes through full extraction on EC2 this
    time — bypassing the screen entirely, same as if it had passed the first
    time. Does not delete the old skip-marker item; marks it rejected/
    superseded so the queue doesn't show it as still-pending once the
    reprocessed version lands as a new item."""
    item = _pdf_queue_manager.load(item_id)
    if item is None:
        return JSONResponse({"error": "Item not found"}, status_code=404)
    if not item.s3_raw_pdf_path:
        return JSONResponse({"error": "This item has no s3_raw_pdf_path — nothing to reprocess"}, status_code=400)

    bucket = getattr(_conf, "PDF_INGESTION_S3_BUCKET", "") if _conf else ""
    queue_url = getattr(_conf, "PDF_SQS_QUEUE_URL", "") if _conf else ""
    handoff_prefix = getattr(_conf, "PDF_HANDOFF_S3_PREFIX", "restbiz/pending_large/") if _conf else "restbiz/pending_large/"
    if not bucket or not queue_url:
        return JSONResponse({"error": "PDF_INGESTION_S3_BUCKET / PDF_SQS_QUEUE_URL not configured"}, status_code=500)

    try:
        import boto3

        s3 = boto3.client("s3", region_name=getattr(_conf, "AWS_REGION", "") or None)
        sqs = boto3.client("sqs", region_name=getattr(_conf, "AWS_REGION", "") or None)

        handoff = {
            "filename": item.filename,
            "s3_raw_pdf_path": item.s3_raw_pdf_path,
            "bucket": bucket,
            "page_count": None,  # unknown here; sqs_consumer.py treats it as optional (progress display only)
            "screen_reason": "manual reprocess triggered by reviewer from the admin UI",
        }
        handoff_key = f"{handoff_prefix}{item.filename}.json"
        s3.put_object(
            Bucket=bucket, Key=handoff_key,
            Body=json.dumps(handoff, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        # Same Records[]-shaped body a real S3 event notification would send —
        # sqs_consumer.py's process_sqs_message() parses this exact shape
        # regardless of whether S3 or this endpoint sent it.
        message_body = json.dumps(
            {"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": handoff_key}}}]},
            ensure_ascii=False,
        )
        sqs.send_message(QueueUrl=queue_url, MessageBody=message_body)
    except Exception as e:
        _LOG.error(f"[pdf_queue_reprocess] Failed to re-queue {item.filename} ({item_id}): {e}")
        return JSONResponse({"error": f"Failed to re-queue for reprocessing: {e}"}, status_code=502)

    item.review_status = "rejected"
    item.reviewer_notes = ((item.reviewer_notes or "") + " [ส่งประมวลผลใหม่ทั้งไฟล์แล้ว — รายการนี้ถูกแทนที่ รอรายการใหม่ในคิว]").strip()
    item.reviewed_at = time.time()
    _pdf_queue_manager.save(item)

    _LOG.info(f"[pdf_queue_reprocess] Re-queued {item.filename} ({item_id}) for full extraction, handoff_key={handoff_key}")
    return JSONResponse({"ok": True, "handoff_key": handoff_key})

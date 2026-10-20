"""
Admin Router — PDF review queue + pipeline status/dashboard.

2026-10 scope note (PDF-M8): since the direct-ingest architectural pivot
(2026-09), real successful content writes straight to Chroma and never
reaches this queue — sqs_consumer.py's _build_knowhow_items/_build_license_
items (the old human-review builders that used to populate it for every
document) are no longer called at all (kept in the file per explicit
instruction, not deleted — see that file's own comment at the point they
stopped being wired in). This queue now ONLY receives three kinds of item:
a Lambda-era legacy skip-marker ("not_relevant", can no longer be produced
by anything but kept for historical items), a give-up-after-N-attempts
failure placeholder ("extraction_failed", pages=[]/llm_drafted_fields=None),
and (2026-10, PDF-M11) a vectorstore-write-failure recovery placeholder
("vectorstore_write_failed" — saved when OCR/topic-splitting/drafting all
succeeded but the final Chroma write failed, so that billed work isn't
silently thrown away with no recovery path; see sqs_consumer.py's
_save_vectorstore_write_failure). None of the three have real drafted
content — /decision rejects an "approved" attempt on any of them (see
that endpoint); use /reprocess instead to retry from the original PDF.
An empty or near-empty queue means the pipeline is working, not that
nothing has happened; check router/admin_pdf.py's /api/pdf-status or the
dashboard for real ingestion activity instead.

Split off from the old router/admin.py (2026-09) into its own file,
originally meant to be served by a separate app/process/port (see
app_pdf.py) from the production chat bot — so a PDF-pipeline dependency
crash (see the boto3 incident, 2026-09-06/07) can never take the bot down
again. The bot's own session-monitoring routes live in router/admin_bot.py.

Included from THREE places as of 2026-10 (all should behave identically —
same router object, same handler functions, no logic duplicated anywhere):
  - app_pdf.py — the originally-intended standalone PDF-admin-only service.
  - router/run_admin_server.py-style production entrypoint, if in use there.
  - app.py (the full chat bot) — added 2026-10 after discovering a real
    deployment (the pure-PDF EC2 instance) runs app.py directly and never
    actually runs app_pdf.py as a separate process; a stale, drifted
    duplicate of these same routes used to live in router/admin.py from
    before this file was split out (removed 2026-10 — see admin.py's own
    comment at the point it used to sit).

static/admin.html (the combined dashboard shell) points its PDF-related
fetch calls at a base URL controlled by admin.html's own PDF_API_BASE
constant — same-origin when already on the right port, otherwise an
explicit cross-origin base. Whichever process actually serves admin.html in
a given deployment needs this router included (directly or reachable at
that base URL) for the PDF Queue tab to work.

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
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from model.pdf_review_queue_manager import PdfReviewQueueManager
from service import pdf_status_tracker
from utils.admin_auth import require_admin_key
from utils.logger import get_logger
from utils.page_ranges import format_page_ranges
from utils.pdf_cost_guard import get_cost_guard
from utils.pdf_ingestion_log import read_recent_events

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
    retried for 13+ hours with nothing visible here beforehand.

    See this module's docstring (PDF-M8, 2026-10) for why the list below
    "in_flight" is now only failure/legacy-skip placeholders, not a pending-
    approval queue — an empty list here does not mean nothing has happened."""
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
    # 2026-10 fix (PDF-M8, extended for PDF-M11): since the direct-ingest
    # pivot, every item that reaches this queue is one of: a Lambda-era
    # legacy skip-marker ("not_relevant"), a give-up-after-N-attempts
    # failure placeholder ("extraction_failed", pages=[] /
    # llm_drafted_fields=None — see sqs_consumer.py's give-up branches), or
    # a vectorstore-write-failure recovery placeholder
    # ("vectorstore_write_failed" — see _save_vectorstore_write_failure,
    # saved when OCR/drafting succeeded but the final Chroma write failed).
    # Real successful content now goes straight to Chroma and never
    # reaches this queue at all. Approving any of these would write an
    # item with NO real drafted content as a new Sheet row. The only
    # sensible actions are "reprocess" (retry extraction) or "rejected"
    # (acknowledge and drop).
    _tier = (item.relevance_check or {}).get("tier")
    if decision.review_status == "approved" and _tier in ("extraction_failed", "not_relevant", "vectorstore_write_failed"):
        return JSONResponse(
            {"error": f"Cannot approve a {_tier!r} item — it has no real drafted content to write. Use /reprocess to retry extraction, or reject to dismiss it."},
            status_code=400,
        )

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
                # know_how items skip the structured Sheet entirely (they
                # write to one of the 2 know-how tabs instead — see
                # knowhow_write_back.py). 2026-09: duplicate-detection now
                # exists for this path too (pdf_knowhow_candidate_matching.py,
                # populated per-topic at extraction time — see
                # sqs_consumer.py), and decision_type="update"/old_row_ref
                # are now respected here instead of being silently ignored —
                # see append_knowhow_topics's own docstring for the one
                # remaining limitation (old_row_ref is per-ITEM, applied to
                # every topic in a multi-topic item, not per-topic).
                from service.knowhow_write_back import append_knowhow_topics

                sheet_result = append_knowhow_topics(
                    item.knowhow_topics, item.filename,
                    decision_type=decision.decision_type, old_row_ref=decision.old_row_ref,
                )
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


@router.get("/api/pdf-status", dependencies=[_auth])
async def pdf_status():
    """Status feed for the pure-PDF project's dashboard (app_pdf.py's
    GET /admin/dashboard, 2026-10) — covers the LIVE direct-ingest pipeline
    this router otherwise says nothing about (every route above only ever
    reads/writes the old human-review queue, which the direct-ingest path
    never populates — see sqs_consumer.py's 2026-09 architectural pivot).

    Three things, each independently best-effort (one source failing must
    not blank out the other two):
      - today's cumulative $ spend vs. a reference amount (utils/
        pdf_cost_guard.py) — OBSERVATIONAL ONLY (2026-10: a calendar-day
        auto-block used to gate here, removed — it couldn't tell a genuine
        bulk upload from an actual malfunction, and punished the former as
        hard as the latter; see sqs_consumer.py's _MAX_HANDOFF_ATTEMPTS /
        lambda/pdf_extraction/handler.py's _MAX_ATTEMPTS for the real fix, a
        bounded per-document retry cap targeting the actual failure mode).
        The color here is still useful as a human-noticeable "spend looks
        unusually high today, go check why" cue — the user explicitly chose
        dashboard-only visibility over any push notification channel — it
        just no longer blocks anything on its own.
      - recent ingestion activity (utils/pdf_ingestion_log.py) — per-
        document new/update/duplicate counts and cost, something to look at
        besides raw CloudWatch/journalctl logs.
      - recently archived (superseded) docs (service/local_vector_store.py's
        list_archived_docs, now sorted newest-first with a date filter —
        2026-10 audit item 4) — the same periodic-review data
        pdf_archive_inspect.py shows from the CLI, now also here."""
    cost_guard = get_cost_guard()
    try:
        today_cost_usd = cost_guard.get_today_total()
    except Exception as e:
        _LOG.warning(f"[pdf_status] Failed to read today's cost total: {e}")
        today_cost_usd = 0.0
    ceiling_usd = cost_guard.ceiling_usd
    pct_of_ceiling = round((today_cost_usd / ceiling_usd) * 100, 1) if ceiling_usd > 0 else 0.0

    bucket = getattr(_conf, "PDF_INGESTION_S3_BUCKET", "") if _conf else ""
    recent_activity = []
    if bucket:
        try:
            recent_activity = read_recent_events(bucket, days=7)
        except Exception as e:
            _LOG.warning(f"[pdf_status] Failed to read recent ingestion activity: {e}")

    # 2026-10 fix (Infra2-C1): list_archived_docs pulls in langchain_core/
    # langchain_openai/langchain_community → chromadb — none of which are in
    # requirements-pdf.txt (the lean PDF-only image deliberately excludes
    # them). A module-level import of this name used to make importing
    # admin_pdf.py itself crash on that image with ModuleNotFoundError
    # before this handler's own try/except ever got a chance to run —
    # moved to a lazy import here so the only consequence of the dependency
    # being absent is this one section of the dashboard degrading to [].
    recent_archived = []
    try:
        from service.local_vector_store import list_archived_docs
        since_epoch = time.time() - 7 * 86400
        recent_archived = [
            {"id": d["id"], "metadata": d["metadata"]}
            for d in list_archived_docs(since_epoch=since_epoch)
        ]
    except Exception as e:
        _LOG.warning(f"[pdf_status] Failed to read recent archived docs: {e}")

    return JSONResponse({
        "today_cost_usd": round(today_cost_usd, 4),
        "ceiling_usd": ceiling_usd,
        "pct_of_ceiling": pct_of_ceiling,
        "over_ceiling": today_cost_usd >= ceiling_usd,
        "recent_activity": recent_activity,
        "recent_archived": recent_archived,
    })


# Dashboard page for pdf_status above (2026-10). Lives on THIS router (not
# wherever it happens to be served from) specifically so it travels with
# admin_pdf_router wherever that router is included — see this module's
# docstring for the full list of places that currently do (app_pdf.py,
# app.py, and whichever standalone admin entrypoint production uses).
#
# No X-Admin-Key dependency on the page itself — it's static HTML/JS with no
# embedded secrets; the data it fetches (pdf_status above) is already
# gated. Same pattern static/admin.html already uses (public shell,
# authenticated API calls).
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<title>Restbiz PDF Pipeline — Dashboard</title>
<style>
  body { font-family: -apple-system, "Segoe UI", sans-serif; background: #0f1115; color: #e6e6e6; margin: 0; padding: 24px; }
  h1 { font-size: 18px; margin: 0 0 16px; }
  .card { background: #1a1d24; border-radius: 10px; padding: 18px; margin-bottom: 20px; }
  .cost-row { display: flex; align-items: baseline; gap: 10px; margin-bottom: 10px; }
  .cost-amount { font-size: 28px; font-weight: 600; }
  .cost-ceiling { color: #999; font-size: 14px; }
  .bar-track { background: #2a2e38; border-radius: 6px; height: 14px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 6px; transition: width 0.3s ease; }
  .bar-fill.ok { background: #3fb950; }
  .bar-fill.warn { background: #d29922; }
  .bar-fill.over { background: #da3633; }
  .pct-label { margin-top: 6px; font-size: 13px; color: #aaa; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #2a2e38; }
  th { color: #999; font-weight: 500; }
  .err { color: #f85149; }
  .flag-high { color: #da3633; font-weight: 600; cursor: help; }
  .muted { color: #777; }
  .refresh-note { color: #666; font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
  <h1>Restbiz — PDF Pipeline Dashboard (feature/pdf-only-source)</h1>

  <div class="card">
    <div class="cost-row">
      <span class="cost-amount" id="costAmount">—</span>
      <span class="cost-ceiling" id="costCeiling"></span>
    </div>
    <div class="bar-track"><div class="bar-fill ok" id="costBar" style="width:0%"></div></div>
    <div class="pct-label" id="costPct"></div>
  </div>

  <div class="card">
    <h2 style="font-size:14px;margin:0 0 10px;">กิจกรรมล่าสุด (7 วัน)</h2>
    <table id="activityTable">
      <thead><tr><th>เวลา</th><th>ไฟล์</th><th>ประเภท</th><th>ใหม่</th><th>อัปเดต</th><th>ซ้ำ</th><th>ต้นทุน</th><th>แฟล็กร้ายแรง</th><th>ผิดพลาด</th></tr></thead>
      <tbody><tr><td colspan="9" class="muted">กำลังโหลด...</td></tr></tbody>
    </table>
  </div>

  <div class="card">
    <h2 style="font-size:14px;margin:0 0 10px;">เอกสารที่ถูกเก็บเข้า archive ล่าสุด (7 วัน)</h2>
    <table id="archivedTable">
      <thead><tr><th>วันที่</th><th>ไฟล์ต้นฉบับ</th><th>หัวข้อ</th><th>เหตุผล</th></tr></thead>
      <tbody><tr><td colspan="4" class="muted">กำลังโหลด...</td></tr></tbody>
    </table>
  </div>

  <div class="refresh-note">อัปเดตอัตโนมัติทุก 30 วินาที</div>

<script>
function getAdminKey() {
  let key = sessionStorage.getItem("pdf_admin_key");
  if (!key) {
    key = window.prompt("X-Admin-Key:") || "";
    sessionStorage.setItem("pdf_admin_key", key);
  }
  return key;
}

function fmtTime(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleString("th-TH");
}

// 2026-10 fix: every value rendered below (filename, shape, error,
// source_file, license_type/main_topic, superseded_reason) can originate
// from an uploaded PDF's own text via the LLM drafting/classification
// steps — this codebase already treats PDF-derived text as attacker-
// influenceable everywhere else (see INJECTION_GUARD). The render
// functions used to interpolate these straight into innerHTML with no
// escaping, which is stored XSS: a PDF crafted so a drafted field becomes
// e.g. "<img src=x onerror=...>" would run arbitrary JS in this page,
// including reading the admin key out of sessionStorage below. Escape
// every interpolated value through this before it reaches innerHTML.
function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  // 2026-10 fix (Infra2-C4/PDF2-M2): the textContent→innerHTML round-trip
  // escapes &, <, > but NOT quote characters — browsers only escape quotes
  // inside attribute-value serialization, never inside a text node. The
  // `examples` tooltip below lands in title="${examples}" (an attribute
  // position), so an unescaped `"` in a crafted PDF's flag text could break
  // out of the attribute and run JS reading the admin key out of
  // sessionStorage. Escape both quote characters explicitly.
  return div.innerHTML.replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

async function apiGet(path) {
  const resp = await fetch(path, { headers: { "X-Admin-Key": getAdminKey() } });
  if (resp.status === 401) {
    sessionStorage.removeItem("pdf_admin_key");
    throw new Error("Invalid admin key");
  }
  if (!resp.ok) throw new Error("HTTP " + resp.status);
  return resp.json();
}

function renderCost(data) {
  document.getElementById("costAmount").textContent = "$" + data.today_cost_usd.toFixed(2);
  document.getElementById("costCeiling").textContent = "/ $" + data.ceiling_usd.toFixed(2) + " ต่อวัน";
  const pct = Math.min(data.pct_of_ceiling, 100);
  const bar = document.getElementById("costBar");
  bar.style.width = pct + "%";
  bar.className = "bar-fill " + (data.over_ceiling ? "over" : pct >= 70 ? "warn" : "ok");
  document.getElementById("costPct").textContent =
    data.over_ceiling
      ? "ค่าใช้จ่ายวันนี้สูงกว่าปกติ — ไม่ได้บล็อกการทำงาน แค่เตือนให้เข้ามาดูว่าผิดปกติหรือไม่ (เช่น เอกสารจำนวนมากพร้อมกัน หรือติดลูปซ้ำ)"
      : data.pct_of_ceiling.toFixed(1) + "% ของค่าอ้างอิงวันนี้";
}

function renderActivity(events) {
  const tbody = document.querySelector("#activityTable tbody");
  if (!events.length) { tbody.innerHTML = '<tr><td colspan="9" class="muted">ไม่มีกิจกรรม</td></tr>'; return; }
  tbody.innerHTML = events.map(e => {
    // 2026-10 fix (PDF-M3): validate_extraction()'s high-severity flags
    // (arithmetic mismatches, dual-model disagreements) used to be computed
    // then silently discarded — this is the one place a human can see them
    // now that the review queue no longer gates successful documents.
    const highCount = e.high_severity_flags || 0;
    const examples = (e.high_severity_flag_examples || []).map(escapeHtml).join("; ");
    const flagCell = highCount > 0
      ? `<span class="flag-high" title="${examples}">⚠ ${highCount}</span>`
      : "-";
    return `
    <tr>
      <td>${escapeHtml(fmtTime(e.timestamp))}</td>
      <td>${escapeHtml(e.filename || "")}</td>
      <td>${escapeHtml(e.shape || "")}</td>
      <td>${e.new_count ?? 0}</td>
      <td>${e.update_count ?? 0}</td>
      <td>${e.duplicate_count ?? 0}</td>
      <td>$${(e.cost_usd || 0).toFixed(4)}</td>
      <td>${flagCell}</td>
      <td class="err">${escapeHtml(e.error || "")}</td>
    </tr>`;
  }).join("");
}

function renderArchived(docs) {
  const tbody = document.querySelector("#archivedTable tbody");
  if (!docs.length) { tbody.innerHTML = '<tr><td colspan="4" class="muted">ไม่มีเอกสาร</td></tr>'; return; }
  tbody.innerHTML = docs.map(d => {
    const m = d.metadata || {};
    const topic = m.license_type || m.main_topic || "";
    return `
    <tr>
      <td>${escapeHtml(fmtTime(parseFloat(m.superseded_at)))}</td>
      <td>${escapeHtml(m.source_file || "")}</td>
      <td>${escapeHtml(topic)}</td>
      <td>${escapeHtml(m.superseded_reason || "")}</td>
    </tr>`;
  }).join("");
}

async function loadAll() {
  try {
    const data = await apiGet("/admin/api/pdf-status");
    renderCost(data);
    renderActivity(data.recent_activity || []);
    renderArchived(data.recent_archived || []);
  } catch (e) {
    document.getElementById("costAmount").textContent = "โหลดข้อมูลไม่สำเร็จ: " + e.message;
  }
}

loadAll();
setInterval(loadAll, 30000);
</script>
</body>
</html>"""


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def admin_pdf_dashboard():
    return HTMLResponse(_DASHBOARD_HTML)

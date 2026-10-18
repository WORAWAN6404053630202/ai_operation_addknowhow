"""
PDF Ingestion Review Dashboard — FastAPI Application Entry Point
=================================================================
Separate app/process/port from the production chat bot (app.py) — split out
2026-09 after a missing PDF-only dependency (boto3) crashed the bot's shared
container (see router/admin.py's git history / the 2026-09-06/07 incident).
A crash in this service can no longer take the chat bot down, and vice versa.

Deliberately thin compared to app.py: no reranker preload, no cache-warming
threads, no vector-store readiness check — none of that applies to the PDF
review pipeline. This process only serves router/admin_pdf.py's endpoints.

static/admin.html (the combined dashboard shell) is still served by the BOT
service (app.py) — its PDF-related fetch calls point at this service's own
base URL/port instead of a relative path, which is why CORS is open here.

NOT ACTUALLY DEPLOYED for the pure-PDF EC2 instance (2026-10 discovery): that
instance's "web" container runs `uvicorn app:app` (the full bot), not this
file — this module exists and works standalone (e.g. for local testing),
but in practice app.py directly registers router/admin_pdf.py's two newest
routes (/admin/api/pdf-status, /admin/dashboard) itself rather than this
service being a separate running process there. See app.py's own comment
for why (router/admin.py, already included in app.py, has its own
un-removed duplicate of the older /api/pdf-queue routes this file also
defines — a wholesale include_router here would collide with those).
"""

import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import conf

from router.admin_pdf import router as admin_pdf_router
from utils.middleware import MonitoringMiddleware, HealthCheckMiddleware, BodySizeLimitMiddleware
from utils.logger import setup_logging, get_logger

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "human")
LOG_FILE = os.getenv("LOG_FILE", None)

setup_logging(level=LOG_LEVEL, log_format=LOG_FORMAT, log_file=LOG_FILE)
logger = get_logger(__name__)

logger.info(f"Starting PDF admin service with LOG_LEVEL={LOG_LEVEL}, LOG_FORMAT={LOG_FORMAT}")

_start_time = time.time()

app = FastAPI(
    title="Restbiz — PDF Ingestion Review",
    description="PDF ingestion review-queue admin API (split out of the main bot service)",
    version="1.0.0",
)

app.add_middleware(HealthCheckMiddleware)
app.add_middleware(MonitoringMiddleware, enable_debug=(LOG_LEVEL == "DEBUG"))

# Open CORS: the dashboard page (static/admin.html) is served by the bot
# service on a different port/origin and fetches this service's endpoints
# directly from the browser — same reasoning as app.py's own CORS setup.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(BodySizeLimitMiddleware, max_bytes=getattr(conf, "MAX_REQUEST_BODY_BYTES", 1_000_000))

app.include_router(admin_pdf_router)
# /admin/dashboard comes along automatically via the include above — the
# route now lives on admin_pdf_router itself (router/admin_pdf.py), not
# defined here, so it's reachable the same way wherever that router is
# included (this app, or app.py's direct registration — see app.py's own
# comment for why app.py registers the two new routes directly instead of
# including this whole router there).


@app.get("/health", tags=["health"], include_in_schema=False)
async def health():
    """Simple liveness check — this service has no vector store/model to warm
    up, so "ready" is just "the process is up", unlike the bot's /health."""
    return {
        "status": "ok",
        "ready": True,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_pdf:app", host="0.0.0.0", port=3001, reload=True)

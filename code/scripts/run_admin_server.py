# code/scripts/run_admin_server.py
"""Minimal standalone FastAPI server for the PDF Review Queue admin panel only
(feature/pdf-ingestion). Serves just router/admin_pdf.py's routes + this
file's own copy of the static/admin.html shell, skipping app.py's full stack
(chat routes, vector store, chromadb/langchain/sentence-transformers) — this
isolated EC2 process's venv only has the light dependencies the PDF review
flow actually needs, not the whole app's ML stack.

Updated 2026-09 (bot/PDF service-split work) to import the now-separate
router/admin_pdf.py instead of the old combined router/admin.py — this is the
service actually running in production for PDF review today (systemd unit
restbiz-pdf-admin.service, port 8001), so it gets its own dashboard-shell
route here rather than relying on the bot's admin_bot.py for that (which
would mean this "isolated" server suddenly depends on the bot being up to
show its own UI — defeats the point of isolating it).

static/admin.html's PDF_API_BASE constant is origin-aware: when the page is
loaded from this server directly (as it is today, via the route below), PDF
fetches stay same-origin/relative; only when the SAME html is loaded from the
bot's own admin dashboard (port 3000, future) does it cross-origin-fetch to
this server's port instead.

Usage:
    export RESTBIZ_ENV_FILE=env.dev.properties
    python code/scripts/run_admin_server.py --port 8001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

code_dir = Path(__file__).parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from router.admin_pdf import router as admin_pdf_router

app = FastAPI(title="Restbiz PDF Review Admin (dev, isolated)")
app.include_router(admin_pdf_router)


@app.get("/admin/", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard():
    """Same dashboard shell as router/admin_bot.py's admin_dashboard() — this
    server needs its own copy since it deliberately doesn't depend on the bot
    service being up at all to show its own UI."""
    html_path = code_dir / "static" / "admin.html"
    if not html_path.exists():
        return HTMLResponse("<h1>admin.html not found</h1>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok", "ready": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

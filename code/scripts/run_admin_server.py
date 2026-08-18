# code/scripts/run_admin_server.py
"""Minimal standalone FastAPI server for the PDF Review Queue admin panel only
(feature/pdf-ingestion). Serves just router/admin.py's routes + static/admin.html,
skipping app.py's full stack (chat routes, vector store, chromadb/langchain/
sentence-transformers) — this isolated EC2 process's venv only has the light
dependencies the PDF review flow actually needs, not the whole app's ML stack.

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

from router.admin import router as admin_router

app = FastAPI(title="Restbiz PDF Review Admin (dev, isolated)")
app.include_router(admin_router)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

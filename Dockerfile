# STAGE: DEVELOP (Development Environment)
FROM python:3.11-slim AS develop

# ติดตั้ง system dependencies สำหรับ sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv is what generated requirements*.txt's hashes (see file header) — plain pip
# fails on this lockfile's `uvicorn[standard]` extra under --require-hashes mode
# even though the file itself is correct; installing with uv avoids that entirely.
#
# 2026-10 fix (Infra2-M5): pinned by digest instead of the mutable `:latest`
# tag — a build tool that can silently change between builds is a supply-chain
# risk (an upstream push could land unreviewed between two otherwise-identical
# deploys). Digest resolved from `:latest` on 2026-10-20; bump deliberately
# (re-resolve + re-test) rather than letting it drift silently.
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:637bbca55303e5c6abee80ec3ef86c89f487e118e45b8d693a4a8889cee2bfe4 /uv /uvx /bin/

# Copy requirements ก่อนเพื่อใช้ Docker cache
COPY requirements.txt requirements-dev.txt /app/

# 2026-10 fix (Infra2-M2, revised after verifying with a real build): this
# stage used to have no CPU-only torch handling at all, unlike `prod` below
# — requirements-dev.txt deliberately excludes torch itself
# (--no-emit-package torch, see header), so when sentence-transformers pulls
# it in transitively, uv needs its own instruction on which build to pick.
# A first attempt added a `pip install torch==X+cpu` pre-install step (same
# as `prod` below always had) — a real `docker buildx build` then proved
# that pre-install line does NOT survive: uv re-resolves torch from scratch
# to satisfy sentence-transformers regardless of what's already installed,
# silently replacing it with a ~1.1GB CUDA build (confirmed via `docker run
# ... python3 -c "import torch; print(torch.__version__)"` returning
# `2.14.1+cu130`). `--torch-backend cpu` on the actual install step is what
# controls the real outcome — same flag this lockfile's own compile command
# already uses (see file header) — so the separate pre-install was pure dead
# weight once this flag is here; removed rather than left as inert cruft.
RUN uv pip install --system --no-cache --torch-backend cpu -r requirements-dev.txt

# Copy code และ data
COPY code/ /app/code/
COPY local_chroma_v3/ /app/local_chroma_v3/

# ตั้งค่า environment variables
ENV PYTHONPATH=/app/code
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 3000

# รัน uvicorn ด้วย auto-reload สำหรับ development
# --reload-exclude: code/data/* holds runtime-written files (session state json/lock,
# chroma dirs) — without this, every chat request's state write is picked up as a "code
# change" and restarts the worker mid-request (confirmed: a live pilot test triggered a
# restart that made the in-flight request wait 250s+ for a fresh reranker/BM25/embedding
# reload). Restricting to *.py changes only fixes this without weakening reload for real edits.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--reload", "--reload-exclude", "*/data/*"]

# STAGE: STAGING (Pre-production Testing)
FROM python:3.11-slim AS staging

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Infra2-M5: pinned by digest, see develop stage comment above
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:637bbca55303e5c6abee80ec3ef86c89f487e118e45b8d693a4a8889cee2bfe4 /uv /uvx /bin/

COPY requirements.txt /app/
# 2026-10 fix (Infra2-M2): --torch-backend cpu — see the matching comment on
# the develop stage above for why (verified via a real build) this is the
# one line that actually controls CPU-vs-CUDA torch, not a pre-install step.
RUN uv pip install --system --no-cache --torch-backend cpu -r requirements.txt

COPY code/ /app/code/
COPY local_chroma_v3/ /app/local_chroma_v3/

ENV PYTHONPATH=/app/code
ENV PYTHONUNBUFFERED=1

EXPOSE 3000

# Staging: production-like (ไม่มี --reload)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1"]

# STAGE: PRODUCTION (Optimized for Performance & Security)
FROM python:3.11-slim AS prod

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Infra2-M5: pinned by digest, see develop stage comment above
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:637bbca55303e5c6abee80ec3ef86c89f487e118e45b8d693a4a8889cee2bfe4 /uv /uvx /bin/

# Copy and install requirements.
# 2026-10 fix (Infra2-M4/M2, revised after verifying with a real build): this
# stage used to have a separate `pip install torch==X --index-url .../cpu`
# pre-install step here, with a comment claiming it meant "torch already
# installed above, so sentence-transformers skips it." A real `docker buildx
# build --target prod-pdf-full` proved that comment WRONG: uv doesn't see
# torch as pinned in requirements.txt (--no-emit-package torch, see file
# header) and, needing it to satisfy sentence-transformers, silently
# re-resolves and REPLACES whatever's already installed — confirmed
# `torch==2.14.1+cu130` (a ~1.1GB CUDA build) actually landed in the image
# before this fix, meaning the "~2.5 GB → ~500 MB" savings that pre-install
# step was supposed to buy was never actually happening in any deployed
# image. `--torch-backend cpu` on THIS install is what actually controls the
# outcome (verified: CPU-tagged torch, ~711MB) — same flag this lockfile's
# own compile command already uses (see file header) — so the separate
# pre-install step was pure dead weight once this flag is here; removed
# rather than left as inert cruft that still downloads an unused wheel.
COPY requirements.txt /app/
RUN uv pip install --system --no-cache --torch-backend cpu -r requirements.txt

# Copy application code
COPY code/ /app/code/
COPY local_chroma_v3/ /app/local_chroma_v3/
COPY pyproject.toml /app/

# Create necessary directories and set permissions
# /app/logs: 2026-10 fix (Infra-M10) — utils/logger.py's RotatingFileHandler
# (added 2026-09 after an undiagnosable incident, see incident_diagnosability
# memory) has had LOG_FILE unset in every env file since it shipped, so it
# has never actually activated in any deployment. Needs to exist and be
# writable by appuser before LOG_FILE=/app/logs/app.log (see env.properties)
# can write to it; docker-compose.prod.yaml mounts a host dir here so the
# file survives container recreation, the whole point of this fix.
RUN mkdir -p /app/code/sessions /app/chroma_db /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONPATH=/app/code
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Expose port
EXPOSE 3000

# Production: Multi-worker setup with optimization
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1", "--loop", "uvloop", "--http", "httptools"]

# STAGE: PROD-PDF (PDF ingestion review dashboard — separate service, 2026-09)
# Split off from the `prod` stage above so a PDF-only dependency crash (see the
# boto3 incident, 2026-09-06/07) can never take the chat bot's container down
# again. Deliberately does NOT install torch/chromadb/sentence-transformers/
# langchain — this service only serves router/admin_pdf.py's endpoints, never
# touches RAG/embeddings, and does not need local_chroma_v3/ at all.
FROM python:3.11-slim AS prod-pdf

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /app

# Infra2-M5: pinned by digest, see develop stage comment above
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:637bbca55303e5c6abee80ec3ef86c89f487e118e45b8d693a4a8889cee2bfe4 /uv /uvx /bin/

COPY requirements-pdf.txt /app/
RUN uv pip install --system --no-cache -r requirements-pdf.txt

# Only the PDF-relevant subtree + shared utils — see requirements-pdf.in's
# header comment for exactly which service/model/utils files this service
# actually imports.
COPY code/ /app/code/
COPY pyproject.toml /app/

# 2026-10 fix (Infra2-C2): env.properties sets LOG_FILE=/app/logs/app.log
# globally (added for the `prod` stage's rotating file log), but this
# lean stage never created that directory — app_pdf.py's setup_logging()
# call would open a RotatingFileHandler against a path that doesn't exist,
# raising FileNotFoundError before the app object is even constructed, and
# crash-loop docker-compose.prod.yaml's pdf-serve service. The `prod` stage
# right above creates /app/code/sessions /app/chroma_db /app/logs together;
# this stage only needs /app/logs since it has neither sessions nor chroma_db.
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app/code
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

EXPOSE 3001

CMD ["uvicorn", "app_pdf:app", "--host", "0.0.0.0", "--port", "3001", "--workers", "1", "--loop", "uvloop", "--http", "httptools"]

# STAGE: PROD-PDF-FULL (dedicated PDF-project instance, 2026-09) — the ONE
# image that needs to run BOTH the chat bot (uvicorn app:app, needs
# torch/chromadb/sentence-transformers from the `prod` stage above) AND
# service/sqs_consumer.py's direct-ingest extraction pipeline (needs
# pymupdf + typhoon-ocr, which are in NEITHER requirements.txt NOR
# requirements-pdf.txt — confirmed by grepping every requirements*.txt/.in
# file in this repo, 2026-09: requirements-pdf.in's own header comment
# names these as dependencies of sqs_consumer.py but they were never
# actually captured in any lockfile; the old EC2 instance's separate
# hand-rolled venv installed them ad hoc, untracked).
#
# Deliberately does NOT add these to requirements.in/.txt itself: that
# would put pymupdf/typhoon-ocr into the REAL production bot's image too,
# undoing the exact isolation the prod-pdf split above was created for
# (a PDF-only dependency crash must never take the chat bot down — see
# that stage's comment, the boto3 incident). This stage extends `prod`
# instead, so only a deployment that explicitly targets prod-pdf-full
# (this dedicated PDF-project instance) carries the extra risk.
#
# Version note: typhoon-ocr==0.4.1 was confirmed working against
# openai==3.8.0 on the old instance's venv — requirements.txt currently
# pins openai==2.53.0 for the base bot.
#
# 2026-10 fix (Infra2-M3): was fully unpinned ("let uv resolve compatible
# versions together") — pinning typhoon-ocr exactly to the one version this
# comment already calls out as confirmed-working. Left pymupdf's minor/patch
# floating (uv still resolves it freely against openai/typhoon-ocr) but
# capped below the just-released pymupdf 2.0 — a major-version jump with an
# unknown API surface that was never run against this codebase's actual
# PyMuPDF usage (fitz.open/page extraction in service/pdf_*.py), so letting
# a routine rebuild silently pick it up is the real risk being closed here,
# not reproducibility down to the exact patch.
FROM prod AS prod-pdf-full

USER root
RUN pip install --no-cache-dir "pymupdf>=1.28,<2" typhoon-ocr==0.4.1
USER appuser

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1", "--loop", "uvloop", "--http", "httptools"]
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
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy requirements ก่อนเพื่อใช้ Docker cache
COPY requirements.txt requirements-dev.txt /app/

# ติดตั้ง Python dependencies รวม dev tools (gradio, pytest)
RUN uv pip install --system --no-cache -r requirements-dev.txt

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

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY requirements.txt /app/
RUN uv pip install --system --no-cache -r requirements.txt

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

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install CPU-only PyTorch first to avoid pulling in the GPU build (~2.5 GB → ~500 MB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements (torch already installed above, so sentence-transformers skips it)
COPY requirements.txt /app/
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY code/ /app/code/
COPY local_chroma_v3/ /app/local_chroma_v3/
COPY pyproject.toml /app/

# Create necessary directories and set permissions
RUN mkdir -p /app/code/sessions /app/chroma_db && \
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

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY requirements-pdf.txt /app/
RUN uv pip install --system --no-cache -r requirements-pdf.txt

# Only the PDF-relevant subtree + shared utils — see requirements-pdf.in's
# header comment for exactly which service/model/utils files this service
# actually imports.
COPY code/ /app/code/
COPY pyproject.toml /app/

RUN chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app/code
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

EXPOSE 3001

CMD ["uvicorn", "app_pdf:app", "--host", "0.0.0.0", "--port", "3001", "--workers", "1", "--loop", "uvloop", "--http", "httptools"]
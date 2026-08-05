# STAGE: DEVELOP (Development Environment)
FROM python:3.11-slim AS develop

# ติดตั้ง system dependencies สำหรับ sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements ก่อนเพื่อใช้ Docker cache
COPY requirements.txt requirements-dev.txt /app/

# ติดตั้ง Python dependencies รวม dev tools (gradio, pytest)
RUN pip install --no-cache-dir -r requirements-dev.txt

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

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

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

# Install CPU-only PyTorch first to avoid pulling in the GPU build (~2.5 GB → ~500 MB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements (torch already installed above, so sentence-transformers skips it)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

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
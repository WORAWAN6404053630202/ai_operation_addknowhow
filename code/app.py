"""
FastAPI Application Entry Point
================================
นี่คือไฟล์หลักที่เปิดตัว web server

สิ่งที่ไฟล์นี้ทำ:
1. สร้าง FastAPI app object
2. เชื่อมต่อ router (เส้นทาง API ทั้งหมดอยู่ใน router/route_v1.py)
3. เสิร์ฟไฟล์ static (รูป, CSS, JS)
4. เสิร์ฟหน้าเว็บ HTML หลัก (static/index.html)
5. ติดตั้ง monitoring middleware และ logging
"""

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import conf

from router.route_v1 import api_v1
from router.monitoring import router as monitoring_router
from router.admin import router as admin_router
from utils.middleware import MonitoringMiddleware, HealthCheckMiddleware, BodySizeLimitMiddleware
from utils.logger import setup_logging, get_logger

# Setup logging based on environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "human")  # "human" or "json"
LOG_FILE = os.getenv("LOG_FILE", None)

setup_logging(level=LOG_LEVEL, log_format=LOG_FORMAT, log_file=LOG_FILE)
logger = get_logger(__name__)

logger.info(f"Starting application with LOG_LEVEL={LOG_LEVEL}, LOG_FORMAT={LOG_FORMAT}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # route_v1's cache-warming threads (topic pool / BM25 / op-group classifier) must
    # start here, not at route_v1 module import time — see start_prewarm_threads'
    # docstring for why (duplicated paid LLM calls under uvicorn --reload otherwise).
    from router.route_v1 import start_prewarm_threads
    start_prewarm_threads()

    import conf as _conf
    if getattr(_conf, "RERANKER_ENABLED", False):
        model_name = getattr(_conf, "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        backend = getattr(_conf, "RERANKER_BACKEND", "pytorch")

        def _load():
            try:
                if backend == "onnx":
                    # Preload the ONNX path actually being used — preloading the
                    # pytorch model here (the old unconditional behavior) would waste
                    # startup time/RAM on a model no request will ever call, while
                    # leaving the real (ONNX) model to cold-load on the first live
                    # request instead.
                    from utils.reranker import _get_onnx_reranker
                    onnx_file = getattr(_conf, "RERANKER_ONNX_FILE", "model_quint8_avx2.onnx")
                    _get_onnx_reranker(model_name, onnx_file)
                    logger.info("[Startup] Reranker (onnx) model ready: %s (onnx/%s)", model_name, onnx_file)
                else:
                    from utils.reranker import _get_reranker
                    _get_reranker(model_name)
                    logger.info("[Startup] Reranker (pytorch) model ready: %s", model_name)
            except Exception as e:
                logger.warning("[Startup] Reranker preload failed (will retry on first request): %s", e)

        threading.Thread(target=_load, daemon=True, name="reranker-preload").start()
        logger.info("[Startup] Reranker preload started in background (model=%s, backend=%s)", model_name, backend)
    yield


app = FastAPI(
    title="Restbiz — น้องสุดยอด",
    description="Thai Regulatory AI Assistant for restaurant businesses",
    version="1.0.0",
    lifespan=lifespan,
)

# Add monitoring middleware (before CORS)
app.add_middleware(HealthCheckMiddleware)
app.add_middleware(MonitoringMiddleware, enable_debug=(LOG_LEVEL == "DEBUG"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # No cookies/browser-ambient auth anywhere in this app (session_id travels
    # in the JSON body, not a cookie) — allow_credentials=True combined with a
    # wildcard origin is spec-invalid anyway (browsers reject it) and was
    # never actually exercised. Dropping it makes this a valid, intentional
    # open API instead of a misconfigured one.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Added LAST so it's the OUTERMOST middleware (Starlette runs add_middleware
# calls most-recently-added-first) — must wrap CORS/Monitoring/HealthCheck so
# none of them (MonitoringMiddleware peeks the body for logging) ever sees a
# request body larger than conf.MAX_REQUEST_BODY_BYTES.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=getattr(conf, "MAX_REQUEST_BODY_BYTES", 1_000_000))

# Include routers
app.include_router(api_v1, prefix="/api/v1")
app.include_router(monitoring_router)
app.include_router(admin_router)


@app.get("/health", tags=["health"], include_in_schema=False)
async def root_health():
    """
    Root-level health check — used by Docker/load-balancer.
    Returns 200 only after the embedding model and vector store are fully loaded.
    """
    import time
    from router.monitoring import _start_time
    try:
        from service.local_vector_store import get_vs_manager
        mgr = get_vs_manager()
        ready = mgr.vectorstore is not None and (mgr._collection_count() or 0) > 0
    except Exception as e:
        logger.warning(f"[health] vectorstore check failed: {e}")
        ready = False
    return {
        "status": "ok" if ready else "starting",
        "ready": ready,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = static_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>index.html not found in code/static/</h1>", status_code=404)

    html = HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    html.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    html.headers["Pragma"] = "no-cache"
    html.headers["Expires"] = "0"
    return html


if __name__ == "__main__":
    import uvicorn
    # reload_excludes: data/* holds runtime-written files (session state json/lock,
    # chroma dirs) — without this, every chat request's state write is picked up as a
    # "code change" and restarts the worker mid-request. See Dockerfile dev CMD comment.
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True, reload_excludes=["data/*"])
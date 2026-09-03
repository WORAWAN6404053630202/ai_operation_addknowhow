"""
Monitoring Middleware for FastAPI
Provides request tracking, logging, and metrics collection.
"""

import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable

from utils.logger import get_logger, set_request_context, clear_request_context
from utils.metrics import metrics

logger = get_logger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request monitoring
    - Assigns request IDs
    - Logs all requests/responses
    - Collects metrics
    - Handles errors gracefully
    """
    
    def __init__(self, app: ASGIApp, enable_debug: bool = False):
        super().__init__(app)
        self.enable_debug = enable_debug
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract session ID if present
        session_id = None
        if request.method == "POST":
            try:
                body = await request.body()
                # Re-populate body for downstream handlers
                request._body = body
                
                import json
                data = json.loads(body) if body else {}
                session_id = data.get("session_id")
            except Exception:
                pass
        
        # Set logging context
        set_request_context(request_id=request_id, session_id=session_id)
        
        # Log request
        logger.info(
            f"→ {request.method} {request.url.path}",
            extra={'extra_data': {
                'request_id': request_id,
                'session_id': session_id,
                'client': request.client.host if request.client else None,
                'user_agent': request.headers.get('user-agent', 'unknown')
            }}
        )
        
        # Track timing
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Log response
            logger.info(
                f"← {request.method} {request.url.path} → {response.status_code}",
                extra={'extra_data': {
                    'request_id': request_id,
                    'status_code': response.status_code,
                    'elapsed_ms': round(elapsed_ms, 2)
                }}
            )
            
            # Record metrics
            metrics.record_request(
                session_id=session_id or "unknown",
                request_id=request_id,
                endpoint=request.url.path,
                elapsed_ms=elapsed_ms,
                success=response.status_code < 400
            )
            
            # Add headers
            response.headers['X-Request-ID'] = request_id
            if self.enable_debug:
                response.headers['X-Response-Time'] = f"{elapsed_ms:.2f}ms"
            
            return response
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Log error
            logger.error(
                f"✗ {request.method} {request.url.path} failed",
                exc_info=True,
                extra={'extra_data': {
                    'request_id': request_id,
                    'elapsed_ms': round(elapsed_ms, 2),
                    'error': str(e)
                }}
            )
            
            # Record error
            metrics.record_request(
                session_id=session_id or "unknown",
                request_id=request_id,
                endpoint=request.url.path,
                elapsed_ms=elapsed_ms,
                success=False,
                error=str(e)
            )
            
            # Re-raise to let FastAPI handle
            raise
            
        finally:
            # Clear context
            clear_request_context()


class BodySizeLimitMiddleware:
    """
    Raw ASGI middleware enforcing a hard cap on POST/PUT/PATCH body size.

    Must run BEFORE any downstream code reads the body — including
    MonitoringMiddleware.dispatch()'s own `await request.body()` peek (used
    just to log session_id), which wraps that read in a bare
    `except Exception: pass`. A naive approach that raises once a streamed
    body exceeds the cap would get silently swallowed right there and never
    actually protect anything. Instead this middleware drains and
    bound-checks the real ASGI stream itself, then hands downstream code a
    small in-memory replay of the already-verified-safe body — so nothing
    downstream can read more than max_bytes no matter how many times (or
    where) it calls request.body().

    Must be the outermost middleware (added LAST in app.py — Starlette runs
    add_middleware() calls in reverse order, most-recently-added first) so
    every layer below it — CORS, monitoring, health-check, routing — only
    ever sees the bounded replay, never the raw stream.
    """

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                pass  # malformed header — fall through to the real byte count below

        chunks = []
        total = 0
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_bytes:
                await self._reject(send)
                return
            chunks.append(chunk)
            more_body = message.get("more_body", False)

        body = b"".join(chunks)
        already_sent = False

        async def replay_receive():
            nonlocal already_sent
            if not already_sent:
                already_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            # After the replayed body, proxy further calls to the real
            # transport instead of synthesizing http.disconnect. Starlette's
            # StreamingResponse (used by /chat/stream's SSE) runs a
            # concurrent task that calls receive() again almost immediately
            # to watch for a real client disconnect — a synthetic disconnect
            # here would fire that instantly and kill the stream mid-flight
            # (caught by testing the SSE endpoint end-to-end: response
            # started fine but the body came back empty).
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _reject(send):
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({
            "type": "http.response.body",
            "body": b'{"detail":"Request body too large"}',
        })


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """
    Fast path for health checks
    Bypasses logging and metrics for /health endpoint
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip monitoring for health checks
        if request.url.path in ["/health", "/healthz", "/ping"]:
            return await call_next(request)
        
        return await call_next(request)

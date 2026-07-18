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

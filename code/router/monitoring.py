"""
Health Check and Monitoring Endpoints
Provides system health status and metrics for monitoring.
"""

import time
import psutil
import os
from datetime import datetime
from fastapi import APIRouter, Depends, Response
from typing import Dict, Any

from utils.admin_auth import require_admin_key
from utils.metrics import metrics
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

_cached_state_manager = None


# Startup time
_start_time = time.time()


def _get_system_info() -> Dict[str, Any]:
    """Get system resource information"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count()
            },
            'memory': {
                'total_mb': round(memory.total / 1024 / 1024, 2),
                'used_mb': round(memory.used / 1024 / 1024, 2),
                'percent': memory.percent
            },
            'disk': {
                'total_gb': round(disk.total / 1024 / 1024 / 1024, 2),
                'used_gb': round(disk.used / 1024 / 1024 / 1024, 2),
                'percent': disk.percent
            }
        }
    except Exception as e:
        logger.warning(f"Could not get system info: {e}")
        return {}


def _check_services() -> Dict[str, bool]:
    """Check if critical services are available"""
    checks = {}
    
    # Check ChromaDB
    try:
        from service.local_vector_store import get_vs_manager
        mgr = get_vs_manager()
        count = mgr._collection_count() or 0
        checks['chromadb'] = count > 0
    except Exception as e:
        logger.warning(f"ChromaDB check failed: {e}")
        checks['chromadb'] = False

    # Check OpenRouter API key
    try:
        from conf import OPENROUTER_API_KEY
        checks['openrouter_key'] = bool(OPENROUTER_API_KEY)
    except Exception as e:
        logger.warning(f"OpenRouter key check failed: {e}")
        checks['openrouter_key'] = False

    # Check session storage
    try:
        global _cached_state_manager
        if _cached_state_manager is None:
            from model.state_manager import StateManager
            _cached_state_manager = StateManager()
        checks['session_storage'] = True
    except Exception as e:
        logger.warning(f"Session storage check failed: {e}")
        checks['session_storage'] = False
    
    return checks


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint
    Returns 200 if service is alive
    """
    uptime = time.time() - _start_time
    
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': round(uptime, 2)
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with service status
    """
    uptime = time.time() - _start_time
    services = _check_services()
    system = _get_system_info()
    
    # Overall health
    all_healthy = all(services.values())
    
    return {
        'status': 'healthy' if all_healthy else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': round(uptime, 2),
        'services': services,
        'system': system
    }


@router.get("/metrics")
async def get_metrics():
    """
    Get application metrics
    """
    return metrics.get_summary()


@router.get("/metrics/llm")
async def get_llm_metrics():
    """
    Get LLM usage metrics
    """
    return metrics.get_llm_stats()


@router.get("/metrics/requests")
async def get_request_metrics():
    """
    Get request metrics
    """
    return metrics.get_request_stats()


@router.get("/metrics/prometheus", response_class=Response)
async def prometheus_metrics():
    """
    Export metrics in Prometheus format
    Compatible with Prometheus scraping
    """
    lines = []
    
    # LLM metrics
    llm_stats = metrics.get_llm_stats()
    lines.append(f"# HELP llm_calls_total Total number of LLM calls")
    lines.append(f"# TYPE llm_calls_total counter")
    lines.append(f"llm_calls_total {llm_stats.get('total_calls', 0)}")
    
    lines.append(f"# HELP llm_tokens_total Total tokens consumed")
    lines.append(f"# TYPE llm_tokens_total counter")
    lines.append(f"llm_tokens_total {llm_stats.get('total_tokens', 0)}")
    
    lines.append(f"# HELP llm_latency_ms Average LLM latency in milliseconds")
    lines.append(f"# TYPE llm_latency_ms gauge")
    lines.append(f"llm_latency_ms {llm_stats.get('avg_latency_ms', 0)}")
    
    # Request metrics
    req_stats = metrics.get_request_stats()
    lines.append(f"# HELP requests_total Total number of requests")
    lines.append(f"# TYPE requests_total counter")
    lines.append(f"requests_total {req_stats.get('total_requests', 0)}")
    
    lines.append(f"# HELP requests_errors_total Total number of failed requests")
    lines.append(f"# TYPE requests_errors_total counter")
    lines.append(f"requests_errors_total {req_stats.get('failed_requests', 0)}")
    
    lines.append(f"# HELP request_latency_ms Average request latency in milliseconds")
    lines.append(f"# TYPE request_latency_ms gauge")
    lines.append(f"request_latency_ms {req_stats.get('avg_latency_ms', 0)}")
    
    # System metrics
    try:
        system = _get_system_info()
        if 'cpu' in system:
            lines.append(f"# HELP cpu_usage_percent CPU usage percentage")
            lines.append(f"# TYPE cpu_usage_percent gauge")
            lines.append(f"cpu_usage_percent {system['cpu']['percent']}")
        
        if 'memory' in system:
            lines.append(f"# HELP memory_usage_percent Memory usage percentage")
            lines.append(f"# TYPE memory_usage_percent gauge")
            lines.append(f"memory_usage_percent {system['memory']['percent']}")
    except Exception:
        pass
    
    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4"
    )


@router.get("/debug/sessions", dependencies=[Depends(require_admin_key)])
async def list_sessions():
    """
    List active sessions (debug endpoint)
    """
    try:
        from model.state_manager import StateManager
        sm = StateManager()

        # Get all session files
        import glob
        session_files = glob.glob(str(sm.dir / "s_*.json"))
        
        sessions = []
        for file in session_files:
            try:
                import json
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sessions.append({
                        'session_id': data.get('session_id'),
                        'current_phase': data.get('current_phase'),
                        'last_modified': os.path.getmtime(file),
                        'file_size': os.path.getsize(file)
                    })
            except Exception:
                pass
        
        return {
            'total_sessions': len(sessions),
            'sessions': sorted(sessions, key=lambda x: x['last_modified'], reverse=True)
        }
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return {'error': str(e)}


@router.post("/debug/reset-metrics", dependencies=[Depends(require_admin_key)])
async def reset_metrics():
    """
    Reset all metrics (debug endpoint)
    """
    metrics.reset()
    logger.info("Metrics reset by user")
    return {'status': 'ok', 'message': 'Metrics reset successfully'}

"""
Professional Logging System
Provides structured logging with request tracking, performance metrics, and debug capabilities.
"""

import logging
import sys
import time
import json
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
from functools import wraps

# Context variable for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    Makes logs machine-readable and easy to parse/analyze
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add request context if available
        request_id = request_id_var.get()
        if request_id:
            log_data['request_id'] = request_id
            
        session_id = session_id_var.get()
        if session_id:
            log_data['session_id'] = session_id
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add custom fields
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for development
    Colorized output with clear structure
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format: [TIME] LEVEL | logger | message
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        parts = [
            f"{color}[{timestamp}]{reset}",
            f"{color}{record.levelname:8s}{reset}",
            f"{record.name:30s}",
            record.getMessage()
        ]
        
        # Add request/session context
        request_id = request_id_var.get()
        session_id = session_id_var.get()
        if request_id or session_id:
            context = []
            if request_id:
                context.append(f"req={request_id[:8]}")
            if session_id:
                context.append(f"sess={session_id[:8]}")
            parts.append(f"[{' '.join(context)}]")
        
        message = " | ".join(parts)
        
        # Add extra_data fields inline (human-readable)
        if hasattr(record, 'extra_data') and isinstance(record.extra_data, dict):
            skip = {'action', 'persona', 'total_docs', 'doc_index'}
            extras = {k: v for k, v in record.extra_data.items() if k not in skip and v not in (None, '', [])}
            if extras:
                kv = '  '.join(f"{k}={v!r}" for k, v in extras.items())
                message += f"\n                                                    ↳ {kv}"
        
        # Add exception if present
        if record.exc_info:
            message += "\n" + "".join(traceback.format_exception(*record.exc_info))
        
        return message


def setup_logging(
    level: str = "INFO",
    log_format: str = "human",  # "human" or "json"
    log_file: Optional[str] = None
):
    """
    Setup application logging
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: "human" for development, "json" for production
        log_file: Optional file path to write logs
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set level
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Choose formatter
    if log_format == "json":
        formatter = StructuredFormatter()
    else:
        formatter = HumanReadableFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with extra methods
    
    Usage:
        logger = get_logger(__name__)
        logger.log_with_data("info", "Message", {'key': 'value'})
    """
    logger = logging.getLogger(name)
    
    # Add custom methods
    def log_with_data(level, message, extra_data=None, **kwargs):
        # Convert string level to int
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        if extra_data:
            record = logger.makeRecord(
                logger.name, level, "(unknown file)", 0,
                message, (), None, **kwargs
            )
            record.extra_data = extra_data
            logger.handle(record)
        else:
            logger.log(level, message, **kwargs)
    
    # Attach method to logger
    logger.log_with_data = log_with_data
    logger.debug_data = lambda msg, **data: log_with_data(logging.DEBUG, msg, data)
    logger.info_data = lambda msg, **data: log_with_data(logging.INFO, msg, data)
    logger.warning_data = lambda msg, **data: log_with_data(logging.WARNING, msg, data)
    logger.error_data = lambda msg, **data: log_with_data(logging.ERROR, msg, data)
    
    return logger


def set_request_context(request_id: Optional[str] = None, session_id: Optional[str] = None):
    """Set request/session context for logging"""
    if request_id:
        request_id_var.set(request_id)
    if session_id:
        session_id_var.set(session_id)


def clear_request_context():
    """Clear request context"""
    request_id_var.set(None)
    session_id_var.set(None)


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with timing
    
    Usage:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            logger.debug(f"→ Calling {func_name}")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.debug(
                    f"← {func_name} completed",
                    extra={'extra_data': {'elapsed_ms': round(elapsed * 1000, 2)}}
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"✗ {func_name} failed after {elapsed:.2f}s: {e}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


# Performance tracking
class PerformanceTracker:
    """Track performance metrics for operations"""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
        self.metrics = {}
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info_data(
                f"✓ {self.operation} completed",
                elapsed_ms=round(elapsed * 1000, 2),
                **self.metrics
            )
        else:
            self.logger.error_data(
                f"✗ {self.operation} failed",
                elapsed_ms=round(elapsed * 1000, 2),
                error=str(exc_val),
                **self.metrics
            )
    
    def add_metric(self, key: str, value: Any):
        """Add a metric to track"""
        self.metrics[key] = value
    
    def checkpoint(self, name: str):
        """Log a checkpoint"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.logger.debug(
                f"  ├─ {name}: {elapsed:.3f}s"
            )


# Alias for backward compatibility
TimingContext = PerformanceTracker


# Example usage in docstring
"""
Example Usage:
==============

# Setup logging
from code.utils.logger import setup_logging, get_logger, set_request_context, PerformanceTracker

setup_logging(level="DEBUG", log_format="human")
logger = get_logger(__name__)

# Basic logging
logger.info("Server started")
logger.warning("API key not found", extra={'extra_data': {'key_name': 'OPENROUTER_API_KEY'}})

# Request context
set_request_context(request_id="abc123", session_id="sess456")
logger.info("Processing request")  # Will include request_id and session_id

# Performance tracking
with PerformanceTracker(logger, "LLM Call") as tracker:
    response = call_llm(prompt)
    tracker.add_metric("tokens", response.usage.total_tokens)
    tracker.checkpoint("Got response")
    result = process_response(response)
    tracker.checkpoint("Processed")

# Function decorator
@log_function_call(logger)
def expensive_operation():
    time.sleep(1)
    return "result"
"""

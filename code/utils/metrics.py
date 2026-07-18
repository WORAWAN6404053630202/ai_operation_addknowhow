"""
Metrics Collection System
Tracks application performance, LLM usage, and business metrics.
"""

import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import threading


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM call"""
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_ms: float
    success: bool
    error: Optional[str] = None
    persona: Optional[str] = None  # academic/practical/supervisor
    operation: Optional[str] = None  # greet/topic_picker/answer/etc


@dataclass
class RequestMetrics:
    """Metrics for a user request"""
    timestamp: datetime
    session_id: str
    request_id: str
    endpoint: str
    elapsed_ms: float
    success: bool
    llm_calls: int = 0
    total_tokens: int = 0
    retrieval_count: int = 0
    error: Optional[str] = None


class MetricsCollector:
    """
    Centralized metrics collection
    Thread-safe singleton for collecting and querying metrics
    """
    
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.llm_calls: deque = deque(maxlen=10000)
        self.requests: deque = deque(maxlen=5000)
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        self._lock = threading.Lock()
    
    # === LLM Metrics ===
    
    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        persona: Optional[str] = None,
        operation: Optional[str] = None
    ):
        """Record an LLM API call"""
        metric = LLMCallMetrics(
            timestamp=datetime.now(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_ms=elapsed_ms,
            success=success,
            error=error,
            persona=persona,
            operation=operation
        )
        
        with self._lock:
            self.llm_calls.append(metric)
            
            # Update counters
            self.counters['llm_calls_total'] += 1
            self.counters['llm_tokens_total'] += metric.total_tokens
            if not success:
                self.counters['llm_errors_total'] += 1
            
            # Track by model
            self.counters[f'llm_calls_{model}'] += 1
            self.counters[f'llm_tokens_{model}'] += metric.total_tokens
            
            # Track timing
            self.timers['llm_latency'].append(elapsed_ms)
    
    def get_llm_stats(self, last_n: Optional[int] = None) -> Dict:
        """Get LLM usage statistics"""
        with self._lock:
            calls = list(self.llm_calls)[-last_n:] if last_n else list(self.llm_calls)
            
            if not calls:
                return {
                    'total_calls': 0,
                    'total_tokens': 0,
                    'avg_latency_ms': 0,
                    'success_rate': 0
                }
            
            total_calls = len(calls)
            successful = sum(1 for c in calls if c.success)
            total_tokens = sum(c.total_tokens for c in calls)
            avg_latency = sum(c.elapsed_ms for c in calls) / total_calls
            
            # Group by model
            by_model = defaultdict(lambda: {'calls': 0, 'tokens': 0})
            for call in calls:
                by_model[call.model]['calls'] += 1
                by_model[call.model]['tokens'] += call.total_tokens
            
            # Group by persona
            by_persona = defaultdict(lambda: {'calls': 0, 'tokens': 0})
            for call in calls:
                if call.persona:
                    by_persona[call.persona]['calls'] += 1
                    by_persona[call.persona]['tokens'] += call.total_tokens
            
            return {
                'total_calls': total_calls,
                'successful_calls': successful,
                'failed_calls': total_calls - successful,
                'success_rate': successful / total_calls,
                'total_tokens': total_tokens,
                'avg_tokens_per_call': total_tokens / total_calls,
                'avg_latency_ms': round(avg_latency, 2),
                'min_latency_ms': round(min(c.elapsed_ms for c in calls), 2),
                'max_latency_ms': round(max(c.elapsed_ms for c in calls), 2),
                'by_model': dict(by_model),
                'by_persona': dict(by_persona)
            }
    
    # === Request Metrics ===
    
    def record_request(
        self,
        session_id: str,
        request_id: str,
        endpoint: str,
        elapsed_ms: float,
        success: bool = True,
        llm_calls: int = 0,
        total_tokens: int = 0,
        retrieval_count: int = 0,
        error: Optional[str] = None
    ):
        """Record a user request"""
        metric = RequestMetrics(
            timestamp=datetime.now(),
            session_id=session_id,
            request_id=request_id,
            endpoint=endpoint,
            elapsed_ms=elapsed_ms,
            success=success,
            llm_calls=llm_calls,
            total_tokens=total_tokens,
            retrieval_count=retrieval_count,
            error=error
        )
        
        with self._lock:
            self.requests.append(metric)
            
            # Update counters
            self.counters['requests_total'] += 1
            if not success:
                self.counters['requests_errors_total'] += 1
            
            # Track timing
            self.timers['request_latency'].append(elapsed_ms)
    
    def get_request_stats(self, last_n: Optional[int] = None) -> Dict:
        """Get request statistics"""
        with self._lock:
            requests = list(self.requests)[-last_n:] if last_n else list(self.requests)
            
            if not requests:
                return {
                    'total_requests': 0,
                    'avg_latency_ms': 0,
                    'success_rate': 0
                }
            
            total = len(requests)
            successful = sum(1 for r in requests if r.success)
            avg_latency = sum(r.elapsed_ms for r in requests) / total
            avg_llm_calls = sum(r.llm_calls for r in requests) / total
            
            return {
                'total_requests': total,
                'successful_requests': successful,
                'failed_requests': total - successful,
                'success_rate': successful / total,
                'avg_latency_ms': round(avg_latency, 2),
                'min_latency_ms': round(min(r.elapsed_ms for r in requests), 2),
                'max_latency_ms': round(max(r.elapsed_ms for r in requests), 2),
                'avg_llm_calls_per_request': round(avg_llm_calls, 2)
            }
    
    # === Generic Counters ===
    
    def increment(self, counter: str, value: int = 1):
        """Increment a counter"""
        with self._lock:
            self.counters[counter] += value
    
    def record_timing(self, timer: str, elapsed_ms: float):
        """Record a timing measurement"""
        with self._lock:
            self.timers[timer].append(elapsed_ms)
    
    def get_counter(self, counter: str) -> int:
        """Get counter value"""
        with self._lock:
            return self.counters.get(counter, 0)
    
    def get_timer_stats(self, timer: str) -> Dict:
        """Get timer statistics"""
        with self._lock:
            values = self.timers.get(timer, [])
            if not values:
                return {'count': 0, 'avg': 0, 'min': 0, 'max': 0}
            
            return {
                'count': len(values),
                'avg': round(sum(values) / len(values), 2),
                'min': round(min(values), 2),
                'max': round(max(values), 2)
            }
    
    # === Summary ===
    
    def get_summary(self) -> Dict:
        """Get comprehensive metrics summary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'llm': self.get_llm_stats(),
            'requests': self.get_request_stats(),
            'counters': dict(self.counters),
            'timers': {k: self.get_timer_stats(k) for k in self.timers.keys()}
        }
    
    def reset(self):
        """Reset all metrics (useful for testing)"""
        with self._lock:
            self.llm_calls.clear()
            self.requests.clear()
            self.counters.clear()
            self.timers.clear()


# Singleton instance
metrics = MetricsCollector()


# Context manager for timing
class timer:
    """
    Context manager for timing operations
    
    Usage:
        with timer(metrics, 'database_query'):
            result = db.query(...)
    """
    
    def __init__(self, metrics_collector: MetricsCollector, name: str):
        self.metrics = metrics_collector
        self.name = name
        self.start = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self.start) * 1000
        self.metrics.record_timing(self.name, elapsed_ms)


# Decorator for function timing
def track_time(timer_name: str):
    """
    Decorator to track function execution time
    
    Usage:
        @track_time('my_function')
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.time() - start) * 1000
                metrics.record_timing(timer_name, elapsed_ms)
        return wrapper
    return decorator

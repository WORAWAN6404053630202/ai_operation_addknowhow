"""Thread-local progress-callback bridge for streaming real pipeline-stage signals
to the SSE endpoint without threading a parameter through the whole call chain.

Must be set on the same worker thread that runs supervisor.handle() (thread-locals
don't cross the event-loop-thread -> executor-thread boundary) — see route_v1.py's
_run_with_progress wrapper.
"""
import threading
from typing import Callable, Optional

_local = threading.local()


def set_progress_callback(cb: Optional[Callable[[str], None]]) -> None:
    _local.cb = cb


def clear_progress_callback() -> None:
    _local.cb = None


def emit_progress(text: str) -> None:
    cb = getattr(_local, "cb", None)
    if cb:
        try:
            cb(text)
        except Exception:
            pass

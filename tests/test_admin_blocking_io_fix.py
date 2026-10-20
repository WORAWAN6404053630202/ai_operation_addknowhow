"""Tests for the 2026-10 fix (core2-M5) to router/admin.py's admin_sessions/
admin_session_detail/admin_stats/admin_logs.

Background: all four were `async def` doing real blocking disk I/O
directly on the single-threaded asyncio event loop — unlike route_v1.py,
which documents extensively why that stalls every concurrent user's
/chat/stream SSE connection, not just the admin request itself.
admin_sessions was also N+1: model.state_manager.StateManager.list_sessions
already parses every session file in full, but admin_sessions used to ALSO
call .load(sid) per row just to re-extract token/cost fields list_sessions
had already read. Both are fixed: list_sessions now returns those fields
directly, and all four handlers route their blocking work through
run_in_executor.

These call the handler functions directly (not through a FastAPI app/
TestClient) to test behavior without needing to wire up the `_auth`
dependency injection — same approach tests/test_admin_auth.py already
uses for this router's auth dependency itself.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import router.admin as admin  # noqa: E402
from model.state_manager import StateManager  # noqa: E402


def _write_session_file(tmp_path: Path, session_id: str, *, user_msgs=1, cost=0.01) -> None:
    messages = []
    for i in range(user_msgs):
        messages.append({"role": "user", "content": f"question {i}"})
        messages.append({"role": "assistant", "content": f"answer {i}"})
    data = {
        "session_id": session_id,
        "persona_id": "practical",
        "messages": messages,
        "context": {},
        "total_prompt_tokens": 100,
        "total_completion_tokens": 50,
        "total_cost": cost,
    }
    (tmp_path / f"{session_id}.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.unit
class TestListSessionsReturnsAggregatesDirectly:
    """StateManager.list_sessions now computes message counts/token/cost
    fields itself (it has already parsed the full file) instead of callers
    re-opening the same file a second time."""

    def test_returns_message_and_cost_aggregates(self, tmp_path):
        sm = StateManager(persist_dir=str(tmp_path))
        _write_session_file(tmp_path, "s1", user_msgs=2, cost=0.0345)

        sessions = sm.list_sessions(limit=10)

        assert len(sessions) == 1
        s = sessions[0]
        assert s["total_messages"] == 4
        assert s["user_messages"] == 2
        assert s["bot_messages"] == 2
        assert s["prompt_tokens"] == 100
        assert s["completion_tokens"] == 50
        assert s["total_cost"] == 0.0345


@pytest.mark.unit
class TestAdminHandlersUseRunInExecutor:
    def test_admin_sessions_does_not_reload_each_session(self, tmp_path, monkeypatch):
        """The actual N+1 bug this closes: before the fix, admin_sessions
        called _state_manager.load(sid) once per row in addition to
        list_sessions. After the fix, .load() must never be called at all
        — everything needed comes from list_sessions directly."""
        sm = StateManager(persist_dir=str(tmp_path))
        _write_session_file(tmp_path, "s1", user_msgs=1, cost=0.01)
        _write_session_file(tmp_path, "s2", user_msgs=3, cost=0.02)
        monkeypatch.setattr(admin, "_state_manager", sm)

        with patch.object(sm, "load", side_effect=AssertionError(".load() must not be called by admin_sessions")):
            result = asyncio.run(admin.admin_sessions(limit=10))

        body = json.loads(result.body)
        assert body["total"] == 2
        total_costs = sorted(row["total_cost"] for row in body["sessions"])
        assert total_costs == [0.01, 0.02]

    def test_admin_session_detail_runs_via_executor_and_returns_full_state(self, tmp_path, monkeypatch):
        sm = StateManager(persist_dir=str(tmp_path))
        _write_session_file(tmp_path, "s1", user_msgs=2, cost=0.05)
        monkeypatch.setattr(admin, "_state_manager", sm)

        result = asyncio.run(admin.admin_session_detail("s1"))
        body = json.loads(result.body)
        assert body["session_id"] == "s1"
        assert body["total_messages"] == 4
        assert body["total_cost"] == 0.05

    def test_admin_session_detail_missing_session_returns_404(self, tmp_path, monkeypatch):
        sm = StateManager(persist_dir=str(tmp_path))
        monkeypatch.setattr(admin, "_state_manager", sm)
        result = asyncio.run(admin.admin_session_detail("does-not-exist"))
        assert result.status_code == 404

    def test_admin_stats_counts_sessions_by_recency(self, tmp_path, monkeypatch):
        sm = StateManager(persist_dir=str(tmp_path))
        _write_session_file(tmp_path, "recent", user_msgs=1)
        monkeypatch.setattr(admin, "_state_manager", sm)
        with patch("router.admin.get_cache") as mock_get_cache:
            mock_get_cache.return_value.get_stats.return_value = {}
            result = asyncio.run(admin.admin_stats())
        body = json.loads(result.body)
        assert body["sessions"]["total"] == 1
        assert body["sessions"]["today"] == 1

    def test_admin_logs_reads_file_via_executor(self, tmp_path, monkeypatch):
        log_file = tmp_path / "app.log"
        log_file.write_text("line1\nline2\nline3\n", encoding="utf-8")
        monkeypatch.setattr(admin, "_LOG_FILE", log_file)

        result = asyncio.run(admin.admin_logs(lines=2))
        body = json.loads(result.body)
        assert body["lines"] == ["line2", "line3"]
        assert body["total_lines"] == 3

    def test_admin_logs_missing_file_reports_error_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.setattr(admin, "_LOG_FILE", tmp_path / "does_not_exist.log")
        result = asyncio.run(admin.admin_logs(lines=10))
        body = json.loads(result.body)
        assert body["lines"] == []
        assert "error" in body

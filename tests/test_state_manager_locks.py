"""
Unit tests for StateManager orphan lock cleanup.

Uses a temporary directory so no real state files are touched.
No LLM or Chroma connection required.
"""
import json
import os
import time
import pytest
from pathlib import Path
from model.state_manager import StateManager


@pytest.mark.unit
class TestOrphanLockCleanup:

    def test_cleanup_removes_stale_lock(self, tmp_path):
        sm = StateManager(persist_dir=str(tmp_path))

        lock_file = tmp_path / "stale_session.lock"
        lock_file.write_text(json.dumps({"pid": 99999, "ts": time.time()}))

        # Backdate modification time well past the default stale threshold (15 s)
        old_mtime = time.time() - 60
        os.utime(str(lock_file), (old_mtime, old_mtime))

        removed = sm.cleanup_orphan_locks()

        assert removed == 1
        assert not lock_file.exists()

    def test_cleanup_preserves_fresh_lock(self, tmp_path):
        sm = StateManager(persist_dir=str(tmp_path))

        lock_file = tmp_path / "active_session.lock"
        lock_file.write_text(json.dumps({"pid": os.getpid(), "ts": time.time()}))
        # mtime is current — not stale

        removed = sm.cleanup_orphan_locks()

        assert removed == 0
        assert lock_file.exists()

    def test_cleanup_returns_zero_when_no_locks(self, tmp_path):
        sm = StateManager(persist_dir=str(tmp_path))
        removed = sm.cleanup_orphan_locks()
        assert removed == 0

    def test_cleanup_handles_multiple_stale_locks(self, tmp_path):
        sm = StateManager(persist_dir=str(tmp_path))
        old_mtime = time.time() - 60

        for i in range(3):
            lf = tmp_path / f"session_{i}.lock"
            lf.write_text(json.dumps({"pid": 99999, "ts": time.time()}))
            os.utime(str(lf), (old_mtime, old_mtime))

        removed = sm.cleanup_orphan_locks()

        assert removed == 3
        assert not any((tmp_path / f"session_{i}.lock").exists() for i in range(3))

    def test_cleanup_mixed_stale_and_fresh(self, tmp_path):
        sm = StateManager(persist_dir=str(tmp_path))

        stale = tmp_path / "stale.lock"
        stale.write_text(json.dumps({"pid": 99999, "ts": 0}))
        os.utime(str(stale), (time.time() - 60, time.time() - 60))

        fresh = tmp_path / "fresh.lock"
        fresh.write_text(json.dumps({"pid": os.getpid(), "ts": time.time()}))

        removed = sm.cleanup_orphan_locks()

        assert removed == 1
        assert not stale.exists()
        assert fresh.exists()

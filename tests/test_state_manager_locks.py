"""
Unit tests for StateManager orphan lock cleanup, plus the 2026-10 fix
(core-M10) to the lock-timeout-vs-staleness gap: STATE_LOCK_TIMEOUT_S used
to be 0.5s against a STATE_LOCK_STALE_S of 90s, so any session whose lock
was orphaned by a process crash mid-save left every request for that
session_id failing with an uncaught TimeoutError (surfaced to the user as
"Chat failed. Please try again." AFTER the LLM call already ran and was
billed) for up to ~89.5 seconds. Fixed by lowering staleness to 5s and
raising the per-request timeout above it.

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

        # Backdate modification time well past the default stale threshold
        # (conf.STATE_LOCK_STALE_S, currently 5s — see conf.py)
        old_mtime = time.time() - 120
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
        old_mtime = time.time() - 120

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
        os.utime(str(stale), (time.time() - 120, time.time() - 120))

        fresh = tmp_path / "fresh.lock"
        fresh.write_text(json.dumps({"pid": os.getpid(), "ts": time.time()}))

        removed = sm.cleanup_orphan_locks()

        assert removed == 1
        assert not stale.exists()
        assert fresh.exists()


@pytest.mark.unit
class TestLockTimeoutExceedsStaleness:
    """Pins the invariant the core-M10 fix depends on: a request waiting on
    an orphaned lock must always live long enough to reach the point where
    that lock becomes reclaimable. If a future edit narrows this gap again,
    this test catches it immediately instead of waiting for a live incident."""

    def test_conf_defaults_satisfy_the_invariant(self):
        import conf
        assert conf.STATE_LOCK_TIMEOUT_S > conf.STATE_LOCK_STALE_S, (
            f"STATE_LOCK_TIMEOUT_S ({conf.STATE_LOCK_TIMEOUT_S}) must exceed "
            f"STATE_LOCK_STALE_S ({conf.STATE_LOCK_STALE_S}) — otherwise a request "
            f"can time out before an orphaned lock becomes reclaimable (core-M10)."
        )


@pytest.mark.unit
class TestOrphanedLockIsReclaimedNotTimedOut:
    """The actual regression test: a request that encounters an orphaned
    (crashed-process) lock older than the staleness threshold must succeed
    in acquiring it well within its own timeout — not raise TimeoutError."""

    def test_acquire_lock_reclaims_a_stale_lock_within_timeout(self, tmp_path, monkeypatch):
        import conf
        monkeypatch.setattr(conf, "STATE_LOCK_STALE_S", 0.2)
        monkeypatch.setattr(conf, "STATE_LOCK_TIMEOUT_S", 1.0)
        monkeypatch.setattr(conf, "STATE_LOCK_POLL_S", 0.02)

        sm = StateManager(persist_dir=str(tmp_path))
        session_id = "crashed_session"
        lock_path = sm._lock_path(session_id)
        lock_path.write_text(json.dumps({"pid": 99999, "ts": time.time()}))
        old_mtime = time.time() - 1.0  # older than the 0.2s staleness threshold
        os.utime(str(lock_path), (old_mtime, old_mtime))

        sm._acquire_lock(session_id)  # must not raise TimeoutError
        sm._release_lock(session_id)

    def test_save_succeeds_despite_a_preexisting_orphaned_lock(self, tmp_path, monkeypatch):
        """End-to-end via the public API a request actually calls."""
        import conf
        monkeypatch.setattr(conf, "STATE_LOCK_STALE_S", 0.2)
        monkeypatch.setattr(conf, "STATE_LOCK_TIMEOUT_S", 1.0)
        monkeypatch.setattr(conf, "STATE_LOCK_POLL_S", 0.02)

        from model.conversation_state import ConversationState

        sm = StateManager(persist_dir=str(tmp_path))
        session_id = "crashed_session_2"
        lock_path = sm._lock_path(session_id)
        lock_path.write_text(json.dumps({"pid": 99999, "ts": time.time()}))
        old_mtime = time.time() - 1.0
        os.utime(str(lock_path), (old_mtime, old_mtime))

        state = ConversationState(session_id=session_id, persona_id="practical", context={})
        sm.save(session_id, state)  # must not raise

        loaded = sm.load(session_id)
        assert loaded is not None
        assert loaded.session_id == session_id

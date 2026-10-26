"""Tests for LocalVectorStoreManager._rebuild_lock (service/local_vector_store.py,
2026-10 fix — data-integrity audit).

Background: nothing previously prevented scripts/ingest_local.py's
destructive create_vectorstore(reset=True) rename-swap from running WHILE
the long-running SQS consumer was mid add_documents()/archive_and_remove()
against the SAME LOCAL_VECTOR_DIR — local Chroma persists to SQLite, a
strict single-writer store with no cross-process lock of its own, and
research (SQLite's own docs/FAQ, and Chroma's own add()-vs-upsert()
guidance) is consistent: serialize concurrent writers in application code.
_rebuild_lock is an advisory fcntl.flock()-based lock: create_vectorstore's
swap takes it EXCLUSIVE; every other mutating method takes it SHARED.

These use two independent LocalVectorStoreManager instances pointed at the
SAME persist_dir (via a real temp directory, no mocking) to simulate two
separate processes/handles contending for the same lock file — flock()
itself is real OS-level advisory locking, so this exercises the actual
mechanism rather than a mock of it.
"""
from __future__ import annotations

import fcntl
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from service.local_vector_store import LocalVectorStoreManager  # noqa: E402


def _manager_at(persist_dir: str) -> LocalVectorStoreManager:
    mgr = LocalVectorStoreManager()
    with patch.object(mgr, "_persist_dir", return_value=persist_dir):
        pass
    # Keep the patch alive for the manager's lifetime rather than just the
    # `with` block above — simplest way is to monkeypatch the bound method.
    mgr._persist_dir = lambda: persist_dir  # type: ignore[method-assign]
    return mgr


@pytest.mark.unit
class TestRebuildLockMutualExclusion:
    def test_exclusive_lock_blocks_a_second_exclusive_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr_a = _manager_at(tmp)
            mgr_b = _manager_at(tmp)
            with mgr_a._rebuild_lock(exclusive=True):
                lock_path = mgr_b._rebuild_lock_path()
                with open(lock_path, "a+") as fh_b:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(fh_b.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_exclusive_lock_blocks_a_shared_acquisition(self):
        """The whole point of the fix: a rebuild holding exclusive must
        block an ordinary write (shared) from proceeding concurrently."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr_a = _manager_at(tmp)
            mgr_b = _manager_at(tmp)
            with mgr_a._rebuild_lock(exclusive=True):
                lock_path = mgr_b._rebuild_lock_path()
                with open(lock_path, "a+") as fh_b:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(fh_b.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)

    def test_two_shared_acquisitions_do_not_block_each_other(self):
        """Ordinary writes (add_documents from two near-simultaneous
        uploads, say) must be able to proceed concurrently with each other
        — only a rebuild's exclusive lock should ever block anything."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr_a = _manager_at(tmp)
            mgr_b = _manager_at(tmp)
            with mgr_a._rebuild_lock(exclusive=False):
                lock_path = mgr_b._rebuild_lock_path()
                with open(lock_path, "a+") as fh_b:
                    # Must NOT raise — two shared holders are compatible.
                    fcntl.flock(fh_b.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    fcntl.flock(fh_b.fileno(), fcntl.LOCK_UN)

    def test_lock_is_released_after_the_context_manager_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr_a = _manager_at(tmp)
            mgr_b = _manager_at(tmp)
            with mgr_a._rebuild_lock(exclusive=True):
                pass  # released on exit

            lock_path = mgr_b._rebuild_lock_path()
            with open(lock_path, "a+") as fh_b:
                # Must succeed now that mgr_a's exclusive lock was released.
                fcntl.flock(fh_b.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh_b.fileno(), fcntl.LOCK_UN)

    def test_lock_is_released_even_if_the_body_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr_a = _manager_at(tmp)
            mgr_b = _manager_at(tmp)
            with pytest.raises(RuntimeError):
                with mgr_a._rebuild_lock(exclusive=True):
                    raise RuntimeError("simulated failure mid-write")

            lock_path = mgr_b._rebuild_lock_path()
            with open(lock_path, "a+") as fh_b:
                fcntl.flock(fh_b.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh_b.fileno(), fcntl.LOCK_UN)

    def test_lock_file_lives_next_to_not_inside_persist_dir_rename_target(self):
        """The lock file must survive create_vectorstore(reset=True)'s
        rename-swap of the persist dir itself — it must be a SIBLING path
        derived from the dir's name, never a file inside the dir, since a
        path inside persist_dir resolves to a different physical file
        before vs. after the swap's rename (the regression this pins:
        _rebuild_lock_path originally returned persist_dir / ".rebuild.lock",
        which breaks lock coordination across the swap)."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _manager_at(tmp)
            lock_path = mgr._rebuild_lock_path()
            assert str(lock_path) == tmp.rstrip("/") + ".rebuild.lock"
            assert lock_path.parent == Path(tmp).parent
            assert not str(lock_path).startswith(tmp.rstrip("/") + "/")

"""Smoke test for the 2026-10 fix (Infra2-C1) to router/admin_pdf.py.

Background: list_archived_docs (service/local_vector_store.py) transitively
pulls in langchain_core/langchain_openai/langchain_community -> chromadb —
none of which are in requirements-pdf.txt, the lean PDF-only image's
dependency lockfile, which deliberately excludes them. A module-level
`from service.local_vector_store import list_archived_docs` used to make
importing router/admin_pdf.py itself raise ModuleNotFoundError on that
image, before the /api/pdf-status handler's own try/except (meant to
degrade this one section to []) ever got a chance to run — restart-looping
the whole lean PDF service, the exact boto3-incident failure mode the
prod-pdf split was built to prevent.

This test proves router/admin_pdf.py imports cleanly with chromadb and the
three langchain packages unimportable, by forcing a fresh import with
those names blocked at the __import__ level.
"""
from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

_BLOCKED = ("chromadb", "langchain_core", "langchain_openai", "langchain_community")


@pytest.mark.unit
def test_admin_pdf_importable_without_chromadb_or_langchain(monkeypatch):
    # Drop any already-imported copies so the import below re-executes the
    # module's top-level code fresh, rather than returning a cached module
    # that was imported earlier in the test session (when the blocked
    # packages were still available).
    for mod_name in list(sys.modules):
        root = mod_name.split(".")[0]
        if mod_name == "router.admin_pdf" or root in _BLOCKED:
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.split(".")[0] in _BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r} (blocked by test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    import router.admin_pdf  # noqa: F401 — must not raise

    # The handler itself still works (degrading list_archived_docs's section
    # to best-effort via its own try/except) — not exercised here since it's
    # async and needs a running loop; this test's scope is specifically
    # "importing the module doesn't crash", which is where the real bug was.
    assert hasattr(router.admin_pdf, "pdf_status")

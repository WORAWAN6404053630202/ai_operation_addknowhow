"""Regression tests for the 2026-10 fix (core2-M12) to code/static/app.js
and code/static/index.html.

marked.parse()'s output used to go straight into bubble.innerHTML with no
sanitization — marked has not sanitized its own output by default since
v5, and the library was loaded unpinned from a third-party CDN with no
SRI. Since the PDF direct-ingest pivot, corpus content can come straight
from an uploaded PDF with no human review, so a crafted PDF's content
becoming retrievable corpus text was a real path to stored XSS in every
user's browser. Fixed by vendoring both marked and DOMPurify locally (no
CDN dependency at all) and sanitizing marked's output through
DOMPurify.sanitize() before assignment.

No JS runtime is available in this Python test suite, so these are
source-level regression checks (same approach as
test_admin_xss_escaping.py) plus existence/sanity checks on the vendored
files themselves.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_STATIC_DIR = Path(__file__).parent.parent / "code" / "static"


@pytest.mark.unit
class TestAppJsSanitizesMarkdownOutput:
    def _source(self) -> str:
        return (_STATIC_DIR / "app.js").read_text(encoding="utf-8")

    def test_marked_output_is_passed_through_dompurify(self):
        src = self._source()
        assert "DOMPurify.sanitize(rawHtml)" in src

    def test_marked_parse_result_is_not_assigned_directly(self):
        """The vulnerable pattern this closes: `return marked.parse(...)`
        with nothing in between it and the caller's innerHTML assignment."""
        src = self._source()
        assert "return marked.parse(String(text || \"\"));" not in src


@pytest.mark.unit
class TestIndexHtmlNoLongerLoadsFromThirdPartyCdn:
    def _source(self) -> str:
        return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

    def test_no_cdn_script_tags_remain(self):
        src = self._source()
        assert "cdn.jsdelivr.net" not in src

    def test_marked_loaded_from_local_vendor_path(self):
        src = self._source()
        assert '<script src="/static/vendor/marked.min.js"></script>' in src

    def test_dompurify_loaded_from_local_vendor_path(self):
        src = self._source()
        assert '<script src="/static/vendor/purify.min.js"></script>' in src

    def test_dompurify_loaded_before_app_js(self):
        """app.js's assistantToHtml references the global DOMPurify at call
        time, not at app.js's own load time, so strict load-order isn't
        required for correctness — but loading it before app.js (as
        written) keeps the dependency relationship between the two
        obvious from reading the markup top to bottom."""
        src = self._source()
        assert src.index("purify.min.js") < src.index('src="/static/app.js"')


@pytest.mark.unit
class TestVendoredFilesArePresentAndReal:
    def test_vendored_marked_file_exists_and_is_non_trivial(self):
        p = _STATIC_DIR / "vendor" / "marked.min.js"
        assert p.exists()
        assert p.stat().st_size > 5000
        assert "marked" in p.read_text(encoding="utf-8")[:300].lower()

    def test_vendored_dompurify_file_exists_and_is_non_trivial(self):
        p = _STATIC_DIR / "vendor" / "purify.min.js"
        assert p.exists()
        assert p.stat().st_size > 5000
        assert "DOMPurify" in p.read_text(encoding="utf-8")[:300]

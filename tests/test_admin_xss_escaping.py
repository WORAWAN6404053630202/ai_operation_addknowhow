"""Regression tests for the 2026-10 fix (core2-C4, Infra2-C4/PDF2-M2) to
both admin dashboards' HTML-escaping helpers.

Both `escHtml` (code/static/admin.html, the bot's dashboard) and
`escapeHtml` (code/router/admin_pdf.py, the PDF dashboard) escaped `&`, `<`,
`>` but not quote characters — fine for a text node, but their output also
landed in attribute positions (an inline onclick string in admin.html, a
title="..." tooltip in admin_pdf.py), so a crafted session_id / PDF-derived
flag message could break out of the attribute and run JS that reads the
admin key out of localStorage/sessionStorage.

No JS runtime is available in this Python test suite, so these are
source-level regression checks: assert the quote-escaping logic is present,
and that admin.html's session row no longer builds its click handler via
raw string interpolation into an inline onclick attribute.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"


@pytest.mark.unit
class TestAdminHtmlEscaping:
    def _source(self) -> str:
        return (_CODE_DIR / "static" / "admin.html").read_text(encoding="utf-8")

    def test_eschtml_escapes_double_quote(self):
        src = self._source()
        assert '.replaceAll(\'"\', "&quot;")' in src or '.replaceAll("\\"","&quot;")' in src

    def test_eschtml_escapes_single_quote(self):
        src = self._source()
        assert "&#39;" in src

    def test_session_row_no_longer_uses_inline_onclick_string_interpolation(self):
        """The actual vulnerable pattern this closes: onclick="selectSession('${s.session_id}')"
        built session_id into a JS-string-literal/attribute position with no
        escaping at the call site at all (escHtml wasn't even called there).
        Strip comment lines first — the fix's own explanatory comment quotes
        the old pattern verbatim, which would otherwise false-positive."""
        code_lines = [
            line for line in self._source().splitlines()
            if not line.strip().startswith("//")
        ]
        assert "onclick=\"selectSession('${s.session_id}')\"" not in "\n".join(code_lines)

    def test_session_row_uses_escaped_data_attribute_instead(self):
        src = self._source()
        assert 'data-sid="${escHtml(s.session_id)}"' in src

    def test_delegated_click_handler_reads_data_sid(self):
        src = self._source()
        assert "row.dataset.sid" in src


@pytest.mark.unit
class TestAdminPdfHtmlEscaping:
    def _source(self) -> str:
        return (_CODE_DIR / "router" / "admin_pdf.py").read_text(encoding="utf-8")

    def test_escapehtml_escapes_double_quote(self):
        src = self._source()
        assert '&quot;' in src

    def test_escapehtml_escapes_single_quote(self):
        src = self._source()
        assert "&#39;" in src

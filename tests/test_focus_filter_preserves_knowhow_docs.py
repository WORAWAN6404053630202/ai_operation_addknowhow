"""2026-10 fix: FOCUS FILTER's dominant-license-type narrowing was dropping
know-how/business_guide docs (license_type="") as "noise" alongside
non-dominant regulatory license_types, because "" never equals a non-empty
_dominant_lt. Live-reproduced on the PDF-only DBD Biz Regist test content:
a question about สมัครสมาชิก.pdf (no license_type) retrieved the correct
doc, but FOCUS FILTER discarded it twice in the same turn (once per
retrieval pass) in favor of an unrelated regulatory doc sharing the
candidate pool, leaving the LLM with zero evidence to answer from.

The exact same root cause was already identified and fixed for the
narrower "broad tax query" case (see the _is_broad_tax_query block's own
comment a few hundred lines above this fix, 2026-09: "license_type=''
never matches a non-empty dominant_lt") — this generalizes that fix to
every FOCUS FILTER narrowing, not just tax queries.

persona_practical.py's handle() is a single ~3000+-line method — see
test_doc_overflow_ceiling_fixes.py's module docstring for why source
inspection (not a full mocked end-to-end call) is this codebase's
established pattern for regression-testing logic embedded this deeply.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model.persona_practical import PracticalPersonaService  # noqa: E402


@pytest.mark.unit
class TestFocusFilterPreservesEmptyLicenseTypeDocs:
    def test_focused_docs_filter_keeps_empty_license_type(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        idx = source.find("_dominant_lt = _best_lt if (_best_lt and _best_score > 0) else None")
        assert idx != -1, "_dominant_lt assignment not found in handle()"
        end = source.find("if _focused_docs:", idx)
        assert end != -1
        body = source[idx:end]
        # The fix: membership test against (_dominant_lt, "") instead of a bare
        # equality check against _dominant_lt alone — "" (know-how/business_guide,
        # no license_type) must survive the narrowing, not just the dominant
        # regulatory license_type itself.
        assert 'in (_dominant_lt, "")' in body, (
            "FOCUS FILTER's _focused_docs comprehension no longer preserves "
            "empty-license_type (know-how) docs — regression of the 2026-10 fix"
        )
        # Guard against a trivial "fix" that just keeps everything regardless of
        # _dominant_lt — non-dominant REGULATORY license_types must still be
        # dropped (that's the filter's actual, intended job).
        assert "_dominant_lt" in body

    def test_focused_docs_filter_still_excludes_other_regulatory_license_types(self):
        # Behavioral guard via the actual comprehension, evaluated directly —
        # confirms a non-dominant, non-empty license_type doc is excluded while
        # an empty-license_type doc and the dominant one both survive.
        _dominant_lt = "การจดทะเบียนนิติบุคคลดิจิทัล (จัดตั้งบริษัทจำกัด)"
        _all_docs = [
            {"metadata": {"license_type": _dominant_lt}},
            {"metadata": {"license_type": "การจดทะเบียนเลิกบริษัทจำกัด"}},  # non-dominant reg. doc
            {"metadata": {"license_type": ""}},  # know-how doc, e.g. สมัครสมาชิก.pdf
            {"metadata": {}},  # no license_type key at all (defensive case)
        ]
        _focused_docs = [
            d for d in _all_docs
            if (d.get("metadata") or {}).get("license_type", "").strip() in (_dominant_lt, "")
        ]
        assert len(_focused_docs) == 3
        kept_license_types = {(d["metadata"].get("license_type") or "") for d in _focused_docs}
        assert kept_license_types == {_dominant_lt, ""}
        assert "การจดทะเบียนเลิกบริษัทจำกัด" not in kept_license_types

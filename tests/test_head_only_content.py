"""
Regression tests for the 2026-09 cost-reduction change: DOCUMENTS sent to the
Practical LLM used to include both a full "content" string (page_content, built at
ingest time by duplicating every metadata field as labeled prose — verified ~75-85%
the size of metadata itself against real corpus data) AND the same fields again via
structured "metadata". Since metadata already carries the complete version of every
field (after raising 4 field caps that were previously tighter than content's own
embedding-time caps — see model/persona_practical.py's _FIELD_CAPS comments),
sending full content too was pure duplication except for 3 fields deliberately
excluded from _LLM_METADATA_WHITELIST (location, area_size,
operation_by_department) — those only ever reached the LLM via content's short
"head" line. _build_head_only_content() reproduces just that head line from
metadata directly, dropping the redundant body.

One retrieval path (sub_topic-completeness aggregation, ~persona_practical.py line
4080) is a genuine exception — its "content" is a synthesis with no metadata
equivalent — marked with "_keep_full_content": True and must NOT go through this
trim; covered by test_docs_json_keep_full_content_marker_respected below.

No network access, no LLM calls.
"""
import sys
import os

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))

import pytest
from model.persona_practical import _build_head_only_content


@pytest.mark.unit
class TestBuildHeadOnlyContent:

    def test_includes_all_head_fields_when_present(self):
        md = {
            "license_type": "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร",
            "operation_by_department": "ยื่นใหม่",
            "location": "กรุงเทพฯ",
            "area_size": "ไม่เกิน 200 ตรม.",
            "entity_type_normalized": "นิติบุคคล",
            "operation_topic": "การขอใบอนุญาต",
            "department": "สำนักงานเขต",
        }
        out = _build_head_only_content(md)
        assert "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร" in out
        assert "ยื่นใหม่" in out
        assert "กรุงเทพฯ" in out
        assert "ไม่เกิน 200 ตรม." in out
        assert "นิติบุคคล" in out
        assert "การขอใบอนุญาต" in out
        assert "สำนักงานเขต" in out

    def test_registration_type_used_only_when_entity_type_absent(self):
        md_with_entity = {"entity_type_normalized": "บุคคลธรรมดา", "registration_type": "บริษัทจำกัด"}
        out = _build_head_only_content(md_with_entity)
        assert "บุคคลธรรมดา" in out
        assert "บริษัทจำกัด" not in out

        md_without_entity = {"registration_type": "บริษัทจำกัด"}
        out2 = _build_head_only_content(md_without_entity)
        assert "บริษัทจำกัด" in out2

    def test_empty_metadata_returns_empty_string(self):
        assert _build_head_only_content({}) == ""

    def test_omits_body_fields_entirely(self):
        """The whole point of this function: never include the long fields that are
        already sent in full via metadata — including them would recreate the
        duplication this change is meant to remove."""
        md = {
            "license_type": "X",
            "operation_steps": "ขั้นตอนที่ 1... ขั้นตอนที่ 2... " * 50,
            "identification_documents": "เอกสาร A, B, C " * 50,
            "fees": "500 บาท",
            "legal_regulatory": "บทลงโทษ..." * 50,
        }
        out = _build_head_only_content(md)
        assert "ขั้นตอนที่" not in out
        assert "เอกสาร A" not in out
        assert "500 บาท" not in out
        assert "บทลงโทษ" not in out
        assert len(out) < 200

    def test_short_relative_to_a_realistic_full_body(self):
        """Sanity check on the actual savings claim, not just structural correctness."""
        md = {
            "license_type": "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร",
            "location": "กรุงเทพฯ",
            "entity_type_normalized": "นิติบุคคล",
            "operation_topic": "การขอใบอนุญาต",
            "department": "สำนักงานเขต",
            "operation_steps": "ยื่นคำขอ กรอกแบบฟอร์ม ชำระค่าธรรมเนียม รอการตรวจสอบ " * 20,
        }
        head = _build_head_only_content(md)
        full_body_len = len(md["operation_steps"])
        assert len(head) < full_body_len * 0.2


@pytest.mark.unit
class TestDocsJsonKeepFullContentMarker:
    """The docs_json build loop (persona_practical.py, "Pass 2") must construct each
    entry as an explicit {"metadata": ..., "content": ...} dict literal — never by
    spreading the source doc dict (e.g. **d) — so the internal "_keep_full_content"
    routing marker (set on the sub_topic-completeness aggregated doc) can never leak
    into what actually reaches the LLM. Source-inspection guard since the build loop
    lives inline inside a large method, not its own testable function."""

    def test_docs_json_append_is_an_explicit_two_key_dict_literal(self):
        import inspect
        import model.persona_practical as pp
        src = inspect.getsource(pp)
        idx = src.index('docs_json.append(')
        snippet = src[idx: idx + 200]
        assert '"metadata": filtered_md' in snippet
        assert '"content": _content_for_llm' in snippet
        assert "**d" not in snippet
        assert ".update(d)" not in snippet

    def test_keep_full_content_marker_is_checked_before_head_only_trim(self):
        import inspect
        import model.persona_practical as pp
        src = inspect.getsource(pp.PracticalPersonaService)
        check_idx = src.index('d.get("_keep_full_content")')
        # _build_head_only_content(md) must appear shortly after the marker check
        # (the else-branch of the same if/else), not unconditionally elsewhere.
        window = src[check_idx: check_idx + 1200]
        assert "_build_head_only_content(md)" in window
        assert 'docs_json.append(' in window

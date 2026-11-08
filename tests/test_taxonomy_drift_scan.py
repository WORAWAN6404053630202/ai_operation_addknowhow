"""
Tests for code/scripts/taxonomy_drift_scan.py — 2026-11, hardcoded-logic
audit follow-up, Phase 3 of the dynamic-taxonomy plan (see service.
live_taxonomy's docstring for the full background).
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "scripts"))
import taxonomy_drift_scan as tds  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.unit
class TestExtractValuesAgainstRealSource:
    """The whole script depends on each regex still matching the real
    structure's current shape — these pin that against the actual files,
    not a fabricated fixture, so a reformat that breaks extraction silently
    is caught immediately rather than only in a next live audit."""

    def _source_for(self, spec) -> str:
        with open(os.path.join(REPO_ROOT, spec.file), "r", encoding="utf-8") as f:
            return f.read()

    def test_every_registered_structure_extracts_at_least_one_value(self):
        for spec in tds.STRUCTURES:
            values = tds._extract_values(spec, self._source_for(spec))
            assert values, f"{spec.name} extracted nothing from {spec.file} — regex may be stale"

    def test_area_sensitive_licenses_extracts_known_value(self):
        spec = next(s for s in tds.STRUCTURES if s.name == "_AREA_SENSITIVE_LICENSES")
        values = tds._extract_values(spec, self._source_for(spec))
        assert "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหารหรือสถานที่สะสมอาหาร" in values

    def test_bank_dept_patterns_extracts_department_values_not_regex(self):
        spec = next(s for s in tds.STRUCTURES if s.name == "_BANK_DEPT_PATTERNS")
        values = tds._extract_values(spec, self._source_for(spec))
        assert "ธนาคารไทยพาณิชย์" in values
        assert "ธนาคารกสิกรไทย" in values
        assert not any("re.IGNORECASE" in v for v in values)

    def test_related_topic_groups_extracts_only_frozenset_members_not_notes(self):
        spec = next(s for s in tds.STRUCTURES if s.name == "_RELATED_TOPIC_GROUPS")
        values = tds._extract_values(spec, self._source_for(spec))
        assert "บริการร้านค้ารับชำระเงิน QR API" in values
        assert not any("ซึ่งเป็น" in v for v in values)  # the trailing note strings, not frozenset members

    def test_rt_canonical_map_extracts_values_not_keys(self):
        spec = next(s for s in tds.STRUCTURES if s.name == "_RT_CANONICAL_MAP")
        values = tds._extract_values(spec, self._source_for(spec))
        assert "บริษัท" in values
        assert "บริษัทจำกัด" not in values

    def test_unknown_structure_name_returns_empty_not_crash(self):
        import re
        spec = tds.StructureSpec(
            name="_NOT_IN_FILE", file="code/model/persona_supervisor.py", field="license_type",
            severity="high", anchor_re=re.compile(r"_NOT_IN_FILE_AT_ALL\s*=\s*\[(.*?)\]", re.DOTALL),
            value_re=re.compile(r'"([^"]+)"'),
        )
        assert tds._extract_values(spec, self._source_for(spec)) == []


@pytest.mark.unit
class TestCheckDrift:
    def _vectorstore_with(self, metadatas: list):
        vs = MagicMock()
        vs._collection.get.return_value = {"metadatas": metadatas}
        return vs

    def _clear_live_taxonomy_cache(self):
        from service.live_taxonomy import _live_values_cache, _llm_resolution_cache
        _live_values_cache.clear()
        _llm_resolution_cache.clear()

    def setup_method(self):
        self._clear_live_taxonomy_cache()

    def teardown_method(self):
        self._clear_live_taxonomy_cache()

    def test_value_present_in_live_corpus_is_not_flagged(self, monkeypatch):
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 1)
        vs = self._vectorstore_with([
            {"license_type": "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหารหรือสถานที่สะสมอาหาร"},
            {"department": "ธนาคารไทยพาณิชย์"},
            {"department": "ธนาคารกสิกรไทย"},
            {"license_type": "บริการร้านค้ารับชำระเงิน QR API"},
            {"location": "กรุงเทพฯ"},
            {"location": "ต่างจังหวัด"},
            {"registration_type": "บริษัท"},
        ])
        findings = tds.check_drift(vs, repo_root=REPO_ROOT)
        by_structure = {(f.structure, f.value) for f in findings}
        assert ("_AREA_SENSITIVE_LICENSES", "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหารหรือสถานที่สะสมอาหาร") not in by_structure
        assert ("_BANK_DEPT_PATTERNS", "ธนาคารไทยพาณิชย์") not in by_structure
        assert ("_RT_CANONICAL_MAP", "บริษัท") not in by_structure
        assert ("_RELATED_TOPIC_GROUPS", "บริการร้านค้ารับชำระเงิน QR API") not in by_structure

    def test_related_topic_groups_resolves_against_license_type_not_main_topic(self, monkeypatch):
        """Live-caught 2026-11-08: _RELATED_TOPIC_GROUPS's values are license_type
        strings (fed into multi_license_topics, populated from license_type — see
        persona_supervisor.py's _lt_unique/_detected_licenses), NOT main_topic. The
        first deploy of this scan mis-assigned the field and flagged all 10 members
        as drift against the wrong field — this pins the fix."""
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 1)
        # "ใบทะเบียนพาณิชย์" only appears as main_topic here; license_type has an
        # unrelated value so the field isn't empty (an empty live field is skipped
        # by design, which would hide the very bug this test exists to catch).
        vs = self._vectorstore_with([
            {"main_topic": "ใบทะเบียนพาณิชย์"},
            {"license_type": "ใบอนุญาตจำหน่ายสุรา"},
        ])
        findings = tds.check_drift(vs, repo_root=REPO_ROOT)
        related_findings = [f for f in findings if f.structure == "_RELATED_TOPIC_GROUPS"]
        # "ใบทะเบียนพาณิชย์" is present only as main_topic here, not license_type ->
        # still flagged, proving the check is really reading the license_type field.
        assert any(f.value == "ใบทะเบียนพาณิชย์" for f in related_findings)

    def test_value_absent_from_live_corpus_is_flagged_high_severity(self, monkeypatch):
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 1)
        # department field has totally unrelated live values -> _BANK_DEPT_PATTERNS'
        # targets ("ธนาคารไทยพาณิชย์"/"ธนาคารกสิกรไทย") won't resolve at all.
        vs = self._vectorstore_with([{"department": "กรมสรรพากร"}])
        findings = tds.check_drift(vs, repo_root=REPO_ROOT)
        bank_findings = [f for f in findings if f.structure == "_BANK_DEPT_PATTERNS"]
        assert len(bank_findings) == 2
        assert all(f.severity == "high" for f in bank_findings)

    def test_empty_live_field_does_not_produce_false_drift(self, monkeypatch):
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 1)
        vs = self._vectorstore_with([])  # nothing in the corpus at all
        findings = tds.check_drift(vs, repo_root=REPO_ROOT)
        assert findings == []

    def test_fuzzy_match_against_live_value_is_not_flagged(self, monkeypatch):
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 1)
        # Near-identical spelling (missing a trailing word) should still fuzzy-resolve.
        vs = self._vectorstore_with([
            {"license_type": "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหารหรือสถานที่สะสมอาหารฯ"},
        ])
        findings = tds.check_drift(vs, repo_root=REPO_ROOT)
        area_findings = [f for f in findings if f.structure == "_AREA_SENSITIVE_LICENSES"]
        assert area_findings == []


@pytest.mark.unit
class TestMainExitCode:
    def test_main_returns_zero_when_no_findings(self, monkeypatch):
        monkeypatch.setattr(tds, "_get_vectorstore_for_cli", lambda: MagicMock())
        monkeypatch.setattr(tds, "check_drift", lambda vs: [])
        assert tds.main() == 0

    def test_main_returns_one_when_high_severity_findings_exist(self, monkeypatch):
        monkeypatch.setattr(tds, "_get_vectorstore_for_cli", lambda: MagicMock())
        monkeypatch.setattr(
            tds, "check_drift",
            lambda vs: [tds.DriftFinding("_X", "f.py", "license_type", "high", "ไม่พบ")],
        )
        assert tds.main() == 1

    def test_main_returns_zero_when_only_info_severity_findings_exist(self, monkeypatch):
        monkeypatch.setattr(tds, "_get_vectorstore_for_cli", lambda: MagicMock())
        monkeypatch.setattr(
            tds, "check_drift",
            lambda vs: [tds.DriftFinding("_MENU_FALLBACK_TOPICS", "f.py", "main_topic", "info", "ไม่พบ")],
        )
        assert tds.main() == 0

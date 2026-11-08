"""
Tests for service.live_taxonomy — 2026-11, hardcoded-logic audit follow-up.
Generalizes persona_practical.py's proven _get_nonreg_main_topics pattern
("zero hardcoded topic names, fetched live, cached per corpus-version")
into a shared utility, so hardcoded keyword->canonical-string mappings
(persona_supervisor.py's _KW_OVERRIDE_LICENSE_TYPES etc.) can be replaced
with dynamic resolution against whatever the corpus actually contains.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from service.live_taxonomy import (
    get_cached_llm_resolution,
    get_live_license_types,
    get_live_values,
    resolve_candidate,
    set_cached_llm_resolution,
    _live_values_cache,
    _llm_resolution_cache,
)


def _vectorstore_with_metadatas(metadatas: list):
    vs = MagicMock()
    vs._collection.get.return_value = {"metadatas": metadatas}
    return vs


@pytest.fixture(autouse=True)
def _clear_caches():
    _live_values_cache.clear()
    _llm_resolution_cache.clear()
    yield
    _live_values_cache.clear()
    _llm_resolution_cache.clear()


@pytest.mark.unit
class TestGetLiveValues:
    def test_returns_distinct_nonempty_values_sorted_longest_first(self, monkeypatch):
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 1)
        vs = _vectorstore_with_metadatas([
            {"license_type": "ก"},
            {"license_type": "กขค"},
            {"license_type": "ก"},
            {"license_type": ""},
            {"license_type": "nan"},
            {},
        ])
        out = get_live_values(vs, "license_type")
        assert out == ["กขค", "ก"]

    def test_vectorstore_none_returns_empty(self, monkeypatch):
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 1)
        assert get_live_values(None, "license_type") == []

    def test_cache_hit_avoids_refetch_until_corpus_version_changes(self, monkeypatch):
        _cv = {"v": 1}
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: _cv["v"])
        vs = _vectorstore_with_metadatas([{"license_type": "a"}])
        first = get_live_values(vs, "license_type")
        vs._collection.get.return_value = {"metadatas": [{"license_type": "b"}]}
        second = get_live_values(vs, "license_type")  # same version -> cached, unchanged
        assert first == second == ["a"]
        _cv["v"] = 2
        third = get_live_values(vs, "license_type")  # version bumped -> refetch
        assert third == ["b"]

    def test_fetch_exception_degrades_to_empty_not_crash(self, monkeypatch):
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 1)
        vs = MagicMock()
        vs._collection.get.side_effect = RuntimeError("boom")
        assert get_live_values(vs, "license_type") == []

    def test_get_live_license_types_is_a_thin_wrapper(self, monkeypatch):
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 1)
        vs = _vectorstore_with_metadatas([{"license_type": "ใบอนุญาตทดสอบ"}])
        assert get_live_license_types(vs) == ["ใบอนุญาตทดสอบ"]


@pytest.mark.unit
class TestResolveCandidate:
    def test_exact_match(self):
        assert resolve_candidate("ก", ["ก", "ข"]) == ("ก", "exact")

    def test_substring_single_match(self):
        value, kind = resolve_candidate("มาตรฐาน", ["ใบรับรองมาตรฐานร้านอาหาร", "ใบอนุญาตจำหน่ายสุรา"])
        assert value == "ใบรับรองมาตรฐานร้านอาหาร"
        assert kind == "fuzzy"

    def test_substring_multiple_matches_is_ambiguous(self):
        value, kind = resolve_candidate("อาหาร", ["ใบอนุญาตจำหน่ายอาหาร", "ใบรับรองมาตรฐานอาหาร"])
        assert value is None
        assert kind == "ambiguous"

    def test_fuzzy_close_match_above_threshold(self):
        value, kind = resolve_candidate(
            "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร",
            ["ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหารหรือสถานที่สะสมอาหาร", "ใบอนุญาตจำหน่ายสุรา"],
        )
        assert kind in ("fuzzy", "exact")
        assert value == "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหารหรือสถานที่สะสมอาหาร"

    def test_no_match_below_threshold_is_empty(self):
        value, kind = resolve_candidate("zzz_totally_unrelated_zzz", ["ก", "ข", "ค"])
        assert value is None
        assert kind == "empty"

    def test_empty_hint_or_empty_candidates(self):
        assert resolve_candidate("", ["ก"]) == (None, "empty")
        assert resolve_candidate("ก", []) == (None, "empty")
        assert resolve_candidate("   ", ["ก"]) == (None, "empty")


@pytest.mark.unit
class TestLlmResolutionCache:
    def test_set_then_get_within_same_corpus_version(self, monkeypatch):
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 7)
        set_cached_llm_resolution("QR", "บริการร้านค้ารับชำระเงิน QR API")
        assert get_cached_llm_resolution("QR") == "บริการร้านค้ารับชำระเงิน QR API"

    def test_normalizes_case_and_whitespace_for_the_cache_key(self, monkeypatch):
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 7)
        set_cached_llm_resolution("  qr  ", "X")
        assert get_cached_llm_resolution("QR") == "X"

    def test_confirmed_no_match_is_cached_as_empty_string_not_none(self, monkeypatch):
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 7)
        set_cached_llm_resolution("obscure", None)
        assert get_cached_llm_resolution("obscure") == ""

    def test_cache_invalidates_on_corpus_version_change(self, monkeypatch):
        _cv = {"v": 1}
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: _cv["v"])
        set_cached_llm_resolution("QR", "X")
        _cv["v"] = 2
        assert get_cached_llm_resolution("QR") is None

    def test_unresolved_hint_returns_none_not_empty_string(self, monkeypatch):
        monkeypatch.setattr("service.live_taxonomy._get_corpus_version", lambda: 7)
        assert get_cached_llm_resolution("never seen before") is None

"""Tests for utils/pdf_draft_cache.py — the content-hash-keyed cache for
PDF topic-splitting/field-drafting LLM output (2026-10). Covers: hash
determinism, get/put round-trip, and the fail-open guarantee (any DynamoDB
error must degrade to "miss"/"silently skipped," never raise). No real
AWS — the DynamoDB client is a MagicMock throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils.pdf_draft_cache import DraftCache, content_hash  # noqa: E402


@pytest.mark.unit
class TestContentHash:
    def test_same_text_same_hash(self):
        assert content_hash("เนื้อหาเดิม") == content_hash("เนื้อหาเดิม")

    def test_different_text_different_hash(self):
        assert content_hash("เนื้อหา A") != content_hash("เนื้อหา B")

    def test_empty_text_does_not_raise(self):
        assert content_hash("") == content_hash(None if False else "")  # sanity: stable on empty


def _cache_with_mock_client():
    cache = DraftCache(table_name="test-table", region_name="ap-southeast-7")
    mock_client = MagicMock()
    cache._client = mock_client
    return cache, mock_client


@pytest.mark.unit
class TestDraftCacheGet:
    def test_miss_returns_none(self):
        cache, mock_client = _cache_with_mock_client()
        mock_client.get_item.return_value = {}
        assert cache.get("some:key") is None

    def test_hit_returns_deserialized_result(self):
        cache, mock_client = _cache_with_mock_client()
        mock_client.get_item.return_value = {
            "Item": {"result_json": {"S": '{"department": "สำนักงานประกันสังคม"}'}}
        }
        result = cache.get("some:key")
        assert result == {"department": "สำนักงานประกันสังคม"}

    def test_dynamodb_error_returns_none_not_raise(self):
        cache, mock_client = _cache_with_mock_client()
        mock_client.get_item.side_effect = RuntimeError("table does not exist")
        assert cache.get("some:key") is None  # must not raise

    def test_malformed_item_returns_none_not_raise(self):
        cache, mock_client = _cache_with_mock_client()
        mock_client.get_item.return_value = {"Item": {"result_json": {"S": "not valid json{{"}}}
        assert cache.get("some:key") is None


@pytest.mark.unit
class TestDraftCachePut:
    def test_put_calls_put_item_with_serialized_result(self):
        cache, mock_client = _cache_with_mock_client()
        cache.put("some:key", {"license_type": "ทะเบียนนายจ้าง"})
        mock_client.put_item.assert_called_once()
        _, kwargs = mock_client.put_item.call_args
        assert kwargs["TableName"] == "test-table"
        assert kwargs["Item"]["cache_key"]["S"] == "some:key"
        assert "ทะเบียนนายจ้าง" in kwargs["Item"]["result_json"]["S"]

    def test_dynamodb_error_on_put_does_not_raise(self):
        cache, mock_client = _cache_with_mock_client()
        mock_client.put_item.side_effect = RuntimeError("throttled")
        cache.put("some:key", {"a": "b"})  # must not raise


@pytest.mark.unit
class TestDraftCacheRoundTrip:
    def test_put_then_get_round_trips(self):
        """Simulates a real table by having the mock's get_item echo back
        whatever put_item was last called with."""
        cache, mock_client = _cache_with_mock_client()
        stored = {}

        def fake_put_item(TableName, Item):
            stored["item"] = Item

        def fake_get_item(TableName, Key):
            if "item" not in stored or stored["item"]["cache_key"]["S"] != Key["cache_key"]["S"]:
                return {}
            return {"Item": stored["item"]}

        mock_client.put_item.side_effect = fake_put_item
        mock_client.get_item.side_effect = fake_get_item

        cache.put("draft:v1:abc123", {"fees": "ไม่มีค่าธรรมเนียม"})
        assert cache.get("draft:v1:abc123") == {"fees": "ไม่มีค่าธรรมเนียม"}
        assert cache.get("draft:v1:different") is None


@pytest.mark.unit
class TestGetDraftCacheSingleton:
    def test_returns_same_instance(self):
        import utils.pdf_draft_cache as mod
        mod._CACHE = None  # reset singleton for this test
        c1 = mod.get_draft_cache()
        c2 = mod.get_draft_cache()
        assert c1 is c2
        mod._CACHE = None  # don't leak state to other tests

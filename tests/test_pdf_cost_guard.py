"""Tests for utils/pdf_cost_guard.py — the daily $ spend TRACKER feeding the
admin dashboard's running-total display (OBSERVATIONAL ONLY since 2026-10;
see the module's docstring for why a prior calendar-day auto-block was
removed). Covers: record_cost's atomic update_item shape, get_today_total's
parsing, and the fail-open guarantee (any DynamoDB error must degrade to
"$0.00 recorded/read," never raise). No real AWS — the DynamoDB client is a
MagicMock throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils.pdf_cost_guard import DailyCostGuard  # noqa: E402


def _guard_with_mock_client(ceiling_usd: float = 25.0):
    guard = DailyCostGuard(table_name="test-cost-table", region_name="ap-southeast-7", ceiling_usd=ceiling_usd)
    mock_client = MagicMock()
    guard._client = mock_client
    return guard, mock_client


@pytest.mark.unit
class TestRecordCost:
    def test_record_cost_calls_update_item_with_amount(self):
        guard, mock_client = _guard_with_mock_client()
        guard.record_cost(1.23)
        mock_client.update_item.assert_called_once()
        _, kwargs = mock_client.update_item.call_args
        assert kwargs["TableName"] == "test-cost-table"
        assert kwargs["ExpressionAttributeValues"][":amount"]["N"] == "1.23"
        assert "date_bucket" in kwargs["Key"]

    def test_zero_or_negative_cost_is_a_no_op(self):
        guard, mock_client = _guard_with_mock_client()
        guard.record_cost(0.0)
        guard.record_cost(-5.0)
        mock_client.update_item.assert_not_called()

    def test_dynamodb_error_on_record_does_not_raise(self):
        guard, mock_client = _guard_with_mock_client()
        mock_client.update_item.side_effect = RuntimeError("throttled")
        guard.record_cost(1.0)  # must not raise


@pytest.mark.unit
class TestGetTodayTotal:
    def test_no_item_returns_zero(self):
        guard, mock_client = _guard_with_mock_client()
        mock_client.get_item.return_value = {}
        assert guard.get_today_total() == 0.0

    def test_existing_item_parses_running_total(self):
        guard, mock_client = _guard_with_mock_client()
        mock_client.get_item.return_value = {"Item": {"running_total_usd": {"N": "12.5"}}}
        assert guard.get_today_total() == 12.5

    def test_dynamodb_error_returns_zero_not_raise(self):
        guard, mock_client = _guard_with_mock_client()
        mock_client.get_item.side_effect = RuntimeError("table does not exist")
        assert guard.get_today_total() == 0.0  # fail-open: never blocks the pipeline


@pytest.mark.unit
class TestCeilingUsdProperty:
    def test_ceiling_usd_is_exposed_for_dashboard_display_only(self):
        """Purely a display reference now — see module docstring for why
        this class no longer gates processing on it."""
        guard, _ = _guard_with_mock_client(ceiling_usd=25.0)
        assert guard.ceiling_usd == 25.0


@pytest.mark.unit
class TestRecordThenReadRoundTrip:
    def test_record_cost_then_get_today_total_reflects_accumulation(self):
        """Simulates a real table: get_item echoes back whatever
        update_item's running total would be after each increment."""
        guard, mock_client = _guard_with_mock_client()
        state = {"total": 0.0}

        def fake_update_item(TableName, Key, UpdateExpression, ExpressionAttributeValues):
            state["total"] += float(ExpressionAttributeValues[":amount"]["N"])

        def fake_get_item(TableName, Key):
            return {"Item": {"running_total_usd": {"N": str(state["total"])}}}

        mock_client.update_item.side_effect = fake_update_item
        mock_client.get_item.side_effect = fake_get_item

        guard.record_cost(5.0)
        guard.record_cost(3.5)
        assert guard.get_today_total() == 8.5


@pytest.mark.unit
class TestGetCostGuardSingleton:
    def test_returns_same_instance(self):
        import utils.pdf_cost_guard as mod
        mod._GUARD = None  # reset singleton for this test
        g1 = mod.get_cost_guard()
        g2 = mod.get_cost_guard()
        assert g1 is g2
        mod._GUARD = None  # don't leak state to other tests

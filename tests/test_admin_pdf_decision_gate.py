"""Tests for the 2026-10 fix (PDF-M8) to router/admin_pdf.py's
POST /admin/api/pdf-queue/{item_id}/decision — a safety gate against
approving a failure/legacy-skip placeholder item.

Background: since the direct-ingest architectural pivot, real successful
content writes straight to Chroma and never reaches this queue —
sqs_consumer.py's old human-review builders are no longer called. This
queue now only ever receives "extraction_failed" (give-up-after-N-attempts
placeholder, pages=[]/llm_drafted_fields=None) or "not_relevant" (Lambda-era
legacy skip-marker) items. Approving either used to be possible and would
write a near-empty row to the real Sheet — this closes that gap.

No real AWS/Sheets/network calls — PdfReviewQueueManager is monkeypatched.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import router.admin_pdf as admin_pdf  # noqa: E402
from model.pdf_review_item import ReviewItem  # noqa: E402


def _item(tier: str | None, **overrides) -> ReviewItem:
    kwargs = dict(filename="test.pdf")
    if tier is not None:
        kwargs["relevance_check"] = {"tier": tier, "reasoning": "x"}
    kwargs.update(overrides)
    return ReviewItem(**kwargs)


@pytest.mark.unit
@pytest.mark.asyncio
class TestApprovalGateOnFailurePlaceholders:
    async def _decide(self, item, review_status="approved", decision_type="new"):
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = item
        original_mgr = admin_pdf._pdf_queue_manager
        admin_pdf._pdf_queue_manager = mock_mgr
        try:
            decision = admin_pdf.PdfReviewDecision(review_status=review_status, decision_type=decision_type)
            resp = await admin_pdf.pdf_queue_decide(item.id, decision)
        finally:
            admin_pdf._pdf_queue_manager = original_mgr
        return resp, mock_mgr

    async def test_approving_extraction_failed_item_is_rejected(self):
        item = _item(tier="extraction_failed")
        resp, mock_mgr = await self._decide(item)
        assert resp.status_code == 400
        mock_mgr.save.assert_not_called()

    async def test_approving_not_relevant_item_is_rejected(self):
        item = _item(tier="not_relevant")
        resp, mock_mgr = await self._decide(item)
        assert resp.status_code == 400
        mock_mgr.save.assert_not_called()

    async def test_rejecting_an_extraction_failed_item_is_still_allowed(self):
        """Rejection (dismissing the placeholder) must stay available even
        though approval is blocked — this is the normal triage action."""
        item = _item(tier="extraction_failed")
        resp, mock_mgr = await self._decide(item, review_status="rejected", decision_type=None)
        assert resp.status_code == 200
        mock_mgr.save.assert_called_once()

    async def test_approving_an_item_with_no_tier_is_unaffected(self):
        """Regression guard: a normal/legacy item with no relevance_check
        tier at all (or a tier outside the blocked set) must not be caught
        by this gate — only exactly 'extraction_failed'/'not_relevant' are."""
        item = _item(tier=None, llm_drafted_fields={"หน่วยงาน": "X"})
        from unittest.mock import patch

        with patch("service.sheet_write_back.append_review_item_to_sheet", return_value={"ok": True}):
            resp, mock_mgr = await self._decide(item)
        assert resp.status_code == 200
        mock_mgr.save.assert_called_once()

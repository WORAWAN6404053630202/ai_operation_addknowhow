"""
Regression + validation tests for utils.text_patterns.is_fee_or_duration_info_query
(and the ACTION_INTENT_RE it shares with PersonaSupervisor._ACTION_Q_RE).

Background: this gate was originally an inline pair of regexes in
persona_practical.py, computed only from `_intent_text` and used to force
action="answer" (all fee tiers combined, no entity_type/operation_group
asking) for purely informational fee/duration questions. Extracted to
utils/text_patterns.py (2026-09) as part of the PDF-book truncation/retrieval
fix chain (see pdf_chunking_truncation_and_retrieval_fixes project memory) so
it could be validated against a real query set before deciding whether the
trigger regex itself needed narrowing.

That validation surfaced a real bug, not in the trigger regex, but in the
negative/action-intent side: the original inline negation
(`อยากจด|ต้องการจด|จะจด|จะสมัคร|อยากสมัคร|ต้องการสมัคร|วิธีจด|ขั้นตอนการจด`)
was a much narrower, independently hand-rolled copy of
PersonaSupervisor._ACTION_Q_RE — missing "อยากขอ", "ต้องการยื่น", "จะขอใบ",
"วิธีลงทะเบียน", the bare "ขอใบ.../ยื่นขอใบ..." pattern, and the modify/cancel/
renew verbs entirely. A real query like "จะขอใบอนุญาตขายสุรา ค่าธรรมเนียม
เท่าไหร่" (real action-intent phrasing, see test_bot_e2e.py categories using
"จะขอใบอนุญาต...") would have been wrongly treated as purely informational.
Fixed by promoting the comprehensive pattern to utils.text_patterns.ACTION_INTENT_RE
and having both call sites share it — see that module's docstring for the
full rationale.

Evidence-based outcome for the TRIGGER regex (ค่าธรรมเนียม/เท่าไหร่/กี่วัน/...):
no bug found. Left unchanged — kept broad, since the 2026-09 fix already
covers the one real gap that broad trigger created (SME tax-rate "...อัตรา
เท่าไหร่" queries hitting a PDF-book narrative/table answer) by widening the
downstream instruction text to handle tiered-table content, not by narrowing
this trigger. This mirrors the RERANKER_MIN_SCORE precedent: don't change a
broad-but-working mechanism without live evidence it's actually wrong.
"""
import pytest

from utils.text_patterns import is_fee_or_duration_info_query, ACTION_INTENT_RE


@pytest.mark.unit
class TestGenuineFeeOrDurationQueriesFireTrue:
    """Real fee/duration follow-up questions from test_bot_e2e.py categories
    2.5 / 18.2 style — must keep returning True or those flows regress."""

    @pytest.mark.parametrize("query", [
        "แล้วค่าธรรมเนียมเท่าไหร่ค่ะ",
        "ค่าธรรมเนียมเท่าไหร่ค่ะ",
        "ใช้เวลากี่วันคะ",
        "ระยะเวลาดำเนินการนานแค่ไหน",
        "มีค่าใช้จ่ายอะไรบ้าง",
        "ต่ออายุใบอนุญาตนี้กี่บาท",
    ])
    def test_fires_true(self, query):
        assert is_fee_or_duration_info_query(query) is True


@pytest.mark.unit
class TestRateStylePdfBookQueriesFireTrue:
    """Q7-style: not a registration-fee question, but a rate/percentage
    question against PDF-book narrative content. Confirmed (2026-09 live
    testing) this SHOULD fire True — the fix for this case was broadening the
    downstream fee/duration instruction text to cover narrative/tiered-table
    content, not excluding these queries from the gate."""

    @pytest.mark.parametrize("query", [
        "ธุรกิจ SME ที่มีทุนไม่เกิน 5 ล้านบาท เสียภาษีเงินได้นิติบุคคลอัตราเท่าไหร่",
        "ค่าเช่าร้านทั่วไปคิดเป็นสัดส่วนเท่าไหร่ของยอดขาย",
    ])
    def test_fires_true(self, query):
        assert is_fee_or_duration_info_query(query) is True


@pytest.mark.unit
class TestActionIntentQueriesFireFalse:
    """User has real action intent (about to register/apply/renew/cancel) —
    must NOT fire, so the normal slot-filling flow still asks what the fee
    actually depends on instead of dumping all tiers immediately."""

    @pytest.mark.parametrize("query", [
        "อยากจดทะเบียนพาณิชย์ ค่าธรรมเนียมเท่าไหร่",
        "ต้องการสมัครใบอนุญาตขายสุรา ใช้เวลากี่วัน",
        # Regression cases for the specific gap this fix closed — these were
        # WRONGLY firing True before ACTION_INTENT_RE was shared:
        "จะขอใบอนุญาตขายสุรา ค่าธรรมเนียมเท่าไหร่",
        "อยากขอใบอนุญาตเปิดร้านอาหาร ต้องเสียค่าใช้จ่ายเท่าไหร่",
        "ต้องการยื่นขอใบอนุญาต ใช้เวลากี่วันคะ",
        "วิธีลงทะเบียนพาณิชย์อิเล็กทรอนิกส์ มีค่าใช้จ่ายเท่าไหร่",
        "ต้องการต่ออายุใบอนุญาต ค่าธรรมเนียมเท่าไหร่",
    ])
    def test_fires_false(self, query):
        assert is_fee_or_duration_info_query(query) is False


@pytest.mark.unit
class TestNonFeeQueriesFireFalse:
    """Sanity: queries with neither trigger keyword should never fire."""

    @pytest.mark.parametrize("query", [
        "ต้องใช้เอกสารอะไรบ้าง",
        "ใบอนุญาตนี้คืออะไร",
        "ปัจจัยสำคัญในการเลือกทำเลร้านมีอะไรบ้างคะ",
    ])
    def test_fires_false(self, query):
        assert is_fee_or_duration_info_query(query) is False


@pytest.mark.unit
class TestActionIntentRegexCoversRealBotE2eCases:
    """Pin ACTION_INTENT_RE against real action-intent phrasings that already
    appear as live steps in test_bot_e2e.py, so any future edit to this
    shared pattern can't silently stop matching real traffic shapes."""

    @pytest.mark.parametrize("query", [
        "จะขอใบอนุญาตเปิดร้านอาหารต้องทำอะไรบ้างค่ะ",
        "แล้วถ้าจะขอใบอนุญาตร้านอาหารด้วยล่ะค่ะ",
        "แล้วถ้าจะขอใบอนุญาตขายสุราด้วยล่ะค่ะ ต้องทำอะไรบ้าง",
        "อยากแก้ไขใบทะเบียนพาณิชย์ค่ะ ต้องทำยังไงบ้าง",
    ])
    def test_matches(self, query):
        assert ACTION_INTENT_RE.search(query)

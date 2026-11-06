"""
Unit tests for CACHE_SKIP_NO_ANSWER_RE (text_patterns.py) and the broadened
"don't cache a no-answer reply" guard it feeds in route_v1.py.

Live-reproduced bug (2026-11): a free-text refusal from the LLM (no
[[INFO_GAP]] tag) got cached under the shared response-cache key and was
replayed verbatim to later sessions asking a similarly-worded question.
These tests pin the two real reproduced phrasings plus a real good answer
that must NOT be skipped, to catch any future drift in either direction.
"""
import pytest

from utils.text_patterns import CACHE_SKIP_NO_ANSWER_RE


def _should_skip_cache(info_gap_detected: bool, bot_reply: str) -> bool:
    """Mirrors route_v1.py's _is_no_answer_reply (both /chat and /chat/stream)."""
    return bool(info_gap_detected) or bool(CACHE_SKIP_NO_ANSWER_RE.search(bot_reply or ""))


@pytest.mark.unit
class TestCacheSkipNoAnswerPattern:

    def test_matches_live_reproduced_case_no_fee_data_noted(self):
        reply = (
            "สำหรับใบรับรองการแจ้งจัดตั้งสถานที่จำหน่ายอาหารหรือสถานที่สะสมอาหาร "
            "(แบบ สอ.4) นั้น **ไม่มีข้อมูลระบุค่าธรรมเนียมในเอกสาร** ครับ"
        )
        assert CACHE_SKIP_NO_ANSWER_RE.search(reply)
        assert _should_skip_cache(False, reply) is True

    def test_matches_live_reproduced_case_no_data_found(self):
        reply = (
            "ไม่พบข้อมูลค่าธรรมเนียมแจ้งรายปีที่ระบุตามขนาดพื้นที่ร้าน 45 ตร.ม. ครับ "
            "หากต้องการทราบรายละเอียดเกี่ยวกับค่าธรรมเนียมประเภทอื่น ๆ "
            "สามารถติดต่อหน่วยงานที่เกี่ยวข้องโดยตรงครับ"
        )
        assert CACHE_SKIP_NO_ANSWER_RE.search(reply)
        assert _should_skip_cache(False, reply) is True

    def test_matches_original_bug_report_wording(self):
        reply = (
            "ผมไม่พบข้อมูลเกี่ยวกับค่าธรรมเนียมหนังสือรับรองการแจ้งรายปีในเอกสารที่มีอยู่ครับ\n\n"
            "แนะนำให้ติดต่อสำนักงานเขต หรือเทศบาล/อบต. ในพื้นที่ที่ร้านตั้งอยู่โดยตรง"
        )
        assert _should_skip_cache(False, reply) is True

    def test_does_not_flag_real_grounded_fee_answer(self):
        """The correct 45-sqm answer (live-verified, 550 baht) must still be cacheable."""
        reply = (
            "สำหรับร้านอาหารที่มีพื้นที่ 45 ตารางเมตร ค่าธรรมเนียมหนังสือรับรองการแจ้งรายปี"
            "จะคำนวณดังนี้ครับ:\n\n"
            "*   **พื้นที่ไม่เกิน 10 ตารางเมตร:** 200 บาทต่อปี\n"
            "*   **พื้นที่เกิน 10 ตารางเมตร:** คิดเพิ่มตารางเมตรละ 10 บาท\n\n"
            "ดังนั้น สำหรับพื้นที่ 45 ตารางเมตร: 200 + (35 * 10) = 550 บาท"
        )
        assert not CACHE_SKIP_NO_ANSWER_RE.search(reply)
        assert _should_skip_cache(False, reply) is False

    def test_tag_flag_alone_still_skips_cache(self):
        """Pre-existing path (INFO_GAP tag already extracted upstream) must keep working."""
        assert _should_skip_cache(True, "ยังไม่พบข้อมูลในเอกสารครับ") is True

    def test_empty_reply_does_not_match(self):
        assert CACHE_SKIP_NO_ANSWER_RE.search("") is None

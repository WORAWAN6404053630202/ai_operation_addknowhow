"""
Regression tests for PracticalPersonaService._extract_inline_service_urls
(model/persona_practical.py).

Live-reproduced bug (2026-09): a real service_channel value packed two
contact entries with URLs onto ONE undelimited line (no newline between
them):

    "ฝ่ายบริการลูกค้า rOS Service 24 ชั่วโมง https://...@442nurvs
     ฝ่าย rOS Operation จันทร์-ศุกร์ 09.00-18.00 น. https://...@253ztzoj"

The final rendered answer showed a garbled link bullet "- น." for the
second URL. Two compounding bugs, both fixed here:

1. Each URL's label was computed from the WHOLE line prefix, not the
   segment since the previous URL — for the 2nd+ URL this pulled the FIRST
   entry's own text (and URL) in too, pushing the segment well past the
   50-char "extract last item" fallback even though each real entry was
   individually short.
2. That fallback's numbered-list-marker regex matched "09.00-18.00 น." (a
   time-of-day, not a list) at both "09." and "18." (digit+dot each),
   splitting the text into garbage fragments and leaving a bare "น." as
   the final label.

No network access, no LLM calls — pure regex/string logic.
"""
import sys
import os

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))

import pytest

from model.persona_practical import PracticalPersonaService


@pytest.mark.unit
class TestInlineServiceUrlExtraction:

    def test_two_undelimited_entries_on_one_line_both_get_clean_labels(self):
        # The exact live-reproduced line (department names shortened only
        # for readability; the real bug depended on entry count and the
        # embedded time-of-day, not on these specific department names).
        line = (
            "ฝ่ายบริการลูกค้า rOS Service 24 ชั่วโมง https://manager.line.biz/account/@442nurvs "
            "ฝ่าย rOS Operation จันทร์-ศุกร์ 09.00-18.00 น. https://manager.line.biz/account/@253ztzoj"
        )
        pairs = PracticalPersonaService._extract_inline_service_urls(line)
        assert len(pairs) == 2
        desc1, url1 = pairs[0]
        desc2, url2 = pairs[1]
        assert url1 == "https://manager.line.biz/account/@442nurvs"
        assert url2 == "https://manager.line.biz/account/@253ztzoj"
        assert desc1 == "ฝ่ายบริการลูกค้า rOS Service 24 ชั่วโมง"
        assert desc2 == "ฝ่าย rOS Operation จันทร์-ศุกร์ 09.00-18.00 น."
        # The specific garbled fragment this bug produced must never reappear.
        assert desc2 != "น."
        assert "rvs" not in desc2  # a stray tail of the FIRST url's ending

    def test_time_of_day_alone_is_not_mistaken_for_a_numbered_list(self):
        # Isolates bug #2 on its own: a single (already-short) entry containing
        # a time-of-day must not trigger the length-based list-splitting path
        # at all, and even if it did, must not treat "09."/"18." as markers.
        line = "เปิดทุกวัน 09.00-18.00 น. https://example.com/contact"
        pairs = PracticalPersonaService._extract_inline_service_urls(line)
        assert len(pairs) == 1
        desc, url = pairs[0]
        assert desc == "เปิดทุกวัน 09.00-18.00 น."

    def test_genuine_numbered_list_still_splits_correctly(self):
        # Regression pin for the fallback path itself — must still work for
        # what it was originally built for.
        line = "1.กรอกข้อมูลร้านค้า 2.กรอกข้อมูลลงในหนังสือ 3.ส่งเอกสาร https://example.com/form " * 1
        # Pad past 50 chars deliberately so the length-based fallback engages.
        line = "0123456789" * 6 + " " + line
        pairs = PracticalPersonaService._extract_inline_service_urls(line)
        assert len(pairs) == 1
        desc, url = pairs[0]
        assert desc == "ส่งเอกสาร"

    def test_single_short_entry_no_truncation_needed(self):
        line = "เว็บไซต์ https://example.com/apply"
        pairs = PracticalPersonaService._extract_inline_service_urls(line)
        assert pairs == [("เว็บไซต์", "https://example.com/apply")]

    def test_three_entries_on_one_line_each_scoped_independently(self):
        line = (
            "สาขา A https://example.com/a "
            "สาขา B https://example.com/b "
            "สาขา C https://example.com/c"
        )
        pairs = PracticalPersonaService._extract_inline_service_urls(line)
        assert pairs == [
            ("สาขา A", "https://example.com/a"),
            ("สาขา B", "https://example.com/b"),
            ("สาขา C", "https://example.com/c"),
        ]

    def test_no_url_on_line_returns_empty(self):
        assert PracticalPersonaService._extract_inline_service_urls("ไม่มีลิงก์ในบรรทัดนี้") == []

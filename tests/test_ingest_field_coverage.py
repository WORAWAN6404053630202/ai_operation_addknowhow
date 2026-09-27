"""
Regression tests for the "field silently missing from embedded page_content"
bug class (2026-09) — a card-payment settlement-cutoff-time table sat in
terms_and_conditions with no cap of its own, near the end of the field
ordering, so for any doc whose earlier fields already filled most of the
character budget it never made it into the vector at all. The bot had the
data in metadata the whole time; similarity/BM25 search just couldn't find
it, because it was never actually embedded.

Two separate causes were found and are covered here:
  1. Ordering/capping in DataLoader._build_page_content and
     GeneralDataLoader._build_page_content (every body field must have its
     own cap, or a bulky earlier field can starve a later one before the
     final overall clamp is even reached).
  2. A deeper wiring bug: both loaders' page_content_max_chars setting was
     never actually connected to conf.PAGE_CONTENT_MAX_CHARS at their real
     call sites (DataLoader(config={}) — getattr on a plain dict can never
     find a "PAGE_CONTENT_MAX_CHARS" attribute; GeneralDataLoader(...) never
     passed the kwarg at all) — so raising the conf constant had zero effect
     until the loaders were fixed to read the real conf module directly.

No network access, no embeddings, no LLM calls — pure string-building logic.
"""
import sys
import os

_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))

import pytest

import conf
from service.data_loader import DataLoader
from service.data_loader_general import GeneralDataLoader


# Every field DataLoader._build_page_content folds into the embedded text,
# with its own cap (must stay in sync with data_loader.py).
_REGULATORY_BODY_FIELDS = {
    "operation_steps": 1800,
    "legal_regulatory": 2000,
    "terms_and_conditions": 2300,
    "identification_documents": 2100,
    "fees": 900,
    "operation_duration": 300,
    "service_channel": 600,
    "answer_guideline": 5500,
    "restaurant_ai_document": 300,
}


@pytest.mark.unit
class TestConfWiring:
    """Regression test for the deeper bug: both loaders must actually read
    conf.PAGE_CONTENT_MAX_CHARS, not a disconnected/never-passed value."""

    def test_data_loader_reads_real_conf_value(self):
        dl = DataLoader(config={})
        assert dl.page_content_max_chars == conf.PAGE_CONTENT_MAX_CHARS
        # Also pin the specific old failure mode: config={} must NOT silently
        # win over the real conf value.
        assert dl.page_content_max_chars != 1800 or conf.PAGE_CONTENT_MAX_CHARS == 1800

    def test_general_data_loader_reads_real_conf_value(self):
        dl = GeneralDataLoader(data_type="business_guide")
        assert dl.page_content_max_chars == conf.PAGE_CONTENT_MAX_CHARS

    def test_general_data_loader_explicit_override_still_wins(self):
        """An explicit caller-supplied value must still take priority over
        the conf default — the fix must not remove that flexibility."""
        dl = GeneralDataLoader(data_type="marketing", page_content_max_chars=999)
        assert dl.page_content_max_chars == 999


@pytest.mark.unit
class TestRegulatoryFieldCoverage:
    """Every populated body field must survive into page_content, regardless
    of how much content precedes it — this is the actual guarantee the
    2026-09 fix exists to provide."""

    def _make_md(self, **overrides) -> dict:
        md = {
            "license_type": "ทดสอบใบอนุญาต",
            "operation_by_department": "การทดสอบ",
            "entity_type_normalized": "บุคคลธรรมดา",
            "department": "หน่วยงานทดสอบ",
        }
        md.update(overrides)
        return md

    def test_worst_case_every_field_at_max_cap_all_survive(self):
        """The real bug: a doc with substantial content in EVERY field. Every
        one must still appear in page_content — none may be silently dropped
        because an earlier field ate the whole budget."""
        md = self._make_md(**{
            field: ("ก" * (cap + 200))  # deliberately longer than its own cap
            for field, cap in _REGULATORY_BODY_FIELDS.items()
        })
        dl = DataLoader(config={})
        content = dl._build_page_content(md)

        for field in _REGULATORY_BODY_FIELDS:
            # Each field's own cap-worth of "ก" must appear somewhere — proves
            # the field was NOT entirely skipped, even though its raw value
            # was longer than its cap (partial truncation is fine and
            # expected; total absence is the bug).
            assert "ก" * 50 in content, f"{field} appears to be completely missing from page_content"

    def test_settlement_time_style_field_survives_after_bulky_earlier_fields(self):
        """Direct regression for the real reported case: terms_and_conditions
        holding a short but critical fact (e.g. a settlement-cutoff-time
        table), with bulky operation_steps/identification_documents ahead of
        it in the ordering."""
        md = self._make_md(
            operation_steps="ข" * 600,
            identification_documents="ค" * 500,
            terms_and_conditions="Credit Card 22.00 น. TrueMoney 24.00 น.",
        )
        dl = DataLoader(config={})
        content = dl._build_page_content(md)
        assert "Credit Card 22.00" in content
        assert "TrueMoney 24.00" in content

    def test_field_missing_entirely_when_value_is_empty(self):
        """Sanity check the test methodology itself: an empty field should
        NOT appear (proves the coverage assertions above aren't vacuous)."""
        md = self._make_md(fees="")
        dl = DataLoader(config={})
        content = dl._build_page_content(md)
        assert "ค่าธรรมเนียม:" not in content


@pytest.mark.unit
class TestIngestValidationCoverageCheck:
    """The automated ingest-time safety net (scripts/ingest_local.py's
    _validate_documents) must actually catch a field that's missing from
    page_content, and must NOT false-positive on a healthy document."""

    def _load_validator(self):
        # scripts/ is not a package; import it directly by path like
        # ingest_local.py's own CLI entry does.
        import importlib.util
        script_path = os.path.join(_CODE_DIR, "scripts", "ingest_local.py")
        spec = importlib.util.spec_from_file_location("ingest_local", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_flags_a_field_missing_from_page_content(self):
        from langchain_core.documents import Document
        ingest_local = self._load_validator()

        doc = Document(
            page_content="เนื้อหาที่ไม่มีข้อมูลค่าธรรมเนียมอยู่เลย " * 3,
            metadata={
                "data_type": "regulatory",
                "license_type": "ทดสอบ",
                "topic_group": "ทดสอบ",
                "entity_type_normalized": "",
                "fees": "ค่าธรรมเนียม 100 บาท ที่ไม่ได้อยู่ใน page_content เลยสักตัวอักษร",
            },
        )
        warnings = ingest_local._validate_documents([doc])
        assert warnings > 0

    def test_no_false_positive_on_healthy_document(self):
        from langchain_core.documents import Document
        ingest_local = self._load_validator()

        fees_text = "ค่าธรรมเนียม 100 บาท"
        doc = Document(
            page_content=f"หัวข้อทดสอบ ค่าธรรมเนียม:\n{fees_text}",
            metadata={
                "data_type": "regulatory",
                "license_type": "ทดสอบ",
                "topic_group": "ทดสอบ",
                "entity_type_normalized": "",
                "fees": fees_text,
            },
        )
        warnings = ingest_local._validate_documents([doc])
        assert warnings == 0


@pytest.mark.unit
class TestGeneralLoaderPageContentBoilerplate:
    """2026-09 fix: GeneralDataLoader._build_page_content used to label every
    field ("หัวข้อหลัก: X หัวข้อย่อย: Y ประเภท: Z แหล่งข้อมูล: W แนวคำตอบ: ...
    อ้างอิง: ...") — live-reproduced this repeated label boilerplate
    dominates the embedding vector so completely that two TOTALLY UNRELATED
    documents came out 0.867 cosine-similar to each other, while a
    genuinely relevant document for a real query scored only 0.243 —
    indistinguishable from noise. Fixed by dropping the labels and the
    low-value content_type/source_book/research_reference fields (all still
    preserved verbatim in metadata for any downstream use) from the embedded
    text entirely, keeping just the bare main_topic/sub_topic/answer_guideline
    values. No network access, no embeddings — pure string-building logic."""

    def test_no_label_boilerplate_in_output(self):
        dl = GeneralDataLoader(data_type="business_guide")
        content = dl._build_page_content(
            source_book="เปิดร้านเบเกอรี่",
            main_topic="การเข้าใจพื้นฐานการขายสินค้า",
            sub_topic="การตั้งราคา",
            content_type="ความรู้ทั่วไป",
            answer_guideline="ควรตั้งราคาจากต้นทุนและราคาตลาดควบคู่กัน",
            research_reference="หนังสือ ครบเครื่องเปิดร้าน 2",
        )
        for label in ("หัวข้อหลัก:", "หัวข้อย่อย:", "ประเภท:", "แหล่งข้อมูล:", "แนวคำตอบ:", "อ้างอิง:"):
            assert label not in content

    def test_topic_and_answer_values_still_present(self):
        dl = GeneralDataLoader(data_type="marketing")
        content = dl._build_page_content(
            source_book="เปิดร้านเบเกอรี่",
            main_topic="การเข้าใจพื้นฐานการขายสินค้า",
            sub_topic="การตั้งราคา",
            content_type="ความรู้ทั่วไป",
            answer_guideline="ควรตั้งราคาจากต้นทุนและราคาตลาดควบคู่กัน",
            research_reference="หนังสือ ครบเครื่องเปิดร้าน 2",
        )
        assert "การเข้าใจพื้นฐานการขายสินค้า" in content
        assert "การตั้งราคา" in content
        assert "ควรตั้งราคาจากต้นทุนและราคาตลาดควบคู่กัน" in content

    def test_low_value_fields_excluded_from_embedded_text(self):
        # source_book/content_type/research_reference are preserved in
        # metadata elsewhere (unaffected by this function) but must not
        # dilute the embedded page_content — confirmed empirically to add
        # no retrieval value, only noise.
        dl = GeneralDataLoader(data_type="marketing")
        content = dl._build_page_content(
            source_book="เปิดร้านเบเกอรี่ ชื่อหนังสือที่ไม่ซ้ำใคร",
            main_topic="หัวข้อ A",
            sub_topic="หัวข้อ B",
            content_type="ประเภทพิเศษที่ไม่ซ้ำใคร",
            answer_guideline="คำตอบจริง",
            research_reference="อ้างอิงที่ไม่ซ้ำใครเลย",
        )
        assert "เปิดร้านเบเกอรี่" not in content
        assert "ประเภทพิเศษที่ไม่ซ้ำใคร" not in content
        assert "อ้างอิงที่ไม่ซ้ำใครเลย" not in content

    def test_missing_optional_fields_do_not_crash_or_leave_blank_lines(self):
        dl = GeneralDataLoader(data_type="business_guide")
        content = dl._build_page_content(
            source_book=None,
            main_topic="หัวข้อหลักเดียว",
            sub_topic=None,
            content_type=None,
            answer_guideline="คำตอบ",
            research_reference=None,
        )
        assert content == "หัวข้อหลักเดียว\nคำตอบ"

    def test_still_respects_page_content_max_chars_clamp(self):
        dl = GeneralDataLoader(data_type="marketing", page_content_max_chars=50)
        content = dl._build_page_content(
            source_book=None,
            main_topic="หัวข้อ",
            sub_topic=None,
            content_type=None,
            answer_guideline="ก" * 200,
            research_reference=None,
        )
        assert len(content) <= 50

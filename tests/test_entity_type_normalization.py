"""
Unit tests for DataLoader._normalize_entity_type.

Covers exact matches, sub-type variants, fuzzy keyword matching,
and invalid / empty inputs. No LLM or Chroma connection required.
"""
import pytest
from service.data_loader import DataLoader


@pytest.mark.unit
class TestEntityTypeNormalization:

    def test_exact_juristic_person(self):
        assert DataLoader._normalize_entity_type("นิติบุคคล") == "นิติบุคคล"

    def test_company_limited(self):
        assert DataLoader._normalize_entity_type("บริษัทจำกัด") == "นิติบุคคล"

    def test_company_public_limited(self):
        assert DataLoader._normalize_entity_type("บริษัทมหาชนจำกัด") == "นิติบุคคล"

    def test_limited_partnership(self):
        assert DataLoader._normalize_entity_type("ห้างหุ้นส่วนจำกัด") == "นิติบุคคล"

    def test_ordinary_partnership(self):
        assert DataLoader._normalize_entity_type("ห้างหุ้นส่วนสามัญ") == "นิติบุคคล"

    def test_individual_exact(self):
        assert DataLoader._normalize_entity_type("บุคคลธรรมดา") == "บุคคลธรรมดา"

    def test_individual_sole_owner(self):
        assert DataLoader._normalize_entity_type("บุคคลธรรมดา กิจการเจ้าของคนเดียว") == "บุคคลธรรมดา"

    def test_fuzzy_company_variant(self):
        # User types "บริษัท ลิมิเต็ด" — fuzzy keyword "บริษัท" should normalize to นิติบุคคล
        assert DataLoader._normalize_entity_type("บริษัท ลิมิเต็ด") == "นิติบุคคล"

    def test_fuzzy_partnership_variant(self):
        assert DataLoader._normalize_entity_type("ห้างหุ้นส่วน ABC") == "นิติบุคคล"

    def test_unknown_value_returns_none(self):
        assert DataLoader._normalize_entity_type("ไม่รู้") is None

    def test_unrelated_string_returns_none(self):
        assert DataLoader._normalize_entity_type("ร้านอาหาร") is None

    def test_empty_string_returns_none(self):
        assert DataLoader._normalize_entity_type("") is None

    def test_none_input_returns_none(self):
        assert DataLoader._normalize_entity_type(None) is None

    def test_whitespace_only_returns_none(self):
        assert DataLoader._normalize_entity_type("   ") is None

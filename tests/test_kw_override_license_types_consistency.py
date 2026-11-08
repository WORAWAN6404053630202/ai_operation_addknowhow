"""
Regression test for the 2026-11 license_type drift fix (hardcoded-logic
audit follow-up). persona_supervisor.py hand-maintains the SAME license_type
strings independently across 3-4 structures: `_KW_OVERRIDE_LICENSE_TYPES`
(a flat list, validated against the real corpus at startup by
`_validate_license_types_at_startup`), `_SUPERVISOR_KW_OVERRIDE` (regex →
license_type), `_MULTI_TOPIC_LICENSE_KEYWORDS` (regex → license_type, a
separate predictive classifier), and `_AREA_SENSITIVE_LICENSES` (a frozenset,
one specific entry only).

This investigation started because 8/9 `_KW_OVERRIDE_LICENSE_TYPES` entries
didn't match any real corpus document — live-queried the real 869-doc
baseline directly to resolve each as either a genuine rename (5 entries,
fixed) or a confirmed content gap (3-4 entries, deliberately left as-is for
their keyword-trigger value). Separately found `_SUPERVISOR_KW_OVERRIDE` had
`'กองทุนเงินทดแทน'` with NO counterpart in `_KW_OVERRIDE_LICENSE_TYPES` at
all — a second, independent instance of exactly the 3-way drift this test
exists to catch. Pure structural test: no corpus/network access, just
asserts the 3 structures agree with each other on principle — it cannot
tell you whether a given string is REAL (that's `_validate_license_types_
at_startup`'s job, which needs a live corpus), only that the lists haven't
silently drifted apart from each other again.
"""
from __future__ import annotations

import pytest

from model.persona_supervisor import PersonaSupervisor


@pytest.mark.unit
class TestKwOverrideListsStayInSync:
    def test_every_kw_override_license_type_is_a_supervisor_kw_override_value(self):
        override_values = {lt for _, lt in PersonaSupervisor._SUPERVISOR_KW_OVERRIDE}
        missing = set(PersonaSupervisor._KW_OVERRIDE_LICENSE_TYPES) - override_values
        assert not missing, (
            f"_KW_OVERRIDE_LICENSE_TYPES has entries with no counterpart in "
            f"_SUPERVISOR_KW_OVERRIDE's values: {missing}"
        )

    def test_every_supervisor_kw_override_value_is_in_kw_override_license_types(self):
        override_values = {lt for _, lt in PersonaSupervisor._SUPERVISOR_KW_OVERRIDE}
        missing = override_values - set(PersonaSupervisor._KW_OVERRIDE_LICENSE_TYPES)
        assert not missing, (
            f"_SUPERVISOR_KW_OVERRIDE has values with no counterpart in "
            f"_KW_OVERRIDE_LICENSE_TYPES (the list _validate_license_types_at_startup "
            f"actually checks against the real corpus — these would never be "
            f"validated at all): {missing}"
        )

    def test_every_kw_override_license_type_is_a_multi_topic_keyword_value(self):
        multi_topic_values = {lt for _, lt in PersonaSupervisor._MULTI_TOPIC_LICENSE_KEYWORDS}
        missing = set(PersonaSupervisor._KW_OVERRIDE_LICENSE_TYPES) - multi_topic_values
        assert not missing, (
            f"_KW_OVERRIDE_LICENSE_TYPES has entries with no counterpart in "
            f"_MULTI_TOPIC_LICENSE_KEYWORDS's values: {missing}"
        )

    def test_area_sensitive_licenses_are_known_license_types(self):
        unknown = PersonaSupervisor._AREA_SENSITIVE_LICENSES - set(
            PersonaSupervisor._KW_OVERRIDE_LICENSE_TYPES
        )
        assert not unknown, (
            f"_AREA_SENSITIVE_LICENSES references license_type(s) not in "
            f"_KW_OVERRIDE_LICENSE_TYPES: {unknown}"
        )

    def test_rare_lt_patterns_values_are_known_license_types(self):
        """_RARE_LT_PATTERNS intentionally excludes some known content-gap
        entries (e.g. ทะเบียนนายจ้าง, removed 2026-11 since its exact-filter
        fetch could never succeed) — but whatever IS in there should still
        be a real, tracked license_type, not a third independent spelling."""
        unknown = set(PersonaSupervisor._RARE_LT_PATTERNS.values()) - set(
            PersonaSupervisor._KW_OVERRIDE_LICENSE_TYPES
        )
        assert not unknown, (
            f"_RARE_LT_PATTERNS references license_type(s) not in "
            f"_KW_OVERRIDE_LICENSE_TYPES: {unknown}"
        )

    def test_no_duplicate_entries_within_kw_override_license_types(self):
        values = PersonaSupervisor._KW_OVERRIDE_LICENSE_TYPES
        assert len(values) == len(set(values))

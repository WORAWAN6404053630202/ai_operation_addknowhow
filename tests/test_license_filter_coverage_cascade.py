"""
Regression pin for the department/license_type retrieval-safety fix in
persona_supervisor.py (2026-11, retrieval-safety audit).

Background: two call sites inside _ensure_practical_retrieval_for_legal
filter on license_type via hardcoded keyword patterns (_RARE_LT_PATTERNS,
_MULTI_TOPIC_LICENSE_KEYWORDS) and, on ANY non-empty result, committed to it
and returned immediately — with no check that the result was complete/
correct, and no use of the regulatory-identity dominance guard
(utils.text_patterns.regulatory_identity_mismatch_should_reject) that a
THIRD, lower-traffic call site in the same method already used (see
TestRegulatoryIdentityGuardWiring in test_supervisor_fallback_quality_gates.py).
This mirrors the shape of the entity_type_normalized contamination saga (see
memory entity_type_retrieval_contamination_5round_saga_2026-11.md) for a
different field pair.

A live-corpus check (code/scripts/field_blank_rate_check.py, run 2026-11
against the real 1023-doc corpus) found 0% blank department among all docs
with a non-blank license_type — so the $in:[val, ""] widening idiom used
elsewhere for entity_type_normalized/location would do ~nothing for this gap
today. The real risk is a candidate doc whose department holds a WRONG value
(confirmed real case: "NTT DATA", a vendor name — persona_practical.py:
3474-3489) winning by exact-match coincidence. Phase 2 (this file) wires the
existing identity guard into both sites; Phase 3 (separate, later) adds a
coverage-adequacy + supplement-merge mechanism for the "thin but non-empty
result" half of the gap, which this guard does not address.

Both new guard sites are gated behind the SAME flag as the sibling site
(conf.REGULATORY_IDENTITY_GUARD_ENFORCE, default False) — log-only soft
launch, no default-behavior change until the flag is explicitly enabled.

Phase 3 (same pass, same file): added a coverage-adequacy check
(PersonaSupervisor._is_structurally_adequate, backed by the now-shared
_structural_field_count — previously duplicated as two independently
drifting copies, _STRUCTURAL_KEYS/_structural_score in the multi-topic
loop and _STRUCTURAL_KEYS2/an inline lambda elsewhere) plus an unfiltered
supplement search, appended (never replacing) a structurally-thin filtered
result. Gated behind two NEW flags (RARE_LT_COVERAGE_SUPPLEMENT_ENABLED,
MULTI_TOPIC_COVERAGE_SUPPLEMENT_ENABLED, both default False) — a genuinely
different mechanism from the identity guard, not a reuse of
REGULATORY_IDENTITY_GUARD_ENFORCE.

A THIRD call site with the identical shape was found while refactoring the
duplicated structural-keys constant: _route_pending_slot_to_persona has its
own, independent multi-topic license_type merge loop (reached via the
"topic" slot-answer path rather than free-text query detection in
_ensure_practical_retrieval_for_legal) with the exact same gap. Both the
identity guard and the coverage-adequacy supplement were applied there too
in this same pass — see the TestTopicSlotMultiTopicLoop* classes below.
"""
from __future__ import annotations

import inspect
import re

import pytest

from model.persona_supervisor import PersonaSupervisor


def _full_source() -> str:
    return inspect.getsource(PersonaSupervisor._ensure_practical_retrieval_for_legal)


def _rare_lt_loop_source() -> str:
    source = _full_source()
    m = re.search(
        r"for _rare_pat, _rare_lt in self\._RARE_LT_PATTERNS\.items\(\):"
        r"(.*?)break  # only match first pattern",
        source, re.DOTALL,
    )
    assert m, "_RARE_LT_PATTERNS loop not found in _ensure_practical_retrieval_for_legal"
    return m.group(1)


def _multi_topic_merge_loop_source() -> str:
    source = _full_source()
    m = re.search(
        r"_merged: List\[Dict\] = \[\](.*?)except Exception as _e_mt:",
        source, re.DOTALL,
    )
    assert m, "multi-topic merge loop not found in _ensure_practical_retrieval_for_legal"
    return m.group(1)


def _topic_slot_merge_loop_source() -> str:
    source = inspect.getsource(PersonaSupervisor._route_pending_slot_to_persona)
    m = re.search(
        r"_merged_mt2: List\[Dict\] = \[\](.*?)_merged_mt2 = _merged_mt2\[:_total_cap2\]",
        source, re.DOTALL,
    )
    assert m, "topic-slot multi-topic merge loop not found in _route_pending_slot_to_persona"
    return m.group(1)


@pytest.mark.unit
class TestRareLtLoopIdentityGuardWiring:
    """This loop filters to a single _rare_lt by construction, so comparing
    license_type is structurally a no-op — only `department` can genuinely
    disagree within the result set. Guard only runs when the top doc has a
    confident "_sim" score (absent on the unscored except-fallback path),
    same skip-if-unscored shape as the sibling fallback site."""

    def test_guard_is_wired_into_the_loop(self):
        block = _rare_lt_loop_source()
        assert "_shared_regulatory_identity_mismatch_should_reject(" in block

    def test_guard_compares_department_only_not_license_type(self):
        """license_type comparison would always be a no-op here (every doc in
        _raw_docs already shares _rare_lt by construction of the `filter=`
        passed to similarity_search) — the implementation must not waste a
        call on it or imply a check that can never fire."""
        block = _rare_lt_loop_source()
        guard_calls = block.count("_shared_regulatory_identity_mismatch_should_reject(")
        assert guard_calls == 1, f"expected exactly 1 guard call (department only), found {guard_calls}"

    def test_guard_is_log_only_by_default_gated_on_the_shared_flag(self):
        block = _rare_lt_loop_source()
        assert "REGULATORY_IDENTITY_GUARD_ENFORCE" in block
        assert "would drop" in block
        assert 'getattr(conf, "REGULATORY_IDENTITY_GUARD_ENFORCE", False)' in block

    def test_guard_never_drops_every_doc(self):
        block = _rare_lt_loop_source()
        assert "if _rare_kept and getattr(conf, \"REGULATORY_IDENTITY_GUARD_ENFORCE\", False)" in block

    def test_guard_skips_when_top_doc_has_no_confident_score(self):
        block = _rare_lt_loop_source()
        assert "if _rare_top_score > 0.0:" in block

    def test_guard_is_exception_safe(self):
        block = _rare_lt_loop_source()
        assert "except Exception as _rare_rid_e" in block


@pytest.mark.unit
class TestMultiTopicMergeLoopIdentityGuardWiring:
    """This loop deliberately merges N DIFFERENT license_types — the guard
    must only ever compare docs WITHIN the same _lt_name's _all_lt_docs,
    never across different _detected_licenses, or it would reject a second,
    genuinely-correct license's docs for disagreeing with the first's
    department. No per-doc similarity score exists on this unscored `.get()`
    path, so the reference value is a within-group department majority vote,
    not a retrieval score."""

    def test_guard_is_wired_into_the_loop(self):
        block = _multi_topic_merge_loop_source()
        assert "_shared_regulatory_identity_mismatch_should_reject(" in block

    def test_guard_runs_before_the_entity_type_filter_not_after(self):
        """Department-consistency and entity_type scoping are orthogonal —
        pin that the guard operates on the full _all_lt_docs fetched for
        this _lt_name, before _filtered_lt narrows it further."""
        block = _multi_topic_merge_loop_source()
        guard_pos = block.find("_shared_regulatory_identity_mismatch_should_reject(")
        entity_filter_pos = block.find("Apply entity_type filter if known")
        assert guard_pos != -1 and entity_filter_pos != -1
        assert guard_pos < entity_filter_pos

    def test_guard_uses_majority_vote_not_a_cross_license_reference(self):
        """Must tally department values WITHIN _all_lt_docs for this one
        _lt_name only — no variable from a previous loop iteration should
        leak in as the reference value."""
        block = _multi_topic_merge_loop_source()
        assert "_mt_dept_tally" in block
        assert "for _d in _all_lt_docs:" in block

    def test_guard_requires_at_least_two_distinct_department_values_to_fire(self):
        """A single doc, or a group where every doc shares the same
        department, has nothing to disagree about — must not even attempt
        the guard in that case (avoids a degenerate self-reject)."""
        block = _multi_topic_merge_loop_source()
        assert "if len(_mt_dept_tally) >= 2:" in block

    def test_guard_is_log_only_by_default_gated_on_the_shared_flag(self):
        block = _multi_topic_merge_loop_source()
        assert "REGULATORY_IDENTITY_GUARD_ENFORCE" in block
        assert "would drop" in block
        assert 'getattr(conf, "REGULATORY_IDENTITY_GUARD_ENFORCE", False)' in block

    def test_guard_never_drops_every_doc(self):
        block = _multi_topic_merge_loop_source()
        assert "if _mt_kept and getattr(conf, \"REGULATORY_IDENTITY_GUARD_ENFORCE\", False)" in block

    def test_guard_is_exception_safe(self):
        block = _multi_topic_merge_loop_source()
        assert "except Exception as _mt_rid_e" in block

    def test_guard_reuses_the_shared_floor_via_synthetic_confident_score(self):
        """Majority consensus (not a retrieval score) stands in for "top_score"
        — passed as 1.0, safely above the shared 0.70 floor, so the reused
        guard function's score-gating behaves as intended without a second
        threshold constant being introduced."""
        block = _multi_topic_merge_loop_source()
        assert "_mt_majority_dept, _d_dept, 1.0," in block


@pytest.mark.unit
class TestNeitherNewGuardTouchesLicenseTypeComparison:
    """Both new sites filter BY license_type already — re-deriving a
    license_type mismatch inside either would always be comparing a value
    against itself. Only department is a genuine cross-check here."""

    def test_rare_lt_loop_guard_call_has_no_license_type_argument(self):
        block = _rare_lt_loop_source()
        m = re.search(
            r"_shared_regulatory_identity_mismatch_should_reject\((.*?)\)",
            block, re.DOTALL,
        )
        assert m
        assert "license_type" not in m.group(1)

    def test_multi_topic_loop_guard_call_has_no_license_type_argument(self):
        block = _multi_topic_merge_loop_source()
        m = re.search(
            r"_shared_regulatory_identity_mismatch_should_reject\((.*?)\)",
            block, re.DOTALL,
        )
        assert m
        assert "license_type" not in m.group(1)


@pytest.mark.unit
class TestLiveCorpusBlankRateInformsThisGap:
    """Pins the finding from code/scripts/field_blank_rate_check.py (run
    2026-11 against the live 1023-doc corpus): among all docs with a
    non-blank license_type, 0% have a blank department. This means the
    $in:[val, ""] widening idiom used for entity_type_normalized/location
    elsewhere in this file would have near-zero effect on THIS gap today —
    documented here so a future reader isn't tempted to "fix" this only by
    adding $in and considering the gap closed; the identity guard above and
    the coverage-adequacy mechanism (Phase 3, separate) are what matter.
    """

    def test_in_widening_pattern_exists_elsewhere_for_other_fields(self):
        source = _full_source()
        assert '"entity_type_normalized"] = {"$in": [_fb_entity_type, ""]}' in source

    def test_in_widening_pattern_does_not_exist_for_department_or_license_type(self):
        source = _full_source()
        assert '"department"] = {"$in"' not in source
        assert '"license_type"] = {"$in"' not in source


@pytest.mark.unit
class TestStructuralAdequacyHelper:
    """Pure-logic tests for PersonaSupervisor._structural_field_count /
    _is_structurally_adequate — the shared helper promoted from two
    independently drifting copies (see this file's module docstring)."""

    def _doc(self, **metadata) -> dict:
        return {"content": "x", "metadata": metadata}

    def test_counts_only_the_five_structural_keys(self):
        doc = self._doc(operation_steps="step 1", fees="100 บาท", irrelevant_field="x")
        assert PersonaSupervisor._structural_field_count(doc) == 2

    def test_blank_nan_and_none_string_values_do_not_count(self):
        doc = self._doc(operation_steps="", fees="nan", legal_regulatory="None", terms_and_conditions="real value")
        assert PersonaSupervisor._structural_field_count(doc) == 1

    def test_missing_metadata_counts_as_zero(self):
        assert PersonaSupervisor._structural_field_count({"content": "x"}) == 0

    def test_adequate_when_one_doc_meets_the_default_threshold(self):
        docs = [self._doc(), self._doc(operation_steps="a", fees="b")]
        assert PersonaSupervisor._is_structurally_adequate(docs) is True

    def test_inadequate_when_no_doc_meets_the_default_threshold(self):
        docs = [self._doc(operation_steps="a"), self._doc(fees="b")]
        assert PersonaSupervisor._is_structurally_adequate(docs) is False

    def test_empty_doc_list_is_inadequate(self):
        assert PersonaSupervisor._is_structurally_adequate([]) is False

    def test_min_fields_threshold_is_configurable(self):
        docs = [self._doc(operation_steps="a", fees="b")]
        assert PersonaSupervisor._is_structurally_adequate(docs, min_fields=2) is True
        assert PersonaSupervisor._is_structurally_adequate(docs, min_fields=3) is False


@pytest.mark.unit
class TestRareLtLoopCoverageSupplement:
    """Phase 3: a non-empty but structurally thin rare-license result used
    to commit and return immediately. Gated on a NEW flag (not the identity
    guard's flag — this is a different mechanism), appends rather than
    replaces, and is exception-safe."""

    def test_supplement_gated_on_its_own_dedicated_flag(self):
        block = _rare_lt_loop_source()
        assert "RARE_LT_COVERAGE_SUPPLEMENT_ENABLED" in block
        # The supplement's own gating condition must not reuse the identity
        # guard's flag for this different mechanism (the guard block ABOVE it
        # legitimately mentions REGULATORY_IDENTITY_GUARD_ENFORCE twice —
        # that's a separate mechanism, not what this test is checking).
        supplement_gate = re.search(
            r"if \(\s*getattr\(conf, \"RARE_LT_COVERAGE_SUPPLEMENT_ENABLED\".*?\):",
            block, re.DOTALL,
        )
        assert supplement_gate, "supplement gate condition not found"
        assert "REGULATORY_IDENTITY_GUARD_ENFORCE" not in supplement_gate.group(0)

    def test_supplement_checks_adequacy_before_searching(self):
        block = _rare_lt_loop_source()
        assert "PersonaSupervisor._is_structurally_adequate(_filtered)" in block

    def test_supplement_appends_rather_than_replaces(self):
        block = _rare_lt_loop_source()
        assert "_filtered = _filtered + _rare_supplement" in block

    def test_supplement_dedupes_against_existing_docs(self):
        block = _rare_lt_loop_source()
        assert "_rare_seen_fp" in block

    def test_supplement_is_exception_safe(self):
        block = _rare_lt_loop_source()
        assert "except Exception as _rare_supp_e" in block


@pytest.mark.unit
class TestMultiTopicMergeLoopCoverageSupplement:
    """Phase 3, per-detected-license variant: only the one thin license's
    docs are supplemented — other, adequately-covered licenses in the same
    merge are untouched."""

    def test_supplement_gated_on_its_own_dedicated_flag(self):
        block = _multi_topic_merge_loop_source()
        assert "MULTI_TOPIC_COVERAGE_SUPPLEMENT_ENABLED" in block

    def test_supplement_checks_adequacy_before_searching(self):
        block = _multi_topic_merge_loop_source()
        assert "PersonaSupervisor._is_structurally_adequate(_filtered_lt)" in block

    def test_supplement_anchors_the_search_on_this_one_license_name(self):
        """Must use f"{_lt_name} {q}" — anchoring on the specific license,
        not a bare unfiltered q, so the supplement stays on-topic for the
        license it's meant to fill a gap for."""
        block = _multi_topic_merge_loop_source()
        assert 'f"{_lt_name} {q}"' in block

    def test_supplement_appends_rather_than_replaces(self):
        block = _multi_topic_merge_loop_source()
        assert "_filtered_lt = _filtered_lt + _mt_supplement" in block

    def test_supplement_is_exception_safe(self):
        block = _multi_topic_merge_loop_source()
        assert "except Exception as _mt_supp_e" in block

    def test_sort_uses_the_shared_helper_not_a_local_copy(self):
        block = _multi_topic_merge_loop_source()
        assert "PersonaSupervisor._structural_field_count" in block
        assert "_STRUCTURAL_KEYS =" not in block
        assert "def _structural_score" not in block


@pytest.mark.unit
class TestTopicSlotMultiTopicLoopIdentityGuardWiring:
    """A THIRD call site with the identical shape, found while refactoring
    the duplicated structural-keys constant: _route_pending_slot_to_persona
    has its own, independent multi-topic license_type merge loop (reached
    via the "topic" slot-answer path, not free-text query detection) with
    the same missing-guard gap as the two sites above. Same fix, same flag,
    applied here too in this same pass rather than left as a known gap."""

    def test_guard_is_wired_into_the_loop(self):
        block = _topic_slot_merge_loop_source()
        assert "_shared_regulatory_identity_mismatch_should_reject(" in block

    def test_guard_uses_majority_vote_within_this_license_only(self):
        block = _topic_slot_merge_loop_source()
        assert "_mt2_dept_tally" in block
        assert "for _d2 in _lt_docs2:" in block

    def test_guard_requires_at_least_two_distinct_department_values_to_fire(self):
        block = _topic_slot_merge_loop_source()
        assert "if len(_mt2_dept_tally) >= 2:" in block

    def test_guard_is_log_only_by_default_gated_on_the_shared_flag(self):
        block = _topic_slot_merge_loop_source()
        assert "REGULATORY_IDENTITY_GUARD_ENFORCE" in block
        assert "would drop" in block

    def test_guard_never_drops_every_doc(self):
        block = _topic_slot_merge_loop_source()
        assert "if _mt2_kept and getattr(conf, \"REGULATORY_IDENTITY_GUARD_ENFORCE\", False)" in block

    def test_guard_is_exception_safe(self):
        block = _topic_slot_merge_loop_source()
        assert "except Exception as _mt2_rid_e" in block

    def test_guard_call_has_no_license_type_argument(self):
        block = _topic_slot_merge_loop_source()
        m = re.search(
            r"_shared_regulatory_identity_mismatch_should_reject\((.*?)\)",
            block, re.DOTALL,
        )
        assert m
        assert "license_type" not in m.group(1)


@pytest.mark.unit
class TestTopicSlotMultiTopicLoopCoverageSupplement:
    """Phase 3 applied to the same third call site."""

    def test_supplement_gated_on_its_own_dedicated_flag(self):
        block = _topic_slot_merge_loop_source()
        assert "MULTI_TOPIC_COVERAGE_SUPPLEMENT_ENABLED" in block

    def test_supplement_checks_adequacy_before_searching(self):
        block = _topic_slot_merge_loop_source()
        assert "PersonaSupervisor._is_structurally_adequate(_lt_docs2)" in block

    def test_supplement_appends_rather_than_replaces(self):
        block = _topic_slot_merge_loop_source()
        assert "_lt_docs2 = _lt_docs2 + _mt2_supplement" in block

    def test_supplement_is_exception_safe(self):
        block = _topic_slot_merge_loop_source()
        assert "except Exception as _mt2_supp_e" in block

    def test_sort_uses_the_shared_helper_not_a_local_copy(self):
        block = _topic_slot_merge_loop_source()
        assert "PersonaSupervisor._structural_field_count" in block
        assert "_STRUCTURAL_KEYS2" not in block

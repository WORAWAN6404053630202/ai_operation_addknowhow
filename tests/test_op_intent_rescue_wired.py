"""
Source-inspection regression pins confirming utils.text_patterns.
rescue_top_sim_doc (2026-11, hardcoded-logic audit) is actually wired in at
all 4 operation-intent/entity doc-filter sites found in the audit, right
before each site's existing "apply the filter" gate — not just added to
text_patterns.py and never used. Same established pattern as
test_group_a_single_filter_sites_safety.py for logic embedded too deeply
inside large stateful methods to drive behaviorally end-to-end.
"""
from __future__ import annotations

import inspect

import pytest

from model.persona_academic import AcademicPersonaService
from model.persona_practical import PracticalPersonaService


@pytest.mark.unit
class TestPracticalOpIntentRescueWired:
    def test_rescue_call_precedes_the_op_pass_docs_gate(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        rescue_idx = source.find("_op_pass_docs = _shared_rescue_top_sim_doc(")
        gate_idx = source.find("if _op_pass_docs:")
        assert rescue_idx != -1, "rescue_top_sim_doc call not found in handle()"
        assert gate_idx != -1
        assert rescue_idx < gate_idx

    def test_rescue_is_gated_by_the_conf_flag(self):
        source = inspect.getsource(PracticalPersonaService.handle)
        assert 'getattr(conf, "OP_INTENT_TOP_SIM_RESCUE_ENFORCE", False)' in source


@pytest.mark.unit
class TestAcademicOpIntentRescueWired:
    def test_entity_filter_rescue_precedes_its_gate(self):
        source = inspect.getsource(AcademicPersonaService._filter_docs_by_known_entity)
        rescue_idx = source.find("_shared_rescue_top_sim_doc(")
        gate_idx = source.find("if len(filtered) >= 2:")
        assert rescue_idx != -1
        assert gate_idx != -1
        assert rescue_idx < gate_idx

    def test_known_operation_filter_rescue_precedes_its_gate(self):
        source = inspect.getsource(AcademicPersonaService._filter_docs_by_known_operation)
        rescue_idx = source.find("_shared_rescue_top_sim_doc(")
        gate_idx = source.find("if len(filtered) >= 1:")
        assert rescue_idx != -1
        assert gate_idx != -1
        assert rescue_idx < gate_idx

    def test_inferred_operation_filter_rescue_precedes_its_gate(self):
        source = inspect.getsource(AcademicPersonaService._filter_docs_by_inferred_operation)
        rescue_idx = source.find("_shared_rescue_top_sim_doc(")
        gate_idx = source.find("if filtered:")
        assert rescue_idx != -1
        assert gate_idx != -1
        assert rescue_idx < gate_idx

    def test_all_three_sites_gated_by_the_conf_flag(self):
        import model.persona_academic as academic_mod
        source = inspect.getsource(academic_mod)
        assert source.count('getattr(conf, "OP_INTENT_TOP_SIM_RESCUE_ENFORCE", False)') == 3


@pytest.mark.unit
class TestRareLtObservabilityWired:
    """persona_supervisor.py's _RARE_LT_PATTERNS deliberately does NOT get
    the same rescue treatment (it's a counter-measure against a documented
    embedding-ranking weakness, not a case where trusting _sim more is
    safe) — only observability logging when the regex list never matched
    but a rare license_type surfaced via normal retrieval anyway."""

    def test_match_flag_is_tracked_and_checked_before_the_identity_guard_block(self):
        from model.persona_supervisor import PersonaSupervisor
        source = inspect.getsource(PersonaSupervisor._ensure_practical_retrieval_for_legal)
        flag_set_idx = source.find("_rare_lt_pattern_matched = True")
        check_idx = source.find("if not _rare_lt_pattern_matched and docs:")
        assert flag_set_idx != -1
        assert check_idx != -1
        assert flag_set_idx < check_idx

    def test_observability_check_never_mutates_docs(self):
        """Zero behavior change: the check block must only read `docs`, log,
        and catch its own exceptions — never reassign docs/state.current_docs."""
        from model.persona_supervisor import PersonaSupervisor
        source = inspect.getsource(PersonaSupervisor._ensure_practical_retrieval_for_legal)
        start = source.find("if not _rare_lt_pattern_matched and docs:")
        assert start != -1
        end = source.find("# 2026-10 fix (Phase-0 codebase audit, QA-reproduced cross-domain", start)
        assert end != -1
        block = source[start:end]
        assert "docs =" not in block
        assert "state.current_docs" not in block

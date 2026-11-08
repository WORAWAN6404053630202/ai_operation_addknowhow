"""
Regression pins for the Tier 3 (hygiene/consistency) and Tier 4 (defensive/
observability) fixes from the 2026-11 hardcoded-logic audit. Each assertion
targets exactly what changed — most of these are text/constant-extraction
fixes with no new behavioral surface, so source-presence/value checks are
the right level of test, matching this codebase's own established pattern
for changes this shape (see tests/test_es_backend_awareness_fixes.py).
"""
from __future__ import annotations

import inspect

import pytest

import conf


@pytest.mark.unit
class TestRerankerModelDefaultConsistency:
    def test_conf_default_matches_reranker_module_default(self):
        """conf.RERANKER_MODEL's os.getenv default and reranker.rerank()'s
        own fallback default used to describe two DIFFERENT models — fixed
        to agree. Checked via source text, not the live conf.RERANKER_MODEL
        value, since a local .env can override it (confirmed in this dev
        environment) independent of what the code's own default says."""
        from utils.reranker import rerank
        reranker_default = inspect.signature(rerank).parameters["model_name"].default
        assert reranker_default == "BAAI/bge-reranker-v2-m3"
        conf_source = inspect.getsource(conf)
        assert 'RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")' in conf_source

    def test_no_remaining_stale_mmarco_fallback_defaults(self):
        """The same stale fallback string used to be hand-copied into
        app.py and all 3 persona files' getattr(conf, "RERANKER_MODEL", ...)
        calls — confirm none of them still reference the old model."""
        import app
        import model.persona_academic as academic_mod
        import model.persona_practical as practical_mod
        import model.persona_supervisor as supervisor_mod

        for mod in (app, academic_mod, practical_mod, supervisor_mod):
            source = inspect.getsource(mod)
            assert "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" not in source, (
                f"{mod.__name__} still has the stale reranker fallback default"
            )


@pytest.mark.unit
class TestNewConfConstants:
    def test_op_intent_rescue_enforce_defaults_false(self):
        assert conf.OP_INTENT_TOP_SIM_RESCUE_ENFORCE is False

    def test_topic_switch_overlap_min_ratio_defaults_to_022(self):
        assert conf.TOPIC_SWITCH_OVERLAP_MIN_RATIO == 0.22

    def test_multi_license_merge_caps_default_to_original_values(self):
        assert conf.MULTI_LICENSE_MERGE_PER_LICENSE_CAP == 3
        assert conf.MULTI_LICENSE_MERGE_TOTAL_CAP == 8

    def test_rare_lt_patterns_enabled_defaults_false_per_live_ab_result(self):
        """2026-11 Phase 2.5 live A/B (real /chat queries, reproduced twice each):
        สุรา showed no difference on vs off; ภาษีป้าย was WORSE with it ON
        (info-gap vs a correct detailed fee answer with it off). Default flipped
        from true to false based on this evidence — see conf.py's own comment."""
        assert conf.RARE_LT_PATTERNS_ENABLED is False


@pytest.mark.unit
class TestBankRegexDeduplication:
    def test_bank_id_patterns_is_a_single_shared_class_attribute(self):
        from model.persona_supervisor import PersonaSupervisor
        assert hasattr(PersonaSupervisor, "_BANK_ID_PATTERNS")
        assert len(PersonaSupervisor._BANK_ID_PATTERNS) == 8

    def test_no_remaining_duplicate_inline_bank_list(self):
        """The 8-entry list used to be hand-copied verbatim at 2 call
        sites — confirm only the one class-attribute definition remains."""
        from model.persona_supervisor import PersonaSupervisor
        source = inspect.getsource(PersonaSupervisor)
        assert source.count("_BANK_ID_PATTERNS: List[Tuple] = [") == 1
        assert source.count('("bbl",   re.compile(r"ธนาคารกรุงเทพ|bbl\\b", re.IGNORECASE)),') == 1


@pytest.mark.unit
class TestPermitLtRePromotedAndRevalidated:
    def test_permit_lt_re_is_a_class_attribute(self):
        from model.persona_supervisor import PersonaSupervisor
        assert hasattr(PersonaSupervisor, "_PERMIT_LT_RE")
        assert PersonaSupervisor._PERMIT_LT_RE.search("ใบอนุญาตทดสอบ")

    def test_startup_validator_rechecks_permit_lt_re(self):
        from model.persona_supervisor import PersonaSupervisor
        source = inspect.getsource(PersonaSupervisor._validate_license_types_at_startup)
        assert "_PERMIT_LT_RE" in source
        assert "_PERMIT_LOOKS_LIKE_RE" in source


@pytest.mark.unit
class TestInfoGapPromptScopeClarified:
    def test_never_say_rule_carves_out_info_gap_and_rule16(self):
        from utils.prompts_practical import SYSTEM_PROMPT
        idx = SYSTEM_PROMPT.find('NEVER say "ไม่พบในเอกสาร"')
        assert idx != -1
        following = SYSTEM_PROMPT[idx:idx + 1200]
        assert "[[INFO_GAP]]" in following
        assert "ไม่มีข้อมูลนี้ในคู่มือ" in following

    def test_multi_entity_exception_mirrors_department_exception_b_clause(self):
        from utils.prompts_practical import SYSTEM_PROMPT
        idx = SYSTEM_PROMPT.find("EXCEPTION — genuine multi-entity comparison")
        assert idx != -1
        assert "already-established one" in SYSTEM_PROMPT[idx:idx + 600]

    def test_fee_shape_default_has_a_heuristic_caveat(self):
        from utils.prompts_practical import SYSTEM_PROMPT
        idx = SYSTEM_PROMPT.find("default to (a)")
        assert idx != -1
        assert "heuristic" in SYSTEM_PROMPT[idx:idx + 500]

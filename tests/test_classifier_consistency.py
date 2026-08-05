"""
Regression + consistency tests for the shared text-classification patterns
(utils/text_patterns.py) used across PersonaSupervisor, PracticalPersonaService,
and AcademicPersonaService.

Background: a 2026-07-31 audit found that these three classes had each grown
their own, independently-maintained regex classifiers for the same concepts
(is this a legal question? is this a greeting? does this name a specific
license?) — several had silently diverged (same class-attribute name, different
matched text), which was the direct root cause of a live topic-drift bug (see
persona_practical_retrieve_retry_loop_bug.md in project memory). The patterns
were consolidated into utils/text_patterns.py so each concept has exactly one
definition. No LLM/Chroma connection required — everything here is a class
attribute, accessible without instantiating (these personas require a live
retriever to construct).

Two kinds of tests:
1. Regression cases pinned from the actual bugs found on 2026-07-31.
2. A generic cross-class consistency check: for every classifier name that
   exists on 2+ of the three persona classes, assert the pattern text and
   flags are identical. This single test would have caught 100% of the
   divergence found in the audit, and keeps catching it for any pattern not
   explicitly covered by a regression case above.
"""
import re

import pytest

from model.persona_supervisor import PersonaSupervisor
from model.persona_practical import PracticalPersonaService
from model.persona_academic import AcademicPersonaService
from utils.text_patterns import is_bare_generic_followup

_PERSONA_CLASSES = {
    "PersonaSupervisor": PersonaSupervisor,
    "PracticalPersonaService": PracticalPersonaService,
    "AcademicPersonaService": AcademicPersonaService,
}


def _compiled_patterns(cls) -> dict:
    """All class-attribute compiled regexes, keyed by attribute name."""
    return {
        name: value
        for name, value in vars(cls).items()
        if isinstance(value, re.Pattern)
    }


@pytest.mark.unit
class TestCrossClassPatternConsistency:
    """
    Generic drift detector: any pattern name shared by 2+ persona classes
    must have identical `.pattern` text and `.flags`. This is the test that
    would have caught the 2026-07-31 divergence automatically.
    """

    def test_shared_pattern_names_are_identical_everywhere(self):
        by_name: dict = {}
        for cls_label, cls in _PERSONA_CLASSES.items():
            for name, pattern in _compiled_patterns(cls).items():
                by_name.setdefault(name, []).append((cls_label, pattern))

        mismatches = []
        for name, entries in by_name.items():
            if len(entries) < 2:
                continue
            first_label, first_pattern = entries[0]
            for label, pattern in entries[1:]:
                if pattern.pattern != first_pattern.pattern or pattern.flags != first_pattern.flags:
                    mismatches.append(
                        f"{name}: {first_label}={first_pattern.pattern!r} (flags={first_pattern.flags}) "
                        f"!= {label}={pattern.pattern!r} (flags={pattern.flags})"
                    )

        assert not mismatches, (
            "Found diverged same-name classifiers across persona classes — each of these "
            "should be a single shared definition in utils/text_patterns.py:\n"
            + "\n".join(mismatches)
        )

    def test_at_least_the_known_shared_patterns_exist_on_all_three(self):
        # Sanity check that the consolidation actually happened (guards against a
        # future refactor silently un-sharing one of these by re-adding a local copy).
        expected_on_all_three = {"_LEGAL_SIGNAL_RE", "_TH_WATDEE_RE", "_ENTITY_NITI_RE", "_ENTITY_NATURAL_RE"}
        for name in expected_on_all_three:
            for cls_label, cls in _PERSONA_CLASSES.items():
                assert hasattr(cls, name), f"{cls_label} is missing shared pattern {name}"


@pytest.mark.unit
class TestLegalSignalRegressionCases:
    """
    _LEGAL_SIGNAL_RE previously had 3 different term lists. Before the fix,
    practical's and academic's narrower copies would have missed these.
    """

    @pytest.mark.parametrize("cls_label,cls", list(_PERSONA_CLASSES.items()))
    def test_qr_payment_bank_terms_recognized_everywhere(self, cls_label, cls):
        # Previously only PersonaSupervisor's copy included these terms.
        assert cls._LEGAL_SIGNAL_RE.search("อยากรู้เรื่อง QR Payment กับธนาคารกสิกรไทยครับ")

    @pytest.mark.parametrize("cls_label,cls", list(_PERSONA_CLASSES.items()))
    def test_financial_risk_terms_recognized_everywhere(self, cls_label, cls):
        # Previously only practical's/academic's copies included these terms —
        # supervisor's copy has them too further down its (longer) pattern.
        assert cls._LEGAL_SIGNAL_RE.search("ผลกระทบต่องบการเงินมีอะไรบ้าง")


@pytest.mark.unit
class TestBareGenericDimensionFollowup:
    """
    Root cause of the 2026-07-31 liquor-license topic-drift bug: a bare
    generic-dimension follow-up ("ต้องใช้เอกสารอะไรบ้าง") with no แค่/แล้ว
    prefix fell through persona_practical.py's short-followup detection,
    got misclassified as a brand-new topic, and triggered a broad multi-topic
    retrieval that returned zero docs about the actual (previous) topic.

    Phase 3 (2026-07-31): the fix (persona_practical.py's now-removed
    `_BARE_GENERIC_DIM_RE`/`_SPECIFIC_TOPIC_NAME_RE` pair) was generalized into
    the shared `is_bare_generic_followup()` helper so persona_academic.py,
    which had the identical structural vulnerability and no protection at all,
    gets the same fix. These tests now target the shared helper directly.
    """

    def test_bare_document_question_is_generic_dimension(self):
        assert is_bare_generic_followup("ต้องใช้เอกสารอะไรบ้าง")

    def test_bare_fee_question_is_generic_dimension(self):
        assert is_bare_generic_followup("มีค่าธรรมเนียมเท่าไหร่")

    def test_question_naming_a_different_specific_topic_is_excluded(self):
        # Regression guard for the overcorrection risk: a generic-dimension
        # question that ALSO names a different specific topic must NOT be
        # treated as a bare generic follow-up — it should fall through to
        # each persona's own overlap-ratio check so the real topic switch
        # is caught.
        text = "ค่าธรรมเนียมทะเบียนพาณิชย์เท่าไหร่"
        assert not is_bare_generic_followup(text)

    def test_plain_document_question_names_no_specific_topic(self):
        assert is_bare_generic_followup("ต้องใช้เอกสารอะไรบ้าง")

    def test_bare_followup_helper_used_consistently_by_practical_and_academic(self):
        # persona_practical.py's _is_short_followup and persona_academic.py's
        # _query_changed both call the shared helper (verified by source
        # reference below rather than behavior, since both are private methods
        # requiring a live retriever to instantiate).
        import inspect

        practical_src = inspect.getsource(PracticalPersonaService._is_short_followup)
        academic_src = inspect.getsource(AcademicPersonaService._start_intake_with_retrieval)
        assert "_shared_is_bare_generic_followup" in practical_src
        assert "_shared_is_bare_generic_followup" in academic_src


@pytest.mark.unit
class TestLikelySelectionRegression:
    """
    persona_practical.py's copy of _LIKELY_SELECTION_RE was missing the
    "at least one digit" lookahead that persona_supervisor.py's copy had,
    so a separator-only string would incorrectly count as a menu selection.
    """

    def test_digit_selection_matches(self):
        assert PracticalPersonaService._LIKELY_SELECTION_RE.match("1, 2")
        assert PersonaSupervisor._LIKELY_SELECTION_RE.match("1, 2")

    def test_separator_only_string_does_not_match(self):
        # AcademicPersonaService doesn't define this classifier at all — only
        # supervisor and practical do.
        for cls in (PracticalPersonaService, PersonaSupervisor):
            assert not cls._LIKELY_SELECTION_RE.match("  -  , /")


@pytest.mark.unit
class TestWatdeeGreetingRegression:
    """
    persona_supervisor.py's/persona_academic.py's copies of _TH_WATDEE_RE
    required an exact "หวัดดี" match; persona_practical.py's copy was
    typo/variant-tolerant. Reconciled to the tolerant version everywhere.
    """

    @pytest.mark.parametrize("cls_label,cls", list(_PERSONA_CLASSES.items()))
    def test_casual_watdee_variant_recognized_everywhere(self, cls_label, cls):
        # "หวัดจ้า" is a real casual greeting shortening that the old
        # exact-match-only pattern (requires literal "หวัดดี") would miss.
        assert cls._TH_WATDEE_RE.search("หวัดจ้า")


@pytest.mark.unit
class TestEntityTypeColloquialRegression:
    """
    Phase 3 (2026-07-31, first pass): persona_academic.py's entity-type
    contradiction guard previously used method-local
    `_NATURAL_ENTITY_RE`/`_JURISTIC_ENTITY_RE` patterns that only matched the
    literal formal terms. Reconciled to persona_supervisor.py's richer
    canonical coverage, shared via utils/text_patterns.py.

    Regex-tightening pass (2026-07-31, second pass): the richer coverage
    from the first pass turned out to be too loose — several of its
    colloquial branches (bare "คนเดียว", "ทำเอง", "เปิดเอง", bare "บริษัท"/
    "ห้าง" without their legal suffix) live-reproduced as false positives on
    ordinary Thai sentences unrelated to declaring a business entity type
    (e.g. "ค่าธรรมเนียมทำเองกับจ้างตัวแทน...ต่างกันไหม" — a DIY-vs-agent fee
    question — silently set entity_type='บุคคลธรรมดา' with zero
    confirmation, permanent for the rest of the session per this project's
    "no slot re-ask" rule). Those terms were removed from the canonical
    pattern (see utils/text_patterns.py's ENTITY_NITI_RE/ENTITY_NATURAL_RE
    comment for the full list and reasoning). "ทำคนเดียวค่ะ" is deliberately
    no longer expected to match — this reverses the first pass's regression
    case above.
    """

    @pytest.mark.parametrize("cls_label,cls", list(_PERSONA_CLASSES.items()))
    def test_deliberate_natural_entity_phrasing_recognized_everywhere(self, cls_label, cls):
        # "เจ้าของคนเดียว" (sole owner) is a specific, deliberate declaration —
        # unlike bare "คนเดียว", the "เจ้าของ" prefix makes it unambiguous.
        assert cls._ENTITY_NATURAL_RE.search("เจ้าของคนเดียวครับ")

    @pytest.mark.parametrize("cls_label,cls", list(_PERSONA_CLASSES.items()))
    def test_formal_juristic_entity_phrasing_recognized_everywhere(self, cls_label, cls):
        assert cls._ENTITY_NITI_RE.search("เป็นนิติบุคคลค่ะ")

    @pytest.mark.parametrize("cls_label,cls", list(_PERSONA_CLASSES.items()))
    def test_ambiguous_diy_phrasing_no_longer_matches(self, cls_label, cls):
        # Live-reproduced false positive (2026-07-31): a DIY-filing-fee
        # question, not an entity-type declaration.
        text = "ค่าธรรมเนียมทำเองกับจ้างตัวแทนยื่นต่างกันไหมครับ"
        assert not cls._ENTITY_NATURAL_RE.search(text)

    @pytest.mark.parametrize("cls_label,cls", list(_PERSONA_CLASSES.items()))
    def test_bare_konnyaw_no_longer_matches(self, cls_label, cls):
        # "อยู่คนเดียว" (live alone) / "มาคนเดียว" (came alone) have nothing to
        # do with sole-proprietorship.
        assert not cls._ENTITY_NATURAL_RE.search("วันนี้มาคนเดียวครับ")

    @pytest.mark.parametrize("cls_label,cls", list(_PERSONA_CLASSES.items()))
    def test_bare_haang_no_longer_matches_as_juristic(self, cls_label, cls):
        # Live-reproduced false positive: a shop merely *located in a mall*
        # ("ห้าง" = shopping mall in everyday Thai) matched as "นิติบุคคล"
        # before the fix. "หุ้นส่วน" is now a required suffix.
        assert not cls._ENTITY_NITI_RE.search("ร้านอยู่ในห้างครับ")

    @pytest.mark.parametrize("cls_label,cls", list(_PERSONA_CLASSES.items()))
    def test_bare_borisat_no_longer_matches_as_juristic(self, cls_label, cls):
        # "บริษัท" alone is the everyday word for "a company" in any context
        # (e.g. a former employer) -- "จำกัด"/"มหาชน" is now required.
        assert not cls._ENTITY_NITI_RE.search("ผมเคยทำงานบริษัทเอกชนมาก่อนครับ")


@pytest.mark.unit
class TestAcademicQueryChangedBareFollowupOverride:
    """
    Phase 3 (2026-07-31): persona_academic.py's `_query_changed` (used inside
    `_start_intake_with_retrieval` to decide whether to re-retrieve) previously
    had NO keyword short-circuit — pure word-overlap ratio only (threshold
    0.30). A bare generic-dimension follow-up like "ต้องใช้เอกสารอะไรบ้าง" has
    near-zero overlap with any prior topic by definition, so Academic mode was
    structurally exposed to the identical topic-drift bug already fixed in
    Practical mode, just never live-triggered. This confirms the shared
    `is_bare_generic_followup` early-override is actually wired into the
    source (behavioral live verification is covered separately since
    `_query_changed` is a closure requiring a live retriever to instantiate
    the enclosing service).
    """

    def test_early_override_wired_before_overlap_ratio_calc(self):
        import inspect

        src = inspect.getsource(AcademicPersonaService._start_intake_with_retrieval)
        override_idx = src.index("_shared_is_bare_generic_followup(new_q)")
        overlap_idx = src.index("old_words & new_words")
        assert override_idx < overlap_idx, (
            "is_bare_generic_followup override must run before the overlap-ratio "
            "calculation so it can short-circuit it"
        )

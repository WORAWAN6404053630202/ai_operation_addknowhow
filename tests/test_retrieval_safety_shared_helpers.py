"""
Pure-logic tests for the 3 shared retrieval-safety helpers added to
PersonaSupervisor (2026-11, retrieval-safety audit — Phase 0 of migrating 13
newly-found risky department/license_type filter sites onto shared helpers
instead of continuing to hand-write the same guard/supplement logic per site).

Background: these generalize two mechanisms already proven at 4 sites in this
file (see tests/test_license_filter_coverage_cascade.py): a majority-vote
identity guard (no per-doc score available, unlike the rare-license site's
scored variant, which is NOT migrated in this round) and a coverage-adequacy
supplement that appends an unfiltered search result when the filtered result
is structurally thin.

No mocking needed — all three helpers are pure/static, operating on plain
dicts shaped like {"content": str, "metadata": dict}, same convention as
TestStructuralAdequacyHelper in test_license_filter_coverage_cascade.py.
"""
from __future__ import annotations

import pytest

from model.persona_supervisor import PersonaSupervisor


def _doc(content="x", **metadata) -> dict:
    return {"content": content, "metadata": metadata}


@pytest.mark.unit
class TestSlimDocHelpers:
    def test_reads_field_from_dict_shaped_doc(self):
        assert PersonaSupervisor._slim_doc_field(_doc(department="กรมสวัสดิการ"), "department") == "กรมสวัสดิการ"

    def test_missing_field_returns_empty_string(self):
        assert PersonaSupervisor._slim_doc_field(_doc(), "department") == ""

    def test_reads_content_from_dict_shaped_doc(self):
        assert PersonaSupervisor._slim_doc_content(_doc(content="hello")) == "hello"

    def test_reads_field_from_document_like_object(self):
        class _FakeDoc:
            def __init__(self):
                self.metadata = {"department": "กรมสวัสดิการ"}
                self.page_content = "hello"
        fd = _FakeDoc()
        assert PersonaSupervisor._slim_doc_field(fd, "department") == "กรมสวัสดิการ"
        assert PersonaSupervisor._slim_doc_content(fd) == "hello"


@pytest.mark.unit
class TestIdentityGuardMajorityVote:
    def test_fewer_than_two_docs_never_fires(self):
        docs = [_doc(department="A")]
        kept, would_drop = PersonaSupervisor._identity_guard_majority_vote(
            docs, identity_fields=["department"], enforce=True,
        )
        assert kept == docs
        assert would_drop is False

    def test_all_agree_never_drops(self):
        docs = [_doc(department="A"), _doc(department="A"), _doc(department="A")]
        kept, would_drop = PersonaSupervisor._identity_guard_majority_vote(
            docs, identity_fields=["department"], enforce=True,
        )
        assert kept == docs
        assert would_drop is False

    def test_minority_value_flagged_as_would_drop(self):
        docs = [_doc(department="A"), _doc(department="A"), _doc(department="B")]
        kept, would_drop = PersonaSupervisor._identity_guard_majority_vote(
            docs, identity_fields=["department"], enforce=False,
        )
        assert would_drop is True
        # enforce=False → log-only, nothing actually dropped
        assert kept == docs

    def test_enforce_true_actually_drops_the_minority(self):
        docs = [_doc(department="A"), _doc(department="A"), _doc(department="B")]
        kept, would_drop = PersonaSupervisor._identity_guard_majority_vote(
            docs, identity_fields=["department"], enforce=True,
        )
        assert would_drop is True
        assert len(kept) == 2
        assert all(PersonaSupervisor._slim_doc_field(d, "department") == "A" for d in kept)

    def test_blank_values_never_enter_the_tally(self):
        docs = [_doc(department="A"), _doc(department="A"), _doc(department="")]
        kept, would_drop = PersonaSupervisor._identity_guard_majority_vote(
            docs, identity_fields=["department"], enforce=True,
        )
        # only one distinct non-blank value ("A") → nothing to disagree about
        assert would_drop is False
        assert kept == docs

    def test_never_drops_every_doc(self):
        """A 1-1 tie still has a 'majority' (first-seen via max()), but dropping
        the other side would leave an empty kept-set for that comparison — the
        guard must not do that."""
        docs = [_doc(department="A"), _doc(department="B")]
        kept, would_drop = PersonaSupervisor._identity_guard_majority_vote(
            docs, identity_fields=["department"], enforce=True,
        )
        assert len(kept) >= 1

    def test_checks_multiple_fields_with_or_semantics(self):
        """A doc disagreeing on EITHER field should be droppable — not just the
        first field checked."""
        docs = [
            _doc(department="A", license_type="X"),
            _doc(department="A", license_type="X"),
            _doc(department="A", license_type="Y"),
        ]
        kept, would_drop = PersonaSupervisor._identity_guard_majority_vote(
            docs, identity_fields=["department", "license_type"], enforce=True,
        )
        assert would_drop is True
        assert len(kept) == 2

    def test_exception_safety_returns_original_docs_unchanged(self):
        docs = [_doc(department="A"), _doc(department="B"), _doc(department="A")]
        # Passing a non-list-of-dicts-or-Document (e.g. plain strings) makes
        # _slim_doc_field's getattr/isinstance checks resolve to "" for every
        # entry rather than raising — so exercise the guard's own
        # exception-safety via a genuinely broken identity_fields type instead.
        kept, would_drop = PersonaSupervisor._identity_guard_majority_vote(
            docs, identity_fields=None, enforce=True,  # type: ignore[arg-type]
        )
        assert kept == docs
        assert would_drop is False

    def test_log_label_is_accepted_without_error(self):
        docs = [_doc(department="A"), _doc(department="A"), _doc(department="B")]
        kept, would_drop = PersonaSupervisor._identity_guard_majority_vote(
            docs, identity_fields=["department"], enforce=False, log_label="test-site",
        )
        assert would_drop is True


@pytest.mark.unit
class TestSupplementIfThin:
    def test_disabled_never_calls_supplement_fn(self):
        called = []
        def _supp():
            called.append(True)
            return [_doc(content="extra", operation_steps="x", fees="y")]
        docs = [_doc(content="thin")]
        result = PersonaSupervisor._supplement_if_thin(docs, enabled=False, supplement_fn=_supp)
        assert result == docs
        assert called == []

    def test_already_adequate_never_calls_supplement_fn(self):
        called = []
        def _supp():
            called.append(True)
            return []
        docs = [_doc(content="rich", operation_steps="a", fees="b")]
        result = PersonaSupervisor._supplement_if_thin(docs, enabled=True, supplement_fn=_supp)
        assert result == docs
        assert called == []

    def test_thin_and_enabled_appends_supplement(self):
        docs = [_doc(content="thin")]
        supplement = [_doc(content="extra", operation_steps="a", fees="b")]
        result = PersonaSupervisor._supplement_if_thin(
            docs, enabled=True, supplement_fn=lambda: supplement,
        )
        assert len(result) == 2
        assert result[0] == docs[0]
        assert result[1] == supplement[0]

    def test_dedupes_against_existing_docs_by_content_fingerprint(self):
        docs = [_doc(content="thin")]
        supplement = [_doc(content="thin"), _doc(content="new", operation_steps="a", fees="b")]
        result = PersonaSupervisor._supplement_if_thin(
            docs, enabled=True, supplement_fn=lambda: supplement,
        )
        assert len(result) == 2  # "thin" deduped out, "new" kept

    def test_empty_supplement_returns_docs_unchanged(self):
        docs = [_doc(content="thin")]
        result = PersonaSupervisor._supplement_if_thin(docs, enabled=True, supplement_fn=lambda: [])
        assert result == docs

    def test_exception_in_supplement_fn_returns_docs_unchanged(self):
        docs = [_doc(content="thin")]
        def _boom():
            raise RuntimeError("boom")
        result = PersonaSupervisor._supplement_if_thin(docs, enabled=True, supplement_fn=_boom)
        assert result == docs


@pytest.mark.unit
class TestMergeValueGroupsWithGuardAndSupplement:
    def test_merges_per_value_fetch_results(self):
        fetch_map = {
            "lt1": [_doc(content="a", department="A")],
            "lt2": [_doc(content="b", department="B")],
        }
        result = PersonaSupervisor._merge_value_groups_with_guard_and_supplement(
            ["lt1", "lt2"],
            fetch_fn=lambda v: fetch_map[v],
            identity_fields=["department"],
            guard_enforce=True,
            supplement_enabled=False,
        )
        assert len(result) == 2

    def test_guard_never_compares_across_different_values(self):
        """lt1's docs all say department=A, lt2's docs all say department=B —
        if the guard compared across groups it would reject one whole group;
        it must not, since merging different license_types on purpose is the
        entire point of this helper."""
        fetch_map = {
            "lt1": [_doc(content="a1", department="A"), _doc(content="a2", department="A")],
            "lt2": [_doc(content="b1", department="B"), _doc(content="b2", department="B")],
        }
        result = PersonaSupervisor._merge_value_groups_with_guard_and_supplement(
            ["lt1", "lt2"],
            fetch_fn=lambda v: fetch_map[v],
            identity_fields=["department"],
            guard_enforce=True,
            supplement_enabled=False,
        )
        assert len(result) == 4

    def test_within_group_disagreement_still_caught(self):
        fetch_map = {
            "lt1": [_doc(content="a1", department="A"), _doc(content="a2", department="A"), _doc(content="wrong", department="NTT DATA")],
        }
        result = PersonaSupervisor._merge_value_groups_with_guard_and_supplement(
            ["lt1"],
            fetch_fn=lambda v: fetch_map[v],
            identity_fields=["department"],
            guard_enforce=True,
            supplement_enabled=False,
        )
        assert len(result) == 2
        assert all(PersonaSupervisor._slim_doc_field(d, "department") == "A" for d in result)

    def test_post_filter_runs_after_the_guard(self):
        calls = []
        def _post(value, docs):
            calls.append((value, len(docs)))
            return docs
        fetch_map = {"lt1": [_doc(content="a", department="A"), _doc(content="b", department="A")]}
        PersonaSupervisor._merge_value_groups_with_guard_and_supplement(
            ["lt1"],
            fetch_fn=lambda v: fetch_map[v],
            identity_fields=["department"],
            guard_enforce=True,
            supplement_enabled=False,
            post_filter_fn=_post,
        )
        assert calls == [("lt1", 2)]

    def test_supplement_fn_receives_the_value_and_current_group_docs(self):
        calls = []
        def _supp(value, group_docs):
            calls.append((value, len(group_docs)))
            return []
        fetch_map = {"lt1": [_doc(content="thin")]}
        PersonaSupervisor._merge_value_groups_with_guard_and_supplement(
            ["lt1"],
            fetch_fn=lambda v: fetch_map[v],
            identity_fields=["department"],
            guard_enforce=True,
            supplement_enabled=True,
            supplement_fn=_supp,
        )
        assert calls == [("lt1", 1)]

    def test_per_group_cap_applied(self):
        fetch_map = {"lt1": [_doc(content=str(i)) for i in range(5)]}
        result = PersonaSupervisor._merge_value_groups_with_guard_and_supplement(
            ["lt1"],
            fetch_fn=lambda v: fetch_map[v],
            identity_fields=["department"],
            guard_enforce=True,
            supplement_enabled=False,
            per_group_cap=2,
        )
        assert len(result) == 2

    def test_total_cap_stops_further_fetching(self):
        fetched = []
        def _fetch(v):
            fetched.append(v)
            return [_doc(content=v)]
        PersonaSupervisor._merge_value_groups_with_guard_and_supplement(
            ["lt1", "lt2", "lt3"],
            fetch_fn=_fetch,
            identity_fields=["department"],
            guard_enforce=True,
            supplement_enabled=False,
            total_cap=2,
        )
        assert fetched == ["lt1", "lt2"]

    def test_empty_values_list_returns_empty(self):
        result = PersonaSupervisor._merge_value_groups_with_guard_and_supplement(
            [],
            fetch_fn=lambda v: [_doc()],
            identity_fields=["department"],
            guard_enforce=True,
            supplement_enabled=False,
        )
        assert result == []

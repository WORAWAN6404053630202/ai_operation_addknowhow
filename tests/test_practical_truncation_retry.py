"""
Unit tests for PracticalPersonaService._call_llm_json_with_truncation_retry
(persona_practical.py) and the related _BOOST_FIELDS department addition —
both from the 2026-09 QA-review fix pass (see project memory for the full
root-cause list).

Background: a QA review found real cases where a long multi-item answer (an
8-item document checklist, a 23-subtopic know-how chapter) was silently cut
short by MAX_TOKENS_PRACTICAL, with no indication to the user. The fix wraps
the existing _call_llm_json with exactly one extra attempt at a higher token
ceiling, ONLY when the first attempt shows a truncation marker
("json_repair" from the regex rescue path, or "Parse error" from the final
fallback) — normal complete answers must never pay for the extra call.

PracticalPersonaService can be constructed with retriever=MagicMock() and no
network access — ChatOpenAI's constructor doesn't validate credentials or
call out. _call_llm_json itself is monkeypatched here since it's the network
boundary; these tests verify the retry orchestration logic around it.
"""
import re
from unittest.mock import MagicMock

import pytest

from model.persona_practical import PracticalPersonaService


@pytest.fixture
def practical():
    return PracticalPersonaService(retriever=MagicMock())


@pytest.mark.unit
class TestTruncationRetryOrchestration:

    def test_non_truncated_first_attempt_returns_immediately_no_retry(self, practical):
        complete_result = {"analysis": "ok", "action": "answer", "answer": "full answer"}
        practical._call_llm_json = MagicMock(return_value=complete_result)

        out = practical._call_llm_json_with_truncation_retry("prompt text")

        assert out == complete_result
        practical._call_llm_json.assert_called_once()

    def test_truncated_first_attempt_retries_with_expanded_llm_override(self, practical):
        truncated = {"analysis": "json_repair", "answer": "partial answer cut off mid"}
        complete = {"analysis": "ok", "action": "answer", "answer": "full complete answer"}
        practical._call_llm_json = MagicMock(side_effect=[truncated, complete])

        out = practical._call_llm_json_with_truncation_retry("prompt text")

        assert out == complete
        assert practical._call_llm_json.call_count == 2
        # First call: no llm_override. Second call: an override was passed and
        # is not the plain self.llm (must be a distinct, higher-max_tokens bind()).
        first_kwargs = practical._call_llm_json.call_args_list[0].kwargs
        second_kwargs = practical._call_llm_json.call_args_list[1].kwargs
        assert "llm_override" not in first_kwargs or first_kwargs.get("llm_override") is None
        assert second_kwargs.get("llm_override") is not None
        assert second_kwargs["llm_override"] is not practical.llm

    def test_parse_error_marker_also_triggers_retry(self, practical):
        # "Parse error" is the OTHER documented truncation marker (final
        # generic-error fallback path), distinct from the regex-rescue's
        # "json_repair" — both must trigger the retry.
        truncated = {"analysis": "Parse error", "answer": ""}
        complete = {"analysis": "ok", "action": "answer", "answer": "full answer"}
        practical._call_llm_json = MagicMock(side_effect=[truncated, complete])

        out = practical._call_llm_json_with_truncation_retry("prompt text")

        assert out == complete
        assert practical._call_llm_json.call_count == 2

    def test_still_truncated_after_retry_falls_back_to_first_result(self, practical):
        # Rare case: even the expanded budget wasn't enough. Must return the
        # FIRST attempt's result unchanged (documented fallback behavior),
        # not the still-truncated retry result.
        truncated_1 = {"analysis": "json_repair", "answer": "first partial"}
        truncated_2 = {"analysis": "json_repair", "answer": "second partial, different"}
        practical._call_llm_json = MagicMock(side_effect=[truncated_1, truncated_2])

        out = practical._call_llm_json_with_truncation_retry("prompt text")

        assert out == truncated_1
        assert practical._call_llm_json.call_count == 2

    def test_uses_configured_retry_token_budget(self, practical, monkeypatch):
        # ChatOpenAI is a pydantic model — .bind is inherited from the Runnable
        # base and can't be monkeypatched by plain attribute assignment
        # ("no field 'bind'"). Instead, inspect the RunnableBinding.kwargs the
        # real .bind() call produced, which is what the LLM actually receives.
        import conf
        monkeypatch.setattr(conf, "MAX_TOKENS_PRACTICAL_RETRY", 9999, raising=False)
        truncated = {"analysis": "json_repair", "answer": ""}
        complete = {"analysis": "ok", "answer": "done"}
        practical._call_llm_json = MagicMock(side_effect=[truncated, complete])

        practical._call_llm_json_with_truncation_retry("prompt text")

        second_kwargs = practical._call_llm_json.call_args_list[1].kwargs
        assert second_kwargs["llm_override"].kwargs.get("max_tokens") == 9999


@pytest.mark.unit
class TestBoostFieldsIncludesDepartment:
    """
    Regression pin for the 2026-09 fix: 'department' added to the metadata
    boost fields inside _retrieve_docs so non-bank departments (e.g. 'NTT
    DATA') can be ranked correctly when named directly in a query — see the
    inline comment at persona_practical.py's _BOOST_FIELDS definition for the
    documented, honest limitation (does not fix the case where the user
    doesn't know the department name to begin with, e.g. "ธนาคารไหน" phrasing;
    that needs live-data verification not possible in this environment).
    """

    def test_boost_fields_tuple_contains_department(self):
        import inspect
        source = inspect.getsource(PracticalPersonaService._retrieve_docs)
        m = re.search(r'_BOOST_FIELDS\s*=\s*\(([^)]*)\)', source)
        assert m, "_BOOST_FIELDS tuple definition not found in _retrieve_docs"
        fields = [f.strip().strip('"\'') for f in m.group(1).split(",") if f.strip()]
        assert "department" in fields
        # Guard the fields this was already covering before the fix.
        for f in ("operation_topic", "sub_topic", "main_topic", "license_type"):
            assert f in fields

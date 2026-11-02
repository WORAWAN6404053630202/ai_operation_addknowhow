"""
Unit tests for scripts/es_vs_chroma_langsmith_eval.py (Round 3 Phase 7,
.claude/plans/mutable-seeking-wave.md) — the real script costs money
(live HTTP + LLM judge calls), so these mock both boundaries: `requests`
for the target function, and the judge LLM's `.invoke()` for the
evaluators. Same "construct real objects, monkeypatch only the network
edge" approach the rest of this ES-project test suite already uses.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

CODE_DIR = Path(__file__).parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from scripts.es_vs_chroma_langsmith_eval import (
    _make_groundedness_evaluator,
    _make_target,
    _parse_judge_json,
    correctness_evaluator,
    relevance_evaluator,
)


def _run(outputs):
    return SimpleNamespace(outputs=outputs)


def _example(inputs, outputs):
    return SimpleNamespace(inputs=inputs, outputs=outputs)


@pytest.mark.unit
class TestParseJudgeJson:
    def test_valid_json(self):
        assert _parse_judge_json('{"correct": true}') == {"correct": True}

    def test_malformed_json_returns_empty_dict_not_raise(self):
        assert _parse_judge_json("not json at all") == {}


@pytest.mark.unit
class TestMakeTarget:
    """`_make_target` does `import requests` locally (inside the function,
    not at module scope) — `requests` is never a module-level attribute of
    this script, so the patch target must be the real `requests.post`
    (the same cached module object any local import binds to), not
    `scripts.es_vs_chroma_langsmith_eval.requests`."""

    def test_posts_question_with_fresh_session_and_include_sources(self):
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": "คำตอบ", "retrieved_sources": [{"metadata": {}}], "total_cost": 0.01}
            mock_post.return_value = mock_resp

            target = _make_target("http://example.test:3001")
            out = target({"question": "ค่าธรรมเนียมเท่าไหร่"})

            call_kwargs = mock_post.call_args.kwargs
            assert mock_post.call_args.args[0] == "http://example.test:3001/api/v1/chat"
            assert call_kwargs["json"]["message"] == "ค่าธรรมเนียมเท่าไหร่"
            assert call_kwargs["json"]["include_sources"] is True
            assert call_kwargs["json"]["session_id"].startswith("s_eval_")
            assert out == {"answer": "คำตอบ", "retrieved_sources": [{"metadata": {}}], "total_cost": 0.01}

    def test_two_calls_get_different_session_ids(self):
        """Each eval question must be a cold session — not accumulating
        multi-turn state across the 113-question run."""
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": "x"}
            mock_post.return_value = mock_resp

            target = _make_target("http://example.test:3001")
            target({"question": "q1"})
            target({"question": "q2"})

            sid1 = mock_post.call_args_list[0].kwargs["json"]["session_id"]
            sid2 = mock_post.call_args_list[1].kwargs["json"]["session_id"]
            assert sid1 != sid2


@pytest.mark.unit
class TestCorrectnessEvaluator:
    def test_skips_when_no_reference_answer(self):
        run = _run({"answer": "x"})
        example = _example({"question": "q"}, {"answer": ""})
        result = correctness_evaluator(run, example)
        assert result["key"] == "correctness"
        assert result["score"] is None

    def test_maps_judge_verdict_to_score(self):
        run = _run({"answer": "ค่าธรรมเนียม 500 บาท"})
        example = _example({"question": "ค่าธรรมเนียมเท่าไหร่"}, {"answer": "500 บาท"})
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = SimpleNamespace(content='{"reasoning": "matches", "correct": true}')
        with patch("scripts.es_vs_chroma_langsmith_eval._judge_llm", return_value=fake_llm):
            result = correctness_evaluator(run, example)
        assert result["score"] == 1.0
        assert result["comment"] == "matches"

    def test_false_verdict_scores_zero(self):
        run = _run({"answer": "wrong"})
        example = _example({"question": "q"}, {"answer": "right"})
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = SimpleNamespace(content='{"reasoning": "mismatch", "correct": false}')
        with patch("scripts.es_vs_chroma_langsmith_eval._judge_llm", return_value=fake_llm):
            result = correctness_evaluator(run, example)
        assert result["score"] == 0.0


@pytest.mark.unit
class TestRelevanceEvaluator:
    def test_maps_judge_verdict_to_score(self):
        run = _run({"answer": "ตอบตรงคำถาม"})
        example = _example({"question": "q"}, {"answer": "ref"})
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = SimpleNamespace(content='{"reasoning": "on topic", "relevant": true}')
        with patch("scripts.es_vs_chroma_langsmith_eval._judge_llm", return_value=fake_llm):
            result = relevance_evaluator(run, example)
        assert result["key"] == "relevance"
        assert result["score"] == 1.0


@pytest.mark.unit
class TestGroundednessEvaluator:
    def test_skipped_when_no_retrieved_sources(self):
        fake_service = MagicMock()
        with patch("model.persona_practical.PracticalPersonaService", return_value=fake_service):
            evaluator = _make_groundedness_evaluator()
        run = _run({"answer": "x", "retrieved_sources": None})
        example = _example({"question": "q"}, {"answer": "ref"})
        result = evaluator(run, example)
        assert result["score"] is None
        fake_service._groundedness_llm_call.assert_not_called()

    def test_grounded_answer_scores_one(self):
        fake_service = MagicMock()
        fake_service._groundedness_llm_call.return_value = {"reasoning": "all claims traceable", "unsupported": False, "claims": []}
        with patch("model.persona_practical.PracticalPersonaService", return_value=fake_service):
            evaluator = _make_groundedness_evaluator()
        run = _run({"answer": "x", "retrieved_sources": [{"metadata": {}, "content": "y"}]})
        example = _example({"question": "q"}, {"answer": "ref"})
        result = evaluator(run, example)
        assert result["key"] == "groundedness"
        assert result["score"] == 1.0

    def test_unsupported_claim_scores_zero(self):
        fake_service = MagicMock()
        fake_service._groundedness_llm_call.return_value = {"reasoning": "invented fee", "unsupported": True, "claims": ["500 บาท"]}
        with patch("model.persona_practical.PracticalPersonaService", return_value=fake_service):
            evaluator = _make_groundedness_evaluator()
        run = _run({"answer": "x", "retrieved_sources": [{"metadata": {}, "content": "y"}]})
        example = _example({"question": "q"}, {"answer": "ref"})
        result = evaluator(run, example)
        assert result["score"] == 0.0
        assert "500 บาท" in result["comment"]

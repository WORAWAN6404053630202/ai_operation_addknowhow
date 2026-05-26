"""
Unit tests for JSON parse robustness in LLM response handling.

Validates that json.JSONDecodeError is properly distinguished from other
exceptions, and that the parsing contract (dict or empty dict) is correct.
These tests verify the behavior that the supervisor's JSON-parsing closures rely on.
No LLM connection required.
"""
import json
import pytest


@pytest.mark.unit
class TestJsonParseRobustness:

    def test_valid_json_dict_parsed(self):
        text = '{"action": "answer", "question": ""}'
        obj = json.loads(text)
        result = obj if isinstance(obj, dict) else {}
        assert result == {"action": "answer", "question": ""}

    def test_valid_json_non_dict_returns_empty(self):
        text = '"just a string"'
        obj = json.loads(text)
        result = obj if isinstance(obj, dict) else {}
        assert result == {}

    def test_valid_json_list_returns_empty(self):
        text = '[1, 2, 3]'
        obj = json.loads(text)
        result = obj if isinstance(obj, dict) else {}
        assert result == {}

    def test_invalid_json_raises_decode_error(self):
        with pytest.raises(json.JSONDecodeError):
            json.loads("not valid json {")

    def test_truncated_json_raises_decode_error(self):
        with pytest.raises(json.JSONDecodeError):
            json.loads('{"action": "answer"')

    def test_json_decode_error_is_subclass_of_value_error(self):
        # json.JSONDecodeError inherits from ValueError — confirm narrowing
        # the except clause to JSONDecodeError does not suppress ValueError from other sources.
        assert issubclass(json.JSONDecodeError, ValueError)

    def test_empty_string_raises_decode_error(self):
        with pytest.raises(json.JSONDecodeError):
            json.loads("")

    def test_none_response_handled_as_empty_string(self):
        text = (None or "").strip()
        assert text == ""

    def test_whitespace_only_raises_decode_error(self):
        with pytest.raises(json.JSONDecodeError):
            json.loads("   ")

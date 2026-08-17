import pytest

from src.core.ci.response_parsing import _parse_triage_response


def test_parse_triage_response_accepts_fenced_json() -> None:
    parsed = _parse_triage_response(
        """```json
{"should_create_issue": true, "title": "fix: bug", "body": "details"}
```"""
    )

    assert parsed["should_create_issue"] is True
    assert parsed["title"] == "fix: bug"


def test_parse_triage_response_defaults_to_no_issue_for_prose() -> None:
    parsed = _parse_triage_response(
        "We need to decide if any listed issue qualifies, but this is not JSON."
    )

    assert parsed == {"should_create_issue": False, "title": "", "body": ""}


def test_parse_triage_response_rejects_json_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="missing required"):
        _parse_triage_response('{"title": "fix: bug", "body": "details"}')


def test_parse_triage_response_rejects_malformed_json_shape() -> None:
    with pytest.raises(ValueError, match="looked like JSON"):
        _parse_triage_response('{"should_create_issue": true, "title": "fix: bug"')

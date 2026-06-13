import pytest

from src.core.llm_manager import _parse_openrouter_json_response
from src.core.mcp_integration import _parse_json_text, _parse_markdown_link_results


class FakeResponse:
    def __init__(self, *, payload=None, text="", status_code=200, error=None):
        self._payload = payload
        self.text = text
        self.status_code = status_code
        self._error = error

    def json(self):
        if self._error:
            raise self._error
        return self._payload


def test_parse_openrouter_json_response_rejects_non_json_body() -> None:
    response = FakeResponse(
        text="<html><title>Bad Gateway</title></html>",
        status_code=502,
        error=ValueError("not json"),
    )

    with pytest.raises(RuntimeError, match="non-JSON response"):
        _parse_openrouter_json_response(response, "chat completion")


def test_parse_openrouter_json_response_rejects_non_object_json() -> None:
    response = FakeResponse(payload=["not", "an", "object"])

    with pytest.raises(RuntimeError, match="invalid JSON shape"):
        _parse_openrouter_json_response(response, "model list fetch")


def test_parse_json_text_distinguishes_json_from_plain_text() -> None:
    parsed, data = _parse_json_text('{"ok": true}', context="test")
    assert parsed is True
    assert data == {"ok": True}

    parsed, data = _parse_json_text("plain search output", context="test")
    assert parsed is False
    assert data is None


def test_parse_markdown_link_results() -> None:
    results = _parse_markdown_link_results(
        "1. [First](https://example.com)\n"
        "Snippet line\n"
        "2. [Second](https://example.org)\n"
    )

    assert results == [
        {
            "title": "First",
            "url": "https://example.com",
            "snippet": "Snippet line",
        },
        {
            "title": "Second",
            "url": "https://example.org",
            "snippet": "",
        },
    ]

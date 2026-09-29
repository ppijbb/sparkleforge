import pytest

from src.core.roadmap.vision_judge import DEFAULT_VISION_MODEL, call_vision_judge


def test_default_model_is_free_tier_compatible():
    # Regression for issue #1761: a paid default 402s for anyone on OpenRouter's free tier.
    assert DEFAULT_VISION_MODEL.endswith(":free")


def test_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        call_vision_judge("prompt", ["data:image/png;base64,abc"])


class _Resp:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


def test_sends_text_and_image_content_blocks(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp(body={"choices": [{"message": {"content": "NO_ACTIONABLE_CLI_UX_ISSUES"}}]})

    monkeypatch.setattr("src.core.roadmap.vision_judge.requests.post", fake_post)

    result = call_vision_judge("judge this", ["data:image/png;base64,abc"])

    assert result == "NO_ACTIONABLE_CLI_UX_ISSUES"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    content = captured["json"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "judge this"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}


def test_retries_and_recovers_from_embedded_200_error(monkeypatch):
    # Regression: upstream capacity errors (e.g. NVIDIA ResourceExhausted) come back
    # as HTTP 200 with an "error" body, not a real non-200 status.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("src.core.roadmap.vision_judge.time.sleep", lambda _seconds: None)
    responses = [
        _Resp(body={"error": {"message": "Upstream error from Nvidia: ResourceExhausted", "code": 502}}),
        _Resp(body={"choices": [{"message": {"content": "NO_ACTIONABLE_CLI_UX_ISSUES"}}]}),
    ]

    def fake_post(url, headers=None, json=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr("src.core.roadmap.vision_judge.requests.post", fake_post)

    result = call_vision_judge("judge this", ["data:image/png;base64,abc"])

    assert result == "NO_ACTIONABLE_CLI_UX_ISSUES"


def test_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("src.core.roadmap.vision_judge.time.sleep", lambda _seconds: None)

    def fake_post(url, headers=None, json=None, timeout=None):
        return _Resp(body={"error": {"message": "Upstream error from Nvidia: ResourceExhausted", "code": 502}})

    monkeypatch.setattr("src.core.roadmap.vision_judge.requests.post", fake_post)

    with pytest.raises(RuntimeError, match="ResourceExhausted"):
        call_vision_judge("judge this", ["data:image/png;base64,abc"], max_retries=2)

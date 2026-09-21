import pytest

from src.core.roadmap.vision_judge import call_vision_judge


def test_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        call_vision_judge("prompt", ["data:image/png;base64,abc"])


def test_sends_text_and_image_content_blocks(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "NO_ACTIONABLE_CLI_UX_ISSUES"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr("src.core.roadmap.vision_judge.requests.post", fake_post)

    result = call_vision_judge("judge this", ["data:image/png;base64,abc"])

    assert result == "NO_ACTIONABLE_CLI_UX_ISSUES"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    content = captured["json"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "judge this"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}

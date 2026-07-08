from src.core.cli_agents.open_code_agent import OpenCodeAgent


def test_open_code_agent_uses_cli_model_override_for_google_api(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")

    agent = OpenCodeAgent(model_path="google/gemini-2.0-flash-exp")

    assert agent._google_model() == "gemini-2.0-flash-exp"
    assert agent._model == "google/gemini-2.0-flash-exp"
    assert agent._max_tokens == 2048


def test_open_code_agent_strips_nested_google_models_prefix(monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)

    agent = OpenCodeAgent(model_path="google/models/gemini-2.0-flash-exp")

    assert agent._google_model() == "gemini-2.0-flash-exp"
    assert agent._model == "google/models/gemini-2.0-flash-exp"


def test_open_code_agent_keeps_openrouter_model_for_openrouter(monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)

    agent = OpenCodeAgent(model_path="moonshotai/kimi-k2.5")

    assert agent._model == "moonshotai/kimi-k2.5"
    assert agent._google_model() == "gemini-3.1-flash-lite-preview"
    assert agent._max_tokens == 4096

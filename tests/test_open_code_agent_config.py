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
    assert agent._google_model() == "gemini-3.5-flash-lite"
    assert agent._max_tokens == 4096


def test_open_code_agent_context_window_uses_model_defaults(monkeypatch):
    monkeypatch.delenv("OPEN_CODE_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("LLM_CONTEXT_WINDOW", raising=False)
    monkeypatch.setenv("OPENCODE_PRIMARY", "openrouter")

    gemini = OpenCodeAgent(model_path="google/gemini-2.0-flash-exp")
    kimi = OpenCodeAgent(model_path="moonshotai/kimi-k2.5")

    assert gemini.context_window() == 1_000_000
    assert kimi.context_window() == 262_144


def test_open_code_agent_context_window_honors_env_override(monkeypatch):
    monkeypatch.setenv("OPEN_CODE_CONTEXT_WINDOW", "64000")

    agent = OpenCodeAgent(model_path="google/gemini-2.0-flash-exp")

    assert agent.context_window() == 64_000


def test_open_code_agent_prompt_budget_reserves_output(monkeypatch):
    monkeypatch.setenv("OPEN_CODE_CONTEXT_WINDOW", "10000")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2000")

    agent = OpenCodeAgent(model_path="moonshotai/kimi-k2.5")

    assert agent.prompt_context_budget() == 4_000

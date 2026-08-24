import pytest

from src.core.cli_agents.open_code_agent import DEFAULT_MODEL, OPENROUTER_FALLBACKS, OpenCodeAgent


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


def test_openrouter_fallback_chain_is_all_free_tier():
    # A paid model id here 402s: OPENROUTER_API_KEY on this account has never
    # purchased credits, so the fallback chain must stay free-tier-only.
    assert DEFAULT_MODEL.endswith(":free")
    assert all(model.endswith(":free") for model in OPENROUTER_FALLBACKS)


def test_open_code_agent_prompt_budget_reserves_output(monkeypatch):
    monkeypatch.setenv("OPEN_CODE_CONTEXT_WINDOW", "10000")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2000")

    agent = OpenCodeAgent(model_path="moonshotai/kimi-k2.5")

    assert agent.prompt_context_budget() == 4_000


def _agent_with_keys(monkeypatch, *, openrouter=True, nvidia=True, google=False):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key" if openrouter else "")
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-key" if nvidia else "")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key" if google else "")
    monkeypatch.setenv("OPENCODE_PRIMARY", "openrouter")
    return OpenCodeAgent(model_path="moonshotai/kimi-k2.5")


async def test_execute_query_never_reports_bare_error_on_empty_exception_message(monkeypatch):
    """A str(exc) == "" exception (e.g. asyncio.TimeoutError) used to surface
    as the literal, content-free "[ERROR] " -- the actual failure reason was
    lost. It must now at least carry the exception's type name."""
    agent = _agent_with_keys(monkeypatch, openrouter=True, nvidia=False, google=False)

    async def _boom(*args, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(agent, "_call_openrouter_chain", _boom)

    result = await agent.execute_query("do something")

    assert result["success"] is False
    assert result["response"] != "[ERROR] "
    assert "TimeoutError" in result["response"]
    assert "TimeoutError" in result["metadata"]["error"]


async def test_non_runtime_error_from_openrouter_still_falls_back_to_nvidia(monkeypatch):
    """Regression test: the openrouter-chain except clause used to only catch
    RuntimeError, so a TimeoutError (or any non-RuntimeError) skipped the
    nvidia/google fallback entirely instead of triggering it."""
    agent = _agent_with_keys(monkeypatch, openrouter=True, nvidia=True, google=False)

    async def _timeout(*args, **kwargs):
        raise TimeoutError()

    async def _nvidia_ok(*args, **kwargs):
        agent._last_backend = "nvidia:some-model"
        return "nvidia saved the day"

    monkeypatch.setattr(agent, "_call_openrouter_chain", _timeout)
    monkeypatch.setattr(agent, "_call_nvidia_nim", _nvidia_ok)

    result = await agent.execute_query("do something")

    assert result["success"] is True
    assert result["response"] == "nvidia saved the day"
    assert result["metadata"]["backend"] == "nvidia:some-model"


async def test_all_backends_failing_reports_every_attempted_backend(monkeypatch):
    agent = _agent_with_keys(monkeypatch, openrouter=True, nvidia=True, google=True)

    async def _fail_openrouter(*args, **kwargs):
        raise RuntimeError("OpenRouter 429: rate limited")

    async def _fail_nvidia(*args, **kwargs):
        raise TimeoutError()

    async def _fail_google(*args, **kwargs):
        raise RuntimeError("Google Gemini 500: internal error")

    monkeypatch.setattr(agent, "_call_openrouter_chain", _fail_openrouter)
    monkeypatch.setattr(agent, "_call_nvidia_nim", _fail_nvidia)
    monkeypatch.setattr(agent, "_call_google_genai", _fail_google)

    result = await agent.execute_query("do something")

    assert result["success"] is False
    assert "rate limited" in result["response"]
    assert "TimeoutError" in result["response"]
    assert "internal error" in result["response"]


async def test_execute_query_reports_which_backend_served_a_successful_call(monkeypatch):
    agent = _agent_with_keys(monkeypatch, openrouter=True, nvidia=False, google=False)

    async def _openrouter_ok(*args, **kwargs):
        agent._last_backend = "openrouter:z-ai/glm-5.2:free"
        return "patch content"

    monkeypatch.setattr(agent, "_call_openrouter_chain", _openrouter_ok)

    result = await agent.execute_query("do something")

    assert result["success"] is True
    assert result["metadata"]["backend"] == "openrouter:z-ai/glm-5.2:free"

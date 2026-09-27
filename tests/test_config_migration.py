"""Tests for backward-compatible environment variable reads and config migration."""

import os
import warnings
import pytest
from src.core import researcher_config
from src.core.cli_agents.sparkle_llm_agent import SparkleLLMAgent, _primary_provider


def test_legacy_env_var_fallbacks(monkeypatch):
    # Clear new env vars and set legacy ones
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SPARKLEFORGE_MIGRATION_V2", raising=False)
    monkeypatch.delenv("SPARKLE_LLM_PRIMARY", raising=False)
    monkeypatch.delenv("SPARKLE_LLM_MODEL_PATH", raising=False)

    monkeypatch.setenv("OPENCODE_PRIMARY", "google")
    monkeypatch.setenv("OPEN_CODE_MODEL_PATH", "custom/legacy-model")

    with pytest.deprecated_call():
        provider = _primary_provider()
    assert provider == "google"

    agent = SparkleLLMAgent()
    assert agent._model == "custom/legacy-model"


def test_default_provider_backward_compatibility(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SPARKLEFORGE_MIGRATION_V2", raising=False)
    monkeypatch.delenv("OPENCODE_PRIMARY", raising=False)
    monkeypatch.delenv("OPEN_CODE_MODEL_PATH", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_MODEL", "deepseek/deepseek-v4")

    with pytest.deprecated_call():
        cfg = researcher_config.load_config_from_env()
    
    # Without migration v2, should gracefully fallback to opencode / maintain old default or handle safely
    assert cfg.llm.provider in ("opencode", "sparkle_llm")

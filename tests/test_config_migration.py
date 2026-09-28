"""Tests for backward-compatible environment variable migration and default provider handling."""

import os
import warnings
import pytest
from src.core import researcher_config


def test_legacy_env_var_fallback(monkeypatch):
    # Clear modern env vars and set legacy ones
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("SPARKLE_LLM_MODEL_PATH", raising=False)
    monkeypatch.delenv("OPENCODE_PRIMARY", raising=False)
    monkeypatch.delenv("OPEN_CODE_MODEL_PATH", raising=False)

    monkeypatch.setenv("OPENCODE_PRIMARY", "legacy-model-v1")
    monkeypatch.setenv("OPEN_CODE_MODEL_PATH", "/path/to/legacy/model")

    with pytest.warns(DeprecationWarning):
        cfg = researcher_config.load_config_from_env()

    assert cfg.llm.provider == "opencode"
    assert cfg.llm.primary_model == "legacy-model-v1"
    assert cfg.llm.sparkle_llm_model_path == "/path/to/legacy/model"


def test_modern_env_var_takes_precedence(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENCODE_PRIMARY", "legacy-model-v1")

    cfg = researcher_config.load_config_from_env()

    assert cfg.llm.provider == "openai"
    assert cfg.llm.primary_model == "gpt-4o-mini"


def test_default_provider_sparkle_llm(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENCODE_PRIMARY", raising=False)
    monkeypatch.delenv("OPEN_CODE_MODEL_PATH", raising=False)
    monkeypatch.delenv("OPENCODE_VERIFY_COMMAND", raising=False)

    # Should load sparkle_llm by default when no legacy vars are present
    cfg = researcher_config.load_config_from_env()
    assert cfg.llm.provider == "sparkle_llm"
    assert cfg.llm.primary_model == "sparkle_llm"

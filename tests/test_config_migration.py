"""Tests for LLM provider default/env-var resolution after the sparkle_llm rebrand.

The rebrand (see issue #1749) intentionally drops OPENCODE_PRIMARY/OPEN_CODE_MODEL_PATH
without a fallback: no external deployment (GH Actions vars, secrets, docs) was found
depending on those exact names at the time of the rename, so there is no legacy
config to preserve. These tests assert that actual behavior, not the fallback these
tests previously (and incorrectly) claimed existed.
"""

import pytest

from src.core import researcher_config


def _no_dotenv(monkeypatch):
    # load_config_from_env() does a local `from dotenv import load_dotenv` and calls
    # it with override=True, which would re-inject this machine's real .env values
    # over whatever the test sets up.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)


def test_legacy_env_vars_are_not_read(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("SPARKLE_LLM_MODEL_PATH", raising=False)

    monkeypatch.setenv("OPENCODE_PRIMARY", "legacy-model-v1")
    monkeypatch.setenv("OPEN_CODE_MODEL_PATH", "/path/to/legacy/model")

    cfg = researcher_config.load_config_from_env()

    # No fallback by design: the legacy names are simply inert now.
    assert cfg.llm.provider == "sparkle_llm"
    assert cfg.llm.primary_model == "sparkle_llm"
    assert cfg.llm.sparkle_llm_model_path is None


def test_modern_env_var_takes_precedence(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENCODE_PRIMARY", "legacy-model-v1")

    cfg = researcher_config.load_config_from_env()

    assert cfg.llm.provider == "openai"
    assert cfg.llm.primary_model == "gpt-4o-mini"


def test_default_provider_sparkle_llm(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENCODE_PRIMARY", raising=False)
    monkeypatch.delenv("OPEN_CODE_MODEL_PATH", raising=False)

    cfg = researcher_config.load_config_from_env()
    assert cfg.llm.provider == "sparkle_llm"
    assert cfg.llm.primary_model == "sparkle_llm"

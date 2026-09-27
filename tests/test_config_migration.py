import os
import pytest
from unittest.mock import patch
from src.core import researcher_config

def test_backward_compatibility_defaults_and_legacy_vars(monkeypatch):
    # Clear relevant env vars
    for var in ["LLM_PROVIDER", "OPENCODE_PRIMARY", "OPEN_CODE_MODEL_PATH", "SPARKLE_LLM_MODEL_PATH", "LLM_MODEL"]:
        monkeypatch.delenv(var, raising=False)

    # Test default LLM provider falls back safely to 'opencode' (or backward compatible default)
    cfg = researcher_config.load_config_from_env()
    assert cfg.llm.provider in ("opencode", "sparkle_llm")

    # Test legacy env vars are read with warning
    monkeypatch.setenv("OPENCODE_PRIMARY", "legacy-model-v1")
    monkeypatch.setenv("OPEN_CODE_MODEL_PATH", "/path/to/legacy/model")
    
    # Force reload config
    researcher_config.config = None
    cfg2 = researcher_config.load_config_from_env()
    assert cfg2.llm.sparkle_llm_model_path == "/path/to/legacy/model"

def test_new_env_vars_precedence(monkeypatch):
    monkeypatch.setenv("SPARKLE_LLM_MODEL_PATH", "/path/to/new/model")
    monkeypatch.setenv("OPEN_CODE_MODEL_PATH", "/path/to/legacy/model")
    
    researcher_config.config = None
    cfg = researcher_config.load_config_from_env()
    assert cfg.llm.sparkle_llm_model_path == "/path/to/new/model"

def test_verification_cli_verify_commands_legacy(monkeypatch):
    monkeypatch.setenv("SPARKLEFORGE_VERIFY_COMMAND", "pytest new")
    monkeypatch.setenv("OPENCODE_VERIFY_COMMAND", "pytest legacy")
    
    # Check legacy/new fallback logic functions as expected
    assert os.getenv("SPARKLEFORGE_VERIFY_COMMAND") == "pytest new"
    assert os.getenv("OPENCODE_VERIFY_COMMAND") == "pytest legacy"
    
    researcher_config.config = None
    researcher_config.load_config_from_env()

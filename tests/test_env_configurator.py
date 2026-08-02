import pytest

from src.core.env_configurator import ConfigurationError, verify_environment


def test_verify_environment_requires_variable_presence(monkeypatch) -> None:
    monkeypatch.delenv("SPARKLEFORGE_ENV", raising=False)

    with pytest.raises(ConfigurationError, match="SPARKLEFORGE_ENV"):
        verify_environment(["SPARKLEFORGE_ENV"])


def test_verify_environment_accepts_empty_present_variable(monkeypatch) -> None:
    monkeypatch.setenv("SPARKLEFORGE_ENV", "")

    assert verify_environment() is True


def test_verify_environment_supports_custom_required_vars(monkeypatch) -> None:
    monkeypatch.setenv("CUSTOM_REQUIRED_ENV", "set")

    assert verify_environment(["CUSTOM_REQUIRED_ENV"]) is True

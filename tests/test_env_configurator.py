import pytest

from src.core.env_configurator import ConfigurationError, EnvConfigurator


def test_verify_environment_requires_variable_presence() -> None:
    config = EnvConfigurator(environ={})
    with pytest.raises(ConfigurationError, match="SPARKLEFORGE_ENV"):
        config.verify_environment()


def test_verify_environment_accepts_empty_present_variable() -> None:
    config = EnvConfigurator(environ={"SPARKLEFORGE_ENV": ""})
    assert config.verify_environment() is True


def test_verify_environment_supports_custom_required_vars() -> None:
    config = EnvConfigurator(environ={"CUSTOM_REQUIRED_ENV": "set"})
    assert config.verify_environment(["CUSTOM_REQUIRED_ENV"]) is True

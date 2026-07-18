"""Issue #690: sandbox memory/cpu/storage limits are configurable, not hardcoded.

get_sandbox() already read SPARKLEFORGE_DOCKER_RUNTIME/SANDBOX_*_IMAGE from
the environment, but memory_limit/cpu_limit/tmpfs_size/pids_limit silently
fell back to hardcoded SandboxConfig defaults regardless of environment
configuration. These tests verify the env vars are now honored, and that
omitting them still falls back to the same defaults as before.
"""

import src.core.sandbox.docker_sandbox as docker_sandbox
from src.core.sandbox.docker_sandbox import SandboxConfig


class _FakeDockerSandbox:
    """Captures the SandboxConfig passed in, without requiring a real Docker install."""

    def __init__(self, config):
        self.config = config


def _get_sandbox_config(monkeypatch, env: dict) -> SandboxConfig:
    monkeypatch.setattr(docker_sandbox, "_sandbox_instance", None)
    monkeypatch.setattr(docker_sandbox, "DockerSandbox", _FakeDockerSandbox)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sandbox = docker_sandbox.get_sandbox()
    return sandbox.config


def test_get_sandbox_honors_resource_limit_env_vars(monkeypatch):
    config = _get_sandbox_config(
        monkeypatch,
        {
            "SPARKLEFORGE_SANDBOX_MEMORY_LIMIT": "1g",
            "SPARKLEFORGE_SANDBOX_CPU_LIMIT": "2.0",
            "SPARKLEFORGE_SANDBOX_TMPFS_SIZE": "256m",
            "SPARKLEFORGE_SANDBOX_PIDS_LIMIT": "64",
        },
    )

    assert config.memory_limit == "1g"
    assert config.cpu_limit == 2.0
    assert config.tmpfs_size == "256m"
    assert config.pids_limit == 64


def test_get_sandbox_falls_back_to_defaults_without_env_vars(monkeypatch):
    for key in (
        "SPARKLEFORGE_SANDBOX_MEMORY_LIMIT",
        "SPARKLEFORGE_SANDBOX_CPU_LIMIT",
        "SPARKLEFORGE_SANDBOX_TMPFS_SIZE",
        "SPARKLEFORGE_SANDBOX_PIDS_LIMIT",
    ):
        monkeypatch.delenv(key, raising=False)

    config = _get_sandbox_config(monkeypatch, {})

    assert config.memory_limit == SandboxConfig.memory_limit
    assert config.cpu_limit == SandboxConfig.cpu_limit
    assert config.tmpfs_size == SandboxConfig.tmpfs_size
    assert config.pids_limit == SandboxConfig.pids_limit

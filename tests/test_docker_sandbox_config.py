"""Issue #690: sandbox memory/cpu/storage limits are configurable, not hardcoded.

get_sandbox() already read SPARKLEFORGE_DOCKER_RUNTIME/SANDBOX_*_IMAGE from
the environment, but memory_limit/cpu_limit/tmpfs_size/pids_limit silently
fell back to hardcoded SandboxConfig defaults regardless of environment
configuration. These tests verify the env vars are now honored, and that
omitting them still falls back to the same defaults as before.

Issue #700: sandbox containers had no way to use custom DNS servers, so code
execution on networks with private/internal DNS (no public resolvers) could
not resolve hostnames even with network access enabled. Covers the new
SPARKLEFORGE_SANDBOX_DNS_SERVERS env var and its plumbing into the Docker
`dns` container option.
"""

import src.core.sandbox.docker_sandbox as docker_sandbox
from src.core.sandbox.docker_sandbox import DockerSandbox, SandboxConfig


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
        "SPARKLEFORGE_SANDBOX_DNS_SERVERS",
    ):
        monkeypatch.delenv(key, raising=False)

    config = _get_sandbox_config(monkeypatch, {})

    assert config.memory_limit == SandboxConfig.memory_limit
    assert config.cpu_limit == SandboxConfig.cpu_limit
    assert config.tmpfs_size == SandboxConfig.tmpfs_size
    assert config.pids_limit == SandboxConfig.pids_limit
    assert config.dns_servers is None


def test_get_sandbox_honors_dns_servers_env_var(monkeypatch):
    config = _get_sandbox_config(
        monkeypatch, {"SPARKLEFORGE_SANDBOX_DNS_SERVERS": "10.0.0.2, 10.0.0.3"}
    )

    assert config.dns_servers == ("10.0.0.2", "10.0.0.3")


def test_get_sandbox_dns_servers_default_to_none_without_env_var(monkeypatch):
    monkeypatch.delenv("SPARKLEFORGE_SANDBOX_DNS_SERVERS", raising=False)

    config = _get_sandbox_config(monkeypatch, {})

    assert config.dns_servers is None


def _sandbox_with_config(config: SandboxConfig) -> DockerSandbox:
    sandbox = object.__new__(DockerSandbox)
    sandbox.config = config
    return sandbox


def test_container_kwargs_includes_dns_when_configured():
    sandbox = _sandbox_with_config(SandboxConfig(dns_servers=("8.8.8.8", "1.1.1.1")))

    kwargs = sandbox._container_kwargs("python:3.11-slim", ["python", "-c", "1"], None)

    assert kwargs["dns"] == ["8.8.8.8", "1.1.1.1"]


def test_container_kwargs_omits_dns_when_not_configured():
    sandbox = _sandbox_with_config(SandboxConfig(dns_servers=None))

    kwargs = sandbox._container_kwargs("python:3.11-slim", ["python", "-c", "1"], None)

    assert "dns" not in kwargs

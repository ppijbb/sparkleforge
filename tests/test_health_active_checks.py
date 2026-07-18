"""Issue #683: `sparkleforge health` performs active subsystem checks.

Previously quick_health_check/run_comprehensive_health_check only reported
passively-collected resource metrics -- there was no active validation that
Docker responds, the sandbox can actually execute a command, or the
OpenRouter API is reachable. These tests cover the three new check methods
and the CLI wiring that fails the command when the sandbox is broken.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.cli.main_commands as main_commands
import src.monitoring.system_monitor as system_monitor
from src.monitoring.system_monitor import HealthMonitor


@pytest.fixture(autouse=True)
def _fake_reliability_config(monkeypatch):
    """HealthMonitor.__init__ reads module-level `config`, which is only
    populated by load_config_from_env() (called from main.py at startup) --
    unset in a bare test process. A minimal stand-in is enough since every
    read goes through getattr(..., default)."""
    monkeypatch.setattr(system_monitor, "config", SimpleNamespace(reliability=SimpleNamespace()))


def test_check_docker_available_reports_missing_binary():
    monitor = HealthMonitor()

    async def _raise_not_found(*args, **kwargs):
        raise FileNotFoundError("no docker")

    with patch("asyncio.create_subprocess_exec", side_effect=_raise_not_found):
        result = asyncio.run(monitor.check_docker_available())

    assert result["ok"] is False
    assert "not found" in result["detail"]


def test_check_docker_available_reports_success():
    monitor = HealthMonitor()

    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(b"", b""))

    async def _fake_exec(*args, **kwargs):
        return fake_proc

    with patch("asyncio.create_subprocess_exec", side_effect=_fake_exec):
        result = asyncio.run(monitor.check_docker_available())

    assert result["ok"] is True


def test_check_sandbox_write_passes_when_echo_succeeds():
    monitor = HealthMonitor()

    fake_result = SimpleNamespace(ok=True, stdout="sparkleforge_health_check\n", stderr="", sandbox_type="subprocess")

    with patch(
        "src.core.guard.sandbox_executor.SandboxExecutor.execute_async",
        new=AsyncMock(return_value=fake_result),
    ):
        result = asyncio.run(monitor.check_sandbox_write())

    assert result["ok"] is True
    assert "subprocess" in result["detail"]


def test_check_sandbox_write_fails_when_command_fails():
    monitor = HealthMonitor()

    fake_result = SimpleNamespace(ok=False, stdout="", stderr="permission denied", sandbox_type="firejail")

    with patch(
        "src.core.guard.sandbox_executor.SandboxExecutor.execute_async",
        new=AsyncMock(return_value=fake_result),
    ):
        result = asyncio.run(monitor.check_sandbox_write())

    assert result["ok"] is False
    assert "permission denied" in result["detail"]


def test_check_openrouter_api_skips_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monitor = HealthMonitor()

    result = asyncio.run(monitor.check_openrouter_api())

    assert result["ok"] is None
    assert "not set" in result["detail"]


def test_check_openrouter_api_reports_success(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monitor = HealthMonitor()

    fake_response = SimpleNamespace(status_code=200)
    with patch("requests.get", return_value=fake_response):
        result = asyncio.run(monitor.check_openrouter_api())

    assert result["ok"] is True


def test_run_active_subsystem_checks_aggregates_all_three():
    monitor = HealthMonitor()
    monitor.check_docker_available = AsyncMock(return_value={"ok": True, "detail": ""})
    monitor.check_sandbox_write = AsyncMock(return_value={"ok": True, "detail": ""})
    monitor.check_openrouter_api = AsyncMock(return_value={"ok": None, "detail": "skipped"})

    result = asyncio.run(monitor.run_active_subsystem_checks())

    assert set(result.keys()) == {"docker", "sandbox_write", "openrouter_api"}
    assert result["sandbox_write"]["ok"] is True


def test_handle_health_command_fails_when_sandbox_is_broken(monkeypatch):
    fake_checks = {
        "docker": {"ok": True, "detail": ""},
        "sandbox_write": {"ok": False, "detail": "boom"},
        "openrouter_api": {"ok": None, "detail": "skipped"},
    }

    monkeypatch.setattr(
        HealthMonitor, "run_active_subsystem_checks", AsyncMock(return_value=fake_checks)
    )
    monkeypatch.setattr(HealthMonitor, "quick_health_check", AsyncMock(return_value=True))

    args = SimpleNamespace(detailed=False)
    result = asyncio.run(main_commands.handle_health_command(args))

    assert result == 1

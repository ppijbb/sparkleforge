"""Issue #682: reject runs when free disk space is below a safety threshold.

Low disk space can cause SQLite writes to fail mid-transaction. These tests
cover the check itself (src/core/observe/system_collector.py) and that the
CLI run/coworker entry points actually reject before doing any real work
when the check fails.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import src.cli.main_commands as main_commands
from src.core.observe.system_collector import check_disk_space_safety


def test_check_disk_space_safety_rejects_when_below_threshold():
    with patch("src.core.observe.system_collector.PSUTIL_AVAILABLE", True), patch(
        "src.core.observe.system_collector.psutil"
    ) as mock_psutil:
        mock_psutil.disk_usage.return_value = SimpleNamespace(free=100 * 1024 * 1024)
        is_safe, message = check_disk_space_safety(min_free_mb=500.0)

    assert is_safe is False
    assert "500" in message


def test_check_disk_space_safety_passes_when_above_threshold():
    with patch("src.core.observe.system_collector.PSUTIL_AVAILABLE", True), patch(
        "src.core.observe.system_collector.psutil"
    ) as mock_psutil:
        mock_psutil.disk_usage.return_value = SimpleNamespace(free=10 * 1024 * 1024 * 1024)
        is_safe, message = check_disk_space_safety(min_free_mb=500.0)

    assert is_safe is True
    assert message == ""


def test_check_disk_space_safety_fails_open_without_psutil():
    with patch("src.core.observe.system_collector.PSUTIL_AVAILABLE", False):
        is_safe, message = check_disk_space_safety()

    assert is_safe is True


def test_handle_run_command_rejects_before_orchestrator_init(monkeypatch):
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_disk_space_safety",
        lambda *args, **kwargs: (False, "no space"),
    )

    def _boom():
        raise AssertionError("orchestrator must not be initialized when disk check fails")

    monkeypatch.setattr(main_commands, "_load_autonomous_orchestrator", _boom, raising=False)

    args = SimpleNamespace(
        mode="research",
        query="test query",
        model=None,
        max_tokens=None,
        task=None,
        session_id=None,
        continue_session=False,
    )
    config = SimpleNamespace(llm=SimpleNamespace(provider="openrouter"))

    result = asyncio.run(main_commands.handle_run_command(args, config))
    assert result == 1


def test_execute_coworker_goal_rejects_before_orchestrator_init(monkeypatch):
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_disk_space_safety",
        lambda *args, **kwargs: (False, "no space"),
    )

    result = asyncio.run(main_commands._execute_coworker_goal("do something"))
    assert result == 1

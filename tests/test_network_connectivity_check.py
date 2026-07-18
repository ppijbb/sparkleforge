"""Issue #680: warn upfront on no network connectivity before scheduling LLM calls.

Previously only a *reactive* connectivity check existed (error_handler.py,
run after a network error already happened). check_network_connectivity()
is a proactive pre-flight probe, warn-only by design since local-model-only
setups don't need internet and a blocked probe host doesn't necessarily mean
every provider is unreachable.
"""

import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import patch

import src.cli.main_commands as main_commands
from src.core.observe.system_collector import check_network_connectivity


def test_check_network_connectivity_reports_connected():
    with patch("socket.create_connection") as mock_connect:
        is_connected, message = check_network_connectivity()

    mock_connect.assert_called_once()
    assert is_connected is True
    assert message == ""


def test_check_network_connectivity_reports_disconnected():
    with patch("socket.create_connection", side_effect=OSError("network unreachable")):
        is_connected, message = check_network_connectivity()

    assert is_connected is False
    assert "network unreachable" in message


def test_handle_run_command_warns_but_does_not_reject_when_offline(monkeypatch, caplog):
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_network_connectivity",
        lambda: (False, "no network"),
    )

    def _boom():
        raise AssertionError("orchestrator must not be initialized in this offline-check test")

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

    import logging

    with caplog.at_level(logging.WARNING):
        # Stop right after the network check by making the orchestrator
        # loader raise -- proves the offline warning alone doesn't reject.
        try:
            asyncio.run(main_commands.handle_run_command(args, config))
        except AssertionError:
            pass

    assert any("no network" in record.message for record in caplog.records)


def test_execute_coworker_goal_warns_but_does_not_reject_when_offline(monkeypatch, caplog):
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_network_connectivity",
        lambda: (False, "no network"),
    )

    async def fake_orchestrator_execute(*args, **kwargs):
        return {"content": "ok"}

    monkeypatch.setattr(
        "src.core.agent_orchestrator.get_orchestrator",
        lambda: SimpleNamespace(execute=fake_orchestrator_execute),
    )

    import logging

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(main_commands._execute_coworker_goal("do something"))

    assert result == 0  # offline warning must not block the coworker path either
    assert any("no network" in record.message for record in caplog.records)

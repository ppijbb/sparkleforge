"""Unit tests for Codex and Hermes CLI Agent adapters in SparkleForge."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from src.core.cli_agents.cli_agent_manager import get_cli_agent_manager
from src.core.cli_agents.codex_agent import CodexCLIAgent
from src.core.cli_agents.hermes_agent import HermesCLIAgent


def test_cli_agent_manager_registration():
    manager = get_cli_agent_manager()
    available = manager.get_available_agents()

    assert "codex" in available
    assert "hermes" in available
    assert "claude_code" in available
    assert "gemini_cli" in available


@pytest.mark.asyncio
async def test_codex_agent_instantiation_and_parse():
    agent = CodexCLIAgent(api_key="test-key")
    assert agent.config.name == "codex"
    assert agent.config.command == "codex"

    # Test output parsing
    mock_result = type(
        "Result",
        (),
        {
            "success": True,
            "output": '{"response": "def foo(): pass", "confidence": 0.9, "usage": {"tokens": 100}}',
            "error": "",
            "exit_code": 0,
            "execution_time": 0.5,
        },
    )()

    parsed = agent.parse_output(mock_result)
    assert parsed["success"] is True
    assert parsed["response"] == "def foo(): pass"
    assert parsed["confidence"] == 0.9


@pytest.mark.asyncio
async def test_hermes_agent_instantiation_and_parse():
    agent = HermesCLIAgent(api_key="test-key")
    assert agent.config.name == "hermes"
    assert agent.config.command == "hermes"

    mock_result = type(
        "Result",
        (),
        {
            "success": True,
            "output": '{"output": "Workflow completed successfully", "confidence": 0.85}',
            "error": "",
            "exit_code": 0,
            "execution_time": 1.2,
        },
    )()

    parsed = agent.parse_output(mock_result)
    assert parsed["success"] is True
    assert parsed["response"] == "Workflow completed successfully"
    assert parsed["confidence"] == 0.85


def test_codex_parse_output_malformed_no_attribute_error(caplog):
    """Malformed output must not raise AttributeError (issue #1124)."""
    agent = CodexCLIAgent(api_key="test-key")

    mock_result = type(
        "Result",
        (),
        {
            "success": True,
            "output": object(),
            "error": "",
            "exit_code": 0,
            "execution_time": 0.5,
        },
    )()

    with caplog.at_level("ERROR"):
        parsed = agent.parse_output(mock_result)

    assert parsed["success"] is False
    assert "Parsing failed" in parsed["error"]


@pytest.mark.asyncio
async def test_hermes_concurrent_calls_on_shared_instance_do_not_mix_args():
    """CLIAgentManager caches one HermesCLIAgent instance per name and batch
    dispatch can run two hermes tasks concurrently against it. If
    execute_query mutated self.config.args in place, one task's command
    could leak into the other's (or get wiped by the other's restore)."""
    agent = HermesCLIAgent(api_key="test-key")
    captured_commands = []

    async def fake_execute_command(command, input_text=None):
        captured_commands.append(list(command))
        # Yield control so a truly concurrent second call can interleave here
        # if execute_query still mutated shared state before awaiting.
        await asyncio.sleep(0.01)
        return type(
            "Result",
            (),
            {"success": True, "output": '{"output": "ok", "confidence": 0.8}', "error": "",
             "exit_code": 0, "execution_time": 0.1},
        )()

    with patch.object(agent, "_execute_command", new=AsyncMock(side_effect=fake_execute_command)):
        await asyncio.gather(
            agent.execute_query("query-A", task_type="agentic"),
            agent.execute_query("query-B", task_type="agentic"),
        )

    assert len(captured_commands) == 2
    cmd_a = next(c for c in captured_commands if "query-A" in c)
    cmd_b = next(c for c in captured_commands if "query-B" in c)
    # Each command must carry only its own query, never the other's.
    assert "query-B" not in cmd_a
    assert "query-A" not in cmd_b
    # Shared config.args must be untouched after both calls finish.
    assert agent.config.args == ["--format", "json"]


def test_hermes_parse_output_malformed_no_attribute_error(caplog):
    """Malformed output must not raise AttributeError (issue #1124)."""
    agent = HermesCLIAgent(api_key="test-key")

    mock_result = type(
        "Result",
        (),
        {
            "success": True,
            "output": object(),
            "error": "",
            "exit_code": 0,
            "execution_time": 1.2,
        },
    )()

    with caplog.at_level("ERROR"):
        parsed = agent.parse_output(mock_result)

    assert parsed["success"] is False
    assert "Parsing failed" in parsed["error"]

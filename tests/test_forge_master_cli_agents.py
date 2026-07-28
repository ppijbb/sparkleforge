"""Unit tests for Codex and Hermes CLI Agent adapters in SparkleForge."""

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

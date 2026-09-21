"""Self-check for BaseCLIAgent._execute_command_pty: the child must see a real tty."""

import pytest
from src.core.cli_agents.base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult


class _EchoAgent(BaseCLIAgent):
    async def execute_query(self, query: str, **kwargs):
        raise NotImplementedError

    def parse_output(self, result: CLIExecutionResult):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_execute_command_pty_gives_child_a_real_tty():
    agent = _EchoAgent(CLIAgentConfig(name="echo_test", command="python3"))
    result = await agent._execute_command_pty(
        ["python3", "-c", "import sys; print(sys.stdout.isatty())"]
    )
    assert result.success
    assert "True" in result.output


@pytest.mark.asyncio
async def test_execute_command_pty_times_out():
    agent = _EchoAgent(CLIAgentConfig(name="echo_test", command="python3"))
    result = await agent._execute_command_pty(["sleep", "5"], timeout=1)
    assert not result.success
    assert result.exit_code == -1

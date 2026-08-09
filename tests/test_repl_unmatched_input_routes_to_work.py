"""Unmatched REPL free text must reach AgentHarness (via work_command),
not the tool-less AutonomousOrchestrator research pipeline.

Regression for: free-text goals silently fell back to `research`, a path
with zero tools registered, so the harness's classify/planner nodes never
got a chance to decide the task actually needed file edits / tool calls.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.cli.repl_cli import REPLCLI


@pytest.mark.asyncio
async def test_unmatched_text_dispatches_to_work_not_research():
    repl = REPLCLI.__new__(REPLCLI)
    repl.console = type("_C", (), {"print": lambda self, *a, **k: None})()
    work_mock = AsyncMock()
    research_mock = AsyncMock()
    repl.command_handlers = {"work": work_mock, "research": research_mock}
    repl._try_route_command = AsyncMock(return_value=False)

    await repl.handle_command("fix the failing test in payments module")

    work_mock.assert_awaited_once_with(repl, ["fix the failing test in payments module"])
    research_mock.assert_not_awaited()

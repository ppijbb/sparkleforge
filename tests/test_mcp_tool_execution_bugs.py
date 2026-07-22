"""Regression tests for the monolith-split refactor bugs in main_commands.py,
code.py, status.py, and execution.py (CLI import, Docker sandbox return path,
execute_tool signature, and MCP-session fallback).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.core.mcp_integration.hub_mixins.execution import ExecutionMixin
from src.core.tools.registry import ToolCategory


def test_main_commands_imports_without_error() -> None:
    # main_commands.py imports WebAppManager, _load_autonomous_orchestrator,
    # and project_root from src.core.autonomous_research_system; a missing
    # export there used to break the CLI entry point at import time.
    import src.cli.main_commands  # noqa: F401


@pytest.mark.asyncio
async def test_docker_sandbox_execution_returns_tool_result() -> None:
    # The Docker execution branch in code.py used to fall through without
    # returning, so every Docker-sandboxed code tool call silently produced
    # None instead of a ToolResult.
    from src.core.mcp_integration.executors.code import _execute_code_tool
    from src.core.sandbox.docker_sandbox import ExecutionResult

    fake_result = ExecutionResult(
        success=True, output="hi\n", error="", exit_code=0, execution_time=0.01
    )
    fake_sandbox = SimpleNamespace(execute_code=AsyncMock(return_value=fake_result))

    with patch(
        "src.core.sandbox.docker_sandbox.get_sandbox", return_value=fake_sandbox
    ):
        result = await _execute_code_tool(
            "code", {"code": "print('hi')", "language": "python", "sandbox": "docker"}
        )

    assert result is not None
    assert result.success is True
    assert result.data["output"] == "hi\n"


def test_execute_tool_signature_matches_status_py_call_sites() -> None:
    # status.py's health_check calls execute_tool(tool, {...}) positionally.
    # execute_tool used to be documented as taking citation_id as its second
    # positional argument elsewhere in the codebase; assert the actual
    # imported symbol accepts (tool_name, parameters) so that shape doesn't
    # silently regress.
    from src.core.mcp_integration.tools import execute_tool

    sig = inspect.signature(execute_tool)
    params = list(sig.parameters.values())

    assert params[0].name == "tool_name"
    assert params[1].name == "parameters"
    # If the second parameter were named/typed as a citation id (str/int)
    # instead of a dict, calling execute_tool(tool, {...}) would be a
    # TypeError at runtime.
    assert "Dict" in str(params[1].annotation) or params[1].annotation is dict


class _FakeRegistry:
    """Routes 'myserver::mytool' to a resolved-but-unconnected MCP server."""

    tool_sources: dict = {}

    def get_tool_info(self, name: str):
        return SimpleNamespace(mcp_server="myserver")

    def get_all_tool_names(self):
        return ["myserver::mytool"]

    def is_mcp_tool(self, name: str) -> bool:
        return True

    def get_mcp_server_info(self, name: str):
        return ("myserver", "mytool")


class _FakeHub(ExecutionMixin):
    def __init__(self):
        self.registry = _FakeRegistry()
        self.mcp_sessions: dict = {}  # 'myserver' is never connected


@pytest.mark.asyncio
async def test_execute_tool_does_not_fall_back_to_local_when_mcp_server_disconnected() -> None:
    # When a tool resolves to an MCP server (mcp_info is truthy) but that
    # server isn't in self.mcp_sessions yet, execution used to fall through
    # to the local-tool-fallback branch below instead of returning an
    # explicit error -- risking silently running a different, same-named
    # local tool instead of the MCP-only one that was requested.
    hub = _FakeHub()

    result = await hub.execute_tool("myserver::mytool", {"x": 1})

    assert result["success"] is False
    assert result["source"] == "mcp"
    assert "myserver" in result["error"]
    assert "not connected" in result["error"]


class _UtilityFakeRegistry:
    """A local, non-MCP tool registered under a category with no dedicated
    dispatch function (e.g. UTILITY for scheduler tools, BROWSER for CDP
    tools)."""

    def __init__(self, executor):
        self.tool_sources = {"create_automation_task": "local"}
        self._executor = executor

    def get_tool_info(self, name: str):
        return SimpleNamespace(category=ToolCategory.UTILITY, mcp_server=None)

    def get_all_tool_names(self):
        return ["create_automation_task"]

    def is_mcp_tool(self, name: str) -> bool:
        return False

    def get_mcp_server_info(self, name: str):
        return None

    async def execute(self, name: str, arguments: dict):
        return await self._executor(name, arguments)


class _UtilityFakeHub(ExecutionMixin):
    def __init__(self, registry):
        self.registry = registry
        self.mcp_sessions: dict = {}


@pytest.mark.asyncio
async def test_execute_tool_dispatches_utility_category_tools_through_registry() -> None:
    # Local tools registered under a category with no dedicated dispatch
    # function (UTILITY, BROWSER, ...) used to fall through to the DATA-tool
    # dispatcher, which only recognizes fetch/filesystem/browser/shell and
    # raised "Unknown data tool" for anything else -- e.g. the scheduler's
    # create_automation_task/list_automation_tasks (#817). They should be
    # invoked through the registry's own registered executor instead.
    created = {"id": "sched-1", "name": "daily-report"}

    async def fake_executor(name, arguments):
        assert name == "create_automation_task"
        assert arguments == {"name": "daily-report", "cron_expression": "0 9 * * *"}
        return created

    hub = _UtilityFakeHub(_UtilityFakeRegistry(fake_executor))

    result = await hub.execute_tool(
        "create_automation_task",
        {"name": "daily-report", "cron_expression": "0 9 * * *"},
    )

    assert result["success"] is True
    assert result["data"] == created
    assert result["source"] == "local"

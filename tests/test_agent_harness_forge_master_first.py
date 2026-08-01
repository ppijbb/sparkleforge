"""Issue #1181: codebase_agent-routed tasks should try the local ForgeMaster
CLI-agent fleet before ever reaching the frontier-LLM ParallelAgentExecutor
path. Frontier API usage should be a last resort, not the default.
"""

import asyncio
from unittest.mock import AsyncMock, patch

from src.core.agent_harness import AgentHarness
from src.core.surface.task_dashboard import TaskDashboard


def _harness() -> AgentHarness:
    harness = object.__new__(AgentHarness)  # skip __init__'s heavy tool registration
    harness.dashboard = TaskDashboard()
    return harness


def test_forge_master_success_removes_task_from_frontier_queue():
    async def run_test():
        harness = _harness()
        tasks = [{"task_id": "t1", "description": "write an add function"}]

        fake_batch_result = {
            "success": True,
            "total": 1,
            "succeeded": 1,
            "results": [{"success": True, "response": "def add(a,b): return a+b", "agent_used": "codex"}],
        }

        with patch(
            "src.core.forge_master.tools._dispatch_batch_to_forge_master_tool",
            new=AsyncMock(return_value=fake_batch_result),
        ):
            unhandled, handled = await harness._dispatch_codebase_tasks_via_forge_master(
                tasks, session_id="s1"
            )

        assert unhandled == []
        assert len(handled) == 1
        assert handled[0]["status"] == "completed"
        assert handled[0]["result"] == "def add(a,b): return a+b"

    asyncio.run(run_test())


def test_forge_master_failure_falls_through_to_frontier_queue():
    async def run_test():
        harness = _harness()
        tasks = [{"task_id": "t1", "description": "write an add function"}]

        fake_batch_result = {
            "success": False,
            "total": 1,
            "succeeded": 0,
            "results": [{"success": False, "error": "adversarial audit rejected"}],
        }

        with patch(
            "src.core.forge_master.tools._dispatch_batch_to_forge_master_tool",
            new=AsyncMock(return_value=fake_batch_result),
        ):
            unhandled, handled = await harness._dispatch_codebase_tasks_via_forge_master(
                tasks, session_id="s1"
            )

        assert handled == []
        assert unhandled == tasks

    asyncio.run(run_test())


def test_forge_master_dispatch_exception_falls_back_to_frontier_queue_untouched():
    async def run_test():
        harness = _harness()
        tasks = [{"task_id": "t1", "description": "write an add function"}]

        with patch(
            "src.core.forge_master.tools._dispatch_batch_to_forge_master_tool",
            new=AsyncMock(side_effect=RuntimeError("cli fleet unavailable")),
        ):
            unhandled, handled = await harness._dispatch_codebase_tasks_via_forge_master(
                tasks, session_id="s1"
            )

        assert handled == []
        assert unhandled == tasks

    asyncio.run(run_test())

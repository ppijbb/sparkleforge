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


def _state(tasks):
    return {"workflow": {"session_id": "s1", "tasks": tasks}}


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
                _state(tasks), tasks
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
                _state(tasks), tasks
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
                _state(tasks), tasks
            )

        assert handled == []
        assert unhandled == tasks

    asyncio.run(run_test())


def test_forge_master_translates_task_id_dependencies_to_batch_indices():
    """Planner-assigned dependencies use task_id references; the batch API
    addresses tasks by position. A dropped translation would let ForgeMaster
    run a dependent task concurrently with its prerequisite instead of
    waiting on it."""
    async def run_test():
        harness = _harness()
        tasks = [
            {"task_id": "t1", "description": "implement the function"},
            {"task_id": "t2", "description": "fix review feedback", "dependencies": ["t1"]},
        ]

        captured_tasks = {}

        async def fake_dispatch(fm_tasks, **kwargs):
            captured_tasks["fm_tasks"] = fm_tasks
            return {
                "success": True,
                "total": 2,
                "succeeded": 2,
                "results": [
                    {"success": True, "response": "impl done", "agent_used": "codex"},
                    {"success": True, "response": "fixed", "agent_used": "codex"},
                ],
            }

        with patch(
            "src.core.forge_master.tools._dispatch_batch_to_forge_master_tool",
            new=AsyncMock(side_effect=fake_dispatch),
        ):
            await harness._dispatch_codebase_tasks_via_forge_master(_state(tasks), tasks)

        # task_id "t1" is task index 0, so t2's dependency must translate to [0].
        assert captured_tasks["fm_tasks"][0].get("dependencies") is None
        assert captured_tasks["fm_tasks"][1]["dependencies"] == [0]

    asyncio.run(run_test())


def test_forge_master_drops_dependency_pointing_outside_the_batch():
    """A dependency on a task_id not present in this batch (e.g. it was
    already resolved by the Anvil engine) has no index to translate to and
    must be dropped rather than crash the lookup."""
    async def run_test():
        harness = _harness()
        tasks = [{"task_id": "t2", "description": "fix feedback", "dependencies": ["t_not_in_batch"]}]

        captured_tasks = {}

        async def fake_dispatch(fm_tasks, **kwargs):
            captured_tasks["fm_tasks"] = fm_tasks
            return {
                "success": True,
                "total": 1,
                "succeeded": 1,
                "results": [{"success": True, "response": "fixed", "agent_used": "codex"}],
            }

        with patch(
            "src.core.forge_master.tools._dispatch_batch_to_forge_master_tool",
            new=AsyncMock(side_effect=fake_dispatch),
        ):
            await harness._dispatch_codebase_tasks_via_forge_master(_state(tasks), tasks)

        assert captured_tasks["fm_tasks"][0].get("dependencies") is None

    asyncio.run(run_test())


def test_forge_master_handled_tasks_get_marked_completed_in_session_control():
    """The legacy ParallelAgentExecutor path reflects per-task completion into
    SessionControl via _update_session_tasks; ForgeMaster-handled tasks must
    get the same treatment or session task views keep showing them pending."""
    async def run_test():
        harness = _harness()
        tasks = [{"task_id": "t1", "description": "write an add function"}]
        state = _state(tasks)

        fake_batch_result = {
            "success": True,
            "total": 1,
            "succeeded": 1,
            "results": [{"success": True, "response": "def add(a,b): return a+b", "agent_used": "codex"}],
        }

        with patch(
            "src.core.forge_master.tools._dispatch_batch_to_forge_master_tool",
            new=AsyncMock(return_value=fake_batch_result),
        ), patch.object(
            harness, "_update_session_tasks"
        ) as mock_update:
            await harness._dispatch_codebase_tasks_via_forge_master(state, tasks)

        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args.args[0] is state
        assert call_args.args[1][0]["task_id"] == "t1"
        assert call_args.kwargs.get("status") == "completed"

    asyncio.run(run_test())


def test_forge_master_failed_tasks_do_not_trigger_session_control_update():
    async def run_test():
        harness = _harness()
        tasks = [{"task_id": "t1", "description": "write an add function"}]
        state = _state(tasks)

        fake_batch_result = {
            "success": False,
            "total": 1,
            "succeeded": 0,
            "results": [{"success": False, "error": "adversarial audit rejected"}],
        }

        with patch(
            "src.core.forge_master.tools._dispatch_batch_to_forge_master_tool",
            new=AsyncMock(return_value=fake_batch_result),
        ), patch.object(
            harness, "_update_session_tasks"
        ) as mock_update:
            await harness._dispatch_codebase_tasks_via_forge_master(state, tasks)

        mock_update.assert_not_called()

    asyncio.run(run_test())

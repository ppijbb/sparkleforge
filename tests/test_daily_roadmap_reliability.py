"""Regression tests for Daily Roadmap execution, routing, dashboard IDs, and synthesis."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.generator_agent import GeneratorAgent
from src.agents.planner_agent import PlannerAgent
from src.core.agent_harness import AgentHarness
from src.core.harness_state import create_initial_harness_state
from src.core.langgraph_checkpointer import build_sqlite_checkpointer
from src.core.llm_manager.routing import RoutingMixin
from src.core.llm_manager.types import ModelConfig, TaskType
from src.core.surface.task_dashboard import TaskDashboard


def test_planner_parse_tasks_ignores_brackets_in_surrounding_prose() -> None:
    planner = PlannerAgent()
    response = (
        "Here is the [plan] and {context} summary before the JSON:\n"
        "```json\n"
        '{"tasks": [{"task_id": "t1", "description": "Inspect docs/ANVIL_PLAN.md"}]}\n'
        "```\n"
        "Trailing note with [another bracket]."
    )
    tasks = planner._parse_tasks(response, "daily roadmap")
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "t1"


def test_routing_resolves_prefixed_explicit_model_and_prioritizes_llm_provider(monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("SPARKLEFORGE_EXPLICIT_MODEL", "google/gemini-3.5-flash-lite")

    order = RoutingMixin._build_provider_rotation_order()
    assert order[0] == "google"

    router = RoutingMixin()
    router.llm_config = SimpleNamespace(
        primary_model="google/gemini-3.5-flash-lite",
        planning_model="google/gemini-3.5-flash-lite",
        reasoning_model="google/gemini-3.5-flash-lite",
        verification_model="google/gemini-3.5-flash-lite",
        generation_model="google/gemini-3.5-flash-lite",
        compression_model="google/gemini-3.5-flash-lite",
        budget_limit=10.0,
    )
    router.models = {
        "gemini-flash-lite": ModelConfig(
            name="gemini-flash-lite",
            provider="google",
            model_id="gemini-3.5-flash-lite",
            temperature=0.1,
            max_tokens=4096,
            cost_per_token=0.0001,
            speed_rating=9.0,
            quality_rating=7.0,
            capabilities=[TaskType.PLANNING],
        ),
        "gemini-pro": ModelConfig(
            name="gemini-pro",
            provider="google",
            model_id="gemini-3.1-pro-preview",
            temperature=0.2,
            max_tokens=4096,
            cost_per_token=0.0005,
            speed_rating=6.0,
            quality_rating=9.0,
            capabilities=[TaskType.ANALYSIS, TaskType.DEEP_REASONING],
        ),
    }
    router.provider_rate_limited = {}
    router.provider_usage_count = {"google": 0, "nvidia": 0}
    router._initialize_dynamic_google_fallback = False  # noqa: SLF001

    assert router.select_model(TaskType.ANALYSIS) == "gemini-flash-lite"


@pytest.mark.asyncio
async def test_generator_agent_synthesizes_final_output_and_skips_non_artifact_python(tmp_path) -> None:
    agent = GeneratorAgent()
    state = create_initial_harness_state(session_id="s_gen", user_query="Generate daily roadmap")
    state["meta"]["output_dir"] = str(tmp_path)
    state["workflow"]["final_output"] = "Executed 1 tasks (0 via Anvil, 1 via Legacy)."
    state["workflow"]["tasks"] = [
        {
            "task_id": "task_1",
            "status": "completed",
            "result": {
                "content": (
                    "# Proposal\n\n## Why now\nSignal.\n\n"
                    "```python\nraise RuntimeError('illustrative snippet must not run')\n```"
                )
            },
        }
    ]

    fake_synth = SimpleNamespace(
        content="# Final Issue\n\n## Why now\nA\n\n## Proposed change\nB\n\n"
        "## Implementation notes\nC\n\n## Acceptance criteria\n- [ ] D\n\n## Validation\npytest"
    )
    with patch("src.core.llm_manager.execute_llm_task", new=AsyncMock(return_value=fake_synth)):
        res = await agent.synthesize(state)

    assert "## Why now" in res["final_output"]
    assert "Executed 1 tasks" not in res["final_output"]
    assert "Marp presentation:" not in res["final_output"]


@pytest.mark.asyncio
async def test_harness_legacy_dashboard_uses_task_record_id_not_object() -> None:
    harness = object.__new__(AgentHarness)
    harness.dashboard = TaskDashboard()
    harness.router = SimpleNamespace(assign_agent_for_task=AsyncMock(return_value="analyzer_agent"))
    harness.anvil_engine = SimpleNamespace(handler_registry={})
    harness.mode_controller = SimpleNamespace(mode=SimpleNamespace(value="research"))
    harness._enforce_session_control = AsyncMock()
    harness._register_session_tasks = MagicMock()
    harness._apply_mode_to_state = MagicMock()
    harness._guard_intent = AsyncMock()
    harness._update_session_tasks = MagicMock()
    harness._record_execution_signal = MagicMock()
    harness._update_session_progress = MagicMock()
    harness._capture_observation_snapshot = AsyncMock(return_value={})
    harness._update_token_budget = MagicMock()

    state = create_initial_harness_state(session_id="s_dash", user_query="test")
    state["workflow"]["route"] = "planner_parallel"
    state["workflow"]["tasks"] = [{"task_id": "t1", "description": "analyze", "status": "pending"}]

    fake_executor = MagicMock()
    fake_executor.execute_parallel_tasks = AsyncMock(
        return_value={"execution_results": [{"result": {"content": "ok"}, "status": "completed"}]}
    )

    with (
        patch("src.core.mcp_integration.get_mcp_hub", side_effect=RuntimeError("skip mcp")),
        patch("src.core.parallel_agent_executor.ParallelAgentExecutor", return_value=fake_executor),
    ):
        updated = await harness._node_executor(state)

    assert updated["workflow"]["tasks"][0]["status"] == "completed"
    records = harness.dashboard.list_tasks()
    assert len(records) == 1
    assert records[0].status.value == "success"


@pytest.mark.asyncio
async def test_build_sqlite_checkpointer_marks_worker_thread_daemon(tmp_path) -> None:
    saver = build_sqlite_checkpointer(str(tmp_path / "cp.db"))
    try:
        assert saver.conn._thread.daemon is True
    finally:
        await saver.conn.close()

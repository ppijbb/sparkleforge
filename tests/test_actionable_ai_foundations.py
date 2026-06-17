from __future__ import annotations

import asyncio
from pathlib import Path

from src.core.bootstrap_graph import BootstrapGraph, BootstrapStage
from src.core.execution_registry import (
    ExecutionRegistry,
    RegisteredTool,
    default_command_registry,
)
from src.core.prompt_router import PromptRouter
from src.core.scheduler import Scheduler
from src.core.trust_gate import TrustContext, TrustGate, TrustLevel


def test_scheduler_run_now_executes_callback(tmp_path: Path):
    scheduler = Scheduler(storage_path=tmp_path)

    async def callback(user_query: str, session_id: str):
        return {"query": user_query, "session_id": session_id}

    scheduler.set_execution_callback(callback)
    schedule = scheduler.create_schedule("daily", "* * * * *", "ping")

    execution = asyncio.run(scheduler.run_now(schedule.schedule_id))

    assert execution.status == "completed"
    assert execution.result["query"] == "ping"
    assert scheduler.get_schedule(schedule.schedule_id).run_count == 1
    assert scheduler.get_schedule(schedule.schedule_id).success_count == 1


def test_prompt_router_prefers_explicit_schedule_command():
    registry = ExecutionRegistry(
        commands=default_command_registry(),
        skills=(),
        tools=(),
        triggers=(),
    )
    router = PromptRouter()

    results = asyncio.run(
        router.route(
            "schedule list",
            registry,
            TrustContext.default(),
        )
    )

    assert results
    assert results[0].target == "schedule list"


def test_trust_gate_reads_project_policy(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SPARKLEFORGE_TRUST_LEVEL", raising=False)
    monkeypatch.delenv("SPARKLEFORGE_DENY_TOOLS", raising=False)
    monkeypatch.delenv("SPARKLEFORGE_DENY_PREFIXES", raising=False)
    monkeypatch.delenv("SPARKLEFORGE_ALLOWED_MCP_SERVERS", raising=False)

    (tmp_path / ".sparkleforge-trust").write_text(
        """
        {
          "level": "read_only",
          "deny_names": ["run_shell_command"],
          "deny_prefixes": ["git_"],
          "allowed_mcp_servers": ["arxiv"]
        }
        """.strip(),
        encoding="utf-8",
    )

    trust = asyncio.run(TrustGate(project_root=tmp_path).evaluate())

    assert trust.level == TrustLevel.READ_ONLY
    assert "run_shell_command" in trust.deny_names
    assert trust.deny_prefixes == ("git_",)
    assert trust.allowed_mcp_servers == frozenset({"arxiv"})
    assert trust.allows_tool("arxiv::search", "arxiv")
    assert not trust.allows_tool("run_shell_command")


def test_execution_registry_filters_tools_by_trust():
    registry = ExecutionRegistry(
        commands=(),
        skills=(),
        tools=(
            RegisteredTool(
                name="run_shell_command",
                description="Run shell",
                source="local",
            ),
            RegisteredTool(
                name="arxiv::search",
                description="Search papers",
                source="mcp",
                mcp_server="arxiv",
            ),
        ),
        triggers=(),
    )

    trust = TrustContext(
        level=TrustLevel.PARTIAL,
        deny_names=frozenset({"run_shell_command"}),
        deny_prefixes=(),
        allowed_mcp_servers=frozenset({"arxiv"}),
    )

    filtered = registry.filter_by_trust(trust)

    assert [tool.name for tool in filtered.tools] == ["arxiv::search"]


def test_bootstrap_graph_reports_stage_failures(tmp_path: Path):
    graph = BootstrapGraph(project_root=tmp_path)

    async def ok_stage():
        return {"ok": True}

    async def failing_stage():
        raise RuntimeError("boom")

    graph._default_stages = lambda: [  # type: ignore[method-assign]
        BootstrapStage("config", ok_stage),
        BootstrapStage("trust_gate", failing_stage, depends_on=("config",)),
    ]

    result = asyncio.run(graph.run())

    assert not result.ok
    assert result.stages[0].ok is True
    assert result.stages[1].ok is False
    assert result.stages[1].error == "boom"


def test_scheduler_timeout_validation_and_handling(tmp_path: Path):
    scheduler = Scheduler(storage_path=tmp_path)

    async def slow_callback(user_query: str, session_id: str):
        await asyncio.sleep(0.5)
        return "done"

    scheduler.set_execution_callback(slow_callback)

    # 1. Valid timeout that triggers timeout error
    schedule = scheduler.create_schedule(
        "slow_job", "* * * * *", "slow", timeout_seconds=1  # 1s is not enough if callback takes 0.5s but wait, let's use timeout_seconds=0.1
    )
    # Wait, let's set timeout_seconds to a small value so it times out
    schedule.timeout_seconds = 0.05
    execution = asyncio.run(scheduler.run_now(schedule.schedule_id))
    assert execution.status == "failed"
    assert "Timeout" in (execution.error or "")

    # 2. Invalid timeout value (infinity) - should be validated safely and run without timeout
    async def fast_callback(user_query: str, session_id: str):
        return "fast"
    scheduler.set_execution_callback(fast_callback)

    schedule_inf = scheduler.create_schedule(
        "inf_job", "* * * * *", "fast", timeout_seconds="inf"  # type: ignore[arg-type]
    )
    execution_inf = asyncio.run(scheduler.run_now(schedule_inf.schedule_id))
    assert execution_inf.status == "completed"
    assert execution_inf.result == "fast"

    # 3. Invalid timeout value (negative) - should be validated safely and run without timeout
    schedule_neg = scheduler.create_schedule(
        "neg_job", "* * * * *", "fast", timeout_seconds=-5
    )
    execution_neg = asyncio.run(scheduler.run_now(schedule_neg.schedule_id))
    assert execution_neg.status == "completed"
    assert execution_neg.result == "fast"

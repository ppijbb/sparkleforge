"""Issue #585: Heat -- timeboxed task execution with a wrap-up report.

SessionQuota.wall_clock_timeout_seconds / SessionControl.check_quotas exist
but were never actually wired into the real execution path (update_session_progress,
the only caller of check_quotas, has zero callers anywhere -- confirmed while
scoping this issue). The real, live-tracked autonomous execution loop is
AgentLoop.run_conversation()'s IterationBudget, so Heat's soft-deadline +
wrap-up-report behavior is built there instead: past HEAT_SOFT_DEADLINE_RATIO
of an optional heat_seconds budget, the loop stops starting new tool-calling
iterations and returns a summary of what was completed/failed, instead of
either running to iteration exhaustion or being cut off mid-tool-call.
"""

import pytest
from types import SimpleNamespace

from src.cli.main_commands import parse_heat_duration
from src.core.agent_loop import AgentLoop, IterationBudget
from src.core.llm_manager import ModelResult, TaskType


class NoopCompressor:
    async def compress_if_needed(self, history):
        return history

    async def compress_by_summarization(self, history):
        return history

    def prune_tool_output(self, output):
        return output


class NoopMemory:
    def get_context_block(self):
        return ""


class FakeMCPHub:
    def __init__(self):
        self.registry = SimpleNamespace(
            tools={
                "search": SimpleNamespace(
                    description="Search test data",
                    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                )
            }
        )
        self.calls = []

    async def initialize_mcp(self):
        return None

    async def execute_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return {"success": True, "data": {"answer": "tool evidence"}}


class FakeOrchestrator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.models = {
            "tool-model": SimpleNamespace(provider="openrouter", capabilities=[TaskType.RESEARCH])
        }

    async def execute_with_model(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def make_loop(orchestrator, mcp_hub):
    loop = AgentLoop.__new__(AgentLoop)
    loop.orchestrator = orchestrator
    loop.mcp_hub = mcp_hub
    loop.compressor = NoopCompressor()
    loop.memory = NoopMemory()
    return loop


def test_iteration_budget_heat_properties_none_when_unset():
    budget = IterationBudget(max_iterations=5)

    assert budget.heat_soft_expired is False
    assert budget.heat_hard_expired is False


def test_iteration_budget_heat_soft_expired_at_ratio(monkeypatch):
    import src.core.agent_loop as agent_loop_module

    # start_time is set explicitly (dataclass field default_factory binds the
    # real time.time at module-import time, so patching time.time afterward
    # cannot retroactively change an already-constructed default); elapsed()
    # itself does a fresh time.time() lookup, so patching it controls "now".
    monkeypatch.setattr(agent_loop_module.time, "time", lambda: 1000.0 + 90.0)
    budget = IterationBudget(max_iterations=5, heat_seconds=100.0, start_time=1000.0)

    assert budget.elapsed == 90.0
    assert budget.heat_soft_expired is True  # 90 >= 100 * 0.85
    assert budget.heat_hard_expired is False  # 90 < 100


def test_iteration_budget_heat_hard_expired_past_full_budget(monkeypatch):
    import src.core.agent_loop as agent_loop_module

    monkeypatch.setattr(agent_loop_module.time, "time", lambda: 1000.0 + 150.0)
    budget = IterationBudget(max_iterations=5, heat_seconds=100.0, start_time=1000.0)

    assert budget.heat_hard_expired is True


@pytest.mark.asyncio
async def test_run_conversation_without_heat_seconds_behaves_as_before():
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search", "arguments": '{"query": "sparkleforge"}'},
    }
    orchestrator = FakeOrchestrator(
        [
            ModelResult("", "tool-model", 0.1, 0.8, 0.0, {"tool_calls": [tool_call]}),
            ModelResult("final answer", "tool-model", 0.1, 0.8, 0.0, {}),
        ]
    )
    loop = make_loop(orchestrator, FakeMCPHub())

    result = await loop.run_conversation(
        [{"role": "user", "content": "research sparkleforge"}], max_iterations=3
    )

    assert result["success"] is True
    assert result["content"] == "final answer"
    assert "heat_report" not in result["metadata"]


@pytest.mark.asyncio
async def test_run_conversation_wraps_up_when_heat_soft_deadline_already_passed():
    # heat_seconds so small that by the time the loop checks heat_soft_expired
    # (a few microseconds after construction), it is already past the 85%
    # threshold -- deterministic without needing to mock time mid-loop.
    orchestrator = FakeOrchestrator([])
    loop = make_loop(orchestrator, FakeMCPHub())

    result = await loop.run_conversation(
        [{"role": "user", "content": "research sparkleforge"}],
        max_iterations=10,
        heat_seconds=1e-6,
    )

    assert result["success"] is True
    assert result["metadata"]["heat_expired"] is True
    report = result["metadata"]["heat_report"]
    assert report["completed"] == []
    assert report["failed"] == []
    assert "resume" in report["next_recommended_action"].lower()
    assert orchestrator.calls == []  # never even called the model


def test_build_heat_report_classifies_completed_and_failed():
    loop = make_loop(FakeOrchestrator([]), FakeMCPHub())
    budget = IterationBudget(max_iterations=5, heat_seconds=100.0)
    tool_results = [
        {"tool_name": "search", "success": True, "data": {"answer": "found it"}},
        {"tool_name": "write_file", "success": False, "error": "disk full"},
    ]

    report = loop._build_heat_report(budget, tool_results, errors=[])

    assert len(report["completed"]) == 1
    assert report["completed"][0]["tool"] == "search"
    assert len(report["failed"]) == 1
    assert report["failed"][0] == {"tool": "write_file", "error": "disk full"}
    assert "write_file" in report["next_recommended_action"]
    assert "disk full" in report["next_recommended_action"]


def test_build_heat_report_recommends_resume_when_all_succeeded():
    loop = make_loop(FakeOrchestrator([]), FakeMCPHub())
    budget = IterationBudget(max_iterations=5, heat_seconds=100.0)
    tool_results = [{"tool_name": "search", "success": True, "data": {}}]

    report = loop._build_heat_report(budget, tool_results, errors=[])

    assert "resume" in report["next_recommended_action"].lower()


class TestParseHeatDuration:
    def test_minutes(self):
        assert parse_heat_duration("30m") == 1800.0

    def test_hours(self):
        assert parse_heat_duration("1h") == 3600.0

    def test_seconds_suffix(self):
        assert parse_heat_duration("90s") == 90.0

    def test_bare_number_is_seconds(self):
        assert parse_heat_duration("45") == 45.0

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            parse_heat_duration("0m")

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_heat_duration("soon")

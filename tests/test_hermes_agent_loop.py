import pytest
from types import SimpleNamespace

from src.core.agent_loop import AgentLoop, IterationBudget
from src.core.autonomous_orchestrator import _autopilot_mode_enabled
from src.core.llm_manager import ModelResult, TaskType
from src.core.orchestrator.execution import ExecutionNode
from src.core.prompt_builder import get_system_prompt


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
    loop.overseer = None
    loop.greedy_overseer = None
    loop.mode_controller = None
    loop.method_resolver = None
    loop.intent_guardrail = None
    return loop


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_calls_until_final_answer():
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
    mcp_hub = FakeMCPHub()
    loop = make_loop(orchestrator, mcp_hub)

    result = await loop.run_conversation(
        [{"role": "user", "content": "research sparkleforge"}], max_iterations=3
    )

    assert result["success"] is True
    assert result["content"] == "final answer"
    assert result["tool_calls_count"] == 1
    assert result["tool_results"][0]["success"] is True
    assert mcp_hub.calls == [("search", {"query": "sparkleforge"})]
    assert orchestrator.calls[0]["model_name"] == "tool-model"


@pytest.mark.asyncio
async def test_agent_loop_records_invalid_tool_arguments_as_structured_error():
    tool_call = {
        "id": "call_bad",
        "type": "function",
        "function": {"name": "search", "arguments": "not-json"},
    }
    orchestrator = FakeOrchestrator(
        [
            ModelResult("", "tool-model", 0.1, 0.8, 0.0, {"tool_calls": [tool_call]}),
            ModelResult("recovered", "tool-model", 0.1, 0.8, 0.0, {}),
        ]
    )
    loop = make_loop(orchestrator, FakeMCPHub())

    result = await loop.run_conversation(
        [{"role": "user", "content": "research"}], max_iterations=3
    )

    assert result["success"] is True
    assert result["content"] == "recovered"
    assert result["errors"][0]["type"] == "invalid_tool_arguments"
    assert result["tool_results"][0]["success"] is False


@pytest.mark.asyncio
async def test_agent_loop_stops_on_repeated_identical_tool_calls():
    # Regression for issue #807: the benchmark's security_scan scenario had
    # the agent call git_status 12+ times in a row with identical arguments
    # until the iteration budget ceiling fired, wasting the whole budget on
    # no forward progress. The loop should detect this and bail out with a
    # stuck_loop reason well before iterations run out.
    repeated_call = {
        "id": "call_repeat",
        "type": "function",
        "function": {"name": "search", "arguments": '{"query": "same"}'},
    }
    orchestrator = FakeOrchestrator(
        [
            ModelResult("", "tool-model", 0.1, 0.8, 0.0, {"tool_calls": [repeated_call]})
            for _ in range(10)
        ]
    )
    mcp_hub = FakeMCPHub()
    loop = make_loop(orchestrator, mcp_hub)

    result = await loop.run_conversation(
        [{"role": "user", "content": "scan the repo"}], max_iterations=20
    )

    assert result["success"] is False
    assert result["metadata"]["error_category"] == "stuck_loop"
    assert result["errors"][-1]["type"] == "stuck_loop"
    # Terminated well before exhausting the iteration budget.
    assert result["iterations"] < 20
    # Only the first 3 identical calls actually executed; the 4th was
    # short-circuited instead of burning another round-trip.
    assert len(mcp_hub.calls) == 3


@pytest.mark.asyncio
async def test_agent_loop_allows_alternating_tool_calls_without_tripping_guard():
    # Non-repeating tool calls (even to the same tool with different args)
    # must not be mistaken for a stuck loop.
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": "search", "arguments": f'{{"query": "q{i}"}}'},
        }
        for i in range(4)
    ]
    orchestrator = FakeOrchestrator(
        [
            ModelResult("", "tool-model", 0.1, 0.8, 0.0, {"tool_calls": [c]})
            for c in calls
        ]
        + [ModelResult("done", "tool-model", 0.1, 0.8, 0.0, {})]
    )
    mcp_hub = FakeMCPHub()
    loop = make_loop(orchestrator, mcp_hub)

    result = await loop.run_conversation(
        [{"role": "user", "content": "research"}], max_iterations=10
    )

    assert result["success"] is True
    assert result["content"] == "done"
    assert len(mcp_hub.calls) == 4


@pytest.mark.asyncio
async def test_execution_node_uses_hermes_results_in_state(monkeypatch):
    agent_config = SimpleNamespace(max_concurrent_research_units=1)
    node = ExecutionNode(
        llm_config=SimpleNamespace(),
        agent_config=agent_config,
        research_depth=SimpleNamespace(adjust_depth_progressively=lambda *_: None),
        streaming_manager=SimpleNamespace(),
    )

    async def fake_hermes(task, state, max_iterations):
        return {
            "task_id": task["id"],
            "task_name": task["name"],
            "tool_used": "hermes_agent_loop",
            "result": "task result",
            "status": "completed",
            "iterations": 2,
            "agent_loop_metadata": {
                "success": True,
                "iterations": 2,
                "tool_calls_count": 3,
                "tool_results": [],
                "errors": [],
                "metadata": {},
            },
        }

    monkeypatch.setattr(node, "_execute_task_with_hermes", fake_hermes)
    state = {
        "user_request": "research request",
        "planned_tasks": [{"id": "t1", "name": "Task 1", "description": "Do research"}],
        "execution_plan": {"strategy": "sequential"},
        "max_iterations": 5,
        "innovation_stats": {},
    }

    result = await node.execute_research(state)

    assert result["execution_results"][0]["result"] == "task result"
    assert result["streaming_data"][0]["tool_calls_count"] == 3
    assert result["innovation_stats"]["hermes_execution_used"] is True
    assert result["innovation_stats"]["hermes_tasks_completed"] == 1
    assert result["innovation_stats"]["tool_calls_count"] == 3


def test_autopilot_defaults_to_autonomous(monkeypatch):
    monkeypatch.delenv("SPARKLEFORGE_AUTOPILOT_MODE", raising=False)

    assert _autopilot_mode_enabled({}) is True
    assert _autopilot_mode_enabled({"autopilot_mode": False}) is False


def test_problem_solving_prompt_does_not_invite_clarification():
    prompt = get_system_prompt("researcher")

    assert "Do not stop to ask the user for clarification" in prompt
    assert "ask for clarification within your thoughts" not in prompt


def test_agent_loop_adds_autonomous_contract():
    loop = make_loop(FakeOrchestrator([]), FakeMCPHub())

    system_message = loop._build_autonomous_system_message("Base")

    assert "Base" in system_message
    assert "Do not ask the user for clarification" in system_message
    assert "hard blocker" in system_message


@pytest.mark.asyncio
async def test_oversee_iteration_skips_research_overseer_for_coworker_tasks():
    loop = make_loop(FakeOrchestrator([]), FakeMCPHub())
    calls = []

    class FakeGreedyOverseer:
        async def evaluate_execution_results(self, state):
            calls.append(state)
            return {"overseer_decision": "proceed"}

    loop.greedy_overseer = FakeGreedyOverseer()
    budget = IterationBudget(max_iterations=5)

    await loop._oversee_iteration(budget, [], [], [], TaskType.GENERATION)
    assert calls == []

    await loop._oversee_iteration(budget, [], [], [], TaskType.RESEARCH)
    assert len(calls) == 1

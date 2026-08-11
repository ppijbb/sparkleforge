import pytest
import tracemalloc
from types import SimpleNamespace

from src.core.agent_loop import AgentLoop, IterationBudget
from src.core.anvil.mode_controller import ExecutionMode, ModeController
from src.core.autonomous_orchestrator import _autopilot_mode_enabled
from src.core.llm_manager import ModelResult, TaskType
from src.core.orchestrator.execution import ExecutionNode
from src.core.prompt_builder import get_system_prompt


class NoopCompressor:
    async def compress_if_needed(self, history):
        return history

    async def compress_if_needed_background(self, history):
        return history

    async def compress_by_summarization(self, history):
        return history

    def discard_pending_background_compaction(self):
        pass

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
async def test_overseer_branch_is_entered_during_loop_iteration():
    """Regression for issue #1208: the overseer branch in run_conversation must
    actually fire when an overseer is wired. Previously the loop guarded on
    ``self.overseer`` while the constructor assigned ``self.greedy_overseer``,
    so the branch was dead code at runtime."""
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

    overseer_calls = []

    class TrackingOverseer:
        async def evaluate_execution_results(self, state):
            overseer_calls.append(state)
            return {"overseer_decision": "proceed"}

    loop.overseer = TrackingOverseer()

    result = await loop.run_conversation(
        [{"role": "user", "content": "research sparkleforge"}], max_iterations=3
    )

    assert result["success"] is True
    assert len(overseer_calls) >= 1
    assert any(call["overseer_iterations"] >= 2 for call in overseer_calls)


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


class FakeAskUserOverseer:
    """Overseer stub that always decides ask_user, regardless of state."""

    def __init__(self):
        self.calls = 0

    async def evaluate_execution_results(self, state):
        self.calls += 1
        return {
            "overseer_decision": "ask_user",
            "overseer_evaluations": [{"reasoning": "quality too low, need guidance"}],
        }


@pytest.mark.asyncio
async def test_agent_loop_stops_and_surfaces_overseer_ask_user_decision():
    # Regression for issue #1300: GreedyOverseerAgent deciding "ask_user" used
    # to be logged into `errors` and then ignored -- the loop kept iterating
    # and calling the model instead of actually stopping for the human.
    orchestrator = FakeOrchestrator(
        [ModelResult("should not be reached", "tool-model", 0.1, 0.8, 0.0, {})]
    )
    loop = make_loop(orchestrator, FakeMCPHub())
    overseer = FakeAskUserOverseer()
    loop.overseer = overseer

    result = await loop.run_conversation(
        [{"role": "user", "content": "research"}], max_iterations=10
    )

    assert result["success"] is True
    assert result["metadata"]["overseer_decision"] == "ask_user"
    assert result["metadata"]["waiting_for_user"] is True
    assert "quality too low, need guidance" in result["content"]
    assert overseer.calls == 1
    assert orchestrator.calls == []


class FailingMCPHub:
    """Like FakeMCPHub, but every tool call raises."""

    def __init__(self):
        self.calls = []

    async def initialize_mcp(self):
        return None

    async def execute_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        raise RuntimeError("simulated tool failure")


@pytest.mark.asyncio
async def test_agent_loop_stops_when_mode_switches_to_hitl_collaborative():
    # Regression for issue #1337: ModeController switching AUTONOMOUS ->
    # HITL_COLLABORATIVE (e.g. after repeated tool failures) was logged
    # ("Execution mode switched: autonomous -> hitl_collaborative") but
    # nothing ever checked mode_controller.mode -- the loop just kept
    # iterating autonomously past the switch. It must actually stop before
    # the next autonomous tool call, the same way the overseer's ask_user
    # decision already does.
    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": "search", "arguments": f'{{"query": "q{i}"}}'},
        }
        for i in range(3)
    ]
    orchestrator = FakeOrchestrator(
        [ModelResult("", "tool-model", 0.1, 0.8, 0.0, {"tool_calls": [c]}) for c in tool_calls]
    )
    mcp_hub = FailingMCPHub()
    loop = make_loop(orchestrator, mcp_hub)
    loop.mode_controller = ModeController()  # default failure_threshold=3

    result = await loop.run_conversation(
        [{"role": "user", "content": "do something"}], max_iterations=10
    )

    assert loop.mode_controller.mode == ExecutionMode.HITL_COLLABORATIVE
    assert result["success"] is True
    assert result["metadata"]["execution_mode"] == "hitl_collaborative"
    assert result["metadata"]["waiting_for_user"] is True
    assert "Waiting for human input" in result["content"]
    # Exactly 3 failing tool calls tripped the switch; a 4th, now-blocked
    # autonomous iteration would have called the model and the tool again --
    # neither should have happened.
    assert len(mcp_hub.calls) == 3
    assert len(orchestrator.calls) == 3


@pytest.mark.asyncio
async def test_agent_loop_injects_momentum_nudge_when_only_re_reading_old_ground():
    # Regression for the lfdb dogfooding session (issue #1216, Anvil Phase Mu):
    # the model kept alternating between two already-read files turn after
    # turn -- never a single *consecutive* repeat (so MAX_STUCK_TOOL_REPEATS
    # never fires) but also never gathering anything new. The loop should
    # notice the run-wide stagnation and push the model toward a real action.
    call_a = {
        "id": "call_a",
        "type": "function",
        "function": {"name": "search", "arguments": '{"query": "a"}'},
    }
    call_b = {
        "id": "call_b",
        "type": "function",
        "function": {"name": "search", "arguments": '{"query": "b"}'},
    }
    orchestrator = FakeOrchestrator(
        [
            ModelResult("", "tool-model", 0.1, 0.8, 0.0, {"tool_calls": [call_a]}),
            ModelResult("", "tool-model", 0.1, 0.8, 0.0, {"tool_calls": [call_b]}),
            ModelResult("", "tool-model", 0.1, 0.8, 0.0, {"tool_calls": [call_a]}),
            ModelResult("", "tool-model", 0.1, 0.8, 0.0, {"tool_calls": [call_b]}),
            ModelResult("done", "tool-model", 0.1, 0.8, 0.0, {}),
        ]
    )
    loop = make_loop(orchestrator, FakeMCPHub())

    result = await loop.run_conversation(
        [{"role": "user", "content": "research"}], max_iterations=10
    )

    assert result["success"] is True
    momentum_messages = [
        m
        for m in result["history"]
        if m.get("role") == "system" and "Momentum check:" in str(m.get("content", ""))
    ]
    assert len(momentum_messages) == 1


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
async def test_record_resolved_capability_gates_on_tool_success():
    loop = make_loop(FakeOrchestrator([]), FakeMCPHub())

    class FakeResolvedMethod:
        def __init__(self, resolved):
            self.resolved = resolved

    class FakeMethodResolver:
        def __init__(self, resolved):
            self._resolved = resolved

        async def resolve(self, capability):
            return FakeResolvedMethod(self._resolved)

    class FakeModeController:
        def __init__(self):
            self.unresolved_calls = []

        def on_unresolved_capability(self, capability):
            self.unresolved_calls.append(capability)

    # Registered tool (resolver finds it) that ran and reported success:
    # no HITL escalation.
    loop.method_resolver = FakeMethodResolver(resolved=True)
    loop.mode_controller = FakeModeController()
    await loop._record_resolved_capability("search", success=True)
    assert loop.mode_controller.unresolved_calls == []

    # Unknown tool: execute_tool returned {"success": False} without raising,
    # and the resolver can't find a handler either -- must still escalate.
    loop.method_resolver = FakeMethodResolver(resolved=False)
    loop.mode_controller = FakeModeController()
    await loop._record_resolved_capability("no_such_tool", success=False)
    assert loop.mode_controller.unresolved_calls == ["no_such_tool"]

    # Registered tool that failed at runtime (success=False) must not be
    # silently swallowed just because the resolver can find a handler for it.
    loop.method_resolver = FakeMethodResolver(resolved=True)
    loop.mode_controller = FakeModeController()
    await loop._record_resolved_capability("search", success=False)
    assert loop.mode_controller.unresolved_calls == ["search"]


@pytest.mark.asyncio
async def test_oversee_iteration_skips_research_overseer_for_coworker_tasks():
    loop = make_loop(FakeOrchestrator([]), FakeMCPHub())
    calls = []

    class FakeGreedyOverseer:
        async def evaluate_execution_results(self, state):
            calls.append(state)
            return {"overseer_decision": "proceed"}

    loop.overseer = FakeGreedyOverseer()
    budget = IterationBudget(max_iterations=5)

    await loop._oversee_iteration(budget, [], [], [], 0, TaskType.GENERATION)
    assert calls == []

    await loop._oversee_iteration(budget, [], [], [], 0, TaskType.RESEARCH)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_tool_signature_tracking_stays_bounded_across_many_iterations():
    # Regression for issue #1309: the per-iteration tool-call signature window
    # must not grow unboundedly as iterations accumulate unique arguments.
    from collections import deque

    from src.core.agent_loop import SIGNATURE_WINDOW_SIZE

    window: deque[set] = deque(maxlen=SIGNATURE_WINDOW_SIZE)
    for i in range(10_000):
        if not window:
            window.append(set())
        window[-1].add(("search", f'{{"query": "unique-{i}"}}'))
        # Simulate iteration rollover by starting a fresh window entry each
        # iteration; deque(maxlen=...) evicts the oldest automatically.
        window.append(set())

    total_entries = sum(len(s) for s in window)
    assert total_entries <= SIGNATURE_WINDOW_SIZE

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    window2: deque[set] = deque(maxlen=SIGNATURE_WINDOW_SIZE)
    for i in range(10_000):
        window2.append({("search", f'{{"query": "unique-{i}"}}')})
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    window_growth = sum(
        stat.size_diff
        for stat in stats
        if "test_hermes_agent_loop" in stat.traceback[0].filename
    )
    # The deque is bounded by SIGNATURE_WINDOW_SIZE entries regardless of how
    # many iterations ran; memory must not scale with iteration count.
    assert window_growth < 1_000_000
    assert len(window2) == SIGNATURE_WINDOW_SIZE

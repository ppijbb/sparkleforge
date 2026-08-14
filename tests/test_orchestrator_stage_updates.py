"""_invoke_with_stage_updates() must behave like graph.ainvoke() (same final
state) while additionally printing each node name as it completes, so a
multi-stage run isn't a silent black box (see item 4: visible pipeline
stage tracker)."""

import asyncio
from typing import TypedDict
from unittest.mock import patch

from langgraph.graph import END, StateGraph

from src.core.autonomous_orchestrator import AutonomousOrchestrator


class _State(TypedDict):
    x: int
    log: list


def _step_a(state):
    return {"x": state["x"] + 1, "log": state["log"] + ["a"]}


def _step_b(state):
    return {"x": state["x"] + 10, "log": state["log"] + ["b"]}


class _FakeOrchestrator(AutonomousOrchestrator):
    """Holds just the `.graph` attribute _invoke_with_stage_updates() needs,
    avoiding AutonomousOrchestrator.__init__'s heavy config/agent wiring."""

    def __init__(self):
        g = StateGraph(_State)
        g.add_node("step_a", _step_a)
        g.add_node("step_b", _step_b)
        g.set_entry_point("step_a")
        g.add_edge("step_a", "step_b")
        g.add_edge("step_b", END)
        self.graph = g.compile()

        # Bypass AutonomousOrchestrator.__init__'s heavy config/agent wiring.


def test_final_state_matches_ainvoke_when_values_yielded_last():
    fake = _FakeOrchestrator()
    input_state = {"x": 0, "log": []}

    with patch("sys.stdout.isatty", return_value=False):
        streamed = asyncio.run(_invoke_with_stage_updates(fake, input_state, {}))
    direct = asyncio.run(fake.graph.ainvoke({"x": 0, "log": []}, {}))

    assert streamed == direct == {"x": 11, "log": ["a", "b"]}


def test_prints_each_node_name_when_a_tty(capsys):
    fake = _FakeOrchestrator()

    with patch("sys.stdout.isatty", return_value=True):
        asyncio.run(fake._invoke_with_stage_updates({"x": 0, "log": []}, {}))

    out = capsys.readouterr().out
    assert "→ step_a" in out
    assert "→ step_b" in out


def test_silent_when_not_a_tty(capsys):
    fake = _FakeOrchestrator()

    with patch("sys.stdout.isatty", return_value=False):
        asyncio.run(fake._invoke_with_stage_updates({"x": 0, "log": []}, {}))

    out = capsys.readouterr().out
    assert "step_a" not in out
    assert "step_b" not in out

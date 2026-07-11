"""Regression tests for the disk-backed LangGraph checkpointer.

Covers issue #491: the orchestrator graph previously compiled with no
checkpointer at all, so a crash mid-run lost all progress with no way to
resume. These tests assert that a run surviving a simulated process
restart (a brand-new checkpointer instance pointed at the same on-disk
sqlite file) can see the state from before the crash and resume from the
last completed node rather than starting over.
"""

import os
from typing import TypedDict

import pytest
from langgraph.graph import END, StateGraph

from src.core.langgraph_checkpointer import build_sqlite_checkpointer


class _State(TypedDict):
    counter: int
    should_fail: bool


def _step_one(state: _State) -> dict:
    return {"counter": state["counter"] + 1}


def _step_two(state: _State) -> dict:
    if state.get("should_fail"):
        raise RuntimeError("simulated crash mid-execution")
    return {"counter": state["counter"] + 10}


def _build_graph(checkpointer):
    workflow = StateGraph(_State)
    workflow.add_node("step_one", _step_one)
    workflow.add_node("step_two", _step_two)
    workflow.set_entry_point("step_one")
    workflow.add_edge("step_one", "step_two")
    workflow.add_edge("step_two", END)
    return workflow.compile(checkpointer=checkpointer)


async def test_sqlite_checkpointer_survives_simulated_process_restart(tmp_path):
    db_path = str(tmp_path / "checkpoints.db")
    config = {"configurable": {"thread_id": "resume-test"}}

    # First "process": run until step_two crashes, after step_one has
    # already been checkpointed.
    graph_before_crash = _build_graph(build_sqlite_checkpointer(db_path))
    with pytest.raises(RuntimeError):
        await graph_before_crash.ainvoke({"counter": 0, "should_fail": True}, config)

    # Simulate a fresh process: an independent checkpointer/graph instance
    # pointed at the same sqlite file, as would happen after a restart.
    graph_after_restart = _build_graph(build_sqlite_checkpointer(db_path))
    state = await graph_after_restart.aget_state(config)
    assert state.values["counter"] == 1  # step_one's result survived the crash

    # Resume from the last completed node instead of starting over.
    await graph_after_restart.aupdate_state(config, {"should_fail": False})
    final = await graph_after_restart.ainvoke(None, config)
    assert final["counter"] == 11


async def test_build_sqlite_checkpointer_creates_parent_directory(tmp_path):
    db_path = str(tmp_path / "nested" / "dir" / "checkpoints.db")
    checkpointer = build_sqlite_checkpointer(db_path)
    assert checkpointer is not None
    # The parent directory is created eagerly; the sqlite file itself is
    # created lazily on first use (aiosqlite defers the real connect()).
    assert os.path.isdir(os.path.dirname(db_path))

    graph = _build_graph(checkpointer)
    await graph.ainvoke(
        {"counter": 0, "should_fail": False}, {"configurable": {"thread_id": "t"}}
    )
    assert os.path.exists(db_path)

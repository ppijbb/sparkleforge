"""QA for Langfuse ops hardening: trace context, token savings snapshot/delta, post-task report.

Verification order (QA convergence):
1. Lint: ruff on src/core/input_router.py, observability.py, session_lane.py,
   context_mode/stats.py, context_compaction/manager.py, agent_orchestrator.py, task_monitoring.py.
2. Unit: trace context ensure_trace_context, get_trace_context, set_trace_context.
3. Unit: SessionStats.snapshot() and .delta() (context_mode.stats).
4. Unit: get_current_turn_compaction_savings / get_current_turn_tool_savings (clear after use).
5. Integration: run workflow to completion and assert state has token_savings_report (or final_report contains savings line).
6. Run: pytest tests/test_langfuse_ops_qa.py -v
"""

from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(project_root))

from src.core.context_compaction.manager import (
    clear_current_turn_compaction_savings,
    get_current_turn_compaction_savings,
)
from src.core.context_mode.stats import (
    SessionStatsSnapshot,
    clear_current_turn_tool_savings,
    get_current_turn_tool_savings,
    get_session_stats,
    record_tool_context_savings,
    reset_session_stats,
)
from src.core.input_router import (
    TRACE_ENTRYPOINT,
    TRACE_REQUEST_ID,
    TRACE_TURN_ID,
    InputEnvelope,
    ensure_trace_context,
    get_trace_context,
    normalize_message,
    set_trace_context,
)


def test_ensure_trace_context_sets_turn_and_request_id():
    """ensure_trace_context sets request_id and turn_id in envelope.metadata."""
    env = normalize_message("s1", "hello")
    ensure_trace_context(env)
    assert env.metadata is not None
    assert env.metadata.get(TRACE_REQUEST_ID)
    assert env.metadata.get(TRACE_TURN_ID)
    assert env.metadata.get(TRACE_ENTRYPOINT) == "message"
    assert env.metadata.get("source_type") == "message"


def test_trace_context_contextvar():
    """set_trace_context / get_trace_context roundtrip."""
    set_trace_context(None)
    assert get_trace_context() is None
    set_trace_context({TRACE_TURN_ID: "t1", TRACE_REQUEST_ID: "r1"})
    ctx = get_trace_context()
    assert ctx is not None
    assert ctx.get(TRACE_TURN_ID) == "t1"
    set_trace_context(None)


def test_session_stats_snapshot_delta():
    """SessionStats.snapshot() and .delta() for task-level savings."""
    reset_session_stats()
    stats = get_session_stats()
    stats.track_response("tool_a", 1000)
    snap = stats.snapshot()
    assert isinstance(snap, SessionStatsSnapshot)
    assert snap.total_bytes_returned == 1000
    assert snap.total_calls == 1

    stats.track_response("tool_a", 500)
    stats.track_response("tool_b", 2000)
    delta = stats.delta(snap)
    assert delta.bytes_returned_delta == 2500
    assert delta.calls_delta == 2


def test_tool_savings_record_and_clear():
    """record_tool_context_savings and get/clear_current_turn_tool_savings."""
    clear_current_turn_tool_savings()
    record_tool_context_savings("t1", 10000, 500, turn_id="turn-x")
    record_tool_context_savings("t2", 5000, 5000, turn_id="turn-x")
    lst = get_current_turn_tool_savings()
    assert len(lst) == 2
    assert lst[0]["tool_name"] == "t1"
    assert lst[0]["kept_out_bytes"] == 9500
    assert lst[1]["kept_out_bytes"] == 0
    clear_current_turn_tool_savings()
    assert len(get_current_turn_tool_savings()) == 0


def test_compaction_savings_get_clear():
    """get_current_turn_compaction_savings and clear (no side effects)."""
    clear_current_turn_compaction_savings()
    lst = get_current_turn_compaction_savings()
    assert lst == []
    clear_current_turn_compaction_savings()

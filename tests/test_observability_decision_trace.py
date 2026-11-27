"""Unit tests for DecisionTraceRecorder and session trace inspection."""

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

import src.core.observability as observability
from src.cli.commands.session import inspect_session, session_inspect_command
from src.core.observability import DecisionTraceRecorder


def test_observability_exports_decision_trace_recorder_and_langfuse_helpers() -> None:
    assert hasattr(observability, "DecisionTraceRecorder")
    assert "class DecisionTraceRecorder" not in (observability.__doc__ or "")
    assert hasattr(observability, "get_langfuse_run_config")
    assert hasattr(observability, "start_turn_trace")
    assert hasattr(observability, "track_harness_transition")


def test_decision_trace_recorder_records_summarizes_and_saves(tmp_path: Path) -> None:
    trace_file = tmp_path / "traces" / "sess-1.jsonl"
    recorder = DecisionTraceRecorder("sess-1", log_path=str(trace_file))

    t0 = recorder.record_turn(
        turn_index=0,
        context_slice_ids=["slice-a", "slice-b"],
        candidate_scores={"web_search": 0.92, "shell": 0.15},
        rejection_reasons={"shell": "low_relevance"},
        latency_ms=120.5,
        tokens_in=500,
        tokens_out=100,
        layer_breakdown={"embedding": 20.5, "llm": 80.0, "tool": 20.0},
    )
    assert t0["turn_index"] == 0
    assert len(recorder.traces) == 1
    assert not trace_file.exists()

    saved_path = recorder.save()
    assert saved_path == str(trace_file)
    assert trace_file.exists()

    lines = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["candidate_scores"]["web_search"] == 0.92

    summary = recorder.get_flamegraph_summary()
    assert summary == {
        "session_id": "sess-1",
        "total_latency_ms": 120.5,
        "layers": {"embedding": 20.5, "llm": 80.0, "tool": 20.0},
        "turns_count": 1,
    }


def test_decision_trace_recorder_auto_flush_and_inspect_session(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    trace_file = trace_dir / "sess-2.jsonl"
    recorder = DecisionTraceRecorder("sess-2", log_path=str(trace_file), auto_flush=True)

    recorder.record_turn(
        turn_index=1,
        context_slice_ids=["ctx-1"],
        candidate_scores={"mcp_read": 0.88},
        rejection_reasons={"bash": "disallowed"},
        latency_ms=0.5,
        tokens_in=40,
        tokens_out=10,
        layer_breakdown={"mcp": 0.25, "llm": 0.25},
    )

    rendered = inspect_session("sess-2", trace_dir=str(trace_dir))
    assert "SparkleForge Flamegraph & Decision Tree Inspector: sess-2" in rendered
    assert "Total Wall Time: 0.50 ms across 1 turns" in rendered
    assert "mcp: 0.25 ms (50.0%)" in rendered
    assert "llm: 0.25 ms (50.0%)" in rendered
    assert "context_slices: ctx-1" in rendered
    assert "candidate_scores: mcp_read=0.88" in rendered
    assert "rejections: bash: disallowed" in rendered


def test_inspect_session_handles_missing_and_invalid_session_id(tmp_path: Path) -> None:
    missing = inspect_session("does-not-exist", trace_dir=str(tmp_path))
    assert "No decision trace found for session does-not-exist" in missing

    invalid = inspect_session("../escape", trace_dir=str(tmp_path))
    assert "Invalid session ID" in invalid


@pytest.mark.asyncio
async def test_session_inspect_command_prints_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    recorder = DecisionTraceRecorder("sess-cli", auto_flush=True)
    recorder.record_turn(
        turn_index=0,
        context_slice_ids=[],
        candidate_scores={},
        rejection_reasons={},
        latency_ms=10.0,
        tokens_in=5,
        tokens_out=2,
        layer_breakdown={"llm": 10.0},
    )

    buffer = io.StringIO()

    class _Cli:
        def __init__(self) -> None:
            self.console = Console(file=buffer, force_terminal=False, width=120)

    cli = _Cli()
    await session_inspect_command(cli, ["sess-cli"])
    out = buffer.getvalue()
    assert "sess-cli" in out
    assert "llm: 10.00 ms (100.0%)" in out

    buffer_usage = io.StringIO()
    cli.console = Console(file=buffer_usage, force_terminal=False, width=120)
    await session_inspect_command(cli, [])
    assert "Usage: session inspect" in buffer_usage.getvalue()

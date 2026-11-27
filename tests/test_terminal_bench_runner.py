from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.run_terminal_bench import run_task, CURATED_TASKS, main as terminal_bench_main
from scripts.generate_benchmark_report import load_terminal_bench_runs, render_block
from tests.benchmark.benchmark_metrics import calculate_metrics


def test_terminal_bench_dry_run(monkeypatch, tmp_path):
    history_file = tmp_path / "terminal_bench_history.jsonl"
    monkeypatch.setattr("sys.argv", ["run_terminal_bench.py", "--dry-run", "--history-out", str(history_file)])
    rc = terminal_bench_main()
    assert rc == 0
    assert history_file.exists()
    lines = history_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(CURATED_TASKS)
    record = json.loads(lines[0])
    assert "task_id" in record
    assert "resolved" in record
    assert "input_tokens" in record
    assert "output_tokens" in record
    assert "estimated_cost_usd" in record
    assert "duration_s" in record


def test_run_task_timeout(monkeypatch):
    task = CURATED_TASKS[0].copy()
    task["max_wall_clock_seconds"] = 0.001  # Force immediate timeout
    res = run_task(task, dry_run=False)
    assert res["resolved"] is False


def test_report_generation_with_terminal_bench():
    runs = [
        {
            "task_id": "tb-sys-clean-01",
            "resolved": True,
            "input_tokens": 1200,
            "output_tokens": 300,
            "estimated_cost_usd": 0.002,
            "duration_s": 1.5,
            "model_id": "gemini-3.1-flash-lite-preview",
        }
    ]
    block = render_block(None, [], None, terminal_bench_runs=runs)
    assert "Terminal-Bench Cost-Normalized Pareto Metrics" in block
    assert "gemini-3.1-flash-lite-preview" in block

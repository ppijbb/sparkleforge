"""Issue #702: offline aggregator for release metrics across multiple runs.

generate_daily_report() only ever appends one entry per day to
results/agent_reports/history.json; previously there was no way to see
performance across a range of runs (e.g. "since the last release") without
reading every row by hand. Covers aggregate_release_metrics() and the
`report aggregate` CLI subcommand.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from rich.panel import Panel

from src.core.monitoring.report_generator import aggregate_release_metrics
from src.cli.commands.report import report_command


def test_aggregate_release_metrics_empty_history():
    summary = aggregate_release_metrics([])

    assert summary["entry_count"] == 0
    assert summary["date_range"] is None
    assert summary["total_attempts"] == 0
    assert summary["average_strict_score"] == 0.0
    assert summary["weighted_success_rate"] == 0.0


def test_aggregate_release_metrics_sums_and_weights_correctly():
    history = [
        {"date": "2026-07-14", "strict_score": 80.0, "total_attempts": 4, "success_rate": 50.0, "total_marks": 1},
        {"date": "2026-07-15", "strict_score": 100.0, "total_attempts": 6, "success_rate": 100.0, "total_marks": 3},
    ]

    summary = aggregate_release_metrics(history)

    assert summary["entry_count"] == 2
    assert summary["date_range"] == ("2026-07-14", "2026-07-15")
    assert summary["total_attempts"] == 10
    assert summary["total_marks"] == 4
    assert summary["average_strict_score"] == 90.0
    # weighted: (50*4 + 100*6) / 10 = 80.0
    assert summary["weighted_success_rate"] == 80.0


def test_aggregate_release_metrics_ignores_zero_attempt_days_in_weighting():
    history = [
        {"date": "2026-07-14", "strict_score": 0.0, "total_attempts": 0, "success_rate": 0.0, "total_marks": 0},
        {"date": "2026-07-15", "strict_score": 100.0, "total_attempts": 2, "success_rate": 100.0, "total_marks": 2},
    ]

    summary = aggregate_release_metrics(history)

    assert summary["total_attempts"] == 2
    assert summary["weighted_success_rate"] == 100.0


def test_report_aggregate_command_prints_summary(tmp_path, monkeypatch):
    reports_dir = tmp_path / "results" / "agent_reports"
    reports_dir.mkdir(parents=True)
    history_file = reports_dir / "history.json"
    history_file.write_text(
        json.dumps([{"date": "2026-07-14", "strict_score": 80.0, "total_attempts": 4, "success_rate": 50.0, "total_marks": 1}]),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    printed = []
    cli = SimpleNamespace(console=SimpleNamespace(print=lambda *a, **k: printed.append(a)))

    asyncio.run(report_command(cli, ["aggregate"]))

    panels = [a for call in printed for a in call if isinstance(a, Panel)]
    assert len(panels) == 1
    assert panels[0].title == "Release Metrics Summary"
    assert "80.0" in panels[0].renderable


def test_report_aggregate_command_handles_missing_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    printed = []
    cli = SimpleNamespace(console=SimpleNamespace(print=lambda *a, **k: printed.append(a)))

    asyncio.run(report_command(cli, ["aggregate"]))

    assert any("No report history found" in str(a) for call in printed for a in call)

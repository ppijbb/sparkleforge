"""Tests for the frontier-equivalent cost ticker's data aggregation (issue #1546)."""

from __future__ import annotations

from src.web import live_dashboard


def test_fetch_cost_metrics_excludes_rows_missing_created_at(monkeypatch):
    rows = [{"metadata": {"estimated_cost": 1.0, "token_usage": {}}}]  # no created_at
    monkeypatch.setattr(live_dashboard, "_safe_select", lambda *a, **kw: rows)

    metrics = live_dashboard._fetch_cost_metrics(days=7)

    assert metrics["request_count"] == 0


def test_fetch_cost_metrics_excludes_rows_with_unparseable_created_at(monkeypatch):
    rows = [
        {
            "created_at": "not-a-date",
            "metadata": {"estimated_cost": 1.0, "token_usage": {}},
        }
    ]
    monkeypatch.setattr(live_dashboard, "_safe_select", lambda *a, **kw: rows)

    metrics = live_dashboard._fetch_cost_metrics(days=7)

    assert metrics["request_count"] == 0


def test_fetch_cost_metrics_includes_row_within_window(monkeypatch):
    from datetime import datetime, timezone

    rows = [
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "estimated_cost": 1.0,
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 100},
            },
        }
    ]
    monkeypatch.setattr(live_dashboard, "_safe_select", lambda *a, **kw: rows)

    metrics = live_dashboard._fetch_cost_metrics(days=7)

    assert metrics["request_count"] == 1
    assert metrics["total_actual_cost"] == 1.0

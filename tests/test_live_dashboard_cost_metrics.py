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


def test_fetch_cost_metrics_zero_tokens_yields_none_savings_pct_not_zero_division(monkeypatch):
    from datetime import datetime, timezone

    rows = [
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            # estimated_cost logged but no token_usage -- request_count > 0
            # while total_frontier_equivalent stays 0.
            "metadata": {"estimated_cost": 1.0, "token_usage": {}},
        }
    ]
    monkeypatch.setattr(live_dashboard, "_safe_select", lambda *a, **kw: rows)

    metrics = live_dashboard._fetch_cost_metrics(days=7)

    assert metrics["request_count"] == 1
    assert metrics["total_frontier_equivalent"] == 0.0
    assert metrics["savings_pct"] is None


def test_render_cost_ticker_does_not_crash_when_savings_pct_is_none():
    metrics = {
        "request_count": 1,
        "total_actual_cost": 1.0,
        "total_frontier_equivalent": 0.0,
        "savings_pct": None,
        "frontier_model": "claude-opus-4",
    }
    live_dashboard._render_cost_ticker(metrics)  # must not raise TypeError

"""Public live telemetry Streamlit dashboard for SparkleForge.

Pulls real-time metrics from Supabase tables populated by
``src/utils/supabase_exporter.py`` and renders them for the community.
Run with::

    streamlit run src/web/live_dashboard.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import streamlit as st

from src.utils.supabase_exporter import (
    DEFAULT_FRONTIER_MODEL,
    frontier_equivalent_cost_usd,
    get_supabase_client,
)

logger = logging.getLogger(__name__)

DASHBOARD_TITLE = "SparkleForge Live Telemetry"
DASHBOARD_URL = "https://sparkleforge.streamlit.app"

# Baseline metrics surfaced in the issue. These are shown when Supabase is
# unreachable so the public dashboard remains informative.
FALLBACK_METRICS: Dict[str, Any] = {
    "mttm_minutes": 141.08,
    "auto_merge_rate": 0.667,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_select(table: str, columns: str = "*", limit: int = 100) -> List[Dict[str, Any]]:
    """Best-effort synchronous Supabase select; returns [] on any failure."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        response = client.table(table).select(columns).limit(limit).execute()
        return list(response.data or [])
    except Exception as exc:  # pragma: no cover - dashboard must stay live
        logger.warning("Supabase select on %s failed: %s", table, exc)
        return []


async def _safe_select_async(table: str, columns: str = "*", limit: int = 100) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(_safe_select, table, columns, limit)


def _fetch_metrics() -> Dict[str, Any]:
    """Aggregate live metrics from Supabase, falling back to baselines."""
    metrics: Dict[str, Any] = dict(FALLBACK_METRICS)
    metrics["source"] = "baseline"

    summary_rows = _safe_select("telemetry_summary", limit=1)
    if summary_rows:
        row = summary_rows[0]
        if "mttm_minutes" in row:
            metrics["mttm_minutes"] = float(row["mttm_minutes"])
        if "auto_merge_rate" in row:
            metrics["auto_merge_rate"] = float(row["auto_merge_rate"])
        metrics["source"] = "supabase"

    return metrics


def _fetch_agent_logs(limit: int = 25) -> List[Dict[str, Any]]:
    """Fetch streaming agent execution step journal entries."""
    rows = _safe_select("agent_execution_logs", limit=limit)
    if not rows:
        return []
    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return rows[:limit]


def _fetch_jobs(limit: int = 25) -> List[Dict[str, Any]]:
    rows = _safe_select("forge_jobs", limit=limit)
    if not rows:
        return []
    rows.sort(key=lambda r: r.get("updated_at", r.get("created_at", "")), reverse=True)
    return rows[:limit]


def _fetch_ci_token_metrics() -> Dict[str, Any]:
    rows = _safe_select("sparkleforge_history_events", "metadata", limit=1000)
    total_tokens = 0
    total_cost = 0.0
    for r in rows:
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            import json
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        usage = meta.get("token_usage") or {}
        total_tokens += usage.get("total_tokens", 0)
        total_cost += meta.get("estimated_cost", 0.0)
    return {"total_tokens": total_tokens, "total_cost_usd": total_cost}

def _render_ci_cost_widget() -> None:
    st.subheader("🤖 CI Auto-Fix Cumulative Cost")
    metrics = _fetch_ci_token_metrics()
    col1, col2 = st.columns(2)
    col1.metric("Total CI Tokens Consumed", f"{metrics['total_tokens']:,}")
    col2.metric("Total Estimated LLM Cost", f"${metrics['total_cost_usd']:.4f}")


def _fetch_cost_metrics(days: int = 7) -> Dict[str, Any]:
    """Aggregate actual vs frontier-equivalent $ cost from logged LLM calls.

    limit=2000 is a pragmatic bound for a 7-day informational ticker, not a
    correctness requirement -- consistent with _fetch_ci_token_metrics'
    all-time limit=1000 above; neither paginates.
    """
    rows = _safe_select("sparkleforge_history_events", "metadata,created_at", limit=2000)
    if len(rows) >= 2000:
        logger.warning(
            "_fetch_cost_metrics hit its row limit (2000); the %d-day cost ticker "
            "may be undercounting.",
            days,
        )
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    total_actual = 0.0
    total_frontier = 0.0
    request_count = 0
    for r in rows:
        created_at = r.get("created_at")
        if not created_at:
            continue
        try:
            # Python 3.11+ fromisoformat() parses a "Z" suffix natively
            # (this project requires >=3.11); no manual "+00:00" swap needed.
            if datetime.fromisoformat(created_at) < cutoff:
                continue
        except ValueError:
            # Can't verify this row is inside the 7-day window -- exclude it
            # rather than risk counting stale/out-of-window cost data.
            continue
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        actual_cost = meta.get("estimated_cost")
        if actual_cost is None:
            continue
        usage = meta.get("token_usage") or {}
        total_actual += actual_cost
        total_frontier += frontier_equivalent_cost_usd(
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        )
        request_count += 1

    savings_pct = (1.0 - (total_actual / total_frontier)) if total_frontier > 0 else None
    return {
        "request_count": request_count,
        "total_actual_cost": total_actual,
        "total_frontier_equivalent": total_frontier,
        "savings_pct": savings_pct,
        "frontier_model": DEFAULT_FRONTIER_MODEL,
    }


def _format_usd(value: float) -> str:
    if value >= 1:
        return f"${value:,.2f}"
    if value >= 0.01:
        return f"${value:.4f}"
    return f"${value:.6f}"


def _render_cost_ticker(cost_metrics: Dict[str, Any]) -> None:
    st.subheader("💰 Frontier-Equivalent Cost Ticker (Last 7 Days)")
    if cost_metrics["request_count"] == 0:
        st.info("No cost metrics logged yet. Run tasks to see the frontier-equivalent comparison.")
        return
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(
        "Actual Spend",
        _format_usd(cost_metrics["total_actual_cost"]),
        help=f"Across {cost_metrics['request_count']} logged LLM calls.",
    )
    col_b.metric(
        f"Frontier-Equivalent ({cost_metrics['frontier_model']})",
        _format_usd(cost_metrics["total_frontier_equivalent"]),
        help=f"What the same token usage would cost on {cost_metrics['frontier_model']}.",
    )
    savings_pct = cost_metrics["savings_pct"]
    # savings_pct is None whenever total_frontier_equivalent is 0 (e.g. logged
    # calls with no token_usage) even though request_count > 0 -- guard
    # separately from the request_count==0 check above.
    col_c.metric("Savings", _format_percent(savings_pct) if savings_pct is not None else "No data yet")


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_header() -> None:
    st.set_page_config(
        page_title=DASHBOARD_TITLE,
        page_icon="⚒️",
        layout="wide",
    )
    st.title(f"⚒️✨ {DASHBOARD_TITLE}")
    st.caption(
        "Public, real-time telemetry streamed from Supabase. "
        "Embed this badge in README and release announcements."
    )


def _render_metric_cards(metrics: Dict[str, Any], cost_savings_pct: float | None) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Mean Time to Merge (MTTM)",
        f"{metrics['mttm_minutes']:.2f} min",
        help="Average time from issue open to PR merge.",
    )
    col2.metric(
        "Autonomous Auto-Merge Rate",
        _format_percent(metrics["auto_merge_rate"]),
        help="Share of PRs validated and merged by CI harnesses.",
    )
    col3.metric(
        "Cost Savings vs Frontier-Equivalent",
        _format_percent(cost_savings_pct) if cost_savings_pct is not None else "No data yet",
        help="Derived from logged actual vs frontier-equivalent $ cost, last 7 days.",
    )
    st.caption(f"Metric source: `{metrics.get('source', 'baseline')}`")


def _render_agent_logs(logs: List[Dict[str, Any]]) -> None:
    st.subheader("🧾 Active Agent Execution Logs")
    if not logs:
        st.info("No live agent execution logs available. Showing baseline metrics above.")
        return
    for entry in logs:
        with st.expander(
            f"{entry.get('created_at', 'unknown')} — {entry.get('agent', 'agent')}"
        ):
            st.json(entry)


def _render_jobs(jobs: List[Dict[str, Any]]) -> None:
    st.subheader("🔨 Recent Forge Jobs")
    if not jobs:
        st.info("No live forge jobs available.")
        return
    st.dataframe(jobs, use_container_width=True)


def _render_footer() -> None:
    st.divider()
    st.caption(
        f"Public dashboard: [{DASHBOARD_URL}]({DASHBOARD_URL}) — "
 "powered by Supabase telemetry from `src/utils/supabase_exporter.py`."
    )
    st.caption(f"Last refreshed: {_utc_now()}")


def main() -> None:
    _render_header()
    metrics = _fetch_metrics()
    cost_metrics = _fetch_cost_metrics(7)
    _render_metric_cards(metrics, cost_metrics["savings_pct"])
    _render_cost_ticker(cost_metrics)
    col_a, col_b = st.columns(2)
    with col_a:
        _render_agent_logs(_fetch_agent_logs())
    with col_b:
        _render_jobs(_fetch_jobs())
    st.divider()
    _render_ci_cost_widget()
    _render_footer()


if __name__ == "__main__":
    main()

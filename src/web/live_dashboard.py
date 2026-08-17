"""Public live telemetry Streamlit dashboard for SparkleForge.

Pulls real-time metrics from Supabase tables populated by
``src/utils/supabase_exporter.py`` and renders them for the community.
Run with::

    streamlit run src/web/live_dashboard.py
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import streamlit as st

from src.utils.supabase_exporter import get_supabase_client

logger = logging.getLogger(__name__)

DASHBOARD_TITLE = "SparkleForge Live Telemetry"
DASHBOARD_URL = "https://sparkleforge.streamlit.app"

# Baseline metrics surfaced in the issue. These are shown when Supabase is
# unreachable so the public dashboard remains informative.
FALLBACK_METRICS: Dict[str, Any] = {
    "mttm_minutes": 141.08,
    "auto_merge_rate": 0.667,
    "token_savings": 0.92,
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
        if "token_savings" in row:
            metrics["token_savings"] = float(row["token_savings"])
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


def _render_metric_cards(metrics: Dict[str, Any]) -> None:
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
        "Zero-Cost Reactive Scheduler Token Savings",
        _format_percent(metrics["token_savings"]),
        help="Token cost reduction during async waiting.",
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
    _render_metric_cards(metrics)
    col_a, col_b = st.columns(2)
    with col_a:
        _render_agent_logs(_fetch_agent_logs())
    with col_b:
        _render_jobs(_fetch_jobs())
    _render_footer()


if __name__ == "__main__":
    main()

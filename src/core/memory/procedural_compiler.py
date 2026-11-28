"""Procedural Playbook Compiler & Local Error Context Ledger.

Compiles verified successful execution trajectories and resolved error fixes into
compact, reusable playbook templates (SKILL.md style) capped at <= 400 tokens,
and provides local SQLite fallback persistence for agent error contexts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.sparkleforge_history import get_local_error_ledger_path, get_local_connection

logger = logging.getLogger(__name__)

MAX_PLAYBOOK_TOKENS = 400


def compile_playbook(
    error_type: str,
    scenario_name: Optional[str],
    root_cause: str,
    verified_steps: List[str],
    anti_patterns: Optional[List[str]] = None,
    verified: bool = True,
) -> Optional[str]:
    """Extract and compile a compact procedural playbook if verified is True.

    Capped at <= 400 tokens (~1600 characters).
    """
    if not verified:
        logger.debug("Skipping playbook compilation: run was not verified.")
        return None

    trigger_sig = f"{error_type}:{scenario_name or 'global'}"
    steps_md = "\n".join(f"- {step}" for step in verified_steps)
    anti_md = "\n".join(f"- {ap}" for ap in (anti_patterns or []))

    playbook = f"""# Playbook: {trigger_sig}

## Trigger Signature
`{trigger_sig}`

## Root Cause
{root_cause}

## Verified Steps
{steps_md}
"""
    if anti_patterns:
        playbook += f"\n## Anti-Patterns to Avoid\n{anti_md}\n"

    # Rough token check (approx 4 chars per token)
    if len(playbook) > MAX_PLAYBOOK_TOKENS * 4:
        # Truncate steps or root cause if too large
        playbook = playbook[: MAX_PLAYBOOK_TOKENS * 4]

    return playbook


def fetch_matching_playbooks(error_type: str, scenario_name: Optional[str] = None) -> List[str]:
    """Fetch historical resolved fix playbooks or error context resolutions for matching signature."""
    conn = get_local_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        query = """
            SELECT resolution_summary, execution_context FROM agent_error_contexts
            WHERE error_type = ? AND remediation_status = 'resolved'
        """
        params: list[Any] = [error_type]
        if scenario_name:
            query += " AND scenario_name = ?"
            params.append(scenario_name)
        query += " ORDER BY created_at DESC LIMIT 5"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        playbooks = []
        for row in rows:
            summary = row[0]
            if summary:
                playbooks.append(summary)
        return playbooks
    except Exception as e:
        logger.debug("Failed to fetch matching playbooks: %s", e)
        return []
    finally:
        conn.close()


def render_report_metrics_section() -> List[str]:
    """Compute and render real Recovery Rate and Historical Remediation Reuse Rate for benchmark report."""
    from src.utils.sparkleforge_history import compute_ledger_metrics
    metrics = compute_ledger_metrics()
    recovery_rate = metrics.get("recovery_rate", 0.0)
    reuse_rate = metrics.get("reuse_rate", 0.0)
    total_errors = metrics.get("total_errors", 0)
    resolved_errors = metrics.get("resolved_errors", 0)

    lines = [
        "#### Local-First Error Remediation & Procedural Memory (Issue #2029)",
        "",
        f"- **Total Recorded Errors**: {total_errors}",
        f"- **Resolved Errors**: {resolved_errors}",
        f"- **Recovery Rate**: **{recovery_rate:.1f}%**",
        f"- **Historical Remediation Reuse Rate**: **{reuse_rate:.1f}%**",
        "",
    ]
    return lines

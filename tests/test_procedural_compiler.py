from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.utils.sparkleforge_history import (
    log_error_context,
    mark_error_resolved,
    compute_ledger_metrics,
    fetch_recent_patch_failures,
)
from src.core.memory.procedural_compiler import (
    compile_playbook,
    fetch_matching_playbooks,
)


def test_local_first_error_ledger_offline(monkeypatch, tmp_path):
    db_path = tmp_path / "error_contexts.db"
    monkeypatch.setattr("src.utils.sparkleforge_history.get_local_error_ledger_path", lambda: db_path)
    monkeypatch.setattr("src.utils.supabase_exporter.get_supabase_client", lambda: None)

    # 1. Record error offline
    log_error_context(
        error_type="PatchApplyFailure",
        error_message="Hunk mismatch at line 42",
        scenario_name="system_cleanup",
        execution_context={"issue_number": "1234"},
    )

    # 2. Query recent patch failures offline
    failures = fetch_recent_patch_failures(issue_number="1234", limit=1)
    assert len(failures) == 1
    assert failures[0]["error_message"] == "Hunk mismatch at line 42"

    # 3. Mark resolved and verify transition
    conn = tmp_path / "error_contexts.db" # using same path via monkeypatch
    # Get the ID of the inserted error
    import sqlite3
    sqlite_conn = sqlite3.connect(str(db_path))
    row_id = sqlite_conn.execute("SELECT id FROM agent_error_contexts").fetchone()[0]
    sqlite_conn.close()

    mark_error_resolved(row_id, resolution_summary="Fixed hunk mismatch by matching exact context.", commit_or_patch_hash="abc1234")

    metrics = compute_ledger_metrics()
    assert metrics["total_errors"] == 1
    assert metrics["resolved_errors"] == 1
    assert metrics["recovery_rate"] == 100.0


def test_procedural_compiler_validation_and_bounds(monkeypatch, tmp_path):
    db_path = tmp_path / "error_contexts.db"
    monkeypatch.setattr("src.utils.sparkleforge_history.

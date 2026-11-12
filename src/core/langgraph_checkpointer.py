"""Shared disk-backed LangGraph checkpointer.

Both `AutonomousOrchestrator` (src/core/orchestrator/graph.py) and
`AgentHarness` (src/core/agent_harness.py) build a LangGraph `StateGraph`
that needs to survive a process crash/restart mid-run. `MemorySaver` (or no
checkpointer at all) only keeps state in process memory, so a crash loses
the entire in-flight plan/execution history with no way to resume.

`AsyncSqliteSaver` (not the sync `SqliteSaver`) is required here because
both graphs are driven via `ainvoke`/`aget_state`, and `SqliteSaver`
explicitly raises `NotImplementedError` on every async method. `aiosqlite`
connections are lazy — `aiosqlite.connect()` just spawns a background worker
thread and defers the real `sqlite3.connect()` call until first awaited use
— so this can still be built synchronously from `__init__`, and
`AsyncSqliteSaver` creates its tables on first use (`setup()` is called
internally, lazily) without an explicit async setup step here.
"""

from __future__ import annotations

import os

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def build_sqlite_checkpointer(db_path: str) -> AsyncSqliteSaver:
    """Build a SQLite-backed async checkpointer at `db_path`."""
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = aiosqlite.connect(db_path)
    # Prevent unclosed aiosqlite worker threads from blocking CLI process shutdown if an early abort skips aclose().
    worker_thread = getattr(conn, "_thread", None)
    if worker_thread is not None and not worker_thread.is_alive():
        worker_thread.daemon = True
    return AsyncSqliteSaver(conn)

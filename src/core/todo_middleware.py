"""Todo list state for task tracking and context management integration.

Provides:
- Session-scoped todo list (task breakdown with status) for progress UI and reporting.
- Integration with progress_tracker and context_compaction: when messages grow,
  compaction is triggered; large tool outputs are offloaded via scratch_pad.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Session ID -> list of todo items {content, status, task_id?}
_session_todos: Dict[str, List[Dict[str, Any]]] = {}


@dataclass
class TodoItem:
    """Single todo entry for task tracking."""

    content: str
    status: str = "pending"  # pending | in_progress | completed | failed
    task_id: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_session_todos(session_id: str) -> List[Dict[str, Any]]:
    """Return the current todo list for a session (for UI / progress)."""
    return list(_session_todos.get(session_id, []))


def set_session_todos(session_id: str, todos: List[Dict[str, Any]]) -> None:
    """Set the full todo list for a session (e.g. from planner task breakdown)."""
    _session_todos[session_id] = [dict(t) for t in todos]


def update_todo_status(
    session_id: str,
    task_id: str | None,
    status: str,
    content: str | None = None,
) -> None:
    """Update a single todo by task_id or by content match."""
    if session_id not in _session_todos:
        return
    for t in _session_todos[session_id]:
        if task_id and t.get("task_id") == task_id:
            t["status"] = status
            if content is not None:
                t["content"] = content
            return
        if content is not None and t.get("content") == content:
            t["status"] = status
            if task_id is not None:
                t["task_id"] = task_id
            return


def sync_todos_from_research_tasks(
    session_id: str, research_tasks: List[Dict[str, Any]]
) -> None:
    """Build todo list from research_tasks (planner output) for tracking."""
    todos = []
    for i, task in enumerate(research_tasks):
        desc = task.get("description") or task.get("objective") or str(task)
        task_id = task.get("task_id") or f"task_{i}"
        todos.append({
            "content": desc[:500] if isinstance(desc, str) else str(desc)[:500],
            "status": "pending",
            "task_id": task_id,
        })
    set_session_todos(session_id, todos)
    logger.debug("Synced %d todos from research_tasks for session %s", len(todos), session_id)


def ensure_context_management(
    session_id: str,
    message_count: int,
    total_tokens_estimate: int,
    context_window: int = 128_000,
    compaction_ratio: float = 0.85,
) -> bool:
    """Signal that context may need compaction (call from orchestrator when appropriate).

    Returns True if compaction was recommended (caller may invoke CompactionManager).
    """
    if total_tokens_estimate <= 0 or context_window <= 0:
        return False
    if total_tokens_estimate >= int(context_window * compaction_ratio):
        logger.info(
            "[TodoMiddleware] Context near limit: %s tokens (%.0f%% of %s); compaction recommended",
            total_tokens_estimate,
            100 * total_tokens_estimate / context_window,
            context_window,
        )
        return True
    return False


def get_context_summary_and_file_bypass_hint() -> str:
    """Return a short hint for the agent about context management (for system prompt)."""
    return (
        "Large tool outputs are automatically offloaded to files (scratch pad); "
        "only a summary and file path are kept in context. "
        "When the conversation is long, older messages may be summarized to stay within the context window."
    )

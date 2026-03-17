"""Isolated task state for sub-agent execution to prevent main context pollution.

When running a task in a sub-agent (ExecutorAgent), pass a scoped state that contains
only what that task needs; after execution, merge back only the task's results
into the main state so the main context window is not filled with every executor's
tool calls and messages.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Keys that sub-agent needs to run a single task (minimal context)
ISOLATED_STATE_KEYS = (
    "session_id",
    "user_query",
    "research_plan",
    "research_tasks",
    "shared_results_manager",
    "discussion_manager",
    "sparkle_ideas",
    "direct_forward_message",
    "direct_forward_from_agent",
    "pending_questions",
    "user_responses",
    "clarification_context",
    "waiting_for_user",
)

# Keys we merge back from sub-agent result into main state (only task outputs)
MERGE_BACK_KEYS = (
    "research_results",
    "research_failed",
    "error",
    "direct_forward_message",
    "direct_forward_from_agent",
)


def create_isolated_task_state(
    full_state: Dict[str, Any],
    assigned_task: Dict[str, Any],
    task_index: int,
) -> Dict[str, Any]:
    """Build a minimal state for a single sub-agent task to keep its context isolated.

    The sub-agent receives only session_id, user_query, research_plan, the single
    assigned_task (in a one-item research_tasks list), and shared managers.
    Message history is not passed so the sub-agent's context is not polluted
    by the main agent's conversation.
    """
    isolated: Dict[str, Any] = {}
    for key in ISOLATED_STATE_KEYS:
        if key in full_state:
            isolated[key] = full_state[key]
    # Single-task list so the executor only sees its own task
    isolated["research_tasks"] = [assigned_task]
    isolated["research_results"] = []
    isolated["verified_results"] = full_state.get("verified_results") or []
    isolated["final_report"] = None
    isolated["current_agent"] = None
    isolated["iteration"] = full_state.get("iteration", 0)
    isolated["verification_failed"] = full_state.get("verification_failed", False)
    isolated["report_failed"] = full_state.get("report_failed", False)
    # Minimal messages: only user query and plan summary to stay under context budget
    messages = full_state.get("messages") or []
    if messages:
        # Keep only last user message and a single system summary if any
        isolated["messages"] = [m for m in messages[-3:] if m]
    else:
        isolated["messages"] = []
    return isolated


def merge_task_result_into_state(
    main_state: Dict[str, Any],
    task_id: str,
    task_index: int,
    result_state: Dict[str, Any],
) -> None:
    """Merge only the task's results from result_state into main_state.

    Appends result_state['research_results'] to main_state['research_results'],
    and updates research_failed/error/direct_forward_* if present. Does not
    copy messages or other large fields to avoid polluting the main context.
    """
    new_results = result_state.get("research_results") or []
    main_results = main_state.get("research_results") or []
    for r in new_results:
        if isinstance(r, dict):
            r["_task_id"] = task_id
            r["_executor_index"] = task_index
        main_results.append(r)
    main_state["research_results"] = main_results

    if result_state.get("research_failed"):
        main_state["research_failed"] = True
    if result_state.get("error"):
        existing = main_state.get("error") or ""
        main_state["error"] = f"{existing}; Task {task_id}: {result_state['error']}".strip(" ;")
    if result_state.get("direct_forward_message"):
        main_state["direct_forward_message"] = result_state["direct_forward_message"]
    if result_state.get("direct_forward_from_agent"):
        main_state["direct_forward_from_agent"] = result_state["direct_forward_from_agent"]


def get_isolated_state_keys() -> tuple:
    """Return the set of keys included in isolated task state (for tests/docs)."""
    return ISOLATED_STATE_KEYS


def get_merge_back_keys() -> tuple:
    """Return the set of keys merged from sub-agent result (for tests/docs)."""
    return MERGE_BACK_KEYS

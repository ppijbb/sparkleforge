"""Langfuse observability for LangChain/LangGraph tracing.

Uses environment variables only (no hardcoded keys):
- LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY: required to enable tracing.
- LANGFUSE_BASE_URL: optional (default cloud; set for self-hosted).

Security:
- API keys are read only from os.environ; this module never logs or returns key values.
- session_id, user_id, and tags are sent to Langfuse as trace metadata; do not pass
  secrets or PII in these fields if you use a shared Langfuse project.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List

from src.core.input_router import (
    TRACE_AGENT_ID,
    TRACE_ENTRYPOINT,
    TRACE_REQUEST_ID,
    TRACE_SOURCE_TYPE,
    TRACE_TASK_ID,
    TRACE_TURN_ID,
    get_trace_context,
)

_LANGFUSE_ENABLED: bool | None = None


def _langfuse_enabled() -> bool:
    """True only if both required Langfuse env vars are set."""
    global _LANGFUSE_ENABLED
    if _LANGFUSE_ENABLED is None:
        _LANGFUSE_ENABLED = bool(
            os.environ.get("LANGFUSE_PUBLIC_KEY")
            and os.environ.get("LANGFUSE_SECRET_KEY")
        )
    return _LANGFUSE_ENABLED


def get_langfuse_callbacks(
    session_id: str | None = None,
    user_id: str | None = None,
    tags: List[str] | None = None,
) -> List[Any]:
    """Return Langfuse callback handlers for LangChain, or empty list if disabled.

    When Langfuse is enabled, pass session_id/user_id/tags via get_langfuse_run_config
    so they are sent as trace metadata (Langfuse SDK v3 uses config metadata).
    """
    if not _langfuse_enabled():
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except ImportError:
        return []


def get_langfuse_run_config(
    session_id: str | None = None,
    user_id: str | None = None,
    tags: List[str] | None = None,
) -> dict:
    """Return run config with callbacks and trace metadata for ainvoke/astream.

    Merges trace context from get_trace_context() when available (turn_id, request_id,
    entrypoint, source_type, task_id, agent_id) so per-turn/agent/task correlation works.
    Use as: graph.ainvoke(state, config=get_langfuse_run_config(session_id=...))
    Returns {} when Langfuse is disabled (safe to pass as config).
    """
    callbacks = get_langfuse_callbacks()
    if not callbacks:
        return {}
    meta: dict = {}
    if session_id is not None:
        meta["langfuse_session_id"] = session_id
    if user_id is not None:
        meta["langfuse_user_id"] = user_id
    if tags is not None:
        meta["langfuse_tags"] = tags
    ctx = get_trace_context()
    tags: List[str] = []
    if ctx:
        for key in (
            TRACE_TURN_ID,
            TRACE_REQUEST_ID,
            TRACE_ENTRYPOINT,
            TRACE_SOURCE_TYPE,
            TRACE_TASK_ID,
            TRACE_AGENT_ID,
        ):
            val = ctx.get(key)
            if val is not None and isinstance(val, str) and len(val) <= 200:
                meta[f"trace_{key}"] = val
        # Langfuse tag strategy: low-cardinality ops tags (env, entrypoint, source_type)
        entrypoint = ctx.get(TRACE_ENTRYPOINT)
        source_type = ctx.get(TRACE_SOURCE_TYPE)
        if entrypoint:
            tags.append(f"entrypoint:{entrypoint}")
        if source_type and source_type != entrypoint:
            tags.append(f"source:{source_type}")
    if tags:
        meta["langfuse_tags"] = tags
    return (
        {"callbacks": callbacks, "metadata": meta} if meta else {"callbacks": callbacks}
    )


def get_langfuse_client() -> Any | None:
    """Return Langfuse client when enabled, for flush() in short-lived processes. Otherwise None."""
    if not _langfuse_enabled():
        return None
    try:
        from langfuse import get_client

        return get_client()
    except ImportError:
        return None


@contextmanager
def start_turn_trace(
    name: str = "turn",
    input: Any = None,
    session_id: str | None = None,
) -> Generator[Any, None, None]:
    """Root span for one user request (turn). Use around graph.ainvoke().

    When Langfuse is disabled, yields None and does nothing.
    """
    if not _langfuse_enabled():
        yield None
        return
    try:
        from langfuse import get_client, propagate_attributes

        client = get_client()
        ctx = get_trace_context() or {}
        sid = session_id or ctx.get(TRACE_TURN_ID) or "default"
        meta: Dict[str, str] = {}
        for k in (TRACE_TURN_ID, TRACE_REQUEST_ID, TRACE_ENTRYPOINT, TRACE_SOURCE_TYPE):
            v = ctx.get(k)
            if v is not None and isinstance(v, str) and len(v) <= 200:
                meta[k] = v
        with client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input or {},
        ) as span:
            with propagate_attributes(session_id=sid, metadata=meta):
                yield span
    except ImportError:
        yield None
    except Exception:
        yield None


@contextmanager
def start_agent_span(
    name: str,
    agent_id: str,
    input: Any = None,
) -> Generator[Any, None, None]:
    """Child span for an agent stage (planner/executor/verifier/generator)."""
    if not _langfuse_enabled():
        yield None
        return
    try:
        from langfuse import get_client, propagate_attributes

        client = get_client()
        with client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input or {},
            metadata={"agent_id": agent_id[:200]},
        ) as span:
            with propagate_attributes(metadata={TRACE_AGENT_ID: agent_id[:200]}):
                yield span
    except (ImportError, Exception):
        yield None


@contextmanager
def start_tool_span(
    name: str,
    tool_name: str,
    input: Any = None,
) -> Generator[Any, None, None]:
    """Child span for MCP or local tool execution."""
    if not _langfuse_enabled():
        yield None
        return
    try:
        from langfuse import get_client

        client = get_client()
        with client.start_as_current_observation(
            as_type="span",
            name=name or tool_name,
            input=input or {},
            metadata={"tool_name": tool_name[:200]},
        ) as span:
            yield span
    except (ImportError, Exception):
        yield None


def update_generation_usage(
    usage_details: Dict[str, int] | None = None,
    cost_details: Dict[str, float] | None = None,
) -> None:
    """Update the current Langfuse generation observation with usage/cost.

    Call after LLM response when actual usage is available.
    When Langfuse is disabled or no current generation, no-op.
    """
    if not _langfuse_enabled() or (not usage_details and not cost_details):
        return
    try:
        from langfuse import get_client

        client = get_client()
        client.update_current_observation(
            usage_details=usage_details,
            cost_details=cost_details,
        )
    except (ImportError, Exception):
        pass


def propagate_trace_context(
    turn_id: str | None = None,
    agent_id: str | None = None,
    task_id: str | None = None,
    model: str | None = None,
    tool_name: str | None = None,
) -> None:
    """Set trace context for the current scope (for downstream spans).

    Values are applied via Langfuse propagate_attributes when inside an active trace.
    String values longer than 200 chars are truncated for Langfuse.
    """
    if not _langfuse_enabled():
        return
    try:
        from langfuse import propagate_attributes

        meta: Dict[str, str] = {}
        if turn_id:
            meta[TRACE_TURN_ID] = turn_id[:200]
        if agent_id:
            meta[TRACE_AGENT_ID] = agent_id[:200]
        if task_id:
            meta[TRACE_TASK_ID] = task_id[:200]
        if model:
            meta["model"] = model[:200]
        if tool_name:
            meta["tool_name"] = tool_name[:200]
        if meta:
            propagate_attributes(metadata=meta)
    except (ImportError, Exception):
        pass

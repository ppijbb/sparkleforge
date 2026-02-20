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
from typing import Any, List, Optional

_LANGFUSE_ENABLED: Optional[bool] = None


def _langfuse_enabled() -> bool:
    """True only if both required Langfuse env vars are set."""
    global _LANGFUSE_ENABLED
    if _LANGFUSE_ENABLED is None:
        _LANGFUSE_ENABLED = bool(
            os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
        )
    return _LANGFUSE_ENABLED


def get_langfuse_callbacks(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
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
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> dict:
    """Return run config with callbacks and trace metadata for ainvoke/astream.

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
    return {"callbacks": callbacks, "metadata": meta} if meta else {"callbacks": callbacks}


def get_langfuse_client() -> Optional[Any]:
    """Return Langfuse client when enabled, for flush() in short-lived processes. Otherwise None."""
    if not _langfuse_enabled():
        return None
    try:
        from langfuse import get_client
        return get_client()
    except ImportError:
        return None

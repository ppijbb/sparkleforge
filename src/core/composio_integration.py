"""Composio Tool Router integration interface (open-claude-cowork style).

When Composio is configured (COMPOSIO_API_KEY and optional composio package),
tools from 500+ apps (Slack, Gmail, GitHub, etc.) can be discovered and executed
through a single router. This module provides the integration interface so that
SparkleForge can optionally register Composio tools alongside MCP tools.

Usage:
  - Set COMPOSIO_API_KEY in the environment.
  - Optionally install: pip install composio-core composio-openai (or composio langchain)
  - Call get_composio_tools() to obtain a list of tools to merge with MCP tools.
  - Tool execution can be routed via Composio when the tool name is in the
    composio registry (e.g. SLACK_SEND_MESSAGE, GMAIL_SEND_EMAIL).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_COMPOSIO_TOOLS_CACHE: List[Any] = []
_COMPOSIO_AVAILABLE: bool | None = None


def is_composio_available() -> bool:
    """Return True if Composio SDK is installed and COMPOSIO_API_KEY is set."""
    global _COMPOSIO_AVAILABLE
    if _COMPOSIO_AVAILABLE is not None:
        return _COMPOSIO_AVAILABLE
    if not os.environ.get("COMPOSIO_API_KEY"):
        _COMPOSIO_AVAILABLE = False
        return False
    try:
        import composio  # noqa: F401
        _COMPOSIO_AVAILABLE = True
        return True
    except ImportError:
        _COMPOSIO_AVAILABLE = False
        return False


def get_composio_tools(
    entity_id: str | None = None,
    tags: List[str] | None = None,
) -> List[Any]:
    """Return LangChain-compatible tools from Composio Tool Router.

    When Composio is not available, returns an empty list so callers can
    merge with MCP tools without error.
    entity_id: Composio entity (e.g. user/app) for which to fetch tools.
    tags: Optional filter by action tags.
    """
    global _COMPOSIO_TOOLS_CACHE
    if not is_composio_available():
        return []
    try:
        from composio.tools import ComposioToolSet
        from langchain_core.tools import BaseTool

        toolset = ComposioToolSet()
        tools = toolset.get_tools(
            entity_id=entity_id or os.environ.get("COMPOSIO_ENTITY_ID"),
            tags=tags or [],
        )
        if isinstance(tools, list):
            _COMPOSIO_TOOLS_CACHE = tools
            return tools
        return []
    except Exception as e:
        logger.warning("Composio get_tools failed: %s", e)
        return []


def execute_composio_tool(
    tool_name: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a tool by name via Composio (when the tool is from Composio registry).

    Returns dict with success, data, error. Call from MCP hub or tool router
    when tool_name is identified as a Composio action.
    """
    if not is_composio_available():
        return {"success": False, "error": "Composio not configured", "data": None}
    try:
        from composio.tools import ComposioToolSet

        toolset = ComposioToolSet()
        result = toolset.execute(
            action=tool_name,
            params=parameters,
        )
        return {"success": True, "data": result, "error": None}
    except Exception as e:
        logger.warning("Composio execute failed for %s: %s", tool_name, e)
        return {"success": False, "error": str(e), "data": None}


def get_composio_tool_names() -> List[str]:
    """Return list of Composio tool/action names for routing (when available)."""
    if not _COMPOSIO_TOOLS_CACHE:
        get_composio_tools()
    return [getattr(t, "name", str(t)) for t in _COMPOSIO_TOOLS_CACHE]

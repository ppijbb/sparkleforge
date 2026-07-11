"""MCP integration subpackage.

Split out from the monolithic ``src/core/mcp_integration.py`` (issue #494)
by concern. This package hosts per-concern modules so the legacy single-file
hub remains reviewable incrementally.
"""

from src.core.mcp.openrouter_client import OpenRouterClient

__all__ = ["OpenRouterClient"]

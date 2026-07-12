"""Universal MCP Hub - package facade.

``src/core/mcp_integration.py`` used to be a single 7,776-line module (issue
#508, Anvil Phase Sigma-1: split from #449/#424, reworking the incomplete
#494 attempt). It's now this package:

- ``parser.py``: pure parsing/formatting helpers for tool calls and results.
- ``client.py``: ``OpenRouterClient`` (disabled; kept for import compat).
- ``tools.py``: ``execute_tool``/``get_mcp_hub`` and the per-category
  ``_execute_*_tool`` dispatchers.
- ``hub.py``: ``UniversalMCPHub`` -- MCP session/transport management and
  tool discovery.

Every name that was importable from ``src.core.mcp_integration`` before the
split is re-exported here unchanged, so ``from src.core.mcp_integration
import X`` keeps working for all existing call sites without modification.

Import order matters: ``tools`` must load before ``hub``, since ``hub``
imports several dispatchers from ``tools`` at module level, while ``tools``
only reaches into ``hub`` lazily inside ``get_mcp_hub()``. See ``hub.py``'s
and ``tools.py``'s module docstrings for why that direction avoids a
circular import.
"""

from src.core.mcp_integration.client import OpenRouterClient
from src.core.mcp_integration.hub import (
    FASTMCP_AVAILABLE,
    HTTP_CLIENT_AVAILABLE,
    LANGCHAIN_AVAILABLE,
    MCP_AVAILABLE,
    UniversalMCPHub,
)
from src.core.mcp_integration.parser import (
    _actionable_error_message,
    _cap_tool_result_for_context,
    _create_tool_trace,
    _format_query_string,
    _infer_tool_type,
    _normalize_mcp_call_params,
    _normalize_mcp_tool_alias,
    _parse_json_text,
    _parse_markdown_link_results,
    _structured_tool_description,
    get_tool_trace_manager,
    set_tool_trace_manager,
)
from src.core.mcp_integration.tools import (
    _execute_academic_tool,
    _execute_academic_tool_sync,
    _execute_browser_tool,
    _execute_code_tool,
    _execute_code_tool_sync,
    _execute_data_tool,
    _execute_data_tool_sync,
    _execute_document_tool,
    _execute_file_tool,
    _execute_git_tool,
    _execute_search_tool,
    _execute_search_tool_sync,
    _execute_shell_tool,
    _fallback_to_ddg_search,
    _get_ddg_lock,
    _playwright_dismiss_google_consent,
    check_mcp_servers,
    execute_tool,
    get_available_tools,
    get_best_tool_for_task,
    get_mcp_hub,
    get_tool_for_category,
    health_check,
    list_tools,
    run_mcp_hub,
)
from src.core.tools.registry import ToolCategory, ToolInfo, ToolResult

__all__ = [
    "ToolCategory",
    "ToolInfo",
    "ToolResult",
    "OpenRouterClient",
    "UniversalMCPHub",
    "MCP_AVAILABLE",
    "HTTP_CLIENT_AVAILABLE",
    "FASTMCP_AVAILABLE",
    "LANGCHAIN_AVAILABLE",
    "get_mcp_hub",
    "get_available_tools",
    "execute_tool",
    "get_best_tool_for_task",
    "get_tool_for_category",
    "health_check",
    "run_mcp_hub",
    "list_tools",
    "check_mcp_servers",
    "_execute_search_tool",
    "_execute_search_tool_sync",
    "_execute_academic_tool",
    "_execute_academic_tool_sync",
    "_execute_data_tool",
    "_execute_data_tool_sync",
    "_execute_code_tool",
    "_execute_code_tool_sync",
    "_execute_browser_tool",
    "_execute_document_tool",
    "_execute_git_tool",
    "_execute_shell_tool",
    "_execute_file_tool",
    "_playwright_dismiss_google_consent",
    "_fallback_to_ddg_search",
    "_get_ddg_lock",
    "_parse_json_text",
    "_parse_markdown_link_results",
    "_normalize_mcp_tool_alias",
    "_infer_tool_type",
    "_format_query_string",
    "_structured_tool_description",
    "_actionable_error_message",
    "_cap_tool_result_for_context",
    "_normalize_mcp_call_params",
    "_create_tool_trace",
    "get_tool_trace_manager",
    "set_tool_trace_manager",
]

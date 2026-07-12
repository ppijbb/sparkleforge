"""
tests/test_mcp_integration_tools.py — regression tests for issue #524.

`src/core/mcp_integration/tools.py` (post monolith split, Anvil Phase Sigma-1)
had three concrete correctness bugs, since fixed and further split into
`src/core/mcp_integration/executors/*` (issue #507):

1. `_execute_data_tool` dropped the result of `_execute_file_tool` /
   `_execute_browser_tool` / `_execute_shell_tool` (missing `return`), so
   every filesystem/browser/shell call fell through to the `else` branch's
   `ValueError`.
2. `_execute_search_tool` had an unreachable `else:` on a `try/except` whose
   `try` body always returns explicitly -- dead code that never runs.
3. A dedent in `_execute_search_tool`'s markdown-parsing branch let a
   fallback text-parsing block run unconditionally after the TAVILY-format
   if/else, silently overwriting a correctly parsed TAVILY result with a
   low-quality single-item fallback.
"""
import pytest

from src.core.mcp_integration.executors import data, search
from src.core.tools.registry import ToolResult


@pytest.mark.asyncio
async def test_execute_data_tool_filesystem_returns_dispatcher_result(monkeypatch):
    sentinel = ToolResult(success=True, data={"ok": "filesystem"})

    async def fake_file_tool(tool_name, parameters):
        return sentinel

    monkeypatch.setattr(data, "_execute_file_tool", fake_file_tool)

    result = await data._execute_data_tool("filesystem", {})

    assert result is sentinel


@pytest.mark.asyncio
async def test_execute_data_tool_browser_returns_dispatcher_result(monkeypatch):
    sentinel = ToolResult(success=True, data={"ok": "browser"})

    async def fake_browser_tool(tool_name, parameters):
        return sentinel

    monkeypatch.setattr(data, "_execute_browser_tool", fake_browser_tool)

    result = await data._execute_data_tool("browser", {})

    assert result is sentinel


@pytest.mark.asyncio
async def test_execute_data_tool_shell_returns_dispatcher_result(monkeypatch):
    sentinel = ToolResult(success=True, data={"ok": "shell"})

    async def fake_shell_tool(tool_name, parameters):
        return sentinel

    monkeypatch.setattr(data, "_execute_shell_tool", fake_shell_tool)

    result = await data._execute_data_tool("shell", {})

    assert result is sentinel


class _FakeHub:
    """Minimal stand-in for UniversalMCPHub, only exposing what
    _execute_search_tool's MCP-server fallback path touches."""

    def __init__(self, server_name, tool_name, response_text):
        self.mcp_sessions = {server_name: object()}
        self.mcp_server_configs = {server_name: {}}
        self.mcp_tools_map = {server_name: {tool_name: {}}}
        self.request_timing_history = {}
        self._response_text = response_text

    async def _execute_via_mcp_server(self, server_name, tool_name, params):
        return self._response_text


@pytest.mark.asyncio
async def test_execute_search_tool_does_not_overwrite_tavily_parse_with_markdown_fallback(
    monkeypatch,
):
    # Force the embedded src.utils.search_duckduckgo path to fail so control
    # falls through to the MCP-server fallback logic further down.
    import src.utils.search_utils as search_utils_module

    async def failing_search(*args, **kwargs):
        raise RuntimeError("embedded search unavailable in test")

    monkeypatch.setattr(search_utils_module, "search_duckduckgo", failing_search)

    tavily_text = "Title: Real Result\nURL: http://example.com\nContent: Real content\n"
    fake_hub = _FakeHub("tavily-mcp", "tavily-search", tavily_text)
    monkeypatch.setattr(search, "get_mcp_hub", lambda: fake_hub)

    result = await search._execute_search_tool("g-search", {"query": "test", "max_results": 2})

    assert result.success is True
    assert result.data["results"][0]["title"] == "Real Result"
    assert result.data["results"][0]["url"] == "http://example.com"

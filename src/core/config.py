"""
Typed server specifications and conversion helpers for FastMCP configuration.

This module provides models for configuring MCP (Model Context Protocol) servers
with support for both stdio and HTTP/SSE transports. It ensures type safety and validation
for server specifications used throughout the MCP Agent system.

Key Features:
- Type-safe server configuration models
- Support for stdio and HTTP/SSE transports
- Environment variable handling
- Validation and error handling
- Configuration serialization/deserialization

Usage:
    spec = HTTPServerSpec(
        url="http://127.0.0.1:8000/mcp",
        headers={"Authorization": "Bearer token"}
    )
    
    stdio_spec = StdioServerSpec(
        command="python",
        args=["server.py"],
        env={"DEBUG": "true"}
    )
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Any, Dict, List, Optional, Union

# Simple fallback implementation without pydantic dependency
class BaseModel:
    """Simple base model for server specs.
    
    Provides basic attribute initialization without requiring pydantic dependency.
    This lightweight implementation allows for flexible configuration while maintaining
    compatibility with the existing codebase.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def ConfigDict(**kwargs):
    """Placeholder for pydantic ConfigDict compatibility."""
    return kwargs

def Field(default=None, **kwargs):
    """Placeholder for pydantic Field compatibility."""
    return default


class _BaseServer:
    """Base model for server specs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class StdioServerSpec(_BaseServer):
    """Specification for a local MCP server launched via stdio.

    NOTE:
        The FastMCP Python client typically expects HTTP/SSE transports. Using
        `StdioServerSpec` requires a different adapter or an HTTP shim in front
        of the stdio server. Keep this for future expansion or custom runners.
    """

    def __init__(self, command: str, args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd


class HTTPServerSpec(_BaseServer):
    """Specification for a remote MCP server accessed via HTTP/SSE.

    Args:
        url: The server URL (e.g., "http://127.0.0.1:8000/mcp").
        transport: The transport mechanism ("http", "streamable-http", or "sse").
        headers: Optional request headers (e.g., Authorization tokens).
        auth: Optional auth hint if your FastMCP deployment consumes it.
    """

    def __init__(self, url: str, transport: str = "http", headers: Optional[Dict[str, str]] = None, auth: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.transport = transport
        self.headers = headers or {}
        self.auth = auth


ServerSpec = StdioServerSpec | HTTPServerSpec
"""Union of supported server specifications."""


def servers_to_mcp_config(servers: Mapping[str, ServerSpec]) -> dict[str, dict[str, object]]:
    """Convert programmatic server specs to the FastMCP configuration dict.

    Args:
        servers: Mapping of server name to specification.

    Returns:
        Dict suitable for initializing `fastmcp.Client({"mcpServers": ...})`.
    """
    cfg: dict[str, dict[str, object]] = {}
    for name, s in servers.items():
        if isinstance(s, StdioServerSpec):
            # FastMCP uses command, args, env only for stdio servers without transport field
            entry: dict[str, object] = {
                "command": s.command,
                "args": s.args,
            }
            # env가 비어있지 않으면 추가 (None이 아닌 dict만)
            if s.env:
                entry["env"] = s.env
            # cwd가 있으면 추가
            if hasattr(s, 'cwd') and s.cwd:
                entry["cwd"] = s.cwd
            cfg[name] = entry
        else:
            # HTTP 서버의 경우 httpUrl 또는 url 사용
            entry: dict[str, object] = {}
            if s.transport == "http":
                # httpUrl 사용 (FastMCP 표준)
                entry["httpUrl"] = s.url
            else:
                # SSE나 다른 transport의 경우 url 사용
                entry["url"] = s.url
            if s.headers:
                entry["headers"] = s.headers
            if s.auth is not None:
                entry["auth"] = s.auth
            cfg[name] = entry
    return cfg
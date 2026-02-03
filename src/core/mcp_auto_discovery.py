"""FastMCP 클라이언트 통합 - MCP 서버 자동 발견 및 연결 관리."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any  # <-- add

from fastmcp import Client as FastMCPClient

from .config import ServerSpec, servers_to_mcp_config


class FastMCPClientWrapper:
    """FastMCP 클라이언트를 통한 MCP 서버 연결 관리."""

    def __init__(self, servers: Mapping[str, ServerSpec]) -> None:
        """FastMCP 클라이언트 초기화.

        Args:
            servers: 서버 이름 -> ServerSpec 매핑
        """
        mcp_cfg = {"mcpServers": servers_to_mcp_config(servers)}
        self._client: Any = FastMCPClient(mcp_cfg)  # <-- annotate as Any to avoid unparameterized generic

    @property
    def client(self) -> Any:  # <-- return Any to avoid unparameterized generic
        """기본 FastMCP 클라이언트 인스턴스 반환."""
        return self._client


class FastMCPMulti:
    """다중 서버를 위한 FastMCP 클라이언트 래퍼."""

    def __init__(self, servers: Mapping[str, ServerSpec] | dict[str, dict[str, Any]]) -> None:
        """다중 서버 FastMCP 클라이언트 초기화.

        Args:
            servers: 서버 이름 -> ServerSpec 매핑 또는 원본 mcp_config.json 형식의 dict
        """
        # 이미 dict 형식이면 그대로 사용, ServerSpec이면 변환
        if servers:
            # 첫 번째 값의 타입을 확인하여 판단
            first_value = next(iter(servers.values()))
            if isinstance(first_value, dict):
                # 원본 mcp_config.json 형식인 경우
                mcp_cfg = {"mcpServers": servers}
            else:
                # ServerSpec 형식인 경우 변환
                mcp_cfg = {"mcpServers": servers_to_mcp_config(servers)}
        else:
            # 빈 dict인 경우
            mcp_cfg = {"mcpServers": {}}
        self._client: Any = FastMCPClient(mcp_cfg)  # <-- annotate as Any to avoid unparameterized generic

    @property
    def client(self) -> Any:  # <-- return Any to avoid unparameterized generic
        """기본 FastMCP 클라이언트 인스턴스 반환."""
        return self._client


async def test_mcp_connection(server_spec: ServerSpec) -> bool:
    """단일 MCP 서버 연결 테스트.

    Args:
        server_spec: 테스트할 서버 스펙

    Returns:
        연결 성공 여부
    """
    try:
        wrapper = FastMCPClientWrapper({"test": server_spec})
        async with wrapper.client:
            tools = await wrapper.client.list_tools()
            return tools is not None
    except Exception:
        return False


async def discover_server_tools(server_spec: ServerSpec) -> list[dict[str, Any]]:
    """서버에서 사용 가능한 도구 목록을 자동으로 발견.

    Args:
        server_spec: 도구를 발견할 서버 스펙

    Returns:
        발견된 도구 목록
    """
    try:
        wrapper = FastMCPClientWrapper({"discovery": server_spec})
        async with wrapper.client:
            tools = await wrapper.client.list_tools()
            return list(tools or [])
    except Exception as e:
        raise RuntimeError(f"Failed to discover tools from MCP server: {e}") from e

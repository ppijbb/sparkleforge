"""UniversalMCPHub: connection/session management and MCP tool discovery.

Extracted from the monolithic ``src/core/mcp_integration.py`` (issue #508,
Anvil Phase Sigma-1), and further split (issue #507/#524 -- the Sigma-1
split never actually reduced this file, which stayed a single 4,390-line
class). ``UniversalMCPHub`` is now composed from mixins defined under
``src/core/mcp_integration/hub_mixins/``, one per responsibility:

- ``registration.py`` (``ToolRegistrationMixin``): LangChain tool wrapper
  construction, manual/auto-discovered tool loading.
- ``connection.py`` (``ConnectionMixin``): stdio/SSE/streamable-HTTP
  transport lifecycle, server config loading, health checks.
- ``execution.py`` (``ExecutionMixin``): routes a tool call to the right
  MCP server or native executor.
- ``lifecycle.py`` (``LifecycleMixin``): shutdown/cleanup.
- ``status.py`` (``StatusMixin``): category lookup, health/server status
  summaries, LangChain tool listing.

Since Python resolves ``self.foo()`` through the instance's MRO at call
time, methods in one mixin can freely call methods defined in another --
no mixin imports another, only ``UniversalMCPHub`` composes them all.
Optional-dependency flags/types (``MCP_AVAILABLE`` and friends) live in
``mcp_runtime.py`` so mixins and this module can share them without
importing each other (which would be circular, since this module imports
the mixins to build the class).
"""

from contextlib import AsyncExitStack
from typing import Any, Dict, List

from src.core.config import HTTPServerSpec
from src.core.mcp_auto_discovery import FastMCPMulti
from src.core.mcp_integration.client import OpenRouterClient
from src.core.mcp_integration.hub_mixins.connection import ConnectionMixin
from src.core.mcp_integration.hub_mixins.execution import ExecutionMixin
from src.core.mcp_integration.hub_mixins.lifecycle import LifecycleMixin
from src.core.mcp_integration.hub_mixins.registration import ToolRegistrationMixin
from src.core.mcp_integration.hub_mixins.status import StatusMixin
from src.core.mcp_integration.mcp_runtime import (
    FASTMCP_AVAILABLE,
    HTTP_CLIENT_AVAILABLE,
    LANGCHAIN_AVAILABLE,
    MCP_AVAILABLE,
    BaseTool,
    ClientSession,
)
from src.core.mcp_tool_loader import MCPToolLoader
from src.core.mcp_tool_loader import ToolInfo as MCPToolInfo
from src.core.researcher_config import get_llm_config, get_mcp_config
from src.core.tools.registry import ToolInfo
from src.core.tools.registry import registry as global_registry


class UniversalMCPHub(
    ToolRegistrationMixin,
    ConnectionMixin,
    ExecutionMixin,
    LifecycleMixin,
    StatusMixin,
):
    """Universal MCP Hub - 2025년 10월 최신 버전."""

    def __init__(self):
        self.config = get_mcp_config()
        self.llm_config = get_llm_config()

        # ToolRegistry 통합 관리
        self.registry = global_registry

        # 실행 컨텍스트별 MCP 세션 관리 (ROMA 스타일)
        # 각 실행마다 독립적인 MCP 세션 풀을 유지
        self._execution_sessions: Dict[str, Dict[str, Any]] = {}
        self.tools: Dict[str, ToolInfo] = {}  # 하위 호환성을 위해 유지 (registry.tools 참조)
        self.openrouter_client: OpenRouterClient | None = None

        # MCP 클라이언트 (기존 시스템)
        self.mcp_sessions: Dict[str, ClientSession] = {}
        self.exit_stacks: Dict[str, AsyncExitStack] = (
            {}
        )  # 참조만 유지, cleanup에서 aclose() 호출 안 함
        self.mcp_tools_map: Dict[str, Dict[str, Any]] = (
            {}
        )  # server_name -> {tool_name -> tool_info}
        self.mcp_server_configs: Dict[str, Dict[str, Any]] = {}
        # 각 서버별 연결 진단 정보
        self.connection_diagnostics: Dict[str, Dict[str, Any]] = {}
        # 종료/차단 플래그 (종료 중 신규 연결 방지)
        self.stopping: bool = False

        # FastMCP Client 인스턴스 저장 (연결 풀링)
        self.fastmcp_clients: Dict[str, Any] = {}  # server_name -> FastMCPClient

        # Anti-bot 우회를 위한 User-Agent 풀 (Skyvern 스타일)
        self.user_agents = [
            # Chrome (Windows)
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            # Chrome (macOS)
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            # Chrome (Linux)
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            # Firefox (Windows)
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
            # Firefox (macOS)
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:132.0) Gecko/20100101 Firefox/132.0",
            # Safari (macOS)
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
            # Edge (Windows)
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
        ]

        # 요청 간격 변동성을 위한 히스토리 (Skyvern 스타일: 인간 행동 패턴 모방)
        self.request_timing_history: Dict[str, List[float]] = {}  # server_name -> [timestamps]

        # FastMCP 자동 발견 시스템 (신규)
        self.fastmcp_servers: Dict[str, HTTPServerSpec] = {}  # 자동 발견용 서버 설정
        self.fastmcp_multi: FastMCPMulti | None = None
        self.fastmcp_tool_loader: MCPToolLoader | None = None
        # FastMCP 설정 저장소 (서버별) - Client는 context manager이므로 매번 새로 생성
        self.fastmcp_configs: Dict[str, Dict[str, Any]] = {}  # server_name -> mcp_config
        self.auto_discovered_tools: Dict[str, BaseTool] = {}  # 자동 발견된 도구들
        self.auto_discovered_tool_infos: Dict[str, MCPToolInfo] = {}  # 도구 메타데이터

        self._load_tools_config()
        self._initialize_tools()
        self._initialize_clients()

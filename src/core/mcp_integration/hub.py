"""UniversalMCPHub: connection/session management and MCP tool discovery.

Extracted from the monolithic ``src/core/mcp_integration.py`` (issue #508,
Anvil Phase Sigma-1). This is the bulk of the original file: MCP session
lifecycle (stdio/SSE/streamable-HTTP transports), server discovery, and
LangChain tool wrapping.

Several methods here bare-reference per-category dispatchers defined under
``src/core/mcp_integration/executors/`` (``_execute_search_tool_sync`` and
friends, issue #507/#524) and helpers defined in ``parser.py`` -- those are
imported at module level below. ``tools.py`` only needs ``UniversalMCPHub``
inside ``get_mcp_hub()``, deferred until first call, so importing it here
at module load time does not create a circular import.
"""

import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# SparkleForge Registry
from src.core.tools.registry import registry as global_registry

# MCP imports
try:
    from urllib.parse import urlencode

    import httpx
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.shared.exceptions import McpError
    from mcp.types import ListToolsResult, TextContent

    MCP_AVAILABLE = True
    HTTP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    HTTP_CLIENT_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    sse_client = None
    streamablehttp_client = None
    urlencode = None
    ListToolsResult = None
    TextContent = None
    McpError = Exception  # Fallback
    httpx = None

# FastMCP imports
try:
    import logging as fastmcp_logging

    from fastmcp import Client as FastMCPClient

    # FastMCP 로거 레벨을 warning으로 설정
    fastmcp_logger = fastmcp_logging.getLogger("fastmcp")
    fastmcp_logger.setLevel(fastmcp_logging.WARNING)
    # fastmcp 관련 모든 로거에 대해 warning 레벨 적용
    for logger_name in ["fastmcp", "fastmcp.client", "fastmcp.runner"]:
        logger_instance = fastmcp_logging.getLogger(logger_name)
        logger_instance.setLevel(fastmcp_logging.WARNING)

    # MCP 클라이언트 로거도 필터링 (heartbeat 오류 방지)
    for logger_name in ["mcp", "mcp.client", "mcp.client.streamable_http", "Runner"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.setLevel(logging.WARNING)

        # heartbeat 관련 메시지 필터링
        class HeartbeatFilter(logging.Filter):
            def filter(self, record):
                msg = record.getMessage()
                # heartbeat 관련 오류 메시지 필터링
                if "heartbeat" in msg.lower() or "invalid_token" in msg.lower():
                    return False
                return True

        logger_instance.addFilter(HeartbeatFilter())

    FASTMCP_AVAILABLE = True
except ImportError:
    FastMCPClient = None
    FASTMCP_AVAILABLE = False

# LangChain imports
try:
    from langchain_core.tools import BaseTool, StructuredTool

    # Pydantic v2 호환성 - 최신 LangChain은 pydantic v2 사용
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        try:
            from pydantic.v1 import BaseModel, Field
        except ImportError:
            from langchain_core.pydantic_v1 import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = None
    StructuredTool = None
    BaseModel = None
    Field = None

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import HTTPServerSpec
from src.core.mcp_auto_discovery import FastMCPMulti
from src.core.mcp_tool_loader import MCPToolLoader
from src.core.mcp_tool_loader import ToolInfo as MCPToolInfo
from src.core.observability import start_tool_span
from src.core.researcher_config import get_llm_config, get_mcp_config

logger = logging.getLogger(__name__)



from src.core.mcp_integration.client import OpenRouterClient
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
)
from src.core.mcp_integration.executors.academic import (
    _execute_academic_tool,
    _execute_academic_tool_sync,
)
from src.core.mcp_integration.executors.browser import _execute_browser_tool
from src.core.mcp_integration.executors.code import _execute_code_tool_sync
from src.core.mcp_integration.executors.data import _execute_data_tool, _execute_data_tool_sync
from src.core.mcp_integration.executors.file import _execute_file_tool
from src.core.mcp_integration.executors.git import _execute_git_tool
from src.core.mcp_integration.executors.search import (
    _execute_search_tool,
    _execute_search_tool_sync,
)
from src.core.mcp_integration.executors.shell import _execute_shell_tool
from src.core.mcp_integration.tools import execute_tool, get_mcp_hub
from src.core.tools.registry import ToolCategory, ToolInfo, ToolResult


class UniversalMCPHub:
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

    def _load_tools_config(self):
        """tools_config.json에서 Tool 메타데이터 로드."""
        # configs 폴더에서 로드 시도 (우선)
        tools_config_file = project_root / "configs" / "tools_config.json"
        if not tools_config_file.exists():
            # 하위 호환성: 루트에서도 시도
            tools_config_file = project_root / "tools_config.json"

        if tools_config_file.exists():
            try:
                with open(tools_config_file, encoding="utf-8") as f:
                    self.tools_config = json.load(f)
                logger.info(f"✅ Loaded tools config from {tools_config_file}")
            except Exception as e:
                logger.warning(f"Failed to load tools config: {e}")
                self.tools_config = {}
        else:
            logger.warning(f"tools_config.json not found at {tools_config_file}")
            self.tools_config = {}

    def _create_langchain_tool_wrapper(
        self, tool_name: str, tool_config: Dict[str, Any]
    ) -> BaseTool | None:
        """tools_config.json의 설정을 기반으로 LangChain Tool 래퍼 생성.

        Args:
            tool_name: Tool 이름
            tool_config: tools_config.json에서 로드된 Tool 설정

        Returns:
            LangChain BaseTool 인스턴스 또는 None
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, cannot create tool wrapper")
            return None

        try:
            # 카테고리 매핑
            category_map = {
                "search": ToolCategory.SEARCH,
                "data": ToolCategory.DATA,
                "code": ToolCategory.CODE,
                "academic": ToolCategory.ACADEMIC,
                "business": ToolCategory.BUSINESS,
                "utility": ToolCategory.UTILITY,
                "browser": ToolCategory.BROWSER,
                "document": ToolCategory.DOCUMENT,
                "file": ToolCategory.FILE,
            }

            category_str = tool_config.get("category", "utility")
            category_map.get(category_str, ToolCategory.UTILITY)
            description = _structured_tool_description(tool_config, tool_name)
            tool_config.get("parameters", {})

            # Pydantic 스키마 생성 - 최신 방식으로 단순화 (args_schema 없이도 동작)
            ToolSchema = None
            # LangChain StructuredTool은 args_schema 없이도 함수 시그니처에서 자동으로 파라미터를 추론함
            # 복잡한 동적 스키마 생성을 피하고 함수 파라미터로 처리

            # Tool 실행 함수 선택 (동기 래퍼 생성) - 함수 시그니처 명시
            def create_sync_func(tool_name_str, func_type):
                """동기 함수 래퍼 생성 - 명시적 함수 시그니처로 LangChain이 파라미터 추론."""
                if func_type == "search":

                    def search_wrapper(
                        query: str,
                        max_results: int = 10,
                        num_results: int = 10,
                        format: str = "detailed",
                    ) -> str:
                        params = {"query": query, "format": format}
                        if max_results:
                            params["max_results"] = max_results
                        elif num_results:
                            params["max_results"] = num_results
                        return _execute_search_tool_sync(tool_name_str, params)

                    return search_wrapper
                elif func_type == "academic":

                    def academic_wrapper(
                        query: str, max_results: int = 10, num_results: int = 10
                    ) -> str:
                        params = {"query": query}
                        if max_results:
                            params["max_results"] = max_results
                        elif num_results:
                            params["max_results"] = num_results
                        return _execute_academic_tool_sync(tool_name_str, params)

                    return academic_wrapper
                elif func_type == "data":
                    if tool_name_str == "fetch":

                        def fetch_wrapper(url: str, format: str = "detailed") -> str:
                            return _execute_data_tool_sync("fetch", {"url": url, "format": format})

                        return fetch_wrapper
                    elif tool_name_str == "filesystem":

                        def filesystem_wrapper(path: str, operation: str = "read") -> str:
                            return _execute_data_tool_sync(
                                "filesystem", {"path": path, "operation": operation}
                            )

                        return filesystem_wrapper
                    else:

                        def data_wrapper(**kwargs) -> str:
                            return _execute_data_tool_sync(tool_name_str, kwargs)

                        return data_wrapper
                elif func_type == "code":
                    if "interpreter" in tool_name_str.lower():

                        def code_wrapper(code: str, language: str = "python") -> str:
                            return _execute_code_tool_sync(
                                tool_name_str, {"code": code, "language": language}
                            )

                        return code_wrapper
                    else:

                        def code_wrapper(code: str) -> str:
                            return _execute_code_tool_sync(tool_name_str, {"code": code})

                        return code_wrapper
                else:
                    return None

            # Tool별 실행 함수 매핑
            func = None
            category_str = tool_config.get("category", "utility")

            if tool_name == "g-search":
                func = create_sync_func("g-search", "search")
            elif tool_name == "fetch":
                func = create_sync_func("fetch", "data")
            elif tool_name == "filesystem":
                func = create_sync_func("filesystem", "data")
            elif tool_name == "python_coder":
                func = create_sync_func("python_coder", "code")
            elif tool_name == "code_interpreter":
                func = create_sync_func("code_interpreter", "code")
            elif tool_name == "arxiv":
                func = create_sync_func("arxiv", "academic")
            elif tool_name == "scholar":
                func = create_sync_func("scholar", "academic")
            else:
                # 카테고리 기반으로 자동 선택 시도
                if category_str == "search":
                    func = create_sync_func(tool_name, "search")
                elif category_str == "data":
                    func = create_sync_func(tool_name, "data")
                elif category_str == "code":
                    func = create_sync_func(tool_name, "code")
                elif category_str == "academic":
                    func = create_sync_func(tool_name, "academic")
                elif category_str == "business":
                    # 전용 business executor가 없으면 일반 검색으로 graceful fallback
                    func = create_sync_func("g-search", "search")

            if func is None:
                logger.warning(
                    f"No execution function for tool: {tool_name}, category: {category_str}"
                )

                # 실행 함수가 없어도 기본 래퍼 함수 생성
                def generic_executor(**kwargs):
                    """Generic executor when specific function not available."""
                    raise RuntimeError(
                        f"Tool {tool_name} execution not implemented yet. Please configure execution function."
                    )

                func = generic_executor

            # StructuredTool 생성 - args_schema 없이도 생성 가능하도록
            try:
                if StructuredTool and ToolSchema:
                    langchain_tool = StructuredTool.from_function(
                        func=func,
                        name=tool_name,
                        description=description,
                        args_schema=ToolSchema,
                    )
                elif StructuredTool:
                    # args_schema 없이 생성 (파라미터는 함수 시그니처에서 자동 추론)
                    langchain_tool = StructuredTool.from_function(
                        func=func, name=tool_name, description=description
                    )
                else:
                    return None

                logger.info(f"✅ Created LangChain tool wrapper for {tool_name}")
                return langchain_tool
            except Exception as schema_error:
                # Schema 생성 실패 시 args_schema 없이 재시도
                logger.warning(
                    f"Failed to create tool with schema for {tool_name}: {schema_error}, trying without schema"
                )
                try:
                    if StructuredTool:
                        langchain_tool = StructuredTool.from_function(
                            func=func, name=tool_name, description=description
                        )
                        logger.info(
                            f"✅ Created LangChain tool wrapper for {tool_name} (without schema)"
                        )
                        return langchain_tool
                except Exception as e2:
                    logger.error(f"Failed to create tool without schema for {tool_name}: {e2}")
                    return None

        except Exception as e:
            logger.error(f"Failed to create LangChain tool wrapper for {tool_name}: {e}")
            return None

    def _initialize_tools(self):
        """도구 초기화 - tools_config.json 기반 + FastMCP 자동 발견."""
        # 1. 수동 등록 도구 초기화
        self._initialize_manual_tools()

        # 2. FastMCP 자동 발견 도구 초기화 (비동기)
        # 이미 실행 중인 이벤트 루프가 있으면 태스크로 실행, 없으면 새로 생성
        try:
            # 실행 중인 이벤트 루프 확인
            try:
                loop = asyncio.get_running_loop()
                # 이미 실행 중인 루프가 있으면 태스크로 실행 (asyncio.run() 사용 금지)
                # 태스크를 생성하지만 await하지 않음 (백그라운드 실행)
                loop.create_task(self._initialize_auto_discovered_tools())
                # 태스크가 완료될 때까지 기다리지 않음 (비동기 초기화)
                logger.debug("Auto-discovered MCP tools initialization started as background task")
            except RuntimeError:
                # 실행 중인 루프가 없으면 새 루프에서 실행
                asyncio.run(self._initialize_auto_discovered_tools())
        except Exception as e:
            logger.warning(f"Failed to initialize auto-discovered MCP tools: {e}")
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")
            # 자동 발견 실패 시에도 계속 진행

        # 3. 도구 통합 및 충돌 해결
        self._merge_tools()

    def _initialize_manual_tools(self):
        """수동 등록 도구 초기화 - tools_config.json 기반."""
        local_tools = self.tools_config.get("local_tools", {})

        for tool_name, tool_config in local_tools.items():
            # MCP 전용 Tool은 건너뛰기 (MCP 서버에서 동적 등록됨)
            if tool_config.get("implementation") == "mcp_only":
                continue

            # 카테고리 매핑
            category_map = {
                "search": ToolCategory.SEARCH,
                "data": ToolCategory.DATA,
                "code": ToolCategory.CODE,
                "academic": ToolCategory.ACADEMIC,
                "business": ToolCategory.BUSINESS,
                "utility": ToolCategory.UTILITY,
                "browser": ToolCategory.BROWSER,
                "document": ToolCategory.DOCUMENT,
                "file": ToolCategory.FILE,
            }

            category_str = tool_config.get("category", "utility")
            category = category_map.get(category_str, ToolCategory.UTILITY)
            description = tool_config.get("description", f"{tool_name} tool")

            # ToolInfo 생성
            tool_info = ToolInfo(
                name=tool_name,
                category=category,
                description=description,
                parameters=tool_config.get("parameters", {}),
                mcp_server=tool_config.get("mcp_server_name", ""),
            )

            # LangChain Tool 래퍼 생성
            langchain_tool = self._create_langchain_tool_wrapper(tool_name, tool_config)

            if langchain_tool:
                # Registry에 등록
                self.registry.register_local_tool(tool_info, langchain_tool)
                # 하위 호환성을 위해 self.tools에도 추가
                self.tools[tool_name] = tool_info
                logger.info(f"✅ Registered local tool: {tool_name}")
            else:
                # LangChain wrapper 생성 실패해도 기본 ToolInfo는 등록 (나중에 실행 시도 가능)
                logger.warning(
                    f"⚠️ Failed to create LangChain wrapper for {tool_name}, registering without wrapper"
                )
                self.registry.tools[tool_name] = tool_info
                self.tools[tool_name] = tool_info

        # ========================================================================
        # NATIVE TOOLS REGISTRATION (Overrides/Fallbacks)
        # ========================================================================
        try:
            from langchain_core.tools import Tool

            from src.core.tools.native_search import search_duckduckgo_json

            native_tool_name = "ddg_search"
            logger.info(f"🛠️ Registering Native Tool: {native_tool_name}")

            native_tool_info = ToolInfo(
                name=native_tool_name,
                category=ToolCategory.SEARCH,
                description="Robust native DuckDuckGo search (No MCP required)",
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results"},
                },
                mcp_server="",
            )

            def native_search_wrapper(query: str, max_results: int = 5):
                # Handler for both string input (query only) and structured input
                if isinstance(query, dict):
                    q = query.get("query", "")
                    m = query.get("max_results", 5)
                    return search_duckduckgo_json(q, m)
                return search_duckduckgo_json(query, max_results)

            native_langchain_tool = Tool(
                name=native_tool_name,
                func=native_search_wrapper,
                description="Search DuckDuckGo natively",
            )

            self.registry.register_local_tool(native_tool_info, native_langchain_tool)
            self.tools[native_tool_name] = native_tool_info

            # Also alias 'search' and 'g-search' to this if not present
            for alias in ["search", "g-search"]:
                if alias not in self.tools and alias not in self.registry.tools:
                    self.tools[alias] = native_tool_info
                    self.registry.register_local_tool(
                        native_tool_info, native_langchain_tool
                    )  # Re-registering with same object might verify alias support? No, straightforward.
                    logger.info(f"✅ Aliased '{alias}' to native {native_tool_name}")

        except Exception as e:
            logger.error(f"❌ Failed to register native tools: {e}")

        # ========================================================================
        # GIT TOOLS REGISTRATION
        # ========================================================================
        try:
            from langchain_core.tools import Tool

            git_tools = [
                {
                    "name": "git_status",
                    "description": "Check Git repository status (branch, staged/unstaged files)",
                    "parameters": {
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path (optional, defaults to current directory)",
                        }
                    },
                },
                {
                    "name": "git_commit",
                    "description": "Create a Git commit with automatic message generation",
                    "parameters": {
                        "message": {
                            "type": "string",
                            "description": "Commit message (optional, auto-generated if not provided)",
                        },
                        "auto_stage": {
                            "type": "boolean",
                            "description": "Automatically stage files (default: true)",
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path (optional)",
                        },
                    },
                },
                {
                    "name": "git_push",
                    "description": "Push Git branch to remote repository",
                    "parameters": {
                        "branch": {
                            "type": "string",
                            "description": "Branch to push (optional, defaults to current branch)",
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force push (default: false)",
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path (optional)",
                        },
                    },
                },
                {
                    "name": "git_create_pr",
                    "description": "Create a Pull Request using GitHub CLI",
                    "parameters": {
                        "title": {
                            "type": "string",
                            "description": "PR title (required)",
                        },
                        "body": {"type": "string", "description": "PR body (optional)"},
                        "base": {
                            "type": "string",
                            "description": "Base branch (default: main)",
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path (optional)",
                        },
                    },
                },
                {
                    "name": "git_commit_push_pr",
                    "description": "Complete workflow: commit, push, and create PR in one step",
                    "parameters": {
                        "commit_message": {
                            "type": "string",
                            "description": "Commit message (optional, auto-generated if not provided)",
                        },
                        "pr_title": {
                            "type": "string",
                            "description": "PR title (optional, uses commit message if not provided)",
                        },
                        "pr_body": {
                            "type": "string",
                            "description": "PR body (optional, auto-generated if not provided)",
                        },
                        "base": {
                            "type": "string",
                            "description": "Base branch (default: main)",
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path (optional)",
                        },
                    },
                },
            ]

            for git_tool_config in git_tools:
                tool_name = git_tool_config["name"]
                logger.info(f"🛠️ Registering Git Tool: {tool_name}")

                tool_info = ToolInfo(
                    name=tool_name,
                    category=ToolCategory.GIT,
                    description=git_tool_config["description"],
                    parameters=git_tool_config["parameters"],
                    mcp_server="",
                )

                # LangChain Tool 래퍼 생성
                def create_git_tool_wrapper(tool_name: str):
                    async def git_tool_wrapper(**kwargs):
                        from src.core.mcp_integration import (
                            _execute_git_tool,
                        )

                        result = await _execute_git_tool(tool_name, kwargs)
                        if result.success:
                            return (
                                result.data
                                if isinstance(result.data, dict)
                                else {"result": result.data}
                            )
                        else:
                            return {"error": result.error}

                    return git_tool_wrapper

                # 동기 래퍼 (LangChain Tool은 동기 함수를 기대)
                def sync_git_wrapper(tool_name: str):
                    def wrapper(**kwargs):
                        import asyncio

                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # 이미 실행 중인 루프가 있으면 새 태스크 생성
                                import concurrent.futures

                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(
                                        asyncio.run,
                                        create_git_tool_wrapper(tool_name)(**kwargs),
                                    )
                                    return future.result()
                            else:
                                return loop.run_until_complete(
                                    create_git_tool_wrapper(tool_name)(**kwargs)
                                )
                        except RuntimeError:
                            return asyncio.run(create_git_tool_wrapper(tool_name)(**kwargs))

                    return wrapper

                langchain_tool = Tool(
                    name=tool_name,
                    func=sync_git_wrapper(tool_name),
                    description=git_tool_config["description"],
                )

                self.registry.register_local_tool(tool_info, langchain_tool)
                self.tools[tool_name] = tool_info
                logger.info(f"✅ Registered Git tool: {tool_name}")

        except Exception as e:
            logger.error(f"❌ Failed to register Git tools: {e}", exc_info=True)

        # Registry의 tools를 self.tools와 동기화
        self.tools.update(self.registry.tools)

        logger.info(
            f"✅ Initialized {len(self.registry.tools)} tools in registry ({len(self.registry.langchain_tools)} with LangChain wrappers)"
        )

    async def _initialize_auto_discovered_tools(self):
        """FastMCP를 통한 자동 발견 도구 초기화."""
        # FastMCP 서버 설정 초기화
        self._initialize_fastmcp_servers()

        if not self.fastmcp_servers:
            logger.info("No FastMCP servers configured for auto-discovery")
            return

        # FastMCP 클라이언트 초기화
        self.fastmcp_multi = FastMCPMulti(self.fastmcp_servers)
        self.fastmcp_tool_loader = MCPToolLoader(self.fastmcp_multi)

        try:
            # 도구 자동 발견
            discovered_tools = await self.fastmcp_tool_loader.get_all_tools()
            tool_infos = await self.fastmcp_tool_loader.list_tool_info()

            # 발견된 도구 저장
            for tool, info in zip(discovered_tools, tool_infos):
                tool_name = info.name
                self.auto_discovered_tools[tool_name] = tool
                self.auto_discovered_tool_infos[tool_name] = info

            logger.info(
                f"Auto-discovered {len(discovered_tools)} tools from {len(self.fastmcp_servers)} FastMCP servers"
            )

        except Exception as e:
            logger.error(f"Failed to auto-discover MCP tools: {e}")
            raise

    def _initialize_fastmcp_servers(self):
        """환경 변수 및 구성에서 FastMCP 서버 설정 초기화."""
        # 환경 변수에서 서버 설정 로드 (예: FASTMCP_SERVERS)
        # 실제로는 config나 환경 변수에서 로드해야 함
        # 여기서는 예시로 빈 설정 유지

    def _merge_tools(self):
        """자동 발견 도구와 수동 등록 도구 통합 및 충돌 해결."""
        # 자동 발견된 도구들을 ToolRegistry에 통합
        for tool_name, tool in self.auto_discovered_tools.items():
            tool_info = self.auto_discovered_tool_infos[tool_name]

            # 카테고리 매핑 (MCP ToolInfo -> 기존 ToolCategory)
            category_map = {
                "search": ToolCategory.SEARCH,
                "data": ToolCategory.DATA,
                "code": ToolCategory.CODE,
                "academic": ToolCategory.ACADEMIC,
                "business": ToolCategory.BUSINESS,
                "utility": ToolCategory.UTILITY,
                "browser": ToolCategory.BROWSER,
                "document": ToolCategory.DOCUMENT,
                "file": ToolCategory.FILE,
            }

            # 도구 설명에서 카테고리 추론 (단순 키워드 기반)
            description_lower = tool_info.description.lower()
            category = ToolCategory.UTILITY  # 기본값
            for keyword, cat in category_map.items():
                if keyword in description_lower:
                    category = cat
                    break

            # 기존 ToolInfo 형식으로 변환
            legacy_tool_info = ToolInfo(
                name=tool_name,
                category=category,
                description=tool_info.description,
                parameters={},  # MCP 스키마는 별도 처리
                mcp_server=tool_info.server_guess,
            )

            # 이름 충돌 확인
            if tool_name in self.registry.tools:
                # 충돌 시 자동 발견 도구 우선 (mcp_auto_ 접두사로 구분)
                auto_tool_name = f"mcp_auto_{tool_name}"
                logger.warning(
                    f"Tool name conflict: '{tool_name}' already exists. Using '{auto_tool_name}' for auto-discovered tool."
                )
                legacy_tool_info.name = auto_tool_name

            # Registry에 등록
            self.registry.register_local_tool(legacy_tool_info, tool)

        # Registry와 self.tools 동기화
        self.tools.update(self.registry.tools)
        logger.info(f"✅ Merged tools: {len(self.registry.tools)} total tools in registry")

    def _initialize_clients(self):
        """클라이언트 초기화 - Gemini 직결 사용, OpenRouter 비활성화."""
        self.openrouter_client = None
        logger.info("✅ LLM routed via llm_manager (Gemini direct). OpenRouter disabled.")

    def _resolve_env_vars_in_value(self, value: Any) -> Any:
        """재귀적으로 객체 내의 환경변수 플레이스홀더를 실제 값으로 치환.
        ${VAR_NAME} 또는 $VAR_NAME 형식 지원.
        """
        if isinstance(value, str):
            import re

            # ${VAR_NAME} 또는 $VAR_NAME 패턴 찾기
            pattern = r"\$\{([^}]+)\}|\$(\w+)"

            def replace_env_var(match):
                var_name = match.group(1) or match.group(2)
                env_value = os.getenv(var_name)
                if env_value is not None:
                    return env_value
                # 환경변수가 없으면 원본 유지 (또는 경고)
                logger.warning(f"Environment variable '{var_name}' not found, keeping placeholder")
                return match.group(0)

            result = re.sub(pattern, replace_env_var, value)
            return result
        elif isinstance(value, dict):
            return {k: self._resolve_env_vars_in_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_env_vars_in_value(item) for item in value]
        else:
            return value

    def _check_server_requirements(self, server_name: str, server_config: Dict[str, Any]) -> bool:
        """서버에 필요한 API 키나 환경변수가 있는지 확인.

        Returns:
            True: 서버를 로드해도 됨
            False: API 키가 없어서 스킵해야 함
        """
        # exa 서버는 EXA_API_KEY 필요
        if server_name == "exa" or "exa" in server_name.lower():
            exa_key = os.getenv("EXA_API_KEY")
            if not exa_key:
                return False
            # headers에 Authorization이 필요한 경우 확인
            headers = server_config.get("headers", {})
            if "Authorization" in headers:
                auth_value = headers.get("Authorization", "")
                # 환경변수 치환이 안된 경우 (${EXA_API_KEY} 형태)
                if "${" in auth_value or not auth_value.replace("Bearer ", "").strip():
                    return False

        # stdio 방식 서버는 API 키 불필요 (npx로 실행)
        # 단, github 서버는 GITHUB_TOKEN이 필요함
        if (
            "command" in server_config
            and "httpUrl" not in server_config
            and "url" not in server_config
        ):
            # github 서버는 GITHUB_TOKEN 체크
            if server_name == "github" or "github" in server_name.lower():
                github_token = os.getenv("GITHUB_TOKEN")
                if not github_token:
                    logger.debug(
                        f"[MCP][check.req] server={server_name} requires GITHUB_TOKEN but not set"
                    )
                    return False
                # env 설정에서도 확인
                env_config = server_config.get("env", {})
                if "GITHUB_PERSONAL_ACCESS_TOKEN" in env_config:
                    env_value = env_config["GITHUB_PERSONAL_ACCESS_TOKEN"]
                    # 환경변수 치환이 안된 경우 (${GITHUB_TOKEN} 형태)
                    if isinstance(env_value, str) and "${" in env_value and not github_token:
                        return False
            logger.debug(f"[MCP][check.req] server={server_name} stdio mode, requirements checked")
            return True

        # HTTP 서버는 설정에 따라 API 키가 필요할 수 있음 (서버별로 다름)
        # 각 서버의 headers 설정에서 환경변수로 API 키를 지정할 수 있음

        # 다른 서버들은 API 키가 없어도 사용 가능 (예: ddg_search)
        return True

    def _load_mcp_servers_from_config(self):
        """MCP 서버 설정을 config에서 로드하고 환경변수 치환."""
        # 중복 실행 방지
        if not hasattr(self, "_mcp_servers_loaded"):
            self._mcp_servers_loaded = False

        if hasattr(self, "_mcp_servers_loaded") and self._mcp_servers_loaded:
            logger.debug("[MCP][load.skip] MCP server configs already loaded, skipping")
            return

        try:
            # configs 폴더에서 로드 시도 (우선)
            config_file = project_root / "configs" / "mcp_config.json"
            if not config_file.exists():
                # 하위 호환성: 루트에서도 시도
                config_file = project_root / "mcp_config.json"

            if config_file.exists():
                with open(config_file) as f:
                    config_data = json.load(f)
                    raw_configs = config_data.get("mcpServers", {})

                    # PROJECT_ROOT 주입 (경로 하드코딩 방지)
                    os.environ["PROJECT_ROOT"] = str(project_root)
                    # 환경변수 치환
                    resolved_configs = self._resolve_env_vars_in_value(raw_configs)
                    from src.core.mcp_python import (
                        normalize_mcp_servers_python_commands,
                    )

                    resolved_configs = normalize_mcp_servers_python_commands(resolved_configs)

                    # API 키 확인 및 필터링
                    filtered_configs = {}
                    for server_name, server_config in resolved_configs.items():
                        # disabled 플래그 확인
                        if server_config.get("disabled"):
                            logger.info(f"[MCP][skip.disabled] server={server_name}")
                            continue

                        # API 키가 필요한 서버 확인
                        if not self._check_server_requirements(server_name, server_config):
                            logger.info(
                                f"[MCP][skip.no-api-key] server={server_name} (API key not configured)"
                            )
                            continue

                        filtered_configs[server_name] = server_config

                    self.mcp_server_configs = filtered_configs
                    logger.info(
                        f"✅ Loaded MCP server configs: {list(self.mcp_server_configs.keys())}"
                    )
                    # 로드 완료 플래그 설정
                    self._mcp_servers_loaded = True
            else:
                # 기본 DuckDuckGo MCP 서버 설정
                self.mcp_server_configs = {
                    "ddg_search": {
                        "command": "npx",
                        "args": [
                            "-y",
                            "@modelcontextprotocol/server-duckduckgo-search@latest",
                        ],
                    }
                }
                logger.info("✅ Using default MCP server config for ddg_search")
                # 로드 완료 플래그 설정
                self._mcp_servers_loaded = True

            # FORCE DISABLE FLAKY SERVERS (DDG only) to use native fallbacks
            if "ddg_search" in self.mcp_server_configs:
                logger.info(
                    "🚫 Disabling flaky 'ddg_search' MCP server to use Native Tool fallback"
                )
                del self.mcp_server_configs["ddg_search"]

            # tavily-mcp는 이제 활성화 (사용자가 요청)

            # Ensure we don't default to them either (tavily-mcp 제외)
            keys_to_remove = [k for k in self.mcp_server_configs if k in ["ddg_search"]]
            for k in keys_to_remove:
                del self.mcp_server_configs[k]

        except Exception as e:
            logger.warning(f"Failed to load MCP server configs: {e}")
            self.mcp_server_configs = {}

    def _get_server_specific_settings(
        self, server_name: str, server_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """각 MCP 서버별 특성에 맞는 설정 반환.

        서버별 특성:
        - stdio 서버 (npx 기반): 프로세스 시작 시간 필요, 더 긴 타임아웃
        - HTTP 서버: 빠른 연결, 짧은 타임아웃
        - 특정 서버: 특수 설정 적용
        """
        is_stdio = (
            "httpUrl" not in server_config
            and "url" not in server_config
            and server_config.get("type") != "http"
        )
        is_npx = is_stdio and "npx" in server_config.get("command", "")

        # 서버별 기본 설정
        settings = {
            "timeout": 30.0,
            "max_retries": 1,
        }

        # npx 기반 stdio 서버는 더 긴 타임아웃 필요
        if is_npx:
            settings["timeout"] = 60.0  # npx 다운로드 및 실행 시간 고려

        # 특정 서버별 커스텀 설정 - 타임아웃만 설정 (초기화 지연은 일괄 처리)
        if server_name == "exa":
            settings["timeout"] = 15.0  # HTTP 서버는 빠르지만 여유 시간 확보
        elif server_name == "semantic_scholar":
            settings["timeout"] = 20.0  # HTTP 서버지만 인증 처리 시간 필요
        elif server_name == "context7-mcp":
            settings["timeout"] = 60.0  # Upstash 서버는 초기화 시간 필요 (npx 기반)
        elif server_name == "parallel-search":
            settings["timeout"] = 60.0  # npx 기반 서버
        elif server_name == "unified-search-mcp-server":
            settings["timeout"] = 60.0
        elif server_name in ["tavily-mcp", "WebSearch-MCP"]:
            settings["timeout"] = 60.0  # npx 기반 서버, API 키 검증 시간 필요
        elif server_name == "ddg_search":
            settings["timeout"] = 45.0  # npx 기반 서버는 초기화 시간 필요
        elif server_name in ["fetch", "docfork"]:
            settings["timeout"] = 60.0  # npx 기반 서버
        elif server_name == "arxiv":
            settings["timeout"] = 60.0  # npx 기반 arXiv MCP 서버

        # HTTP 서버는 빠르지만 여유 시간 확보
        if not is_stdio:
            settings["timeout"] = max(settings["timeout"], 20.0)  # 최소 20초

        return settings

    async def _check_connection_health(self, server_name: str) -> bool:
        """Check if existing MCP server connection is healthy.

        Args:
            server_name: Server name to check

        Returns:
            True if connection is healthy, False otherwise
        """
        # FastMCP Client 확인
        if server_name in self.fastmcp_clients:
            try:
                fastmcp_client = self.fastmcp_clients[server_name]
                # FastMCP Client는 context manager이므로 간단한 health check
                # 실제로는 연결 테스트를 위해 list_tools를 호출할 수 있지만,
                # 여기서는 클라이언트 존재 여부만 확인 (성능 고려)
                return fastmcp_client is not None
            except Exception as e:
                logger.debug(f"FastMCP connection health check failed for {server_name}: {e}")
                return False

        # 기존 ClientSession 방식 확인
        if server_name not in self.mcp_sessions:
            return False

        try:
            session = self.mcp_sessions[server_name]
            # Try to list tools as a health check (lightweight operation)
            # This will fail if connection is broken
            if hasattr(session, "list_tools"):
                # Quick health check - just verify session is still valid
                # We don't actually call list_tools to avoid overhead
                return True
            return True  # Assume healthy if session exists
        except Exception as e:
            logger.debug(f"Connection health check failed for {server_name}: {e}")
            return False

    async def _connect_to_mcp_server(
        self, server_name: str, server_config: Dict[str, Any], timeout: float = None
    ):
        """MCP 서버에 연결 - Connection pooling with health check and auto-reconnection."""
        # Lazy loading: 설정이 로드되지 않았다면 로드
        if not self.mcp_server_configs:
            self._load_mcp_servers_from_config()

        # 설정이 없는 서버는 연결 불가
        if server_name not in self.mcp_server_configs:
            logger.error(f"[MCP][connect.error] No config for server={server_name}")
            return False

        # 이미 연결 중인지 확인 (재귀 방지)
        if server_name in self.mcp_sessions and not self.stopping:
            return True

        if self.stopping:
            logger.warning(f"[MCP][skip.stopping] server={server_name}")
            return False

        # Connection pooling: Check if connection already exists and is healthy
        if server_name in self.mcp_sessions:
            is_healthy = await self._check_connection_health(server_name)
            if is_healthy:
                logger.debug(f"[MCP][connect.pool] Reusing existing connection for {server_name}")
                return True
            else:
                logger.warning(
                    f"[MCP][connect.reconnect] Connection unhealthy for {server_name}, reconnecting..."
                )
                # Disconnect unhealthy connection
                try:
                    await self._disconnect_from_mcp_server(server_name)
                except Exception as e:
                    logger.debug(f"Error disconnecting unhealthy connection: {e}")

        # 서버별 설정 가져오기
        server_settings = self._get_server_specific_settings(server_name, server_config)
        if timeout is None:
            timeout = server_settings["timeout"]

        logger.info(
            f"[MCP][connect.start] server={server_name} type={server_config.get('type', 'stdio')} url={(server_config.get('httpUrl') or server_config.get('url'))} timeout={timeout}"
        )
        self.connection_diagnostics[server_name] = {
            "server": server_name,
            "type": (
                "http"
                if (
                    server_config.get("httpUrl")
                    or server_config.get("url")
                    or server_config.get("type") == "http"
                )
                else "stdio"
            ),
            "url": server_config.get("httpUrl") or server_config.get("url"),
            "stage": "start",
            "ok": False,
            "error": None,
            "traceback": None,
            "init_ms": None,
            "list_ms": None,
        }
        if not MCP_AVAILABLE:
            logger.error("MCP package not available")
            return False

        try:
            exit_stack = AsyncExitStack()
            self.exit_stacks[server_name] = exit_stack

            # 서버 타입 확인 (stdio vs HTTP)
            server_type = server_config.get("type", "stdio")
            is_stdio = server_type == "stdio" or (
                "command" in server_config
                and "httpUrl" not in server_config
                and "url" not in server_config
            )

            if is_stdio:
                # stdio 서버 연결 (표준 MCP 방식 - OpenManus 스타일)
                if (
                    not MCP_AVAILABLE
                    or ClientSession is None
                    or StdioServerParameters is None
                    or stdio_client is None
                ):
                    logger.error(f"MCP package not available for stdio server {server_name}")
                    return False

                command = server_config.get("command")
                args = server_config.get("args", [])
                if not command:
                    logger.error(f"No command provided for stdio server {server_name}")
                    return False

                # stdio MCP 서버는 현재 실행 중인 Python 인터프리터(.venv)를 강제로 사용
                # 그래야 fastmcp/httpx 등 프로젝트 가상환경 의존성을 자식 프로세스가 동일하게 본다.
                if command in {"python", "python3"}:
                    import sys

                    command = sys.executable

                # 환경변수 처리 (github 등 env가 필요한 서버)
                env_vars = server_config.get("env", {})
                resolved_env = {}
                if env_vars:
                    for env_key, env_value in env_vars.items():
                        # 환경변수 치환 (${VAR} 형식)
                        if isinstance(env_value, str) and "${" in env_value:
                            import re

                            env_var_pattern = r"\$\{([^}]+)\}"
                            matches = re.findall(env_var_pattern, env_value)
                            resolved_value = env_value
                            for env_var in matches:
                                actual_value = os.getenv(env_var)
                                if actual_value:
                                    resolved_value = resolved_value.replace(
                                        f"${{{env_var}}}", actual_value
                                    )
                                else:
                                    logger.warning(
                                        f"[MCP][stdio.connect] server={server_name} env var {env_var} not found, keeping placeholder"
                                    )
                            resolved_env[env_key] = resolved_value
                        else:
                            resolved_env[env_key] = env_value

                    # 환경변수가 모두 비어있으면 서버 스킵
                    if all(
                        not v or (isinstance(v, str) and "${" in v) for v in resolved_env.values()
                    ):
                        logger.warning(
                            f"[MCP][stdio.connect] server={server_name} required env vars not set, skipping"
                        )
                        self.connection_diagnostics[server_name].update(
                            {
                                "ok": False,
                                "error": "Required environment variables not set",
                                "stage": "failed",
                            }
                        )
                        return False

                logger.info(
                    f"[MCP][stdio.connect] server={server_name} command={command} args={args} env={list(resolved_env.keys()) if resolved_env else 'none'}"
                )

                # npm 캐시 손상 문제 해결: npx 캐시 정리
                if command == "npx":
                    try:
                        import shutil
                        import subprocess

                        # npx 캐시 디렉토리 정리 시도
                        npx_cache_dir = os.path.expanduser("~/.npm/_npx")

                        # ERR_MODULE_NOT_FOUND 오류가 발생하는 경우, 손상된 캐시 디렉토리 전체 삭제
                        if os.path.exists(npx_cache_dir):
                            # zod 모듈 오류가 있는 디렉토리 찾기
                            for item in os.listdir(npx_cache_dir):
                                item_path = os.path.join(npx_cache_dir, item)
                                if os.path.isdir(item_path):
                                    # zod 모듈이 손상된 경우 해당 디렉토리 전체 삭제
                                    zod_path = os.path.join(item_path, "node_modules", "zod")
                                    if os.path.exists(zod_path):
                                        # zod 파일들이 없는 경우 (TAR_ENTRY_ERROR)
                                        zod_external = os.path.join(zod_path, "v3", "external.js")
                                        if not os.path.exists(zod_external):
                                            # 손상된 패키지 디렉토리 전체 삭제
                                            try:
                                                shutil.rmtree(item_path, ignore_errors=True)
                                                logger.info(
                                                    f"[MCP][stdio.connect] Cleaned corrupted npx cache directory: {item}"
                                                )
                                            except Exception as e:
                                                logger.debug(
                                                    f"[MCP][stdio.connect] Failed to remove cache dir {item}: {e}"
                                                )
                    except Exception as e:
                        logger.debug(f"[MCP][stdio.connect] Failed to clean npm cache: {e}")

                try:
                    # 표준 MCP 방식으로 연결 (OpenManus 스타일)
                    # StdioServerParameters에 env 전달
                    server_params = StdioServerParameters(
                        command=command,
                        args=args,
                        env=resolved_env if resolved_env else None,
                    )

                    # AsyncExitStack으로 연결 유지 (OpenManus 방식)
                    stdio_transport = await exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    read, write = stdio_transport
                    session = await exit_stack.enter_async_context(ClientSession(read, write))

                    # 세션 초기화 및 도구 목록 가져오기
                    if self.stopping:
                        raise asyncio.CancelledError("Stopping flag is set, skipping initialize")

                    await asyncio.wait_for(session.initialize(), timeout=timeout)
                    response = await asyncio.wait_for(session.list_tools(), timeout=timeout)

                    # 도구 등록
                    for tool in response.tools:
                        # register_mcp_tool 사용
                        self.registry.register_mcp_tool(server_name, tool, tool)

                        if server_name not in self.mcp_tools_map:
                            self.mcp_tools_map[server_name] = {}
                        # MCPToolInfo 생성 (자동 발견용)
                        tool_info = MCPToolInfo(
                            server_guess=server_name,
                            name=f"{server_name}::{tool.name}",
                            description=tool.description or "",
                            input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                        )
                        self.mcp_tools_map[server_name][tool.name] = tool_info

                    # 세션 저장 (연결 유지)
                    self.mcp_sessions[server_name] = session

                    logger.info(
                        f"[MCP][stdio.connect] ✅ Connected to {server_name}, tools: {len(response.tools)}"
                    )

                    self.connection_diagnostics[server_name].update(
                        {
                            "ok": True,
                            "stage": "connected",
                            "tools_count": len(response.tools),
                        }
                    )

                    return True
                except asyncio.CancelledError:
                    logger.debug(f"[MCP][stdio.connect] Connection cancelled for {server_name}")
                    raise
                except Exception as e:
                    error_str = str(e).lower()
                    error_msg = str(e)

                    # npm 404 에러는 패키지가 존재하지 않으므로 재시도 불필요
                    is_npm_404 = "404" in error_str and (
                        "npm" in error_str
                        or "not found" in error_str
                        or "not in this registry" in error_str
                    )
                    is_server_error = "server error detected" in error_str
                    is_npm_404 = is_npm_404 or (
                        "not found" in error_str and "server error" in error_str
                    )

                    # npm 오류 감지
                    is_npm_enotempty = "enotempty" in error_str or (
                        "npm error" in error_str and "directory not empty" in error_str
                    )
                    is_npm_tar_error = "tar_entry_error" in error_str or (
                        "enoent" in error_str and "zod" in error_str
                    )
                    is_module_not_found = "err_module_not_found" in error_str or (
                        "cannot find module" in error_str and "zod" in error_str
                    )

                    # Connection closed 오류는 서버 연결 실패
                    is_connection_closed = (
                        "connection closed" in error_str or "client failed to connect" in error_str
                    )

                    # npm 캐시 손상 오류 해결: 캐시 정리 후 재시도
                    if (
                        (is_npm_enotempty or is_npm_tar_error or is_module_not_found)
                        and command == "npx"
                        and not is_server_error
                    ):
                        try:
                            import shutil
                            import subprocess

                            # npm cache clean --force 실행
                            try:
                                subprocess.run(
                                    ["npm", "cache", "clean", "--force"],
                                    capture_output=True,
                                    timeout=10,
                                    check=False,
                                )
                            except Exception:
                                pass

                            # npx 캐시 디렉토리 전체 정리 시도
                            npx_cache_dir = os.path.expanduser("~/.npm/_npx")
                            if os.path.exists(npx_cache_dir):
                                # 손상된 패키지 디렉토리 찾기 및 삭제
                                for item in os.listdir(npx_cache_dir):
                                    item_path = os.path.join(npx_cache_dir, item)
                                    if os.path.isdir(item_path):
                                        try:
                                            # zod 모듈이 손상된 경우 해당 디렉토리 전체 삭제
                                            zod_path = os.path.join(
                                                item_path, "node_modules", "zod"
                                            )
                                            if os.path.exists(zod_path):
                                                # zod 파일들이 없는 경우 (TAR_ENTRY_ERROR 또는 MODULE_NOT_FOUND)
                                                zod_external = os.path.join(
                                                    zod_path, "v3", "external.js"
                                                )
                                                if not os.path.exists(zod_external):
                                                    # 손상된 패키지 디렉토리 전체 삭제
                                                    shutil.rmtree(item_path, ignore_errors=True)
                                                    logger.info(
                                                        f"[MCP][stdio.connect] Cleaned corrupted npx cache directory: {item}"
                                                    )
                                        except Exception:
                                            pass

                                # 재시도 (한 번만) - 표준 MCP 방식으로
                                logger.info(
                                    f"[MCP][stdio.connect] Retrying connection to {server_name} after npm cache cleanup..."
                                )
                                try:
                                    # 표준 MCP 방식으로 재시도
                                    retry_server_params = StdioServerParameters(
                                        command=command,
                                        args=args,
                                        env=resolved_env if resolved_env else None,
                                    )

                                    retry_stdio_transport = await exit_stack.enter_async_context(
                                        stdio_client(retry_server_params)
                                    )
                                    retry_read, retry_write = retry_stdio_transport
                                    retry_session = await exit_stack.enter_async_context(
                                        ClientSession(retry_read, retry_write)
                                    )

                                    await retry_session.initialize()
                                    retry_response = await asyncio.wait_for(
                                        retry_session.list_tools(), timeout=timeout
                                    )

                                    # 도구 등록
                                    for tool in retry_response.tools:
                                        self.registry.register_mcp_tool(server_name, tool, tool)
                                        if server_name not in self.mcp_tools_map:
                                            self.mcp_tools_map[server_name] = {}
                                        tool_info = MCPToolInfo(
                                            server_guess=server_name,
                                            name=f"{server_name}::{tool.name}",
                                            description=tool.description or "",
                                            input_schema=(
                                                tool.inputSchema
                                                if hasattr(tool, "inputSchema")
                                                else {}
                                            ),
                                        )
                                        self.mcp_tools_map[server_name][tool.name] = tool_info

                                    # 세션 저장
                                    self.mcp_sessions[server_name] = retry_session

                                    logger.info(
                                        f"[MCP][stdio.connect] ✅ Connected to {server_name} after cache cleanup, tools: {len(retry_response.tools)}"
                                    )
                                    self.connection_diagnostics[server_name].update(
                                        {
                                            "ok": True,
                                            "stage": "connected",
                                            "tools_count": len(retry_response.tools),
                                        }
                                    )
                                    return True
                                except Exception as retry_e:
                                    logger.warning(
                                        f"[MCP][stdio.connect] Retry failed for {server_name}: {retry_e}"
                                    )
                        except Exception as cleanup_e:
                            logger.debug(f"[MCP][stdio.connect] Cache cleanup failed: {cleanup_e}")

                    # 조용히 처리할 오류들 (WARNING 레벨로만 로깅)
                    if is_npm_404:
                        logger.warning(
                            f"[MCP][stdio.connect] Package/Server error for {server_name} (npm 404/Server Error), skipping"
                        )
                    elif is_connection_closed:
                        logger.warning(
                            f"[MCP][stdio.connect] Connection closed for {server_name}, skipping"
                        )
                    else:
                        # 다른 오류는 WARNING 레벨로 로깅
                        logger.warning(
                            f"[MCP][stdio.connect] Failed to connect to {server_name}: {error_msg[:200]}"
                        )

                    self.connection_diagnostics[server_name].update(
                        {
                            "ok": False,
                            "error": error_msg[:200],  # 긴 에러 메시지 자르기
                            "stage": "failed",
                            "is_npm_404": is_npm_404,
                            "is_npm_enotempty": is_npm_enotempty,
                            "is_connection_closed": is_connection_closed,
                        }
                    )

                    # npm 404, Connection closed, Server Error는 재시도 불필요
                    if is_npm_404 or is_connection_closed or is_server_error:
                        return False

                    return False

                except Exception as e:
                    logger.error(
                        f"[MCP][stdio.connect] Error setting up stdio connection for {server_name}: {e}",
                        exc_info=True,
                    )
                    self.connection_diagnostics[server_name].update(
                        {"ok": False, "error": str(e), "stage": "failed"}
                    )
                    return False
            else:
                # HTTP 서버 연결 (기존 로직)
                # FastMCP 기반 연결 (모든 서버를 HTTP로 처리)
                if not FASTMCP_AVAILABLE or FastMCPClient is None:
                    logger.error(f"FastMCP client not available for server {server_name}")
                    return False

                # 서버 설정을 FastMCP 형식으로 변환
                base_url = server_config.get("httpUrl") or server_config.get("url")
                if not base_url:
                    logger.error(f"No URL provided for MCP server {server_name}")
                    return False

            # Headers 구성 (환경 변수 치환 포함)
            headers = server_config.get("headers", {}).copy()

            # 환경 변수 치환 (${VAR} 형식) - Bearer ${API_KEY} 같은 형식 지원
            resolved_headers = {}
            for k, v in headers.items():
                if isinstance(v, str):
                    # ${VAR} 형식이 포함되어 있는지 확인 (전체 값이 ${VAR}이거나 Bearer ${VAR} 같은 형식)
                    import re

                    env_var_pattern = r"\$\{([^}]+)\}"
                    matches = re.findall(env_var_pattern, v)
                    if matches:
                        resolved_value = v
                        for env_var in matches:
                            env_value = os.getenv(env_var, "")
                            if env_value:
                                # ${VAR}를 실제 값으로 치환
                                resolved_value = resolved_value.replace(
                                    f"${{{env_var}}}", env_value
                                )
                                logger.debug(
                                    f"[MCP][auth.env] server={server_name} Resolved {k} from {env_var}"
                                )
                            else:
                                logger.warning(
                                    f"[MCP][auth.env] server={server_name} {env_var} not found in environment"
                                )
                        resolved_headers[k] = resolved_value
                    else:
                        resolved_headers[k] = v
                else:
                    resolved_headers[k] = v

            # Authorization 헤더는 서버 설정에서 명시적으로 지정해야 함
            # 환경변수 치환을 통해 각 서버별 API 키를 설정할 수 있음

            # FastMCP 설정 구성
            # FastMCP는 httpUrl이 아니라 url을 기대함
            # FastMCP는 headers를 지원하므로 Authorization 헤더를 그대로 전달
            server_config_dict = {"url": base_url}

            # headers가 있으면 추가 (FastMCP는 headers를 지원함)
            if resolved_headers:
                server_config_dict["headers"] = resolved_headers

            mcp_config = {"mcpServers": {server_name: server_config_dict}}

            logger.info(
                f"[MCP][fastmcp.connect] server={server_name} url={base_url} headers={list(resolved_headers.keys()) if resolved_headers else 'None'}"
            )

            try:
                # FastMCP Client 직접 사용 (가이드에 따른 올바른 사용법)
                # 기존 클라이언트가 있으면 재사용, 없으면 새로 생성
                if server_name in self.fastmcp_clients:
                    fastmcp_client = self.fastmcp_clients[server_name]
                    logger.debug(
                        f"[MCP][fastmcp.reuse] server={server_name} Reusing existing FastMCP client"
                    )
                else:
                    # FastMCP Client 생성
                    fastmcp_client = FastMCPClient(mcp_config)
                    self.fastmcp_clients[server_name] = fastmcp_client
                    logger.debug(
                        f"[MCP][fastmcp.create] server={server_name} Created new FastMCP client"
                    )

                # FastMCP Client를 Context Manager로 사용 (가이드에 따른 올바른 사용법)
                try:
                    # stopping 플래그 재확인
                    if self.stopping:
                        logger.warning(
                            f"[MCP][skip.stopping] server={server_name} stopping flag is set"
                        )
                        raise asyncio.CancelledError("Stopping flag is set")

                    # Context Manager로 사용하여 연결 테스트 및 도구 목록 가져오기
                    async with fastmcp_client:
                        # stopping 플래그 체크
                        if self.stopping:
                            logger.info(
                                f"[MCP][skip.stopping] server={server_name} stopping flag is set, skipping connection"
                            )
                            raise asyncio.CancelledError("Stopping flag is set")

                        # 도구 목록 가져오기 (타임아웃 설정, shield 제거하여 취소 가능)
                        try:
                            tools = await asyncio.wait_for(
                                fastmcp_client.list_tools(), timeout=timeout
                            )
                        except TimeoutError:
                            logger.warning(
                                f"[MCP][list_tools.timeout] server={server_name} list_tools timeout after {timeout}s"
                            )
                            raise
                        except asyncio.CancelledError:
                            if self.stopping:
                                logger.info(
                                    f"[MCP][list_tools.cancelled] server={server_name} cancelled due to stopping flag"
                                )
                                raise
                            else:
                                logger.warning(
                                    f"[MCP][list_tools.cancelled] server={server_name} list_tools was cancelled unexpectedly"
                                )
                                raise

                    # 도구 정보 저장
                    tools_dict = {}
                    if tools:
                        for tool in tools:
                            tools_dict[tool.name] = {
                                "name": tool.name,
                                "description": getattr(tool, "description", "") or "",
                                "inputSchema": (
                                    getattr(tool, "inputSchema", None)
                                    or getattr(tool, "input_schema", None)
                                    or {}
                                ),
                            }

                    self.mcp_tools_map[server_name] = tools_dict
                    logger.info(
                        f"[MCP][fastmcp.success] server={server_name} Connected, {len(tools_dict)} tools available"
                    )

                    # 연결 진단 정보 업데이트
                    di = self.connection_diagnostics.get(server_name, {})
                    di.update(
                        {
                            "ok": True,
                            "tools_count": len(tools_dict),
                            "client_type": "FastMCP",
                        }
                    )
                    self.connection_diagnostics[server_name] = di

                    # FastMCP Client 인스턴스 저장 (나중에 도구 호출 시 사용)
                    # 주의: FastMCP Client는 context manager이므로, 도구 호출 시마다 async with로 사용해야 함
                    # 세션은 저장하지 않고 클라이언트만 저장
                    self.mcp_sessions[server_name] = fastmcp_client  # FastMCP Client 저장

                    return True

                except Exception as fastmcp_error:
                    error_msg = str(fastmcp_error)
                    error_type = type(fastmcp_error).__name__
                    logger.error(
                        f"[MCP][fastmcp.error] server={server_name} err={error_type}: {error_msg}"
                    )
                    logger.exception(f"[MCP][fastmcp.error] server={server_name} full traceback:")

                    # 연결 실패 시 클라이언트 제거
                    if server_name in self.fastmcp_clients:
                        del self.fastmcp_clients[server_name]

                    di = self.connection_diagnostics.get(server_name, {})
                    di.update(
                        {
                            "stage": "fastmcp_connect",
                            "error": error_msg,
                            "error_type": error_type,
                            "ok": False,
                        }
                    )
                    self.connection_diagnostics[server_name] = di
                    return False

            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                logger.error(
                    f"[MCP][fastmcp.error] server={server_name} err={error_type}: {error_msg}"
                )
                logger.exception(f"[MCP][fastmcp.error] server={server_name} full traceback:")
                di = self.connection_diagnostics.get(server_name, {})
                di.update(
                    {
                        "stage": "fastmcp_connect",
                        "error": error_msg,
                        "error_type": error_type,
                    }
                )
                self.connection_diagnostics[server_name] = di
                if server_name in self.fastmcp_configs:
                    del self.fastmcp_configs[server_name]
                return False

            # 도구 맵 생성 및 Registry에 동적 등록
            self.mcp_tools_map[server_name] = {}
            for tool in response.tools:
                self.mcp_tools_map[server_name][tool.name] = tool
                # ToolRegistry에 server_name::tool_name 형식으로 등록
                self.registry.register_mcp_tool(server_name, tool.name, tool)
                logger.debug(f"[MCP][register] {server_name}::{tool.name}")

            # Registry tools를 self.tools에 동기화
            self.tools.update(self.registry.tools)

            tool_names = [t for t in self.mcp_tools_map.get(server_name, {}).keys()]
            logger.info(f"[MCP][connect.ok] server={server_name} tools={tool_names}")
            logger.info(
                f"✅ Connected to MCP server {server_name} with {len(response.tools)} tools"
            )
            return True

        except asyncio.CancelledError:
            # 작업이 취소된 경우 (종료 신호 등) - 정상적인 동작
            logger.info(
                f"[MCP][connect.cancelled] server={server_name} stage=generic (shutdown in progress)"
            )
            try:
                await self._disconnect_from_mcp_server(server_name)
            except Exception:
                pass  # cleanup 중 오류는 무시
            return False  # raise하지 않고 False 반환하여 다른 서버 연결 계속 진행
        except TimeoutError:
            logger.error(f"[MCP][connect.timeout] server={server_name} stage=generic")
            di = self.connection_diagnostics.get(server_name, {})
            di.update({"stage": "timeout_generic", "error": f"timeout_{timeout}s"})
            self.connection_diagnostics[server_name] = di
            # 타임아웃 발생 시 exit_stack 참조만 제거 (aclose() 호출하지 않음 - anyio 오류 방지)
            if server_name in self.exit_stacks:
                del self.exit_stacks[server_name]
            await self._disconnect_from_mcp_server(server_name)
            return False
        except Exception as e:
            logger.exception(f"[MCP][connect.error] server={server_name} err={e}")
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")
            di = self.connection_diagnostics.get(server_name, {})
            di.update(
                {
                    "stage": "exception",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            self.connection_diagnostics[server_name] = di
            # 실패 시 exit_stack 참조만 제거 (aclose() 호출하지 않음 - anyio 오류 방지)
            if server_name in self.exit_stacks:
                del self.exit_stacks[server_name]
            try:
                await self._disconnect_from_mcp_server(server_name)
            except:
                pass
            return False

    async def _register_dynamic_server(self, server_name: str, server_path: Path) -> bool:
        """동적으로 생성된 MCP 서버를 등록하고 시작.

        Args:
            server_name: 서버 이름
            server_path: 서버 파일 경로

        Returns:
            등록 성공 여부
        """
        try:
            logger.info(f"[MCP][builder.register] Registering dynamic server: {server_name}")

            # 서버 설정 생성 (stdio 방식)
            server_config = {
                "type": "stdio",
                "command": "python",
                "args": [str(server_path)],
            }

            # mcp_server_configs에 추가
            self.mcp_server_configs[server_name] = server_config

            # mcp_config.json에도 추가 (선택적, 영구 저장)
            try:
                config_file = project_root / "configs" / "mcp_config.json"
                if config_file.exists():
                    with open(config_file, encoding="utf-8") as f:
                        config_data = json.load(f)

                    if "mcpServers" not in config_data:
                        config_data["mcpServers"] = {}

                    # 동적 서버 추가 (기존 서버와 충돌 방지)
                    if server_name not in config_data["mcpServers"]:
                        config_data["mcpServers"][server_name] = server_config

                        # 백업 후 저장
                        backup_file = config_file.with_suffix(".json.bak")
                        if not backup_file.exists():
                            import shutil

                            shutil.copy2(config_file, backup_file)

                        with open(config_file, "w", encoding="utf-8") as f:
                            json.dump(config_data, f, indent=2, ensure_ascii=False)
                        logger.debug(
                            f"[MCP][builder.register] Added {server_name} to mcp_config.json"
                        )
            except Exception as config_error:
                logger.warning(
                    f"[MCP][builder.register] Failed to update mcp_config.json: {config_error}"
                )
                # 계속 진행 (메모리에는 등록됨)

            # 서버 연결 시도
            connected = await self._connect_to_mcp_server(server_name, server_config, timeout=30.0)

            if connected:
                logger.info(
                    f"[MCP][builder.register] ✅ Dynamic server registered and connected: {server_name}"
                )

                # ProcessManager에 등록 (서버 프로세스 추적)
                try:
                    from src.core.process_manager import get_process_manager

                    get_process_manager()
                    # 서버 프로세스는 _connect_to_mcp_server에서 시작되므로 여기서는 로깅만
                    logger.debug(
                        f"[MCP][builder.register] Server {server_name} process will be tracked by ProcessManager"
                    )
                except Exception as pm_error:
                    logger.debug(
                        f"[MCP][builder.register] ProcessManager registration skipped: {pm_error}"
                    )

                return True
            else:
                logger.error(
                    f"[MCP][builder.register] ❌ Failed to connect to dynamic server: {server_name}"
                )
                # 등록은 했지만 연결 실패 - 설정은 유지 (재시도 가능)
                return False

        except Exception as e:
            logger.error(
                f"[MCP][builder.register] Failed to register dynamic server {server_name}: {e}",
                exc_info=True,
            )
            return False

    async def _disconnect_from_mcp_server(self, server_name: str):
        """MCP 서버 연결 해제 - 안전한 비동기 정리."""
        try:
            # FastMCP Client 정리
            if server_name in self.fastmcp_clients:
                try:
                    fastmcp_client = self.fastmcp_clients[server_name]
                    # FastMCP Client 명시적 종료 시도
                    if hasattr(fastmcp_client, "close"):
                        try:
                            await asyncio.wait_for(fastmcp_client.close(), timeout=0.5)
                        except (TimeoutError, Exception) as e:
                            logger.debug(
                                f"FastMCP client close timeout/error for {server_name}: {e}"
                            )
                    elif hasattr(fastmcp_client, "__aexit__"):
                        # Context manager의 __aexit__ 호출 시도
                        try:
                            await asyncio.wait_for(
                                fastmcp_client.__aexit__(None, None, None), timeout=0.5
                            )
                        except (TimeoutError, Exception) as e:
                            logger.debug(
                                f"FastMCP client __aexit__ timeout/error for {server_name}: {e}"
                            )
                    # 참조 제거
                    del self.fastmcp_clients[server_name]
                    logger.debug(f"Removed FastMCP client for {server_name}")
                except Exception as e:
                    logger.debug(f"Error removing FastMCP client for {server_name}: {e}")
                    # 오류가 있어도 참조는 제거
                    if server_name in self.fastmcp_clients:
                        del self.fastmcp_clients[server_name]

            # 세션 먼저 제거 및 종료 (heartbeat 무한 루프 방지)
            if server_name in self.mcp_sessions:
                session = self.mcp_sessions[server_name]
                # FastMCP Client인 경우 context manager이므로 명시적 shutdown 불필요
                is_fastmcp_client = (
                    session and FASTMCP_AVAILABLE and isinstance(session, FastMCPClient)
                )
                if not is_fastmcp_client:
                    # 기존 ClientSession 방식
                    try:
                        # 세션 종료 시도 (안전하게) - heartbeat 중지
                        if hasattr(session, "shutdown"):
                            await asyncio.wait_for(session.shutdown(), timeout=1.0)
                        elif hasattr(session, "close"):
                            await asyncio.wait_for(session.close(), timeout=1.0)
                    except (TimeoutError, AttributeError, Exception) as e:
                        logger.debug(f"Session shutdown timeout/error for {server_name}: {e}")
                        # 타임아웃이어도 세션은 제거 (heartbeat 중지)
                # 세션 제거 (heartbeat 무한 루프 방지)
                del self.mcp_sessions[server_name]

            # Exit stack 정리: aclose() 호출하지 않음 (anyio cancel scope 오류 방지)
            # 참조만 제거 - 컨텍스트는 원래 태스크에서 정리됨
            if server_name in self.exit_stacks:
                del self.exit_stacks[server_name]

            if server_name in self.mcp_tools_map:
                del self.mcp_tools_map[server_name]

            logger.debug(f"Disconnected from MCP server: {server_name}")

        except Exception as e:
            logger.debug(f"Error disconnecting from MCP server {server_name}: {e}")
            # 예외가 발생해도 세션/클라이언트는 제거 시도 (heartbeat 무한 루프 방지)
            if server_name in self.mcp_sessions:
                try:
                    del self.mcp_sessions[server_name]
                except:
                    pass
            if server_name in self.fastmcp_clients:
                try:
                    del self.fastmcp_clients[server_name]
                except:
                    pass

    async def initialize_mcp(self):
        """MCP 초기화 - OpenRouter와 MCP 서버 연결."""
        if not self.config.enabled:
            logger.warning("MCP is disabled. Continuing with limited functionality.")
            return
        if self.stopping:
            logger.warning("MCP initialization requested during stopping state; skipping")
            return

        try:
            logger.info("Initializing MCP Hub with MCP servers (no OpenRouter)...")

            # 일괄 초기화 대기 시간 (agent 시작 초기에 모든 서버 준비 시간 확보)
            batch_init_delay = float(os.getenv("MCP_BATCH_INIT_DELAY", "3.0"))  # 기본 3초
            if batch_init_delay > 0:
                logger.info(
                    f"[MCP][init.batch] Waiting {batch_init_delay}s for batch initialization before connecting servers..."
                )
                await asyncio.sleep(batch_init_delay)

            # MCP 서버 연결 (모든 서버) - 병렬 + 타임아웃 적용
            timeout_per_server = float(
                os.getenv("MCP_CONNECT_TIMEOUT", "60")
            )  # 서버당 최대 60초(환경변수로 조정, npx 서버 고려)
            max_concurrency = int(
                os.getenv("MCP_MAX_CONCURRENCY", "3")
            )  # 동시 연결 수 제한 (기본 3개)
            semaphore = asyncio.Semaphore(max_concurrency)
            logger.info(
                f"[MCP][init] max_concurrency={max_concurrency}, timeout_per_server={timeout_per_server}s"
            )

            # disabled=true 설정된 서버는 건너뛰기 + 허용 서버 화이트리스트 적용
            allowlist_str = os.getenv("MCP_ALLOWED_SERVERS", "").strip()
            allowlist = [s.strip() for s in allowlist_str.split(",") if s.strip()]
            base_items = [
                (n, c) for n, c in self.mcp_server_configs.items() if not c.get("disabled")
            ]
            if allowlist:
                # 화이트리스트가 있으면 그것만 연결
                enabled_server_items = [(n, c) for n, c in base_items if n in allowlist]
                logger.info(f"[MCP][allowlist] enabled={[n for n, _ in enabled_server_items]}")
            else:
                # 화이트리스트가 없으면 disabled가 아닌 모든 서버 연결 시도
                enabled_server_items = base_items
                logger.info(
                    f"[MCP][allowlist] not set; connecting to all enabled servers: {[n for n, _ in enabled_server_items]}"
                )

            # 서버별 타임아웃 설정 적용 (재시도 로직 포함)
            async def connect_one_with_settings(name: str, cfg: Dict[str, Any]) -> tuple[str, bool]:
                try:
                    # stopping 플래그 체크
                    if self.stopping:
                        logger.info(f"[MCP][skip.stopping] server={name} stopping flag is set")
                        return name, False

                    async with semaphore:
                        # semaphore 획득 후 다시 체크
                        if self.stopping:
                            logger.info(
                                f"[MCP][skip.stopping] server={name} stopping flag is set after semaphore"
                            )
                            return name, False

                        if cfg.get("disabled"):
                            logger.warning(f"[MCP][skip.disabled] server={name}")
                            return name, False

                        # 서버별 설정 가져오기
                        server_settings = self._get_server_specific_settings(name, cfg)
                        server_timeout = server_settings["timeout"]

                        # 재시도 로직: 타임아웃이나 일시적 에러는 재시도
                        max_connection_retries = 3
                        connection_success = False

                        for retry_attempt in range(max_connection_retries):
                            # 재시도 전 stopping 플래그 체크
                            if self.stopping:
                                logger.info(
                                    f"[MCP][skip.stopping] server={name} stopping flag is set before retry {retry_attempt + 1}"
                                )
                                return name, False

                            try:
                                logger.info(
                                    f"Connecting to MCP server {name} (timeout: {server_timeout}s, attempt {retry_attempt + 1}/{max_connection_retries})..."
                                )
                                # stopping 플래그 체크
                                if self.stopping:
                                    logger.info(
                                        f"[MCP][skip.stopping] server={name} stopping flag is set, skipping connection"
                                    )
                                    return name, False
                                # shield 제거하여 취소 가능하도록 (stopping 플래그로 제어)
                                ok = await self._connect_to_mcp_server(
                                    name, cfg, timeout=server_timeout
                                )
                                if ok:
                                    connection_success = True
                                    if retry_attempt > 0:
                                        logger.info(
                                            f"[MCP][init.success] server={name} connected after {retry_attempt + 1} attempts"
                                        )
                                    break
                                else:
                                    # 연결 실패
                                    if retry_attempt < max_connection_retries - 1:
                                        wait_time = 2**retry_attempt  # 지수 백오프: 1초, 2초
                                        logger.warning(
                                            f"[MCP][init.retry] server={name} connection failed (attempt {retry_attempt + 1}/{max_connection_retries}), retrying in {wait_time}s..."
                                        )
                                        await asyncio.sleep(wait_time)
                                        continue
                                    else:
                                        logger.warning(
                                            f"[MCP][init.failed] server={name} failed after {max_connection_retries} attempts"
                                        )
                                        break

                            except TimeoutError:
                                # 타임아웃 에러는 재시도 가능
                                if retry_attempt < max_connection_retries - 1:
                                    wait_time = 2**retry_attempt  # 지수 백오프: 1초, 2초
                                    logger.warning(
                                        f"[MCP][init.timeout] server={name} timeout (attempt {retry_attempt + 1}/{max_connection_retries}), retrying in {wait_time}s..."
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    logger.warning(
                                        f"[MCP][init.timeout] server={name} timeout after {max_connection_retries} attempts"
                                    )
                                    break

                            except Exception as e:
                                error_str = str(e).lower()
                                error_msg = str(e)

                                # npm 404 에러는 패키지가 존재하지 않으므로 재시도 불필요
                                is_npm_404 = "404" in error_str and (
                                    "npm" in error_str
                                    or "not found" in error_str
                                    or "not in this registry" in error_str
                                    or "package not found" in error_str
                                )

                                # npm ENOTEMPTY 오류는 디렉토리 관련 문제로, 재시도 불필요
                                is_npm_enotempty = "enotempty" in error_str or (
                                    "npm error" in error_str and "directory not empty" in error_str
                                )

                                # Connection closed 오류는 서버 연결 실패로, 재시도 불필요
                                is_connection_closed = (
                                    "connection closed" in error_str
                                    or "client failed to connect" in error_str
                                )

                                # 조용히 처리할 오류들 (재시도 불필요)
                                if is_npm_404:
                                    logger.warning(
                                        f"[MCP][init.skip] server={name} package not found (npm 404), skipping"
                                    )
                                    break
                                elif is_npm_enotempty:
                                    logger.warning(
                                        f"[MCP][init.skip] server={name} npm directory issue (ENOTEMPTY), skipping"
                                    )
                                    break
                                elif is_connection_closed:
                                    logger.warning(
                                        f"[MCP][init.skip] server={name} connection closed, skipping"
                                    )
                                    break

                                # 504, 502, 503 등 서버 에러는 재시도
                                is_retryable = any(
                                    code in error_str
                                    for code in [
                                        "504",
                                        "502",
                                        "503",
                                        "500",
                                        "gateway",
                                        "timeout",
                                        "unavailable",
                                    ]
                                )

                                if is_retryable and retry_attempt < max_connection_retries - 1:
                                    wait_time = 2**retry_attempt  # 지수 백오프: 1초, 2초
                                    logger.warning(
                                        f"[MCP][init.retry] server={name} error (attempt {retry_attempt + 1}/{max_connection_retries}): {error_msg[:100]}, retrying in {wait_time}s..."
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    # 재시도 불가능한 에러 또는 최대 재시도 횟수 초과
                                    logger.warning(
                                        f"[MCP][connect.error] server={name} error: {error_msg[:200]}"
                                    )
                                    break

                        return name, connection_success

                except asyncio.CancelledError:
                    # shutdown 중 취소는 정상적인 동작 - 다른 서버 연결은 계속 진행
                    logger.info(f"[MCP][init.cancelled] server={name} (shutdown in progress)")
                    return name, False
                except Exception as e:
                    logger.exception(f"[MCP][connect.error] server={name} unexpected err={e}")
                    return name, False

            tasks = [
                asyncio.create_task(connect_one_with_settings(n, c))
                for n, c in enabled_server_items
            ]
            # return_exceptions=True로 변경하여 일부 실패해도 계속 진행
            # 전체 초기화 타임아웃 설정 (서버 수 * 타임아웃, 최대 300초)
            total_timeout = min(len(enabled_server_items) * timeout_per_server, 300.0)
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=total_timeout,
                )
            except TimeoutError:
                logger.warning(
                    f"[MCP][init.timeout] MCP initialization timeout after {total_timeout}s, cancelling remaining tasks..."
                )
                # 남은 작업 취소
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # 완료된 작업만 결과 수집
                results = []
                for task in tasks:
                    try:
                        result = await task
                        results.append(result)
                    except (asyncio.CancelledError, Exception):
                        results.append(None)

            # 결과 파싱 (예외가 포함될 수 있음)
            connected_servers = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    server_name = enabled_server_items[i][0]
                    if isinstance(result, asyncio.CancelledError):
                        logger.info(
                            f"[MCP][init.cancelled] server={server_name} (task was cancelled)"
                        )
                    else:
                        logger.warning(f"[MCP][init.exception] server={server_name} error={result}")
                elif isinstance(result, tuple) and len(result) == 2:
                    name, ok = result
                    if ok:
                        connected_servers.append(name)

            if connected_servers:
                logger.info(
                    f"✅ Successfully connected to {len(connected_servers)} MCP servers: {', '.join(connected_servers)}"
                )
            else:
                logger.warning("⚠️ No MCP servers connected successfully")

            # OpenRouter 연결 테스트 제거 (Gemini는 llm_manager 경유)
            logger.info("✅ MCP Hub initialized (OpenRouter disabled)")
            logger.info(f"Available tools: {len(self.tools)}")
            logger.info(f"MCP servers: {list(self.mcp_sessions.keys())}")
            logger.info(f"Primary model: {self.llm_config.primary_model}")
            # 서버별 연결 진단 요약 출력
            if self.connection_diagnostics:
                logger.info("[MCP][diagnostics] server connection summary")
                for name, di in self.connection_diagnostics.items():
                    init_ms = di.get("init_ms")
                    list_ms = di.get("list_ms")
                    logger.info(
                        "[MCP][diag] server=%s type=%s url=%s stage=%s ok=%s init_ms=%s list_ms=%s err=%s",
                        name,
                        di.get("type"),
                        di.get("url"),
                        di.get("stage"),
                        di.get("ok"),
                        f"{init_ms:.0f}" if isinstance(init_ms, (int, float)) else "-",
                        f"{list_ms:.0f}" if isinstance(list_ms, (int, float)) else "-",
                        di.get("error"),
                    )

            # 필수 도구 검증 - 실패 시 warning만
            await self._validate_essential_tools()

        except Exception as e:
            logger.warning(
                f"⚠️ MCP Hub initialization failed: {e} - continuing with graceful degradation"
            )
            logger.info("ℹ️ System will continue with limited functionality (no API calls)")
            # Don't raise, allow graceful degradation

    async def _execute_via_mcp_server(
        self, server_name: str, tool_name: str, params: Dict[str, Any]
    ) -> Any | None:
        """MCP 서버를 통해 도구 실행 (with connection pooling and health check)."""
        # Connection pooling: Check if connection exists and is healthy
        if server_name not in self.mcp_sessions:
            # Lazy loading: 연결 시점에 연결 시도
            connected = await self.ensure_server_connected(server_name)
            if not connected:
                return None
        elif not await self._check_connection_health(server_name):
                logger.warning(f"Connection to {server_name} is unhealthy, reconnecting...")
                # Auto-reconnection
                if server_name in self.mcp_server_configs:
                    try:
                        await self._disconnect_from_mcp_server(server_name)
                    except Exception:
                        pass
                    server_config = self.mcp_server_configs[server_name]
                    connected = await self._connect_to_mcp_server(server_name, server_config)
                    if not connected:
                        logger.error(f"Failed to reconnect to server {server_name}")
                        return None
                else:
                    logger.error(f"Cannot reconnect to {server_name}: no config found")
                    return None

        return await self._execute_via_mcp_server_internal(server_name, tool_name, params)

    async def ensure_server_connected(self, server_name: str) -> bool:
        """서버 연결 보장 (Lazy loading)."""
        if not self.mcp_server_configs:
            self._load_mcp_servers_from_config()
        if server_name in self.mcp_server_configs:
            return await self._connect_to_mcp_server(server_name, self.mcp_server_configs[server_name])
        return False

    async def _execute_via_mcp_server_internal(
        self, server_name: str, tool_name: str, params: Dict[str, Any]
    ) -> Any | None:
        """실제 도구 실행 로직."""
        if server_name not in self.mcp_sessions:
            logger.error(f"Server {server_name} still not in mcp_sessions after connection attempt")
            return None

        if server_name not in self.mcp_tools_map:
            logger.error(
                f"Server {server_name} not in mcp_tools_map. Available: {list(self.mcp_tools_map.keys())}"
            )
            return None

        if tool_name not in self.mcp_tools_map[server_name]:
            available_tools = list(self.mcp_tools_map[server_name].keys())
            logger.error(
                f"Tool {tool_name} not found in server {server_name}. Available tools: {available_tools}"
            )
            return None

        # 재시도 + 재연결(ClosedResource/429) + 간단한 백오프
        max_attempts = 3
        backoff_seconds = [0.5, 1.5, 3.0]

        for attempt in range(max_attempts):
            try:
                # 표준 MCP ClientSession 방식 (우선)
                if server_name in self.mcp_sessions:
                    session = self.mcp_sessions[server_name]
                    logger.debug(
                        f"Calling tool {tool_name} on server {server_name} using ClientSession (attempt {attempt + 1}/{max_attempts})"
                    )

                    if session is None:
                        logger.error(f"[MCP][exec.error] Session is None for {server_name}")
                        return None

                    if not hasattr(session, "call_tool"):
                        logger.error(
                            f"[MCP][exec.error] Session does not have call_tool method: {type(session)}"
                        )
                        return None

                    tool_def = self.mcp_tools_map.get(server_name, {}).get(tool_name)
                    call_params = _normalize_mcp_call_params(tool_def, params)

                    # 기존 ClientSession 방식
                    try:
                        result = await session.call_tool(tool_name, call_params)
                    except McpError as e:
                        error_msg = str(e) if e else ""
                        should_retry_wrapped = (
                            "Missing required argument" in error_msg
                            and "input" in error_msg
                            and "Unexpected keyword argument" in error_msg
                            and "input" not in call_params
                        )
                        if not should_retry_wrapped:
                            raise
                        wrapped_params = {"input": _normalize_mcp_call_params(tool_def, params)}
                        logger.debug(
                            "[MCP][exec.retry] Retrying %s/%s with FastMCP input wrapper",
                            server_name,
                            tool_name,
                        )
                        result = await session.call_tool(tool_name, wrapped_params)

                    # 결과를 TextContent에서 추출 (ClientSession 방식)
                    if result and hasattr(result, "content") and result.content:
                        content_parts = []
                        for item in result.content:
                            if isinstance(item, TextContent):
                                content_parts.append(item.text)
                            else:
                                # 다른 타입의 content도 처리
                                content_parts.append(str(item))

                        content_str = " ".join(content_parts)
                        raw_bytes = len(content_str.encode("utf-8", errors="replace"))
                        # Context-mode interceptor: reduce large tool output before it enters LLM context (95%+ token savings)
                        try:
                            from src.core.context_mode.interceptor import (
                                process as context_mode_process,
                            )
                            from src.core.context_mode.stats import (
                                record_tool_context_savings,
                            )
                            from src.core.input_router import (
                                TRACE_TURN_ID,
                                get_trace_context,
                            )

                            synthetic = {"content": [{"type": "text", "text": content_str}]}
                            processed = context_mode_process(tool_name, synthetic)
                            if processed.get("content") and len(processed["content"]) > 0:
                                block = processed["content"][0]
                                if isinstance(block, dict) and block.get("type") == "text":
                                    content_str = block.get("text", content_str)
                            returned_bytes = len(content_str.encode("utf-8", errors="replace"))
                            ctx = get_trace_context() or {}
                            record_tool_context_savings(
                                tool_name,
                                raw_bytes,
                                returned_bytes,
                                turn_id=ctx.get(TRACE_TURN_ID),
                            )
                        except Exception as interceptor_err:
                            logger.debug("Context-mode interceptor skip: %s", interceptor_err)
                        logger.debug(
                            f"Tool {tool_name} returned content length: {len(content_str)}"
                        )
                        return content_str
                    else:
                        logger.warning(f"Tool {tool_name} returned empty result")
                        return None
                else:
                    logger.error(
                        f"[MCP][exec.error] Server {server_name} not found in fastmcp_clients or mcp_sessions"
                    )
                    return None

            except McpError as e:
                error_msg = str(e) if e else "Unknown MCP error"
                error_code = getattr(e.error, "code", None) if hasattr(e, "error") else None
                error_data = getattr(e.error, "data", None) if hasattr(e, "error") else None

                # 레이트리밋 / 토큰 오류 감지
                is_rate_limit = "Too Many Requests" in error_msg or (error_code == 429)
                is_auth_error = "invalid_token" in error_msg.lower() or (error_code == 401)

                error_details = (
                    f"[MCP][exec.error] server={server_name} tool={tool_name} operation=call_tool"
                )
                if error_code:
                    error_details += f" code={error_code}"
                if error_data:
                    error_details += f" data={error_data}"
                error_details += f" error={error_msg}"
                logger.error(error_details)

                if is_rate_limit and attempt < max_attempts - 1:
                    wait = backoff_seconds[attempt]
                    logger.warning(
                        f"[MCP][exec.retry] Rate limit hit, retrying in {wait}s (attempt {attempt + 2}/{max_attempts})"
                    )
                    await asyncio.sleep(wait)
                    continue

                if is_auth_error:
                    logger.error(
                        "[MCP][auth] invalid or expired token; refresh credentials and re-init MCP"
                    )
                return None

            except (RuntimeError, ConnectionError, OSError) as e:
                # ClosedResourceError, connection reset 등
                error_type = type(e).__name__
                error_msg = str(e)
                closed_like = (
                    "closed" in error_msg.lower() or "connection reset" in error_msg.lower()
                )

                if (
                    closed_like
                    and server_name in self.mcp_server_configs
                    and attempt < max_attempts - 1
                ):
                    logger.warning(
                        f"[MCP][exec.retry] server={server_name} tool={tool_name} connection closed, reconnecting (attempt {attempt + 2}/{max_attempts})"
                    )
                    try:
                        await self._disconnect_from_mcp_server(server_name)
                    except Exception:
                        pass
                    server_config = self.mcp_server_configs[server_name]
                    reconnected = await self._connect_to_mcp_server(server_name, server_config)
                    if reconnected:
                        wait = backoff_seconds[attempt]
                        await asyncio.sleep(wait)
                        continue
                    logger.error(f"[MCP][exec.error] Reconnect failed for {server_name}")
                    # Reconnect failed, session/client is bad or gone
                    if server_name in self.mcp_sessions:
                        del self.mcp_sessions[server_name]
                    if server_name in self.fastmcp_clients:
                        del self.fastmcp_clients[server_name]
                    return None

                logger.error(
                    f"[MCP][exec.error] server={server_name} tool={tool_name} operation=call_tool type={error_type} error={error_msg}"
                )
                # Invalidate session/client on fatal error if it looks like a connection issue
                if closed_like or "broken pipe" in error_msg.lower():
                    if server_name in self.mcp_sessions:
                        logger.warning(
                            f"[MCP][session.invalidate] Removing dead session for {server_name}"
                        )
                        del self.mcp_sessions[server_name]
                    if server_name in self.fastmcp_clients:
                        logger.warning(
                            f"[MCP][client.invalidate] Removing dead FastMCP client for {server_name}"
                        )
                        del self.fastmcp_clients[server_name]

                import traceback

                logger.debug(
                    f"[MCP][exec.exception] server={server_name} tool={tool_name} - Full traceback:\n{traceback.format_exc()}"
                )
                return None

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                logger.error(
                    f"[MCP][exec.error] server={server_name} tool={tool_name} operation=call_tool type={error_type} error={error_msg}"
                )
                import traceback

                logger.debug(
                    f"[MCP][exec.exception] server={server_name} tool={tool_name} - Full traceback:\n{traceback.format_exc()}"
                )
                return None

    async def _validate_essential_tools(self):
        """필수 MCP 도구 검증 - Tool이 등록되어 있는지 확인만 (실제 실행은 선택적)."""
        essential_tools = ["g-search", "fetch", "filesystem"]
        missing_tools = []

        logger.info("Validating essential tools availability...")

        # 등록된 모든 tool 목록 확인
        all_tools = self.registry.get_all_tool_names()
        logger.info(f"Registered tools: {all_tools}")

        for tool in essential_tools:
            # tool_name으로 직접 찾기
            tool_found = False

            # 1. 직접 등록된 tool 확인
            if tool in all_tools:
                tool_found = True
                logger.info(f"✅ Found essential tool: {tool}")

            # 2. server_name::tool_name 형식으로도 찾기
            if not tool_found:
                for registered_name in all_tools:
                    if "::" in registered_name:
                        _, original_tool_name = registered_name.split("::", 1)
                        if original_tool_name == tool:
                            tool_found = True
                            logger.info(f"✅ Found essential tool: {tool} as {registered_name}")
                            break

            if not tool_found:
                missing_tools.append(tool)
                logger.warning(f"⚠️ Essential tool {tool} not found in registry")

        # 누락된 tool이 있으면 경고만 (실제 실행 전까지는 정확한 검증 불가)
        if missing_tools:
            logger.warning(f"⚠️ Some essential tools not found: {missing_tools}")
            logger.warning(
                "⚠️ Tools may be registered later when MCP servers connect or may need manual configuration"
            )
            logger.warning("⚠️ System will continue, but these tools may not be available")
        else:
            logger.info("✅ All essential tools found in registry")

        # 실제 실행 테스트는 선택적 (timeout으로 인한 false negative 방지)
        # Production 환경에서는 실제 사용 시점에 검증하는 것이 더 안전

    async def cleanup(self):
        """MCP 연결 정리 - Production-grade cleanup."""
        logger.info("Cleaning up MCP Hub...")
        # 신규 연결 차단
        self.stopping = True

        # OpenRouter 클라이언트 사용 안 함
        self.openrouter_client = None

        # FastMCP Client 정리 (병렬로 빠르게 종료)
        async def close_fastmcp_client(server_name: str, client: Any):
            """FastMCP Client 종료 헬퍼"""
            try:
                # 명시적 종료 시도
                if hasattr(client, "close"):
                    try:
                        await asyncio.wait_for(client.close(), timeout=0.5)
                    except (TimeoutError, Exception):
                        pass
                elif hasattr(client, "__aexit__"):
                    try:
                        await asyncio.wait_for(client.__aexit__(None, None, None), timeout=0.5)
                    except (TimeoutError, Exception):
                        pass
                logger.debug(f"Closed FastMCP client for {server_name}")
            except Exception as e:
                logger.debug(f"Error closing FastMCP client for {server_name}: {e}")

        # 모든 FastMCP Client를 병렬로 종료 (최대 1초 타임아웃)
        if self.fastmcp_clients:
            close_tasks = [
                close_fastmcp_client(name, client)
                for name, client in list(self.fastmcp_clients.items())
            ]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*close_tasks, return_exceptions=True), timeout=1.0
                )
            except TimeoutError:
                logger.warning("FastMCP clients cleanup timed out (continuing)")
            except Exception as e:
                logger.debug(f"Error during parallel FastMCP cleanup: {e}")
            finally:
                # 참조는 무조건 제거
                self.fastmcp_clients.clear()

        # 모든 MCP 서버 연결 해제 (역순으로 정리)
        server_names = list(self.mcp_sessions.keys())
        for server_name in reversed(server_names):
            try:
                # 세션 제거
                if server_name in self.mcp_sessions:
                    session = self.mcp_sessions.get(server_name)
                    # FastMCP Client인 경우 명시적 종료 시도
                    if (
                        session and isinstance(session, FastMCPClient)
                        if FASTMCP_AVAILABLE
                        else False
                    ):
                        try:
                            # FastMCP Client 명시적 종료
                            if hasattr(session, "close"):
                                await asyncio.wait_for(session.close(), timeout=0.5)
                            elif hasattr(session, "__aexit__"):
                                await asyncio.wait_for(
                                    session.__aexit__(None, None, None), timeout=0.5
                                )
                        except (TimeoutError, Exception) as e:
                            logger.debug(
                                f"FastMCP session close timeout/error for {server_name}: {e}"
                            )
                    elif session and hasattr(session, "shutdown"):
                        # 기존 ClientSession 방식
                        try:
                            await asyncio.wait_for(session.shutdown(), timeout=0.5)  # 타임아웃 단축
                        except:
                            pass
                    del self.mcp_sessions[server_name]

                # Exit stack 정리: anyio cancel scope 오류 무시하고 시도
                if server_name in self.exit_stacks:
                    exit_stack = self.exit_stacks[server_name]
                    try:
                        # anyio RuntimeError는 완전히 무시 (다른 태스크에서 닫히려 할 때 발생)
                        await asyncio.wait_for(exit_stack.aclose(), timeout=2.0)
                    except RuntimeError as e:
                        if "cancel scope" in str(e).lower() or "different task" in str(e).lower():
                            # anyio cancel scope 오류는 무시
                            pass
                        else:
                            logger.debug(
                                f"RuntimeError during exit_stack cleanup for {server_name}: {e}"
                            )
                    except (TimeoutError, Exception) as e:
                        # 기타 오류는 무시
                        logger.debug(f"Error closing exit_stack for {server_name}: {e}")
                    finally:
                        del self.exit_stacks[server_name]

                if server_name in self.mcp_tools_map:
                    del self.mcp_tools_map[server_name]

            except Exception as e:
                logger.debug(f"Error disconnecting from {server_name}: {e}")

        # 정리 완료 대기
        try:
            await asyncio.sleep(0.1)
        except:
            pass

        # 동적으로 생성된 서버 정리 (auto_cleanup이 활성화된 경우)
        if self.config.builder_auto_cleanup:
            try:
                from src.core.mcp_server_builder import get_mcp_server_builder

                get_mcp_server_builder()
                # 빌드된 서버 디렉토리 정리 (선택적)
                # 실제 서버 프로세스는 ProcessManager가 관리하므로 여기서는 로깅만
                logger.debug("[MCP][cleanup] Dynamic servers will be cleaned up by ProcessManager")
            except Exception as e:
                logger.debug(f"[MCP][cleanup] Builder cleanup skipped: {e}")

        logger.info("MCP Hub cleanup completed")

    def start_shutdown(self):
        """외부에서 종료 시작 시 호출 - 신규 연결 차단"""
        self.stopping = True

    async def call_llm_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> Dict[str, Any]:
        """LLM 호출은 llm_manager를 통해 수행하도록 강제 (Gemini 직결)."""
        raise RuntimeError("call_llm_async via MCP Hub is disabled. Use llm_manager for Gemini.")

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        citation_id: str | None = None,
        *,
        _skip_builder_retry: bool = False,
    ) -> Dict[str, Any]:
        """Tool 실행 - MCP 프로토콜만 사용 (9대 혁신: ToolTrace 추적 통합).

        실행 우선순위:
        1. MCP 서버에서 Tool 실행 (server_name::tool_name 형식 또는 tool_name으로 찾기)
        2. 실패 시 명확한 에러 반환 (fallback 없음)

        Args:
            tool_name: 도구 이름
            parameters: 도구 파라미터
            citation_id: Citation ID (optional, ToolTrace 추적용)
            _skip_builder_retry: 내부용. True면 builder 재시도 생략 (재귀 방지)
        """
        import uuid

        start_time = time.time()

        # 실행 컨텍스트에서 execution_id 가져오기 (ROMA 스타일)
        execution_id = None
        try:
            from src.core.recursive_context_manager import ExecutionContext

            ctx = ExecutionContext.get()
            if ctx:
                execution_id = ctx.execution_id
                logger.debug(f"Tool execution in execution context: {execution_id}")
        except Exception as e:
            logger.debug(f"Failed to get ExecutionContext: {e}")

        # 9대 혁신: ToolTrace 추적 준비
        tool_id = f"tool_{uuid.uuid4().hex[:8]}"
        tool_type = _infer_tool_type(tool_name)
        query_str = _format_query_string(tool_name, parameters)

        # 로컬 도구 우선 처리 (suna-style: 실제 동작하는 도구 우선)
        local_tools = {
            # 실제 동작하는 도구들
            "browser_navigate",
            "browser_extract",
            "browser_screenshot",
            "browser_interact",
            "run_shell_command",
            "run_interactive_command",
            "run_background_command",
            "create_file",
            "read_file",
            "write_file",
            "edit_file",
            "list_files",
            "delete_file",
            "filesystem",
            "browser",
            "shell",  # 일반적인 이름도 지원
        }

        if tool_name in local_tools or any(
            tool_name.startswith(prefix) for prefix in ["browser_", "shell_", "file_"]
        ):
            logger.debug(f"Executing local tool: {tool_name}")
            try:
                # ToolResult를 Dict로 변환하여 반환
                if tool_name.startswith("browser") or tool_name == "browser":
                    result = await _execute_browser_tool(tool_name, parameters)  # noqa: F823
                elif tool_name.startswith(("shell", "run_")) or tool_name == "shell":
                    result = await _execute_shell_tool(tool_name, parameters)  # noqa: F823
                elif (
                    tool_name.startswith(
                        (
                            "file",
                            "create_",
                            "read_",
                            "write_",
                            "edit_",
                            "list_",
                            "delete_",
                        )
                    )
                    or tool_name == "filesystem"
                ):
                    result = await _execute_file_tool(tool_name, parameters)  # noqa: F823
                else:
                    # 일반적인 경우 data tool로 처리
                    result = await _execute_data_tool(tool_name, parameters)

                execution_time = time.time() - start_time
                err = _actionable_error_message(tool_name, result.error) if result.error else None
                out = {
                    "success": result.success,
                    "data": result.data,
                    "error": err,
                    "execution_time": result.execution_time,
                    "confidence": result.confidence,
                    "source": "local_tool",
                }
                return _cap_tool_result_for_context(out, tool_name)

            except Exception as e:
                logger.error(f"Local tool execution failed: {tool_name} - {e}")
                execution_time = time.time() - start_time
                return {
                    "success": False,
                    "error": _actionable_error_message(tool_name, e),
                    "execution_time": execution_time,
                    "confidence": 0.0,
                    "source": "local_tool",
                }

        # MCP 서버 정보 추출
        mcp_server = None
        mcp_tool_name = None
        if "::" in tool_name:
            parts = tool_name.split("::", 1)
            mcp_server = parts[0]
            mcp_tool_name = parts[1] if len(parts) > 1 else tool_name

        semantic_scholar_aliases = (
            "semantic_scholar::",
            "semanticscholar::",
            "semantic-scholar-mcp::",
        )
        if tool_name.startswith(semantic_scholar_aliases):
            logger.warning(
                "[MCP][semantic_scholar.disabled] Routing %s through arxiv fallback",
                tool_name,
            )
            fallback_result = await self.execute_tool(
                "arxiv",
                parameters,
                citation_id,
                _skip_builder_retry=True,
            )
            fallback_result["source"] = "semantic_scholar_arxiv_fallback"
            fallback_result.setdefault("metadata", {})
            if isinstance(fallback_result["metadata"], dict):
                fallback_result["metadata"]["requested_tool"] = tool_name
                fallback_result["metadata"]["fallback_tool"] = "arxiv"
            return fallback_result

        # Citation ID가 없으면 생성 (임시)
        if not citation_id:
            try:
                # 전역 citation manager가 있다면 사용, 없으면 임시 생성
                # 실제로는 orchestrator에서 관리하는 citation_id를 전달받아야 함
                citation_id = f"TEMP-{tool_id}"
            except Exception:
                citation_id = f"TEMP-{tool_id}"

        # 출력 매니저 통합
        from src.utils.output_manager import (
            OutputLevel,
            ToolExecutionResult,
            get_output_manager,
        )

        output_manager = get_output_manager()

        # 도구 실행 시작 알림
        await output_manager.output(
            f"🔧 도구 '{tool_name}' 실행 시작...",
            level=OutputLevel.SERVICE,
            agent_name="mcp_integration",
        )

        logger.info(f"[MCP][exec.start] tool={tool_name} params_keys={list(parameters.keys())}")
        logger.info(f"[MCP][exec.start] parameters_preview={str(parameters)[:200]}...")

        # 학술 도구 라우팅 (arxiv, scholar) - MCP 서버 우선 사용
        if tool_name in ["arxiv", "scholar"]:
            logger.info(
                f"[MCP][exec.academic] Routing {tool_name} to _execute_academic_tool (MCP server first)"
            )
            try:
                # 먼저 MCP 서버에서 시도
                mcp_hub = get_mcp_hub()

                # MCP 서버 연결 확인
                if not mcp_hub.mcp_sessions:
                    logger.warning("No MCP servers connected, attempting to initialize...")
                    try:
                        await mcp_hub.initialize_mcp()
                    except Exception as e:
                        logger.warning(f"Failed to initialize MCP servers: {e}")

                # arXiv MCP 서버에서 시도
                mcp_result = None
                if tool_name == "arxiv":
                    # arXiv MCP 서버 도구 찾기
                    if "arxiv" in mcp_hub.mcp_sessions and "arxiv" in mcp_hub.mcp_tools_map:
                        tools = mcp_hub.mcp_tools_map["arxiv"]
                        arxiv_tool_name = None

                        # arxiv_search, arxiv_get_paper 등 찾기
                        for tool_key in tools.keys():
                            tool_lower = tool_key.lower()
                            if "search" in tool_lower or "query" in tool_lower:
                                arxiv_tool_name = tool_key
                                break

                        if arxiv_tool_name:
                            logger.info(f"Using arXiv MCP server with tool: {arxiv_tool_name}")
                            mcp_result = await mcp_hub._execute_via_mcp_server(
                                "arxiv", arxiv_tool_name, parameters
                            )

                # MCP 결과가 있으면 사용, 없으면 로컬 fallback
                if mcp_result:
                    from src.core.mcp_integration import ToolResult

                    tool_result = ToolResult(
                        success=True,
                        data=mcp_result if isinstance(mcp_result, dict) else {"result": mcp_result},
                        execution_time=time.time() - start_time,
                        confidence=0.9,
                    )
                else:
                    # 로컬 fallback
                    from src.core.mcp_integration import (
                        ToolResult,
                        _execute_academic_tool,
                    )

                    tool_result = await _execute_academic_tool(tool_name, parameters)

                execution_time = time.time() - start_time
                logger.info(
                    f"[MCP][exec.academic.success] {tool_name} routing succeeded: success={tool_result.success}"
                )

                # 도구 실행 결과 표시
                result_summary = ""
                if tool_result.success and tool_result.data:
                    if isinstance(tool_result.data, dict) and "results" in tool_result.data:
                        result_count = len(tool_result.data["results"])
                        result_summary = f"{result_count}개 논문 검색됨"
                    else:
                        result_summary = f"데이터 반환됨 ({type(tool_result.data).__name__})"
                elif tool_result.error:
                    result_summary = f"오류: {tool_result.error[:100]}..."

                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=tool_result.success,
                    execution_time=execution_time,
                    result_summary=result_summary,
                    confidence=tool_result.confidence,
                    error_message=tool_result.error,
                )
                await output_manager.output_tool_execution(tool_exec_result)

                out = {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "mcp_academic" if mcp_result else "local_academic",
                }
                return _cap_tool_result_for_context(out, tool_name)
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"[MCP][exec.academic.error] {tool_name} routing failed: {e}",
                    exc_info=True,
                )

                # 도구 실행 실패 결과 표시
                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    execution_time=execution_time,
                    result_summary=f"학술 도구 실행 실패: {str(e)[:100]}...",
                    confidence=0.0,
                    error_message=str(e),
                )
                await output_manager.output_tool_execution(tool_exec_result)

                return {
                    "success": False,
                    "data": None,
                    "error": f"Academic tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0,
                    "source": "academic_routing_failed",
                }

        # 검색 도구는 먼저 라우팅 확인 (도구 찾기 전에)
        if tool_name in ["g-search", "ddg_search", "mcp_search", "tavily", "exa"]:
            logger.info(
                f"[MCP][exec.route] Routing {tool_name} to _execute_search_tool (tool_name type: {type(tool_name)})"
            )
            try:
                from src.core.mcp_integration import ToolResult, _execute_search_tool

                tool_result = await _execute_search_tool(tool_name, parameters)
                execution_time = time.time() - start_time
                logger.info(
                    f"[MCP][exec.route.success] {tool_name} routing succeeded: success={tool_result.success}"
                )

                # 도구 실행 결과 표시
                result_summary = ""
                if tool_result.success and tool_result.data:
                    if isinstance(tool_result.data, dict) and "results" in tool_result.data:
                        result_count = len(tool_result.data["results"])
                        result_summary = f"{result_count}개 결과 검색됨"
                    else:
                        result_summary = f"데이터 반환됨 ({type(tool_result.data).__name__})"
                elif tool_result.error:
                    result_summary = f"오류: {tool_result.error[:100]}..."

                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=tool_result.success,
                    execution_time=execution_time,
                    result_summary=result_summary,
                    confidence=tool_result.confidence,
                    error_message=tool_result.error,
                )
                await output_manager.output_tool_execution(tool_exec_result)

                out = {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "mcp_search",
                }
                return _cap_tool_result_for_context(out, tool_name)
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"[MCP][exec.route.error] {tool_name} routing failed: {e}",
                    exc_info=True,
                )

                # 도구 실행 실패 결과 표시
                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    execution_time=execution_time,
                    result_summary=f"라우팅 실패: {str(e)[:100]}...",
                    confidence=0.0,
                    error_message=str(e),
                )
                await output_manager.output_tool_execution(tool_exec_result)

                # 라우팅 실패 시 일반 도구 찾기로 fallback
                # 하지만 라우팅이 실패하면 검색 도구 자체가 문제이므로 빈 결과 반환
                return {
                    "success": False,
                    "data": None,
                    "error": f"Search tool routing failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0,
                    "source": "mcp_search_routing_failed",
                }

        # 브라우저 도구 라우팅 (우선 처리)
        if tool_name.startswith("browser_"):
            logger.info(f"[MCP][exec.browser] Routing {tool_name} to _execute_browser_tool")
            try:
                # IMPORTANT: avoid binding the name `_execute_browser_tool` in this scope.
                # If we import it with the same identifier, Python treats it as a local variable
                # across the whole function (leading to UnboundLocalError in the local-tool branch).
                from src.core.mcp_integration import (
                    ToolResult,
                )
                from src.core.mcp_integration import (
                    _execute_browser_tool as browser_execute_tool,
                )

                tool_result = await browser_execute_tool(tool_name, parameters)
                execution_time = time.time() - start_time

                result_summary = ""
                if tool_result.success and tool_result.data:
                    if isinstance(tool_result.data, dict):
                        if "extracted_data" in tool_result.data:
                            result_summary = (
                                f"콘텐츠 추출 완료 ({tool_result.data.get('content_length', 0)}자)"
                            )
                        elif "screenshot_path" in tool_result.data:
                            result_summary = f"스크린샷 저장: {tool_result.data['screenshot_path']}"
                        elif "actions" in tool_result.data:
                            result_summary = f"{len(tool_result.data['actions'])}개 액션 실행"
                        else:
                            result_summary = "브라우저 작업 완료"
                    else:
                        result_summary = f"데이터 반환됨 ({type(tool_result.data).__name__})"
                elif tool_result.error:
                    result_summary = f"오류: {tool_result.error[:100]}..."

                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=tool_result.success,
                    execution_time=execution_time,
                    result_summary=result_summary,
                    confidence=tool_result.confidence,
                    error_message=tool_result.error,
                )
                await output_manager.output_tool_execution(tool_exec_result)

                return {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "browser",
                }
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[MCP][exec.browser.error] {tool_name} failed: {e}", exc_info=True)

                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    execution_time=execution_time,
                    result_summary=f"브라우저 도구 실행 실패: {str(e)[:100]}...",
                    confidence=0.0,
                    error_message=str(e),
                )
                await output_manager.output_tool_execution(tool_exec_result)

                return {
                    "success": False,
                    "data": None,
                    "error": f"Browser tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0,
                }

        # 문서 생성 도구 라우팅
        if tool_name.startswith("generate_"):
            logger.info(f"[MCP][exec.document] Routing {tool_name} to _execute_document_tool")
            try:
                from src.core.mcp_integration import ToolResult, _execute_document_tool

                tool_result = await _execute_document_tool(tool_name, parameters)
                execution_time = time.time() - start_time

                result_summary = ""
                if tool_result.success and tool_result.data:
                    if isinstance(tool_result.data, dict) and "file_path" in tool_result.data:
                        result_summary = f"문서 생성 완료: {tool_result.data['file_path']}"
                    else:
                        result_summary = "문서 생성 완료"
                elif tool_result.error:
                    result_summary = f"오류: {tool_result.error[:100]}..."

                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=tool_result.success,
                    execution_time=execution_time,
                    result_summary=result_summary,
                    confidence=tool_result.confidence,
                    error_message=tool_result.error,
                )
                await output_manager.output_tool_execution(tool_exec_result)

                return {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "document",
                }
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[MCP][exec.document.error] {tool_name} failed: {e}", exc_info=True)

                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    execution_time=execution_time,
                    result_summary=f"문서 생성 실패: {str(e)[:100]}...",
                    confidence=0.0,
                    error_message=str(e),
                )
                await output_manager.output_tool_execution(tool_exec_result)

                return {
                    "success": False,
                    "data": None,
                    "error": f"Document tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0,
                }

        # Shell 도구 라우팅
        if tool_name in [
            "run_shell_command",
            "run_interactive_command",
            "run_background_command",
        ]:
            logger.info(f"[MCP][exec.shell] Routing {tool_name} to _execute_shell_tool")
            try:
                # IMPORTANT: avoid binding the name `_execute_shell_tool` in this scope.
                # If we import it with the same identifier, Python treats it as a local variable
                # across the whole function (leading to UnboundLocalError in the local-tool branch).
                from src.core.mcp_integration import (
                    ToolResult,
                )
                from src.core.mcp_integration import (
                    _execute_shell_tool as shell_execute_tool,
                )

                tool_result = await shell_execute_tool(tool_name, parameters)
                execution_time = time.time() - start_time

                result_summary = ""
                if tool_result.success and tool_result.data:
                    if isinstance(tool_result.data, dict):
                        if "stdout" in tool_result.data:
                            stdout_preview = tool_result.data["stdout"][:100]
                            result_summary = f"명령 실행 완료: {stdout_preview}..."
                        elif "pid" in tool_result.data:
                            result_summary = f"백그라운드 작업 시작: PID {tool_result.data['pid']}"
                        else:
                            result_summary = "Shell 명령 실행 완료"
                    else:
                        result_summary = "Shell 명령 실행 완료"
                elif tool_result.error:
                    result_summary = f"오류: {tool_result.error[:100]}..."

                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=tool_result.success,
                    execution_time=execution_time,
                    result_summary=result_summary,
                    confidence=tool_result.confidence,
                    error_message=tool_result.error,
                )
                await output_manager.output_tool_execution(tool_exec_result)

                result_dict = {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "shell",
                }

                # 9대 혁신: ToolTrace 생성
                _create_tool_trace(
                    tool_id=tool_id,
                    citation_id=citation_id or f"TEMP-{tool_id}",
                    tool_type=tool_type,
                    query=query_str,
                    result=result_dict,
                    mcp_server=mcp_server,
                    mcp_tool_name=mcp_tool_name,
                )

                return result_dict
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[MCP][exec.shell.error] {tool_name} failed: {e}", exc_info=True)

                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    execution_time=execution_time,
                    result_summary=f"Shell 명령 실행 실패: {str(e)[:100]}...",
                    confidence=0.0,
                    error_message=str(e),
                )
                await output_manager.output_tool_execution(tool_exec_result)

                return {
                    "success": False,
                    "data": None,
                    "error": f"Shell tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0,
                    "source": "shell",
                }

        # Git 도구 라우팅
        if tool_name in [
            "git_status",
            "git_commit",
            "git_push",
            "git_create_pr",
            "git_commit_push_pr",
            "git_create_branch",
        ]:
            logger.info(f"[MCP][exec.git] Routing {tool_name} to _execute_git_tool")
            try:
                from src.core.mcp_integration import ToolResult, _execute_git_tool

                tool_result = await _execute_git_tool(tool_name, parameters)
                execution_time = time.time() - start_time

                result_summary = ""
                if tool_result.success and tool_result.data:
                    if isinstance(tool_result.data, dict):
                        if "commit_hash" in tool_result.data:
                            result_summary = f"커밋 완료: {tool_result.data['commit_hash'][:8]}"
                        elif "pr_url" in tool_result.data:
                            result_summary = f"PR 생성 완료: {tool_result.data['pr_url']}"
                        elif "branch" in tool_result.data:
                            result_summary = f"브랜치 작업 완료: {tool_result.data['branch']}"
                        else:
                            result_summary = "Git 작업 완료"
                    else:
                        result_summary = "Git 작업 완료"
                elif tool_result.error:
                    result_summary = f"오류: {tool_result.error[:100]}..."

                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=tool_result.success,
                    execution_time=execution_time,
                    result_summary=result_summary,
                    confidence=tool_result.confidence,
                    error_message=tool_result.error,
                )
                await output_manager.output_tool_execution(tool_exec_result)

                result_dict = {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "git",
                }

                # 9대 혁신: ToolTrace 생성
                _create_tool_trace(
                    tool_id=tool_id,
                    citation_id=citation_id or f"TEMP-{tool_id}",
                    tool_type=tool_type,
                    query=query_str,
                    result=result_dict,
                    mcp_server=mcp_server,
                    mcp_tool_name=mcp_tool_name,
                )

                return result_dict
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[MCP][exec.git.error] {tool_name} failed: {e}", exc_info=True)

                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    execution_time=execution_time,
                    result_summary=f"Git 작업 실패: {str(e)[:100]}...",
                    confidence=0.0,
                    error_message=str(e),
                )
                await output_manager.output_tool_execution(tool_exec_result)

                return {
                    "success": False,
                    "data": None,
                    "error": f"Git tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0,
                    "source": "git",
                }

        # 파일 도구 라우팅
        if tool_name in [
            "create_file",
            "read_file",
            "write_file",
            "edit_file",
            "list_files",
            "delete_file",
        ]:
            logger.info(f"[MCP][exec.file] Routing {tool_name} to _execute_file_tool")
            try:
                # IMPORTANT: avoid binding the name `_execute_file_tool` in this scope.
                # If we import it with the same identifier, Python treats it as a local variable
                # across the whole function (leading to UnboundLocalError in the local-tool branch).
                from src.core.mcp_integration import (
                    ToolResult,
                )
                from src.core.mcp_integration import (
                    _execute_file_tool as file_execute_tool,
                )

                tool_result = await file_execute_tool(tool_name, parameters)
                execution_time = time.time() - start_time

                result_summary = ""
                if tool_result.success and tool_result.data:
                    if isinstance(tool_result.data, dict):
                        if "file_path" in tool_result.data:
                            result_summary = f"파일 작업 완료: {tool_result.data['file_path']}"
                        elif "files" in tool_result.data:
                            result_summary = f"{len(tool_result.data['files'])}개 파일/디렉토리"
                        else:
                            result_summary = "파일 작업 완료"
                    else:
                        result_summary = "파일 작업 완료"
                elif tool_result.error:
                    result_summary = f"오류: {tool_result.error[:100]}..."

                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=tool_result.success,
                    execution_time=execution_time,
                    result_summary=result_summary,
                    confidence=tool_result.confidence,
                    error_message=tool_result.error,
                )
                await output_manager.output_tool_execution(tool_exec_result)

                return {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "file",
                }
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[MCP][exec.file.error] {tool_name} failed: {e}", exc_info=True)

                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    execution_time=execution_time,
                    result_summary=f"파일 작업 실패: {str(e)[:100]}...",
                    confidence=0.0,
                    error_message=str(e),
                )
                await output_manager.output_tool_execution(tool_exec_result)

                return {
                    "success": False,
                    "data": None,
                    "error": f"File tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0,
                }

        # Tool 찾기 (server_name::tool_name 또는 tool_name)
        # 먼저 tool_name이 이미 server_name::tool_name 형식인지 확인
        if "::" in tool_name:
            # 이미 전체 이름 형식이면 직접 찾기
            tool_info = self.registry.get_tool_info(tool_name)
        else:
            # tool_name만 주어진 경우 Registry에서 찾기
            tool_info = self.registry.get_tool_info(tool_name)

        # suffix 매칭으로 찾은 경우, MCP 호출 시 이 이름을 써야 함 (2단계 이상 :: 인 경우)
        resolved_registered_name: str | None = None

        # tool_name으로 직접 찾기 실패 시, 모든 MCP 서버에서 server_name::tool_name 형식으로 찾기
        if not tool_info:
            for registered_name in self.registry.get_all_tool_names():
                # 이미 전체 이름 형식이면 정확히 매칭
                if "::" in tool_name and registered_name == tool_name:
                    tool_info = self.registry.get_tool_info(registered_name)
                    resolved_registered_name = registered_name
                    logger.info(f"Found tool by exact match: {tool_name}")
                    break
                # server_name::tool_name 형식에서 tool_name 부분만 추출하여 비교
                elif "::" in registered_name:
                    _, original_tool_name = registered_name.split("::", 1)
                    if original_tool_name == tool_name:
                        tool_info = self.registry.get_tool_info(registered_name)
                        resolved_registered_name = registered_name
                        logger.info(f"Found tool {tool_name} as {registered_name}")
                        break
                    # SEP-986 별칭: legacy 이름을 안전한 MCP 이름으로 정규화해 매칭
                    if "::" in tool_name and original_tool_name == _normalize_mcp_tool_alias(
                        tool_name
                    ):
                        tool_info = self.registry.get_tool_info(registered_name)
                        resolved_registered_name = registered_name
                        logger.info(f"Found tool {tool_name} as {registered_name} (SEP-986 alias)")
                        break
                elif registered_name == tool_name:
                    tool_info = self.registry.get_tool_info(registered_name)
                    resolved_registered_name = registered_name
                    break

        if not tool_info:
            # Registry에서 직접 찾기
            tool_info = self.registry.tools.get(tool_name)

        if not tool_info:
            # 하위 호환성: self.tools에서 찾기
            tool_info = self.tools.get(tool_name)

        if not tool_info:
            # MCP Builder를 통한 자동 서버 생성 시도 (재시도 시 스킵하여 재귀 방지)
            if self.config.builder_enabled and not _skip_builder_retry:
                logger.info(
                    f"[MCP][builder] Tool '{tool_name}' not found, attempting auto-build..."
                )
                try:
                    from src.core.mcp_server_builder import get_mcp_server_builder

                    builder = get_mcp_server_builder()

                    # 서버 빌드
                    build_result = await builder.build_mcp_server(
                        tool_name=tool_name, parameters=parameters, error_context=None
                    )

                    if build_result.get("success"):
                        server_name = build_result["server_name"]
                        server_path = build_result["server_path"]

                        logger.info(f"[MCP][builder] Server built successfully: {server_name}")

                        # 동적 서버 등록
                        registered = await self._register_dynamic_server(server_name, server_path)

                        if registered:
                            logger.info(
                                f"[MCP][builder] Server registered: {server_name}, retrying tool execution..."
                            )
                            # 도구 실행 재시도 (한 번만, builder 스킵하여 재귀 방지)
                            return await self.execute_tool(
                                tool_name, parameters, citation_id, _skip_builder_retry=True
                            )
                        else:
                            logger.warning(
                                f"[MCP][builder] Failed to register server: {server_name}"
                            )
                    else:
                        logger.warning(
                            f"[MCP][builder] Server build failed: {build_result.get('error')}"
                        )
                except Exception as builder_error:
                    logger.error(
                        "[MCP][builder] Builder error: %s: %s",
                        type(builder_error).__name__,
                        builder_error,
                    )

            # 사용 가능한 모든 tool 목록 로깅
            available_tools = self.registry.get_all_tool_names()
            execution_time = time.time() - start_time
            logger.error(f"[MCP][exec.unknown] tool={tool_name} available={available_tools}")

            # 도구 찾기 실패 결과 표시
            available_preview = ", ".join(available_tools[:5]) + (
                "..." if len(available_tools) > 5 else ""
            )
            tool_exec_result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                execution_time=execution_time,
                result_summary=f"알 수 없는 도구. 사용 가능한 도구: {available_preview}",
                confidence=0.0,
                error_message=f"Unknown tool: {tool_name}",
            )
            await output_manager.output_tool_execution(tool_exec_result)

            err = f"Unknown tool: {tool_name}. How to fix: Use one of: {', '.join(available_tools[:10])}."
            return {
                "success": False,
                "data": None,
                "error": _actionable_error_message(tool_name, err),
                "execution_time": execution_time,
                "confidence": 0.0,
            }

        try:
            # 1. MCP Tool인지 확인 및 실행 시도 - tool_info에서 직접 정보 추출
            found_tool_name = tool_name
            mcp_info = None

            # suffix 매칭으로 등록명이 정해진 경우, 반드시 그 이름으로 server/tool 해석 (2단계 이상 :: 대응)
            if resolved_registered_name and "::" in resolved_registered_name:
                _server, _tool = resolved_registered_name.split("::", 1)
                # Re-derive mcp_server/mcp_tool_name from the resolved registered
                # name so trace metadata reflects the actually executed tool
                # rather than the originally requested (possibly mismatched) name.
                mcp_server = _server
                mcp_tool_name = _tool
                if self.registry.is_mcp_tool(resolved_registered_name):
                    mcp_info = self.registry.get_mcp_server_info(resolved_registered_name)
                    found_tool_name = resolved_registered_name
                    logger.info(
                        f"[MCP][exec.resolve] Using resolved name: {tool_name} -> {resolved_registered_name} (server={_server}, tool={_tool})"
                    )

            # tool_info가 있으면 MCP 도구인지 확인하고 mcp_info 추출 (아직 안 찾았을 때만)
            if tool_info and not mcp_info:
                # tool_info에서 mcp_server 정보 확인
                mcp_server = tool_info.mcp_server
                if mcp_server:
                    # server_name::tool_name 형식에서 server_name과 tool_name 추출
                    if "::" in tool_name:
                        server_name, original_tool_name = tool_name.split("::", 1)
                        mcp_info = (server_name, original_tool_name)
                        found_tool_name = tool_name
                        logger.info(
                            f"[MCP][exec.resolve] Using tool_info: {tool_name} -> server={server_name}, tool={original_tool_name}"
                        )
                    else:
                        # tool_name만 있는 경우 tool_info의 mcp_server 사용
                        # tool_name이 실제 서버의 원본 tool name인지 확인 필요
                        # registry에서 찾기
                        for registered_name in self.registry.get_all_tool_names():
                            if registered_name == tool_name and self.registry.is_mcp_tool(
                                registered_name
                            ):
                                mcp_info = self.registry.get_mcp_server_info(registered_name)
                                found_tool_name = registered_name
                                break
                            elif "::" in registered_name:
                                _, original_tool_name = registered_name.split("::", 1)
                                if original_tool_name == tool_name:
                                    mcp_info = self.registry.get_mcp_server_info(registered_name)
                                    found_tool_name = registered_name
                                    logger.info(
                                        f"[MCP][exec.resolve] Found {tool_name} as {registered_name}"
                                    )
                                    break

            # tool_info에서 찾지 못한 경우 기존 로직 사용
            if not mcp_info:
                # 이미 server_name::tool_name 형식인 경우
                if "::" in tool_name:
                    if self.registry.is_mcp_tool(tool_name):
                        mcp_info = self.registry.get_mcp_server_info(tool_name)
                        found_tool_name = tool_name
                        logger.info(f"[MCP][exec.resolve] Using full name: {tool_name}")
                elif self.registry.is_mcp_tool(tool_name):
                    mcp_info = self.registry.get_mcp_server_info(tool_name)
                    found_tool_name = tool_name
                else:
                    # server_name::tool_name 형식으로 찾기
                    for registered_name in self.registry.get_all_tool_names():
                        if "::" in registered_name:
                            server_part, original_tool_name = registered_name.split("::", 1)
                            if original_tool_name == tool_name and self.registry.is_mcp_tool(
                                registered_name
                            ):
                                mcp_info = self.registry.get_mcp_server_info(registered_name)
                                found_tool_name = registered_name
                                logger.info(f"[MCP][exec.resolve] {tool_name} -> {registered_name}")
                                break
                        elif registered_name == tool_name and self.registry.is_mcp_tool(
                            registered_name
                        ):
                            mcp_info = self.registry.get_mcp_server_info(registered_name)
                            found_tool_name = registered_name
                            break

            if mcp_info:
                server_name, original_tool_name = mcp_info

                # MCP 서버 연결 확인
                if server_name in self.mcp_sessions:
                    try:
                        logger.info(
                            f"[MCP][exec.try] server={server_name} tool={tool_name} as={found_tool_name}"
                        )
                        mcp_result = await self._execute_via_mcp_server(
                            server_name, original_tool_name, parameters
                        )

                        if mcp_result:
                            # MCP 결과를 ToolResult 형식으로 변환
                            # 에러 응답 체크
                            import json
                            import re

                            result_lower = str(mcp_result).lower() if mcp_result else ""
                            error_patterns = [
                                r"\b(failed|error|invalid_token|authentication failed)\b",
                                r"\b(401|404|500|502|503|504)\b",
                                r"bad gateway",
                                r"not found",
                                r"unauthorized",
                                r"<!doctype html>",
                                r"<html",
                                r"<title>.*error.*</title>",
                            ]

                            is_error = False
                            error_msg = None
                            for pattern in error_patterns:
                                if re.search(pattern, result_lower):
                                    is_error = True
                                    if "401" in result_lower:
                                        error_msg = "Authentication failed (401)"
                                    elif "404" in result_lower:
                                        error_msg = "Not found (404)"
                                    elif "502" in result_lower or "bad gateway" in result_lower:
                                        error_msg = "Bad gateway (502)"
                                    elif "500" in result_lower:
                                        error_msg = "Internal server error (500)"
                                    else:
                                        error_msg = "Server error detected"
                                    break

                            if is_error:
                                execution_time = time.time() - start_time
                                logger.error(f"MCP tool {tool_name} returned error: {error_msg}")

                                # MCP 도구 에러 결과 표시
                                tool_exec_result = ToolExecutionResult(
                                    tool_name=tool_name,
                                    success=False,
                                    execution_time=execution_time,
                                    result_summary=f"MCP 도구 에러: {error_msg[:100]}...",
                                    confidence=0.0,
                                    error_message=error_msg,
                                )
                                await output_manager.output_tool_execution(tool_exec_result)

                                return {
                                    "success": False,
                                    "data": None,
                                    "error": _actionable_error_message(tool_name, error_msg),
                                    "execution_time": execution_time,
                                    "confidence": 0.0,
                                    "source": "mcp",
                                }

                            # 문자열인 경우 마크다운 파싱 시도
                            if isinstance(mcp_result, str):
                                # JSON 시도
                                parsed_json, parsed_data = _parse_json_text(
                                    mcp_result, context=f"MCP tool {tool_name}"
                                )
                                if parsed_json:
                                    result_data = parsed_data
                                else:
                                    # 마크다운 파싱
                                    results = _parse_markdown_link_results(mcp_result)

                                    if results:
                                        result_data = {"results": results}
                                    else:
                                        # 파싱 실패 시 원본 텍스트
                                        result_data = {"result": mcp_result}
                            else:
                                result_data = (
                                    mcp_result
                                    if isinstance(mcp_result, dict)
                                    else {"result": mcp_result}
                                )

                            execution_time = time.time() - start_time

                            # MCP 도구 성공 결과 표시
                            result_summary = ""
                            if isinstance(result_data, dict):
                                if "results" in result_data and isinstance(
                                    result_data["results"], list
                                ):
                                    result_count = len(result_data["results"])
                                    result_summary = f"{result_count}개 결과 반환됨"
                                elif "result" in result_data:
                                    result_summary = (
                                        f"결과 반환됨 ({type(result_data['result']).__name__})"
                                    )
                                else:
                                    result_summary = f"데이터 반환됨 ({len(result_data)}개 필드)"
                            else:
                                result_summary = f"결과 반환됨 ({type(result_data).__name__})"

                            tool_exec_result = ToolExecutionResult(
                                tool_name=tool_name,
                                success=True,
                                execution_time=execution_time,
                                result_summary=result_summary,
                                confidence=0.9,
                            )
                            await output_manager.output_tool_execution(tool_exec_result)

                            result_dict = {
                                "success": True,
                                "data": result_data,
                                "error": None,
                                "execution_time": execution_time,
                                "confidence": 0.9,
                                "source": "mcp",
                            }

                            # 9대 혁신: ToolTrace 생성
                            _create_tool_trace(
                                tool_id=tool_id,
                                citation_id=citation_id or f"TEMP-{tool_id}",
                                tool_type=tool_type,
                                query=query_str,
                                result=result_dict,
                                mcp_server=mcp_server,
                                mcp_tool_name=mcp_tool_name,
                            )

                            return _cap_tool_result_for_context(result_dict, tool_name)
                    except Exception as mcp_error:
                        execution_time = time.time() - start_time
                        logger.error(
                            f"[MCP][exec.error] server={server_name} tool={tool_name} err={mcp_error}"
                        )

                        # MCP 실행 실패 결과 표시
                        tool_exec_result = ToolExecutionResult(
                            tool_name=tool_name,
                            success=False,
                            execution_time=execution_time,
                            result_summary=f"MCP 실행 실패: {str(mcp_error)[:100]}...",
                            confidence=0.0,
                            error_message=str(mcp_error),
                        )
                        await output_manager.output_tool_execution(tool_exec_result)

                        # MCP 실패 시 에러 반환 (Actionable error message)
                        return {
                            "success": False,
                            "data": None,
                            "error": _actionable_error_message(tool_name, mcp_error),
                            "execution_time": execution_time,
                            "confidence": 0.0,
                            "source": "mcp",
                        }

            # MCP 도구가 아닌 경우 로컬 도구 확인
            tool_info = self.registry.get_tool_info(tool_name)
            if tool_info and self.registry.tool_sources.get(tool_name) == "local":
                # 로컬 도구 실행
                logger.info(f"[MCP][exec.local] Executing local tool: {tool_name}")
                try:
                    # 로컬 도구는 카테고리에 따라 다른 실행 함수 사용
                    category = tool_info.category

                    if category == ToolCategory.SEARCH:
                        from src.core.mcp_integration import (
                            ToolResult,
                            _execute_search_tool,
                        )

                        tool_result = await _execute_search_tool(tool_name, parameters)
                    elif category == ToolCategory.DATA:
                        # IMPORTANT: alias to avoid shadowing the module-level name in this scope
                        # (a same-name local import makes it local across the whole function,
                        # causing UnboundLocalError in the earlier local-tool branch).
                        from src.core.mcp_integration import (
                            _execute_data_tool as data_execute_tool,
                        )

                        tool_result = await data_execute_tool(tool_name, parameters)
                    elif category == ToolCategory.CODE:
                        from src.core.mcp_integration import _execute_code_tool

                        tool_result = await _execute_code_tool(tool_name, parameters)
                    elif category == ToolCategory.ACADEMIC:
                        from src.core.mcp_integration import _execute_academic_tool

                        tool_result = await _execute_academic_tool(tool_name, parameters)
                    elif category == ToolCategory.GIT:
                        from src.core.mcp_integration import _execute_git_tool

                        tool_result = await _execute_git_tool(tool_name, parameters)
                    else:
                        # 기본적으로 데이터 도구로 처리 (동일 이름 shadowing 방지 alias)
                        from src.core.mcp_integration import (
                            _execute_data_tool as data_execute_tool,
                        )

                        tool_result = await data_execute_tool(tool_name, parameters)

                    execution_time = time.time() - start_time

                    # 결과 요약 생성
                    result_summary = ""
                    if tool_result.success and tool_result.data:
                        if isinstance(tool_result.data, dict):
                            if "results" in tool_result.data:
                                result_count = len(tool_result.data["results"])
                                result_summary = f"{result_count}개 결과 반환됨"
                            elif "content" in tool_result.data:
                                content_len = len(str(tool_result.data["content"]))
                                result_summary = f"콘텐츠 반환됨 ({content_len}자)"
                            else:
                                result_summary = (
                                    f"데이터 반환됨 ({type(tool_result.data).__name__})"
                                )
                        else:
                            result_summary = f"결과 반환됨 ({type(tool_result.data).__name__})"
                    elif tool_result.error:
                        result_summary = f"오류: {tool_result.error[:100]}..."

                    tool_exec_result = ToolExecutionResult(
                        tool_name=tool_name,
                        success=tool_result.success,
                        execution_time=execution_time,
                        result_summary=result_summary,
                        confidence=tool_result.confidence,
                        error_message=tool_result.error,
                    )
                    await output_manager.output_tool_execution(tool_exec_result)

                    result_dict = {
                        "success": tool_result.success,
                        "data": tool_result.data,
                        "error": tool_result.error,
                        "execution_time": execution_time,
                        "confidence": tool_result.confidence,
                        "source": "local",
                    }

                    # 9대 혁신: ToolTrace 생성
                    _create_tool_trace(
                        tool_id=tool_id,
                        citation_id=citation_id or f"TEMP-{tool_id}",
                        tool_type=tool_type,
                        query=query_str,
                        result=result_dict,
                        mcp_server=mcp_server,
                        mcp_tool_name=mcp_tool_name,
                    )

                    return result_dict

                except Exception as local_error:
                    execution_time = time.time() - start_time
                    logger.error(
                        f"[MCP][exec.local.error] Local tool execution failed: {local_error}"
                    )

                    tool_exec_result = ToolExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        execution_time=execution_time,
                        result_summary=f"로컬 도구 실행 실패: {str(local_error)[:100]}...",
                        confidence=0.0,
                        error_message=str(local_error),
                    )
                    await output_manager.output_tool_execution(tool_exec_result)

                    return {
                        "success": False,
                        "data": None,
                        "error": f"Local tool execution failed: {str(local_error)}",
                        "execution_time": execution_time,
                        "confidence": 0.0,
                        "source": "local",
                    }

            # MCP 도구도 로컬 도구도 아닌 경우 MCP Builder 시도 (재시도 시 한 번만 시도하므로 스킵)
            if self.config.builder_enabled and not _skip_builder_retry:
                logger.info(
                    f"[MCP][builder] Tool '{tool_name}' not available, attempting auto-build..."
                )
                try:
                    from src.core.mcp_server_builder import get_mcp_server_builder

                    builder = get_mcp_server_builder()

                    # 서버 빌드
                    build_result = await builder.build_mcp_server(
                        tool_name=tool_name,
                        parameters=parameters,
                        error_context="Tool not found in MCP servers or local tools",
                    )

                    if build_result.get("success"):
                        server_name = build_result["server_name"]
                        server_path = build_result["server_path"]

                        logger.info(f"[MCP][builder] Server built successfully: {server_name}")

                        # 동적 서버 등록
                        registered = await self._register_dynamic_server(server_name, server_path)

                        if registered:
                            logger.info(
                                f"[MCP][builder] Server registered: {server_name}, retrying tool execution..."
                            )
                            # 도구 실행 재시도 (한 번만, builder 재시도 스킵하여 재귀 방지)
                            return await self.execute_tool(
                                tool_name, parameters, citation_id, _skip_builder_retry=True
                            )
                        else:
                            logger.warning(
                                f"[MCP][builder] Failed to register server: {server_name}"
                            )
                    else:
                        logger.warning(
                            f"[MCP][builder] Server build failed: {build_result.get('error')}"
                        )
                except Exception as builder_error:
                    # exc_info=True 제거: 긴 체인 포맷 시 RecursionError 유발 방지
                    logger.error(
                        "[MCP][builder] Builder error: %s: %s",
                        type(builder_error).__name__,
                        builder_error,
                    )

            # MCP 도구도 로컬 도구도 아닌 경우 에러 반환
            error_msg = f"Tool '{tool_name}' is not available (neither MCP nor local)"
            execution_time = time.time() - start_time
            logger.error(f"[MCP][exec.error] {error_msg}")

            # 도구 없음 결과 표시
            tool_exec_result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                execution_time=execution_time,
                result_summary="도구를 사용할 수 없음 (MCP 서버 및 로컬 도구 모두 확인됨)",
                confidence=0.0,
                error_message=error_msg,
            )
            await output_manager.output_tool_execution(tool_exec_result)

            return {
                "success": False,
                "data": None,
                "error": error_msg,
                "execution_time": execution_time,
                "confidence": 0.0,
                "source": "unknown",
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"[MCP][exec.error] tool={tool_name} err={e}")

            # 일반 예외 결과 표시
            tool_exec_result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                execution_time=execution_time,
                result_summary=f"예외 발생: {str(e)[:100]}...",
                confidence=0.0,
                error_message=str(e),
            )
            await output_manager.output_tool_execution(tool_exec_result)

            return {
                "success": False,
                "data": None,
                "error": str(e),
                "execution_time": execution_time,
                "confidence": 0.0,
            }

    def get_tool_for_category(self, category: ToolCategory) -> str | None:
        """카테고리에 해당하는 도구 반환 - Registry 기반."""
        tools_in_category = self.registry.get_tools_by_category(category)
        return tools_in_category[0] if tools_in_category else None

    def get_allowed_tools(self, trust: Any | None = None) -> List[str]:
        """Return tools filtered by the current trust context."""
        if trust is None:
            from src.core.trust_gate import get_current_trust_context

            trust = get_current_trust_context()

        allowed: List[str] = []
        for tool_name in self.registry.get_all_tool_names():
            info = self.registry.get_tool_info(tool_name)
            mcp_server = info.mcp_server if info else None
            if trust.allows_tool(tool_name, mcp_server):
                allowed.append(tool_name)
        return allowed

    def get_available_tools(self) -> List[str]:
        """사용 가능한 도구 목록 반환 - Registry 기반."""
        return self.get_allowed_tools()

    async def get_tool_for_execution(
        self, tool_name: str, execution_id: str | None = None
    ) -> Any | None:
        """실행 컨텍스트별 도구 반환 (ROMA 스타일).

        각 실행마다 독립적인 도구 인스턴스를 관리하여 실행 간 격리를 보장합니다.

        Args:
            tool_name: 도구 이름
            execution_id: 실행 ID (None이면 ExecutionContext에서 가져옴)

        Returns:
            도구 인스턴스 또는 None
        """
        # ExecutionContext에서 execution_id 가져오기
        if execution_id is None:
            try:
                from src.core.recursive_context_manager import ExecutionContext

                ctx = ExecutionContext.get()
                if ctx:
                    execution_id = ctx.execution_id
            except Exception:
                pass

        # execution_id가 없으면 기본 도구 반환 (하위 호환성)
        if not execution_id:
            return self.registry.get_tool(tool_name)

        # 실행별 세션 초기화
        if execution_id not in self._execution_sessions:
            self._execution_sessions[execution_id] = {
                "tools": {},
                "created_at": datetime.now(),
            }

        execution_session = self._execution_sessions[execution_id]

        # 도구가 이미 캐시되어 있으면 반환
        if tool_name in execution_session["tools"]:
            return execution_session["tools"][tool_name]

        # 도구 초기화 및 캐싱
        # LangChain Tool이 있으면 반환, 없으면 ToolInfo 반환
        tool = self.registry.get_langchain_tool(tool_name)
        if not tool:
            # LangChain Tool이 없으면 ToolInfo 반환
            tool = self.registry.get_tool_info(tool_name)

        if tool:
            execution_session["tools"][tool_name] = tool
            logger.debug(f"Tool {tool_name} cached for execution {execution_id}")

        return tool

    async def cleanup_execution(self, execution_id: str):
        """실행 종료 시 세션 정리 (ROMA 스타일).

        실행별로 관리된 도구 인스턴스와 세션을 정리합니다.

        Args:
            execution_id: 정리할 실행 ID
        """
        if execution_id in self._execution_sessions:
            session = self._execution_sessions[execution_id]
            tools_count = len(session.get("tools", {}))

            # 세션 정리
            del self._execution_sessions[execution_id]

            logger.info(f"Cleaned up execution session {execution_id} ({tools_count} tools)")
        else:
            logger.debug(f"Execution session {execution_id} not found (already cleaned up?)")

    def get_all_langchain_tools(self) -> List[BaseTool]:
        """모든 LangChain Tool 리스트 반환."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available")
            return []
        return self.registry.get_all_langchain_tools()

    async def check_mcp_servers(self) -> Dict[str, Any]:
        """모든 MCP 서버 연결 상태 확인 - mcp_config.json에 정의된 모든 서버."""
        server_status = {
            "timestamp": datetime.now().isoformat(),
            "total_servers": len(self.mcp_server_configs),
            "connected_servers": len(self.mcp_sessions),
            "servers": {},
        }

        logger.info(f"Checking {len(self.mcp_server_configs)} MCP servers...")

        for server_name, server_config in self.mcp_server_configs.items():
            server_info = {
                "name": server_name,
                "type": server_config.get("type", "stdio"),
                "connected": server_name in self.mcp_sessions,
                "tools_count": 0,
                "tools": [],
                "error": None,
            }

            # 연결 타입 정보
            if (
                server_config.get("type") == "http"
                or "httpUrl" in server_config
                or "url" in server_config
            ):
                server_info["type"] = "http"
                server_info["url"] = server_config.get("httpUrl") or server_config.get(
                    "url", "unknown"
                )
            else:
                server_info["type"] = "stdio"
                server_info["command"] = server_config.get("command", "unknown")
                server_info["args"] = server_config.get("args", [])

            # 연결 상태 확인
            if server_name in self.mcp_sessions:
                server_info["connected"] = await self._check_connection_health(server_name)
                if not server_info["connected"]:
                    server_info["error"] = "Session health check failed"

                # 제공하는 Tool 목록 확인
                if server_name in self.mcp_tools_map:
                    tools = self.mcp_tools_map[server_name]
                    server_info["tools_count"] = len(tools)
                    server_info["tools"] = list(tools.keys())

                    # 등록된 Tool 이름 (server_name::tool_name 형식)
                    registered_tools = [
                        name
                        for name in self.registry.get_all_tool_names()
                        if name.startswith(f"{server_name}::")
                    ]
                    server_info["registered_tools"] = registered_tools
                else:
                    server_info["tools_count"] = 0
                    server_info["tools"] = []
                    server_info["error"] = "No tools discovered"
            else:
                server_info["connected"] = False
                server_info["error"] = "Not connected"
                # 연결 시도는 하지 않음 (별도의 initialize_mcp 호출 필요)
                # check_mcp_servers는 상태 확인만 수행

            server_status["servers"][server_name] = server_info

        # 통계 요약
        connected = sum(1 for s in server_status["servers"].values() if s["connected"])
        total_tools = sum(s["tools_count"] for s in server_status["servers"].values())

        server_status["summary"] = {
            "connected_servers": connected,
            "total_servers": len(self.mcp_server_configs),
            "total_tools_available": total_tools,
            "connection_rate": f"{connected}/{len(self.mcp_server_configs)}",
        }

        return server_status

    def print_server_status(self, server_status: Dict[str, Any], verbose: bool = False) -> None:
        """CLI용: :meth:`check_mcp_servers` 결과를 사람이 읽기 쉽게 출력."""
        summary = server_status.get("summary", {})
        print("=" * 80)
        print("MCP server status")
        print("=" * 80)
        print(f"Timestamp: {server_status.get('timestamp', '')}")
        print(f"Total configured: {server_status.get('total_servers', 0)}")
        print(f"Connected sessions: {server_status.get('connected_servers', 0)}")
        print(f"Connection rate: {summary.get('connection_rate', 'n/a')}")
        print(f"Total tools (discovered): {summary.get('total_tools_available', 0)}")
        print()
        for name, info in server_status.get("servers", {}).items():
            icon = "OK " if info.get("connected") else "ERR"
            print(
                f"[{icon}] {name}  tools={info.get('tools_count', 0)}  "
                f"type={info.get('type', '?')}"
            )
            if verbose and info.get("error"):
                print(f"      error: {info['error']}")
            if verbose and info.get("tools"):
                for t in info["tools"][:15]:
                    print(f"      - {name}::{t}")
                if len(info["tools"]) > 15:
                    print(f"      ... +{len(info['tools']) - 15} more")
        print("=" * 80)

    async def health_check(self) -> Dict[str, Any]:
        """강화된 헬스 체크 - OpenRouter, Gemini 2.5 Flash Lite, MCP 도구 검증."""
        try:
            health_status = {
                "mcp_enabled": self.config.enabled,
                "tools_available": len(self.tools),
                "timestamp": datetime.now().isoformat(),
            }

            # 1. OpenRouter 연결 테스트
            try:
                test_messages = [
                    {"role": "system", "content": "Health check test."},
                    {
                        "role": "user",
                        "content": "Respond with 'OK' if you can process this request.",
                    },
                ]

                test_response = await self.openrouter_client.generate_response(
                    model=self.llm_config.primary_model,
                    messages=test_messages,
                    temperature=0.1,
                    max_tokens=50,
                )

                openrouter_healthy = test_response and "choices" in test_response
                health_status.update(
                    {
                        "openrouter_connected": openrouter_healthy,
                        "primary_model": self.llm_config.primary_model,
                        "rate_limit_remaining": getattr(
                            self.openrouter_client, "rate_limit_remaining", "unknown"
                        ),
                    }
                )

                if not openrouter_healthy:
                    health_status["overall_health"] = "unhealthy"
                    health_status["critical_error"] = "OpenRouter connection failed"
                    return health_status

            except Exception as e:
                health_status.update(
                    {
                        "openrouter_connected": False,
                        "openrouter_error": str(e),
                        "overall_health": "unhealthy",
                        "critical_error": f"OpenRouter health check failed: {e}",
                    }
                )
                return health_status

            # 2. 필수 MCP 도구 검증
            essential_tools = ["g-search", "fetch", "filesystem"]
            tool_health = {}
            failed_tools = []

            for tool in essential_tools:
                try:
                    # 간단한 테스트 실행
                    if tool == "g-search":
                        test_result = await execute_tool(tool, {"query": "test", "max_results": 1})
                    elif tool == "fetch":
                        test_result = await execute_tool(tool, {"url": "https://httpbin.org/get"})
                    elif tool == "filesystem":
                        test_result = await execute_tool(tool, {"path": ".", "operation": "list"})

                    tool_health[tool] = test_result.get("success", False)
                    if not test_result.get("success", False):
                        failed_tools.append(tool)

                except Exception as e:
                    tool_health[tool] = False
                    failed_tools.append(tool)
                    logger.warning(f"Tool {tool} health check failed: {e}")

            health_status.update(
                {
                    "tool_health": tool_health,
                    "failed_tools": failed_tools,
                    "essential_tools_healthy": len(failed_tools) == 0,
                }
            )

            # 3. 전체 상태 결정
            if len(failed_tools) > 0:
                health_status["overall_health"] = "unhealthy"
                health_status["critical_error"] = (
                    f"Essential tools failed: {', '.join(failed_tools)}"
                )
            else:
                health_status["overall_health"] = "healthy"

            return health_status

        except Exception as e:
            return {
                "mcp_enabled": self.config.enabled,
                "tools_available": len(self.tools),
                "openrouter_connected": False,
                "error": str(e),
                "overall_health": "unhealthy",
                "critical_error": f"Health check failed: {e}",
                "timestamp": datetime.now().isoformat(),
            }

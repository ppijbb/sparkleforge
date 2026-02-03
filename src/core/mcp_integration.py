"""
Universal MCP Hub - 2025년 10월 최신 버전

Model Context Protocol 통합을 위한 범용 허브.
OpenRouter와 Gemini 2.5 Flash Lite 기반의 최신 MCP 연결.
Production 수준의 안정성과 신뢰성 보장.
"""

import asyncio
import sys
import json
import logging
import time
import aiohttp
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import subprocess
import os
from datetime import datetime, timedelta
from contextlib import AsyncExitStack
import random
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import ListToolsResult, TextContent
    from mcp.shared.exceptions import McpError
    from urllib.parse import urlencode
    import httpx
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
    from fastmcp import Client as FastMCPClient
    import logging as fastmcp_logging
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

from src.core.researcher_config import get_mcp_config, get_llm_config
from src.core.mcp_auto_discovery import FastMCPMulti
from src.core.mcp_tool_loader import MCPToolLoader, ToolInfo as MCPToolInfo
from src.core.config import HTTPServerSpec

logger = logging.getLogger(__name__)

# 9대 혁신: ToolTrace 추적 시스템
_tool_trace_manager = None

def get_tool_trace_manager():
    """전역 ToolTraceManager 인스턴스 반환 (싱글톤 패턴)."""
    global _tool_trace_manager
    if _tool_trace_manager is None:
        from src.core.tool_trace import ToolTraceManager
        _tool_trace_manager = ToolTraceManager()
    return _tool_trace_manager

def set_tool_trace_manager(manager):
    """전역 ToolTraceManager 인스턴스 설정."""
    global _tool_trace_manager
    _tool_trace_manager = manager

def _create_tool_trace(
    tool_id: str,
    citation_id: str,
    tool_type: str,
    query: str,
    result: Dict[str, Any],
    mcp_server: Optional[str] = None,
    mcp_tool_name: Optional[str] = None,
) -> Optional[Any]:
    """
    ToolTrace 생성 헬퍼 함수 (9대 혁신: ToolTrace 추적 시스템).
    
    Args:
        tool_id: Tool ID
        citation_id: Citation ID
        tool_type: Tool type
        query: Query string
        result: 도구 실행 결과
        mcp_server: MCP 서버 이름 (optional)
        mcp_tool_name: MCP 도구 이름 (optional)
    
    Returns:
        ToolTrace 객체 (생성 성공 시), None (실패 시)
    """
    try:
        from src.core.tool_trace import ToolTrace
        
        # raw_answer 생성 (result를 JSON 문자열로)
        raw_answer = json.dumps(result, ensure_ascii=False, indent=2) if result else "{}"
        
        # summary 생성 (간단한 요약)
        if result.get("success"):
            if isinstance(result.get("data"), dict):
                if "results" in result["data"]:
                    summary = f"Found {len(result['data']['results'])} results"
                elif "content" in result["data"]:
                    content = str(result["data"]["content"])
                    summary = f"Content: {content[:100]}..." if len(content) > 100 else f"Content: {content}"
                else:
                    summary = "Tool executed successfully"
            else:
                summary = "Tool executed successfully"
        else:
            summary = f"Tool execution failed: {result.get('error', 'Unknown error')[:100]}"
        
        trace = ToolTrace.create_with_size_limit(
            tool_id=tool_id,
            citation_id=citation_id,
            tool_type=tool_type,
            query=query,
            raw_answer=raw_answer,
            summary=summary,
            mcp_server=mcp_server,
            mcp_tool_name=mcp_tool_name,
        )
        
        # ToolTraceManager에 추가
        try:
            manager = get_tool_trace_manager()
            manager.add_trace(trace)
        except Exception as e:
            logger.debug(f"Failed to add ToolTrace to manager: {e}")
        
        return trace
    except Exception as e:
        logger.debug(f"Failed to create ToolTrace: {e}")
        return None

def _infer_tool_type(tool_name: str) -> str:
    """
    도구 이름에서 도구 타입 추론.
    
    Args:
        tool_name: 도구 이름
    
    Returns:
        도구 타입
    """
    tool_lower = tool_name.lower()
    
    if "::" in tool_name:
        # MCP 도구
        return "mcp_tool"
    elif "search" in tool_lower or "google" in tool_lower or "tavily" in tool_lower:
        return "web_search"
    elif "arxiv" in tool_lower or "scholar" in tool_lower or "paper" in tool_lower:
        return "paper_search"
    elif "rag" in tool_lower or "query" in tool_lower:
        return "rag_hybrid" if "hybrid" in tool_lower else "rag_naive"
    elif "code" in tool_lower or "python" in tool_lower or "execute" in tool_lower:
        return "run_code"
    elif "browser" in tool_lower:
        return "browser"
    elif "generate" in tool_lower or "document" in tool_lower:
        return "document_generation"
    elif "file" in tool_lower:
        return "file_operation"
    else:
        return "unknown"

def _format_query_string(tool_name: str, parameters: Dict[str, Any]) -> str:
    """
    도구 파라미터를 쿼리 문자열로 포맷.
    
    Args:
        tool_name: 도구 이름
        parameters: 도구 파라미터
    
    Returns:
        포맷된 쿼리 문자열
    """
    # 주요 파라미터 추출
    query_keys = ["query", "question", "text", "input", "url", "path", "code"]
    for key in query_keys:
        if key in parameters:
            value = parameters[key]
            if isinstance(value, str):
                return value[:200]  # 최대 200자
            elif isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False)[:200]
    
    # 파라미터 전체를 JSON으로
    return json.dumps(parameters, ensure_ascii=False)[:200]


class ToolCategory(Enum):
    """MCP 도구 카테고리."""
    SEARCH = "search"
    DATA = "data"
    CODE = "code"
    ACADEMIC = "academic"
    BUSINESS = "business"
    UTILITY = "utility"
    BROWSER = "browser"  # 브라우저 자동화
    DOCUMENT = "document"  # 문서 생성
    FILE = "file"  # 파일 작업
    GIT = "git"  # Git 워크플로우


@dataclass
class ToolInfo:
    """도구 정보."""
    name: str
    category: ToolCategory
    description: str
    parameters: Dict[str, Any]
    mcp_server: str


@dataclass
class ToolResult:
    """도구 실행 결과."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    confidence: float = 0.0
    tool_name: Optional[str] = None
    source: Optional[str] = None


class ToolRegistry:
    """Tool 중앙 관리 시스템 - MCP 및 로컬 Tool 통합 관리."""
    
    def __init__(self):
        """ToolRegistry 초기화."""
        self.tools: Dict[str, ToolInfo] = {}  # tool_name -> ToolInfo
        self.langchain_tools: Dict[str, BaseTool] = {}  # tool_name -> LangChain Tool
        self.tool_sources: Dict[str, str] = {}  # tool_name -> source (mcp/local)
        self.mcp_tool_mapping: Dict[str, Tuple[str, str]] = {}  # tool_name -> (server_name, original_tool_name)
        
    def register_mcp_tool(self, server_name: str, tool: Any, tool_def: Any = None):
        """
        MCP Tool을 server_name::tool_name 형식으로 등록.
        
        Args:
            server_name: MCP 서버 이름
            tool: MCP Tool 객체 또는 tool name
            tool_def: MCP Tool 정의 (description, inputSchema 등 포함)
        """
        if isinstance(tool, str):
            tool_name = tool
        else:
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
        
        # server_name::tool_name 형식으로 등록
        registered_name = f"{server_name}::{tool_name}"
        
        # ToolInfo 생성
        if tool_def and hasattr(tool_def, 'description'):
            description = tool_def.description
            input_schema = tool_def.inputSchema if hasattr(tool_def, 'inputSchema') else {}
        else:
            description = f"Tool from MCP server {server_name}"
            input_schema = {}
        
        # 카테고리 추론 (기본값: UTILITY)
        category = ToolCategory.UTILITY
        tool_lower = tool_name.lower()
        if 'search' in tool_lower:
            category = ToolCategory.SEARCH
        elif 'scholar' in tool_lower or 'arxiv' in tool_lower or 'paper' in tool_lower:
            category = ToolCategory.ACADEMIC
        elif 'browser' in tool_lower:
            category = ToolCategory.BROWSER
        elif tool_lower.startswith('generate_') or 'document' in tool_lower or 'pdf' in tool_lower or 'docx' in tool_lower or 'pptx' in tool_lower:
            category = ToolCategory.DOCUMENT
        elif 'file' in tool_lower and 'fetch' not in tool_lower:
            category = ToolCategory.FILE
        elif 'fetch' in tool_lower:
            category = ToolCategory.DATA
        elif 'code' in tool_lower or 'python' in tool_lower:
            category = ToolCategory.CODE
        
        tool_info = ToolInfo(
            name=registered_name,
            category=category,
            description=description,
            parameters=input_schema,
            mcp_server=server_name
        )
        
        self.tools[registered_name] = tool_info
        self.tool_sources[registered_name] = "mcp"
        self.mcp_tool_mapping[registered_name] = (server_name, tool_name)
        
        logger.debug(f"Registered MCP tool: {registered_name} from server {server_name}")
        
    def register_local_tool(self, tool_info: ToolInfo, langchain_tool: BaseTool):
        """
        로컬 Tool을 LangChain Tool과 함께 등록.
        
        Args:
            tool_info: ToolInfo 객체
            langchain_tool: LangChain BaseTool 인스턴스
        """
        tool_name = tool_info.name
        
        self.tools[tool_name] = tool_info
        self.langchain_tools[tool_name] = langchain_tool
        self.tool_sources[tool_name] = "local"
        
        logger.debug(f"Registered local tool: {tool_name}")
        
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Tool 정보 조회."""
        return self.tools.get(tool_name)
        
    def get_langchain_tool(self, tool_name: str) -> Optional[BaseTool]:
        """LangChain Tool 조회."""
        return self.langchain_tools.get(tool_name)
        
    def get_all_langchain_tools(self) -> List[BaseTool]:
        """모든 Tool을 LangChain Tool 리스트로 반환."""
        return list(self.langchain_tools.values())
        
    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """카테고리별 Tool 목록 반환."""
        return [
            name for name, info in self.tools.items()
            if info.category == category
        ]
        
    def is_mcp_tool(self, tool_name: str) -> bool:
        """Tool이 MCP Tool인지 확인."""
        return self.tool_sources.get(tool_name) == "mcp"
        
    def get_mcp_server_info(self, tool_name: str) -> Optional[Tuple[str, str]]:
        """MCP Tool의 서버 정보 반환: (server_name, original_tool_name)."""
        return self.mcp_tool_mapping.get(tool_name)
        
    def get_all_tool_names(self) -> List[str]:
        """등록된 모든 Tool 이름 반환."""
        return list(self.tools.keys())
        
    def remove_tool(self, tool_name: str):
        """Tool 제거."""
        if tool_name in self.tools:
            del self.tools[tool_name]
        if tool_name in self.langchain_tools:
            del self.langchain_tools[tool_name]
        if tool_name in self.tool_sources:
            del self.tool_sources[tool_name]
        if tool_name in self.mcp_tool_mapping:
            del self.mcp_tool_mapping[tool_name]


class OpenRouterClient:
    """(비활성화) OpenRouter 경유는 사용하지 않습니다."""
    def __init__(self, api_key: str):
        self.api_key = api_key
    async def __aenter__(self):
        raise RuntimeError("OpenRouter is disabled. Use Gemini direct path via llm_manager.")
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False
    async def generate_response(self, *args, **kwargs):
        raise RuntimeError("OpenRouter is disabled. Use Gemini direct path via llm_manager.")



class UniversalMCPHub:
    """Universal MCP Hub - 2025년 10월 최신 버전."""

    def __init__(self):
        self.config = get_mcp_config()
        self.llm_config = get_llm_config()

        # ToolRegistry 통합 관리
        self.registry = ToolRegistry()
        
        # 실행 컨텍스트별 MCP 세션 관리 (ROMA 스타일)
        # 각 실행마다 독립적인 MCP 세션 풀을 유지
        self._execution_sessions: Dict[str, Dict[str, Any]] = {}
        self.tools: Dict[str, ToolInfo] = {}  # 하위 호환성을 위해 유지 (registry.tools 참조)
        self.openrouter_client: Optional[OpenRouterClient] = None

        # MCP 클라이언트 (기존 시스템)
        self.mcp_sessions: Dict[str, ClientSession] = {}
        self.exit_stacks: Dict[str, AsyncExitStack] = {}  # 참조만 유지, cleanup에서 aclose() 호출 안 함
        self.mcp_tools_map: Dict[str, Dict[str, Any]] = {}  # server_name -> {tool_name -> tool_info}
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
        self.fastmcp_multi: Optional[FastMCPMulti] = None
        self.fastmcp_tool_loader: Optional[MCPToolLoader] = None
        # FastMCP 설정 저장소 (서버별) - Client는 context manager이므로 매번 새로 생성
        self.fastmcp_configs: Dict[str, Dict[str, Any]] = {}  # server_name -> mcp_config
        self.auto_discovered_tools: Dict[str, BaseTool] = {}  # 자동 발견된 도구들
        self.auto_discovered_tool_infos: Dict[str, MCPToolInfo] = {}  # 도구 메타데이터

        # ERA 서버 관리자 (안전한 코드 실행)
        self.era_server_manager: Optional[Any] = None
        try:
            from src.core.researcher_config import get_era_config
            from src.core.era_server_manager import ERAServerManager
            
            era_config = get_era_config()
            if era_config.enabled:
                # 서버 주소 파싱
                server_addr = ":8080"  # 기본값
                if ':' in era_config.server_url:
                    try:
                        port = era_config.server_url.split(':')[-1]
                        server_addr = f":{port}"
                    except:
                        pass
                
                self.era_server_manager = ERAServerManager(
                    agent_binary_path=era_config.agent_binary_path,
                    server_url=era_config.server_url,
                    server_addr=server_addr,
                    auto_start=era_config.auto_start
                )
                logger.info("ERA server manager initialized")
        except ImportError:
            logger.debug("ERA modules not available")
        except Exception as e:
            logger.warning(f"Failed to initialize ERA server manager: {e}")

        self._load_tools_config()
        self._initialize_tools()
        self._initialize_clients()
        self._load_mcp_servers_from_config()
        
    def _load_tools_config(self):
        """tools_config.json에서 Tool 메타데이터 로드."""
        # configs 폴더에서 로드 시도 (우선)
        tools_config_file = project_root / "configs" / "tools_config.json"
        if not tools_config_file.exists():
            # 하위 호환성: 루트에서도 시도
            tools_config_file = project_root / "tools_config.json"
        
        if tools_config_file.exists():
            try:
                with open(tools_config_file, 'r', encoding='utf-8') as f:
                    self.tools_config = json.load(f)
                logger.info(f"✅ Loaded tools config from {tools_config_file}")
            except Exception as e:
                logger.warning(f"Failed to load tools config: {e}")
                self.tools_config = {}
        else:
            logger.warning(f"tools_config.json not found at {tools_config_file}")
            self.tools_config = {}
    
    def _create_langchain_tool_wrapper(self, tool_name: str, tool_config: Dict[str, Any]) -> Optional[BaseTool]:
        """
        tools_config.json의 설정을 기반으로 LangChain Tool 래퍼 생성.
        
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
                "file": ToolCategory.FILE
            }
            
            category_str = tool_config.get("category", "utility")
            category = category_map.get(category_str, ToolCategory.UTILITY)
            description = tool_config.get("description", f"{tool_name} tool")
            params_config = tool_config.get("parameters", {})
            
            # Pydantic 스키마 생성 - 최신 방식으로 단순화 (args_schema 없이도 동작)
            ToolSchema = None
            # LangChain StructuredTool은 args_schema 없이도 함수 시그니처에서 자동으로 파라미터를 추론함
            # 복잡한 동적 스키마 생성을 피하고 함수 파라미터로 처리
            
            # Tool 실행 함수 선택 (동기 래퍼 생성) - 함수 시그니처 명시
            def create_sync_func(tool_name_str, func_type):
                """동기 함수 래퍼 생성 - 명시적 함수 시그니처로 LangChain이 파라미터 추론."""
                if func_type == "search":
                    def search_wrapper(query: str, max_results: int = 10, num_results: int = 10) -> str:
                        params = {"query": query}
                        if max_results:
                            params["max_results"] = max_results
                        elif num_results:
                            params["max_results"] = num_results
                        return _execute_search_tool_sync(tool_name_str, params)
                    return search_wrapper
                elif func_type == "academic":
                    def academic_wrapper(query: str, max_results: int = 10, num_results: int = 10) -> str:
                        params = {"query": query}
                        if max_results:
                            params["max_results"] = max_results
                        elif num_results:
                            params["max_results"] = num_results
                        return _execute_academic_tool_sync(tool_name_str, params)
                    return academic_wrapper
                elif func_type == "data":
                    if tool_name_str == "fetch":
                        def fetch_wrapper(url: str) -> str:
                            return _execute_data_tool_sync("fetch", {"url": url})
                        return fetch_wrapper
                    elif tool_name_str == "filesystem":
                        def filesystem_wrapper(path: str, operation: str = "read") -> str:
                            return _execute_data_tool_sync("filesystem", {"path": path, "operation": operation})
                        return filesystem_wrapper
                    else:
                        def data_wrapper(**kwargs) -> str:
                            return _execute_data_tool_sync(tool_name_str, kwargs)
                        return data_wrapper
                elif func_type == "code":
                    if "interpreter" in tool_name_str.lower():
                        def code_wrapper(code: str, language: str = "python") -> str:
                            return _execute_code_tool_sync(tool_name_str, {"code": code, "language": language})
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
            
            if func is None:
                logger.warning(f"No execution function for tool: {tool_name}, category: {category_str}")
                # 실행 함수가 없어도 기본 래퍼 함수 생성
                def generic_executor(**kwargs):
                    """Generic executor when specific function not available."""
                    raise RuntimeError(f"Tool {tool_name} execution not implemented yet. Please configure execution function.")
                func = generic_executor
            
            # StructuredTool 생성 - args_schema 없이도 생성 가능하도록
            try:
                if StructuredTool and ToolSchema:
                    langchain_tool = StructuredTool.from_function(
                        func=func,
                        name=tool_name,
                        description=description,
                        args_schema=ToolSchema
                    )
                elif StructuredTool:
                    # args_schema 없이 생성 (파라미터는 함수 시그니처에서 자동 추론)
                    langchain_tool = StructuredTool.from_function(
                        func=func,
                        name=tool_name,
                        description=description
                    )
                else:
                    return None
                
                logger.info(f"✅ Created LangChain tool wrapper for {tool_name}")
                return langchain_tool
            except Exception as schema_error:
                # Schema 생성 실패 시 args_schema 없이 재시도
                logger.warning(f"Failed to create tool with schema for {tool_name}: {schema_error}, trying without schema")
                try:
                    if StructuredTool:
                        langchain_tool = StructuredTool.from_function(
                            func=func,
                            name=tool_name,
                            description=description
                        )
                        logger.info(f"✅ Created LangChain tool wrapper for {tool_name} (without schema)")
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
                task = loop.create_task(self._initialize_auto_discovered_tools())
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
                "file": ToolCategory.FILE
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
                mcp_server=tool_config.get("mcp_server_name", "")
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
                logger.warning(f"⚠️ Failed to create LangChain wrapper for {tool_name}, registering without wrapper")
                self.registry.tools[tool_name] = tool_info
                self.tools[tool_name] = tool_info

        # ========================================================================
        # NATIVE TOOLS REGISTRATION (Overrides/Fallbacks)
        # ========================================================================
        try:
            from src.core.tools.native_search import search_duckduckgo_json
            from langchain_core.tools import Tool
            
            native_tool_name = "ddg_search"
            logger.info(f"🛠️ Registering Native Tool: {native_tool_name}")
            
            native_tool_info = ToolInfo(
                name=native_tool_name,
                category=ToolCategory.SEARCH,
                description="Robust native DuckDuckGo search (No MCP required)",
                parameters={"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Max results"}},
                mcp_server="" 
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
                description="Search DuckDuckGo natively"
            )
            
            self.registry.register_local_tool(native_tool_info, native_langchain_tool)
            self.tools[native_tool_name] = native_tool_info
            
            # Also alias 'search' and 'g-search' to this if not present
            for alias in ["search", "g-search"]:
                if alias not in self.tools and alias not in self.registry.tools:
                    self.tools[alias] = native_tool_info
                    self.registry.register_local_tool(native_tool_info, native_langchain_tool) # Re-registering with same object might verify alias support? No, straightforward.
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
                        "repo_path": {"type": "string", "description": "Repository path (optional, defaults to current directory)"}
                    }
                },
                {
                    "name": "git_commit",
                    "description": "Create a Git commit with automatic message generation",
                    "parameters": {
                        "message": {"type": "string", "description": "Commit message (optional, auto-generated if not provided)"},
                        "auto_stage": {"type": "boolean", "description": "Automatically stage files (default: true)"},
                        "repo_path": {"type": "string", "description": "Repository path (optional)"}
                    }
                },
                {
                    "name": "git_push",
                    "description": "Push Git branch to remote repository",
                    "parameters": {
                        "branch": {"type": "string", "description": "Branch to push (optional, defaults to current branch)"},
                        "force": {"type": "boolean", "description": "Force push (default: false)"},
                        "repo_path": {"type": "string", "description": "Repository path (optional)"}
                    }
                },
                {
                    "name": "git_create_pr",
                    "description": "Create a Pull Request using GitHub CLI",
                    "parameters": {
                        "title": {"type": "string", "description": "PR title (required)"},
                        "body": {"type": "string", "description": "PR body (optional)"},
                        "base": {"type": "string", "description": "Base branch (default: main)"},
                        "repo_path": {"type": "string", "description": "Repository path (optional)"}
                    }
                },
                {
                    "name": "git_commit_push_pr",
                    "description": "Complete workflow: commit, push, and create PR in one step",
                    "parameters": {
                        "commit_message": {"type": "string", "description": "Commit message (optional, auto-generated if not provided)"},
                        "pr_title": {"type": "string", "description": "PR title (optional, uses commit message if not provided)"},
                        "pr_body": {"type": "string", "description": "PR body (optional, auto-generated if not provided)"},
                        "base": {"type": "string", "description": "Base branch (default: main)"},
                        "repo_path": {"type": "string", "description": "Repository path (optional)"}
                    }
                }
            ]
            
            for git_tool_config in git_tools:
                tool_name = git_tool_config["name"]
                logger.info(f"🛠️ Registering Git Tool: {tool_name}")
                
                tool_info = ToolInfo(
                    name=tool_name,
                    category=ToolCategory.GIT,
                    description=git_tool_config["description"],
                    parameters=git_tool_config["parameters"],
                    mcp_server=""
                )
                
                # LangChain Tool 래퍼 생성
                def create_git_tool_wrapper(tool_name: str):
                    async def git_tool_wrapper(**kwargs):
                        from src.core.mcp_integration import _execute_git_tool, ToolResult
                        result = await _execute_git_tool(tool_name, kwargs)
                        if result.success:
                            return result.data if isinstance(result.data, dict) else {"result": result.data}
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
                                    future = executor.submit(asyncio.run, create_git_tool_wrapper(tool_name)(**kwargs))
                                    return future.result()
                            else:
                                return loop.run_until_complete(create_git_tool_wrapper(tool_name)(**kwargs))
                        except RuntimeError:
                            return asyncio.run(create_git_tool_wrapper(tool_name)(**kwargs))
                    return wrapper
                
                langchain_tool = Tool(
                    name=tool_name,
                    func=sync_git_wrapper(tool_name),
                    description=git_tool_config["description"]
                )
                
                self.registry.register_local_tool(tool_info, langchain_tool)
                self.tools[tool_name] = tool_info
                logger.info(f"✅ Registered Git tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to register Git tools: {e}", exc_info=True)
        
        # Registry의 tools를 self.tools와 동기화
        self.tools.update(self.registry.tools)
        
        logger.info(f"✅ Initialized {len(self.registry.tools)} tools in registry ({len(self.registry.langchain_tools)} with LangChain wrappers)")

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

            logger.info(f"Auto-discovered {len(discovered_tools)} tools from {len(self.fastmcp_servers)} FastMCP servers")

        except Exception as e:
            logger.error(f"Failed to auto-discover MCP tools: {e}")
            raise

    def _initialize_fastmcp_servers(self):
        """환경 변수 및 구성에서 FastMCP 서버 설정 초기화."""
        # 환경 변수에서 서버 설정 로드 (예: FASTMCP_SERVERS)
        # 실제로는 config나 환경 변수에서 로드해야 함
        # 여기서는 예시로 빈 설정 유지
        pass

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
                "file": ToolCategory.FILE
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
                mcp_server=tool_info.server_guess
            )

            # 이름 충돌 확인
            if tool_name in self.registry.tools:
                # 충돌 시 자동 발견 도구 우선 (mcp_auto_ 접두사로 구분)
                auto_tool_name = f"mcp_auto_{tool_name}"
                logger.warning(f"Tool name conflict: '{tool_name}' already exists. Using '{auto_tool_name}' for auto-discovered tool.")
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
        """
        재귀적으로 객체 내의 환경변수 플레이스홀더를 실제 값으로 치환.
        ${VAR_NAME} 또는 $VAR_NAME 형식 지원.
        """
        if isinstance(value, str):
            import re
            # ${VAR_NAME} 또는 $VAR_NAME 패턴 찾기
            pattern = r'\$\{([^}]+)\}|\$(\w+)'
            
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
        """
        서버에 필요한 API 키나 환경변수가 있는지 확인.
        
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
        if "command" in server_config and "httpUrl" not in server_config and "url" not in server_config:
            # github 서버는 GITHUB_TOKEN 체크
            if server_name == "github" or "github" in server_name.lower():
                github_token = os.getenv("GITHUB_TOKEN")
                if not github_token:
                    logger.debug(f"[MCP][check.req] server={server_name} requires GITHUB_TOKEN but not set")
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
        if hasattr(self, '_mcp_servers_loaded') and self._mcp_servers_loaded:
            logger.debug("[MCP][load.skip] MCP server configs already loaded, skipping")
            return

        try:
            # configs 폴더에서 로드 시도 (우선)
            config_file = project_root / "configs" / "mcp_config.json"
            if not config_file.exists():
                # 하위 호환성: 루트에서도 시도
                config_file = project_root / "mcp_config.json"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    raw_configs = config_data.get("mcpServers", {})
                    
                    # 환경변수 치환
                    resolved_configs = self._resolve_env_vars_in_value(raw_configs)
                    
                    # API 키 확인 및 필터링
                    filtered_configs = {}
                    for server_name, server_config in resolved_configs.items():
                        # disabled 플래그 확인
                        if server_config.get("disabled"):
                            logger.info(f"[MCP][skip.disabled] server={server_name}")
                            continue
                        
                        # API 키가 필요한 서버 확인
                        if not self._check_server_requirements(server_name, server_config):
                            logger.info(f"[MCP][skip.no-api-key] server={server_name} (API key not configured)")
                            continue
                        
                        filtered_configs[server_name] = server_config
                    
                    self.mcp_server_configs = filtered_configs
                    logger.info(f"✅ Loaded MCP server configs: {list(self.mcp_server_configs.keys())}")
                    # 로드 완료 플래그 설정
                    self._mcp_servers_loaded = True
            else:
                # 기본 DuckDuckGo MCP 서버 설정
                self.mcp_server_configs = {
                    "ddg_search": {
                        "command": "npx",
                        "args": [
                            "-y",
                            "@modelcontextprotocol/server-duckduckgo-search@latest"
                        ]
                    }
                }
                logger.info("✅ Using default MCP server config for ddg_search")
                # 로드 완료 플래그 설정
                self._mcp_servers_loaded = True
            
            # FORCE DISABLE FLAKY SERVERS (DDG only) to use native fallbacks
            if "ddg_search" in self.mcp_server_configs:
                logger.info("🚫 Disabling flaky 'ddg_search' MCP server to use Native Tool fallback")
                del self.mcp_server_configs["ddg_search"]

            # tavily-mcp는 이제 활성화 (사용자가 요청)
                
            # Ensure we don't default to them either (tavily-mcp 제외)
            keys_to_remove = [k for k in self.mcp_server_configs if k in ["ddg_search"]]
            for k in keys_to_remove:
                del self.mcp_server_configs[k]
                
        except Exception as e:
            logger.warning(f"Failed to load MCP server configs: {e}")
            self.mcp_server_configs = {}
    
    def _get_server_specific_settings(self, server_name: str, server_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        각 MCP 서버별 특성에 맞는 설정 반환.
        
        서버별 특성:
        - stdio 서버 (npx 기반): 프로세스 시작 시간 필요, 더 긴 타임아웃
        - HTTP 서버: 빠른 연결, 짧은 타임아웃
        - 특정 서버: 특수 설정 적용
        """
        is_stdio = "httpUrl" not in server_config and "url" not in server_config and server_config.get("type") != "http"
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
        """
        Check if existing MCP server connection is healthy.
        
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
            if hasattr(session, 'list_tools'):
                # Quick health check - just verify session is still valid
                # We don't actually call list_tools to avoid overhead
                return True
            return True  # Assume healthy if session exists
        except Exception as e:
            logger.debug(f"Connection health check failed for {server_name}: {e}")
            return False
    
    async def _connect_to_mcp_server(self, server_name: str, server_config: Dict[str, Any], timeout: float = None):
        """MCP 서버에 연결 - Connection pooling with health check and auto-reconnection."""
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
                logger.warning(f"[MCP][connect.reconnect] Connection unhealthy for {server_name}, reconnecting...")
                # Disconnect unhealthy connection
                try:
                    await self._disconnect_from_mcp_server(server_name)
                except Exception as e:
                    logger.debug(f"Error disconnecting unhealthy connection: {e}")
        
        # 서버별 설정 가져오기
        server_settings = self._get_server_specific_settings(server_name, server_config)
        if timeout is None:
            timeout = server_settings["timeout"]
        
        logger.info(f"[MCP][connect.start] server={server_name} type={server_config.get('type','stdio')} url={(server_config.get('httpUrl') or server_config.get('url'))} timeout={timeout}")
        self.connection_diagnostics[server_name] = {
            "server": server_name,
            "type": ("http" if (server_config.get("httpUrl") or server_config.get("url") or server_config.get("type") == "http") else "stdio"),
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
            is_stdio = server_type == "stdio" or ("command" in server_config and "httpUrl" not in server_config and "url" not in server_config)
            
            if is_stdio:
                # stdio 서버 연결 (표준 MCP 방식 - OpenManus 스타일)
                if not MCP_AVAILABLE or ClientSession is None or StdioServerParameters is None or stdio_client is None:
                    logger.error(f"MCP package not available for stdio server {server_name}")
                    return False
                
                command = server_config.get("command")
                args = server_config.get("args", [])
                if not command:
                    logger.error(f"No command provided for stdio server {server_name}")
                    return False
                
                # 환경변수 처리 (github 등 env가 필요한 서버)
                env_vars = server_config.get("env", {})
                resolved_env = {}
                if env_vars:
                    for env_key, env_value in env_vars.items():
                        # 환경변수 치환 (${VAR} 형식)
                        if isinstance(env_value, str) and "${" in env_value:
                            import re
                            env_var_pattern = r'\$\{([^}]+)\}'
                            matches = re.findall(env_var_pattern, env_value)
                            resolved_value = env_value
                            for env_var in matches:
                                actual_value = os.getenv(env_var)
                                if actual_value:
                                    resolved_value = resolved_value.replace(f"${{{env_var}}}", actual_value)
                                else:
                                    logger.warning(f"[MCP][stdio.connect] server={server_name} env var {env_var} not found, keeping placeholder")
                            resolved_env[env_key] = resolved_value
                        else:
                            resolved_env[env_key] = env_value
                    
                    # 환경변수가 모두 비어있으면 서버 스킵
                    if all(not v or (isinstance(v, str) and "${" in v) for v in resolved_env.values()):
                        logger.warning(f"[MCP][stdio.connect] server={server_name} required env vars not set, skipping")
                        self.connection_diagnostics[server_name].update({
                            "ok": False,
                            "error": "Required environment variables not set",
                            "stage": "failed"
                        })
                        return False
                
                logger.info(f"[MCP][stdio.connect] server={server_name} command={command} args={args} env={list(resolved_env.keys()) if resolved_env else 'none'}")
                
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
                                                logger.info(f"[MCP][stdio.connect] Cleaned corrupted npx cache directory: {item}")
                                            except Exception as e:
                                                logger.debug(f"[MCP][stdio.connect] Failed to remove cache dir {item}: {e}")
                    except Exception as e:
                        logger.debug(f"[MCP][stdio.connect] Failed to clean npm cache: {e}")
                
                try:
                    # 표준 MCP 방식으로 연결 (OpenManus 스타일)
                    # StdioServerParameters에 env 전달
                    server_params = StdioServerParameters(
                        command=command,
                        args=args,
                        env=resolved_env if resolved_env else None
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
                    
                    await session.initialize()
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
                            input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                        )
                        self.mcp_tools_map[server_name][tool.name] = tool_info
                    
                    # 세션 저장 (연결 유지)
                    self.mcp_sessions[server_name] = session
                    
                    logger.info(f"[MCP][stdio.connect] ✅ Connected to {server_name}, tools: {len(response.tools)}")
                    
                    self.connection_diagnostics[server_name].update({
                        "ok": True,
                        "stage": "connected",
                        "tools_count": len(response.tools)
                    })
                    
                    return True
                except asyncio.CancelledError:
                    logger.debug(f"[MCP][stdio.connect] Connection cancelled for {server_name}")
                    raise
                except Exception as e:
                        error_str = str(e).lower()
                        error_msg = str(e)
                        
                        # npm 404 에러는 패키지가 존재하지 않으므로 재시도 불필요
                        is_npm_404 = "404" in error_str and ("npm" in error_str or "not found" in error_str or "not in this registry" in error_str)
                        
                        # npm 오류 감지
                        is_npm_enotempty = "enotempty" in error_str or ("npm error" in error_str and "directory not empty" in error_str)
                        is_npm_tar_error = "tar_entry_error" in error_str or ("enoent" in error_str and "zod" in error_str)
                        is_module_not_found = "err_module_not_found" in error_str or ("cannot find module" in error_str and "zod" in error_str)
                        
                        # Connection closed 오류는 서버 연결 실패
                        is_connection_closed = "connection closed" in error_str or "client failed to connect" in error_str
                        
                        # npm 캐시 손상 오류 해결: 캐시 정리 후 재시도
                        if (is_npm_enotempty or is_npm_tar_error or is_module_not_found) and command == "npx":
                            try:
                                import shutil
                                import subprocess
                                # npm cache clean --force 실행
                                try:
                                    subprocess.run(
                                        ["npm", "cache", "clean", "--force"],
                                        capture_output=True,
                                        timeout=10,
                                        check=False
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
                                                zod_path = os.path.join(item_path, "node_modules", "zod")
                                                if os.path.exists(zod_path):
                                                    # zod 파일들이 없는 경우 (TAR_ENTRY_ERROR 또는 MODULE_NOT_FOUND)
                                                    zod_external = os.path.join(zod_path, "v3", "external.js")
                                                    if not os.path.exists(zod_external):
                                                        # 손상된 패키지 디렉토리 전체 삭제
                                                        shutil.rmtree(item_path, ignore_errors=True)
                                                        logger.info(f"[MCP][stdio.connect] Cleaned corrupted npx cache directory: {item}")
                                            except Exception:
                                                pass
                                    
                                    # 재시도 (한 번만) - 표준 MCP 방식으로
                                    logger.info(f"[MCP][stdio.connect] Retrying connection to {server_name} after npm cache cleanup...")
                                    try:
                                        # 표준 MCP 방식으로 재시도
                                        retry_server_params = StdioServerParameters(
                                            command=command,
                                            args=args,
                                            env=resolved_env if resolved_env else None
                                        )
                                        
                                        retry_stdio_transport = await exit_stack.enter_async_context(
                                            stdio_client(retry_server_params)
                                        )
                                        retry_read, retry_write = retry_stdio_transport
                                        retry_session = await exit_stack.enter_async_context(ClientSession(retry_read, retry_write))
                                        
                                        await retry_session.initialize()
                                        retry_response = await asyncio.wait_for(retry_session.list_tools(), timeout=timeout)
                                        
                                        # 도구 등록
                                        for tool in retry_response.tools:
                                            self.registry.register_mcp_tool(server_name, tool, tool)
                                            if server_name not in self.mcp_tools_map:
                                                self.mcp_tools_map[server_name] = {}
                                            tool_info = MCPToolInfo(
                                                server_guess=server_name,
                                                name=f"{server_name}::{tool.name}",
                                                description=tool.description or "",
                                                input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                                            )
                                            self.mcp_tools_map[server_name][tool.name] = tool_info
                                        
                                        # 세션 저장
                                        self.mcp_sessions[server_name] = retry_session
                                        
                                        logger.info(f"[MCP][stdio.connect] ✅ Connected to {server_name} after cache cleanup, tools: {len(retry_response.tools)}")
                                        self.connection_diagnostics[server_name].update({
                                            "ok": True,
                                            "stage": "connected",
                                            "tools_count": len(retry_response.tools)
                                        })
                                        return True
                                    except Exception as retry_e:
                                        logger.warning(f"[MCP][stdio.connect] Retry failed for {server_name}: {retry_e}")
                            except Exception as cleanup_e:
                                logger.debug(f"[MCP][stdio.connect] Cache cleanup failed: {cleanup_e}")
                        
                        # 조용히 처리할 오류들 (WARNING 레벨로만 로깅)
                        if is_npm_404:
                            logger.warning(f"[MCP][stdio.connect] Package not found for {server_name} (npm 404), skipping")
                        elif is_connection_closed:
                            logger.warning(f"[MCP][stdio.connect] Connection closed for {server_name}, skipping")
                        else:
                            # 다른 오류는 WARNING 레벨로 로깅
                            logger.warning(f"[MCP][stdio.connect] Failed to connect to {server_name}: {error_msg[:200]}")
                        
                        self.connection_diagnostics[server_name].update({
                            "ok": False,
                            "error": error_msg[:200],  # 긴 에러 메시지 자르기
                            "stage": "failed",
                            "is_npm_404": is_npm_404,
                            "is_npm_enotempty": is_npm_enotempty,
                            "is_connection_closed": is_connection_closed
                        })
                        
                        # npm 404, Connection closed는 재시도 불필요
                        if is_npm_404 or is_connection_closed:
                            return False
                        
                        return False
                        
                except Exception as e:
                    logger.error(f"[MCP][stdio.connect] Error setting up stdio connection for {server_name}: {e}", exc_info=True)
                    self.connection_diagnostics[server_name].update({
                        "ok": False,
                        "error": str(e),
                        "stage": "failed"
                    })
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
                    env_var_pattern = r'\$\{([^}]+)\}'
                    matches = re.findall(env_var_pattern, v)
                    if matches:
                        resolved_value = v
                        for env_var in matches:
                            env_value = os.getenv(env_var, "")
                            if env_value:
                                # ${VAR}를 실제 값으로 치환
                                resolved_value = resolved_value.replace(f"${{{env_var}}}", env_value)
                                logger.debug(f"[MCP][auth.env] server={server_name} Resolved {k} from {env_var}")
                            else:
                                logger.warning(f"[MCP][auth.env] server={server_name} {env_var} not found in environment")
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
            server_config_dict = {
                "url": base_url
            }
            
            # headers가 있으면 추가 (FastMCP는 headers를 지원함)
            if resolved_headers:
                server_config_dict["headers"] = resolved_headers
            
            mcp_config = {
                "mcpServers": {
                    server_name: server_config_dict
                }
            }
            
            
            logger.info(f"[MCP][fastmcp.connect] server={server_name} url={base_url} headers={list(resolved_headers.keys()) if resolved_headers else 'None'}")
            
            try:
                # FastMCP Client 직접 사용 (가이드에 따른 올바른 사용법)
                # 기존 클라이언트가 있으면 재사용, 없으면 새로 생성
                if server_name in self.fastmcp_clients:
                    fastmcp_client = self.fastmcp_clients[server_name]
                    logger.debug(f"[MCP][fastmcp.reuse] server={server_name} Reusing existing FastMCP client")
                else:
                    # FastMCP Client 생성
                    fastmcp_client = FastMCPClient(mcp_config)
                    self.fastmcp_clients[server_name] = fastmcp_client
                    logger.debug(f"[MCP][fastmcp.create] server={server_name} Created new FastMCP client")
                
                # FastMCP Client를 Context Manager로 사용 (가이드에 따른 올바른 사용법)
                try:
                    # stopping 플래그 재확인
                    if self.stopping:
                        logger.warning(f"[MCP][skip.stopping] server={server_name} stopping flag is set")
                        raise asyncio.CancelledError("Stopping flag is set")
                    
                    # Context Manager로 사용하여 연결 테스트 및 도구 목록 가져오기
                    async with fastmcp_client:
                        # stopping 플래그 체크
                        if self.stopping:
                            logger.info(f"[MCP][skip.stopping] server={server_name} stopping flag is set, skipping connection")
                            raise asyncio.CancelledError("Stopping flag is set")
                        
                        # 도구 목록 가져오기 (타임아웃 설정, shield 제거하여 취소 가능)
                        try:
                            tools = await asyncio.wait_for(
                                fastmcp_client.list_tools(),
                                timeout=timeout
                            )
                        except asyncio.TimeoutError:
                            logger.warning(f"[MCP][list_tools.timeout] server={server_name} list_tools timeout after {timeout}s")
                            raise
                        except asyncio.CancelledError as e:
                            if self.stopping:
                                logger.info(f"[MCP][list_tools.cancelled] server={server_name} cancelled due to stopping flag")
                                raise
                            else:
                                logger.warning(f"[MCP][list_tools.cancelled] server={server_name} list_tools was cancelled unexpectedly")
                                raise
                    
                    # 도구 정보 저장
                    tools_dict = {}
                    if tools:
                        for tool in tools:
                            tools_dict[tool.name] = {
                                "name": tool.name,
                                "description": getattr(tool, "description", "") or "",
                                "inputSchema": getattr(tool, "inputSchema", {}) or {}
                            }
                    
                    self.mcp_tools_map[server_name] = tools_dict
                    logger.info(f"[MCP][fastmcp.success] server={server_name} Connected, {len(tools_dict)} tools available")
                    
                    # 연결 진단 정보 업데이트
                    di = self.connection_diagnostics.get(server_name, {})
                    di.update({
                        "ok": True,
                        "tools_count": len(tools_dict),
                        "client_type": "FastMCP"
                    })
                    self.connection_diagnostics[server_name] = di
                    
                    # FastMCP Client 인스턴스 저장 (나중에 도구 호출 시 사용)
                    # 주의: FastMCP Client는 context manager이므로, 도구 호출 시마다 async with로 사용해야 함
                    # 세션은 저장하지 않고 클라이언트만 저장
                    self.mcp_sessions[server_name] = fastmcp_client  # FastMCP Client 저장
                    
                    return True
                    
                except Exception as fastmcp_error:
                    error_msg = str(fastmcp_error)
                    error_type = type(fastmcp_error).__name__
                    logger.error(f"[MCP][fastmcp.error] server={server_name} err={error_type}: {error_msg}")
                    logger.exception(f"[MCP][fastmcp.error] server={server_name} full traceback:")
                    
                    # 연결 실패 시 클라이언트 제거
                    if server_name in self.fastmcp_clients:
                        del self.fastmcp_clients[server_name]
                    
                    di = self.connection_diagnostics.get(server_name, {})
                    di.update({
                        "stage": "fastmcp_connect",
                        "error": error_msg,
                        "error_type": error_type,
                        "ok": False
                    })
                    self.connection_diagnostics[server_name] = di
                    return False
                    
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                logger.error(f"[MCP][fastmcp.error] server={server_name} err={error_type}: {error_msg}")
                logger.exception(f"[MCP][fastmcp.error] server={server_name} full traceback:")
                di = self.connection_diagnostics.get(server_name, {})
                di.update({"stage": "fastmcp_connect", "error": error_msg, "error_type": error_type})
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
            if 't1' in locals() and 't2' in locals():
                logger.info(f"[MCP][connect.ok] server={server_name} init_ms={(t1-t0)*1000:.0f} list_ms={(t2-t1)*1000:.0f} tools={tool_names}")
            else:
                logger.info(f"[MCP][connect.ok] server={server_name} tools={tool_names}")
            logger.info(f"✅ Connected to MCP server {server_name} with {len(response.tools)} tools")
            return True
            
        except asyncio.CancelledError:
            # 작업이 취소된 경우 (종료 신호 등) - 정상적인 동작
            logger.info(f"[MCP][connect.cancelled] server={server_name} stage=generic (shutdown in progress)")
            try:
                await self._disconnect_from_mcp_server(server_name)
            except Exception:
                pass  # cleanup 중 오류는 무시
            return False  # raise하지 않고 False 반환하여 다른 서버 연결 계속 진행
        except asyncio.TimeoutError:
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
            di.update({"stage": "exception", "error": str(e), "traceback": traceback.format_exc()})
            self.connection_diagnostics[server_name] = di
            # 실패 시 exit_stack 참조만 제거 (aclose() 호출하지 않음 - anyio 오류 방지)
            if server_name in self.exit_stacks:
                del self.exit_stacks[server_name]
            try:
                await self._disconnect_from_mcp_server(server_name)
            except:
                pass
            return False
    
    async def _register_dynamic_server(
        self,
        server_name: str,
        server_path: Path
    ) -> bool:
        """
        동적으로 생성된 MCP 서버를 등록하고 시작.
        
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
                "args": [str(server_path)]
            }
            
            # mcp_server_configs에 추가
            self.mcp_server_configs[server_name] = server_config
            
            # mcp_config.json에도 추가 (선택적, 영구 저장)
            try:
                config_file = project_root / "configs" / "mcp_config.json"
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    if "mcpServers" not in config_data:
                        config_data["mcpServers"] = {}
                    
                    # 동적 서버 추가 (기존 서버와 충돌 방지)
                    if server_name not in config_data["mcpServers"]:
                        config_data["mcpServers"][server_name] = server_config
                        
                        # 백업 후 저장
                        backup_file = config_file.with_suffix('.json.bak')
                        if not backup_file.exists():
                            import shutil
                            shutil.copy2(config_file, backup_file)
                        
                        with open(config_file, 'w', encoding='utf-8') as f:
                            json.dump(config_data, f, indent=2, ensure_ascii=False)
                        logger.debug(f"[MCP][builder.register] Added {server_name} to mcp_config.json")
            except Exception as config_error:
                logger.warning(f"[MCP][builder.register] Failed to update mcp_config.json: {config_error}")
                # 계속 진행 (메모리에는 등록됨)
            
            # 서버 연결 시도
            connected = await self._connect_to_mcp_server(server_name, server_config, timeout=30.0)
            
            if connected:
                logger.info(f"[MCP][builder.register] ✅ Dynamic server registered and connected: {server_name}")
                
                # ProcessManager에 등록 (서버 프로세스 추적)
                try:
                    from src.core.process_manager import get_process_manager
                    pm = get_process_manager()
                    # 서버 프로세스는 _connect_to_mcp_server에서 시작되므로 여기서는 로깅만
                    logger.debug(f"[MCP][builder.register] Server {server_name} process will be tracked by ProcessManager")
                except Exception as pm_error:
                    logger.debug(f"[MCP][builder.register] ProcessManager registration skipped: {pm_error}")
                
                return True
            else:
                logger.error(f"[MCP][builder.register] ❌ Failed to connect to dynamic server: {server_name}")
                # 등록은 했지만 연결 실패 - 설정은 유지 (재시도 가능)
                return False
                
        except Exception as e:
            logger.error(f"[MCP][builder.register] Failed to register dynamic server {server_name}: {e}", exc_info=True)
            return False
    
    async def _disconnect_from_mcp_server(self, server_name: str):
        """MCP 서버 연결 해제 - 안전한 비동기 정리."""
        try:
            # FastMCP Client 정리
            if server_name in self.fastmcp_clients:
                try:
                    fastmcp_client = self.fastmcp_clients[server_name]
                    # FastMCP Client 명시적 종료 시도
                    if hasattr(fastmcp_client, 'close'):
                        try:
                            await asyncio.wait_for(fastmcp_client.close(), timeout=0.5)
                        except (asyncio.TimeoutError, Exception) as e:
                            logger.debug(f"FastMCP client close timeout/error for {server_name}: {e}")
                    elif hasattr(fastmcp_client, '__aexit__'):
                        # Context manager의 __aexit__ 호출 시도
                        try:
                            await asyncio.wait_for(fastmcp_client.__aexit__(None, None, None), timeout=0.5)
                        except (asyncio.TimeoutError, Exception) as e:
                            logger.debug(f"FastMCP client __aexit__ timeout/error for {server_name}: {e}")
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
                is_fastmcp_client = session and FASTMCP_AVAILABLE and isinstance(session, FastMCPClient)
                if not is_fastmcp_client:
                    # 기존 ClientSession 방식
                    try:
                        # 세션 종료 시도 (안전하게) - heartbeat 중지
                        if hasattr(session, 'shutdown'):
                            await asyncio.wait_for(session.shutdown(), timeout=1.0)
                        elif hasattr(session, 'close'):
                            await asyncio.wait_for(session.close(), timeout=1.0)
                    except (asyncio.TimeoutError, AttributeError, Exception) as e:
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
        
        # ERA 서버 시작 (코드 실행을 위해)
        if self.era_server_manager:
            try:
                if await self.era_server_manager.ensure_server_running():
                    logger.info("✅ ERA server is running (safe code execution enabled)")
                else:
                    logger.warning("⚠️ ERA server is not available (code execution will use fallback)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to start ERA server: {e} (code execution will use fallback)")
        
        try:
            logger.info("Initializing MCP Hub with MCP servers (no OpenRouter)...")
            
            # 일괄 초기화 대기 시간 (agent 시작 초기에 모든 서버 준비 시간 확보)
            batch_init_delay = float(os.getenv("MCP_BATCH_INIT_DELAY", "3.0"))  # 기본 3초
            if batch_init_delay > 0:
                logger.info(f"[MCP][init.batch] Waiting {batch_init_delay}s for batch initialization before connecting servers...")
                await asyncio.sleep(batch_init_delay)
            
            # MCP 서버 연결 (모든 서버) - 병렬 + 타임아웃 적용
            timeout_per_server = float(os.getenv("MCP_CONNECT_TIMEOUT", "60"))  # 서버당 최대 60초(환경변수로 조정, npx 서버 고려)
            max_concurrency = int(os.getenv("MCP_MAX_CONCURRENCY", "3"))  # 동시 연결 수 제한 (기본 3개)
            semaphore = asyncio.Semaphore(max_concurrency)
            logger.info(f"[MCP][init] max_concurrency={max_concurrency}, timeout_per_server={timeout_per_server}s")


            # disabled=true 설정된 서버는 건너뛰기 + 허용 서버 화이트리스트 적용
            allowlist_str = os.getenv("MCP_ALLOWED_SERVERS", "").strip()
            allowlist = [s.strip() for s in allowlist_str.split(",") if s.strip()]
            base_items = [(n, c) for n, c in self.mcp_server_configs.items() if not c.get("disabled")]
            if allowlist:
                # 화이트리스트가 있으면 그것만 연결
                enabled_server_items = [(n, c) for n, c in base_items if n in allowlist]
                logger.info(f"[MCP][allowlist] enabled={ [n for n,_ in enabled_server_items] }")
            else:
                # 화이트리스트가 없으면 disabled가 아닌 모든 서버 연결 시도
                enabled_server_items = base_items
                logger.info(f"[MCP][allowlist] not set; connecting to all enabled servers: { [n for n,_ in enabled_server_items] }")

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
                            logger.info(f"[MCP][skip.stopping] server={name} stopping flag is set after semaphore")
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
                                logger.info(f"[MCP][skip.stopping] server={name} stopping flag is set before retry {retry_attempt + 1}")
                                return name, False
                            
                            try:
                                logger.info(f"Connecting to MCP server {name} (timeout: {server_timeout}s, attempt {retry_attempt + 1}/{max_connection_retries})...")
                                # stopping 플래그 체크
                                if self.stopping:
                                    logger.info(f"[MCP][skip.stopping] server={name} stopping flag is set, skipping connection")
                                    return name, False
                                # shield 제거하여 취소 가능하도록 (stopping 플래그로 제어)
                                ok = await self._connect_to_mcp_server(name, cfg, timeout=server_timeout)
                                if ok:
                                    connection_success = True
                                    if retry_attempt > 0:
                                        logger.info(f"[MCP][init.success] server={name} connected after {retry_attempt + 1} attempts")
                                    break
                                else:
                                    # 연결 실패
                                    if retry_attempt < max_connection_retries - 1:
                                        wait_time = 2 ** retry_attempt  # 지수 백오프: 1초, 2초
                                        logger.warning(f"[MCP][init.retry] server={name} connection failed (attempt {retry_attempt + 1}/{max_connection_retries}), retrying in {wait_time}s...")
                                        await asyncio.sleep(wait_time)
                                        continue
                                    else:
                                        logger.warning(f"[MCP][init.failed] server={name} failed after {max_connection_retries} attempts")
                                        break
                                        
                            except asyncio.TimeoutError:
                                # 타임아웃 에러는 재시도 가능
                                if retry_attempt < max_connection_retries - 1:
                                    wait_time = 2 ** retry_attempt  # 지수 백오프: 1초, 2초
                                    logger.warning(f"[MCP][init.timeout] server={name} timeout (attempt {retry_attempt + 1}/{max_connection_retries}), retrying in {wait_time}s...")
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    logger.warning(f"[MCP][init.timeout] server={name} timeout after {max_connection_retries} attempts")
                                    break
                                    
                            except Exception as e:
                                error_str = str(e).lower()
                                error_msg = str(e)
                                
                                # npm 404 에러는 패키지가 존재하지 않으므로 재시도 불필요
                                is_npm_404 = "404" in error_str and ("npm" in error_str or "not found" in error_str or "not in this registry" in error_str or "package not found" in error_str)
                                
                                # npm ENOTEMPTY 오류는 디렉토리 관련 문제로, 재시도 불필요
                                is_npm_enotempty = "enotempty" in error_str or ("npm error" in error_str and "directory not empty" in error_str)
                                
                                # Connection closed 오류는 서버 연결 실패로, 재시도 불필요
                                is_connection_closed = "connection closed" in error_str or "client failed to connect" in error_str
                                
                                # 조용히 처리할 오류들 (재시도 불필요)
                                if is_npm_404:
                                    logger.warning(f"[MCP][init.skip] server={name} package not found (npm 404), skipping")
                                    break
                                elif is_npm_enotempty:
                                    logger.warning(f"[MCP][init.skip] server={name} npm directory issue (ENOTEMPTY), skipping")
                                    break
                                elif is_connection_closed:
                                    logger.warning(f"[MCP][init.skip] server={name} connection closed, skipping")
                                    break
                                
                                # 504, 502, 503 등 서버 에러는 재시도
                                is_retryable = any(code in error_str for code in ["504", "502", "503", "500", "gateway", "timeout", "unavailable"])
                                
                                if is_retryable and retry_attempt < max_connection_retries - 1:
                                    wait_time = 2 ** retry_attempt  # 지수 백오프: 1초, 2초
                                    logger.warning(f"[MCP][init.retry] server={name} error (attempt {retry_attempt + 1}/{max_connection_retries}): {error_msg[:100]}, retrying in {wait_time}s...")
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    # 재시도 불가능한 에러 또는 최대 재시도 횟수 초과
                                    logger.warning(f"[MCP][connect.error] server={name} error: {error_msg[:200]}")
                                    break
                        
                        return name, connection_success
                        
                except asyncio.CancelledError:
                    # shutdown 중 취소는 정상적인 동작 - 다른 서버 연결은 계속 진행
                    logger.info(f"[MCP][init.cancelled] server={name} (shutdown in progress)")
                    return name, False
                except Exception as e:
                    logger.exception(f"[MCP][connect.error] server={name} unexpected err={e}")
                    return name, False

            tasks = [asyncio.create_task(connect_one_with_settings(n, c)) for n, c in enabled_server_items]
            # return_exceptions=True로 변경하여 일부 실패해도 계속 진행
            # 전체 초기화 타임아웃 설정 (서버 수 * 타임아웃, 최대 300초)
            total_timeout = min(len(enabled_server_items) * timeout_per_server, 300.0)
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=total_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"[MCP][init.timeout] MCP initialization timeout after {total_timeout}s, cancelling remaining tasks...")
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
                        logger.info(f"[MCP][init.cancelled] server={server_name} (task was cancelled)")
                    else:
                        logger.warning(f"[MCP][init.exception] server={server_name} error={result}")
                elif isinstance(result, tuple) and len(result) == 2:
                    name, ok = result
                    if ok:
                        connected_servers.append(name)
            
            if connected_servers:
                logger.info(f"✅ Successfully connected to {len(connected_servers)} MCP servers: {', '.join(connected_servers)}")
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
                    init_ms = di.get('init_ms')
                    list_ms = di.get('list_ms')
                    logger.info(
                        "[MCP][diag] server=%s type=%s url=%s stage=%s ok=%s init_ms=%s list_ms=%s err=%s",
                        name,
                        di.get("type"),
                        di.get("url"),
                        di.get("stage"),
                        di.get("ok"),
                        f"{init_ms:.0f}" if isinstance(init_ms, (int, float)) else "-",
                        f"{list_ms:.0f}" if isinstance(list_ms, (int, float)) else "-",
                        di.get("error")
                    )
            
            # 필수 도구 검증 - 실패 시 warning만
            await self._validate_essential_tools()
            
        except Exception as e:
            logger.warning(f"⚠️ MCP Hub initialization failed: {e} - continuing with graceful degradation")
            logger.info("ℹ️ System will continue with limited functionality (no API calls)")
            # Don't raise, allow graceful degradation
    
    async def _execute_via_mcp_server(self, server_name: str, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """MCP 서버를 통해 도구 실행 (with connection pooling and health check)."""
        
        # Connection pooling: Check if connection exists and is healthy
        if server_name not in self.mcp_sessions:
            # Try to connect if not in sessions
            logger.debug(f"Server {server_name} not in sessions, attempting connection...")
            if server_name in self.mcp_server_configs:
                server_config = self.mcp_server_configs[server_name]
                connected = await self._connect_to_mcp_server(server_name, server_config)
                if not connected:
                    logger.error(f"Failed to connect to server {server_name}")
                    return None
            else:
                logger.error(f"Server {server_name} not in mcp_sessions and no config found. Available: {list(self.mcp_sessions.keys())}")
                return None
        else:
            # Health check existing connection
            is_healthy = await self._check_connection_health(server_name)
            if not is_healthy:
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
        
        if server_name not in self.mcp_sessions:
            logger.error(f"Server {server_name} still not in mcp_sessions after connection attempt")
            return None
        
        if server_name not in self.mcp_tools_map:
            logger.error(f"Server {server_name} not in mcp_tools_map. Available: {list(self.mcp_tools_map.keys())}")
            return None
        
        if tool_name not in self.mcp_tools_map[server_name]:
            available_tools = list(self.mcp_tools_map[server_name].keys())
            logger.error(f"Tool {tool_name} not found in server {server_name}. Available tools: {available_tools}")
            return None
        
        # 재시도 + 재연결(ClosedResource/429) + 간단한 백오프
        max_attempts = 3
        backoff_seconds = [0.5, 1.5, 3.0]
        
        for attempt in range(max_attempts):
            try:
                # 표준 MCP ClientSession 방식 (우선)
                if server_name in self.mcp_sessions:
                    session = self.mcp_sessions[server_name]
                    logger.debug(f"Calling tool {tool_name} on server {server_name} using ClientSession (attempt {attempt+1}/{max_attempts})")
                    
                    if session is None:
                        logger.error(f"[MCP][exec.error] Session is None for {server_name}")
                        return None
                    
                    if not hasattr(session, "call_tool"):
                        logger.error(f"[MCP][exec.error] Session does not have call_tool method: {type(session)}")
                        return None
                    
                    # 기존 ClientSession 방식
                    result = await session.call_tool(tool_name, params)
                    
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
                        logger.debug(f"Tool {tool_name} returned content length: {len(content_str)}")
                        return content_str
                    else:
                        logger.warning(f"Tool {tool_name} returned empty result")
                        return None
                else:
                    logger.error(f"[MCP][exec.error] Server {server_name} not found in fastmcp_clients or mcp_sessions")
                    return None
            
            except McpError as e:
                error_msg = str(e) if e else "Unknown MCP error"
                error_code = getattr(e.error, 'code', None) if hasattr(e, 'error') else None
                error_data = getattr(e.error, 'data', None) if hasattr(e, 'error') else None
                
                # 레이트리밋 / 토큰 오류 감지
                is_rate_limit = "Too Many Requests" in error_msg or (error_code == 429)
                is_auth_error = "invalid_token" in error_msg.lower() or (error_code == 401)
                
                error_details = f"[MCP][exec.error] server={server_name} tool={tool_name} operation=call_tool"
                if error_code:
                    error_details += f" code={error_code}"
                if error_data:
                    error_details += f" data={error_data}"
                error_details += f" error={error_msg}"
                logger.error(error_details)
                
                if is_rate_limit and attempt < max_attempts - 1:
                    wait = backoff_seconds[attempt]
                    logger.warning(f"[MCP][exec.retry] Rate limit hit, retrying in {wait}s (attempt {attempt+2}/{max_attempts})")
                    await asyncio.sleep(wait)
                    continue
                
                if is_auth_error:
                    logger.error("[MCP][auth] invalid or expired token; refresh credentials and re-init MCP")
                return None
            
            except (RuntimeError, ConnectionError, OSError) as e:
                # ClosedResourceError, connection reset 등
                error_type = type(e).__name__
                error_msg = str(e)
                closed_like = "closed" in error_msg.lower() or "connection reset" in error_msg.lower()
                
                if closed_like and server_name in self.mcp_server_configs and attempt < max_attempts - 1:
                    logger.warning(f"[MCP][exec.retry] server={server_name} tool={tool_name} connection closed, reconnecting (attempt {attempt+2}/{max_attempts})")
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
                
                logger.error(f"[MCP][exec.error] server={server_name} tool={tool_name} operation=call_tool type={error_type} error={error_msg}")
                # Invalidate session/client on fatal error if it looks like a connection issue
                if closed_like or "broken pipe" in error_msg.lower():
                    if server_name in self.mcp_sessions:
                        logger.warning(f"[MCP][session.invalidate] Removing dead session for {server_name}")
                        del self.mcp_sessions[server_name]
                    if server_name in self.fastmcp_clients:
                        logger.warning(f"[MCP][client.invalidate] Removing dead FastMCP client for {server_name}")
                        del self.fastmcp_clients[server_name]
                
                import traceback
                logger.debug(f"[MCP][exec.exception] server={server_name} tool={tool_name} - Full traceback:\n{traceback.format_exc()}")
                return None
            
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                logger.error(f"[MCP][exec.error] server={server_name} tool={tool_name} operation=call_tool type={error_type} error={error_msg}")
                import traceback
                logger.debug(f"[MCP][exec.exception] server={server_name} tool={tool_name} - Full traceback:\n{traceback.format_exc()}")
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
            logger.warning("⚠️ Tools may be registered later when MCP servers connect or may need manual configuration")
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
        
        # ERA 서버 정리
        if self.era_server_manager:
            try:
                self.era_server_manager.cleanup()
                logger.info("✅ ERA server cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up ERA server: {e}")
        
        # OpenRouter 클라이언트 사용 안 함
        self.openrouter_client = None
        
        # FastMCP Client 정리 (병렬로 빠르게 종료)
        async def close_fastmcp_client(server_name: str, client: Any):
            """FastMCP Client 종료 헬퍼"""
            try:
                # 명시적 종료 시도
                if hasattr(client, 'close'):
                    try:
                        await asyncio.wait_for(client.close(), timeout=0.5)
                    except (asyncio.TimeoutError, Exception):
                        pass
                elif hasattr(client, '__aexit__'):
                    try:
                        await asyncio.wait_for(client.__aexit__(None, None, None), timeout=0.5)
                    except (asyncio.TimeoutError, Exception):
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
                    asyncio.gather(*close_tasks, return_exceptions=True),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
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
                    if session and isinstance(session, FastMCPClient) if FASTMCP_AVAILABLE else False:
                        try:
                            # FastMCP Client 명시적 종료
                            if hasattr(session, 'close'):
                                await asyncio.wait_for(session.close(), timeout=0.5)
                            elif hasattr(session, '__aexit__'):
                                await asyncio.wait_for(session.__aexit__(None, None, None), timeout=0.5)
                        except (asyncio.TimeoutError, Exception) as e:
                            logger.debug(f"FastMCP session close timeout/error for {server_name}: {e}")
                    elif session and hasattr(session, 'shutdown'):
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
                            logger.debug(f"RuntimeError during exit_stack cleanup for {server_name}: {e}")
                    except (asyncio.TimeoutError, Exception) as e:
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
                builder = get_mcp_server_builder()
                # 빌드된 서버 디렉토리 정리 (선택적)
                # 실제 서버 프로세스는 ProcessManager가 관리하므로 여기서는 로깅만
                logger.debug("[MCP][cleanup] Dynamic servers will be cleaned up by ProcessManager")
            except Exception as e:
                logger.debug(f"[MCP][cleanup] Builder cleanup skipped: {e}")
        
        logger.info("MCP Hub cleanup completed")

    def start_shutdown(self):
        """외부에서 종료 시작 시 호출 - 신규 연결 차단"""
        self.stopping = True
    
    async def call_llm_async(self, model: str, messages: List[Dict[str, str]], 
                           temperature: float = 0.1, max_tokens: int = 4000) -> Dict[str, Any]:
        """LLM 호출은 llm_manager를 통해 수행하도록 강제 (Gemini 직결)."""
        raise RuntimeError("call_llm_async via MCP Hub is disabled. Use llm_manager for Gemini.")
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], citation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Tool 실행 - MCP 프로토콜만 사용 (9대 혁신: ToolTrace 추적 통합).

        실행 우선순위:
        1. MCP 서버에서 Tool 실행 (server_name::tool_name 형식 또는 tool_name으로 찾기)
        2. 실패 시 명확한 에러 반환 (fallback 없음)
        
        Args:
            tool_name: 도구 이름
            parameters: 도구 파라미터
            citation_id: Citation ID (optional, ToolTrace 추적용)
        """
        import time
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
            "browser_navigate", "browser_extract", "browser_screenshot", "browser_interact",
            "run_shell_command", "run_interactive_command", "run_background_command",
            "create_file", "read_file", "write_file", "edit_file", "list_files", "delete_file",
            "filesystem", "browser", "shell"  # 일반적인 이름도 지원
        }

        if tool_name in local_tools or any(tool_name.startswith(prefix) for prefix in ["browser_", "shell_", "file_"]):
            logger.debug(f"Executing local tool: {tool_name}")
            try:
                # ToolResult를 Dict로 변환하여 반환
                if tool_name.startswith("browser") or tool_name == "browser":
                    result = await _execute_browser_tool(tool_name, parameters)
                elif tool_name.startswith("shell") or tool_name == "shell":
                    result = await _execute_shell_tool(tool_name, parameters)
                elif tool_name.startswith(("file", "create_", "read_", "write_", "edit_", "list_", "delete_")) or tool_name == "filesystem":
                    result = await _execute_file_tool(tool_name, parameters)
                else:
                    # 일반적인 경우 data tool로 처리
                    result = await _execute_data_tool(tool_name, parameters)

                execution_time = time.time() - start_time
                return {
                    "success": result.success,
                    "data": result.data,
                    "error": result.error,
                    "execution_time": result.execution_time,
                    "confidence": result.confidence,
                    "source": "local_tool"
                }

            except Exception as e:
                logger.error(f"Local tool execution failed: {tool_name} - {e}")
                execution_time = time.time() - start_time
                return {
                    "success": False,
                    "error": f"Local tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0,
                    "source": "local_tool"
                }

        # MCP 서버 정보 추출
        mcp_server = None
        mcp_tool_name = None
        if "::" in tool_name:
            parts = tool_name.split("::", 1)
            mcp_server = parts[0]
            mcp_tool_name = parts[1] if len(parts) > 1 else tool_name
        
        # Citation ID가 없으면 생성 (임시)
        if not citation_id:
            try:
                from src.generation.citation_manager import CitationManager
                # 전역 citation manager가 있다면 사용, 없으면 임시 생성
                # 실제로는 orchestrator에서 관리하는 citation_id를 전달받아야 함
                citation_id = f"TEMP-{tool_id}"
            except Exception:
                citation_id = f"TEMP-{tool_id}"

        # 출력 매니저 통합
        from src.utils.output_manager import get_output_manager, OutputLevel, ToolExecutionResult
        output_manager = get_output_manager()

        # 도구 실행 시작 알림
        await output_manager.output(
            f"🔧 도구 '{tool_name}' 실행 시작...",
            level=OutputLevel.SERVICE,
            agent_name="mcp_integration"
        )

        logger.info(f"[MCP][exec.start] tool={tool_name} params_keys={list(parameters.keys())}")
        logger.info(f"[MCP][exec.start] parameters_preview={str(parameters)[:200]}...")
        
        # 학술 도구 라우팅 (arxiv, scholar) - MCP 서버 우선 사용
        if tool_name in ["arxiv", "scholar"]:
            logger.info(f"[MCP][exec.academic] Routing {tool_name} to _execute_academic_tool (MCP server first)")
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
                                "arxiv",
                                arxiv_tool_name,
                                parameters
                            )
                
                # MCP 결과가 있으면 사용, 없으면 로컬 fallback
                if mcp_result:
                    from src.core.mcp_integration import ToolResult
                    tool_result = ToolResult(
                        success=True,
                        data=mcp_result if isinstance(mcp_result, dict) else {"result": mcp_result},
                        execution_time=time.time() - start_time,
                        confidence=0.9
                    )
                else:
                    # 로컬 fallback
                    from src.core.mcp_integration import _execute_academic_tool, ToolResult
                    tool_result = await _execute_academic_tool(tool_name, parameters)
                
                execution_time = time.time() - start_time
                logger.info(f"[MCP][exec.academic.success] {tool_name} routing succeeded: success={tool_result.success}")

                # 도구 실행 결과 표시
                result_summary = ""
                if tool_result.success and tool_result.data:
                    if isinstance(tool_result.data, dict) and 'results' in tool_result.data:
                        result_count = len(tool_result.data['results'])
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
                    error_message=tool_result.error
                )
                await output_manager.output_tool_execution(tool_exec_result)

                return {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "mcp_academic" if mcp_result else "local_academic"
                }
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[MCP][exec.academic.error] {tool_name} routing failed: {e}", exc_info=True)

                # 도구 실행 실패 결과 표시
                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    execution_time=execution_time,
                    result_summary=f"학술 도구 실행 실패: {str(e)[:100]}...",
                    confidence=0.0,
                    error_message=str(e)
                )
                await output_manager.output_tool_execution(tool_exec_result)

                return {
                    "success": False,
                    "data": None,
                    "error": f"Academic tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0,
                    "source": "academic_routing_failed"
                }
        
        # 검색 도구는 먼저 라우팅 확인 (도구 찾기 전에)
        if tool_name in ["g-search", "ddg_search", "mcp_search", "tavily", "exa"]:
            logger.info(f"[MCP][exec.route] Routing {tool_name} to _execute_search_tool (tool_name type: {type(tool_name)})")
            try:
                from src.core.mcp_integration import _execute_search_tool, ToolResult
                tool_result = await _execute_search_tool(tool_name, parameters)
                execution_time = time.time() - start_time
                logger.info(f"[MCP][exec.route.success] {tool_name} routing succeeded: success={tool_result.success}")

                # 도구 실행 결과 표시
                result_summary = ""
                if tool_result.success and tool_result.data:
                    if isinstance(tool_result.data, dict) and 'results' in tool_result.data:
                        result_count = len(tool_result.data['results'])
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
                    error_message=tool_result.error
                )
                await output_manager.output_tool_execution(tool_exec_result)

                return {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "mcp_search"
                }
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[MCP][exec.route.error] {tool_name} routing failed: {e}", exc_info=True)

                # 도구 실행 실패 결과 표시
                tool_exec_result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    execution_time=execution_time,
                    result_summary=f"라우팅 실패: {str(e)[:100]}...",
                    confidence=0.0,
                    error_message=str(e)
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
                    "source": "mcp_search_routing_failed"
                }
        
        # 브라우저 도구 라우팅 (우선 처리)
        if tool_name.startswith("browser_"):
            logger.info(f"[MCP][exec.browser] Routing {tool_name} to _execute_browser_tool")
            try:
                from src.core.mcp_integration import _execute_browser_tool, ToolResult
                tool_result = await _execute_browser_tool(tool_name, parameters)
                execution_time = time.time() - start_time
                
                result_summary = ""
                if tool_result.success and tool_result.data:
                    if isinstance(tool_result.data, dict):
                        if "extracted_data" in tool_result.data:
                            result_summary = f"콘텐츠 추출 완료 ({tool_result.data.get('content_length', 0)}자)"
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
                    error_message=tool_result.error
                )
                await output_manager.output_tool_execution(tool_exec_result)
                
                return {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "browser"
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
                    error_message=str(e)
                )
                await output_manager.output_tool_execution(tool_exec_result)
                
                return {
                    "success": False,
                    "data": None,
                    "error": f"Browser tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0
                }
        
        # 문서 생성 도구 라우팅
        if tool_name.startswith("generate_"):
            logger.info(f"[MCP][exec.document] Routing {tool_name} to _execute_document_tool")
            try:
                from src.core.mcp_integration import _execute_document_tool, ToolResult
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
                    error_message=tool_result.error
                )
                await output_manager.output_tool_execution(tool_exec_result)
                
                return {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "document"
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
                    error_message=str(e)
                )
                await output_manager.output_tool_execution(tool_exec_result)
                
                return {
                    "success": False,
                    "data": None,
                    "error": f"Document tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0
                }
        
        # Shell 도구 라우팅
        if tool_name in ["run_shell_command", "run_interactive_command", "run_background_command"]:
            logger.info(f"[MCP][exec.shell] Routing {tool_name} to _execute_shell_tool")
            try:
                from src.core.mcp_integration import _execute_shell_tool, ToolResult
                tool_result = await _execute_shell_tool(tool_name, parameters)
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
                    error_message=tool_result.error
                )
                await output_manager.output_tool_execution(tool_exec_result)
                
                result_dict = {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "shell"
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
                    error_message=str(e)
                )
                await output_manager.output_tool_execution(tool_exec_result)
                
                return {
                    "success": False,
                    "data": None,
                    "error": f"Shell tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0,
                    "source": "shell"
                }
        
        # Git 도구 라우팅
        if tool_name in ["git_status", "git_commit", "git_push", "git_create_pr", "git_commit_push_pr", "git_create_branch"]:
            logger.info(f"[MCP][exec.git] Routing {tool_name} to _execute_git_tool")
            try:
                from src.core.mcp_integration import _execute_git_tool, ToolResult
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
                    error_message=tool_result.error
                )
                await output_manager.output_tool_execution(tool_exec_result)
                
                result_dict = {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "git"
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
                    error_message=str(e)
                )
                await output_manager.output_tool_execution(tool_exec_result)
                
                return {
                    "success": False,
                    "data": None,
                    "error": f"Git tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0,
                    "source": "git"
                }
        
        # 파일 도구 라우팅
        if tool_name in ["create_file", "read_file", "write_file", "edit_file", "list_files", "delete_file"]:
            logger.info(f"[MCP][exec.file] Routing {tool_name} to _execute_file_tool")
            try:
                from src.core.mcp_integration import _execute_file_tool, ToolResult
                tool_result = await _execute_file_tool(tool_name, parameters)
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
                    error_message=tool_result.error
                )
                await output_manager.output_tool_execution(tool_exec_result)
                
                return {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error": tool_result.error,
                    "execution_time": execution_time,
                    "confidence": tool_result.confidence,
                    "source": "file"
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
                    error_message=str(e)
                )
                await output_manager.output_tool_execution(tool_exec_result)
                
                return {
                    "success": False,
                    "data": None,
                    "error": f"File tool execution failed: {str(e)}",
                    "execution_time": execution_time,
                    "confidence": 0.0
                }
        
        # Tool 찾기 (server_name::tool_name 또는 tool_name)
        # 먼저 tool_name이 이미 server_name::tool_name 형식인지 확인
        if "::" in tool_name:
            # 이미 전체 이름 형식이면 직접 찾기
            tool_info = self.registry.get_tool_info(tool_name)
        else:
            # tool_name만 주어진 경우 Registry에서 찾기
            tool_info = self.registry.get_tool_info(tool_name)
        
        # tool_name으로 직접 찾기 실패 시, 모든 MCP 서버에서 server_name::tool_name 형식으로 찾기
        if not tool_info:
            for registered_name in self.registry.get_all_tool_names():
                # 이미 전체 이름 형식이면 정확히 매칭
                if "::" in tool_name and registered_name == tool_name:
                    tool_info = self.registry.get_tool_info(registered_name)
                    logger.info(f"Found tool by exact match: {tool_name}")
                    break
                # server_name::tool_name 형식에서 tool_name 부분만 추출하여 비교
                elif "::" in registered_name:
                    _, original_tool_name = registered_name.split("::", 1)
                    if original_tool_name == tool_name:
                        tool_info = self.registry.get_tool_info(registered_name)
                        logger.info(f"Found tool {tool_name} as {registered_name}")
                        break
                elif registered_name == tool_name:
                    tool_info = self.registry.get_tool_info(registered_name)
                    break
        
        if not tool_info:
            # Registry에서 직접 찾기
            tool_info = self.registry.tools.get(tool_name)
        
        if not tool_info:
            # 하위 호환성: self.tools에서 찾기
            tool_info = self.tools.get(tool_name)
        
        if not tool_info:
            # MCP Builder를 통한 자동 서버 생성 시도
            if self.config.builder_enabled:
                logger.info(f"[MCP][builder] Tool '{tool_name}' not found, attempting auto-build...")
                try:
                    from src.core.mcp_server_builder import get_mcp_server_builder
                    builder = get_mcp_server_builder()
                    
                    # 서버 빌드
                    build_result = await builder.build_mcp_server(
                        tool_name=tool_name,
                        parameters=parameters,
                        error_context=None
                    )
                    
                    if build_result.get("success"):
                        server_name = build_result["server_name"]
                        server_path = build_result["server_path"]
                        
                        logger.info(f"[MCP][builder] Server built successfully: {server_name}")
                        
                        # 동적 서버 등록
                        registered = await self._register_dynamic_server(server_name, server_path)
                        
                        if registered:
                            logger.info(f"[MCP][builder] Server registered: {server_name}, retrying tool execution...")
                            # 도구 실행 재시도
                            return await self.execute_tool(tool_name, parameters)
                        else:
                            logger.warning(f"[MCP][builder] Failed to register server: {server_name}")
                    else:
                        logger.warning(f"[MCP][builder] Server build failed: {build_result.get('error')}")
                except Exception as builder_error:
                    logger.error(f"[MCP][builder] Builder error: {builder_error}", exc_info=True)
            
            # 사용 가능한 모든 tool 목록 로깅
            available_tools = self.registry.get_all_tool_names()
            execution_time = time.time() - start_time
            logger.error(f"[MCP][exec.unknown] tool={tool_name} available={available_tools}")

            # 도구 찾기 실패 결과 표시
            available_preview = ', '.join(available_tools[:5]) + ('...' if len(available_tools) > 5 else '')
            tool_exec_result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                execution_time=execution_time,
                result_summary=f"알 수 없는 도구. 사용 가능한 도구: {available_preview}",
                confidence=0.0,
                error_message=f"Unknown tool: {tool_name}"
            )
            await output_manager.output_tool_execution(tool_exec_result)

            return {
                "success": False,
                "data": None,
                "error": f"Unknown tool: {tool_name}. Available tools: {', '.join(available_tools[:10])}",
                "execution_time": execution_time,
                "confidence": 0.0
            }

        try:
            # 1. MCP Tool인지 확인 및 실행 시도 - tool_info에서 직접 정보 추출
            found_tool_name = tool_name
            mcp_info = None
            
            # tool_info가 있으면 MCP 도구인지 확인하고 mcp_info 추출
            if tool_info:
                # tool_info에서 mcp_server 정보 확인
                mcp_server = tool_info.mcp_server
                if mcp_server:
                    # server_name::tool_name 형식에서 server_name과 tool_name 추출
                    if "::" in tool_name:
                        server_name, original_tool_name = tool_name.split("::", 1)
                        mcp_info = (server_name, original_tool_name)
                        found_tool_name = tool_name
                        logger.info(f"[MCP][exec.resolve] Using tool_info: {tool_name} -> server={server_name}, tool={original_tool_name}")
                    else:
                        # tool_name만 있는 경우 tool_info의 mcp_server 사용
                        # tool_name이 실제 서버의 원본 tool name인지 확인 필요
                        # registry에서 찾기
                        for registered_name in self.registry.get_all_tool_names():
                            if registered_name == tool_name and self.registry.is_mcp_tool(registered_name):
                                mcp_info = self.registry.get_mcp_server_info(registered_name)
                                found_tool_name = registered_name
                                break
                            elif "::" in registered_name:
                                _, original_tool_name = registered_name.split("::", 1)
                                if original_tool_name == tool_name:
                                    mcp_info = self.registry.get_mcp_server_info(registered_name)
                                    found_tool_name = registered_name
                                    logger.info(f"[MCP][exec.resolve] Found {tool_name} as {registered_name}")
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
                            if original_tool_name == tool_name and self.registry.is_mcp_tool(registered_name):
                                mcp_info = self.registry.get_mcp_server_info(registered_name)
                                found_tool_name = registered_name
                                logger.info(f"[MCP][exec.resolve] {tool_name} -> {registered_name}")
                                break
                        elif registered_name == tool_name and self.registry.is_mcp_tool(registered_name):
                            mcp_info = self.registry.get_mcp_server_info(registered_name)
                            found_tool_name = registered_name
                            break
            
            if mcp_info:
                server_name, original_tool_name = mcp_info
                
                # MCP 서버 연결 확인
                if server_name in self.mcp_sessions:
                    try:
                        logger.info(f"[MCP][exec.try] server={server_name} tool={tool_name} as={found_tool_name}")
                        mcp_result = await self._execute_via_mcp_server(
                            server_name,
                            original_tool_name,
                            parameters
                        )
                        
                        if mcp_result:
                            # MCP 결과를 ToolResult 형식으로 변환
                            # 에러 응답 체크
                            import json
                            import re
                            
                            result_lower = str(mcp_result).lower() if mcp_result else ""
                            error_patterns = [
                                r'\b(failed|error|invalid_token|authentication failed)\b',
                                r'\b(401|404|500|502|503|504)\b',
                                r'bad gateway',
                                r'not found',
                                r'unauthorized',
                                r'<!doctype html>',
                                r'<html',
                                r'<title>.*error.*</title>'
                            ]
                            
                            is_error = False
                            error_msg = None
                            for pattern in error_patterns:
                                if re.search(pattern, result_lower):
                                    is_error = True
                                    if '401' in result_lower:
                                        error_msg = "Authentication failed (401)"
                                    elif '404' in result_lower:
                                        error_msg = "Not found (404)"
                                    elif '502' in result_lower or 'bad gateway' in result_lower:
                                        error_msg = "Bad gateway (502)"
                                    elif '500' in result_lower:
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
                                    error_message=error_msg
                                )
                                await output_manager.output_tool_execution(tool_exec_result)

                                return {
                                    "success": False,
                                    "data": None,
                                    "error": f"MCP tool returned error: {error_msg}",
                                    "execution_time": execution_time,
                                    "confidence": 0.0,
                                    "source": "mcp"
                                }
                            
                            # 문자열인 경우 마크다운 파싱 시도
                            if isinstance(mcp_result, str):
                                # JSON 시도
                                try:
                                    result_data = json.loads(mcp_result)
                                except:
                                    # 마크다운 파싱
                                    results = []
                                    lines = mcp_result.strip().split('\n')
                                    current_result = None
                                    
                                    for line in lines:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        
                                        link_match = re.match(r'^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)', line)
                                        if link_match:
                                            if current_result:
                                                results.append(current_result)
                                            title = link_match.group(1)
                                            url = link_match.group(2)
                                            current_result = {"title": title, "url": url, "snippet": ""}
                                        elif current_result and line:
                                            if current_result["snippet"]:
                                                current_result["snippet"] += " " + line
                                            else:
                                                current_result["snippet"] = line
                                    
                                    if current_result:
                                        results.append(current_result)
                                    
                                    if results:
                                        result_data = {"results": results}
                                    else:
                                        # 파싱 실패 시 원본 텍스트
                                        result_data = {"result": mcp_result}
                            else:
                                result_data = mcp_result if isinstance(mcp_result, dict) else {"result": mcp_result}

                            execution_time = time.time() - start_time

                            # MCP 도구 성공 결과 표시
                            result_summary = ""
                            if isinstance(result_data, dict):
                                if 'results' in result_data and isinstance(result_data['results'], list):
                                    result_count = len(result_data['results'])
                                    result_summary = f"{result_count}개 결과 반환됨"
                                elif 'result' in result_data:
                                    result_summary = f"결과 반환됨 ({type(result_data['result']).__name__})"
                                else:
                                    result_summary = f"데이터 반환됨 ({len(result_data)}개 필드)"
                            else:
                                result_summary = f"결과 반환됨 ({type(result_data).__name__})"

                            tool_exec_result = ToolExecutionResult(
                                tool_name=tool_name,
                                success=True,
                                execution_time=execution_time,
                                result_summary=result_summary,
                                confidence=0.9
                            )
                            await output_manager.output_tool_execution(tool_exec_result)

                            result_dict = {
                                "success": True,
                                "data": result_data,
                                "error": None,
                                "execution_time": execution_time,
                                "confidence": 0.9,
                                "source": "mcp"
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
                    except Exception as mcp_error:
                        execution_time = time.time() - start_time
                        logger.error(f"[MCP][exec.error] server={server_name} tool={tool_name} err={mcp_error}")

                        # MCP 실행 실패 결과 표시
                        tool_exec_result = ToolExecutionResult(
                            tool_name=tool_name,
                            success=False,
                            execution_time=execution_time,
                            result_summary=f"MCP 실행 실패: {str(mcp_error)[:100]}...",
                            confidence=0.0,
                            error_message=str(mcp_error)
                        )
                        await output_manager.output_tool_execution(tool_exec_result)

                        # MCP 실패 시 에러 반환 (fallback 제거)
                        return {
                            "success": False,
                            "data": None,
                            "error": f"MCP tool execution failed: {str(mcp_error)}",
                            "execution_time": execution_time,
                            "confidence": 0.0,
                            "source": "mcp"
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
                        from src.core.mcp_integration import _execute_search_tool, ToolResult
                        tool_result = await _execute_search_tool(tool_name, parameters)
                    elif category == ToolCategory.DATA:
                        from src.core.mcp_integration import _execute_data_tool
                        tool_result = await _execute_data_tool(tool_name, parameters)
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
                        # 기본적으로 데이터 도구로 처리
                        from src.core.mcp_integration import _execute_data_tool
                        tool_result = await _execute_data_tool(tool_name, parameters)
                    
                    execution_time = time.time() - start_time
                    
                    # 결과 요약 생성
                    result_summary = ""
                    if tool_result.success and tool_result.data:
                        if isinstance(tool_result.data, dict):
                            if 'results' in tool_result.data:
                                result_count = len(tool_result.data['results'])
                                result_summary = f"{result_count}개 결과 반환됨"
                            elif 'content' in tool_result.data:
                                content_len = len(str(tool_result.data['content']))
                                result_summary = f"콘텐츠 반환됨 ({content_len}자)"
                            else:
                                result_summary = f"데이터 반환됨 ({type(tool_result.data).__name__})"
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
                        error_message=tool_result.error
                    )
                    await output_manager.output_tool_execution(tool_exec_result)
                    
                    result_dict = {
                        "success": tool_result.success,
                        "data": tool_result.data,
                        "error": tool_result.error,
                        "execution_time": execution_time,
                        "confidence": tool_result.confidence,
                        "source": "local"
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
                    logger.error(f"[MCP][exec.local.error] Local tool execution failed: {local_error}")
                    
                    tool_exec_result = ToolExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        execution_time=execution_time,
                        result_summary=f"로컬 도구 실행 실패: {str(local_error)[:100]}...",
                        confidence=0.0,
                        error_message=str(local_error)
                    )
                    await output_manager.output_tool_execution(tool_exec_result)
                    
                    return {
                        "success": False,
                        "data": None,
                        "error": f"Local tool execution failed: {str(local_error)}",
                        "execution_time": execution_time,
                        "confidence": 0.0,
                        "source": "local"
                    }
            
            # MCP 도구도 로컬 도구도 아닌 경우 MCP Builder 시도
            if self.config.builder_enabled:
                logger.info(f"[MCP][builder] Tool '{tool_name}' not available, attempting auto-build...")
                try:
                    from src.core.mcp_server_builder import get_mcp_server_builder
                    builder = get_mcp_server_builder()
                    
                    # 서버 빌드
                    build_result = await builder.build_mcp_server(
                        tool_name=tool_name,
                        parameters=parameters,
                        error_context=f"Tool not found in MCP servers or local tools"
                    )
                    
                    if build_result.get("success"):
                        server_name = build_result["server_name"]
                        server_path = build_result["server_path"]
                        
                        logger.info(f"[MCP][builder] Server built successfully: {server_name}")
                        
                        # 동적 서버 등록
                        registered = await self._register_dynamic_server(server_name, server_path)
                        
                        if registered:
                            logger.info(f"[MCP][builder] Server registered: {server_name}, retrying tool execution...")
                            # 도구 실행 재시도
                            return await self.execute_tool(tool_name, parameters)
                        else:
                            logger.warning(f"[MCP][builder] Failed to register server: {server_name}")
                    else:
                        logger.warning(f"[MCP][builder] Server build failed: {build_result.get('error')}")
                except Exception as builder_error:
                    logger.error(f"[MCP][builder] Builder error: {builder_error}", exc_info=True)
            
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
                error_message=error_msg
            )
            await output_manager.output_tool_execution(tool_exec_result)

            return {
                "success": False,
                "data": None,
                "error": error_msg,
                "execution_time": execution_time,
                "confidence": 0.0,
                "source": "unknown"
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
                error_message=str(e)
            )
            await output_manager.output_tool_execution(tool_exec_result)

            return {
                "success": False,
                "data": None,
                "error": str(e),
                "execution_time": execution_time,
                "confidence": 0.0
            }
    
    def get_tool_for_category(self, category: ToolCategory) -> Optional[str]:
        """카테고리에 해당하는 도구 반환 - Registry 기반."""
        tools_in_category = self.registry.get_tools_by_category(category)
        return tools_in_category[0] if tools_in_category else None
    
    def get_available_tools(self) -> List[str]:
        """사용 가능한 도구 목록 반환 - Registry 기반."""
        # Registry의 모든 Tool 이름 반환
        return self.registry.get_all_tool_names()
    
    async def get_tool_for_execution(self, tool_name: str, execution_id: Optional[str] = None) -> Optional[Any]:
        """
        실행 컨텍스트별 도구 반환 (ROMA 스타일).
        
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
                'tools': {},
                'created_at': datetime.now()
            }
        
        execution_session = self._execution_sessions[execution_id]
        
        # 도구가 이미 캐시되어 있으면 반환
        if tool_name in execution_session['tools']:
            return execution_session['tools'][tool_name]
        
        # 도구 초기화 및 캐싱
        # LangChain Tool이 있으면 반환, 없으면 ToolInfo 반환
        tool = self.registry.get_langchain_tool(tool_name)
        if not tool:
            # LangChain Tool이 없으면 ToolInfo 반환
            tool = self.registry.get_tool_info(tool_name)
        
        if tool:
            execution_session['tools'][tool_name] = tool
            logger.debug(f"Tool {tool_name} cached for execution {execution_id}")
        
        return tool
    
    async def cleanup_execution(self, execution_id: str):
        """
        실행 종료 시 세션 정리 (ROMA 스타일).
        
        실행별로 관리된 도구 인스턴스와 세션을 정리합니다.
        
        Args:
            execution_id: 정리할 실행 ID
        """
        if execution_id in self._execution_sessions:
            session = self._execution_sessions[execution_id]
            tools_count = len(session.get('tools', {}))
            
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
            "servers": {}
        }
        
        logger.info(f"Checking {len(self.mcp_server_configs)} MCP servers...")
        
        for server_name, server_config in self.mcp_server_configs.items():
            server_info = {
                "name": server_name,
                "type": server_config.get("type", "stdio"),
                "connected": server_name in self.mcp_sessions,
                "tools_count": 0,
                "tools": [],
                "error": None
            }
            
            # 연결 타입 정보
            if server_config.get("type") == "http" or "httpUrl" in server_config or "url" in server_config:
                server_info["type"] = "http"
                server_info["url"] = server_config.get("httpUrl") or server_config.get("url", "unknown")
            else:
                server_info["type"] = "stdio"
                server_info["command"] = server_config.get("command", "unknown")
                server_info["args"] = server_config.get("args", [])
            
            # 연결 상태 확인
            if server_name in self.mcp_sessions:
                session = self.mcp_sessions[server_name]
                # 세션이 유효한지 확인
                try:
                    if hasattr(session, '_transport') and session._transport:
                        server_info["connected"] = True
                    else:
                        server_info["connected"] = False
                        server_info["error"] = "Session transport not available"
                except:
                    server_info["connected"] = False
                    server_info["error"] = "Session check failed"
                
                # 제공하는 Tool 목록 확인
                if server_name in self.mcp_tools_map:
                    tools = self.mcp_tools_map[server_name]
                    server_info["tools_count"] = len(tools)
                    server_info["tools"] = list(tools.keys())
                    
                    # 등록된 Tool 이름 (server_name::tool_name 형식)
                    registered_tools = [
                        name for name in self.registry.get_all_tool_names()
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
            "connection_rate": f"{connected}/{len(self.mcp_server_configs)}"
        }
        
        return server_status
    
    async def health_check(self) -> Dict[str, Any]:
        """강화된 헬스 체크 - OpenRouter, Gemini 2.5 Flash Lite, MCP 도구 검증."""
        try:
            health_status = {
                "mcp_enabled": self.config.enabled,
                "tools_available": len(self.tools),
                "timestamp": datetime.now().isoformat()
            }
            
            # 1. OpenRouter 연결 테스트
            try:
                test_messages = [
                    {"role": "system", "content": "Health check test."},
                    {"role": "user", "content": "Respond with 'OK' if you can process this request."}
                ]
                
                test_response = await self.openrouter_client.generate_response(
                    model=self.llm_config.primary_model,
                    messages=test_messages,
                    temperature=0.1,
                    max_tokens=50
                )
                
                openrouter_healthy = test_response and "choices" in test_response
                health_status.update({
                    "openrouter_connected": openrouter_healthy,
                    "primary_model": self.llm_config.primary_model,
                    "rate_limit_remaining": getattr(self.openrouter_client, 'rate_limit_remaining', 'unknown')
                })
                
                if not openrouter_healthy:
                    health_status["overall_health"] = "unhealthy"
                    health_status["critical_error"] = "OpenRouter connection failed"
                    return health_status
                    
            except Exception as e:
                health_status.update({
                    "openrouter_connected": False,
                    "openrouter_error": str(e),
                    "overall_health": "unhealthy",
                    "critical_error": f"OpenRouter health check failed: {e}"
                })
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
                    
                    tool_health[tool] = test_result.get('success', False)
                    if not test_result.get('success', False):
                        failed_tools.append(tool)
                        
                except Exception as e:
                    tool_health[tool] = False
                    failed_tools.append(tool)
                    logger.warning(f"Tool {tool} health check failed: {e}")
            
            health_status.update({
                "tool_health": tool_health,
                "failed_tools": failed_tools,
                "essential_tools_healthy": len(failed_tools) == 0
            })
            
            # 3. 전체 상태 결정
            if len(failed_tools) > 0:
                health_status["overall_health"] = "unhealthy"
                health_status["critical_error"] = f"Essential tools failed: {', '.join(failed_tools)}"
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
                "timestamp": datetime.now().isoformat()
            }


# Global MCP Hub instance (lazy initialization)
_mcp_hub = None

def get_mcp_hub() -> 'UniversalMCPHub':
    """Get or initialize global MCP Hub."""
    global _mcp_hub
    if _mcp_hub is None:
        _mcp_hub = UniversalMCPHub()
    return _mcp_hub

async def get_available_tools() -> List[str]:
    """사용 가능한 도구 목록 반환."""
    mcp_hub = get_mcp_hub()
    return mcp_hub.get_available_tools()


async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """MCP 도구 실행 - UniversalMCPHub의 execute_tool 사용 (with caching)."""
    from src.core.result_cache import get_result_cache
    
    result_cache = get_result_cache()
    
    # 캐시 확인
    cached_result = await result_cache.get(
        tool_name=tool_name,
        parameters=parameters,
        check_similarity=True
    )
    
    if cached_result:
        logger.debug(f"[MCP][execute_tool] Cache hit for {tool_name}")
        return cached_result
    
    # MCP Hub 실행
    mcp_hub = get_mcp_hub()
    
    # MCP Hub가 초기화되지 않았으면 초기화
    if not mcp_hub.mcp_sessions:
        logger.info("[MCP][execute_tool] MCP Hub not initialized, initializing...")
        await mcp_hub.initialize_mcp()
    
    result = await mcp_hub.execute_tool(tool_name, parameters)
    
    # 성공한 결과만 캐시에 저장
    if result.get("success", False):
        # TTL 결정: 검색/데이터 도구는 1시간, 다른 도구는 30분
        ttl = 3600 if any(keyword in tool_name.lower() for keyword in ['search', 'fetch', 'data']) else 1800
        await result_cache.set(
            tool_name=tool_name,
            parameters=parameters,
            value=result,
            ttl=ttl
        )
        logger.debug(f"[MCP][execute_tool] Cached result for {tool_name}")
    
    return result


# 동기화 헬퍼 함수들 (LangChain Tool용)
def _execute_search_tool_sync(tool_name: str, parameters: Dict[str, Any]) -> str:
    """동기 버전 - LangChain Tool에서 호출."""
    try:
        # 이미 실행 중인 이벤트 루프가 있으면 새 스레드에서 실행
        import concurrent.futures
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 실행 중인 루프가 있으면 새 스레드에서 실행
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _execute_search_tool(tool_name, parameters))
                    # timeout 설정으로 무한 대기 방지
                    result = future.result(timeout=300)  # 최대 5분
            else:
                result = loop.run_until_complete(_execute_search_tool(tool_name, parameters))
        except RuntimeError:
            # 이벤트 루프가 없으면 새로 생성
            result = asyncio.run(_execute_search_tool(tool_name, parameters))
        
        if result.success:
            import json
            return json.dumps(result.data, ensure_ascii=False, indent=2)
        else:
            raise RuntimeError(result.error or "Tool execution failed")
    except Exception as e:
        raise RuntimeError(f"Tool execution failed: {str(e)}")

def _execute_academic_tool_sync(tool_name: str, parameters: Dict[str, Any]) -> str:
    """동기 버전 - LangChain Tool에서 호출."""
    try:
        import concurrent.futures
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _execute_academic_tool(tool_name, parameters))
                    # timeout 설정으로 무한 대기 방지
                    result = future.result(timeout=300)  # 최대 5분
            else:
                result = loop.run_until_complete(_execute_academic_tool(tool_name, parameters))
        except RuntimeError:
            result = asyncio.run(_execute_academic_tool(tool_name, parameters))
        
        if result.success:
            import json
            return json.dumps(result.data, ensure_ascii=False, indent=2)
        else:
            raise RuntimeError(result.error or "Tool execution failed")
    except Exception as e:
        raise RuntimeError(f"Tool execution failed: {str(e)}")

def _execute_data_tool_sync(tool_name: str, parameters: Dict[str, Any]) -> str:
    """동기 버전 - LangChain Tool에서 호출."""
    try:
        import concurrent.futures
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _execute_data_tool(tool_name, parameters))
                    # timeout 설정으로 무한 대기 방지
                    result = future.result(timeout=300)  # 최대 5분
            else:
                result = loop.run_until_complete(_execute_data_tool(tool_name, parameters))
        except RuntimeError:
            result = asyncio.run(_execute_data_tool(tool_name, parameters))
        
        if result.success:
            import json
            return json.dumps(result.data, ensure_ascii=False, indent=2)
        else:
            raise RuntimeError(result.error or "Tool execution failed")
    except Exception as e:
        raise RuntimeError(f"Tool execution failed: {str(e)}")

def _execute_code_tool_sync(tool_name: str, parameters: Dict[str, Any]) -> str:
    """동기 버전 - LangChain Tool에서 호출."""
    try:
        import concurrent.futures
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _execute_code_tool(tool_name, parameters))
                    # timeout 설정으로 무한 대기 방지
                    result = future.result(timeout=300)  # 최대 5분
            else:
                result = loop.run_until_complete(_execute_code_tool(tool_name, parameters))
        except RuntimeError:
            result = asyncio.run(_execute_code_tool(tool_name, parameters))
        
        if result.success:
            import json
            return json.dumps(result.data, ensure_ascii=False, indent=2)
        else:
            raise RuntimeError(result.error or "Tool execution failed")
    except Exception as e:
        raise RuntimeError(f"Tool execution failed: {str(e)}")


# DuckDuckGo 요청 빈도 제한을 위한 전역 변수
_ddg_last_request_time = {}
_ddg_request_lock = None

def _get_ddg_lock():
    """DuckDuckGo 요청 락을 지연 초기화."""
    global _ddg_request_lock
    if _ddg_request_lock is None:
        _ddg_request_lock = asyncio.Lock()
    return _ddg_request_lock

async def _fallback_to_ddg_search(query: str, max_results: int) -> ToolResult:
    """MCP 서버 실패 시 DDG search로 fallback."""
    try:
        from src.core.tools.native_search import search_duckduckgo_json
        
        logger.info(f"[MCP][fallback] Using DDG search fallback for query: {query}")
        result = search_duckduckgo_json(query, max_results)
        
        if result and isinstance(result, dict):
            results = result.get("results", [])
            if results:
                return ToolResult(
                    success=True,
                    data={"results": results, "total_results": len(results)},
                    tool_name="ddg_search",
                    source="native_ddg_fallback"
                )
        
        # 결과가 없거나 형식이 잘못된 경우
        return ToolResult(
            success=False,
            data=None,
            error="DDG search fallback returned no results",
            tool_name="ddg_search",
            source="native_ddg_fallback"
        )
    except Exception as e:
        logger.error(f"[MCP][fallback] DDG search fallback failed: {e}")
        return ToolResult(
            success=False,
            data=None,
            error=f"DDG search fallback error: {str(e)}",
            tool_name="ddg_search",
            source="native_ddg_fallback"
        )

async def _execute_search_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """검색 도구 실행 - src/utils에서 직접 사용."""
    import time
    start_time = time.time()
    
    # src/utils에서 직접 사용 (MCP 서버로 실행하지 않음)
    try:
        from src.utils.search_utils import search_duckduckgo
        
        query = parameters.get("query", "")
        num_results = parameters.get("num_results", 10) or parameters.get("max_results", 10)
        
        if not query:
            return ToolResult(
                success=False,
                data=None,
                error="query parameter is required",
                execution_time=time.time() - start_time,
                confidence=0.0
            )
        
        # src/utils의 search_duckduckgo 직접 호출
        result = await search_duckduckgo(query, num_results)
        
        if result.get("success"):
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": result.get("results", []),
                    "count": result.get("count", 0),
                    "provider": result.get("provider", "duckduckgo"),
                    "source": "embedded_search"
                },
                execution_time=time.time() - start_time,
                confidence=0.9
            )
        else:
            return ToolResult(
                success=False,
                data=None,
                error=result.get("error", "Search failed"),
                execution_time=time.time() - start_time,
                confidence=0.0
            )
    except ImportError:
        # embedded_mcp_servers가 없으면 기존 로직 사용
        logger.debug("src.utils.search_utils not available, using existing MCP server logic")
    except Exception as e:
        logger.warning(f"Embedded search failed: {e}, falling back to MCP servers")
        # 기존 로직으로 fallback
    else:
        # src/utils 성공적으로 실행됨
        return
    
    # 기존 로직 (MCP 서버 연결 시도) - src/utils 실패 시에만 실행
    """MCP 서버를 통한 검색 도구 실행 (with caching and bot detection bypass)."""
    import time
    from src.core.result_cache import get_result_cache
    
    # ToolResult는 이미 파일 상단에서 정의되어 있으므로 import 불필요
    start_time = time.time()
    query = parameters.get("query", "")
    max_results = parameters.get("max_results", 10) or parameters.get("num_results", 10)
    
    # DuckDuckGo 요청 빈도 제한 (동시 요청 방지)
    global _ddg_last_request_time
    
    # 캐시 확인
    result_cache = get_result_cache()
    cached_result = await result_cache.get(
        tool_name=tool_name,
        parameters=parameters,
        check_similarity=True
    )
    
    if cached_result:
        logger.debug(f"[MCP][_execute_search_tool] Cache hit for {tool_name}")
        # ToolResult 형식으로 변환
        return ToolResult(
            success=cached_result.get("success", False),
            data=cached_result.get("data"),
            error=cached_result.get("error"),
            execution_time=cached_result.get("execution_time", 0.0),
            confidence=cached_result.get("confidence", 0.8)
        )
    
    try:
        # 모든 검색 도구를 g-search와 동일하게 처리
        if tool_name in ["g-search", "ddg_search", "mcp_search"]:
            # mcp_config.json에 정의된 모든 MCP 서버에서 검색 시도
            mcp_hub = get_mcp_hub()
            
            # MCP 서버 연결 확인 및 재연결
            if not mcp_hub.mcp_sessions:
                logger.warning("No MCP servers connected, attempting to initialize...")
                try:
                    await mcp_hub.initialize_mcp()
                except Exception as e:
                    logger.warning(f"Failed to initialize MCP servers: {e}")
            
            # 검색 서버 목록 (github 등 실패하는 서버 제외)
            # fetch, docfork, context7-mcp, github 등은 search 도구가 없거나 실패하므로 제외
            non_search_servers = {"fetch", "docfork", "context7-mcp", "github", "financial_agent", "TodoList"}
            
            # 검색 가능한 서버만 필터링
            all_servers = list(mcp_hub.mcp_server_configs.keys())
            search_servers = [s for s in all_servers if s not in non_search_servers]
            
            # 이미 연결된 서버 우선 사용
            connected_servers = [s for s in search_servers if s in mcp_hub.mcp_sessions]
            unconnected_servers = [s for s in search_servers if s not in mcp_hub.mcp_sessions]
            server_order = connected_servers + unconnected_servers
            
            logger.info(f"[MCP][_execute_search_tool] Trying search servers: {server_order}")
            
            # MCP 서버가 없거나 모두 실패하면 DDG search로 즉시 fallback
            if not server_order:
                logger.warning("[MCP][_execute_search_tool] No MCP search servers available, using DDG search fallback")
                return await _fallback_to_ddg_search(query, max_results)
            
            # mcp_config.json에 정의된 모든 서버 확인 (우선순위 순서로)
            failed_servers = []  # 실패한 서버 추적
            for server_name in server_order:
                logger.info(f"[MCP][_execute_search_tool] 🔍 Attempting server {server_name} ({server_order.index(server_name) + 1}/{len(server_order)})")
                
                # 연결이 안 되어 있으면 연결 시도 (타임아웃 10초로 제한, 재시도 로직 포함)
                if server_name not in mcp_hub.mcp_sessions:
                    logger.info(f"MCP server {server_name} not connected, attempting connection (timeout: 10s)...")
                    server_config = mcp_hub.mcp_server_configs[server_name]
                    
                    # 재시도 로직: 타임아웃이나 일시적 에러는 재시도
                    max_connection_retries = 3
                    connection_success = False
                    
                    for retry_attempt in range(max_connection_retries):
                        try:
                            # 타임아웃 10초로 제한하여 빠르게 실패
                            success = await asyncio.wait_for(
                                mcp_hub._connect_to_mcp_server(server_name, server_config),
                                timeout=10.0
                            )
                            if success:
                                connection_success = True
                                logger.info(f"[MCP][_execute_search_tool] ✅ Successfully connected to {server_name} (attempt {retry_attempt + 1}/{max_connection_retries})")
                                break
                            else:
                                # 연결 실패 (서버가 False 반환)
                                if retry_attempt < max_connection_retries - 1:
                                    wait_time = 2 ** retry_attempt  # 지수 백오프: 1초, 2초
                                    logger.warning(f"[MCP][_execute_search_tool] ⚠️ Connection to {server_name} failed (attempt {retry_attempt + 1}/{max_connection_retries}), retrying in {wait_time}s...")
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    logger.warning(f"[MCP][_execute_search_tool] ❌ Failed to connect to MCP server {server_name} after {max_connection_retries} attempts")
                                    failed_servers.append({"server": server_name, "reason": "connection_failed"})
                                    break

                        except asyncio.TimeoutError:
                            # 타임아웃 에러는 재시도 가능
                            if retry_attempt < max_connection_retries - 1:
                                wait_time = 2 ** retry_attempt  # 지수 백오프: 1초, 2초
                                logger.warning(f"[MCP][_execute_search_tool] ⚠️ MCP server {server_name} connection timeout (10s, attempt {retry_attempt + 1}/{max_connection_retries}), retrying in {wait_time}s...")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                logger.warning(f"[MCP][_execute_search_tool] ❌ MCP server {server_name} connection timeout after {max_connection_retries} attempts, skipping...")
                                failed_servers.append({"server": server_name, "reason": "timeout"})
                                break

                        except Exception as e:
                            error_str = str(e).lower()
                            error_msg = str(e)
                            
                            # npm ENOTEMPTY 오류는 디렉토리 관련 문제로, 재시도 불필요
                            is_npm_enotempty = "enotempty" in error_str or ("npm error" in error_str and "directory not empty" in error_str)
                            
                            # Connection closed 오류는 서버 연결 실패로, 재시도 불필요
                            is_connection_closed = "connection closed" in error_str or "client failed to connect" in error_str
                            
                            # 조용히 처리할 오류들 (재시도 불필요)
                            if is_npm_enotempty or is_connection_closed:
                                logger.debug(f"[MCP][_execute_search_tool] server={server_name} connection issue, skipping")
                                failed_servers.append({"server": server_name, "reason": "connection_issue"})
                                break
                            
                            # 504, 502, 503 등 서버 에러는 재시도
                            is_retryable = any(code in error_str for code in ["504", "502", "503", "500", "gateway", "timeout", "unavailable"])

                            if is_retryable and retry_attempt < max_connection_retries - 1:
                                wait_time = 2 ** retry_attempt  # 지수 백오프: 1초, 2초
                                logger.warning(f"[MCP][_execute_search_tool] ⚠️ Error connecting to {server_name} (attempt {retry_attempt + 1}/{max_connection_retries}): {error_msg[:100]}, retrying in {wait_time}s...")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                logger.debug(f"[MCP][_execute_search_tool] Error connecting to MCP server {server_name}: {error_msg[:100]}, skipping...")
                                failed_servers.append({"server": server_name, "reason": f"connection_error: {error_msg[:100]}"})
                                break
                    
                    if not connection_success:
                        # 연결 실패 시 다음 서버 시도
                        logger.warning(f"[MCP][_execute_search_tool] Failed to connect to {server_name}, trying next server...")
                        failed_servers.append({"server": server_name, "reason": "connection_failed"})
                        continue
                
                # 도구 맵 확인
                if server_name not in mcp_hub.mcp_tools_map:
                    logger.warning(f"[MCP][_execute_search_tool] ❌ MCP server {server_name} has no tools map")
                    failed_servers.append({"server": server_name, "reason": "no_tools_map"})
                    continue
                
                try:
                    tools = mcp_hub.mcp_tools_map[server_name]
                    if not tools:
                        logger.warning(f"[MCP][_execute_search_tool] ❌ MCP server {server_name} has no tools available")
                        failed_servers.append({"server": server_name, "reason": "no_tools_available"})
                        continue
                    
                    search_tool_name = None
                    
                    # 검색 도구 찾기 (search, query, ddg, tavily, web_search 등 키워드로)
                    # 서버별 우선순위 도구 이름
                    server_specific_tools = {
                        "tavily-mcp": ["tavily-search", "search"],
                        "exa": ["web_search_exa", "search"],
                        "WebSearch-MCP": ["web_search", "search"],
                        "ddg_search": ["search", "query"],
                    }
                    
                    # 서버별 우선순위 도구 먼저 찾기
                    if server_name in server_specific_tools:
                        for preferred_tool in server_specific_tools[server_name]:
                            if preferred_tool in tools:
                                search_tool_name = preferred_tool
                                logger.info(f"Found preferred search tool '{search_tool_name}' in server {server_name}")
                                break
                    
                    # 우선순위 도구를 못 찾으면 일반 검색
                    if not search_tool_name:
                        for tool_name_key in tools.keys():
                            tool_lower = tool_name_key.lower()
                            if any(keyword in tool_lower for keyword in ["search", "query", "ddg", "tavily", "web_search"]):
                                search_tool_name = tool_name_key
                                logger.info(f"Found search tool '{search_tool_name}' in server {server_name}")
                                break
                    
                    if not search_tool_name:
                        logger.warning(f"[MCP][_execute_search_tool] ❌ No search tool found in MCP server {server_name}, available tools: {list(tools.keys())}")
                        failed_servers.append({"server": server_name, "reason": f"no_search_tool_found (available: {list(tools.keys())})"})
                        continue
                    
                    # DuckDuckGo 봇 감지 우회: Skyvern 스타일 개선 (자연스러운 요청 패턴)
                    if server_name == "ddg_search":
                        async with _get_ddg_lock():
                            current_time = time.time()
                            
                            # 요청 히스토리 초기화 (최근 10개만 유지)
                            if server_name not in mcp_hub.request_timing_history:
                                mcp_hub.request_timing_history[server_name] = []
                            history = mcp_hub.request_timing_history[server_name]
                            
                            # 오래된 히스토리 제거 (최근 1시간 이내만 유지)
                            history[:] = [t for t in history if current_time - t < 3600]
                            
                            # 마지막 요청 시간 확인
                            if "last_request" in _ddg_last_request_time:
                                time_since_last = current_time - _ddg_last_request_time["last_request"]
                                min_interval = 2.0  # 최소 2초 간격
                                
                                if time_since_last < min_interval:
                                    wait_time = min_interval - time_since_last
                                    logger.debug(f"[MCP][_execute_search_tool] Rate limiting: waiting {wait_time:.2f}s before DuckDuckGo request")
                                    await asyncio.sleep(wait_time)
                            
                            # Skyvern 스타일: 인간 행동 패턴 모방 - 가변 딜레이
                            # 히스토리가 있으면 평균 간격을 계산하여 자연스러운 변동성 추가
                            if len(history) > 0:
                                # 평균 간격 계산
                                intervals = [history[i+1] - history[i] for i in range(len(history)-1)]
                                avg_interval = sum(intervals) / len(intervals) if intervals else 3.0
                                
                                # 평균 간격을 기준으로 ±50% 변동 (최소 1.5초, 최대 5초)
                                base_delay = max(1.5, min(5.0, avg_interval * random.uniform(0.5, 1.5)))
                            else:
                                # 첫 요청: 2~4초 랜덤 딜레이
                                base_delay = random.uniform(2.0, 4.0)
                            
                            # 추가 변동성: ±0.5초 랜덤 추가 (더 자연스러운 패턴)
                            delay = base_delay + random.uniform(-0.5, 0.5)
                            delay = max(1.5, delay)  # 최소 1.5초 보장
                            
                            logger.debug(f"[MCP][_execute_search_tool] Skyvern-style delay: {delay:.2f}s before DuckDuckGo request (history: {len(history)} requests)")
                            await asyncio.sleep(delay)
                            
                            # 마지막 요청 시간 업데이트
                            _ddg_last_request_time["last_request"] = time.time()
                            # 히스토리에 추가
                            history.append(time.time())
                    
                    # 검색 실행 (재시도 로직 포함, 봇 감지 우회)
                    logger.info(f"Using MCP server {server_name} with tool {search_tool_name} for search: {query}")
                    result = None
                    max_retries = 3 if server_name == "ddg_search" else 1
                    bot_detection_indicators = ["bot detection", "no results were found", "try again"]
                    
                    for retry_attempt in range(max_retries):
                        try:
                            result = await mcp_hub._execute_via_mcp_server(
                                server_name,
                                search_tool_name,
                                {"query": query, "max_results": max_results}
                            )

                            # 결과가 없으면 재시도
                            if not result:
                                if retry_attempt < max_retries - 1:
                                    wait_time = 2 * (2 ** retry_attempt)
                                    logger.debug(f"[MCP][_execute_search_tool] No result from {server_name}, retrying after {wait_time}s")
                                    await asyncio.sleep(wait_time)
                                    continue
                                break
                            
                            # 봇 감지 메시지 확인 (DuckDuckGo만) - 즉시 확인
                            if server_name == "ddg_search" and result:
                                result_str = str(result).lower() if isinstance(result, str) else str(result).lower()
                                is_bot_detected = any(indicator in result_str for indicator in bot_detection_indicators)
                                
                                if is_bot_detected:
                                    if retry_attempt < max_retries - 1:
                                        wait_time = 3 * (2 ** retry_attempt)  # 봇 감지 시 더 긴 딜레이: 3초, 6초, 12초
                                        logger.warning(f"[MCP][_execute_search_tool] Bot detection detected from {server_name} (attempt {retry_attempt + 1}/{max_retries}), retrying after {wait_time}s")
                                        await asyncio.sleep(wait_time)
                                        result = None  # 재시도를 위해 None으로 설정
                                        continue
                                    else:
                                        logger.error(f"[MCP][_execute_search_tool] Bot detection persisted after {max_retries} attempts, skipping {server_name}")
                                        result = None  # 모든 재시도 실패
                                        break
                            
                            # 유효한 결과가 있으면 재시도 루프 종료
                            if result:
                                break
                                
                        except Exception as e:
                            logger.warning(f"[MCP][_execute_search_tool] Attempt {retry_attempt + 1}/{max_retries} failed for {server_name}: {e}")
                            if retry_attempt < max_retries - 1:
                                # 지수 백오프: 2초, 4초, 8초
                                wait_time = 2 * (2 ** retry_attempt)
                                logger.debug(f"[MCP][_execute_search_tool] Retrying {server_name} after {wait_time}s delay")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.error(f"[MCP][_execute_search_tool] All {max_retries} attempts failed for {server_name}")
                                result = None
                    
                    if not result:
                        logger.warning(f"[MCP][_execute_search_tool] ❌ MCP server {server_name} tool {search_tool_name} returned no result after {max_retries} attempts")
                        failed_servers.append({"server": server_name, "reason": "no_result_returned", "tool": search_tool_name})
                        continue
                    
                    # 결과 파싱 - 실제 외부 서버 응답 형식 처리 및 에러 체크
                    import json
                    import re
                    
                    # 에러 응답 체크 (failed, 401, 404, 502 등)
                    result_lower = str(result).lower() if result else ""
                    error_patterns = [
                        r'\b(failed|error|invalid_token|authentication failed)\b',
                        r'\b(401|404|500|502|503|504)\b',
                        r'bad gateway',
                        r'not found',
                        r'unauthorized',
                        r'<!doctype html>',  # HTML 에러 페이지
                        r'<html',
                        r'<title>.*error.*</title>'
                    ]
                    
                    is_error = False
                    error_msg = None
                    for pattern in error_patterns:
                        if re.search(pattern, result_lower):
                            is_error = True
                            if not error_msg:
                                # 에러 메시지 추출 시도
                                if '401' in result_lower or 'invalid_token' in result_lower:
                                    error_msg = "Authentication failed (401)"
                                elif '404' in result_lower:
                                    error_msg = "Not found (404)"
                                elif '502' in result_lower or 'bad gateway' in result_lower:
                                    error_msg = "Bad gateway (502) - Server temporarily unavailable"
                                elif '500' in result_lower:
                                    error_msg = "Internal server error (500)"
                                else:
                                    error_msg = "Server error detected in response"
                            break
                    
                    if is_error:
                        logger.error(f"[MCP][_execute_search_tool] ❌ MCP server {server_name} returned error response: {error_msg}")
                        failed_servers.append({"server": server_name, "reason": f"error_response: {error_msg}"})
                        continue  # 다음 서버 시도
                    
                    # result가 dict이고 'result' 키가 문자열인 경우 (tavily-mcp 등)
                    if isinstance(result, dict) and "result" in result and isinstance(result.get("result"), str):
                        result_str = result.get("result", "")
                        logger.debug(f"[MCP][_execute_search_tool] Server {server_name} returned string result (length: {len(result_str)})")
                        # 문자열 결과를 dict로 변환
                        result = result_str
                    
                    if isinstance(result, str):
                        # 텍스트 결과를 파싱 시도
                        # 1. JSON 형식 시도
                        try:
                            result_data = json.loads(result)
                        except:
                            # 2. TAVILY 형식 파싱 시도 ("Title: ... URL: ... Content: ...")
                            if "Title:" in result and "URL:" in result:
                                results = []
                                lines = result.strip().split('\n')
                                current_result = {}
                                
                                for line in lines:
                                    line = line.strip()
                                    if not line:
                                        # 빈 줄이면 현재 결과 저장하고 새로 시작
                                        if current_result and current_result.get("title"):
                                            results.append(current_result)
                                            current_result = {}
                                        continue
                                    
                                    # TAVILY 형식: "Title: ...", "URL: ...", "Content: ..."
                                    if line.startswith("Title:"):
                                        if current_result and current_result.get("title"):
                                            results.append(current_result)
                                        current_result = {"title": line[6:].strip(), "url": "", "snippet": ""}
                                    elif line.startswith("URL:"):
                                        if current_result:
                                            current_result["url"] = line[4:].strip()
                                    elif line.startswith("Content:"):
                                        if current_result:
                                            current_result["snippet"] = line[8:].strip()
                                    elif current_result:
                                        # Content 다음 줄들
                                        if current_result.get("snippet"):
                                            current_result["snippet"] += " " + line
                                        else:
                                            current_result["snippet"] = line
                                
                                # 마지막 결과 추가
                                if current_result and current_result.get("title"):
                                    results.append(current_result)
                                
                                if results:
                                    logger.debug(f"[MCP][_execute_search_tool] Parsed {len(results)} results from TAVILY format")
                                    result_data = {"results": results}
                                else:
                                    # TAVILY 파싱 실패, 마크다운 형식 시도
                                    results = []
                                    current_result = None
                                    
                                    for line in lines:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        
                                        # 마크다운 링크 패턴: [Title](url)
                                        link_match = re.match(r'^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)', line)
                                        if link_match:
                                            if current_result:
                                                results.append(current_result)
                                            title = link_match.group(1)
                                            url = link_match.group(2)
                                            current_result = {
                                                "title": title,
                                                "url": url,
                                                "snippet": ""
                                            }
                                        elif current_result and line:
                                            if current_result["snippet"]:
                                                current_result["snippet"] += " " + line
                                            else:
                                                current_result["snippet"] = line
                                    
                                    if current_result:
                                        results.append(current_result)
                                    
                                    if results:
                                        result_data = {"results": results}
                                    else:
                                        logger.debug(f"[MCP][_execute_search_tool] Could not parse result format, using raw text: {result[:100]}")
                                        result_data = {"results": [{"title": "Search Results", "snippet": result[:500], "url": ""}]}
                            else:
                                # 3. 마크다운 형식 텍스트 파싱 (ddg_search 등이 반환하는 형식)
                                # 예: "1. [Title](url)\n   Description..."
                                results = []
                                lines = result.strip().split('\n')
                            current_result = None
                            
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue
                                
                                # 마크다운 링크 패턴: [Title](url)
                                link_match = re.match(r'^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)', line)
                                if link_match:
                                    # 이전 결과 저장
                                    if current_result:
                                        results.append(current_result)
                                    
                                    title = link_match.group(1)
                                    url = link_match.group(2)
                                    current_result = {
                                        "title": title,
                                        "url": url,
                                        "snippet": ""
                                    }
                                elif current_result and line:
                                    # 설명 텍스트
                                    if current_result["snippet"]:
                                        current_result["snippet"] += " " + line
                                    else:
                                        current_result["snippet"] = line
                            
                            # 마지막 결과 추가
                            if current_result:
                                results.append(current_result)
                            
                            if results:
                                result_data = {"results": results}
                            else:
                                # 파싱 실패 시 원본 텍스트를 snippet으로 사용
                                    logger.debug(f"[MCP][_execute_search_tool] Could not parse markdown format, using raw text: {result[:100]}")
                                    result_data = {"results": [{"title": "Search Results", "snippet": result[:500], "url": ""}]}
                    else:
                        result_data = result
                    
                    # 결과 형식 정규화
                    results = result_data.get("results", [])
                    if not results and isinstance(result_data, dict):
                        # 다른 형식 시도
                        results = result_data.get("items", result_data.get("data", []))
                    
                    if results:
                        # 결과 내용 검증: 봇 감지나 에러 메시지가 포함된 결과 필터링
                        valid_results = []
                        invalid_indicators = [
                            "no results were found", "bot detection",
                            "no results", "not found", "try again",
                            "unable to", "error occurred", "no matches"
                        ]
                        
                        for result_item in (results if isinstance(results, list) else [results]):
                            if isinstance(result_item, dict):
                                snippet = result_item.get("snippet", result_item.get("content", result_item.get("description", "")))
                                title = result_item.get("title", result_item.get("name", ""))
                                
                                snippet_lower = str(snippet).lower() if snippet else ""
                                title_lower = str(title).lower() if title else ""
                                
                                # 에러 메시지가 포함된 결과 필터링
                                is_invalid = False
                                matched_indicators = []
                                
                                for indicator in invalid_indicators:
                                    if indicator in snippet_lower:
                                        is_invalid = True
                                        matched_indicators.append(indicator)
                                    elif indicator in title_lower:
                                        is_invalid = True
                                        matched_indicators.append(indicator)
                                
                                # "Search Results" 제목 + 빈 내용 또는 에러 메시지인 경우
                                if "search results" in title_lower and (not snippet or is_invalid):
                                    is_invalid = True
                                
                                if is_invalid:
                                    logger.warning(f"[MCP][_execute_search_tool] Filtering invalid result from {server_name}: matched indicators: {', '.join(matched_indicators)}")
                                    continue
                                
                                valid_results.append(result_item)
                            elif isinstance(result_item, str):
                                # 문자열 결과도 검증
                                result_lower = result_item.lower()
                                is_invalid = any(indicator in result_lower for indicator in invalid_indicators)
                                
                                if is_invalid:
                                    logger.warning(f"[MCP][_execute_search_tool] Filtering invalid string result from {server_name}: contains error message")
                                    continue
                                
                                # 문자열 결과를 dict 형식으로 변환
                                valid_results.append({
                                    "title": "Search Result",
                                    "snippet": result_item,
                                    "url": ""
                                })
                        
                        # 유효한 결과가 있는지 확인
                        if not valid_results:
                            original_count = len(results) if isinstance(results, list) else 1
                            logger.warning(f"[MCP][_execute_search_tool] ❌ All {original_count} results from {server_name} were filtered out (bot detection or error messages), trying next server...")
                            failed_servers.append({"server": server_name, "reason": f"all_results_filtered ({original_count} results filtered)"})
                            continue  # 다음 서버 시도
                        
                        original_count = len(results) if isinstance(results, list) else 1
                        filtered_count = original_count - len(valid_results)
                        logger.info(f"✅ Search successful via MCP server {server_name}: {len(valid_results)} valid results (filtered {filtered_count} invalid results)")
                        tool_result = ToolResult(
                            success=True,
                            data={
                                "query": query,
                                "results": valid_results,
                                "total_results": len(valid_results),
                                "source": f"{server_name}-mcp"
                            },
                            execution_time=time.time() - start_time,
                            confidence=0.9
                        )
                        
                        # 캐시에 저장 (TTL: 1시간)
                        cache_dict = {
                            "success": tool_result.success,
                            "data": tool_result.data,
                            "error": tool_result.error,
                            "execution_time": tool_result.execution_time,
                            "confidence": tool_result.confidence
                        }
                        await result_cache.set(
                            tool_name=tool_name,
                            parameters=parameters,
                            value=cache_dict,
                            ttl=3600  # 1 hour for search results
                        )
                        logger.debug(f"[MCP][_execute_search_tool] Cached result for {tool_name}")
                        
                        return tool_result
                    else:
                        logger.warning(f"[MCP][_execute_search_tool] ❌ MCP server {server_name} returned empty results")
                        failed_servers.append({"server": server_name, "reason": "empty_results"})
                        continue
                    
                except Exception as mcp_error:
                    error_str = str(mcp_error)
                    # ToolResult 관련 오류는 명확히 처리
                    if "ToolResult" in error_str or "cannot access local variable" in error_str:
                        logger.error(f"[MCP][_execute_search_tool] ❌ MCP 서버 {server_name} 검색 실패 (코드 오류): {mcp_error}")
                        failed_servers.append({"server": server_name, "reason": f"code_error: {str(mcp_error)[:100]}"})
                        # 다음 서버로 계속 진행
                        continue
                    else:
                        logger.warning(f"[MCP][_execute_search_tool] ❌ MCP 서버 {server_name} 검색 실패: {mcp_error}, 다음 서버 시도")
                        failed_servers.append({"server": server_name, "reason": f"exception: {str(mcp_error)[:100]}"})
                        import traceback
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        continue
            
            # 모든 MCP 서버 실패 시 duckduckgo_search 라이브러리 fallback 사용
            logger.warning(f"[MCP][_execute_search_tool] ⚠️ All {len(server_order)} MCP search servers failed for query: '{query}'")
            logger.info(f"[MCP][_execute_search_tool] 📋 Failed servers summary:")
            for i, failed in enumerate(failed_servers, 1):
                logger.info(f"[MCP][_execute_search_tool]   {i}. {failed['server']}: {failed['reason']}")
            
            # 모든 MCP 서버 실패 시 DDG search로 fallback
            logger.warning(f"[MCP][_execute_search_tool] 🔄 Falling back to DDG search...")
            return await _fallback_to_ddg_search(query, max_results)
        
        elif tool_name == "tavily":
            # MCP 서버를 통해 tavily 사용 (mcp_config.json에 정의된 서버)
            mcp_hub = get_mcp_hub()
            
            # 모든 연결된 MCP 서버에서 tavily 도구 찾아서 시도
            for server_name in mcp_hub.mcp_sessions.keys():
                if server_name not in mcp_hub.mcp_tools_map:
                    continue
                
                try:
                    tools = mcp_hub.mcp_tools_map[server_name]
                    tavily_tool_name = None
                    
                    # tavily 도구 찾기
                    for tool_name_key in tools.keys():
                        tool_lower = tool_name_key.lower()
                        if "tavily" in tool_lower:
                            tavily_tool_name = tool_name_key
                            break
                    
                    if tavily_tool_name:
                        logger.info(f"Using MCP server {server_name} with tool {tavily_tool_name}")
                        result = await mcp_hub._execute_via_mcp_server(
                            server_name,
                            tavily_tool_name,
                            {"query": query, "max_results": max_results}
                        )
                        
                        if result:
                            import json
                            import re
                            
                            # 에러 응답 체크
                            result_lower = str(result).lower() if result else ""
                            error_patterns = [
                                r'\b(failed|error|invalid_token|authentication failed)\b',
                                r'\b(401|404|500|502|503|504)\b',
                                r'bad gateway',
                                r'not found',
                                r'unauthorized',
                                r'<!doctype html>',
                                r'<html',
                                r'<title>.*error.*</title>'
                            ]
                            
                            is_error = False
                            for pattern in error_patterns:
                                if re.search(pattern, result_lower):
                                    is_error = True
                                    logger.warning(f"Error detected in tavily response, skipping")
                                    break
                            
                            if is_error:
                                continue  # 다음 서버 시도
                            
                            if isinstance(result, str):
                                try:
                                    result_data = json.loads(result)
                                except:
                                    # 마크다운 형식 파싱
                                    results = []
                                    lines = result.strip().split('\n')
                                    current_result = None
                                    
                                    for line in lines:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        
                                        link_match = re.match(r'^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)', line)
                                        if link_match:
                                            if current_result:
                                                results.append(current_result)
                                            title = link_match.group(1)
                                            url = link_match.group(2)
                                            current_result = {"title": title, "url": url, "snippet": ""}
                                        elif current_result and line:
                                            if current_result["snippet"]:
                                                current_result["snippet"] += " " + line
                                            else:
                                                current_result["snippet"] = line
                                    
                                    if current_result:
                                        results.append(current_result)
                                    
                                    if results:
                                        result_data = {"results": results}
                                    else:
                                        result_data = {"results": [{"title": "Search Results", "snippet": result, "url": ""}]}
                            else:
                                result_data = result
                            
                            results = result_data.get("results", [])
                            if not results and isinstance(result_data, dict):
                                results = result_data.get("items", result_data.get("data", []))
                            
                            if results:
                                tool_result = ToolResult(
                                    success=True,
                                    data={
                                        "query": query,
                                        "results": results if isinstance(results, list) else [results],
                                        "total_results": len(results) if isinstance(results, list) else 1,
                                        "source": f"{server_name}-mcp"
                                    },
                                    execution_time=time.time() - start_time,
                                    confidence=0.85
                                )
                                
                                # 캐시에 저장 (TTL: 1시간)
                                cache_dict = {
                                    "success": tool_result.success,
                                    "data": tool_result.data,
                                    "error": tool_result.error,
                                    "execution_time": tool_result.execution_time,
                                    "confidence": tool_result.confidence
                                }
                                await result_cache.set(
                                    tool_name=tool_name,
                                    parameters=parameters,
                                    value=cache_dict,
                                    ttl=3600  # 1 hour for search results
                                )
                                logger.debug(f"[MCP][_execute_search_tool] Cached result for {tool_name}")
                                
                                return tool_result
                    
                except Exception as mcp_error:
                    logger.warning(f"MCP 서버 {server_name} tavily 실패: {mcp_error}, 다음 서버 시도")
                    continue
            
            # MCP 서버에 tavily가 없으면 에러 (fallback 제거)
            raise ValueError("Tavily MCP server not found. Add tavily server to mcp_config.json")
        
        elif tool_name == "exa":
            # MCP 서버를 통해 exa 사용 (mcp_config.json에 정의된 서버)
            mcp_hub = get_mcp_hub()
            
            # 모든 연결된 MCP 서버에서 exa 도구 찾아서 시도
            for server_name in mcp_hub.mcp_sessions.keys():
                if server_name not in mcp_hub.mcp_tools_map:
                    continue
                
                try:
                    tools = mcp_hub.mcp_tools_map[server_name]
                    exa_tool_name = None
                    
                    # exa 도구 찾기
                    for tool_name_key in tools.keys():
                        tool_lower = tool_name_key.lower()
                        if "exa" in tool_lower:
                            exa_tool_name = tool_name_key
                            break
                    
                    if exa_tool_name:
                        logger.info(f"Using MCP server {server_name} with tool {exa_tool_name}")
                        result = await mcp_hub._execute_via_mcp_server(
                            server_name,
                            exa_tool_name,
                            {"query": query, "numResults": max_results}
                        )
                        
                        if result:
                            import json
                            import re
                            
                            # 에러 응답 체크
                            result_lower = str(result).lower() if result else ""
                            error_patterns = [
                                r'\b(failed|error|invalid_token|authentication failed)\b',
                                r'\b(401|404|500|502|503|504)\b',
                                r'bad gateway',
                                r'not found',
                                r'unauthorized',
                                r'<!doctype html>',
                                r'<html',
                                r'<title>.*error.*</title>'
                            ]
                            
                            is_error = False
                            for pattern in error_patterns:
                                if re.search(pattern, result_lower):
                                    is_error = True
                                    logger.warning(f"Error detected in tavily response, skipping")
                                    break
                            
                            if is_error:
                                continue  # 다음 서버 시도
                            
                            if isinstance(result, str):
                                try:
                                    result_data = json.loads(result)
                                except:
                                    # 마크다운 형식 파싱
                                    results = []
                                    lines = result.strip().split('\n')
                                    current_result = None
                                    
                                    for line in lines:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        
                                        link_match = re.match(r'^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)', line)
                                        if link_match:
                                            if current_result:
                                                results.append(current_result)
                                            title = link_match.group(1)
                                            url = link_match.group(2)
                                            current_result = {"title": title, "url": url, "snippet": ""}
                                        elif current_result and line:
                                            if current_result["snippet"]:
                                                current_result["snippet"] += " " + line
                                            else:
                                                current_result["snippet"] = line
                                    
                                    if current_result:
                                        results.append(current_result)
                                    
                                    if results:
                                        result_data = {"results": results}
                                    else:
                                        result_data = {"results": [{"title": "Search Results", "snippet": result, "url": ""}]}
                            else:
                                result_data = result
                            
                            results = result_data.get("results", [])
                            if not results and isinstance(result_data, dict):
                                results = result_data.get("items", result_data.get("data", []))
                            
                            if results:
                                tool_result = ToolResult(
                                    success=True,
                                    data={
                                        "query": query,
                                        "results": results if isinstance(results, list) else [results],
                                        "total_results": len(results) if isinstance(results, list) else 1,
                                        "source": f"{server_name}-mcp"
                                    },
                                    execution_time=time.time() - start_time,
                                    confidence=0.85
                                )
                                
                                # 캐시에 저장 (TTL: 1시간)
                                cache_dict = {
                                    "success": tool_result.success,
                                    "data": tool_result.data,
                                    "error": tool_result.error,
                                    "execution_time": tool_result.execution_time,
                                    "confidence": tool_result.confidence
                                }
                                await result_cache.set(
                                    tool_name=tool_name,
                                    parameters=parameters,
                                    value=cache_dict,
                                    ttl=3600  # 1 hour for search results
                                )
                                logger.debug(f"[MCP][_execute_search_tool] Cached result for {tool_name}")
                                
                                return tool_result
                    
                except Exception as mcp_error:
                    logger.warning(f"MCP 서버 {server_name} exa 실패: {mcp_error}, 다음 서버 시도")
                    continue
            
            # MCP 서버에 exa가 없으면 에러 (fallback 제거)
            raise ValueError("Exa MCP server not found. Add exa server to mcp_config.json")
        
        else:
            raise ValueError(f"Unknown search tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"Search tool execution failed: {tool_name} - {e}")
        return ToolResult(
            success=False,
            data=None,
            error=f"Search tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def _execute_academic_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """학술 도구 실행 - src/utils에서 직접 사용."""
    import time
    
    start_time = time.time()
    query = parameters.get("query", "")
    max_results = parameters.get("max_results", 10) or parameters.get("num_results", 10)
    
    # src/utils에서 직접 사용 (MCP 서버로 실행하지 않음)
    if tool_name == "arxiv":
        try:
            from src.utils.academic_utils import search_arxiv
            
            if not query:
                return ToolResult(
                    success=False,
                    data=None,
                    error="query parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0
                )
            
            # src/utils의 search_arxiv 직접 호출
            result = await search_arxiv(query, max_results)
            
            if result.get("success"):
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "results": result.get("results", []),
                        "total_results": result.get("total_results", 0),
                        "count": result.get("count", 0),
                        "source": "embedded_arxiv"
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.95
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=result.get("error", "arXiv search failed"),
                    execution_time=time.time() - start_time,
                    confidence=0.0
                )
        except ImportError:
            logger.debug("src.utils.academic_utils not available, using existing logic")
        except Exception as e:
            logger.warning(f"Embedded arXiv search failed: {e}, falling back to existing logic")
    
    # 기존 로직 (src/utils 실패 시 또는 다른 tool_name)
    try:
        if tool_name == "arxiv":
            # arXiv API (100% 무료)
            import arxiv
            
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in client.results(search):
                results.append({
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "url": paper.entry_id,
                    "published": paper.published.isoformat(),
                    "pdf_url": paper.pdf_url
                })
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                    "source": "arxiv"
                },
                execution_time=time.time() - start_time,
                confidence=0.95
            )
        
        elif tool_name == "scholar":
            # Google Scholar (무료, rate limit 있음)
            from scholarly import scholarly
            
            search_query = scholarly.search_pubs(query)
            results = []
            
            for i, pub in enumerate(search_query):
                if i >= max_results:
                    break
                    
                results.append({
                    "title": pub.get("bib", {}).get("title", ""),
                    "authors": pub.get("bib", {}).get("author", ""),
                    "abstract": pub.get("bib", {}).get("abstract", ""),
                    "url": pub.get("pub_url", ""),
                    "year": pub.get("bib", {}).get("pub_year", ""),
                    "citations": pub.get("num_citations", 0)
                })
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                    "source": "scholar"
                },
                execution_time=time.time() - start_time,
                confidence=0.8
            )
        
        else:
            raise ValueError(f"Unknown academic tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"Academic tool execution failed: {tool_name} - {e}")
        return ToolResult(
            success=False,
            data=None,
            error=f"Academic tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def _execute_data_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """실제 데이터 도구 실행."""
    import time
    
    start_time = time.time()
    
    try:
        if tool_name == "fetch":
            # src/utils에서 직접 사용
            try:
                from src.utils.web_utils import fetch_url
                
                url = parameters.get("url", "")
                max_length = parameters.get("max_length", 50000)
                timeout = parameters.get("timeout", 30)
                
                if not url:
                    raise ValueError("URL parameter is required for fetch tool")
                
                # src/utils의 fetch_url 직접 호출
                result = await fetch_url(url, max_length, timeout)
                
                if result.get("success"):
                    return ToolResult(
                        success=True,
                        data={
                            "url": url,
                            "content": result.get("content", ""),
                            "content_type": result.get("content_type", "unknown"),
                            "status_code": result.get("status_code", 200),
                            "character_count": result.get("character_count", 0),
                            "source": "embedded_fetch"
                        },
                        execution_time=time.time() - start_time,
                        confidence=0.9
                    )
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=result.get("error", "Fetch failed"),
                        execution_time=time.time() - start_time,
                        confidence=0.0
                    )
            except ImportError:
                logger.debug("src.utils.web_utils not available, using existing logic")
            except Exception as e:
                logger.warning(f"Embedded fetch failed: {e}, falling back to existing logic")
            
            # 기존 로직 (fallback)
            url = parameters.get("url", "")
            if not url:
                raise ValueError("URL parameter is required for fetch tool")
            
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                return ToolResult(
                    success=True,
                    data={
                        "url": url,
                        "status": response.status_code,
                        "content": response.text[:10000],  # 처음 10000자만
                        "content_length": len(response.text),
                        "headers": dict(response.headers)
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.9
                )
        
        elif tool_name == "filesystem":
            # 파일시스템 접근 (실제 구현)
            await _execute_file_tool(tool_name, parameters)

        elif tool_name == "browser":
            # 브라우저 자동화 (실제 구현)
            await _execute_browser_tool(tool_name, parameters)

        elif tool_name == "shell":
            # 쉘 명령 실행 (실제 구현)
            await _execute_shell_tool(tool_name, parameters)
        
        else:
            raise ValueError(f"Unknown data tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"Data tool execution failed: {tool_name} - {e}")
        return ToolResult(
            success=False,
            data=None,
            error=f"Data tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def _execute_code_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """실제 코드 도구 실행 - ERA 또는 Docker 샌드박스 선택."""
    import time

    start_time = time.time()
    code = parameters.get("code", "")
    language = parameters.get("language", "python")
    sandbox_type = parameters.get("sandbox", "era")  # "era" or "docker"
    
    # 1. 리소스 제한 체크
    try:
        from src.core.resource_limits import ResourceLimits, CodeSizeLimitError

        code_bytes = code.encode('utf-8')
        if ResourceLimits.exceeds_code_limit(len(code_bytes)):
            error_msg = (
                f"Code size ({ResourceLimits.format_bytes(len(code_bytes))}) exceeds limit "
                f"({ResourceLimits.MAX_CODE_SIZE_HUMAN}). "
                f"Please reduce the code size or split into smaller chunks."
            )
            logger.error(error_msg)
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
                execution_time=time.time() - start_time,
                confidence=0.0
            )
    except ImportError:
        # ResourceLimits 모듈이 없으면 경고만 하고 계속 진행
        logger.debug("ResourceLimits module not available, skipping size check")

    # 2. 샌드박스 타입에 따른 실행
    if sandbox_type == "docker":
        # Docker 샌드박스 사용
        try:
            from src.core.sandbox.docker_sandbox import get_sandbox
            sandbox = get_sandbox()

            result = await sandbox.execute_code(code, language)
            execution_time = time.time() - start_time

            return ToolResult(
                success=result.success,
                data={
                    "code": code,
                    "language": language,
                    "output": result.output,
                    "error": result.error,
                    "exit_code": result.exit_code,
                    "sandbox_type": "docker",
                    "container_id": result.container_id
                },
                execution_time=execution_time,
                confidence=0.9 if result.success else 0.5
            )

        except Exception as e:
            logger.error(f"Docker sandbox execution failed: {e}")
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                data=None,
                error=f"Docker sandbox failed: {str(e)}",
                execution_time=execution_time,
                confidence=0.0
            )

    # 3. ERA 설정 확인 (기본값)
    try:
        from src.core.researcher_config import get_era_config
        from src.core.era_client import ERAClient
        from src.core.era_server_manager import get_era_server_manager
        
        era_config = get_era_config()
        
        # ERA가 비활성화되어 있으면 에러 반환
        if not era_config.enabled:
            error_msg = "ERA is disabled. Code execution requires ERA to be enabled for security."
            logger.error(error_msg)
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
                execution_time=time.time() - start_time,
                confidence=0.0
            )
        
        # 싱글톤 ERA 서버 관리자 사용
        server_manager = get_era_server_manager()
        
        # 강화된 서버 시작 (재시도 포함)
        if not await server_manager.ensure_server_running_with_retry():
            error_msg = (
                f"ERA server is not available after {server_manager.max_retries} retries. "
                f"Please ensure ERA Agent is installed and running. "
                f"Binary path: {server_manager.agent_binary_path or 'not found'}"
            )
            logger.error(error_msg)
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
                execution_time=time.time() - start_time,
                confidence=0.0
            )
        
        # ERA 클라이언트 생성
        era_client = ERAClient(
            base_url=era_config.server_url,
            api_key=era_config.api_key,
            timeout=float(era_config.default_timeout) + 10.0  # 여유 시간 추가
        )
        
        try:
            # 언어별 명령 생성
            # Cloudflare Worker 방식: 코드를 base64로 인코딩하여 command에 포함
            # 이 방식이 가장 안전하고 확실함 (이스케이프 문제 없음)
            import base64
            import uuid
            
            # 코드를 base64로 인코딩
            code_bytes = code.encode('utf-8')
            code_base64 = base64.b64encode(code_bytes).decode('ascii')
            
            # 언어별 실행 명령 생성
            unique_id = uuid.uuid4().hex[:8]
            
            if language.lower() in ["python", "py"]:
                # Python: base64 디코딩 후 파이프로 전달
                # 긴 코드나 멀티라인 코드는 임시 파일 사용
                if len(code) > 1000 or code.count('\n') > 10:
                    # 긴 코드는 임시 파일로 실행
                    tmp_file = f"/tmp/code_{unique_id}.py"
                    # base64 문자열을 안전하게 전달 (single quote 사용)
                    command = f'sh -c "echo \'{code_base64}\' | base64 -d > {tmp_file} && python {tmp_file} && rm -f {tmp_file}"'
                else:
                    # 짧은 코드는 파이프로 실행
                    command = f'sh -c "echo \'{code_base64}\' | base64 -d | python"'
                
                result = await era_client.run_temp(
                    language="python",
                    command=command,
                    cpu=era_config.default_cpu,
                    memory=era_config.default_memory,
                    network=era_config.network_mode,
                    timeout=era_config.default_timeout
                )
            elif language.lower() in ["javascript", "js", "node", "nodejs"]:
                # JavaScript/Node: base64 디코딩 후 파이프로 전달
                if len(code) > 1000 or code.count('\n') > 10:
                    # 긴 코드는 임시 파일로 실행
                    tmp_file = f"/tmp/code_{unique_id}.js"
                    command = f'sh -c "echo \'{code_base64}\' | base64 -d > {tmp_file} && node {tmp_file} && rm -f {tmp_file}"'
                else:
                    # 짧은 코드는 파이프로 실행
                    command = f'sh -c "echo \'{code_base64}\' | base64 -d | node"'
                
                result = await era_client.run_temp(
                    language="javascript",
                    command=command,
                    cpu=era_config.default_cpu,
                    memory=era_config.default_memory,
                    network=era_config.network_mode,
                    timeout=era_config.default_timeout
                )
            else:
                error_msg = f"Unsupported language for ERA: {language}"
                logger.error(error_msg)
                return ToolResult(
                    success=False,
                    data=None,
                    error=error_msg,
                    execution_time=time.time() - start_time,
                    confidence=0.0
                )
            
            # ERA 실행 결과를 ToolResult로 변환
            execution_time = time.time() - start_time
            
            return ToolResult(
                success=result.exit_code == 0,
                    data={
                        "code": code,
                        "language": language,
                        "output": result.stdout,
                        "error": result.stderr,
                        "return_code": result.exit_code,
                        "vm_id": result.vm_id,
                        "duration": result.duration,
                        "sandbox_type": "era"
                    },
                execution_time=execution_time,
                confidence=0.9 if result.exit_code == 0 else 0.5
                )
        finally:
            await era_client.close()
            
    except ImportError as e:
        # ERA 모듈이 없으면 에러 반환
        error_msg = f"ERA modules not available: {e}. Please install ERA Agent dependencies."
        logger.error(error_msg)
        return ToolResult(
            success=False,
            data=None,
            error=error_msg,
            execution_time=time.time() - start_time,
            confidence=0.0
        )
    except ConnectionError as e:
        # ERA 연결 실패 - 에러 반환
        error_msg = f"ERA connection failed: {e}. Please ensure ERA server is running."
        logger.error(error_msg)
        return ToolResult(
            success=False,
            data=None,
            error=error_msg,
            execution_time=time.time() - start_time,
            confidence=0.0
        )
    except ValueError as e:
        # ERA 설정 오류 - 에러 반환
        error_msg = f"ERA configuration error: {e}"
        logger.error(error_msg)
        return ToolResult(
            success=False,
            data=None,
            error=error_msg,
            execution_time=time.time() - start_time,
            confidence=0.0
        )
    except Exception as e:
        # 기타 ERA 오류 - 에러 반환
        error_msg = f"ERA execution error: {e}"
        logger.error(error_msg, exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=error_msg,
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def _execute_browser_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """브라우저 자동화 도구 실행."""
    import time
    start_time = time.time()
    
    try:
        from src.automation.browser_manager import BrowserManager
        
        # BrowserManager 인스턴스 생성 (싱글톤 패턴 고려)
        browser_manager = BrowserManager()
        
        # 브라우저 초기화 (아직 안 되어 있으면)
        if not browser_manager.browser_available:
            await browser_manager.initialize_browser()
        
        if tool_name == "browser_navigate":
            # URL로 이동 및 콘텐츠 추출
            url = parameters.get("url", "")
            extraction_goal = parameters.get("extraction_goal", "extract_all_content")
            
            if not url:
                raise ValueError("URL parameter is required for browser_navigate")
            
            result = await browser_manager.navigate_and_extract(url, extraction_goal)
            
            return ToolResult(
                success=result.get("success", False),
                data=result,
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        elif tool_name == "browser_extract":
            # 특정 목표에 맞는 콘텐츠 추출
            url = parameters.get("url", "")
            extraction_goal = parameters.get("extraction_goal", "extract_all_content")
            
            if not url:
                raise ValueError("URL parameter is required for browser_extract")
            
            result = await browser_manager.navigate_and_extract(url, extraction_goal)
            
            return ToolResult(
                success=result.get("success", False),
                data=result,
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        elif tool_name == "browser_screenshot":
            # 스크린샷 캡처
            url = parameters.get("url", "")
            output_path = parameters.get("output_path", None)
            
            if not url:
                raise ValueError("URL parameter is required for browser_screenshot")
            
            # Playwright를 사용한 스크린샷
            try:
                from playwright.async_api import async_playwright
                PLAYWRIGHT_AVAILABLE = True
            except ImportError:
                PLAYWRIGHT_AVAILABLE = False
            
            if PLAYWRIGHT_AVAILABLE:
                from playwright.async_api import async_playwright
                
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()
                    await page.goto(url, wait_until="networkidle")
                    
                    if output_path:
                        await page.screenshot(path=output_path, full_page=True)
                    else:
                        # 임시 파일에 저장
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            output_path = tmp.name
                            await page.screenshot(path=output_path, full_page=True)
                    
                    await browser.close()
                    
                    return ToolResult(
                        success=True,
                        data={"screenshot_path": output_path, "url": url},
                        execution_time=time.time() - start_time,
                        confidence=0.9
                    )
            else:
                raise RuntimeError("Playwright not available for screenshot")
        
        elif tool_name == "browser_interact":
            # 버튼 클릭, 폼 작성 등 상호작용
            url = parameters.get("url", "")
            actions = parameters.get("actions", [])  # List of action dicts
            
            if not url:
                raise ValueError("URL parameter is required for browser_interact")
            
            if not actions:
                raise ValueError("actions parameter is required for browser_interact")
            
            # Playwright를 사용한 상호작용
            try:
                from playwright.async_api import async_playwright
                PLAYWRIGHT_AVAILABLE = True
            except ImportError:
                PLAYWRIGHT_AVAILABLE = False
            
            if PLAYWRIGHT_AVAILABLE:
                from playwright.async_api import async_playwright
                
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()
                    await page.goto(url, wait_until="networkidle")
                    
                    results = []
                    for action in actions:
                        action_type = action.get("type")
                        selector = action.get("selector")
                        value = action.get("value")
                        
                        try:
                            if action_type == "click":
                                await page.click(selector)
                                results.append({"type": "click", "selector": selector, "success": True})
                            elif action_type == "fill":
                                await page.fill(selector, value)
                                results.append({"type": "fill", "selector": selector, "success": True})
                            elif action_type == "select":
                                await page.select_option(selector, value)
                                results.append({"type": "select", "selector": selector, "success": True})
                            elif action_type == "wait":
                                await page.wait_for_selector(selector, timeout=5000)
                                results.append({"type": "wait", "selector": selector, "success": True})
                            else:
                                results.append({"type": action_type, "success": False, "error": "Unknown action type"})
                        except Exception as e:
                            results.append({"type": action_type, "success": False, "error": str(e)})
                    
                    # 최종 페이지 콘텐츠 추출
                    final_content = await page.content()
                    
                    await browser.close()
                    
                    return ToolResult(
                        success=all(r.get("success", False) for r in results),
                        data={
                            "url": url,
                            "actions": results,
                            "final_content": final_content[:10000]  # 처음 10000자만
                        },
                        execution_time=time.time() - start_time,
                        confidence=0.8 if all(r.get("success", False) for r in results) else 0.5
                    )
            else:
                raise RuntimeError("Playwright not available for browser interaction")
        
        else:
            raise ValueError(f"Unknown browser tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"Browser tool execution failed: {tool_name} - {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=f"Browser tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def _execute_document_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """문서 생성 도구 실행."""
    import time
    start_time = time.time()
    
    try:
        from src.generation.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        research_data = parameters.get("research_data", {})
        report_type = parameters.get("report_type", "comprehensive")
        
        if not research_data:
            raise ValueError("research_data parameter is required for document generation")
        
        # 도구 이름에서 형식 추출
        if tool_name == "generate_pdf":
            output_format = "pdf"
        elif tool_name == "generate_docx":
            output_format = "docx"
        elif tool_name == "generate_pptx":
            output_format = "pptx"
        elif tool_name == "generate_html":
            output_format = "html"
        elif tool_name == "generate_markdown":
            output_format = "markdown"
        else:
            raise ValueError(f"Unknown document tool: {tool_name}")
        
        # 문서 생성
        file_path = await generator.generate_research_report(
            research_data=research_data,
            report_type=report_type,
            output_format=output_format
        )
        
        return ToolResult(
            success=True,
            data={
                "file_path": file_path,
                "format": output_format,
                "report_type": report_type
            },
            execution_time=time.time() - start_time,
            confidence=0.9
        )
        
    except Exception as e:
        logger.error(f"Document tool execution failed: {tool_name} - {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=f"Document tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def _execute_git_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """Git 워크플로우 도구 실행."""
    import time
    start_time = time.time()
    
    try:
        from src.core.git_workflow import GitWorkflow
        from pathlib import Path
        
        # 저장소 경로 확인
        repo_path = parameters.get("repo_path")
        if repo_path:
            repo_path = Path(repo_path)
        else:
            repo_path = None
        
        # GitWorkflow 생성
        git_workflow = GitWorkflow(repo_path=repo_path)
        
        if tool_name == "git_status":
            result = await git_workflow.git_status()
            return ToolResult(
                success=True,
                data=result,
                execution_time=time.time() - start_time,
                confidence=0.9
            )
        
        elif tool_name == "git_commit":
            message = parameters.get("message")
            auto_stage = parameters.get("auto_stage", True)
            result = await git_workflow.git_commit(message=message, auto_stage=auto_stage)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        elif tool_name == "git_push":
            branch = parameters.get("branch")
            force = parameters.get("force", False)
            result = await git_workflow.git_push(branch=branch, force=force)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        elif tool_name == "git_create_pr":
            title = parameters.get("title")
            body = parameters.get("body")
            base = parameters.get("base", "main")
            
            if not title:
                return ToolResult(
                    success=False,
                    data=None,
                    error="title parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0
                )
            
            result = await git_workflow.git_create_pr(title=title, body=body, base=base)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        elif tool_name == "git_commit_push_pr":
            commit_message = parameters.get("commit_message")
            pr_title = parameters.get("pr_title")
            pr_body = parameters.get("pr_body")
            base = parameters.get("base", "main")
            
            result = await git_workflow.git_commit_push_pr(
                commit_message=commit_message,
                pr_title=pr_title,
                pr_body=pr_body,
                base=base
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        else:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown git tool: {tool_name}",
                execution_time=time.time() - start_time,
                confidence=0.0
            )
    
    except Exception as e:
        logger.error(f"Git tool execution failed: {tool_name} - {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=f"Git tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def _execute_shell_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """Shell 명령 실행 도구 (완전 자동형 SparkleForge)."""
    import time
    start_time = time.time()
    
    try:
        from src.core.shell_executor import ShellExecutor
        from pathlib import Path
        
        # 작업 디렉토리 확인
        working_dir = parameters.get("working_dir")
        if working_dir:
            working_dir = Path(working_dir)
        else:
            working_dir = None
        
        # ShellExecutor 생성
        executor = ShellExecutor(
            require_confirmation=parameters.get("require_confirmation", False),
            max_execution_time=parameters.get("timeout", 300)
        )
        
        if tool_name == "run_shell_command":
            command = parameters.get("command")
            if not command:
                return ToolResult(
                    success=False,
                    data=None,
                    error="command parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0
                )
            
            confirm = parameters.get("confirm")
            timeout = parameters.get("timeout")
            result = await executor.run(
                command=command,
                working_dir=working_dir,
                confirm=confirm,
                timeout=timeout
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        elif tool_name == "run_interactive_command":
            command = parameters.get("command")
            if not command:
                return ToolResult(
                    success=False,
                    data=None,
                    error="command parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0
                )
            
            input_data = parameters.get("input")
            result = await executor.run_interactive(
                command=command,
                working_dir=working_dir,
                input_data=input_data
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        elif tool_name == "run_background_command":
            command = parameters.get("command")
            if not command:
                return ToolResult(
                    success=False,
                    data=None,
                    error="command parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0
                )
            
            result = await executor.run_background(
                command=command,
                working_dir=working_dir
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        else:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown shell tool: {tool_name}",
                execution_time=time.time() - start_time,
                confidence=0.0
            )
    
    except Exception as e:
        logger.error(f"Shell tool execution failed: {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=str(e),
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def _execute_git_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """Git 워크플로우 도구 실행 (완전 자동형 SparkleForge)."""
    import time
    start_time = time.time()
    
    try:
        from src.core.git_workflow import GitWorkflow
        from pathlib import Path
        
        # 저장소 경로 확인
        repo_path = parameters.get("repo_path")
        if repo_path:
            repo_path = Path(repo_path)
        else:
            repo_path = None  # 현재 디렉토리 사용
        
        git_workflow = GitWorkflow(repo_path=repo_path)
        
        if tool_name == "git_status":
            result = await git_workflow.git_status()
            return ToolResult(
                success=True,
                data=result,
                execution_time=time.time() - start_time,
                confidence=0.9
            )
        
        elif tool_name == "git_commit":
            message = parameters.get("message")
            auto_stage = parameters.get("auto_stage", True)
            result = await git_workflow.git_commit(message=message, auto_stage=auto_stage)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        elif tool_name == "git_push":
            branch = parameters.get("branch")
            force = parameters.get("force", False)
            result = await git_workflow.git_push(branch=branch, force=force)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        elif tool_name == "git_create_pr":
            title = parameters.get("title")
            body = parameters.get("body")
            base = parameters.get("base", "main")
            result = await git_workflow.git_create_pr(title=title, body=body, base=base)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        elif tool_name == "git_commit_push_pr":
            commit_message = parameters.get("commit_message")
            pr_title = parameters.get("pr_title")
            pr_body = parameters.get("pr_body")
            base = parameters.get("base", "main")
            result = await git_workflow.git_commit_push_pr(
                commit_message=commit_message,
                pr_title=pr_title,
                pr_body=pr_body,
                base=base
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0
            )
        
        else:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown git tool: {tool_name}",
                execution_time=time.time() - start_time,
                confidence=0.0
            )
    
    except Exception as e:
        logger.error(f"Git tool execution failed: {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=str(e),
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def _execute_file_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """파일 작업 도구 실행."""
    import time
    start_time = time.time()
    
    try:
        from pathlib import Path
        import os
        
        # 안전성 검증: 작업 디렉토리 제한
        allowed_dirs = [
            Path.cwd(),  # 현재 작업 디렉토리
            Path("./outputs"),  # 출력 디렉토리
            Path("./workspace"),  # 워크스페이스
            Path("./temp"),  # 임시 디렉토리
        ]
        
        def _is_safe_path(file_path: str) -> bool:
            """경로 안전성 검증."""
            try:
                path = Path(file_path).resolve()
                # 상대 경로만 허용
                if path.is_absolute() and not any(path.is_relative_to(allowed) for allowed in allowed_dirs):
                    # 절대 경로인 경우 허용된 디렉토리 내에 있는지 확인
                    for allowed in allowed_dirs:
                        try:
                            path.relative_to(allowed.resolve())
                            return True
                        except ValueError:
                            continue
                    return False
                # 상대 경로는 허용
                return True
            except Exception:
                return False
        
        if tool_name == "create_file":
            file_path = parameters.get("file_path", "")
            content = parameters.get("content", "")
            
            if not file_path:
                raise ValueError("file_path parameter is required")
            if not _is_safe_path(file_path):
                raise ValueError(f"Unsafe file path: {file_path}")
            
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            
            return ToolResult(
                success=True,
                data={"file_path": str(path), "size": len(content)},
                execution_time=time.time() - start_time,
                confidence=0.9
            )
        
        elif tool_name == "read_file":
            file_path = parameters.get("file_path", "")
            
            if not file_path:
                raise ValueError("file_path parameter is required")
            if not _is_safe_path(file_path):
                raise ValueError(f"Unsafe file path: {file_path}")
            
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            content = path.read_text(encoding='utf-8')
            
            return ToolResult(
                success=True,
                data={"file_path": str(path), "content": content, "size": len(content)},
                execution_time=time.time() - start_time,
                confidence=0.9
            )
        
        elif tool_name == "write_file":
            file_path = parameters.get("file_path", "")
            content = parameters.get("content", "")
            
            if not file_path:
                raise ValueError("file_path parameter is required")
            if not _is_safe_path(file_path):
                raise ValueError(f"Unsafe file path: {file_path}")
            
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            
            return ToolResult(
                success=True,
                data={"file_path": str(path), "size": len(content)},
                execution_time=time.time() - start_time,
                confidence=0.9
            )
        
        elif tool_name == "edit_file":
            file_path = parameters.get("file_path", "")
            old_string = parameters.get("old_string", "")
            new_string = parameters.get("new_string", "")
            
            if not file_path:
                raise ValueError("file_path parameter is required")
            if not _is_safe_path(file_path):
                raise ValueError(f"Unsafe file path: {file_path}")
            
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            content = path.read_text(encoding='utf-8')
            if old_string not in content:
                raise ValueError(f"Old string not found in file: {file_path}")
            
            new_content = content.replace(old_string, new_string)
            path.write_text(new_content, encoding='utf-8')
            
            return ToolResult(
                success=True,
                data={"file_path": str(path), "replacements": content.count(old_string)},
                execution_time=time.time() - start_time,
                confidence=0.9
            )
        
        elif tool_name == "list_files":
            directory_path = parameters.get("directory_path", ".")
            recursive = parameters.get("recursive", False)
            
            if not _is_safe_path(directory_path):
                raise ValueError(f"Unsafe directory path: {directory_path}")
            
            path = Path(directory_path)
            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            if not path.is_dir():
                raise ValueError(f"Path is not a directory: {directory_path}")
            
            files = []
            if recursive:
                for item in path.rglob("*"):
                    files.append({
                        "name": item.name,
                        "path": str(item.relative_to(path)),
                        "is_file": item.is_file(),
                        "size": item.stat().st_size if item.is_file() else 0
                    })
            else:
                for item in path.iterdir():
                    files.append({
                        "name": item.name,
                        "path": item.name,
                        "is_file": item.is_file(),
                        "size": item.stat().st_size if item.is_file() else 0
                    })
            
            return ToolResult(
                success=True,
                data={"directory": str(path), "files": files, "count": len(files)},
                execution_time=time.time() - start_time,
                confidence=0.9
            )
        
        elif tool_name == "delete_file":
            file_path = parameters.get("file_path", "")
            
            if not file_path:
                raise ValueError("file_path parameter is required")
            if not _is_safe_path(file_path):
                raise ValueError(f"Unsafe file path: {file_path}")
            
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File or directory not found: {file_path}")
            
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                import shutil
                shutil.rmtree(path)
            
            return ToolResult(
                success=True,
                data={"file_path": str(path), "deleted": True},
                execution_time=time.time() - start_time,
                confidence=0.9
            )
        
        else:
            raise ValueError(f"Unknown file tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"File tool execution failed: {tool_name} - {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=f"File tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0
        )


async def get_tool_for_category(category: ToolCategory) -> Optional[str]:
    """카테고리에 해당하는 도구 반환."""
    mcp_hub = get_mcp_hub()
    return mcp_hub.get_tool_for_category(category)


async def health_check() -> Dict[str, Any]:
    """헬스 체크."""
    mcp_hub = get_mcp_hub()
    return await mcp_hub.health_check()


# CLI 실행 함수들
async def run_mcp_hub():
    """MCP Hub 실행 (CLI)."""
    mcp_hub = get_mcp_hub()
    print("🚀 Starting Universal MCP Hub...")
    try:
        await mcp_hub.initialize_mcp()
        print("✅ MCP Hub started successfully")
        print(f"Available tools: {len(mcp_hub.tools)}")
        
        # Hub 유지
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n✅ MCP Hub stopped")
    except Exception as e:
        print(f"❌ MCP Hub failed to start: {e}")
        await mcp_hub.cleanup()
        sys.exit(1)


async def list_tools():
    """도구 목록 출력 (CLI)."""
    print("🔧 Available MCP Tools:")
    available_tools = await get_available_tools()
    for tool_name in available_tools:
        print(f"  - {tool_name}")

async def check_mcp_servers():
    """MCP 서버 상태 확인 (CLI)."""
    mcp_hub = get_mcp_hub()
    try:
        # 초기화 (이미 초기화되어 있으면 재초기화하지 않음)
        if not mcp_hub.mcp_sessions:
            logger.info("Initializing MCP Hub to check servers...")
            await mcp_hub.initialize_mcp()
        
        server_status = await mcp_hub.check_mcp_servers()
        
        print("\n" + "=" * 80)
        print("📊 MCP 서버 연결 상태 확인")
        print("=" * 80)
        print(f"전체 서버 수: {server_status['total_servers']}")
        print(f"연결된 서버: {server_status['connected_servers']}")
        print(f"연결률: {server_status['summary']['connection_rate']}")
        print(f"전체 사용 가능한 Tool 수: {server_status['summary']['total_tools_available']}")
        print("\n")
        
        for server_name, info in server_status["servers"].items():
            status_icon = "✅" if info["connected"] else "❌"
            print(f"{status_icon} 서버: {server_name}")
            print(f"   타입: {info['type']}")
            
            if info["type"] == "http":
                print(f"   URL: {info.get('url', 'unknown')}")
            else:
                cmd = info.get('command', 'unknown')
                args_preview = ' '.join(info.get('args', [])[:3])
                print(f"   명령어: {cmd} {args_preview}...")
            
            print(f"   연결 상태: {'연결됨' if info['connected'] else '연결 안 됨'}")
            print(f"   제공 Tool 수: {info['tools_count']}")
            
            if info["tools"]:
                print(f"   Tool 목록:")
                for tool in info["tools"][:5]:  # 처음 5개만 표시
                    registered_name = f"{server_name}::{tool}"
                    print(f"     - {registered_name}")
                if len(info["tools"]) > 5:
                    print(f"     ... 및 {len(info['tools']) - 5}개 더")
            
            if info.get("error"):
                print(f"   ⚠️ 오류: {info['error']}")
            print()
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 서버 상태 확인 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 정리하지 않고 세션 유지 (다른 작업에서 사용 가능)
        pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal MCP Hub - MCP Only")
    parser.add_argument("--start", action="store_true", help="Start MCP Hub")
    parser.add_argument("--list-tools", action="store_true", help="List available tools")
    parser.add_argument("--health", action="store_true", help="Show health status")
    parser.add_argument("--check-servers", action="store_true", help="Check all MCP server connections")
    
    args = parser.parse_args()
    
    if args.start:
        asyncio.run(run_mcp_hub())
    elif args.list_tools:
        asyncio.run(list_tools())
    elif args.check_servers:
        asyncio.run(check_mcp_servers())
    elif args.health:
        async def show_health():
            mcp_hub = get_mcp_hub()
            try:
                await mcp_hub.initialize_mcp()
                health = await health_check()
                print("🏥 Health Status:")
                for key, value in health.items():
                    print(f"  {key}: {value}")
                await mcp_hub.cleanup()
            except Exception as e:
                print(f"❌ Health check failed: {e}")
        asyncio.run(show_health())
    else:
        parser.print_help()
"""Tool-execution dispatch layer for the MCP integration hub.

Extracted from the monolithic ``src/core/mcp_integration.py`` (issue #508,
Anvil Phase Sigma-1): ``execute_tool``/``get_mcp_hub`` and the per-category
``_execute_*_tool`` dispatchers that route a tool call to the right MCP
server, native adapter, or fallback.

``UniversalMCPHub`` (``src.core.mcp_integration.hub``) needs several of
these dispatchers as bare module-level references (see ``hub.py``'s
imports), so this module intentionally does *not* import from ``hub`` at
module load time -- only inside ``get_mcp_hub()``, deferred until first
call, to avoid a circular import.
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
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from src.core.mcp_integration.hub import UniversalMCPHub

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



from src.core.mcp_integration.parser import (
    _parse_json_text,
    _parse_markdown_link_results,
)
from src.core.tools.registry import ToolCategory, ToolInfo, ToolResult

# Global MCP Hub instance (lazy initialization)
_mcp_hub = None

def get_mcp_hub() -> "UniversalMCPHub":
    """Get or initialize global MCP Hub."""
    global _mcp_hub
    if _mcp_hub is None:
        try:
            get_mcp_config()
        except RuntimeError as e:
            if "Configuration not loaded" not in str(e):
                raise
            from src.core.researcher_config import load_config_from_env

            load_config_from_env()
        # Deferred import: hub.py imports several dispatchers from this module
        # at load time, so importing UniversalMCPHub here (rather than at this
        # module's top level) avoids a circular import.
        from src.core.mcp_integration.hub import UniversalMCPHub

        _mcp_hub = UniversalMCPHub()
    return _mcp_hub


async def get_available_tools() -> List[str]:
    """사용 가능한 도구 목록 반환."""
    mcp_hub = get_mcp_hub()
    return mcp_hub.get_available_tools()


async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """MCP 도구 실행 - UniversalMCPHub의 execute_tool 사용 (with caching)."""
    from src.core.result_cache import get_result_cache

    mcp_hub = get_mcp_hub()

    # Startup-time trust filtering is enforced here as a defensive runtime check.
    try:
        from src.core.trust_gate import get_current_trust_context

        trust = get_current_trust_context()
        tool_info = mcp_hub.registry.get_tool_info(tool_name)
        mcp_server = tool_info.mcp_server if tool_info else None
        if not trust.allows_tool(tool_name, mcp_server):
            return {
                "success": False,
                "error": f"Tool execution denied by TrustGate: {tool_name}",
                "data": None,
            }
    except Exception as trust_err:
        logger.debug("TrustGate check skipped: %s", trust_err)

    # Lifecycle hooks: PreToolUse (can block execution with exit code 2)
    try:
        from src.core.skills_manager import get_skill_manager

        hook_runner = get_skill_manager().get_hook_runner()
        if hook_runner:
            session_id = getattr(get_skill_manager(), "_current_session_id", "") or ""
            allowed = await hook_runner.run_pre_tool_use(tool_name, parameters, session_id)
            if not allowed:
                return {
                    "success": False,
                    "error": "Tool execution blocked by PreToolUse hook",
                    "data": None,
                }
    except Exception as hook_err:
        logger.debug("PreToolUse hook skipped: %s", hook_err)

    result_cache = get_result_cache()

    # 캐시 확인
    cached_result = await result_cache.get(
        tool_name=tool_name, parameters=parameters, check_similarity=True
    )

    if cached_result:
        logger.debug(f"[MCP][execute_tool] Cache hit for {tool_name}")
        return cached_result

    # MCP Hub 실행
    # MCP Hub가 초기화되지 않았으면 초기화
    if not mcp_hub.mcp_sessions:
        logger.info("[MCP][execute_tool] MCP Hub not initialized, initializing...")
        await mcp_hub.initialize_mcp()

    # SSE tool visualization: emit tool_use before execution
    try:
        from src.core.streaming_manager import get_streaming_manager

        sm = get_streaming_manager()
        session_id = getattr(get_skill_manager(), "_current_session_id", "") or "default"
        await sm.stream_tool_use(session_id, tool_name, parameters)
    except Exception as viz_err:
        logger.debug("Tool visualization stream_tool_use skipped: %s", viz_err)

    with start_tool_span(
        name=f"tool:{tool_name}",
        tool_name=tool_name,
        input={"tool_name": tool_name, "parameters_keys": list(parameters.keys())},
    ):
        result = await mcp_hub.execute_tool(tool_name, parameters)

    # SSE tool visualization: emit tool_result after execution
    try:
        from src.core.streaming_manager import get_streaming_manager

        sm = get_streaming_manager()
        session_id = getattr(get_skill_manager(), "_current_session_id", "") or "default"
        summary = ""
        if result.get("success") and result.get("data"):
            d = result["data"]
            summary = str(d)[:500] if not isinstance(d, str) else d[:500]
        else:
            summary = result.get("error", "failed") or "failed"
        await sm.stream_tool_result(session_id, tool_name, result.get("success", False), summary)
    except Exception as viz_err:
        logger.debug("Tool visualization stream_tool_result skipped: %s", viz_err)

    # Lifecycle hooks: PostToolUse
    try:
        from src.core.skills_manager import get_skill_manager

        hook_runner = get_skill_manager().get_hook_runner()
        if hook_runner:
            session_id = getattr(get_skill_manager(), "_current_session_id", "") or ""
            await hook_runner.run_post_tool_use(tool_name, parameters, result, session_id)
    except Exception as hook_err:
        logger.debug("PostToolUse hook skipped: %s", hook_err)

    # Tool Design: format=concise 이면 응답 크기 제한 (Response Format Optimization)
    if (
        result.get("success")
        and parameters.get("format") == "concise"
        and result.get("data") is not None
    ):
        data = result["data"]
        max_concise_chars = 1500
        if isinstance(data, str) and len(data) > max_concise_chars:
            result = {
                **result,
                "data": data[:max_concise_chars] + "\n...[truncated (format=concise)]",
            }
        elif isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
            result = {
                **result,
                "data": {
                    **data,
                    "results": data["results"][:5],
                    "_truncated": "format=concise: first 5 results only",
                },
            }
        elif isinstance(data, dict) and "content" in data:
            c = data["content"]
            if isinstance(c, str) and len(c) > max_concise_chars:
                result = {
                    **result,
                    "data": {**data, "content": c[:max_concise_chars] + "\n...[truncated]"},
                }

    # Filesystem Context: 대용량 출력 시 Scratch Pad로 오프로드 (Agent-Skills-for-Context-Engineering)
    if result.get("success", False) and os.getenv("ENABLE_SCRATCH_PAD", "true").lower() == "true":
        try:
            from src.core.scratch_pad import (
                build_result_with_scratch_ref,
                write_tool_output,
            )

            threshold = int(os.getenv("SCRATCH_PAD_THRESHOLD_CHARS", "8000"))
            scratch_path, summary = write_tool_output(tool_name, result, threshold_chars=threshold)
            if scratch_path:
                result = build_result_with_scratch_ref(result, scratch_path, summary)
        except Exception as e:
            logger.debug("Scratch pad offload skipped: %s", e)

    # 성공한 결과만 캐시에 저장
    if result.get("success", False):
        # TTL 결정: 검색/데이터 도구는 1시간, 다른 도구는 30분
        ttl = (
            3600
            if any(keyword in tool_name.lower() for keyword in ["search", "fetch", "data"])
            else 1800
        )
        await result_cache.set(tool_name=tool_name, parameters=parameters, value=result, ttl=ttl)
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
                    future = executor.submit(
                        asyncio.run, _execute_search_tool(tool_name, parameters)
                    )
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
                    future = executor.submit(
                        asyncio.run, _execute_academic_tool(tool_name, parameters)
                    )
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
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "[MCP][fallback] DDG returned non-JSON string, wrapping as plain text result"
                )
                return ToolResult(
                    success=True,
                    data={"results": [{"content": result}], "total_results": 1},
                    tool_name="ddg_search",
                    source="native_ddg_fallback",
                )

        if result and isinstance(result, dict):
            results = result.get("results", [])
            if results:
                return ToolResult(
                    success=True,
                    data={"results": results, "total_results": len(results)},
                    tool_name="ddg_search",
                    source="native_ddg_fallback",
                )
        elif result and isinstance(result, list):
            return ToolResult(
                success=True,
                data={"results": result, "total_results": len(result)},
                tool_name="ddg_search",
                source="native_ddg_fallback",
            )

        # 결과가 없거나 형식이 잘못된 경우
        return ToolResult(
            success=False,
            data=None,
            error="DDG search fallback returned no results",
            tool_name="ddg_search",
            source="native_ddg_fallback",
        )
    except Exception as e:
        logger.error(f"[MCP][fallback] DDG search fallback failed: {e}")
        return ToolResult(
            success=False,
            data=None,
            error=f"DDG search fallback error: {str(e)}",
            tool_name="ddg_search",
            source="native_ddg_fallback",
        )


async def _execute_search_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """검색 도구 실행 - src/utils에서 직접 사용."""
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
                confidence=0.0,
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
                    "source": "embedded_search",
                },
                execution_time=time.time() - start_time,
                confidence=0.9,
            )
        else:
            return ToolResult(
                success=False,
                data=None,
                error=result.get("error", "Search failed"),
                execution_time=time.time() - start_time,
                confidence=0.0,
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
    from src.core.result_cache import get_result_cache

    # ToolResult는 이미 파일 상단에서 정의되어 있으므로 import 불필요
    start_time = time.time()
    query = parameters.get("query", "")
    max_results = parameters.get("max_results", 10) or parameters.get("num_results", 10)

    # API 키 없으면 tavily/exa 스킵 (에러 대신 빈 결과 반환)
    if tool_name == "tavily" and not os.getenv("TAVILY_API_KEY"):
        logger.debug("TAVILY_API_KEY not set, skipping tavily")
        return ToolResult(
            success=True,
            data={"results": [], "query": query, "count": 0},
            error=None,
            execution_time=time.time() - start_time,
            confidence=0.0,
        )
    if tool_name == "exa" and not os.getenv("EXA_API_KEY"):
        logger.debug("EXA_API_KEY not set, skipping exa")
        return ToolResult(
            success=True,
            data={"results": [], "query": query, "count": 0},
            error=None,
            execution_time=time.time() - start_time,
            confidence=0.0,
        )

    # DuckDuckGo 요청 빈도 제한 (동시 요청 방지)
    global _ddg_last_request_time

    # 캐시 확인
    result_cache = get_result_cache()
    cached_result = await result_cache.get(
        tool_name=tool_name, parameters=parameters, check_similarity=True
    )

    if cached_result:
        logger.debug(f"[MCP][_execute_search_tool] Cache hit for {tool_name}")
        # ToolResult 형식으로 변환
        return ToolResult(
            success=cached_result.get("success", False),
            data=cached_result.get("data"),
            error=cached_result.get("error"),
            execution_time=cached_result.get("execution_time", 0.0),
            confidence=cached_result.get("confidence", 0.8),
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
            non_search_servers = {
                "fetch",
                "docfork",
                "context7-mcp",
                "github",
                "financial_agent",
                "TodoList",
            }

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
                logger.warning(
                    "[MCP][_execute_search_tool] No MCP search servers available, using DDG search fallback"
                )
                return await _fallback_to_ddg_search(query, max_results)

            # mcp_config.json에 정의된 모든 서버 확인 (우선순위 순서로)
            failed_servers = []  # 실패한 서버 추적
            for server_name in server_order:
                logger.info(
                    f"[MCP][_execute_search_tool] 🔍 Attempting server {server_name} ({server_order.index(server_name) + 1}/{len(server_order)})"
                )

                # 연결이 안 되어 있으면 연결 시도 (타임아웃 10초로 제한, 재시도 로직 포함)
                if server_name not in mcp_hub.mcp_sessions:
                    logger.info(
                        f"MCP server {server_name} not connected, attempting connection (timeout: 10s)..."
                    )
                    server_config = mcp_hub.mcp_server_configs[server_name]

                    # 재시도 로직: 타임아웃이나 일시적 에러는 재시도
                    max_connection_retries = 3
                    connection_success = False

                    for retry_attempt in range(max_connection_retries):
                        try:
                            # 타임아웃 10초로 제한하여 빠르게 실패
                            success = await asyncio.wait_for(
                                mcp_hub._connect_to_mcp_server(server_name, server_config),
                                timeout=10.0,
                            )
                            if success:
                                connection_success = True
                                logger.info(
                                    f"[MCP][_execute_search_tool] ✅ Successfully connected to {server_name} (attempt {retry_attempt + 1}/{max_connection_retries})"
                                )
                                break
                            else:
                                # 연결 실패 (서버가 False 반환)
                                if retry_attempt < max_connection_retries - 1:
                                    wait_time = 2**retry_attempt  # 지수 백오프: 1초, 2초
                                    logger.warning(
                                        f"[MCP][_execute_search_tool] ⚠️ Connection to {server_name} failed (attempt {retry_attempt + 1}/{max_connection_retries}), retrying in {wait_time}s..."
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    logger.warning(
                                        f"[MCP][_execute_search_tool] ❌ Failed to connect to MCP server {server_name} after {max_connection_retries} attempts"
                                    )
                                    failed_servers.append(
                                        {
                                            "server": server_name,
                                            "reason": "connection_failed",
                                        }
                                    )
                                    break

                        except TimeoutError:
                            # 타임아웃 에러는 재시도 가능
                            if retry_attempt < max_connection_retries - 1:
                                wait_time = 2**retry_attempt  # 지수 백오프: 1초, 2초
                                logger.warning(
                                    f"[MCP][_execute_search_tool] ⚠️ MCP server {server_name} connection timeout (10s, attempt {retry_attempt + 1}/{max_connection_retries}), retrying in {wait_time}s..."
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                logger.warning(
                                    f"[MCP][_execute_search_tool] ❌ MCP server {server_name} connection timeout after {max_connection_retries} attempts, skipping..."
                                )
                                failed_servers.append({"server": server_name, "reason": "timeout"})
                                break

                        except Exception as e:
                            error_str = str(e).lower()
                            error_msg = str(e)

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
                            if is_npm_enotempty or is_connection_closed:
                                logger.debug(
                                    f"[MCP][_execute_search_tool] server={server_name} connection issue, skipping"
                                )
                                failed_servers.append(
                                    {
                                        "server": server_name,
                                        "reason": "connection_issue",
                                    }
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
                                    f"[MCP][_execute_search_tool] ⚠️ Error connecting to {server_name} (attempt {retry_attempt + 1}/{max_connection_retries}): {error_msg[:100]}, retrying in {wait_time}s..."
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                logger.debug(
                                    f"[MCP][_execute_search_tool] Error connecting to MCP server {server_name}: {error_msg[:100]}, skipping..."
                                )
                                failed_servers.append(
                                    {
                                        "server": server_name,
                                        "reason": f"connection_error: {error_msg[:100]}",
                                    }
                                )
                                break

                    if not connection_success:
                        # 연결 실패 시 다음 서버 시도
                        logger.warning(
                            f"[MCP][_execute_search_tool] Failed to connect to {server_name}, trying next server..."
                        )
                        failed_servers.append(
                            {"server": server_name, "reason": "connection_failed"}
                        )
                        continue

                # 도구 맵 확인
                if server_name not in mcp_hub.mcp_tools_map:
                    logger.warning(
                        f"[MCP][_execute_search_tool] ❌ MCP server {server_name} has no tools map"
                    )
                    failed_servers.append({"server": server_name, "reason": "no_tools_map"})
                    continue

                try:
                    tools = mcp_hub.mcp_tools_map[server_name]
                    if not tools:
                        logger.warning(
                            f"[MCP][_execute_search_tool] ❌ MCP server {server_name} has no tools available"
                        )
                        failed_servers.append(
                            {"server": server_name, "reason": "no_tools_available"}
                        )
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
                                logger.info(
                                    f"Found preferred search tool '{search_tool_name}' in server {server_name}"
                                )
                                break

                    # 우선순위 도구를 못 찾으면 일반 검색
                    if not search_tool_name:
                        for tool_name_key in tools.keys():
                            tool_lower = tool_name_key.lower()
                            if any(
                                keyword in tool_lower
                                for keyword in [
                                    "search",
                                    "query",
                                    "ddg",
                                    "tavily",
                                    "web_search",
                                ]
                            ):
                                search_tool_name = tool_name_key
                                logger.info(
                                    f"Found search tool '{search_tool_name}' in server {server_name}"
                                )
                                break

                    if not search_tool_name:
                        logger.warning(
                            f"[MCP][_execute_search_tool] ❌ No search tool found in MCP server {server_name}, available tools: {list(tools.keys())}"
                        )
                        failed_servers.append(
                            {
                                "server": server_name,
                                "reason": f"no_search_tool_found (available: {list(tools.keys())})",
                            }
                        )
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
                                time_since_last = (
                                    current_time - _ddg_last_request_time["last_request"]
                                )
                                min_interval = 2.0  # 최소 2초 간격

                                if time_since_last < min_interval:
                                    wait_time = min_interval - time_since_last
                                    logger.debug(
                                        f"[MCP][_execute_search_tool] Rate limiting: waiting {wait_time:.2f}s before DuckDuckGo request"
                                    )
                                    await asyncio.sleep(wait_time)

                            # Skyvern 스타일: 인간 행동 패턴 모방 - 가변 딜레이
                            # 히스토리가 있으면 평균 간격을 계산하여 자연스러운 변동성 추가
                            if len(history) > 0:
                                # 평균 간격 계산
                                intervals = [
                                    history[i + 1] - history[i] for i in range(len(history) - 1)
                                ]
                                avg_interval = sum(intervals) / len(intervals) if intervals else 3.0

                                # 평균 간격을 기준으로 ±50% 변동 (최소 1.5초, 최대 5초)
                                base_delay = max(
                                    1.5,
                                    min(5.0, avg_interval * random.uniform(0.5, 1.5)),
                                )
                            else:
                                # 첫 요청: 2~4초 랜덤 딜레이
                                base_delay = random.uniform(2.0, 4.0)

                            # 추가 변동성: ±0.5초 랜덤 추가 (더 자연스러운 패턴)
                            delay = base_delay + random.uniform(-0.5, 0.5)
                            delay = max(1.5, delay)  # 최소 1.5초 보장

                            logger.debug(
                                f"[MCP][_execute_search_tool] Skyvern-style delay: {delay:.2f}s before DuckDuckGo request (history: {len(history)} requests)"
                            )
                            await asyncio.sleep(delay)

                            # 마지막 요청 시간 업데이트
                            _ddg_last_request_time["last_request"] = time.time()
                            # 히스토리에 추가
                            history.append(time.time())

                    # 검색 실행 (재시도 로직 포함, 봇 감지 우회)
                    logger.info(
                        f"Using MCP server {server_name} with tool {search_tool_name} for search: {query}"
                    )
                    result = None
                    max_retries = 3 if server_name == "ddg_search" else 1
                    bot_detection_indicators = [
                        "bot detection",
                        "no results were found",
                        "try again",
                    ]

                    for retry_attempt in range(max_retries):
                        try:
                            result = await mcp_hub._execute_via_mcp_server(
                                server_name,
                                search_tool_name,
                                {"query": query, "max_results": max_results},
                            )

                            # 결과가 없으면 재시도
                            if not result:
                                if retry_attempt < max_retries - 1:
                                    wait_time = 2 * (2**retry_attempt)
                                    logger.debug(
                                        f"[MCP][_execute_search_tool] No result from {server_name}, retrying after {wait_time}s"
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue
                                break

                            # 봇 감지 메시지 확인 (DuckDuckGo만) - 즉시 확인
                            if server_name == "ddg_search" and result:
                                result_str = (
                                    str(result).lower()
                                    if isinstance(result, str)
                                    else str(result).lower()
                                )
                                is_bot_detected = any(
                                    indicator in result_str
                                    for indicator in bot_detection_indicators
                                )

                                if is_bot_detected:
                                    if retry_attempt < max_retries - 1:
                                        wait_time = 3 * (
                                            2**retry_attempt
                                        )  # 봇 감지 시 더 긴 딜레이: 3초, 6초, 12초
                                        logger.warning(
                                            f"[MCP][_execute_search_tool] Bot detection detected from {server_name} (attempt {retry_attempt + 1}/{max_retries}), retrying after {wait_time}s"
                                        )
                                        await asyncio.sleep(wait_time)
                                        result = None  # 재시도를 위해 None으로 설정
                                        continue
                                    else:
                                        logger.error(
                                            f"[MCP][_execute_search_tool] Bot detection persisted after {max_retries} attempts, skipping {server_name}"
                                        )
                                        result = None  # 모든 재시도 실패
                                        break

                            # 유효한 결과가 있으면 재시도 루프 종료
                            if result:
                                break

                        except Exception as e:
                            logger.warning(
                                f"[MCP][_execute_search_tool] Attempt {retry_attempt + 1}/{max_retries} failed for {server_name}: {e}"
                            )
                            if retry_attempt < max_retries - 1:
                                # 지수 백오프: 2초, 4초, 8초
                                wait_time = 2 * (2**retry_attempt)
                                logger.debug(
                                    f"[MCP][_execute_search_tool] Retrying {server_name} after {wait_time}s delay"
                                )
                                await asyncio.sleep(wait_time)
                            else:
                                logger.error(
                                    f"[MCP][_execute_search_tool] All {max_retries} attempts failed for {server_name}"
                                )
                                result = None

                    if not result:
                        logger.warning(
                            f"[MCP][_execute_search_tool] ❌ MCP server {server_name} tool {search_tool_name} returned no result after {max_retries} attempts"
                        )
                        failed_servers.append(
                            {
                                "server": server_name,
                                "reason": "no_result_returned",
                                "tool": search_tool_name,
                            }
                        )
                        continue

                    # 결과 파싱 - 실제 외부 서버 응답 형식 처리 및 에러 체크
                    import json
                    import re

                    # 에러 응답 체크 (failed, 401, 404, 502 등)
                    result_lower = str(result).lower() if result else ""
                    error_patterns = [
                        r"\b(failed|error|invalid_token|authentication failed)\b",
                        r"\b(401|404|500|502|503|504)\b",
                        r"bad gateway",
                        r"not found",
                        r"unauthorized",
                        r"<!doctype html>",  # HTML 에러 페이지
                        r"<html",
                        r"<title>.*error.*</title>",
                    ]

                    is_error = False
                    error_msg = None
                    for pattern in error_patterns:
                        if re.search(pattern, result_lower):
                            is_error = True
                            if not error_msg:
                                # 에러 메시지 추출 시도
                                if "401" in result_lower or "invalid_token" in result_lower:
                                    error_msg = "Authentication failed (401)"
                                elif "404" in result_lower:
                                    error_msg = "Not found (404)"
                                elif "502" in result_lower or "bad gateway" in result_lower:
                                    error_msg = "Bad gateway (502) - Server temporarily unavailable"
                                elif "500" in result_lower:
                                    error_msg = "Internal server error (500)"
                                else:
                                    error_msg = "Server error detected in response"
                            break

                    if is_error:
                        logger.error(
                            f"[MCP][_execute_search_tool] ❌ MCP server {server_name} returned error response: {error_msg}"
                        )
                        failed_servers.append(
                            {
                                "server": server_name,
                                "reason": f"error_response: {error_msg}",
                            }
                        )
                        continue  # 다음 서버 시도

                    # result가 dict이고 'result' 키가 문자열인 경우 (tavily-mcp 등)
                    if (
                        isinstance(result, dict)
                        and "result" in result
                        and isinstance(result.get("result"), str)
                    ):
                        result_str = result.get("result", "")
                        logger.debug(
                            f"[MCP][_execute_search_tool] Server {server_name} returned string result (length: {len(result_str)})"
                        )
                        # 문자열 결과를 dict로 변환
                        result = result_str

                    if isinstance(result, str):
                        # 텍스트 결과를 파싱 시도
                        # 1. JSON 형식 시도
                        parsed_json, parsed_data = _parse_json_text(
                            result, context=f"MCP search result from {server_name}"
                        )
                        if parsed_json:
                            result_data = parsed_data
                        else:
                            # 2. TAVILY 형식 파싱 시도 ("Title: ... URL: ... Content: ...")
                            if "Title:" in result and "URL:" in result:
                                results = []
                                lines = result.strip().split("\n")
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
                                        current_result = {
                                            "title": line[6:].strip(),
                                            "url": "",
                                            "snippet": "",
                                        }
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
                                    logger.debug(
                                        f"[MCP][_execute_search_tool] Parsed {len(results)} results from TAVILY format"
                                    )
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
                                        link_match = re.match(
                                            r"^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)", line
                                        )
                                        if link_match:
                                            if current_result:
                                                results.append(current_result)
                                            title = link_match.group(1)
                                            url = link_match.group(2)
                                            current_result = {
                                                "title": title,
                                                "url": url,
                                                "snippet": "",
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
                                        logger.debug(
                                            f"[MCP][_execute_search_tool] Could not parse result format, using raw text: {result[:100]}"
                                        )
                                        result_data = {
                                            "results": [
                                                {
                                                    "title": "Search Results",
                                                    "snippet": result[:500],
                                                    "url": "",
                                                }
                                            ]
                                        }
                            else:
                                # 3. 마크다운 형식 텍스트 파싱 (ddg_search 등이 반환하는 형식)
                                # 예: "1. [Title](url)\n   Description..."
                                results = []
                                lines = result.strip().split("\n")
                            current_result = None

                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue

                                # 마크다운 링크 패턴: [Title](url)
                                link_match = re.match(r"^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)", line)
                                if link_match:
                                    # 이전 결과 저장
                                    if current_result:
                                        results.append(current_result)

                                    title = link_match.group(1)
                                    url = link_match.group(2)
                                    current_result = {
                                        "title": title,
                                        "url": url,
                                        "snippet": "",
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
                                logger.debug(
                                    f"[MCP][_execute_search_tool] Could not parse markdown format, using raw text: {result[:100]}"
                                )
                                result_data = {
                                    "results": [
                                        {
                                            "title": "Search Results",
                                            "snippet": result[:500],
                                            "url": "",
                                        }
                                    ]
                                }
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
                            "no results were found",
                            "bot detection",
                            "no results",
                            "not found",
                            "try again",
                            "unable to",
                            "error occurred",
                            "no matches",
                        ]

                        for result_item in (results if isinstance(results, list) else [results]):
                            if isinstance(result_item, dict):
                                snippet = result_item.get(
                                    "snippet",
                                    result_item.get("content", result_item.get("description", "")),
                                )
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
                                    logger.warning(
                                        f"[MCP][_execute_search_tool] Filtering invalid result from {server_name}: matched indicators: {', '.join(matched_indicators)}"
                                    )
                                    continue

                                valid_results.append(result_item)
                            elif isinstance(result_item, str):
                                # 문자열 결과도 검증
                                result_lower = result_item.lower()
                                is_invalid = any(
                                    indicator in result_lower for indicator in invalid_indicators
                                )

                                if is_invalid:
                                    logger.warning(
                                        f"[MCP][_execute_search_tool] Filtering invalid string result from {server_name}: contains error message"
                                    )
                                    continue

                                # 문자열 결과를 dict 형식으로 변환
                                valid_results.append(
                                    {
                                        "title": "Search Result",
                                        "snippet": result_item,
                                        "url": "",
                                    }
                                )

                        # 유효한 결과가 있는지 확인
                        if not valid_results:
                            original_count = len(results) if isinstance(results, list) else 1
                            logger.warning(
                                f"[MCP][_execute_search_tool] ❌ All {original_count} results from {server_name} were filtered out (bot detection or error messages), trying next server..."
                            )
                            failed_servers.append(
                                {
                                    "server": server_name,
                                    "reason": f"all_results_filtered ({original_count} results filtered)",
                                }
                            )
                            continue  # 다음 서버 시도

                        original_count = len(results) if isinstance(results, list) else 1
                        filtered_count = original_count - len(valid_results)
                        logger.info(
                            f"✅ Search successful via MCP server {server_name}: {len(valid_results)} valid results (filtered {filtered_count} invalid results)"
                        )
                        tool_result = ToolResult(
                            success=True,
                            data={
                                "query": query,
                                "results": valid_results,
                                "total_results": len(valid_results),
                                "source": f"{server_name}-mcp",
                            },
                            execution_time=time.time() - start_time,
                            confidence=0.9,
                        )

                        # 캐시에 저장 (TTL: 1시간)
                        cache_dict = {
                            "success": tool_result.success,
                            "data": tool_result.data,
                            "error": tool_result.error,
                            "execution_time": tool_result.execution_time,
                            "confidence": tool_result.confidence,
                        }
                        await result_cache.set(
                            tool_name=tool_name,
                            parameters=parameters,
                            value=cache_dict,
                            ttl=3600,  # 1 hour for search results
                        )
                        logger.debug(f"[MCP][_execute_search_tool] Cached result for {tool_name}")

                        return tool_result
                    else:
                        logger.warning(
                            f"[MCP][_execute_search_tool] ❌ MCP server {server_name} returned empty results"
                        )
                        failed_servers.append({"server": server_name, "reason": "empty_results"})
                        continue

                except Exception as mcp_error:
                    error_str = str(mcp_error)
                    # ToolResult 관련 오류는 명확히 처리
                    if "ToolResult" in error_str or "cannot access local variable" in error_str:
                        logger.error(
                            f"[MCP][_execute_search_tool] ❌ MCP 서버 {server_name} 검색 실패 (코드 오류): {mcp_error}"
                        )
                        failed_servers.append(
                            {
                                "server": server_name,
                                "reason": f"code_error: {str(mcp_error)[:100]}",
                            }
                        )
                        # 다음 서버로 계속 진행
                        continue
                    else:
                        logger.warning(
                            f"[MCP][_execute_search_tool] ❌ MCP 서버 {server_name} 검색 실패: {mcp_error}, 다음 서버 시도"
                        )
                        failed_servers.append(
                            {
                                "server": server_name,
                                "reason": f"exception: {str(mcp_error)[:100]}",
                            }
                        )
                        import traceback

                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        continue

            # 모든 MCP 서버 실패 시 duckduckgo_search 라이브러리 fallback 사용
            logger.warning(
                f"[MCP][_execute_search_tool] ⚠️ All {len(server_order)} MCP search servers failed for query: '{query}'"
            )
            logger.info("[MCP][_execute_search_tool] 📋 Failed servers summary:")
            for i, failed in enumerate(failed_servers, 1):
                logger.info(
                    f"[MCP][_execute_search_tool]   {i}. {failed['server']}: {failed['reason']}"
                )

            # 모든 MCP 서버 실패 시 DDG search로 fallback
            logger.warning("[MCP][_execute_search_tool] 🔄 Falling back to DDG search...")
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
                            {"query": query, "max_results": max_results},
                        )

                        if result:
                            import json
                            import re

                            # 에러 응답 체크
                            result_lower = str(result).lower() if result else ""
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
                            for pattern in error_patterns:
                                if re.search(pattern, result_lower):
                                    is_error = True
                                    logger.warning("Error detected in tavily response, skipping")
                                    break

                            if is_error:
                                continue  # 다음 서버 시도

                            if isinstance(result, str):
                                parsed_json, parsed_data = _parse_json_text(
                                    result, context=f"Tavily result from {server_name}"
                                )
                                if parsed_json:
                                    result_data = parsed_data
                                else:
                                    # 마크다운 형식 파싱
                                    results = _parse_markdown_link_results(result)

                                    if results:
                                        result_data = {"results": results}
                                    else:
                                        result_data = {
                                            "results": [
                                                {
                                                    "title": "Search Results",
                                                    "snippet": result,
                                                    "url": "",
                                                }
                                            ]
                                        }
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
                                        "results": (
                                            results if isinstance(results, list) else [results]
                                        ),
                                        "total_results": (
                                            len(results) if isinstance(results, list) else 1
                                        ),
                                        "source": f"{server_name}-mcp",
                                    },
                                    execution_time=time.time() - start_time,
                                    confidence=0.85,
                                )

                                # 캐시에 저장 (TTL: 1시간)
                                cache_dict = {
                                    "success": tool_result.success,
                                    "data": tool_result.data,
                                    "error": tool_result.error,
                                    "execution_time": tool_result.execution_time,
                                    "confidence": tool_result.confidence,
                                }
                                await result_cache.set(
                                    tool_name=tool_name,
                                    parameters=parameters,
                                    value=cache_dict,
                                    ttl=3600,  # 1 hour for search results
                                )
                                logger.debug(
                                    f"[MCP][_execute_search_tool] Cached result for {tool_name}"
                                )

                                return tool_result

                except Exception as mcp_error:
                    logger.warning(
                        f"MCP 서버 {server_name} tavily 실패: {mcp_error}, 다음 서버 시도"
                    )
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
                            {"query": query, "numResults": max_results},
                        )

                        if result:
                            import json
                            import re

                            # 에러 응답 체크
                            result_lower = str(result).lower() if result else ""
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
                            for pattern in error_patterns:
                                if re.search(pattern, result_lower):
                                    is_error = True
                                    logger.warning("Error detected in tavily response, skipping")
                                    break

                            if is_error:
                                continue  # 다음 서버 시도

                            if isinstance(result, str):
                                parsed_json, parsed_data = _parse_json_text(
                                    result, context=f"Tavily result from {server_name}"
                                )
                                if parsed_json:
                                    result_data = parsed_data
                                else:
                                    # 마크다운 형식 파싱
                                    results = _parse_markdown_link_results(result)

                                    if results:
                                        result_data = {"results": results}
                                    else:
                                        result_data = {
                                            "results": [
                                                {
                                                    "title": "Search Results",
                                                    "snippet": result,
                                                    "url": "",
                                                }
                                            ]
                                        }
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
                                        "results": (
                                            results if isinstance(results, list) else [results]
                                        ),
                                        "total_results": (
                                            len(results) if isinstance(results, list) else 1
                                        ),
                                        "source": f"{server_name}-mcp",
                                    },
                                    execution_time=time.time() - start_time,
                                    confidence=0.85,
                                )

                                # 캐시에 저장 (TTL: 1시간)
                                cache_dict = {
                                    "success": tool_result.success,
                                    "data": tool_result.data,
                                    "error": tool_result.error,
                                    "execution_time": tool_result.execution_time,
                                    "confidence": tool_result.confidence,
                                }
                                await result_cache.set(
                                    tool_name=tool_name,
                                    parameters=parameters,
                                    value=cache_dict,
                                    ttl=3600,  # 1 hour for search results
                                )
                                logger.debug(
                                    f"[MCP][_execute_search_tool] Cached result for {tool_name}"
                                )

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
            confidence=0.0,
        )


async def _execute_academic_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """학술 도구 실행 - src/utils에서 직접 사용."""
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
                    confidence=0.0,
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
                        "source": "embedded_arxiv",
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.95,
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=result.get("error", "arXiv search failed"),
                    execution_time=time.time() - start_time,
                    confidence=0.0,
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
                sort_by=arxiv.SortCriterion.Relevance,
            )

            results = []
            for paper in client.results(search):
                results.append(
                    {
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "abstract": paper.summary,
                        "url": paper.entry_id,
                        "published": paper.published.isoformat(),
                        "pdf_url": paper.pdf_url,
                    }
                )

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                    "source": "arxiv",
                },
                execution_time=time.time() - start_time,
                confidence=0.95,
            )

        elif tool_name == "scholar":
            # Google Scholar (무료, rate limit 있음)
            from scholarly import scholarly

            search_query = scholarly.search_pubs(query)
            results = []

            for i, pub in enumerate(search_query):
                if i >= max_results:
                    break

                results.append(
                    {
                        "title": pub.get("bib", {}).get("title", ""),
                        "authors": pub.get("bib", {}).get("author", ""),
                        "abstract": pub.get("bib", {}).get("abstract", ""),
                        "url": pub.get("pub_url", ""),
                        "year": pub.get("bib", {}).get("pub_year", ""),
                        "citations": pub.get("num_citations", 0),
                    }
                )

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                    "source": "scholar",
                },
                execution_time=time.time() - start_time,
                confidence=0.8,
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
            confidence=0.0,
        )


async def _execute_data_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """실제 데이터 도구 실행."""
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
                            "source": "embedded_fetch",
                        },
                        execution_time=time.time() - start_time,
                        confidence=0.9,
                    )
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=result.get("error", "Fetch failed"),
                        execution_time=time.time() - start_time,
                        confidence=0.0,
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
                        "headers": dict(response.headers),
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.9,
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
            confidence=0.0,
        )


async def _execute_code_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """실제 코드 도구 실행 - Docker/gVisor 샌드박스 우선."""
    start_time = time.time()
    code = parameters.get("code", "")
    language = parameters.get("language", "python")
    sandbox_type = str(parameters.get("sandbox", "docker")).lower()

    # 1. 리소스 제한 체크
    try:
        from src.core.resource_limits import ResourceLimits

        code_bytes = code.encode("utf-8")
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
                confidence=0.0,
            )
    except ImportError:
        # ResourceLimits 모듈이 없으면 경고만 하고 계속 진행
        logger.debug("ResourceLimits module not available, skipping size check")

    # 2. Optional remote sandbox (Runloop/Daytona/Modal) from env SANDBOX_BACKEND
    import os

    backend_name = (os.getenv("SANDBOX_BACKEND") or "").strip().lower()
    if backend_name in ("runloop", "daytona", "modal"):
        try:
            from src.core.sandbox.factory import get_sandbox_backend

            backend = get_sandbox_backend()
            if backend is not None:
                resp = await backend.execute_code(code, language)
                execution_time = time.time() - start_time
                return ToolResult(
                    success=resp.exit_code == 0,
                    data={
                        "code": code,
                        "language": language,
                        "output": resp.output,
                        "error": resp.error,
                        "exit_code": resp.exit_code,
                        "sandbox_type": backend.id,
                    },
                    error=resp.error,
                    execution_time=execution_time,
                    confidence=0.9 if resp.exit_code == 0 else 0.5,
                )
        except Exception as e:
            logger.debug("Remote sandbox (%s) failed, falling back: %s", backend_name, e)

    # 3. Docker/gVisor 샌드박스 사용 (기본값)
    if sandbox_type in ("docker", "gvisor", "runsc", "container"):
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
                    "container_id": result.container_id,
                },
                error=result.error if not result.success else None,
                execution_time=execution_time,
                confidence=0.9 if result.success else 0.5,
            )

        except Exception as e:
            logger.error(f"Docker sandbox execution failed: {e}")
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                data=None,
                error=f"Docker sandbox failed: {str(e)}",
                execution_time=execution_time,
                confidence=0.0,
            )

    if sandbox_type not in ("docker", "gvisor", "runsc", "container"):
        return ToolResult(
            success=False,
            data=None,
            error=(f"Unsupported sandbox '{sandbox_type}'. " "Use 'docker' or 'runsc'."),
            execution_time=time.time() - start_time,
            confidence=0.0,
        )


async def _playwright_dismiss_google_consent(page: Any) -> None:
    """Google 검색 진입 시 지역·쿠키 동의 UI가 뜨면 닫기 시도."""
    candidates = [
        'button:has-text("Accept all")',
        'button:has-text("Accept All")',
        'button:has-text("I agree")',
        'button:has-text("동의")',
        'button:has-text("모두 동의")',
        '[aria-label="Accept all"]',
        'form[action*="consent"] button',
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if await loc.is_visible(timeout=1200):
                await loc.click(timeout=2500)
                await page.wait_for_timeout(400)
                break
        except Exception:
            continue


async def _execute_browser_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """브라우저 자동화 도구 실행."""
    start_time = time.time()

    try:
        from src.automation.browser_manager import BrowserManager

        # BrowserManager 인스턴스 생성 (싱글톤 패턴 고려)
        browser_manager = BrowserManager()

        # browser-use 기반 브라우저 유틸은 `browser_navigate`/`browser_extract`에서만 사용됩니다.
        # `browser_search` 등 Playwright 전용 경로에서는 browser-use가 없어도 동작해야 합니다.
        if (
            tool_name in {"browser_navigate", "browser_extract"}
            and not browser_manager.browser_available
        ):
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
                confidence=0.9 if result.get("success") else 0.0,
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
                confidence=0.9 if result.get("success") else 0.0,
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
                        confidence=0.9,
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
                                results.append(
                                    {
                                        "type": "click",
                                        "selector": selector,
                                        "success": True,
                                    }
                                )
                            elif action_type == "fill":
                                await page.fill(selector, value)
                                results.append(
                                    {
                                        "type": "fill",
                                        "selector": selector,
                                        "success": True,
                                    }
                                )
                            elif action_type == "select":
                                await page.select_option(selector, value)
                                results.append(
                                    {
                                        "type": "select",
                                        "selector": selector,
                                        "success": True,
                                    }
                                )
                            elif action_type == "wait":
                                await page.wait_for_selector(selector, timeout=5000)
                                results.append(
                                    {
                                        "type": "wait",
                                        "selector": selector,
                                        "success": True,
                                    }
                                )
                            else:
                                results.append(
                                    {
                                        "type": action_type,
                                        "success": False,
                                        "error": "Unknown action type",
                                    }
                                )
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
                            "final_content": final_content[:10000],  # 처음 10000자만
                        },
                        execution_time=time.time() - start_time,
                        confidence=0.8 if all(r.get("success", False) for r in results) else 0.5,
                    )
            else:
                raise RuntimeError("Playwright not available for browser interaction")

        elif tool_name == "browser_search":
            # Headless Playwright 검색. Wikipedia는 안정적, Google은 SERP 파싱(차단 가능).
            import urllib.parse

            query = parameters.get("query", "")
            engine = (
                (
                    parameters.get("engine")
                    or os.getenv("SPARKLEFORGE_BROWSER_SEARCH_ENGINE", "wikipedia")
                )
                .lower()
                .strip()
            )
            max_results = int(min(20, max(1, int(parameters.get("max_results", 3) or 3))))

            if not query:
                raise ValueError("query parameter is required for browser_search")

            if engine not in {"wikipedia", "google", "bing", "duckduckgo"}:
                raise ValueError(
                    f"Unsupported browser_search engine: {engine}. "
                    "Use 'wikipedia', 'google', 'bing', or 'duckduckgo'."
                )

            from src.automation.browser_manager import BrowserManager

            browser_manager = BrowserManager()
            await browser_manager.initialize_playwright()
            if not browser_manager.playwright_page:
                raise RuntimeError("Playwright page not initialized for browser_search")

            page = browser_manager.playwright_page

            async def _wikipedia_search() -> ToolResult:
                """Playwright로 Wikipedia 검색을 수행하고 결과를 ToolResult로 반환."""
                q_encoded = urllib.parse.quote(query)
                url = (
                    f"https://en.wikipedia.org/w/index.php?search={q_encoded}"
                    f"&title=Special:Search&ns0=1"
                )
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(1200)
                wiki_results = await page.evaluate(
                    """
                    (maxResults) => {
                        const clean = (s) => (s || '').toString().trim();
                        const pageUrl = window.location.href;
                        const resultEls = Array.from(
                            document.querySelectorAll(
                                '#mw-content-text .mw-search-result, #mw-content-text li.mw-search-result'
                            )
                        );
                        const out = [];
                        for (const el of resultEls) {
                            const a = el.querySelector('a');
                            if (!a) continue;
                            const title = clean(a.textContent);
                            let href = a.href || a.getAttribute('href') || '';
                            if (href && href.startsWith('/')) {
                                href = new URL(href, location.origin).href;
                            }
                            const snippetEl =
                                el.querySelector('.searchresult') ||
                                el.querySelector('.mw-search-result-data') ||
                                el.querySelector('p') ||
                                el;
                            const snippet = clean(snippetEl.textContent);
                            if (title && href) {
                                out.push({
                                    title,
                                    url: href,
                                    snippet: snippet.slice(0, 500),
                                    source: 'wikipedia',
                                });
                            }
                            if (out.length >= maxResults) break;
                        }
                        if (out.length) return out.slice(0, maxResults);
                        const h1 = document.querySelector('h1');
                        const title = clean(h1 ? h1.innerText : document.title);
                        const ps = Array.from(
                            document.querySelectorAll('#mw-content-text .mw-parser-output p')
                        );
                        let snippet = '';
                        for (const p of ps) {
                            const t = clean(p.innerText);
                            if (t && t.length >= 30) {
                                snippet = t;
                                break;
                            }
                        }
                        return [{
                            title,
                            url: pageUrl,
                            snippet: clean(snippet).slice(0, 500),
                            source: 'wikipedia',
                        }].slice(0, maxResults);
                    }
                    """,
                    max_results,
                )

                if not isinstance(wiki_results, list) or len(wiki_results) == 0:
                    return ToolResult(
                        success=False,
                        data={"results": [], "query": query, "engine": "wikipedia"},
                        execution_time=time.time() - start_time,
                        confidence=0.0,
                        error="wikipedia returned no results",
                    )

                return ToolResult(
                    success=True,
                    data={
                        "results": wiki_results,
                        "query": query,
                        "engine": "wikipedia",
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.9,
                )

            if engine == "wikipedia":
                return await _wikipedia_search()

            elif engine == "google":
                hl = os.getenv("BROWSER_SEARCH_GOOGLE_HL", "ko")
                gl = os.getenv("BROWSER_SEARCH_GOOGLE_GL", "kr")
                num = min(max_results, 15)
                q_enc = urllib.parse.quote(query)
                g_url = (
                    f"https://www.google.com/search?q={q_enc}"
                    f"&hl={urllib.parse.quote(hl)}&gl={urllib.parse.quote(gl)}"
                    f"&num={num}&pws=0"
                )
                await page.goto(g_url, wait_until="domcontentloaded", timeout=45000)
                await _playwright_dismiss_google_consent(page)
                await page.wait_for_timeout(800)
                try:
                    await page.wait_for_selector(
                        "#search, #rso, form#captcha-form, div#recaptcha",
                        timeout=15000,
                    )
                except Exception:
                    pass
                if await page.query_selector("form#captcha-form"):
                    return await _wikipedia_search()

                body_lower = ((await page.content()) or "")[:120000].lower()
                if (
                    "detected unusual traffic" in body_lower
                    or "unusual traffic from your computer network" in body_lower
                    or "/recaptcha/" in body_lower
                ):
                    return await _wikipedia_search()

                results = await page.evaluate(
                    """
                    (maxResults) => {
                        const clean = (s) => (s || '').toString().replace(/\\s+/g, ' ').trim();
                        const out = [];
                        const seen = new Set();
                        const skipUrl = (u) => {
                            if (!u || !u.startsWith('http')) return true;
                            try {
                                const h = new URL(u).hostname.toLowerCase();
                                if (h === 'google.com' || h.endsWith('.google.com')) return true;
                                if (h.includes('gstatic.com')) return true;
                                if (h.includes('youtube.com')) return true;
                            } catch (e) { return true; }
                            return false;
                        };
                        let nodes = document.querySelectorAll('#search a h3');
                        if (!nodes.length) nodes = document.querySelectorAll('#rso a h3');
                        if (!nodes.length) nodes = document.querySelectorAll('div[data-hveid] a h3');
                        for (const h3 of nodes) {
                            const a = h3.closest('a');
                            if (!a || !a.href) continue;
                            let href = a.href;
                            if (href.startsWith('/url?')) {
                                try {
                                    const sp = new URL(href, location.origin).searchParams;
                                    href = sp.get('q') || sp.get('url') || href;
                                } catch (e) {}
                            }
                            if (href.startsWith('/')) {
                                try { href = new URL(href, location.origin).href; } catch (e) {}
                            }
                            if (skipUrl(href)) continue;
                            const title = clean(h3.textContent);
                            if (!title || seen.has(href)) continue;
                            seen.add(href);
                            let snippet = '';
                            const block =
                                a.closest('div[data-sokoban-container]') ||
                                a.closest('div.Gx5Zad') ||
                                a.closest('div.g') ||
                                a.closest('div');
                            if (block) {
                                const st = clean(block.innerText || '');
                                if (st.length > title.length + 8) snippet = st.slice(0, 500);
                            }
                            out.push({ title, url: href, snippet, source: 'google' });
                            if (out.length >= maxResults) break;
                        }
                        return out.slice(0, maxResults);
                    }
                    """,
                    max_results,
                )

                if not isinstance(results, list) or len(results) == 0:
                    return await _wikipedia_search()

                return ToolResult(
                    success=True,
                    data={"results": results, "query": query, "engine": engine},
                    execution_time=time.time() - start_time,
                    confidence=0.85,
                )

            elif engine == "bing":
                q_enc = urllib.parse.quote(query)
                b_url = (
                    f"https://www.bing.com/search?q={q_enc}" f"&setlang=en-US&cc=US&form=QBLH&sp=-1"
                )
                await page.goto(b_url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(1000)
                results = await page.evaluate(
                    """
                    (maxResults) => {
                        const clean = (s) => (s || '').toString().replace(/\\s+/g, ' ').trim();
                        const out = [];
                        const nodes = document.querySelectorAll('#b_results .b_algo h2 a');
                        for (let i = 0; i < nodes.length && out.length < maxResults; i++) {
                            const a = nodes[i];
                            const title = clean(a.textContent);
                            const href = a.href || '';
                            let snippet = '';
                            const li = a.closest('li') || a.parentElement;
                            if (li) {
                                const p = li.querySelector('p');
                                if (p) snippet = clean(p.textContent);
                                else {
                                    const cap = li.querySelector('.b_caption p');
                                    if (cap) snippet = clean(cap.textContent);
                                }
                            }
                            if (title && href) {
                                out.push({ title, url: href, snippet: snippet.slice(0, 500), source: 'bing' });
                            }
                        }
                        return out;
                    }
                    """,
                    max_results,
                )
                if not isinstance(results, list) or len(results) == 0:
                    return await _wikipedia_search()
                return ToolResult(
                    success=True,
                    data={"results": results, "query": query, "engine": engine},
                    execution_time=time.time() - start_time,
                    confidence=0.8,
                )

            elif engine == "duckduckgo":
                q_enc = urllib.parse.quote(query)
                ddg_url = f"https://duckduckgo.com/html/?q={q_enc}&kl=us-en&kp=1"
                await page.goto(ddg_url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(1200)
                results = await page.evaluate(
                    """
                    (maxResults) => {
                        const clean = (s) => (s || '').toString().replace(/\\s+/g, ' ').trim();
                        const out = [];
                        const blocks = Array.from(document.querySelectorAll('.result'));
                        for (const b of blocks) {
                            const a = b.querySelector('a.result__a');
                            if (!a) continue;
                            const title = clean(a.textContent);
                            let href = a.href || b.querySelector('a')?.getAttribute('href') || '';
                            if (!href || !href.startsWith('http')) {
                                try {
                                    href = new URL(href, location.origin).href;
                                } catch (e) {}
                            }
                            const sn = b.querySelector('.result__snippet');
                            const snippet = sn ? clean(sn.textContent) : '';
                            if (title && href) {
                                out.push({ title, url: href, snippet: snippet.slice(0, 500), source: 'duckduckgo' });
                            }
                            if (out.length >= maxResults) break;
                        }
                        return out;
                    }
                    """,
                    max_results,
                )
                if not isinstance(results, list) or len(results) == 0:
                    return await _wikipedia_search()
                return ToolResult(
                    success=True,
                    data={"results": results, "query": query, "engine": engine},
                    execution_time=time.time() - start_time,
                    confidence=0.75,
                )

        else:
            raise ValueError(f"Unknown browser tool: {tool_name}")

    except Exception as e:
        logger.error(f"Browser tool execution failed: {tool_name} - {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=f"Browser tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0,
        )


async def _execute_document_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """문서 생성 도구 실행."""
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
            output_format=output_format,
        )

        return ToolResult(
            success=True,
            data={
                "file_path": file_path,
                "format": output_format,
                "report_type": report_type,
            },
            execution_time=time.time() - start_time,
            confidence=0.9,
        )

    except Exception as e:
        logger.error(f"Document tool execution failed: {tool_name} - {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=f"Document tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0,
        )


async def _execute_git_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """Git 워크플로우 도구 실행."""
    start_time = time.time()

    try:
        from pathlib import Path

        from src.core.git_workflow import GitWorkflow

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
                confidence=0.9,
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
                confidence=0.9 if result.get("success") else 0.0,
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
                confidence=0.9 if result.get("success") else 0.0,
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
                    confidence=0.0,
                )

            result = await git_workflow.git_create_pr(title=title, body=body, base=base)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
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
                base=base,
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
            )

        else:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown git tool: {tool_name}",
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

    except Exception as e:
        logger.error(f"Git tool execution failed: {tool_name} - {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=f"Git tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0,
        )


async def _execute_shell_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """Shell 명령 실행 도구 (완전 자동형 SparkleForge)."""
    start_time = time.time()

    try:
        from pathlib import Path

        from src.core.shell_executor import ShellExecutor

        # 작업 디렉토리 확인
        working_dir = parameters.get("working_dir")
        if working_dir:
            working_dir = Path(working_dir)
        else:
            working_dir = None

        # ShellExecutor 생성
        executor = ShellExecutor(
            require_confirmation=parameters.get("require_confirmation", False),
            max_execution_time=parameters.get("timeout", 300),
        )

        if tool_name == "run_shell_command":
            command = parameters.get("command")
            if not command:
                return ToolResult(
                    success=False,
                    data=None,
                    error="command parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )

            confirm = parameters.get("confirm")
            timeout = parameters.get("timeout")
            result = await executor.run(
                command=command,
                working_dir=working_dir,
                confirm=confirm,
                timeout=timeout,
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
            )

        elif tool_name == "run_interactive_command":
            command = parameters.get("command")
            if not command:
                return ToolResult(
                    success=False,
                    data=None,
                    error="command parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )

            input_data = parameters.get("input")
            result = await executor.run_interactive(
                command=command, working_dir=working_dir, input_data=input_data
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
            )

        elif tool_name == "run_background_command":
            command = parameters.get("command")
            if not command:
                return ToolResult(
                    success=False,
                    data=None,
                    error="command parameter is required",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                )

            result = await executor.run_background(command=command, working_dir=working_dir)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
            )

        else:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown shell tool: {tool_name}",
                execution_time=time.time() - start_time,
                confidence=0.0,
            )

    except Exception as e:
        logger.error(f"Shell tool execution failed: {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=str(e),
            execution_time=time.time() - start_time,
            confidence=0.0,
        )


async def _execute_file_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """파일 작업 도구 실행."""
    start_time = time.time()

    try:
        from pathlib import Path

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
                if path.is_absolute() and not any(
                    path.is_relative_to(allowed) for allowed in allowed_dirs
                ):
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

        if tool_name == "filesystem":
            # 범용 filesystem 도구: operation/action 파라미터를 구체 도구로 매핑
            operation = str(parameters.get("operation") or parameters.get("action") or "").lower()
            op_map = {
                "create": "create_file",
                "read": "read_file",
                "write": "write_file",
                "edit": "edit_file",
                "list": "list_files",
                "delete": "delete_file",
            }
            mapped = op_map.get(operation)
            if not mapped:
                raise ValueError(f"Unknown filesystem operation: {operation or '(missing)'}")
            # read 대상이 디렉토리면 목록 조회로 처리
            target = parameters.get("path") or parameters.get("file_path") or ""
            if mapped == "read_file" and target and Path(target).is_dir():
                mapped = "list_files"
            if "file_path" not in parameters and "path" in parameters:
                parameters = {**parameters, "file_path": parameters["path"]}
            if mapped == "list_files" and "directory_path" not in parameters:
                parameters = {
                    **parameters,
                    "directory_path": parameters.get("path", parameters.get("file_path", ".")),
                }
            return await _execute_file_tool(mapped, parameters)

        if tool_name == "create_file":
            file_path = parameters.get("file_path", "")
            content = parameters.get("content", "")

            if not file_path:
                raise ValueError("file_path parameter is required")
            if not _is_safe_path(file_path):
                raise ValueError(f"Unsafe file path: {file_path}")

            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

            return ToolResult(
                success=True,
                data={"file_path": str(path), "size": len(content)},
                execution_time=time.time() - start_time,
                confidence=0.9,
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

            content = path.read_text(encoding="utf-8")

            return ToolResult(
                success=True,
                data={"file_path": str(path), "content": content, "size": len(content)},
                execution_time=time.time() - start_time,
                confidence=0.9,
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
            path.write_text(content, encoding="utf-8")

            return ToolResult(
                success=True,
                data={"file_path": str(path), "size": len(content)},
                execution_time=time.time() - start_time,
                confidence=0.9,
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

            content = path.read_text(encoding="utf-8")
            if old_string not in content:
                raise ValueError(f"Old string not found in file: {file_path}")

            new_content = content.replace(old_string, new_string)
            path.write_text(new_content, encoding="utf-8")

            return ToolResult(
                success=True,
                data={
                    "file_path": str(path),
                    "replacements": content.count(old_string),
                },
                execution_time=time.time() - start_time,
                confidence=0.9,
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
                    files.append(
                        {
                            "name": item.name,
                            "path": str(item.relative_to(path)),
                            "is_file": item.is_file(),
                            "size": item.stat().st_size if item.is_file() else 0,
                        }
                    )
            else:
                for item in path.iterdir():
                    files.append(
                        {
                            "name": item.name,
                            "path": item.name,
                            "is_file": item.is_file(),
                            "size": item.stat().st_size if item.is_file() else 0,
                        }
                    )

            return ToolResult(
                success=True,
                data={"directory": str(path), "files": files, "count": len(files)},
                execution_time=time.time() - start_time,
                confidence=0.9,
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
                confidence=0.9,
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
            confidence=0.0,
        )


async def get_tool_for_category(category: ToolCategory) -> str | None:
    """카테고리에 해당하는 도구 반환."""
    mcp_hub = get_mcp_hub()
    return mcp_hub.get_tool_for_category(category)


async def get_best_tool_for_task(
    task_type: str, category: ToolCategory | None = None
) -> str | None:
    """태스크 타입에 가장 적합한 도구 반환."""
    if category is not None:
        return await get_tool_for_category(category)
    mcp_hub = get_mcp_hub()
    # task_type 키워드로 카테고리 추론
    keyword_map = {
        "search": ToolCategory.SEARCH,
        "academic": ToolCategory.ACADEMIC,
        "data": ToolCategory.DATA,
        "code": ToolCategory.CODE,
        "file": ToolCategory.FILE,
        "browser": ToolCategory.BROWSER,
        "document": ToolCategory.DOCUMENT,
        "git": ToolCategory.GIT,
    }
    for keyword, cat in keyword_map.items():
        if keyword in task_type.lower():
            return mcp_hub.get_tool_for_category(cat)
    return None


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
                cmd = info.get("command", "unknown")
                args_preview = " ".join(info.get("args", [])[:3])
                print(f"   명령어: {cmd} {args_preview}...")

            print(f"   연결 상태: {'연결됨' if info['connected'] else '연결 안 됨'}")
            print(f"   제공 Tool 수: {info['tools_count']}")

            if info["tools"]:
                print("   Tool 목록:")
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
    parser.add_argument(
        "--check-servers", action="store_true", help="Check all MCP server connections"
    )

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

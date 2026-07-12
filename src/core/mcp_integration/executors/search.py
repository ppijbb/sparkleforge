"""Search tool dispatch (ToolCategory.SEARCH): embedded DDG search, MCP-server fallback across configured search servers, and the final duckduckgo_search library fallback."""
import asyncio
import json
import logging
import os
import random
import re
import time
from typing import Any, Dict

from src.core.mcp_integration.parser import (
    _parse_json_text,
    _parse_markdown_link_results,
)
from src.core.mcp_integration.tools import get_mcp_hub
from src.core.tools.registry import ToolResult

logger = logging.getLogger(__name__)


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

    from src.core.result_cache import get_result_cache

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
                                    link_match = re.match(
                                        r"^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)", line
                                    )
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

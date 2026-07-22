"""Tool-execution mixin for UniversalMCPHub: routes a tool call to the right MCP server or native executor, with caching, tracing, and error normalization."""
import asyncio
import logging
import time
from typing import Any, Dict, List

from src.core.mcp_integration.executors.browser import _execute_browser_tool
from src.core.mcp_integration.executors.data import _execute_data_tool
from src.core.mcp_integration.executors.file import _execute_file_tool
from src.core.mcp_integration.executors.shell import _execute_shell_tool
from src.core.mcp_integration.mcp_runtime import McpError, TextContent
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
)
from src.core.mcp_integration.tools import get_mcp_hub
from src.core.tools.registry import ToolCategory

logger = logging.getLogger(__name__)


def _infer_required_capability(tool_name: str) -> str | None:
    """Best-effort map from a tool name to an existing BUILTIN_CAPABILITIES entry.

    Reuses the same prefix conventions execute_tool() already dispatches
    local tools on (see below), rather than inventing a parallel tool
    taxonomy. Returns None for tool names that don't clearly map to one of
    the existing coarse action-risk capabilities -- InvocationGateway
    treats that as "no capability requirement for this call", not "deny".
    """
    if tool_name.startswith(("shell", "run_")) or tool_name == "shell":
        return "execute_shell"
    if tool_name.startswith(("write_", "edit_", "create_", "delete_")):
        return "write_file"
    if tool_name.startswith(("read_", "list_")) or tool_name == "filesystem":
        return "read_file"
    if tool_name.startswith("browser") or tool_name == "browser":
        return "network_request"
    return None


class ExecutionMixin:
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

        # 단일 강제 진입점 (issue #568): 모든 MCP 도구 호출은 IntentGuardrail/
        # CapabilityManager 체크를 강제하는 InvocationGateway를 반드시 거친다.
        from src.core.agent_security import get_current_agent_name
        from src.core.guard.invocation_gateway import InvocationKind, get_invocation_gateway

        actor = get_current_agent_name() or "system"
        gateway_decision = get_invocation_gateway().authorize(
            kind=InvocationKind.MCP_TOOL,
            actor=actor,
            target=tool_name,
            description=query_str,
            required_capability=_infer_required_capability(tool_name),
        )
        if not gateway_decision.allowed:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "data": None,
                "error": f"Denied by invocation gateway: {gateway_decision.reason}",
                "execution_time": execution_time,
            }

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
                else:
                    # 이 도구는 MCP 전용으로 등록되어 있는데 해당 서버 세션이 아직
                    # 연결되지 않은 상태. 로컬 도구 폴백으로 흘러가면 동명의 다른
                    # 도구가 실행되거나 조용히 잘못된 결과를 반환할 수 있으므로,
                    # 여기서 명시적으로 에러를 반환한다.
                    execution_time = time.time() - start_time
                    error_msg = f"MCP server '{server_name}' is not connected"
                    logger.error(f"[MCP][exec.error] {error_msg} (tool={tool_name})")

                    tool_exec_result = ToolExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        execution_time=execution_time,
                        result_summary=f"MCP 서버 연결 안 됨: {server_name}",
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
                        # Categories with no dedicated dispatch function (e.g. UTILITY,
                        # BROWSER for CDP tools) are plain Python callables registered
                        # directly in the ToolRegistry -- invoke them through it instead
                        # of assuming they belong to the DATA dispatcher, which only
                        # knows about fetch/filesystem/browser/shell and would raise
                        # "Unknown data tool" for anything else (e.g. the scheduler's
                        # create_automation_task/list_automation_tasks).
                        from src.core.mcp_integration import ToolResult

                        raw_result = await self.registry.execute(tool_name, parameters)
                        tool_result = (
                            raw_result
                            if isinstance(raw_result, ToolResult)
                            else ToolResult(
                                success=True,
                                data=raw_result,
                                confidence=1.0,
                                tool_name=tool_name,
                            )
                        )

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

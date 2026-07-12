"""MCP transport/connection-lifecycle mixin for UniversalMCPHub: stdio/SSE/streamable-HTTP connections, server config loading, health checks."""
import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict

from src.core.mcp_integration.mcp_runtime import (
    ClientSession,
    FASTMCP_AVAILABLE,
    FastMCPClient,
    MCP_AVAILABLE,
    StdioServerParameters,
    project_root,
    stdio_client,
)
from src.core.mcp_tool_loader import ToolInfo as MCPToolInfo

logger = logging.getLogger(__name__)

class ConnectionMixin:
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
    async def ensure_server_connected(self, server_name: str) -> bool:
        """서버 연결 보장 (Lazy loading)."""
        if not self.mcp_server_configs:
            self._load_mcp_servers_from_config()
        if server_name in self.mcp_server_configs:
            return await self._connect_to_mcp_server(server_name, self.mcp_server_configs[server_name])
        return False

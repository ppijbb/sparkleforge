"""Status/introspection mixin for UniversalMCPHub: category lookup, server health summaries, langchain tool listing."""
import logging
from datetime import datetime
from typing import Any, Dict, List

from src.core.mcp_integration.tools import execute_tool
from src.core.tools.registry import ToolCategory

logger = logging.getLogger(__name__)

class StatusMixin:
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

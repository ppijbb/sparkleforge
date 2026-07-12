"""Tool-execution dispatch layer for the MCP integration hub.

Extracted from the monolithic ``src/core/mcp_integration.py`` (issue #508,
Anvil Phase Sigma-1). This module owns only the top-level dispatch surface
-- ``execute_tool``/``get_mcp_hub`` and the CLI helpers. Each
``ToolCategory``'s ``_execute_*_tool`` implementation lives in its own
module under ``src/core/mcp_integration/executors/`` (issue #507/#524:
this file used to be 3,257 lines with all nine dispatchers inlined).

``UniversalMCPHub`` (``src.core.mcp_integration.hub``) needs ``get_mcp_hub``
as a bare module-level reference (see ``hub.py``'s imports), so this module
intentionally does *not* import from ``hub`` at module load time -- only
inside ``get_mcp_hub()``, deferred until first call, to avoid a circular
import.
"""

import asyncio
import logging
import os
import sys
from typing import TYPE_CHECKING, Any, Dict, List

from src.core.observability import start_tool_span
from src.core.researcher_config import get_mcp_config
from src.core.tools.registry import ToolCategory

if TYPE_CHECKING:
    from src.core.mcp_integration.hub import UniversalMCPHub

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

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

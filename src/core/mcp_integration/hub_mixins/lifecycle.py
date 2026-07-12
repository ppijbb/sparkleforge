"""Shutdown/cleanup mixin for UniversalMCPHub."""
import asyncio
import logging
from typing import Any

from src.core.mcp_integration.mcp_runtime import (
    FASTMCP_AVAILABLE,
    FastMCPClient,
)

logger = logging.getLogger(__name__)

class LifecycleMixin:
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

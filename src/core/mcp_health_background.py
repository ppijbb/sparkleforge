"""
MCP Health Background Service - 백그라운드 헬스체크 서비스

기존 MCPHub와 독립적으로 동작하는 백그라운드 헬스체크 서비스.
주기적으로 연결 상태를 확인하고 필요시 재연결을 시도합니다.
기존 MCPHub 코드는 수정하지 않고, 기존 메서드만 호출합니다.
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MCPHealthBackgroundService:
    """백그라운드 헬스체크 서비스 - 기존 시스템과 독립"""
    
    def __init__(self, mcp_hub, interval: int = 60):
        """
        초기화
        
        Args:
            mcp_hub: 기존 MCPHub 인스턴스 (수정하지 않음)
            interval: 헬스체크 간격 (초)
        """
        self.mcp_hub = mcp_hub
        self.interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.last_check: dict = {}  # server_name -> last_check_time
        logger.info(f"MCP Health Background Service initialized (interval: {interval}s)")
    
    async def start(self):
        """백그라운드에서 주기적 헬스체크 시작"""
        if self._running:
            logger.warning("Health background service is already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("MCP Health Background Service started")
    
    async def stop(self):
        """백그라운드 헬스체크 중지"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("MCP Health Background Service stopped")
    
    async def _monitor_loop(self):
        """모니터링 루프"""
        while self._running:
            try:
                await asyncio.sleep(self.interval)
                if not self._running:
                    break
                
                await self._check_all_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor loop error: {e}")
                await asyncio.sleep(5)  # 에러 발생 시 5초 대기 후 재시도
    
    async def _check_all_connections(self):
        """모든 연결 상태 확인"""
        try:
            if not hasattr(self.mcp_hub, 'mcp_sessions'):
                return
            
            # 기존 MCPHub의 세션 목록 가져오기 (수정 없음)
            server_names = list(self.mcp_hub.mcp_sessions.keys())
            
            if not server_names:
                return
            
            logger.debug(f"Health check: Checking {len(server_names)} MCP servers")
            
            for server_name in server_names:
                try:
                    # 기존 MCPHub의 헬스체크 메서드 사용 (수정 없음)
                    if hasattr(self.mcp_hub, '_check_connection_health'):
                        is_healthy = await self.mcp_hub._check_connection_health(server_name)
                        self.last_check[server_name] = datetime.now()
                        
                        if not is_healthy:
                            logger.warning(f"Health check: Server {server_name} is unhealthy, attempting reconnection")
                            await self._reconnect_server(server_name)
                    else:
                        logger.debug(f"Health check: _check_connection_health method not available")
                except Exception as e:
                    logger.error(f"Health check failed for {server_name}: {e}")
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    async def _reconnect_server(self, server_name: str):
        """서버 재연결 시도"""
        try:
            # 기존 MCPHub의 재연결 메서드 사용 (수정 없음)
            if hasattr(self.mcp_hub, '_reconnect_server'):
                await self.mcp_hub._reconnect_server(server_name)
            elif hasattr(self.mcp_hub, '_connect_to_mcp_server'):
                # 재연결 시도
                if hasattr(self.mcp_hub, 'mcp_server_configs') and server_name in self.mcp_hub.mcp_server_configs:
                    server_config = self.mcp_hub.mcp_server_configs[server_name]
                    # 기존 연결 정리
                    if hasattr(self.mcp_hub, '_disconnect_from_mcp_server'):
                        try:
                            await self.mcp_hub._disconnect_from_mcp_server(server_name)
                        except Exception:
                            pass
                    # 재연결
                    await self.mcp_hub._connect_to_mcp_server(server_name, server_config)
                    logger.info(f"Health check: Reconnected to {server_name}")
            else:
                logger.warning(f"Health check: No reconnection method available for {server_name}")
        except Exception as e:
            logger.error(f"Reconnection failed for {server_name}: {e}")


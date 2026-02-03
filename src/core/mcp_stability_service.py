"""
MCP Stability Service - MCP 안정성 강화 서비스

기존 MCPHub를 래핑하여 토큰 갱신, 헬스체크, 스로틀링 등의 안정성 기능을 제공합니다.
기존 MCPHub 코드는 수정하지 않고, 이 서비스를 통해 안정성을 강화합니다.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class TokenRefresher:
    """토큰 갱신 관리자"""
    
    def __init__(self):
        self.token_cache: Dict[str, Dict[str, Any]] = {}  # server_name -> {token, expires_at}
        self.refresh_threshold = timedelta(minutes=5)  # 만료 5분 전 갱신
    
    async def ensure_valid_token(self, mcp_hub, server_name: Optional[str] = None):
        """
        토큰이 유효한지 확인하고 필요시 갱신
        
        Args:
            mcp_hub: 기존 MCPHub 인스턴스
            server_name: 확인할 서버 이름 (None이면 모든 서버)
        """
        try:
            # 현재는 기본 구현만 (실제 토큰 갱신 로직은 서버별로 다를 수 있음)
            # 401 에러 발생 시 재연결 로직이 이미 mcp_integration.py에 있으므로
            # 여기서는 추가 검증만 수행
            if server_name:
                # 특정 서버의 연결 상태 확인
                if hasattr(mcp_hub, '_check_connection_health'):
                    is_healthy = await mcp_hub._check_connection_health(server_name)
                    if not is_healthy:
                        logger.warning(f"Token refresh: Server {server_name} is unhealthy, reconnection may be needed")
            else:
                # 모든 서버 확인
                if hasattr(mcp_hub, 'mcp_sessions'):
                    for srv_name in list(mcp_hub.mcp_sessions.keys()):
                        if hasattr(mcp_hub, '_check_connection_health'):
                            is_healthy = await mcp_hub._check_connection_health(srv_name)
                            if not is_healthy:
                                logger.warning(f"Token refresh: Server {srv_name} is unhealthy")
        except Exception as e:
            logger.debug(f"Token refresh check failed: {e}")


class HealthMonitor:
    """헬스체크 모니터"""
    
    def __init__(self):
        self.last_check: Dict[str, datetime] = {}  # server_name -> last_check_time
        self.check_interval = timedelta(seconds=30)  # 30초마다 체크
    
    async def check_and_reconnect(self, mcp_hub, server_name: Optional[str] = None):
        """
        연결 상태 확인 및 필요시 재연결
        
        Args:
            mcp_hub: 기존 MCPHub 인스턴스
            server_name: 확인할 서버 이름 (None이면 모든 서버)
        """
        try:
            if not hasattr(mcp_hub, 'mcp_sessions'):
                return
            
            servers_to_check = [server_name] if server_name else list(mcp_hub.mcp_sessions.keys())
            
            for srv_name in servers_to_check:
                if srv_name not in mcp_hub.mcp_sessions:
                    continue
                
                # 마지막 체크 시간 확인
                last_check = self.last_check.get(srv_name)
                if last_check and datetime.now() - last_check < self.check_interval:
                    continue
                
                # 헬스체크 수행
                if hasattr(mcp_hub, '_check_connection_health'):
                    is_healthy = await mcp_hub._check_connection_health(srv_name)
                    self.last_check[srv_name] = datetime.now()
                    
                    if not is_healthy:
                        logger.warning(f"Health check: Server {srv_name} is unhealthy, attempting reconnection")
                        if hasattr(mcp_hub, '_reconnect_server'):
                            await mcp_hub._reconnect_server(srv_name)
                        elif hasattr(mcp_hub, '_connect_to_mcp_server'):
                            # 재연결 시도
                            if srv_name in mcp_hub.mcp_server_configs:
                                server_config = mcp_hub.mcp_server_configs[srv_name]
                                await mcp_hub._connect_to_mcp_server(srv_name, server_config)
        except Exception as e:
            logger.debug(f"Health check failed: {e}")


class SearchThrottler:
    """검색 도구 스로틀링 관리자"""
    
    def __init__(self):
        self.last_request_time: Dict[str, float] = {}  # tool_name -> last_request_timestamp
        self.min_interval = 1.0  # 최소 요청 간격 (초)
        self.search_tools = ['ddg_search', 'tavily', 'g_search', 'search']  # 검색 도구 목록
    
    async def wait_if_needed(self, tool_name: str):
        """
        필요시 스로틀링 대기
        
        Args:
            tool_name: 도구 이름
        """
        # 검색 도구인지 확인
        is_search_tool = any(search_tool in tool_name.lower() for search_tool in self.search_tools)
        if not is_search_tool:
            return
        
        current_time = time.time()
        last_time = self.last_request_time.get(tool_name, 0)
        elapsed = current_time - last_time
        
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            logger.debug(f"Throttling: Waiting {wait_time:.2f}s before {tool_name} request")
            await asyncio.sleep(wait_time)
        
        self.last_request_time[tool_name] = time.time()


class MCPStabilityService:
    """MCP 안정성 서비스 - 기존 MCPHub를 래핑하여 안정성 강화"""
    
    def __init__(self):
        self.token_refresher = TokenRefresher()
        self.health_monitor = HealthMonitor()
        self.throttler = SearchThrottler()
        logger.info("MCP Stability Service initialized")
    
    async def enhance_mcp_call(
        self,
        mcp_hub,
        tool_name: str,
        params: Dict[str, Any],
        server_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        기존 MCPHub 호출을 래핑하여 안정성 강화
        
        Args:
            mcp_hub: 기존 MCPHub 인스턴스 (수정하지 않음)
            tool_name: 도구 이름
            params: 도구 파라미터
            server_name: 서버 이름 (선택적)
            
        Returns:
            기존 execute_tool()의 반환값과 동일
        """
        try:
            # 1. 토큰 갱신 확인
            await self.token_refresher.ensure_valid_token(mcp_hub, server_name)
            
            # 2. 헬스체크 및 재연결
            await self.health_monitor.check_and_reconnect(mcp_hub, server_name)
            
            # 3. 스로틀링 (검색 도구인 경우)
            await self.throttler.wait_if_needed(tool_name)
            
            # 4. 기존 MCPHub 메서드 그대로 호출 (변경 없음)
            if hasattr(mcp_hub, 'execute_tool'):
                result = await mcp_hub.execute_tool(tool_name, params)
                return result
            else:
                logger.error("MCPHub does not have execute_tool method")
                return {
                    'success': False,
                    'error': 'MCPHub.execute_tool not available',
                    'data': None
                }
        except Exception as e:
            logger.error(f"MCP stability service error: {e}")
            # 에러 발생 시 기존 방식으로 fallback
            if hasattr(mcp_hub, 'execute_tool'):
                return await mcp_hub.execute_tool(tool_name, params)
            else:
                return {
                    'success': False,
                    'error': str(e),
                    'data': None
                }


"""Tool Governor for Agent Harness.

모든 도구(MCP, 로컬 도구) 호출은 이 거버넌스 계층을 통과해야 합니다.
호출 빈도 제한, 권한 제어, 서킷 브레이커, 결과 정제 기능을 제공합니다.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple

from src.core.mcp_integration import execute_tool as mcp_execute_tool
from src.core.error_handler import get_error_handler

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """도구 연속 실패 시 일시 차단하는 서킷 브레이커"""
    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}
        
    def record_failure(self, tool_name: str) -> None:
        self.failures[tool_name] = self.failures.get(tool_name, 0) + 1
        self.last_failure_time[tool_name] = time.time()
        
    def record_success(self, tool_name: str) -> None:
        if tool_name in self.failures:
            self.failures[tool_name] = 0
            
    def is_open(self, tool_name: str) -> bool:
        """서킷 브레이커가 열려있는지(호출 차단 상태인지) 확인"""
        failures = self.failures.get(tool_name, 0)
        if failures >= self.failure_threshold:
            last_time = self.last_failure_time.get(tool_name, 0)
            if time.time() - last_time < self.reset_timeout:
                return True
            else:
                # 타임아웃 지났으면 반개방(Half-Open) 상태로 간주, 기회 제공
                self.failures[tool_name] = self.failure_threshold - 1 
        return False


class ToolGovernor:
    """도구 실행 제어기"""
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
        self.rate_limits: Dict[str, float] = {}  # 도구별 호출 최소 간격 (초)
        self.last_called: Dict[str, float] = {}
        self.error_handler = get_error_handler()
        
        # 시스템 기본 정책
        # 검색 도구는 1초 제한, 파일/코드 도구는 0.5초 제한
        self.default_rate_limit = 0.5
        
    def _check_rate_limit(self, tool_name: str) -> Tuple[bool, float]:
        """속도 제한 확인. 반환: (통과여부, 대기해야할시간)"""
        limit = self.rate_limits.get(tool_name, self.default_rate_limit)
        last_time = self.last_called.get(tool_name, 0)
        elapsed = time.time() - last_time
        
        if elapsed < limit:
            return False, limit - elapsed
        return True, 0.0
        
    def _sanitize_result(self, result: Dict[str, Any], max_len: int = 50000) -> Dict[str, Any]:
        """결과 크기 제한 및 정제"""
        if not result or not isinstance(result, dict):
            return result
            
        # 결과가 너무 길면 자르기 (데이터 오염, 프롬프트 초과 방지)
        sanitized = dict(result)
        if 'data' in sanitized:
            data_str = str(sanitized['data'])
            if len(data_str) > max_len:
                sanitized['data'] = data_str[:max_len] + f"\n... [결과가 너무 길어 {max_len}자로 잘렸습니다.]"
                logger.warning(f"ToolGovernor: Result truncated from {len(data_str)} to {max_len} chars.")
                
        return sanitized

    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        agent_id: str = "system"
    ) -> Dict[str, Any]:
        """
        거버넌스 정책이 적용된 도구 실행
        """
        # 1. Circuit Breaker 확인
        if self.circuit_breaker.is_open(tool_name):
            logger.warning(f"ToolGovernor: Circuit breaker open for {tool_name}. Blocked.")
            return {
                "success": False,
                "error": f"Circuit breaker open. '{tool_name}' failed too many times recently.",
                "data": None
            }
            
        # 2. Rate Limit 조절
        import asyncio
        passed, wait_time = self._check_rate_limit(tool_name)
        if not passed:
            logger.debug(f"ToolGovernor: Rate limit triggered for {tool_name}. Waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            
        # 3. Trust Gate / 권한 확인 (향후 구현 시 확장 지점)
        # TrustGate.evaluate()를 래핑할 수 있습니다.
        
        # 4. 실행 기록 (호출 전)
        self.last_called[tool_name] = time.time()
        start_time = time.time()
        
        # 5. 실제 도구 실행 (에러 복구 포함)
        try:
            raw_result = await mcp_execute_tool(tool_name, parameters)
            
            if raw_result and raw_result.get("success", False):
                self.circuit_breaker.record_success(tool_name)
            else:
                self.circuit_breaker.record_failure(tool_name)
                
            # 6. 결과 정제
            sanitized = self._sanitize_result(raw_result)
            
            # 실행 시간 메타데이터 추가
            if isinstance(sanitized, dict):
                sanitized["execution_time"] = time.time() - start_time
                
            return sanitized
            
        except Exception as e:
            logger.error(f"ToolGovernor: Execution of {tool_name} raised exception: {e}")
            self.circuit_breaker.record_failure(tool_name)
            
            # ErrorHandler를 통한 복구 시도
            try:
                recovery_result, recovery_success = await self.error_handler.handle_error(
                    e, mcp_execute_tool, tool_name, parameters
                )
                if recovery_success and recovery_result:
                    logger.info(f"ToolGovernor: Recovered from error in {tool_name}")
                    self.circuit_breaker.record_success(tool_name)
                    return self._sanitize_result(recovery_result)
            except Exception as recovery_error:
                logger.debug(f"ToolGovernor: Recovery failed: {recovery_error}")
                
            return {
                "success": False,
                "error": f"Tool execution error: {str(e)}",
                "data": None,
                "execution_time": time.time() - start_time
            }

# 전역 인스턴스
_governor = None

def get_tool_governor() -> ToolGovernor:
    global _governor
    if _governor is None:
        _governor = ToolGovernor()
    return _governor

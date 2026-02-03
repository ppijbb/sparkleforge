"""
Production-Grade Reliability (혁신 8)

Circuit Breaker, Exponential Backoff, State Persistence, Health Check,
Graceful Degradation, Detailed Logging을 통한 프로덕션급 안정성 보장.
"""

import asyncio
import time
import logging
import json
import redis
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import structlog
from tenacity import (
    retry, stop_after_attempt, wait_exponential, wait_fixed,
    retry_if_exception_type, retry_if_not_exception_type,
    Retrying, RetryError
)

from src.core.researcher_config import get_reliability_config

logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit Breaker 상태."""
    CLOSED = "closed"      # 정상 동작
    OPEN = "open"          # 차단됨
    HALF_OPEN = "half_open"  # 부분 복구 시도


@dataclass
class CircuitBreakerConfig:
    """Circuit Breaker 설정."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3
    timeout: float = 30.0


@dataclass
class RetryConfig:
    """재시도 설정."""
    max_attempts: int = 5  # Increased from 3 to 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


@dataclass
class HealthStatus:
    """헬스 상태."""
    component: str
    status: str  # "healthy", "degraded", "critical", "down"
    last_check: datetime
    error_count: int = 0
    success_count: int = 0
    avg_response_time: float = 0.0
    details: Dict[str, Any] = None


class CircuitBreaker:
    """Circuit Breaker 패턴 구현."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
    def can_execute(self) -> bool:
        """실행 가능 여부 확인."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # 복구 시간 체크
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self):
        """성공 기록."""
        self.success_count += 1
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} closed after recovery")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        """실패 기록."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.name} opened due to failures")
    
    def get_state(self) -> Dict[str, Any]:
        """상태 정보 반환."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time
        }


class StateManager:
    """상태 지속성 관리자."""
    
    def __init__(self, backend: str = "redis", ttl: int = 3600):
        self.backend = backend
        self.ttl = ttl
        self.redis_client = None
        
        if backend == "redis":
            try:
                self.redis_client = redis.Redis(
                    host="localhost",
                    port=6379,
                    db=0,
                    decode_responses=True
                )
                # 연결 테스트
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using memory storage")
                self.backend = "memory"
                self.memory_storage = {}
    
    async def save_state(self, key: str, state: Dict[str, Any]) -> str:
        """상태 저장."""
        state_id = f"{key}_{int(time.time())}"
        state_data = {
            "id": state_id,
            "timestamp": time.time(),
            "data": state
        }
        
        if self.backend == "redis" and self.redis_client:
            try:
                self.redis_client.setex(
                    state_id,
                    self.ttl,
                    json.dumps(state_data)
                )
            except Exception as e:
                logger.error(f"Failed to save state to Redis: {e}")
                return None
        else:
            # 메모리 저장
            self.memory_storage[state_id] = state_data
        
        logger.info(f"State saved: {state_id}")
        return state_id
    
    async def load_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """상태 로드."""
        if self.backend == "redis" and self.redis_client:
            try:
                data = self.redis_client.get(state_id)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Failed to load state from Redis: {e}")
        else:
            # 메모리에서 로드
            return self.memory_storage.get(state_id)
        
        return None
    
    async def delete_state(self, state_id: str) -> bool:
        """상태 삭제."""
        if self.backend == "redis" and self.redis_client:
            try:
                return bool(self.redis_client.delete(state_id))
            except Exception as e:
                logger.error(f"Failed to delete state from Redis: {e}")
                return False
        else:
            # 메모리에서 삭제
            return self.memory_storage.pop(state_id, None) is not None


class HealthMonitor:
    """헬스 모니터링."""
    
    def __init__(self):
        self.components: Dict[str, HealthStatus] = {}
        self.check_interval = 30  # 30초마다 체크
        self.monitoring_task = None
    
    def register_component(self, name: str, check_func: Callable[[], bool]):
        """컴포넌트 등록."""
        self.components[name] = HealthStatus(
            component=name,
            status="unknown",
            last_check=datetime.now(),
            details={}
        )
        self.components[name].check_func = check_func
    
    async def start_monitoring(self):
        """모니터링 시작."""
        if self.monitoring_task:
            return
        
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """모니터링 중지."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self):
        """모니터링 루프."""
        while True:
            try:
                await self._check_all_components()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_components(self):
        """모든 컴포넌트 체크."""
        for name, status in self.components.items():
            try:
                start_time = time.time()
                is_healthy = await status.check_func()
                response_time = time.time() - start_time
                
                if is_healthy:
                    status.status = "healthy"
                    status.success_count += 1
                else:
                    status.status = "degraded"
                    status.error_count += 1
                
                status.last_check = datetime.now()
                status.avg_response_time = (
                    (status.avg_response_time * (status.success_count + status.error_count - 1) + response_time) /
                    (status.success_count + status.error_count)
                )
                
            except Exception as e:
                status.status = "critical"
                status.error_count += 1
                status.last_check = datetime.now()
                logger.error(f"Health check failed for {name}: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """헬스 상태 반환."""
        overall_status = "healthy"
        critical_count = 0
        degraded_count = 0
        
        for status in self.components.values():
            if status.status == "critical":
                critical_count += 1
            elif status.status == "degraded":
                degraded_count += 1
        
        if critical_count > 0:
            overall_status = "critical"
        elif degraded_count > len(self.components) * 0.5:
            overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "components": {name: asdict(status) for name, status in self.components.items()},
            "summary": {
                "total_components": len(self.components),
                "healthy": len([s for s in self.components.values() if s.status == "healthy"]),
                "degraded": degraded_count,
                "critical": critical_count
            }
        }


class ProductionReliability:
    """프로덕션급 안정성 관리자."""
    
    def __init__(self):
        self.config = get_reliability_config()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.state_manager = StateManager(
            backend=self.config.state_backend,
            ttl=self.config.state_ttl
        )
        self.health_monitor = HealthMonitor()
        
        # Circuit breaker 설정
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=self.config.failure_threshold,
            recovery_timeout=self.config.recovery_timeout
        )
        
        # 재시도 설정
        self.retry_config = RetryConfig(
            max_attempts=5,  # Increased from 3 to 5
            base_delay=1.0,
            max_delay=60.0
        )
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Circuit breaker 가져오기."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, self.circuit_config)
        return self.circuit_breakers[name]
    
    async def execute_with_reliability(
        self,
        func: Callable,
        *args,
        component_name: str = None,
        save_state: bool = False,
        **kwargs
    ) -> Any:
        """안정성 보장 실행."""
        if not component_name:
            component_name = func.__name__
        
        circuit_breaker = self.get_circuit_breaker(component_name)
        
        # Circuit breaker 체크
        if not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker open for {component_name}")
            if self.config.enable_graceful_degradation:
                return await self._graceful_degradation(func, component_name, *args, **kwargs)
            else:
                raise RuntimeError(f"Circuit breaker open for {component_name}")
        
        # 상태 저장
        state_id = None
        if save_state and self.config.enable_state_persistence:
            state_data = {
                "function": func.__name__,
                "args": args,
                "kwargs": kwargs,
                "timestamp": time.time()
            }
            state_id = await self.state_manager.save_state(component_name, state_data)
        
        start_time = time.time()
        
        try:
            # 재시도 로직과 함께 실행
            result = await self._execute_with_retry(func, *args, **kwargs)
            
            # 성공 기록
            execution_time = time.time() - start_time
            circuit_breaker.record_success()
            
            logger.info(f"Function {func.__name__} executed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            # 실패 기록
            circuit_breaker.record_failure()
            
            logger.error(f"Function {func.__name__} failed: {e}")
            
            # Graceful degradation
            if self.config.enable_graceful_degradation:
                return await self._graceful_degradation(func, component_name, e, *args, **kwargs)
            else:
                raise
    
    async def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """재시도 로직과 함께 실행 - 에러 타입별 전략."""
        if not self.config.enable_exponential_backoff:
            return await func(*args, **kwargs)
        
        # Try to execute and catch exception to determine retry strategy
        last_exception = None
        for attempt in range(self.retry_config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except asyncio.TimeoutError as e:
                last_exception = e
                # TimeoutError: Immediate retry with increased timeout
                if attempt < self.retry_config.max_attempts - 1:
                    wait_time = min(self.retry_config.base_delay * (2 ** attempt), self.retry_config.max_delay)
                    logger.debug(f"TimeoutError on attempt {attempt + 1}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except (ConnectionError, OSError) as e:
                last_exception = e
                # ConnectionError: Exponential backoff
                if attempt < self.retry_config.max_attempts - 1:
                    wait_time = min(
                        self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
                        self.retry_config.max_delay
                    )
                    logger.debug(f"ConnectionError on attempt {attempt + 1}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except ValueError as e:
                last_exception = e
                # ValueError: No retry (validation error)
                logger.warning(f"ValueError (validation error), no retry: {e}")
                raise
            except Exception as e:
                last_exception = e
                # Other exceptions: Standard exponential backoff
                if attempt < self.retry_config.max_attempts - 1:
                    wait_time = min(
                        self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
                        self.retry_config.max_delay
                    )
                    logger.debug(f"Exception on attempt {attempt + 1}, retrying in {wait_time}s: {type(e).__name__}")
                    await asyncio.sleep(wait_time)
                    continue
                raise
        
        # If we get here, all retries exhausted
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry exhausted without exception")
    
    async def _graceful_degradation(
        self,
        func: Callable,
        component_name: str,
        error: Exception,
        *args,
        **kwargs
    ) -> Any:
        """Graceful degradation - 실패 시 명확한 에러 반환 (fallback 데이터 없음)."""
        error_msg = f"Component {component_name} failed: {str(error)}"
        logger.error(error_msg)
        
        # 모든 경우에서 명확한 실패 반환 (더미 데이터 없이)
        raise RuntimeError(error_msg)
    
    async def start_monitoring(self):
        """모니터링 시작."""
        if self.config.enable_health_check:
            await self.health_monitor.start_monitoring()
    
    async def stop_monitoring(self):
        """모니터링 중지."""
        await self.health_monitor.stop_monitoring()
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환."""
        return {
            "circuit_breakers": {
                name: cb.get_state() for name, cb in self.circuit_breakers.items()
            },
            "health_status": self.health_monitor.get_health_status(),
            "config": {
                "circuit_breaker_enabled": self.config.enable_circuit_breaker,
                "exponential_backoff_enabled": self.config.enable_exponential_backoff,
                "state_persistence_enabled": self.config.enable_state_persistence,
                "health_check_enabled": self.config.enable_health_check,
                "graceful_degradation_enabled": self.config.enable_graceful_degradation
            }
        }


# Global reliability manager (lazy initialization)
_reliability_manager = None

def get_reliability_manager() -> 'ProductionReliability':
    """Get or initialize global reliability manager."""
    global _reliability_manager
    if _reliability_manager is None:
        _reliability_manager = ProductionReliability()
    return _reliability_manager


async def execute_with_reliability(
    func: Callable,
    *args,
    component_name: str = None,
    save_state: bool = False,
    **kwargs
) -> Any:
    """안정성 보장 실행."""
    reliability_manager = get_reliability_manager()
    return await reliability_manager.execute_with_reliability(
        func, *args, component_name=component_name, save_state=save_state, **kwargs
    )


async def get_system_status() -> Dict[str, Any]:
    """시스템 상태 반환."""
    reliability_manager = get_reliability_manager()
    return reliability_manager.get_system_status()


async def start_reliability_monitoring():
    """안정성 모니터링 시작."""
    reliability_manager = get_reliability_manager()
    await reliability_manager.start_monitoring()


async def stop_reliability_monitoring():
    """안정성 모니터링 중지."""
    global _reliability_manager
    if _reliability_manager is not None:
        await _reliability_manager.stop_monitoring()

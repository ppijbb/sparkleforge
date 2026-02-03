"""
Long-Running Session Manager (24/7 연중무휴 지원)

장기 세션 유지, 자동 상태 저장, 메모리 관리, 자동 재시작, 헬스체크 및 복구 기능 제공.
"""

import asyncio
import gc
import logging
import psutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import signal
import sys
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SessionHealth:
    """세션 헬스 상태."""
    session_id: str
    is_healthy: bool
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    last_activity: datetime
    error_count: int = 0
    warning_count: int = 0
    auto_save_count: int = 0
    last_save_time: Optional[datetime] = None


@dataclass
class LongRunningConfig:
    """장기 실행 설정."""
    auto_save_interval: int = 300  # 5분마다 자동 저장
    health_check_interval: int = 60  # 1분마다 헬스체크
    memory_cleanup_interval: int = 600  # 10분마다 메모리 정리
    max_memory_mb: float = 2048  # 최대 메모리 사용량 (2GB)
    max_uptime_hours: float = 24  # 최대 실행 시간 (24시간, 0이면 무제한)
    auto_restart_on_error: bool = True  # 에러 시 자동 재시작
    auto_restart_on_memory_limit: bool = True  # 메모리 한계 시 자동 재시작
    gc_threshold: tuple = (700, 10, 10)  # GC 임계값


class LongRunningSessionManager:
    """24/7 연중무휴 장기 세션 관리자."""
    
    def __init__(
        self,
        session_id: str,
        config: Optional[LongRunningConfig] = None,
        save_callback: Optional[Callable[[str], None]] = None,
        restore_callback: Optional[Callable[[str], Dict[str, Any]]] = None
    ):
        """
        초기화.
        
        Args:
            session_id: 세션 ID
            config: 장기 실행 설정
            save_callback: 상태 저장 콜백 함수
            restore_callback: 상태 복원 콜백 함수
        """
        self.session_id = session_id
        self.config = config or LongRunningConfig()
        self.save_callback = save_callback
        self.restore_callback = restore_callback
        
        self.start_time = time.time()
        self.last_activity = datetime.now()
        self.is_running = False
        self._tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # 헬스 상태
        self.health = SessionHealth(
            session_id=session_id,
            is_healthy=True,
            uptime_seconds=0.0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            last_activity=self.last_activity
        )
        
        # 메트릭 히스토리
        self.metrics_history: deque = deque(maxlen=1000)
        
        # 프로세스 정보
        self.process = psutil.Process()
        
        logger.info(f"LongRunningSessionManager initialized: {session_id}")
    
    async def start(self):
        """장기 세션 시작."""
        if self.is_running:
            logger.warning("Session manager already running")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # 백그라운드 작업 시작
        self._tasks = [
            asyncio.create_task(self._auto_save_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._memory_cleanup_loop()),
            asyncio.create_task(self._uptime_monitor_loop())
        ]
        
        logger.info(f"✅ Long-running session started: {self.session_id}")
    
    async def stop(self):
        """장기 세션 중지."""
        if not self.is_running:
            return
        
        logger.info("Stopping long-running session manager...")
        self.is_running = False
        
        # 최종 상태 저장
        await self.save_state()
        
        # 백그라운드 작업 중지
        self._shutdown_event.set()
        for task in self._tasks:
            task.cancel()
        
        # 작업 완료 대기
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info("Long-running session manager stopped")
    
    async def _auto_save_loop(self):
        """자동 상태 저장 루프."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.auto_save_interval)
                
                if not self.is_running:
                    break
                
                await self.save_state()
                self.health.auto_save_count += 1
                self.health.last_save_time = datetime.now()
                
                logger.debug(f"Auto-saved session state: {self.session_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-save loop: {e}", exc_info=True)
                self.health.error_count += 1
    
    async def _health_check_loop(self):
        """헬스체크 루프."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if not self.is_running:
                    break
                
                await self._perform_health_check()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}", exc_info=True)
                self.health.error_count += 1
    
    async def _memory_cleanup_loop(self):
        """메모리 정리 루프."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.memory_cleanup_interval)
                
                if not self.is_running:
                    break
                
                await self._cleanup_memory()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory cleanup loop: {e}", exc_info=True)
    
    async def _uptime_monitor_loop(self):
        """실행 시간 모니터링 루프."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 1분마다 체크
                
                if not self.is_running:
                    break
                
                uptime_hours = (time.time() - self.start_time) / 3600
                
                # 최대 실행 시간 체크
                if self.config.max_uptime_hours > 0 and uptime_hours >= self.config.max_uptime_hours:
                    logger.info(f"Max uptime reached ({uptime_hours:.1f}h), initiating graceful restart...")
                    await self._graceful_restart()
                    break
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in uptime monitor loop: {e}", exc_info=True)
    
    async def _perform_health_check(self):
        """헬스체크 수행."""
        try:
            # 메모리 사용량
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # CPU 사용량
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            # 실행 시간
            uptime = time.time() - self.start_time
            
            # 헬스 상태 업데이트
            self.health.uptime_seconds = uptime
            self.health.memory_usage_mb = memory_mb
            self.health.cpu_usage_percent = cpu_percent
            self.health.last_activity = self.last_activity
            
            # 메모리 한계 체크
            if memory_mb > self.config.max_memory_mb:
                self.health.is_healthy = False
                self.health.warning_count += 1
                logger.warning(
                    f"Memory limit exceeded: {memory_mb:.1f}MB > {self.config.max_memory_mb}MB"
                )
                
                if self.config.auto_restart_on_memory_limit:
                    logger.info("Auto-restarting due to memory limit...")
                    await self._graceful_restart()
                    return
            
            # CPU 사용량 체크 (90% 이상이면 경고)
            if cpu_percent > 90:
                self.health.warning_count += 1
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            # 메트릭 저장
            self.metrics_history.append({
                'timestamp': datetime.now().isoformat(),
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'uptime_seconds': uptime
            })
            
            # 헬스 상태 로깅 (10분마다)
            if int(uptime) % 600 == 0:
                logger.info(
                    f"Session health: {memory_mb:.1f}MB memory, "
                    f"{cpu_percent:.1f}% CPU, {uptime/3600:.1f}h uptime"
                )
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}", exc_info=True)
            self.health.error_count += 1
            self.health.is_healthy = False
    
    async def _cleanup_memory(self):
        """메모리 정리."""
        try:
            # GC 실행
            collected = gc.collect()
            
            # 메트릭 히스토리 정리 (오래된 항목 제거)
            if len(self.metrics_history) > 500:
                # 최근 500개만 유지
                items_to_remove = len(self.metrics_history) - 500
                for _ in range(items_to_remove):
                    self.metrics_history.popleft()
            
            logger.debug(f"Memory cleanup: GC collected {collected} objects")
            
        except Exception as e:
            logger.error(f"Error in memory cleanup: {e}", exc_info=True)
    
    async def _graceful_restart(self):
        """우아한 재시작."""
        logger.info("Initiating graceful restart...")
        
        # 상태 저장
        await self.save_state()
        
        # 재시작 정보 저장
        restart_info = {
            'session_id': self.session_id,
            'restart_time': datetime.now().isoformat(),
            'reason': 'scheduled' if self.config.max_uptime_hours > 0 else 'memory_limit',
            'uptime_hours': (time.time() - self.start_time) / 3600
        }
        
        restart_file = Path(f"./storage/restart_{self.session_id}.json")
        restart_file.parent.mkdir(parents=True, exist_ok=True)
        restart_file.write_text(json.dumps(restart_info, indent=2), encoding='utf-8')
        
        # 세션 중지
        await self.stop()
        
        # 재시작 스크립트 실행 (또는 외부 프로세스 관리자에게 신호)
        logger.info("Restart info saved. Use external process manager (systemd, supervisor, etc.) for auto-restart.")
    
    async def save_state(self):
        """상태 저장."""
        if not self.save_callback:
            return
        
        try:
            state = {
                'session_id': self.session_id,
                'start_time': self.start_time,
                'last_activity': self.last_activity.isoformat(),
                'health': {
                    'uptime_seconds': self.health.uptime_seconds,
                    'memory_usage_mb': self.health.memory_usage_mb,
                    'cpu_usage_percent': self.health.cpu_usage_percent,
                    'error_count': self.health.error_count,
                    'warning_count': self.health.warning_count,
                    'auto_save_count': self.health.auto_save_count
                }
            }
            
            if asyncio.iscoroutinefunction(self.save_callback):
                await self.save_callback(json.dumps(state))
            else:
                self.save_callback(json.dumps(state))
            
        except Exception as e:
            logger.error(f"Error saving state: {e}", exc_info=True)
    
    def update_activity(self):
        """활동 시간 업데이트."""
        self.last_activity = datetime.now()
        self.health.last_activity = self.last_activity
    
    def get_health(self) -> SessionHealth:
        """헬스 상태 반환."""
        # 최신 정보 업데이트
        self.health.uptime_seconds = time.time() - self.start_time
        try:
            memory_info = self.process.memory_info()
            self.health.memory_usage_mb = memory_info.rss / 1024 / 1024
            self.health.cpu_usage_percent = self.process.cpu_percent(interval=0.1)
        except Exception:
            pass
        
        return self.health
    
    def get_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """메트릭 히스토리 반환."""
        return list(self.metrics_history)[-limit:]


class LongRunningService:
    """24/7 연중무휴 서비스 래퍼."""
    
    def __init__(self, config: Optional[LongRunningConfig] = None):
        """초기화."""
        self.config = config or LongRunningConfig()
        self.session_managers: Dict[str, LongRunningSessionManager] = {}
        self._main_task: Optional[asyncio.Task] = None
    
    async def start_service(
        self,
        service_name: str,
        service_func: Callable,
        session_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        """
        장기 실행 서비스 시작.
        
        Args:
            service_name: 서비스 이름
            service_func: 서비스 함수 (async)
            session_id: 세션 ID (None이면 자동 생성)
            *args, **kwargs: 서비스 함수 인자
        """
        if session_id is None:
            session_id = f"{service_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 세션 관리자 생성
        session_manager = LongRunningSessionManager(
            session_id=session_id,
            config=self.config,
            save_callback=self._create_save_callback(session_id),
            restore_callback=self._create_restore_callback(session_id)
        )
        
        self.session_managers[session_id] = session_manager
        
        # 세션 관리자 시작
        await session_manager.start()
        
        # 서비스 실행 (에러 시 자동 재시작)
        while True:
            try:
                logger.info(f"Starting service: {service_name} (session: {session_id})")
                await service_func(*args, **kwargs)
                break  # 정상 종료
            except Exception as e:
                logger.error(f"Service error: {e}", exc_info=True)
                session_manager.health.error_count += 1
                
                if self.config.auto_restart_on_error:
                    logger.info("Auto-restarting service after error...")
                    await asyncio.sleep(5)  # 5초 대기 후 재시작
                    continue
                else:
                    raise
            finally:
                await session_manager.stop()
    
    def _create_save_callback(self, session_id: str) -> Callable:
        """상태 저장 콜백 생성."""
        async def save_callback(state_json: str):
            from src.core.checkpoint_manager import CheckpointManager
            checkpoint_manager = CheckpointManager()
            await checkpoint_manager.save_checkpoint(
                state=json.loads(state_json),
                metadata={"type": "long_running", "session_id": session_id}
            )
        return save_callback
    
    def _create_restore_callback(self, session_id: str) -> Callable:
        """상태 복원 콜백 생성."""
        async def restore_callback(checkpoint_id: str) -> Dict[str, Any]:
            from src.core.checkpoint_manager import CheckpointManager
            checkpoint_manager = CheckpointManager()
            restored = await checkpoint_manager.restore_checkpoint(checkpoint_id)
            return restored or {}
        return restore_callback


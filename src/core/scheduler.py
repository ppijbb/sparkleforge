"""
Scheduler System (주기적 세션 실행)

Cron 표현식 기반 스케줄링, 주기적 세션 실행, 스케줄 관리 기능 제공.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import re

try:
    from croniter import croniter
    CRONITER_AVAILABLE = True
except ImportError:
    CRONITER_AVAILABLE = False
    logging.warning("croniter not available. Install with: pip install croniter")

logger = logging.getLogger(__name__)


class ScheduleStatus(Enum):
    """스케줄 상태."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class ScheduleConfig:
    """스케줄 설정."""
    schedule_id: str
    name: str
    cron_expression: str  # Cron 표현식 (예: "0 9 * * *" = 매일 오전 9시)
    user_query: str  # 실행할 쿼리
    enabled: bool = True
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    max_runs: Optional[int] = None  # 최대 실행 횟수 (None이면 무제한)
    timeout_seconds: Optional[int] = None  # 타임아웃 (초)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.last_run:
            data['last_run'] = self.last_run.isoformat()
        if self.next_run:
            data['next_run'] = self.next_run.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduleConfig':
        """딕셔너리에서 생성."""
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('last_run'), str):
            data['last_run'] = datetime.fromisoformat(data['last_run'])
        if isinstance(data.get('next_run'), str):
            data['next_run'] = datetime.fromisoformat(data['next_run'])
        if isinstance(data.get('status'), str):
            data['status'] = ScheduleStatus(data['status'])
        return cls(**data)


@dataclass
class ScheduleExecution:
    """스케줄 실행 기록."""
    execution_id: str
    schedule_id: str
    session_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, cancelled
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


class Scheduler:
    """스케줄 관리 및 실행 시스템."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        초기화.
        
        Args:
            storage_path: 스케줄 저장 경로 (None이면 기본 경로)
        """
        if storage_path is None:
            storage_path = Path.home() / ".sparkleforge" / "schedules"
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.schedules_file = self.storage_path / "schedules.json"
        self.executions_file = self.storage_path / "executions.json"
        
        # 메모리 내 스케줄 저장소
        self.schedules: Dict[str, ScheduleConfig] = {}
        self.executions: List[ScheduleExecution] = []
        
        # 실행 중인 작업 추적
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # 스케줄러 태스크
        self.scheduler_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # 실행 콜백
        self.execution_callback: Optional[Callable[[str, str], Any]] = None
        
        # 스케줄 로드
        self._load_schedules()
        self._load_executions()
        
        logger.info(f"Scheduler initialized: {len(self.schedules)} schedules loaded")
    
    def _load_schedules(self):
        """스케줄 로드."""
        if not self.schedules_file.exists():
            return
        
        try:
            data = json.loads(self.schedules_file.read_text(encoding='utf-8'))
            for schedule_data in data.get('schedules', []):
                schedule = ScheduleConfig.from_dict(schedule_data)
                self.schedules[schedule.schedule_id] = schedule
        except Exception as e:
            logger.error(f"Failed to load schedules: {e}")
    
    def _save_schedules(self):
        """스케줄 저장."""
        try:
            data = {
                'schedules': [s.to_dict() for s in self.schedules.values()],
                'updated_at': datetime.now().isoformat()
            }
            self.schedules_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"Failed to save schedules: {e}")
    
    def _load_executions(self):
        """실행 기록 로드."""
        if not self.executions_file.exists():
            return
        
        try:
            data = json.loads(self.executions_file.read_text(encoding='utf-8'))
            # 최근 1000개만 메모리에 로드
            for exec_data in data.get('executions', [])[-1000:]:
                execution = ScheduleExecution(
                    execution_id=exec_data['execution_id'],
                    schedule_id=exec_data['schedule_id'],
                    session_id=exec_data['session_id'],
                    started_at=datetime.fromisoformat(exec_data['started_at']),
                    completed_at=datetime.fromisoformat(exec_data['completed_at']) if exec_data.get('completed_at') else None,
                    status=exec_data.get('status', 'completed'),
                    result=exec_data.get('result'),
                    error=exec_data.get('error'),
                    duration_seconds=exec_data.get('duration_seconds')
                )
                self.executions.append(execution)
        except Exception as e:
            logger.error(f"Failed to load executions: {e}")
    
    def _save_executions(self):
        """실행 기록 저장."""
        try:
            # 최근 1000개만 저장
            recent_executions = self.executions[-1000:]
            data = {
                'executions': [
                    {
                        'execution_id': e.execution_id,
                        'schedule_id': e.schedule_id,
                        'session_id': e.session_id,
                        'started_at': e.started_at.isoformat(),
                        'completed_at': e.completed_at.isoformat() if e.completed_at else None,
                        'status': e.status,
                        'result': e.result,
                        'error': e.error,
                        'duration_seconds': e.duration_seconds
                    }
                    for e in recent_executions
                ],
                'updated_at': datetime.now().isoformat()
            }
            self.executions_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"Failed to save executions: {e}")
    
    def create_schedule(
        self,
        name: str,
        cron_expression: str,
        user_query: str,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        max_runs: Optional[int] = None,
        timeout_seconds: Optional[int] = None
    ) -> ScheduleConfig:
        """
        스케줄 생성.
        
        Args:
            name: 스케줄 이름
            cron_expression: Cron 표현식
            user_query: 실행할 쿼리
            enabled: 활성화 여부
            metadata: 추가 메타데이터
            tags: 태그
            max_runs: 최대 실행 횟수
            timeout_seconds: 타임아웃 (초)
        
        Returns:
            ScheduleConfig
        """
        # Cron 표현식 검증
        if not self._validate_cron_expression(cron_expression):
            raise ValueError(f"Invalid cron expression: {cron_expression}")
        
        schedule_id = f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 다음 실행 시간 계산
        next_run = self._calculate_next_run(cron_expression)
        
        schedule = ScheduleConfig(
            schedule_id=schedule_id,
            name=name,
            cron_expression=cron_expression,
            user_query=user_query,
            enabled=enabled,
            status=ScheduleStatus.ACTIVE if enabled else ScheduleStatus.DISABLED,
            next_run=next_run,
            metadata=metadata or {},
            tags=tags or [],
            max_runs=max_runs,
            timeout_seconds=timeout_seconds
        )
        
        self.schedules[schedule_id] = schedule
        self._save_schedules()
        
        logger.info(f"Schedule created: {schedule_id} ({name})")
        return schedule
    
    def _validate_cron_expression(self, cron_expr: str) -> bool:
        """Cron 표현식 검증."""
        if not CRONITER_AVAILABLE:
            # 간단한 검증 (5개 필드)
            parts = cron_expr.split()
            return len(parts) == 5
        
        try:
            croniter(cron_expr)
            return True
        except Exception:
            return False
    
    def _calculate_next_run(self, cron_expr: str, base_time: Optional[datetime] = None) -> datetime:
        """다음 실행 시간 계산."""
        if base_time is None:
            base_time = datetime.now()
        
        if not CRONITER_AVAILABLE:
            # 간단한 구현 (매일 정시 실행 가정)
            # 실제로는 croniter를 사용하는 것이 좋습니다
            return base_time + timedelta(days=1)
        
        try:
            cron = croniter(cron_expr, base_time)
            return cron.get_next(datetime)
        except Exception as e:
            logger.error(f"Failed to calculate next run: {e}")
            return base_time + timedelta(days=1)
    
    def get_schedule(self, schedule_id: str) -> Optional[ScheduleConfig]:
        """스케줄 조회."""
        return self.schedules.get(schedule_id)
    
    def list_schedules(
        self,
        enabled_only: bool = False,
        status: Optional[ScheduleStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[ScheduleConfig]:
        """
        스케줄 목록 조회.
        
        Args:
            enabled_only: 활성화된 것만
            status: 상태 필터
            tags: 태그 필터
        
        Returns:
            스케줄 리스트
        """
        schedules = list(self.schedules.values())
        
        if enabled_only:
            schedules = [s for s in schedules if s.enabled]
        
        if status:
            schedules = [s for s in schedules if s.status == status]
        
        if tags:
            schedules = [s for s in schedules if any(tag in s.tags for tag in tags)]
        
        return sorted(schedules, key=lambda x: x.created_at, reverse=True)
    
    def update_schedule(
        self,
        schedule_id: str,
        name: Optional[str] = None,
        cron_expression: Optional[str] = None,
        user_query: Optional[str] = None,
        enabled: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        스케줄 업데이트.
        
        Args:
            schedule_id: 스케줄 ID
            name: 이름
            cron_expression: Cron 표현식
            user_query: 쿼리
            enabled: 활성화 여부
            metadata: 메타데이터
            tags: 태그
        
        Returns:
            성공 여부
        """
        if schedule_id not in self.schedules:
            return False
        
        schedule = self.schedules[schedule_id]
        
        if name is not None:
            schedule.name = name
        
        if cron_expression is not None:
            if not self._validate_cron_expression(cron_expression):
                return False
            schedule.cron_expression = cron_expression
            schedule.next_run = self._calculate_next_run(cron_expression)
        
        if user_query is not None:
            schedule.user_query = user_query
        
        if enabled is not None:
            schedule.enabled = enabled
            schedule.status = ScheduleStatus.ACTIVE if enabled else ScheduleStatus.DISABLED
        
        if metadata is not None:
            schedule.metadata.update(metadata)
        
        if tags is not None:
            schedule.tags = tags
        
        self._save_schedules()
        logger.info(f"Schedule updated: {schedule_id}")
        return True
    
    def delete_schedule(self, schedule_id: str) -> bool:
        """
        스케줄 삭제.
        
        Args:
            schedule_id: 스케줄 ID
        
        Returns:
            성공 여부
        """
        if schedule_id not in self.schedules:
            return False
        
        # 실행 중인 작업 취소
        if schedule_id in self.running_tasks:
            self.running_tasks[schedule_id].cancel()
            del self.running_tasks[schedule_id]
        
        del self.schedules[schedule_id]
        self._save_schedules()
        
        logger.info(f"Schedule deleted: {schedule_id}")
        return True
    
    def pause_schedule(self, schedule_id: str) -> bool:
        """스케줄 일시 중지."""
        if schedule_id not in self.schedules:
            return False
        
        schedule = self.schedules[schedule_id]
        schedule.status = ScheduleStatus.PAUSED
        schedule.enabled = False
        self._save_schedules()
        
        logger.info(f"Schedule paused: {schedule_id}")
        return True
    
    def resume_schedule(self, schedule_id: str) -> bool:
        """스케줄 재개."""
        if schedule_id not in self.schedules:
            return False
        
        schedule = self.schedules[schedule_id]
        schedule.status = ScheduleStatus.ACTIVE
        schedule.enabled = True
        schedule.next_run = self._calculate_next_run(schedule.cron_expression)
        self._save_schedules()
        
        logger.info(f"Schedule resumed: {schedule_id}")
        return True
    
    def set_execution_callback(self, callback: Callable[[str, str], Any]):
        """
        실행 콜백 설정.
        
        Args:
            callback: (user_query, session_id) -> result
        """
        self.execution_callback = callback
    
    async def start(self):
        """스케줄러 시작."""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        self.is_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Scheduler started")
    
    async def stop(self):
        """스케줄러 중지."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # 실행 중인 작업 대기
        for task in self.running_tasks.values():
            task.cancel()
        
        await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        logger.info("Scheduler stopped")
    
    async def _scheduler_loop(self):
        """스케줄러 루프."""
        while self.is_running:
            try:
                now = datetime.now()
                
                # 실행할 스케줄 찾기
                for schedule in self.schedules.values():
                    if not schedule.enabled or schedule.status != ScheduleStatus.ACTIVE:
                        continue
                    
                    # 최대 실행 횟수 체크
                    if schedule.max_runs and schedule.run_count >= schedule.max_runs:
                        schedule.status = ScheduleStatus.COMPLETED
                        schedule.enabled = False
                        self._save_schedules()
                        continue
                    
                    # 실행 시간 체크
                    if schedule.next_run and schedule.next_run <= now:
                        # 실행
                        asyncio.create_task(self._execute_schedule(schedule))
                
                # 1분마다 체크
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _execute_schedule(self, schedule: ScheduleConfig):
        """스케줄 실행."""
        if schedule.schedule_id in self.running_tasks:
            logger.warning(f"Schedule already running: {schedule.schedule_id}")
            return
        
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        execution = ScheduleExecution(
            execution_id=execution_id,
            schedule_id=schedule.schedule_id,
            session_id=session_id,
            started_at=datetime.now()
        )
        self.executions.append(execution)
        
        async def run_with_timeout():
            try:
                if self.execution_callback:
                    result = await self.execution_callback(schedule.user_query, session_id)
                    execution.status = "completed"
                    execution.result = result
                else:
                    logger.warning("No execution callback set")
                    execution.status = "failed"
                    execution.error = "No execution callback"
            except asyncio.TimeoutError:
                execution.status = "failed"
                execution.error = "Timeout"
            except Exception as e:
                execution.status = "failed"
                execution.error = str(e)
                logger.error(f"Schedule execution failed: {e}", exc_info=True)
            finally:
                execution.completed_at = datetime.now()
                if execution.started_at:
                    execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
                
                # 스케줄 통계 업데이트
                schedule.last_run = datetime.now()
                schedule.run_count += 1
                if execution.status == "completed":
                    schedule.success_count += 1
                else:
                    schedule.failure_count += 1
                
                # 다음 실행 시간 계산
                schedule.next_run = self._calculate_next_run(schedule.cron_expression)
                
                self._save_schedules()
                self._save_executions()
                
                if schedule.schedule_id in self.running_tasks:
                    del self.running_tasks[schedule.schedule_id]
        
        # 타임아웃 설정
        if schedule.timeout_seconds:
            task = asyncio.create_task(asyncio.wait_for(
                run_with_timeout(),
                timeout=schedule.timeout_seconds
            ))
        else:
            task = asyncio.create_task(run_with_timeout())
        
        self.running_tasks[schedule.schedule_id] = task
        
        try:
            await task
        except asyncio.CancelledError:
            execution.status = "cancelled"
            execution.completed_at = datetime.now()
            self._save_executions()
        except Exception as e:
            logger.error(f"Error executing schedule: {e}", exc_info=True)
    
    def get_execution_history(
        self,
        schedule_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ScheduleExecution]:
        """
        실행 기록 조회.
        
        Args:
            schedule_id: 스케줄 ID 필터
            limit: 최대 결과 수
        
        Returns:
            실행 기록 리스트
        """
        executions = self.executions
        
        if schedule_id:
            executions = [e for e in executions if e.schedule_id == schedule_id]
        
        # 최신순 정렬
        executions.sort(key=lambda x: x.started_at, reverse=True)
        
        return executions[:limit]
    
    def get_schedule_statistics(self) -> Dict[str, Any]:
        """스케줄 통계 조회."""
        total = len(self.schedules)
        active = len([s for s in self.schedules.values() if s.status == ScheduleStatus.ACTIVE])
        paused = len([s for s in self.schedules.values() if s.status == ScheduleStatus.PAUSED])
        disabled = len([s for s in self.schedules.values() if s.status == ScheduleStatus.DISABLED])
        
        total_runs = sum(s.run_count for s in self.schedules.values())
        total_success = sum(s.success_count for s in self.schedules.values())
        total_failure = sum(s.failure_count for s in self.schedules.values())
        
        return {
            'total_schedules': total,
            'active_schedules': active,
            'paused_schedules': paused,
            'disabled_schedules': disabled,
            'total_runs': total_runs,
            'total_success': total_success,
            'total_failure': total_failure,
            'success_rate': total_success / total_runs if total_runs > 0 else 0.0
        }


# 전역 인스턴스
_scheduler: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    """전역 스케줄러 인스턴스 반환."""
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler()
    return _scheduler


"""
Guardian Agent - Self-Healing Monitoring System

Hephaestus 영감을 받은 자가 치유 모니터링 에이전트.
에이전트 상태를 모니터링하고, stuck 감지, 자동 복구 수행.

핵심 특징:
- Agent health monitoring
- Stuck detection and recovery
- Automatic task reassignment
- System-wide coordination
- Anomaly detection
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentHealthStatus(Enum):
    """에이전트 건강 상태."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STUCK = "stuck"
    UNRESPONSIVE = "unresponsive"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """복구 조치."""
    NONE = "none"
    RETRY = "retry"
    REASSIGN = "reassign"
    RESTART = "restart"
    ESCALATE = "escalate"
    TERMINATE = "terminate"


class AgentHealthMetrics(BaseModel):
    """에이전트 건강 지표."""
    agent_id: str
    status: AgentHealthStatus = AgentHealthStatus.UNKNOWN
    
    # 성능 지표
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_response_time: float = 0.0
    last_activity: Optional[datetime] = None
    
    # 리소스 지표
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # 오류 지표
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    error_rate: float = 0.0
    
    # Stuck 감지
    idle_time_seconds: float = 0.0
    stuck_threshold_seconds: float = 300.0  # 5분
    
    class Config:
        arbitrary_types_allowed = True


class HealthCheckResult(BaseModel):
    """헬스 체크 결과."""
    agent_id: str
    status: AgentHealthStatus
    timestamp: datetime = Field(default_factory=datetime.now)
    metrics: AgentHealthMetrics
    recommended_action: RecoveryAction = RecoveryAction.NONE
    details: str = ""


class RecoveryEvent(BaseModel):
    """복구 이벤트."""
    event_id: str
    agent_id: str
    action: RecoveryAction
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = False
    details: str = ""
    duration_seconds: float = 0.0


@dataclass
class AgentTrajectory:
    """에이전트 실행 궤적 (stuck 감지용)."""
    agent_id: str
    checkpoints: deque = field(default_factory=lambda: deque(maxlen=20))
    last_progress_time: datetime = field(default_factory=datetime.now)
    
    def add_checkpoint(self, state_hash: str, output_length: int):
        """체크포인트 추가."""
        self.checkpoints.append({
            "state_hash": state_hash,
            "output_length": output_length,
            "timestamp": datetime.now()
        })
        
        # 진행 여부 확인
        if len(self.checkpoints) >= 2:
            prev = self.checkpoints[-2]
            curr = self.checkpoints[-1]
            
            # 상태 변화 감지
            if prev["state_hash"] != curr["state_hash"] or \
               prev["output_length"] != curr["output_length"]:
                self.last_progress_time = datetime.now()
    
    def get_idle_time(self) -> float:
        """비활성 시간 (초)."""
        return (datetime.now() - self.last_progress_time).total_seconds()
    
    def is_stuck(self, threshold_seconds: float = 300.0) -> bool:
        """stuck 여부 확인."""
        return self.get_idle_time() > threshold_seconds


class GuardianAgent:
    """
    Guardian Agent - 자가 치유 모니터링 에이전트.
    
    모든 에이전트를 모니터링하고 문제 발생 시 자동 복구.
    """
    
    def __init__(
        self,
        check_interval_seconds: float = 10.0,
        stuck_threshold_seconds: float = 300.0,
        max_consecutive_failures: int = 3,
        auto_recovery: bool = True
    ):
        self.check_interval = check_interval_seconds
        self.stuck_threshold = stuck_threshold_seconds
        self.max_consecutive_failures = max_consecutive_failures
        self.auto_recovery = auto_recovery
        
        # 에이전트 추적
        self.agent_metrics: Dict[str, AgentHealthMetrics] = {}
        self.agent_trajectories: Dict[str, AgentTrajectory] = {}
        
        # 복구 이력
        self.recovery_events: List[RecoveryEvent] = []
        
        # 콜백
        self.on_health_change: Optional[Callable] = None
        self.on_recovery_start: Optional[Callable] = None
        self.on_recovery_complete: Optional[Callable] = None
        
        # 모니터링 상태
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"GuardianAgent initialized: interval={check_interval_seconds}s, "
            f"stuck_threshold={stuck_threshold_seconds}s, auto_recovery={auto_recovery}"
        )
    
    def register_agent(self, agent_id: str, initial_metrics: Optional[Dict[str, Any]] = None):
        """에이전트 등록."""
        if agent_id in self.agent_metrics:
            logger.warning(f"Agent {agent_id} already registered")
            return
        
        metrics = AgentHealthMetrics(
            agent_id=agent_id,
            status=AgentHealthStatus.HEALTHY,
            last_activity=datetime.now(),
            stuck_threshold_seconds=self.stuck_threshold
        )
        
        if initial_metrics:
            for key, value in initial_metrics.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
        
        self.agent_metrics[agent_id] = metrics
        self.agent_trajectories[agent_id] = AgentTrajectory(agent_id=agent_id)
        
        logger.info(f"Registered agent: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """에이전트 등록 해제."""
        self.agent_metrics.pop(agent_id, None)
        self.agent_trajectories.pop(agent_id, None)
        logger.info(f"Unregistered agent: {agent_id}")
    
    def report_activity(
        self,
        agent_id: str,
        state_hash: Optional[str] = None,
        output_length: int = 0,
        task_completed: bool = False,
        task_failed: bool = False,
        error: Optional[str] = None,
        response_time: Optional[float] = None
    ):
        """에이전트 활동 보고."""
        if agent_id not in self.agent_metrics:
            self.register_agent(agent_id)
        
        metrics = self.agent_metrics[agent_id]
        trajectory = self.agent_trajectories[agent_id]
        
        # 활동 시간 업데이트
        metrics.last_activity = datetime.now()
        
        # 궤적 업데이트
        if state_hash:
            trajectory.add_checkpoint(state_hash, output_length)
        
        # 태스크 완료/실패 카운트
        if task_completed:
            metrics.tasks_completed += 1
            metrics.consecutive_failures = 0
        
        if task_failed:
            metrics.tasks_failed += 1
            metrics.consecutive_failures += 1
            metrics.last_error = error
        
        # 응답 시간 업데이트 (이동 평균)
        if response_time is not None:
            if metrics.avg_response_time == 0:
                metrics.avg_response_time = response_time
            else:
                metrics.avg_response_time = (
                    0.9 * metrics.avg_response_time + 0.1 * response_time
                )
        
        # 오류율 계산
        total_tasks = metrics.tasks_completed + metrics.tasks_failed
        if total_tasks > 0:
            metrics.error_rate = metrics.tasks_failed / total_tasks
        
        # 상태 업데이트
        self._update_agent_status(agent_id)
    
    def _update_agent_status(self, agent_id: str):
        """에이전트 상태 업데이트."""
        metrics = self.agent_metrics[agent_id]
        trajectory = self.agent_trajectories.get(agent_id)
        
        old_status = metrics.status
        
        # Stuck 체크
        if trajectory and trajectory.is_stuck(self.stuck_threshold):
            metrics.status = AgentHealthStatus.STUCK
            metrics.idle_time_seconds = trajectory.get_idle_time()
        
        # 연속 실패 체크
        elif metrics.consecutive_failures >= self.max_consecutive_failures:
            metrics.status = AgentHealthStatus.FAILED
        
        # 비응답 체크 (마지막 활동이 stuck_threshold * 2 이상 전)
        elif metrics.last_activity:
            idle_seconds = (datetime.now() - metrics.last_activity).total_seconds()
            if idle_seconds > self.stuck_threshold * 2:
                metrics.status = AgentHealthStatus.UNRESPONSIVE
        
        # 성능 저하 체크
        elif metrics.error_rate > 0.3 or metrics.avg_response_time > 60:
            metrics.status = AgentHealthStatus.DEGRADED
        
        # 정상
        else:
            metrics.status = AgentHealthStatus.HEALTHY
        
        # 상태 변경 시 콜백
        if old_status != metrics.status and self.on_health_change:
            asyncio.create_task(
                self._safe_callback(
                    self.on_health_change,
                    agent_id,
                    old_status,
                    metrics.status
                )
            )
    
    async def check_health(self, agent_id: str) -> HealthCheckResult:
        """개별 에이전트 헬스 체크."""
        if agent_id not in self.agent_metrics:
            return HealthCheckResult(
                agent_id=agent_id,
                status=AgentHealthStatus.UNKNOWN,
                metrics=AgentHealthMetrics(agent_id=agent_id),
                details="Agent not registered"
            )
        
        metrics = self.agent_metrics[agent_id]
        self._update_agent_status(agent_id)
        
        # 권장 복구 조치 결정
        action = self._determine_recovery_action(metrics)
        
        return HealthCheckResult(
            agent_id=agent_id,
            status=metrics.status,
            metrics=metrics,
            recommended_action=action,
            details=self._get_health_details(metrics)
        )
    
    def _determine_recovery_action(self, metrics: AgentHealthMetrics) -> RecoveryAction:
        """복구 조치 결정."""
        if metrics.status == AgentHealthStatus.HEALTHY:
            return RecoveryAction.NONE
        
        if metrics.status == AgentHealthStatus.DEGRADED:
            return RecoveryAction.RETRY
        
        if metrics.status == AgentHealthStatus.STUCK:
            if metrics.idle_time_seconds < self.stuck_threshold * 1.5:
                return RecoveryAction.RETRY
            else:
                return RecoveryAction.REASSIGN
        
        if metrics.status == AgentHealthStatus.FAILED:
            if metrics.consecutive_failures < self.max_consecutive_failures * 2:
                return RecoveryAction.RESTART
            else:
                return RecoveryAction.ESCALATE
        
        if metrics.status == AgentHealthStatus.UNRESPONSIVE:
            return RecoveryAction.RESTART
        
        return RecoveryAction.NONE
    
    def _get_health_details(self, metrics: AgentHealthMetrics) -> str:
        """상세 상태 메시지 생성."""
        details = []
        
        if metrics.status == AgentHealthStatus.HEALTHY:
            details.append("Agent is operating normally")
        
        if metrics.status == AgentHealthStatus.STUCK:
            details.append(f"Agent stuck for {metrics.idle_time_seconds:.0f}s")
        
        if metrics.consecutive_failures > 0:
            details.append(f"{metrics.consecutive_failures} consecutive failures")
        
        if metrics.error_rate > 0.1:
            details.append(f"Error rate: {metrics.error_rate:.1%}")
        
        if metrics.last_error:
            details.append(f"Last error: {metrics.last_error[:100]}")
        
        return "; ".join(details) if details else "No issues detected"
    
    async def recover(
        self,
        agent_id: str,
        action: RecoveryAction,
        recovery_handler: Optional[Callable] = None
    ) -> RecoveryEvent:
        """복구 수행."""
        import uuid
        
        event = RecoveryEvent(
            event_id=str(uuid.uuid4())[:8],
            agent_id=agent_id,
            action=action
        )
        
        start_time = datetime.now()
        
        # 콜백
        if self.on_recovery_start:
            await self._safe_callback(self.on_recovery_start, agent_id, action)
        
        try:
            logger.info(f"Starting recovery for {agent_id}: {action.value}")
            
            if action == RecoveryAction.NONE:
                event.success = True
                event.details = "No action needed"
            
            elif action == RecoveryAction.RETRY:
                # 재시도는 외부에서 처리
                event.success = True
                event.details = "Retry requested"
                
                # 궤적 리셋
                if agent_id in self.agent_trajectories:
                    self.agent_trajectories[agent_id] = AgentTrajectory(agent_id=agent_id)
            
            elif action == RecoveryAction.REASSIGN:
                if recovery_handler:
                    result = await recovery_handler("reassign", agent_id)
                    event.success = result.get("success", False)
                    event.details = result.get("message", "Reassigned")
                else:
                    event.details = "Reassignment handler not provided"
            
            elif action == RecoveryAction.RESTART:
                if recovery_handler:
                    result = await recovery_handler("restart", agent_id)
                    event.success = result.get("success", False)
                    event.details = result.get("message", "Restarted")
                else:
                    # 기본 리셋
                    self._reset_agent(agent_id)
                    event.success = True
                    event.details = "Agent metrics reset"
            
            elif action == RecoveryAction.ESCALATE:
                event.details = "Escalated to system administrator"
                logger.warning(f"Agent {agent_id} requires manual intervention")
            
            elif action == RecoveryAction.TERMINATE:
                self.unregister_agent(agent_id)
                event.success = True
                event.details = "Agent terminated"
            
        except Exception as e:
            event.success = False
            event.details = f"Recovery failed: {str(e)}"
            logger.error(f"Recovery failed for {agent_id}: {e}")
        
        finally:
            event.duration_seconds = (datetime.now() - start_time).total_seconds()
            self.recovery_events.append(event)
            
            # 콜백
            if self.on_recovery_complete:
                await self._safe_callback(self.on_recovery_complete, event)
        
        return event
    
    def _reset_agent(self, agent_id: str):
        """에이전트 메트릭 리셋."""
        if agent_id in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentHealthMetrics(
                agent_id=agent_id,
                status=AgentHealthStatus.HEALTHY,
                last_activity=datetime.now()
            )
        
        if agent_id in self.agent_trajectories:
            self.agent_trajectories[agent_id] = AgentTrajectory(agent_id=agent_id)
    
    async def start_monitoring(self, recovery_handler: Optional[Callable] = None):
        """모니터링 시작."""
        if self._monitoring:
            logger.warning("Monitoring already running")
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitoring_loop(recovery_handler)
        )
        logger.info("Guardian monitoring started")
    
    async def stop_monitoring(self):
        """모니터링 중지."""
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Guardian monitoring stopped")
    
    async def _monitoring_loop(self, recovery_handler: Optional[Callable] = None):
        """모니터링 루프."""
        while self._monitoring:
            try:
                for agent_id in list(self.agent_metrics.keys()):
                    # 헬스 체크
                    result = await self.check_health(agent_id)
                    
                    # 자동 복구
                    if self.auto_recovery and result.recommended_action != RecoveryAction.NONE:
                        await self.recover(
                            agent_id,
                            result.recommended_action,
                            recovery_handler
                        )
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 전체 건강 상태."""
        total_agents = len(self.agent_metrics)
        
        if total_agents == 0:
            return {
                "status": "unknown",
                "message": "No agents registered",
                "agents": {}
            }
        
        status_counts = {status: 0 for status in AgentHealthStatus}
        for metrics in self.agent_metrics.values():
            status_counts[metrics.status] += 1
        
        # 시스템 상태 결정
        if status_counts[AgentHealthStatus.HEALTHY] == total_agents:
            system_status = "healthy"
        elif status_counts[AgentHealthStatus.FAILED] > 0 or \
             status_counts[AgentHealthStatus.UNRESPONSIVE] > 0:
            system_status = "critical"
        elif status_counts[AgentHealthStatus.STUCK] > 0 or \
             status_counts[AgentHealthStatus.DEGRADED] > 0:
            system_status = "degraded"
        else:
            system_status = "unknown"
        
        return {
            "status": system_status,
            "total_agents": total_agents,
            "status_breakdown": {
                status.value: count 
                for status, count in status_counts.items()
                if count > 0
            },
            "recent_recoveries": len([
                e for e in self.recovery_events
                if (datetime.now() - e.timestamp).total_seconds() < 3600
            ]),
            "agents": {
                agent_id: {
                    "status": metrics.status.value,
                    "tasks_completed": metrics.tasks_completed,
                    "error_rate": metrics.error_rate,
                    "idle_time": metrics.idle_time_seconds
                }
                for agent_id, metrics in self.agent_metrics.items()
            }
        }
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """안전한 콜백 실행."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Callback error (non-fatal): {e}")


# Singleton instance
_guardian_agent: Optional[GuardianAgent] = None


def get_guardian_agent(
    check_interval_seconds: float = 10.0,
    stuck_threshold_seconds: float = 300.0,
    auto_recovery: bool = True
) -> GuardianAgent:
    """GuardianAgent 싱글톤 인스턴스 반환."""
    global _guardian_agent
    
    if _guardian_agent is None:
        _guardian_agent = GuardianAgent(
            check_interval_seconds=check_interval_seconds,
            stuck_threshold_seconds=stuck_threshold_seconds,
            auto_recovery=auto_recovery
        )
    
    return _guardian_agent

"""
실시간 진행 상황 추적 시스템

워크플로우 단계별 진행, 에이전트 실행 상태, 예상 완료 시간,
중간 결과 미리보기를 제공하는 진행 상황 추적 시스템
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict


class WorkflowStage(Enum):
    """워크플로우 단계."""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentStatus(Enum):
    """에이전트 상태."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentProgress:
    """에이전트 진행 상황."""
    agent_id: str
    agent_type: str
    status: AgentStatus = AgentStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: float = 0.0  # 0.0 to 1.0
    current_task: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """실행 시간 (초)."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return None

    @property
    def is_active(self) -> bool:
        """활성 상태인지 확인."""
        return self.status in [AgentStatus.RUNNING, AgentStatus.PENDING]


@dataclass
class WorkflowProgress:
    """워크플로우 진행 상황."""
    session_id: str
    current_stage: WorkflowStage = WorkflowStage.INITIALIZING
    overall_progress: float = 0.0  # 0.0 to 1.0
    start_time: Optional[float] = None
    estimated_completion: Optional[float] = None
    agents: Dict[str, AgentProgress] = field(default_factory=dict)
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """총 실행 시간 (초)."""
        if self.start_time:
            return time.time() - self.start_time
        return None

    @property
    def active_agents(self) -> List[AgentProgress]:
        """활성 에이전트 목록."""
        return [agent for agent in self.agents.values() if agent.is_active]

    @property
    def completed_agents(self) -> List[AgentProgress]:
        """완료된 에이전트 목록."""
        return [agent for agent in self.agents.values() if agent.status == AgentStatus.COMPLETED]

    @property
    def failed_agents(self) -> List[AgentProgress]:
        """실패한 에이전트 목록."""
        return [agent for agent in self.agents.values() if agent.status == AgentStatus.FAILED]


class ProgressTracker:
    """
    실시간 진행 상황 추적 시스템.

    워크플로우 및 에이전트별 진행 상황을 추적하고 실시간 업데이트를 제공.
    """

    def __init__(
        self,
        session_id: str,
        enable_real_time_updates: bool = True,
        update_interval: float = 1.0
    ):
        """초기화."""
        self.session_id = session_id
        self.enable_real_time_updates = enable_real_time_updates
        self.update_interval = max(0.5, update_interval)  # 최소 0.5초 간격으로 제한

        # 진행 상황 데이터
        self.workflow_progress = WorkflowProgress(session_id=session_id)
        self.workflow_progress.start_time = time.time()

        # 콜백 함수들
        self.progress_callbacks: List[Callable] = []
        self.stage_change_callbacks: List[Callable] = []
        self.agent_status_callbacks: List[Callable] = []

        # 업데이트 태스크
        self.update_task: Optional[asyncio.Task] = None
        self.running = False

        # 잠금
        self._lock = asyncio.Lock()

        # 통계
        self.stats = {
            'total_agents_created': 0,
            'agents_completed': 0,
            'agents_failed': 0,
            'stage_transitions': 0,
            'last_update': time.time()
        }

    async def start_tracking(self):
        """진행 상황 추적 시작."""
        async with self._lock:
            self.running = True
            if self.enable_real_time_updates:
                self.update_task = asyncio.create_task(self._periodic_update())

    async def stop_tracking(self):
        """진행 상황 추적 중지."""
        async with self._lock:
            self.running = False
            if self.update_task:
                self.update_task.cancel()
                try:
                    await self.update_task
                except asyncio.CancelledError:
                    pass

    async def _periodic_update(self):
        """주기적 업데이트."""
        while self.running:
            try:
                await self._update_progress()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # 업데이트 실패 시 로깅만 하고 계속 진행
                try:
                    from src.utils.output_manager import get_output_manager, OutputLevel
                    output_manager = get_output_manager()
                    await output_manager.output(
                        f"진행 상황 업데이트 실패: {e}",
                        level=OutputLevel.DEBUG
                    )
                except:
                    pass  # 출력 매니저도 실패하면 무시

    async def _update_progress(self):
        """진행 상황 업데이트."""
        current_time = time.time()

        # 전체 진행률 계산
        self.workflow_progress.overall_progress = self._calculate_overall_progress()

        # 예상 완료 시간 업데이트
        self.workflow_progress.estimated_completion = self._estimate_completion_time()

        # 콜백 호출
        await self._trigger_callbacks()

        self.stats['last_update'] = current_time

    def _calculate_overall_progress(self) -> float:
        """전체 진행률 계산."""
        # 단계별 가중치
        stage_weights = {
            WorkflowStage.INITIALIZING: 0.05,
            WorkflowStage.PLANNING: 0.15,
            WorkflowStage.EXECUTING: 0.40,
            WorkflowStage.VERIFYING: 0.25,
            WorkflowStage.GENERATING: 0.10,
            WorkflowStage.COMPLETED: 1.0,
            WorkflowStage.FAILED: 1.0
        }

        base_progress = stage_weights.get(self.workflow_progress.current_stage, 0.0)
        
        # 에이전트가 없어도 단계별 기본 진행률은 반환
        if not self.workflow_progress.agents:
            # 초기화 단계에서는 시간 기반으로 약간의 진행률 표시
            if self.workflow_progress.current_stage == WorkflowStage.INITIALIZING:
                if self.workflow_progress.start_time:
                    elapsed = time.time() - self.workflow_progress.start_time
                    # 초기화는 최대 5초로 가정하고, 그 동안 1-5% 진행률 표시
                    init_progress = min(0.05, 0.01 + (elapsed / 5.0) * 0.04)
                    return init_progress
            return base_progress

        # 현재 단계 내 세부 진행률
        if self.workflow_progress.current_stage in [WorkflowStage.EXECUTING, WorkflowStage.VERIFYING]:
            agents_in_stage = [
                agent for agent in self.workflow_progress.agents.values()
                if self._is_agent_in_current_stage(agent)
            ]

            if agents_in_stage:
                stage_progress = sum(agent.progress for agent in agents_in_stage) / len(agents_in_stage)
                # 단계 내 진행률을 전체 진행률에 반영
                stage_weight = stage_weights[self.workflow_progress.current_stage]
                additional_progress = stage_progress * stage_weight
                base_progress += additional_progress
        elif self.workflow_progress.current_stage == WorkflowStage.PLANNING:
            # Planning 단계에서는 에이전트 진행률 반영
            planning_agents = [
                agent for agent in self.workflow_progress.agents.values()
                if 'planner' in agent.agent_type.lower()
            ]
            if planning_agents:
                stage_progress = sum(agent.progress for agent in planning_agents) / len(planning_agents)
                base_progress += stage_progress * 0.15
        elif self.workflow_progress.current_stage == WorkflowStage.GENERATING:
            # Generating 단계에서는 에이전트 진행률 반영
            generating_agents = [
                agent for agent in self.workflow_progress.agents.values()
                if 'generator' in agent.agent_type.lower()
            ]
            if generating_agents:
                stage_progress = sum(agent.progress for agent in generating_agents) / len(generating_agents)
                base_progress += stage_progress * 0.10

        return min(base_progress, 1.0)

    def _is_agent_in_current_stage(self, agent: AgentProgress) -> bool:
        """에이전트가 현재 단계에 속하는지 확인."""
        stage_agent_types = {
            WorkflowStage.PLANNING: ['planner'],
            WorkflowStage.EXECUTING: ['executor', 'parallel_executor'],
            WorkflowStage.VERIFYING: ['verifier', 'parallel_verifier'],
            WorkflowStage.GENERATING: ['generator']
        }

        current_stage_types = stage_agent_types.get(self.workflow_progress.current_stage, [])
        return any(agent_type in agent.agent_type.lower() for agent_type in current_stage_types)

    def _estimate_completion_time(self) -> Optional[float]:
        """예상 완료 시간 추정."""
        if not self.workflow_progress.start_time:
            return None

        elapsed = time.time() - self.workflow_progress.start_time

        if self.workflow_progress.overall_progress > 0:
            total_estimated = elapsed / self.workflow_progress.overall_progress
            remaining = total_estimated - elapsed
            return time.time() + remaining

        return None

    async def _trigger_callbacks(self):
        """콜백 함수들 호출."""
        # 진행 상황 콜백
        for callback in self.progress_callbacks:
            try:
                await callback(self.workflow_progress)
            except Exception as e:
                # 콜백 실패는 로깅만 하고 계속 진행
                try:
                    from src.utils.output_manager import get_output_manager, OutputLevel
                    output_manager = get_output_manager()
                    await output_manager.output(
                        f"진행 상황 콜백 실패: {e}",
                        level=OutputLevel.DEBUG
                    )
                except:
                    pass

    # 에이전트 관리 메서드들
    def register_agent(self, agent_id: str, agent_type: str) -> AgentProgress:
        """새 에이전트 등록."""
        agent = AgentProgress(
            agent_id=agent_id,
            agent_type=agent_type,
            status=AgentStatus.PENDING
        )

        self.workflow_progress.agents[agent_id] = agent
        self.stats['total_agents_created'] += 1

        return agent

    def update_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
        current_task: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """에이전트 상태 업데이트."""
        if agent_id not in self.workflow_progress.agents:
            return

        agent = self.workflow_progress.agents[agent_id]
        old_status = agent.status

        agent.status = status
        agent.current_task = current_task or agent.current_task

        if error_message:
            agent.error_message = error_message

        if metadata:
            agent.metadata.update(metadata)

        # 시간 기록
        if status == AgentStatus.RUNNING and not agent.start_time:
            agent.start_time = time.time()
        elif status in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED]:
            agent.end_time = time.time()

        # 통계 업데이트
        if status == AgentStatus.COMPLETED and old_status != AgentStatus.COMPLETED:
            self.stats['agents_completed'] += 1
        elif status == AgentStatus.FAILED and old_status != AgentStatus.FAILED:
            self.stats['agents_failed'] += 1

        # 에이전트 상태 변경 콜백
        for callback in self.agent_status_callbacks:
            try:
                asyncio.create_task(callback(agent_id, agent))
            except Exception as e:
                # 콜백 실패는 로깅만 하고 계속 진행
                pass

    def update_agent_progress(self, agent_id: str, progress: float, current_task: Optional[str] = None):
        """에이전트 진행률 업데이트."""
        if agent_id not in self.workflow_progress.agents:
            return

        agent = self.workflow_progress.agents[agent_id]
        agent.progress = max(0.0, min(1.0, progress))  # 0.0-1.0 범위 제한

        if current_task:
            agent.current_task = current_task

    # 워크플로우 관리 메서드들
    def set_workflow_stage(self, stage: WorkflowStage, metadata: Optional[Dict[str, Any]] = None):
        """워크플로우 단계 설정."""
        old_stage = self.workflow_progress.current_stage
        self.workflow_progress.current_stage = stage

        # 단계 변경 기록
        stage_record = {
            'stage': stage.value,
            'timestamp': time.time(),
            'previous_stage': old_stage.value if old_stage else None,
            'metadata': metadata or {}
        }
        self.workflow_progress.stage_history.append(stage_record)

        # 메타데이터 업데이트
        if metadata:
            self.workflow_progress.metadata.update(metadata)

        self.stats['stage_transitions'] += 1

        # 단계 변경 콜백
        for callback in self.stage_change_callbacks:
            try:
                asyncio.create_task(callback(old_stage, stage, self.workflow_progress))
            except Exception as e:
                # 콜백 실패는 로깅만 하고 계속 진행
                pass

    # 콜백 등록 메서드들
    def add_progress_callback(self, callback: Callable):
        """진행 상황 콜백 등록."""
        self.progress_callbacks.append(callback)

    def add_stage_change_callback(self, callback: Callable):
        """단계 변경 콜백 등록."""
        self.stage_change_callbacks.append(callback)

    def add_agent_status_callback(self, callback: Callable):
        """에이전트 상태 콜백 등록."""
        self.agent_status_callbacks.append(callback)

    # 정보 조회 메서드들
    def get_workflow_summary(self) -> Dict[str, Any]:
        """워크플로우 요약 정보."""
        return {
            'session_id': self.session_id,
            'current_stage': self.workflow_progress.current_stage.value,
            'overall_progress': self.workflow_progress.overall_progress,
            'duration': self.workflow_progress.duration,
            'estimated_completion': self.workflow_progress.estimated_completion,
            'active_agents': len(self.workflow_progress.active_agents),
            'completed_agents': len(self.workflow_progress.completed_agents),
            'failed_agents': len(self.workflow_progress.failed_agents),
            'total_agents': len(self.workflow_progress.agents)
        }

    def get_agent_summary(self, agent_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """에이전트 요약 정보."""
        if agent_id:
            agent = self.workflow_progress.agents.get(agent_id)
            if not agent:
                return {}
            return {
                'agent_id': agent.agent_id,
                'agent_type': agent.agent_type,
                'status': agent.status.value,
                'progress': agent.progress,
                'duration': agent.duration,
                'current_task': agent.current_task,
                'error_message': agent.error_message
            }
        else:
            return [
                {
                    'agent_id': agent.agent_id,
                    'agent_type': agent.agent_type,
                    'status': agent.status.value,
                    'progress': agent.progress,
                    'duration': agent.duration,
                    'current_task': agent.current_task,
                    'error_message': agent.error_message
                }
                for agent in self.workflow_progress.agents.values()
            ]

    def get_stage_history(self) -> List[Dict[str, Any]]:
        """단계 변경 히스토리."""
        return self.workflow_progress.stage_history.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보."""
        return {
            **self.stats,
            'current_time': time.time(),
            'uptime': time.time() - self.workflow_progress.start_time if self.workflow_progress.start_time else 0
        }


# 전역 진행 추적기 인스턴스
_progress_tracker = None
_current_session_id = None

def get_progress_tracker(session_id: Optional[str] = None) -> ProgressTracker:
    """전역 진행 추적기 인스턴스 반환."""
    global _progress_tracker, _current_session_id

    if session_id != _current_session_id or _progress_tracker is None:
        _progress_tracker = ProgressTracker(session_id or "default")
        _current_session_id = session_id

    return _progress_tracker

def set_progress_tracker(tracker: ProgressTracker):
    """전역 진행 추적기 설정."""
    global _progress_tracker
    _progress_tracker = tracker


# 편의 함수들
def create_agent_progress_context(agent_id: str, agent_type: str):
    """에이전트 진행 상황 컨텍스트 매니저."""
    class AgentProgressContext:
        def __init__(self, agent_id: str, agent_type: str):
            self.agent_id = agent_id
            self.agent_type = agent_type
            self.tracker = get_progress_tracker()

        async def __aenter__(self):
            self.tracker.register_agent(self.agent_id, self.agent_type)
            self.tracker.update_agent_status(
                self.agent_id,
                AgentStatus.RUNNING,
                current_task="Initializing"
            )
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                self.tracker.update_agent_status(
                    self.agent_id,
                    AgentStatus.FAILED,
                    error_message=str(exc_val)
                )
            else:
                self.tracker.update_agent_status(
                    self.agent_id,
                    AgentStatus.COMPLETED
                )

        def update_progress(self, progress: float, task: Optional[str] = None):
            self.tracker.update_agent_progress(self.agent_id, progress, task)

        def update_status(self, status: AgentStatus, task: Optional[str] = None, error: Optional[str] = None):
            self.tracker.update_agent_status(self.agent_id, status, task, error)

    return AgentProgressContext(agent_id, agent_type)

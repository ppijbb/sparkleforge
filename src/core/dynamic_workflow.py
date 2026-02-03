"""
Phase-Based Dynamic Workflow System

Hephaestus 영감을 받은 동적 워크플로우 시스템.
에이전트가 발견에 따라 새 태스크를 동적으로 생성하고,
Phase 기반 실행으로 복잡한 연구 과제를 체계적으로 처리.

핵심 특징:
- Phase-based workflow (Analysis → Implementation → Validation)
- Dynamic task spawning based on discoveries
- Priority bumping for urgent tasks
- Inter-agent communication
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from datetime import datetime
from enum import Enum
from collections import defaultdict
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WorkflowPhase(Enum):
    """워크플로우 단계."""
    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    COMPLETE = "complete"


class TaskPriority(Enum):
    """태스크 우선순위."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(Enum):
    """태스크 상태."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DynamicTask(BaseModel):
    """동적 생성 태스크."""
    task_id: str = Field(description="태스크 고유 ID")
    name: str = Field(description="태스크 이름")
    description: str = Field(description="태스크 설명")
    phase: WorkflowPhase = Field(description="소속 단계")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="우선순위")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="상태")
    
    # 의존성
    depends_on: List[str] = Field(default_factory=list, description="의존 태스크 ID들")
    blocks: List[str] = Field(default_factory=list, description="이 태스크가 블록하는 태스크 ID들")
    
    # 실행 정보
    assigned_agent: Optional[str] = Field(default=None, description="할당된 에이전트")
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    
    # 결과
    result: Optional[Dict[str, Any]] = Field(default=None, description="실행 결과")
    error: Optional[str] = Field(default=None, description="오류 메시지")
    
    # 메타데이터
    spawned_by: Optional[str] = Field(default=None, description="이 태스크를 생성한 태스크 ID")
    spawned_tasks: List[str] = Field(default_factory=list, description="이 태스크가 생성한 태스크 ID들")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")
    
    class Config:
        arbitrary_types_allowed = True


class PhaseTransition(BaseModel):
    """단계 전환 기록."""
    from_phase: WorkflowPhase
    to_phase: WorkflowPhase
    timestamp: datetime = Field(default_factory=datetime.now)
    reason: str = Field(default="")
    completed_tasks: int = Field(default=0)
    pending_tasks: int = Field(default=0)


class DynamicTaskSpawner:
    """
    동적 태스크 생성기.
    
    에이전트가 발견한 내용을 바탕으로 새 태스크를 동적으로 생성.
    """
    
    def __init__(self, max_spawned_per_task: int = 5):
        if max_spawned_per_task <= 0:
            raise ValueError("max_spawned_per_task must be positive")
        self.max_spawned_per_task = max_spawned_per_task
        self.spawn_count: Dict[str, int] = defaultdict(int)
    
    def spawn(
        self,
        parent_task_id: str,
        name: str,
        description: str,
        phase: WorkflowPhase,
        priority: TaskPriority = TaskPriority.MEDIUM,
        depends_on: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[DynamicTask]:
        """
        새 태스크 생성.
        
        Args:
            parent_task_id: 부모 태스크 ID
            name: 태스크 이름
            description: 태스크 설명
            phase: 소속 단계
            priority: 우선순위
            depends_on: 의존 태스크들
            metadata: 추가 메타데이터
            
        Returns:
            생성된 태스크 (제한 초과 시 None)
        """
        # 생성 제한 체크
        if self.spawn_count[parent_task_id] >= self.max_spawned_per_task:
            logger.warning(
                f"Task {parent_task_id} reached spawn limit ({self.max_spawned_per_task})"
            )
            return None
        
        task = DynamicTask(
            task_id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            phase=phase,
            priority=priority,
            depends_on=depends_on or [],
            spawned_by=parent_task_id,
            metadata=metadata or {}
        )
        
        self.spawn_count[parent_task_id] += 1
        
        logger.info(
            f"Spawned task {task.task_id}: {name} (phase={phase.value}, priority={priority.value})"
        )
        
        return task
    
    def reset_count(self, task_id: str):
        """태스크의 생성 카운트 초기화."""
        self.spawn_count[task_id] = 0


class PhaseManager:
    """
    워크플로우 단계 관리자.
    
    Analysis → Implementation → Validation 3단계 워크플로우 관리.
    """
    
    def __init__(self):
        self.current_phase = WorkflowPhase.ANALYSIS
        self.phase_history: List[PhaseTransition] = []
        self.phase_tasks: Dict[WorkflowPhase, List[str]] = {
            phase: [] for phase in WorkflowPhase
        }
        
        # 단계별 완료 조건
        self.phase_completion_threshold = {
            WorkflowPhase.ANALYSIS: 0.9,  # 90% 태스크 완료
            WorkflowPhase.IMPLEMENTATION: 0.95,
            WorkflowPhase.VALIDATION: 1.0
        }
    
    def add_task_to_phase(self, task_id: str, phase: WorkflowPhase):
        """태스크를 단계에 추가."""
        if task_id not in self.phase_tasks[phase]:
            self.phase_tasks[phase].append(task_id)
    
    def can_transition(
        self,
        tasks: Dict[str, DynamicTask],
        to_phase: WorkflowPhase
    ) -> Tuple[bool, str]:
        """
        단계 전환 가능 여부 확인.
        
        Returns:
            (전환 가능 여부, 이유)
        """
        # 순서 체크
        phase_order = [WorkflowPhase.ANALYSIS, WorkflowPhase.IMPLEMENTATION, WorkflowPhase.VALIDATION]
        current_idx = phase_order.index(self.current_phase)
        target_idx = phase_order.index(to_phase)
        
        if target_idx != current_idx + 1:
            return False, f"Invalid transition: {self.current_phase.value} -> {to_phase.value}"
        
        # 현재 단계 완료율 체크
        phase_task_ids = self.phase_tasks[self.current_phase]
        if not phase_task_ids:
            return True, "No tasks in current phase"
        
        completed = sum(
            1 for tid in phase_task_ids
            if tid in tasks and tasks[tid].status == TaskStatus.COMPLETED
        )
        completion_rate = completed / len(phase_task_ids)
        threshold = self.phase_completion_threshold[self.current_phase]
        
        if completion_rate >= threshold:
            return True, f"Completion rate {completion_rate:.0%} >= {threshold:.0%}"
        else:
            return False, f"Completion rate {completion_rate:.0%} < {threshold:.0%}"
    
    def transition(
        self,
        to_phase: WorkflowPhase,
        tasks: Dict[str, DynamicTask],
        reason: str = ""
    ) -> bool:
        """단계 전환 실행."""
        can_transition, msg = self.can_transition(tasks, to_phase)
        
        if not can_transition:
            logger.warning(f"Cannot transition: {msg}")
            return False
        
        # 전환 기록
        phase_task_ids = self.phase_tasks[self.current_phase]
        completed = sum(
            1 for tid in phase_task_ids
            if tid in tasks and tasks[tid].status == TaskStatus.COMPLETED
        )
        pending = len(phase_task_ids) - completed
        
        transition = PhaseTransition(
            from_phase=self.current_phase,
            to_phase=to_phase,
            reason=reason or msg,
            completed_tasks=completed,
            pending_tasks=pending
        )
        self.phase_history.append(transition)
        
        # 전환 실행
        old_phase = self.current_phase
        self.current_phase = to_phase
        
        logger.info(f"Phase transition: {old_phase.value} -> {to_phase.value}")
        
        return True
    
    def get_phase_statistics(
        self,
        tasks: Dict[str, DynamicTask]
    ) -> Dict[str, Any]:
        """단계별 통계 반환."""
        stats = {}
        
        for phase in WorkflowPhase:
            if phase == WorkflowPhase.COMPLETE:
                continue
                
            phase_task_ids = self.phase_tasks[phase]
            if not phase_task_ids:
                stats[phase.value] = {
                    "total": 0,
                    "completed": 0,
                    "in_progress": 0,
                    "failed": 0,
                    "completion_rate": 1.0
                }
                continue
            
            completed = sum(
                1 for tid in phase_task_ids
                if tid in tasks and tasks[tid].status == TaskStatus.COMPLETED
            )
            in_progress = sum(
                1 for tid in phase_task_ids
                if tid in tasks and tasks[tid].status == TaskStatus.IN_PROGRESS
            )
            failed = sum(
                1 for tid in phase_task_ids
                if tid in tasks and tasks[tid].status == TaskStatus.FAILED
            )
            
            stats[phase.value] = {
                "total": len(phase_task_ids),
                "completed": completed,
                "in_progress": in_progress,
                "failed": failed,
                "completion_rate": completed / len(phase_task_ids)
            }
        
        return stats


class TaskQueue:
    """
    우선순위 기반 태스크 큐.
    """
    
    def __init__(self):
        self.tasks: Dict[str, DynamicTask] = {}
        self.priority_queues: Dict[TaskPriority, List[str]] = {
            priority: [] for priority in TaskPriority
        }
    
    def add(self, task: DynamicTask):
        """태스크 추가."""
        self.tasks[task.task_id] = task
        self.priority_queues[task.priority].append(task.task_id)
    
    def get_next(self, phase: Optional[WorkflowPhase] = None) -> Optional[DynamicTask]:
        """
        다음 실행할 태스크 반환.
        
        Args:
            phase: 특정 단계의 태스크만 반환 (선택)
        """
        # Cache priorities list for better performance
        if not hasattr(self, '_sorted_priorities'):
            self._sorted_priorities = sorted(TaskPriority, key=lambda p: p.value)
        
        for priority in self._sorted_priorities:
            queue = self.priority_queues.get(priority, [])
            if not queue:
                continue
                
            for task_id in queue:
                task = self.tasks.get(task_id)
                if not task:
                    continue
                
                # 상태 체크
                if task.status != TaskStatus.PENDING:
                    continue
                
                # 단계 체크
                if phase and task.phase != phase:
                    continue
                
                # 의존성 체크
                if self._has_unmet_dependencies(task):
                    continue
                
                return task
        
        return None
    
    def _has_unmet_dependencies(self, task: DynamicTask) -> bool:
        """미충족 의존성 확인."""
        for dep_id in task.depends_on:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return True
        return False
    
    def bump_priority(self, task_id: str, new_priority: TaskPriority):
        """태스크 우선순위 상향."""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        old_priority = task.priority
        if new_priority.value >= old_priority.value:
            return  # 같거나 낮은 우선순위로는 변경 안 함
        
        # 기존 큐에서 제거
        if task_id in self.priority_queues[old_priority]:
            self.priority_queues[old_priority].remove(task_id)
        
        # 새 큐에 추가
        self.priority_queues[new_priority].insert(0, task_id)
        task.priority = new_priority
        
        logger.info(f"Priority bumped for {task_id}: {old_priority.value} -> {new_priority.value}")
    
    def get_pending_count(self, phase: Optional[WorkflowPhase] = None) -> int:
        """대기 중인 태스크 수."""
        count = 0
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                if phase is None or task.phase == phase:
                    count += 1
        return count
    
    def complete_task(self, task_id: str, result: Dict[str, Any]):
        """태스크 완료 처리."""
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
    
    def fail_task(self, task_id: str, error: str):
        """태스크 실패 처리."""
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = error


# Tuple already imported above


class DynamicWorkflowEngine:
    """
    동적 워크플로우 엔진.
    
    Phase 기반 워크플로우 + 동적 태스크 생성을 통합 관리.
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 5,
        auto_phase_transition: bool = True
    ):
        if max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        
        self.task_spawner = DynamicTaskSpawner()
        self.phase_manager = PhaseManager()
        self.task_queue = TaskQueue()
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.auto_phase_transition = auto_phase_transition
        
        # 실행 중인 태스크
        self.running_tasks: Set[str] = set()
        
        # 콜백
        self.on_task_complete: Optional[Callable] = None
        self.on_phase_change: Optional[Callable] = None
        self.on_task_spawned: Optional[Callable] = None
        
        logger.info(
            f"DynamicWorkflowEngine initialized: "
            f"max_concurrent={max_concurrent_tasks}, auto_transition={auto_phase_transition}"
        )
    
    async def run(
        self,
        initial_tasks: List[DynamicTask],
        task_executor: Callable
    ) -> Dict[str, Any]:
        """
        워크플로우 실행.
        
        Args:
            initial_tasks: 초기 태스크들
            task_executor: 태스크 실행 함수 (async)
            
        Returns:
            실행 결과 요약
        """
        # 초기 태스크 등록
        for task in initial_tasks:
            self.task_queue.add(task)
            self.phase_manager.add_task_to_phase(task.task_id, task.phase)
        
        logger.info(f"Starting workflow with {len(initial_tasks)} initial tasks")
        
        while True:
            # 현재 단계에서 다음 태스크 가져오기
            current_phase = self.phase_manager.current_phase
            
            if current_phase == WorkflowPhase.COMPLETE:
                break
            
            # 동시 실행 제한 체크
            while len(self.running_tasks) < self.max_concurrent_tasks:
                task = self.task_queue.get_next(phase=current_phase)
                
                if not task:
                    break
                
                # 태스크 실행 시작
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now()
                self.running_tasks.add(task.task_id)
                
                # 비동기 실행
                asyncio.create_task(
                    self._execute_task(task, task_executor)
                )
            
            # 실행 중인 태스크가 없고 더 이상 대기 태스크도 없으면
            if not self.running_tasks:
                pending = self.task_queue.get_pending_count(phase=current_phase)
                
                if pending == 0:
                    # 단계 전환 시도
                    if self.auto_phase_transition:
                        next_phase = self._get_next_phase(current_phase)
                        if next_phase:
                            success = self.phase_manager.transition(
                                next_phase,
                                self.task_queue.tasks,
                                reason="All tasks completed"
                            )
                            if success and self.on_phase_change:
                                await self._safe_callback(
                                    self.on_phase_change,
                                    current_phase,
                                    next_phase
                                )
                        else:
                            # 마지막 단계 완료
                            self.phase_manager.current_phase = WorkflowPhase.COMPLETE
                            break
                    else:
                        break
            
            # 잠시 대기
            await asyncio.sleep(0.1)
        
        # 결과 수집
        return self._collect_results()
    
    async def _execute_task(
        self,
        task: DynamicTask,
        executor: Callable
    ):
        """개별 태스크 실행."""
        try:
            logger.info(f"Executing task {task.task_id}: {task.name}")
            
            # 실행
            result = await executor(task)
            
            # 완료 처리
            self.task_queue.complete_task(task.task_id, result)
            
            # 콜백
            if self.on_task_complete:
                await self._safe_callback(self.on_task_complete, task, result)
            
            # 동적 태스크 처리
            if result and "spawn_tasks" in result:
                for spawn_info in result["spawn_tasks"]:
                    new_task = self.task_spawner.spawn(
                        parent_task_id=task.task_id,
                        name=spawn_info.get("name", "Spawned Task"),
                        description=spawn_info.get("description", ""),
                        phase=WorkflowPhase(spawn_info.get("phase", task.phase.value)),
                        priority=TaskPriority(spawn_info.get("priority", TaskPriority.MEDIUM.value)),
                        metadata=spawn_info.get("metadata", {})
                    )
                    
                    if new_task:
                        self.task_queue.add(new_task)
                        self.phase_manager.add_task_to_phase(new_task.task_id, new_task.phase)
                        task.spawned_tasks.append(new_task.task_id)
                        
                        if self.on_task_spawned:
                            await self._safe_callback(self.on_task_spawned, task, new_task)
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            self.task_queue.fail_task(task.task_id, str(e))
        
        finally:
            self.running_tasks.discard(task.task_id)
    
    def _get_next_phase(self, current: WorkflowPhase) -> Optional[WorkflowPhase]:
        """다음 단계 반환."""
        phase_order = [
            WorkflowPhase.ANALYSIS,
            WorkflowPhase.IMPLEMENTATION,
            WorkflowPhase.VALIDATION
        ]
        
        try:
            idx = phase_order.index(current)
            if idx < len(phase_order) - 1:
                return phase_order[idx + 1]
        except ValueError:
            pass
        
        return None
    
    def _collect_results(self) -> Dict[str, Any]:
        """실행 결과 수집."""
        tasks = self.task_queue.tasks
        
        completed = [t for t in tasks.values() if t.status == TaskStatus.COMPLETED]
        failed = [t for t in tasks.values() if t.status == TaskStatus.FAILED]
        
        return {
            "total_tasks": len(tasks),
            "completed": len(completed),
            "failed": len(failed),
            "phase_statistics": self.phase_manager.get_phase_statistics(tasks),
            "phase_history": [
                {
                    "from": t.from_phase.value,
                    "to": t.to_phase.value,
                    "timestamp": t.timestamp.isoformat(),
                    "reason": t.reason
                }
                for t in self.phase_manager.phase_history
            ],
            "task_results": {
                t.task_id: {
                    "name": t.name,
                    "phase": t.phase.value,
                    "status": t.status.value,
                    "result": t.result,
                    "error": t.error,
                    "spawned_tasks": t.spawned_tasks
                }
                for t in tasks.values()
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
_dynamic_workflow_engine: Optional[DynamicWorkflowEngine] = None


def get_dynamic_workflow_engine(
    max_concurrent_tasks: int = 5,
    auto_phase_transition: bool = True
) -> DynamicWorkflowEngine:
    """DynamicWorkflowEngine 싱글톤 인스턴스 반환."""
    global _dynamic_workflow_engine
    
    if _dynamic_workflow_engine is None:
        _dynamic_workflow_engine = DynamicWorkflowEngine(
            max_concurrent_tasks=max_concurrent_tasks,
            auto_phase_transition=auto_phase_transition
        )
    
    return _dynamic_workflow_engine

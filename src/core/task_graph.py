"""통합 Task Graph Management

이 모듈은 `task_queue.py`의 TopicBlock과 `dynamic_workflow.py`의 DynamicTask를 통합하여
단일 모델(UnifiedTask)로 관리하고, DAG 기반 의존성 검사 및 우선순위 스케줄링을 제공합니다.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class UnifiedTask:
    """통합된 단일 타스크 인스턴스"""

    task_id: str
    description: str
    phase: str = "execution"  # analysis | execution | verification
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | ready | running | completed | failed
    assigned_agent: str | None = None
    result: Any | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedTask":
        task_id = data.get("task_id", str(uuid.uuid4()))
        return cls(
            task_id=task_id,
            description=data.get("description", ""),
            phase=data.get("phase", "execution"),
            priority=data.get("priority", 1),
            dependencies=data.get("dependencies", []),
            status=data.get("status", "pending"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "phase": self.phase,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "status": self.status,
            "assigned_agent": self.assigned_agent,
            "result": self.result,
            "error": self.error,
        }


class TaskGraph:
    """DAG 기반 의존성 관리 및 통합 큐"""

    def __init__(self):
        self.tasks: Dict[str, UnifiedTask] = {}
        # task_id -> set of task_ids that depend on it
        self.dependents: Dict[str, Set[str]] = {}

    def add_task(self, task: UnifiedTask) -> None:
        """태스크 추가 및 DAG 검증"""
        self.tasks[task.task_id] = task
        if task.task_id not in self.dependents:
            self.dependents[task.task_id] = set()

        for dep_id in task.dependencies:
            if dep_id not in self.dependents:
                self.dependents[dep_id] = set()
            self.dependents[dep_id].add(task.task_id)

        self._update_task_status(task.task_id)

    def add_task_from_dict(self, task_dict: Dict[str, Any]) -> str:
        """딕셔너리에서 태스크 생성 및 추가"""
        task = UnifiedTask.from_dict(task_dict)
        self.add_task(task)
        return task.task_id

    def spawn_subtask(self, parent_id: str, new_task_dict: Dict[str, Any]) -> str:
        """실행 중 새 태스크를 동적으로 생성"""
        task_id = new_task_dict.get("task_id", str(uuid.uuid4()))
        new_task_dict["task_id"] = task_id

        # 새 태스크는 부모에 종속되지 않지만, 다른 태스크가 이걸 기다리게 수정 가능
        task = UnifiedTask.from_dict(new_task_dict)
        self.add_task(task)
        logger.info(f"[TaskGraph] Dynamically spawned subtask: {task_id} from {parent_id}")
        return task_id

    def _update_task_status(self, task_id: str) -> None:
        """의존성이 모두 해결되었는지 확인 후 상태 변경"""
        task = self.tasks.get(task_id)
        if not task or task.status not in ("pending", "ready"):
            return

        # 모든 의존 태스크가 'completed'인지 확인
        all_resolved = True
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != "completed":
                all_resolved = False
                break

        if all_resolved:
            task.status = "ready"
        else:
            task.status = "pending"

    def get_ready_tasks(self, phase: str | None = None) -> List[UnifiedTask]:
        """실행 준비가 된 태스크 반환 (우선순위 정렬)"""
        ready_tasks = []
        for task in self.tasks.values():
            if task.status == "ready":
                if phase is None or task.phase == phase:
                    ready_tasks.append(task)

        # 우선순위가 높은 것(숫자가 작은 것) 먼저
        return sorted(ready_tasks, key=lambda t: t.priority)

    def mark_running(self, task_id: str, agent_id: str) -> None:
        task = self.tasks.get(task_id)
        if task and task.status == "ready":
            task.status = "running"
            task.assigned_agent = agent_id

    def mark_completed(self, task_id: str, result: Any) -> None:
        """태스크 완료 및 의존하는 태스크들 상태 업데이트"""
        task = self.tasks.get(task_id)
        if not task:
            return

        task.status = "completed"
        task.result = result

        # 이 태스크를 기다리던 태스크들 업데이트
        if task_id in self.dependents:
            for dependent_id in self.dependents[task_id]:
                self._update_task_status(dependent_id)

    def mark_failed(self, task_id: str, error: str) -> None:
        task = self.tasks.get(task_id)
        if task:
            task.status = "failed"
            task.error = error

    def get_execution_order(self) -> List[List[str]]:
        """DAG 기반 실행 레벨(병렬 그룹) 반환 (간단한 위상 정렬)"""
        levels = []
        in_degree = {task_id: len(task.dependencies) for task_id, task in self.tasks.items()}
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]

        while queue:
            levels.append(queue)
            next_queue = []
            for task_id in queue:
                for dependent_id in self.dependents.get(task_id, set()):
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_queue.append(dependent_id)
            queue = next_queue

        return levels

    # --- Legacy Compatibility Methods ---

    def has_pending_tasks(self) -> bool:
        """대기 중이거나 실행 중인 태스크가 있는지 확인"""
        for task in self.tasks.values():
            if task.status in ("pending", "ready", "running"):
                return True
        return False

    def get_next_task_group(self, max_group_size: int) -> List[str]:
        """준비된 태스크들을 최대 동시 실행 수만큼 반환"""
        ready = self.get_ready_tasks()
        return [t.task_id for t in ready[:max_group_size]]

    def get_task(self, task_id: str) -> UnifiedTask | None:
        """태스크 객체 반환"""
        return self.tasks.get(task_id)

    def get_progress(self) -> Dict[str, Any]:
        """진행 상태 반환"""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == "completed")
        failed = sum(1 for t in self.tasks.values() if t.status == "failed")
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "percent": (completed / total * 100) if total > 0 else 0,
        }

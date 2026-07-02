"""
task_dashboard.py — Task queue progress & execution result dashboard.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    SUCCESS    = "success"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


@dataclass
class TaskRecord:
    task_id: str
    name: str
    description: str
    agent_id: str
    status: TaskStatus = TaskStatus.QUEUED
    progress: float = 0.0       # 0.0 – 1.0
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at) * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["duration_ms"] = self.duration_ms
        return d


class TaskDashboard:
    """
    In-memory task queue and execution result dashboard.
    Provides real-time status updates via callback hooks.
    """

    _instance: Optional["TaskDashboard"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "TaskDashboard":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock_data = threading.RLock()
        self._update_callbacks: List[Callable[[TaskRecord], None]] = []

    def register_update_callback(self, cb: Callable[[TaskRecord], None]) -> None:
        """Register a callback invoked on every task status update."""
        self._update_callbacks.append(cb)

    def _notify(self, task: TaskRecord) -> None:
        for cb in self._update_callbacks:
            try:
                cb(task)
            except Exception as e:
                logger.error("Task update callback error: %s", e)

    def submit(
        self,
        name: str,
        description: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskRecord:
        """Submit a new task to the queue."""
        task = TaskRecord(
            task_id=str(uuid.uuid4()),
            name=name,
            description=description,
            agent_id=agent_id,
            metadata=metadata or {},
        )
        with self._lock_data:
            self._tasks[task.task_id] = task
        self._notify(task)
        logger.info("Task submitted: %s (%s)", name, task.task_id[:8])
        return task

    def start(self, task_id: str) -> bool:
        """Mark a task as running."""
        with self._lock_data:
            task = self._tasks.get(task_id)
            if not task:
                return False
            task.status     = TaskStatus.RUNNING
            task.started_at = time.time()
        self._notify(task)
        return True

    def update_progress(self, task_id: str, progress: float) -> bool:
        """Update task progress (0.0 – 1.0)."""
        with self._lock_data:
            task = self._tasks.get(task_id)
            if not task:
                return False
            task.progress = max(0.0, min(1.0, progress))
        self._notify(task)
        return True

    def complete(self, task_id: str, result: Any = None, error: Optional[str] = None) -> bool:
        """Mark a task as completed (success or failure)."""
        with self._lock_data:
            task = self._tasks.get(task_id)
            if not task:
                return False
            task.status      = TaskStatus.SUCCESS if error is None else TaskStatus.FAILED
            task.result      = result
            task.error       = error
            task.progress    = 1.0 if error is None else task.progress
            task.finished_at = time.time()
        self._notify(task)
        logger.info(
            "Task %s: %s (%s) in %.0fms",
            task.task_id[:8], task.status.value, task.name,
            task.duration_ms or 0,
        )
        return True

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued or running task."""
        with self._lock_data:
            task = self._tasks.get(task_id)
            if not task or task.status not in (TaskStatus.QUEUED, TaskStatus.RUNNING):
                return False
            task.status      = TaskStatus.CANCELLED
            task.finished_at = time.time()
        self._notify(task)
        return True

    def get(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock_data:
            return self._tasks.get(task_id)

    def list_tasks(
        self,
        agent_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        limit: int = 50,
    ) -> List[TaskRecord]:
        """List tasks with optional filters."""
        with self._lock_data:
            tasks = list(self._tasks.values())
        if agent_id:
            tasks = [t for t in tasks if t.agent_id == agent_id]
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)[:limit]

    def summary(self) -> Dict[str, int]:
        """Return a count summary by status."""
        with self._lock_data:
            tasks = list(self._tasks.values())
        counts: Dict[str, int] = {s.value: 0 for s in TaskStatus}
        for t in tasks:
            counts[t.status.value] += 1
        counts["total"] = len(tasks)
        return counts

    def reset(self) -> None:
        with self._lock_data:
            self._tasks.clear()

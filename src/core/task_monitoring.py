"""Task-level monitoring: queue depth, throughput, latency, success/failure.

Feeds HealthMonitor and TaskProcessingLevel; enables monitoring-based task processing updates.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TaskProcessingLevel(str, Enum):
    """Task processing level derived from monitoring (모니터링 기반 처리 수준)."""

    REDUCED = "reduced"  # High load/errors → lower concurrency
    NORMAL = "normal"
    ELEVATED = "elevated"  # Low load, queue building → higher concurrency


@dataclass
class TaskSnapshot:
    """Point-in-time task metrics for one session or globally."""

    timestamp: datetime
    queue_depth: int
    running: int
    completed_total: int
    failed_total: int
    latency_sec: float | None  # last or p50
    envelope_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class TaskMetrics:
    """Aggregated task metrics for monitoring and level control."""

    timestamp: datetime
    total_queue_depth: int
    total_running: int
    total_completed: int
    total_failed: int
    session_count: int
    latency_p50_sec: float | None
    latency_p95_sec: float | None
    throughput_per_min: float  # completed per minute, rolling
    error_rate: float  # failed / (completed + failed)
    by_session: Dict[str, TaskSnapshot] = field(default_factory=dict)


class TaskMetricsCollector:
    """Collects task start/end and queue depth; exposes metrics for HealthMonitor and level controller."""

    def __init__(self, window_sec: float = 300.0, max_latency_samples: int = 500):
        self.window_sec = window_sec
        self.max_latency_samples = max_latency_samples
        self._lock = Lock()

        # Per-session
        self._queue_depth: Dict[str, int] = defaultdict(int)
        self._running: Dict[str, int] = defaultdict(int)
        self._completed: Dict[str, int] = defaultdict(int)
        self._failed: Dict[str, int] = defaultdict(int)
        self._running_since: Dict[str, float] = {}  # task_id -> start time

        # Global latency samples (recent)
        self._latency_samples: List[float] = []
        self._completed_timestamps: List[float] = []  # for throughput

    def record_enqueued(self, session_id: str, envelope_type: str = "message") -> None:
        """Call when an item is enqueued (e.g. SessionLane.enqueue)."""
        with self._lock:
            self._queue_depth[session_id] += 1

    def record_dequeued(self, session_id: str) -> None:
        """Call when worker picks an item (queue depth decreases, running increases)."""
        with self._lock:
            self._queue_depth[session_id] = max(0, self._queue_depth[session_id] - 1)
            self._running[session_id] += 1

    def record_start(self, session_id: str, task_id: str | None = None) -> None:
        """Optionally record start for latency (task_id used as key for end)."""
        key = task_id or f"{session_id}_{id(self)}"
        with self._lock:
            self._running_since[key] = time.monotonic()

    def record_end(
        self,
        session_id: str,
        success: bool,
        task_id: str | None = None,
        latency_sec: float | None = None,
    ) -> None:
        """Record task end; latency_sec can be provided or computed from record_start."""
        key = task_id or f"{session_id}_{id(self)}"
        with self._lock:
            self._running[session_id] = max(0, self._running[session_id] - 1)
            if success:
                self._completed[session_id] += 1
            else:
                self._failed[session_id] += 1
            if key in self._running_since and latency_sec is None:
                latency_sec = time.monotonic() - self._running_since.pop(key, 0)
            elif key in self._running_since:
                self._running_since.pop(key, None)
        if latency_sec is not None and latency_sec >= 0:
            with self._lock:
                self._latency_samples.append(latency_sec)
                if len(self._latency_samples) > self.max_latency_samples:
                    self._latency_samples.pop(0)
                self._completed_timestamps.append(time.monotonic())
                cutoff = time.monotonic() - self.window_sec
                self._completed_timestamps = [t for t in self._completed_timestamps if t >= cutoff]

    def get_queue_depth(self, session_id: str | None = None) -> int:
        """Total or per-session queue depth."""
        with self._lock:
            if session_id is not None:
                return self._queue_depth.get(session_id, 0)
            return sum(self._queue_depth.values())

    def get_running(self, session_id: str | None = None) -> int:
        """Total or per-session running count."""
        with self._lock:
            if session_id is not None:
                return self._running.get(session_id, 0)
            return sum(self._running.values())

    def get_global_metrics(self) -> TaskMetrics:
        """Aggregated metrics for monitoring and level controller."""
        with self._lock:
            total_queue = sum(self._queue_depth.values())
            total_running = sum(self._running.values())
            total_completed = sum(self._completed.values())
            total_failed = sum(self._failed.values())
            sessions = set(self._queue_depth) | set(self._running) | set(self._completed) | set(self._failed)
            n = len(self._latency_samples)
            if n:
                sorted_lat = sorted(self._latency_samples)
                p50 = sorted_lat[int((n - 1) * 0.5)]
                p95 = (
                    sorted_lat[int((n - 1) * 0.95)]
                    if n >= 2
                    else sorted_lat[-1]
                )
            else:
                p50 = p95 = None
            now = time.monotonic()
            cutoff = now - 60.0  # last minute for throughput
            completed_last_min = sum(1 for t in self._completed_timestamps if t >= cutoff)
            throughput = completed_last_min
            total_done = total_completed + total_failed
            error_rate = total_failed / total_done if total_done else 0.0

            by_session = {}
            for sid in sessions:
                by_session[sid] = TaskSnapshot(
                    timestamp=datetime.now(),
                    queue_depth=self._queue_depth.get(sid, 0),
                    running=self._running.get(sid, 0),
                    completed_total=self._completed.get(sid, 0),
                    failed_total=self._failed.get(sid, 0),
                    latency_sec=p50,
                    envelope_types={},
                )

        return TaskMetrics(
            timestamp=datetime.now(),
            total_queue_depth=total_queue,
            total_running=total_running,
            total_completed=total_completed,
            total_failed=total_failed,
            session_count=len(sessions),
            latency_p50_sec=p50,
            latency_p95_sec=p95,
            throughput_per_min=float(throughput),
            error_rate=error_rate,
            by_session=by_session,
        )

    def get_research_task_count(self) -> int:
        """Total tasks in queue + running (for HealthMonitor compatibility)."""
        m = self.get_global_metrics()
        return m.total_queue_depth + m.total_running

    def get_agent_status_summary(self) -> Dict[str, str]:
        """Summary for HealthMonitor.agent_status: busy if running, idle if not."""
        with self._lock:
            total_running = sum(self._running.values())
            total_queue = sum(self._queue_depth.values())
        if total_running > 0 or total_queue > 0:
            return {
                "orchestrator": "busy",
                "session_lane": "active",
                "tasks_queued": str(total_queue),
                "tasks_running": str(total_running),
            }
        return {
            "orchestrator": "idle",
            "session_lane": "idle",
        }


_collector: TaskMetricsCollector | None = None


def get_task_metrics_collector() -> TaskMetricsCollector:
    """Global TaskMetricsCollector instance."""
    global _collector
    if _collector is None:
        _collector = TaskMetricsCollector()
    return _collector


class TaskProcessingLevelController:
    """모니터링 기반으로 task 처리 수준(REDUCED/NORMAL/ELEVATED) 및 권장 동시 실행 수 결정."""

    def __init__(
        self,
        base_max_concurrent: int = 5,
        min_concurrent: int = 1,
        max_concurrent: int = 20,
        error_rate_threshold: float = 0.2,
        queue_depth_threshold_high: int = 50,
        queue_depth_threshold_low: int = 5,
    ):
        self.base_max_concurrent = base_max_concurrent
        self.min_concurrent = min_concurrent
        self.max_concurrent = max_concurrent
        self.error_rate_threshold = error_rate_threshold
        self.queue_depth_threshold_high = queue_depth_threshold_high
        self.queue_depth_threshold_low = queue_depth_threshold_low
        self._current_level = TaskProcessingLevel.NORMAL
        self._suggested_max_concurrent = base_max_concurrent
        self._lock = Lock()

    def update(self, task_metrics: TaskMetrics | None = None, cpu_percent: float | None = None, memory_percent: float | None = None) -> None:
        """Update level and suggested concurrency from metrics."""
        if task_metrics is None:
            task_metrics = get_task_metrics_collector().get_global_metrics()
        with self._lock:
            level = TaskProcessingLevel.NORMAL
            suggested = self.base_max_concurrent

            # Error rate → REDUCED
            if task_metrics.error_rate >= self.error_rate_threshold:
                level = TaskProcessingLevel.REDUCED
                suggested = max(self.min_concurrent, self.base_max_concurrent // 2)
            # High queue + low throughput / high latency → consider REDUCED if system stressed
            elif task_metrics.total_queue_depth >= self.queue_depth_threshold_high:
                if cpu_percent is not None and cpu_percent >= 80:
                    level = TaskProcessingLevel.REDUCED
                    suggested = max(self.min_concurrent, self.base_max_concurrent - 2)
                else:
                    level = TaskProcessingLevel.ELEVATED
                    suggested = min(self.max_concurrent, self.base_max_concurrent + 2)
            elif task_metrics.total_queue_depth <= self.queue_depth_threshold_low and task_metrics.total_running == 0:
                if cpu_percent is not None and cpu_percent < 60 and (memory_percent is None or memory_percent < 75):
                    level = TaskProcessingLevel.ELEVATED
                    suggested = min(self.max_concurrent, self.base_max_concurrent + 1)
            else:
                suggested = self.base_max_concurrent

            suggested = max(self.min_concurrent, min(self.max_concurrent, suggested))
            self._current_level = level
            self._suggested_max_concurrent = suggested

    def get_level(self) -> TaskProcessingLevel:
        """Current processing level."""
        with self._lock:
            return self._current_level

    def get_suggested_max_concurrent(self) -> int:
        """Suggested max concurrent tasks from monitoring."""
        with self._lock:
            return self._suggested_max_concurrent


_level_controller: TaskProcessingLevelController | None = None


def get_task_processing_level_controller() -> TaskProcessingLevelController:
    """Global TaskProcessingLevelController instance."""
    global _level_controller
    if _level_controller is None:
        _level_controller = TaskProcessingLevelController()
    return _level_controller

"""Anvil Workflow Engine - 동적 DAG 기반 태스크 스케줄링 엔진.

경직된 순차 실행 흐름 대신, 의존성 기반 토폴로지 정렬과
병렬 실행을 지원하는 동적 태스크 그래프 엔진.
Riemannian manifold geodesic explorer for optimal design pathfinding.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ManifoldPoint:
    """A point in the Riemannian manifold solution space."""

    coordinates: List[float]
    design_id: str
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiemannianManifoldExplorer:
    """Maps high-dimensional architecture requirements onto a Riemannian
    manifold and finds shortest geodesic paths toward optimal system designs.

    The manifold is modelled as a weighted graph where each node is a design
    configuration (a point on the manifold) and edge weights approximate the
    Riemannian distance metric. Geodesics are discovered via Dijkstra's
    algorithm, which yields the locally shortest path on the discretised
    manifold.
    """

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.points: Dict[str, ManifoldPoint] = {}
        # adjacency: design_id -> {neighbour_id -> distance}
        self.adjacency: Dict[str, Dict[str, float]] = {}
        logger.info(
            f"[ManifoldExplorer] Initialized {dimension}-dimensional manifold"
        )

    def add_point(self, point: ManifoldPoint) -> None:
        """Insert a design point into the manifold."""
        self.points[point.design_id] = point
        self.adjacency.setdefault(point.design_id, {})
        logger.debug(f"[ManifoldExplorer] Point added: {point.design_id}")

    def connect(self, source_id: str, target_id: str) -> None:
        """Connect two design points with a Riemannian distance edge."""
        if source_id not in self.points or target_id not in self.points:
            raise ValueError("Both points must exist on the manifold before connecting")
        distance = self._riemannian_distance(
            self.points[source_id], self.points[target_id]
        )
        self.adjacency[source_id][target_id] = distance
        self.adjacency[target_id][source_id] = distance

    def _riemannian_distance(self, a: ManifoldPoint, b: ManifoldPoint) -> float:
        """Approximate Riemannian distance between two points.

        Uses the Euclidean metric scaled by a diagonal metric tensor derived
        from the per-coordinate cost sensitivity stored in point metadata.
        """
        if len(a.coordinates) != len(b.coordinates):
            raise ValueError("Manifold points must share dimensionality")
        metric = [
            max((a.metadata.get("metric") or {}).get(i, 1.0), 1e-6)
            for i in range(len(a.coordinates))
        ]
        total = 0.0
        for i, (x, y) in enumerate(zip(a.coordinates, b.coordinates)):
            delta = x - y
            total += metric[i] * delta * delta
        return total ** 0.5

    def shortest_geodesic(
        self, start_id: str, goal_id: str
    ) -> tuple[List[str], float]:
        """Find the shortest geodesic path between two design points.

        Returns:
            A (path, distance) tuple where path is the list of design_ids
            from start to goal and distance is the accumulated Riemannian
            length of the geodesic.

        Raises:
            ValueError: if either endpoint is unknown or no path exists.
        """
        if start_id not in self.points:
            raise ValueError(f"Start point {start_id} not on manifold")
        if goal_id not in self.points:
            raise ValueError(f"Goal point {goal_id} not on manifold")

        # Dijkstra over the discretised manifold graph.
        import heapq

        dist: Dict[str, float] = {pid: float("inf") for pid in self.points}
        prev: Dict[str, Optional[str]] = {pid: None for pid in self.points}
        dist[start_id] = 0.0
        heap: list[tuple[float, str]] = [(0.0, start_id)]

        while heap:
            current_dist, current = heapq.heappop(heap)
            if current == goal_id:
                break
            if current_dist > dist[current]:
                continue
            for neighbour, weight in self.adjacency.get(current, {}).items():
                candidate = current_dist + weight
                if candidate < dist[neighbour]:
                    dist[neighbour] = candidate
                    prev[neighbour] = current
                    heapq.heappush(heap, (candidate, neighbour))

        if dist[goal_id] == float("inf"):
            raise ValueError(
                f"No geodesic path exists from {start_id} to {goal_id}"
            )

        path: List[str] = []
        node: Optional[str] = goal_id
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()

        logger.info(
            f"[ManifoldExplorer] Geodesic {start_id}->{goal_id}: "
            f"{len(path)} hops, distance={dist[goal_id]:.4f}"
        )
        return path, dist[goal_id]


@dataclass
class AnvilTask:
    """단일 실행 태스크 정의."""

    task_id: str
    name: str
    handler: str  # 실행할 핸들러 함수 이름 (예: "_analyze_topic")
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | completed | failed | skipped
    result: Any = None
    retry_count: int = 0
    max_retries: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class AnvilWorkflowEngine:
    """동적 DAG 기반 태스크 스케줄링 엔진.

    Features:
        - 의존성 기반 토폴로지 정렬로 실행 순서 자동 결정
        - 독립 태스크 병렬 실행
        - 런타임 중 태스크 동적 삽입/제거
        - 태스크별 재시도 정책
        - 핸들러 레지스트리를 통한 확장 가능한 디스패치
    """

    def __init__(
        self,
        skill_repository: Any | None = None,
        handler_registry: Dict[str, Callable] | None = None,
    ):
        self.tasks: Dict[str, AnvilTask] = {}
        self.skill_repository = skill_repository
        self.handler_registry: Dict[str, Callable] = handler_registry or {}
        logger.info("[AnvilEngine] Initialized")

    def register_handler(self, name: str, handler: Callable) -> None:
        """핸들러 함수를 레지스트리에 등록."""
        self.handler_registry[name] = handler
        logger.debug(f"[AnvilEngine] Handler registered: {name}")

    def add_task(self, task: AnvilTask) -> None:
        """태스크를 DAG에 추가."""
        self.tasks[task.task_id] = task
        logger.info(f"[AnvilEngine] Task added: {task.task_id} ({task.name})")

    def remove_task(self, task_id: str) -> None:
        """태스크를 DAG에서 제거. 다른 태스크의 의존성도 정리."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            # 삭제된 태스크를 참조하는 의존성 정리
            for t in self.tasks.values():
                if task_id in t.dependencies:
                    t.dependencies.remove(task_id)
            logger.info(f"[AnvilEngine] Task removed: {task_id}")

    def insert_task_after(self, after_task_id: str, new_task: AnvilTask) -> None:
        """특정 태스크 이후에 새 태스크를 동적으로 삽입."""
        if after_task_id not in self.tasks:
            raise ValueError(f"Task {after_task_id} not found in DAG")
        if after_task_id not in new_task.dependencies:
            new_task.dependencies.append(after_task_id)
        self.add_task(new_task)
        logger.info(
            f"[AnvilEngine] Task {new_task.task_id} inserted after {after_task_id}"
        )

    def _topological_sort(self) -> List[List[str]]:
        """태스크 의존성 기반 토폴로지 정렬. 병렬 실행 가능한 레벨별 그룹 반환.

        Returns:
            List of task_id groups. 같은 그룹 내 태스크는 병렬 실행 가능.

        Raises:
            ValueError: 순환 의존성이 감지된 경우.
        """
        in_degree: Dict[str, int] = {tid: 0 for tid in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.task_id] += 1

        # BFS 레벨별 정렬
        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        levels: List[List[str]] = []
        processed = 0

        while queue:
            level = list(queue)
            levels.append(level)
            next_queue: deque[str] = deque()
            for tid in level:
                processed += 1
                for task in self.tasks.values():
                    if tid in task.dependencies:
                        in_degree[task.task_id] -= 1
                        if in_degree[task.task_id] == 0:
                            next_queue.append(task.task_id)
            queue = next_queue

        if processed < len(self.tasks):
            unprocessed = [
                tid for tid in self.tasks if tid not in {t for lvl in levels for t in lvl}
            ]
            raise ValueError(
                f"[AnvilEngine] Circular dependency detected among tasks: {unprocessed}"
            )

        return levels

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """DAG를 토폴로지 정렬하고 레벨별로 병렬 실행.

        Args:
            context: 실행 컨텍스트 (공유 상태).

        Returns:
            실행 결과 딕셔너리.
        """
        if not self.tasks:
            logger.warning("[AnvilEngine] No tasks to execute")
            return {"status": "empty", "results": {}}

        try:
            levels = self._topological_sort()
        except ValueError as e:
            logger.error(str(e))
            return {"status": "error", "error": str(e), "results": {}}

        logger.info(
            f"[AnvilEngine] Executing {len(self.tasks)} tasks in {len(levels)} levels"
        )

        for level_idx, level in enumerate(levels):
            logger.info(
                f"[AnvilEngine] Level {level_idx + 1}/{len(levels)}: {level}"
            )
            coroutines = [
                self._execute_task(self.tasks[tid], context)
                for tid in level
                if self.tasks[tid].status == "pending"
            ]
            if coroutines:
                await asyncio.gather(*coroutines, return_exceptions=True)

        # 실행 결과 집계
        completed = sum(1 for t in self.tasks.values() if t.status == "completed")
        failed = sum(1 for t in self.tasks.values() if t.status == "failed")

        status = "completed" if failed == 0 else "partial_failure"
        logger.info(
            f"[AnvilEngine] Execution finished: {completed} completed, {failed} failed"
        )

        return {
            "status": status,
            "completed": completed,
            "failed": failed,
            "results": {t.task_id: t.result for t in self.tasks.values()},
        }

    async def _execute_task(self, task: AnvilTask, context: Dict[str, Any]) -> Any:
        """단일 태스크 실행 (핸들러 디스패치 + 재시도).

        핸들러 우선순위:
            1. handler_registry에 등록된 callable
            2. skill_repository에 저장된 스킬
            3. 미등록 시 경고 로그 + skip
        """
        task.status = "running"
        logger.info(f"[AnvilEngine] Executing task: {task.task_id} ({task.name})")

        handler = self.handler_registry.get(task.handler)
        if handler is None and self.skill_repository:
            skill = self.skill_repository.get_skill(task.handler)
            if skill:
                logger.info(
                    f"[AnvilEngine] Using skill '{task.handler}' from repository"
                )
                handler = lambda ctx, s=skill: self.skill_repository.execute_skill(
                    s.name, context=ctx
                )

        if handler is None:
            logger.warning(
                f"[AnvilEngine] No handler found for '{task.handler}', skipping task {task.task_id}"
            )
            task.status = "skipped"
            return None

        while task.retry_count <= task.max_retries:
            try:
                if asyncio.iscoroutinefunction(handler):
                    task.result = await handler(context)
                else:
                    task.result = handler(context)
                task.status = "completed"
                logger.info(f"[AnvilEngine] Task completed: {task.task_id}")
                return task.result
            except Exception as e:
                task.retry_count += 1
                task.error = str(e)
                logger.warning(
                    f"[AnvilEngine] Task {task.task_id} failed (attempt {task.retry_count}/{task.max_retries + 1}): {e}"
                )
                if task.retry_count > task.max_retries:
                    task.status = "failed"
                    logger.error(
                        f"[AnvilEngine] Task {task.task_id} exhausted retries"
                    )
                    return None

    def reset(self) -> None:
        """엔진 상태 초기화 (태스크 전부 제거)."""
        self.tasks.clear()
        logger.info("[AnvilEngine] Engine reset")

    def get_status(self) -> Dict[str, str]:
        """전체 엔진 상태 반환."""
        return {tid: t.status for tid, t in self.tasks.items()}

import logging
import threading
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Task Registry and Delegation Graph with Cycle Detection and Safe Tree Rendering."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tasks: Dict[str, Dict[str, Any]] = {}
            cls._instance._lock_data = threading.Lock()
        return cls._instance

    def reset(self) -> None:
        with self._lock_data:
            self._tasks.clear()

    def submit(
        self,
        task_id: str,
        parent_task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock_data:
            if parent_task_id is not None:
                if parent_task_id not in self._tasks:
                    raise KeyError(f"Parent task not found: {parent_task_id}")

                # Check for direct or multi-level cycles
                current = parent_task_id
                while current is not None:
                    if current == task_id:
                        raise ValueError(f"Cycle detected: task {task_id} cannot be its own ancestor")
                    parent_info = self._tasks.get(current)
                    current = parent_info.get("parent_task_id") if parent_info else None

            self._tasks[task_id] = {
                "task_id": task_id,
                "parent_task_id": parent_task_id,
                "metadata": metadata or {},
            }

    def render_tree(self, root_task_id: Optional[str] = None, max_depth: int = 100) -> str:
        """Render task delegation tree with cycle protection and depth limit."""
        with self._lock_data:
            lines: List[str] = []

            # Find roots if not specified
            roots = [
                tid
                for tid, t in self._tasks.items()
                if t.get("parent_task_id") is None
                or t.get("parent_task_id") not in self._tasks
            ]
            if root_task_id:
                if root_task_id not in self._tasks:
                    return f"Task not found: {root_task_id}"
                roots = [root_task_id]

            def _recurse(task_id: str, prefix: str, visited: Set[str], depth: int) -> None:
                if depth > max_depth:
                    lines.fappend(f"{prefix}max depth reached")
                    return
                if task_id in visited:
                    logger.warning(f"Cycle detected during render_tree at node {task_id}")
                    lines.append(f"{prefix}[CYCLE DETECTED: {task_id}]")
                    return

                visited.add(task_id)
                task_info = self._tasks.get(task_id, {})
                lines.append(f"{prefix}- {task_id}")

                # Find children
                children = [
                    tid
                    for tid, t in self._tasks.items()
                    if t.get("parent_task_id") == task_id
                ]
                for child_id in children:
                    _recurse(child_id, prefix + "  ", visited.copy(), depth + 1)

            for r in roots:
                _recurse(r, "", set(), 0)

            return "\n".join(lines)


# Tests
def test_submit_rejects_cycle_direct_parent():
    registry = TaskRegistry()
    registry.reset()
    registry.submit("t1")
    try:
        registry.submit("t1", parent_task_id="t1")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_submit_rejects_cycle_multi_level():
    registry = TaskRegistry()
    registry.reset()
    registry.submit("t1")
    registry.submit("t2", parent_task_id="t1")
    registry.submit("t3", parent_task_id="t2")
    try:
        registry.submit("t1", parent_task_id="t3")
        assert False, "Expected ValueError"
    except ValueError:
        pass

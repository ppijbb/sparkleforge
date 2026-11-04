import logging
import threading
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Registry for tasks, handling parent-child relationships and tree rendering with cycle detection."""

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock_data = threading.Lock()

    def submit(
        self,
        task_id: str,
        parent_task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Submit a task with optional parent_task_id, verifying existence and acyclicity."""
        with self._lock_data:
            if parent_task_id is not None:
                if parent_task_id not in self._tasks:
                    raise KeyError(f"Parent task '{parent_task_id}' does not exist.")
                
                # Check for direct or indirect cycles
                current = parent_task_id
                visited_ancestors: Set[str] = {task_id}
                while current is not None:
                    if current == task_id:
                        raise ValueError(f"Cycle detected: task '{task_id}' cannot be its own ancestor.")
                    if current in visited_ancestors:
                        break
                    visited_ancestors.add(current)
                    current = self._tasks.get(current, {}).get("parent_task_id")

            self._tasks[task_id] = {
                "task_id": task_id,
                "parent_task_id": parent_task_id,
                "metadata": metadata or {},
            }

    def render_tree(self, task_id: Optional[str] = None, max_depth: int = 100) -> str:
        """Render the task tree starting from a given task_id or roots, protected against cycles."""
        with self._lock_data:
            lines: List[str] = []
            visited: Set[str] = set()

            def _recurse(curr_id: str, depth: int, current_visited: Set[str]) -> None:
                if depth > max_depth:
                    logger.warning(f"Max depth {max_depth} reached while rendering tree at task '{curr_id}'.")
                    return
                if curr_id in current_visited:
                    logger.warning(f"Cycle detected during render_tree at task '{curr_id}'. Skipping branch.")
                    return
                
                current_visited.add(curr_id)
                task_info = self._tasks.get(curr_id, {})
                indent = "  " * depth
                lines.append(f"{indent}- {curr_id}")

                # Find children
                children = [
                    tid for tid, tdata in self._tasks.items()
                    if tdata.get("parent_task_id") == curr_id
                ]
                for child_id in children:
                    _recurse(child_id, depth + 1, set(current_visited))

            if task_id is not None:
                if task_id not in self._tasks:
                    raise KeyError(f"Task '{task_id}' not found.")
                _recurse(task_id, 0, visited)
            else:
                # Render all roots
                roots = [
                    tid for tid, tdata in self._tasks.items()
                    if not tdata.get("parent_task_id") or tdata.get("parent_task_id") not in self._tasks
                ]
                for root_id in roots:
                    _recurse(root_id, 0, set(visited))

            return "\n".join(lines)


class TaskDelegationGraph:
    """Facade/delegation graph manager mapping to TaskRegistry."""

    def __init__(self, registry: Optional[TaskRegistry] = None):
        self.registry = registry or TaskRegistry()

    def submit(self, task_id: str, parent_task_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.registry.submit(task_id, parent_task_id, metadata)

    def render_tree(self, task_id: Optional[str] = None, max_depth: int = 100) -> str:
        return self.registry.render_tree(task_id, max_depth=max_depth)

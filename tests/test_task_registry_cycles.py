import pytest
from src.core.task_registry import TaskRegistry

def test_submit_rejects_cycle_direct_parent():
    registry = TaskRegistry()
    registry.reset()
    registry.submit("t1")
    with pytest.raises(ValueError, match="Cycle detected"):
        registry.submit("t1", parent_task_id="t1")

def test_submit_rejects_cycle_multi_level():
    registry = TaskRegistry()
    registry.reset()
    registry.submit("t1")
    registry.submit("t2", parent_task_id="t1")
    registry.submit("t3", parent_task_id="t2")
    with pytest.raises(ValueError, match="Cycle detected"):
        registry.submit("t1", parent_task_id="t3")

def test_submit_rejects_missing_parent():
    registry = TaskRegistry()
    registry.reset()
    with pytest.raises(KeyError, match="Parent task not found"):
        registry.submit("t1", parent_task_id="nonexistent")

def test_render_tree_handles_cycle_gracefully():
    # Construct a cycle manually by bypassing check or injecting state if needed,
    # or test that render_tree handles cyclic references if any exist.
    registry = TaskRegistry()
    registry.reset()
    # Force a cycle internally for testing render_tree safety
    with registry._lock_data:
        registry._tasks["a"] = {"task_id": "a", "parent_task_id": "b", "metadata": {}}
        registry._tasks["b"] = {"task_id": "b", "parent_task_id": "a", "metadata": {}}

    tree = registry.render_tree()
    assert "

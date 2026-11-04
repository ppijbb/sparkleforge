import pytest
from src.core.task_registry import TaskRegistry


def test_submit_rejects_cycle_direct_parent():
    registry = TaskRegistry()
    registry.submit("task_A")
    with pytest.raises(ValueError, match="Cycle detected"):
        registry.submit("task_A", parent_task_id="task_A")


def test_submit_rejects_cycle_multi_level():
    registry = TaskRegistry()
    registry.submit("task_A")
    registry.submit("task_B", parent_task_id="task_A")
    registry.submit("task_C", parent_task_id="task_B")
    
    with pytest.raises(ValueError, match="Cycle detected"):
        registry.submit("task_A", parent_task_id="task_C")


def test_render_tree_handles_cycle_gracefully():
    registry = TaskRegistry()
    # Manually inject a cycle bypassing submit validation for legacy/edge scenarios
    registry._tasks = {
        "task_A": {"task_id": "task_A", "parent_task_id": "task_B", "metadata": {}},
        "task_B": {"task_id": "task_B", "parent_task_id": "task_A", "metadata": {}},
    }
    # Should not raise RecursionError or hang
    tree = registry.render_tree("task_A")
    assert "task_A" in tree
    assert "task_B" in tree


def test_submit_rejects_missing_parent():
    registry = TaskRegistry()
    with pytest.raises(KeyError, match="does not exist"):
        registry.submit("task_A", parent_task_id="non_existent")


def test_render_tree_normal():
    registry = TaskRegistry()
    registry.submit("root")
    registry.submit("child1", parent_task_id="root")
    registry.submit("child2", parent_task_id="root")
    registry.submit("grandchild", parent_task_id="child1")

    tree = registry.render_tree("root")
    assert "- root" in tree
    assert "  - child1" in tree
    assert "  - child2" in tree
    assert "    - grandchild" in tree

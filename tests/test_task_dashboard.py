"""Tests for the live multi-agent delegation graph in TaskDashboard."""

from src.core.surface.task_dashboard import TaskDashboard


def _fresh_dashboard() -> TaskDashboard:
    """Reset the process-wide singleton and return a clean instance.

    TaskDashboard() always returns the same singleton (see __new__); reset()
    clears its state and unsets the class-level _instance, so the following
    TaskDashboard() call constructs a genuinely fresh one rather than
    returning stale state from a previous test.
    """
    TaskDashboard().reset()
    return TaskDashboard()


def test_submit_with_parent_task_id_registers_delegation_edge():
    dashboard = _fresh_dashboard()
    parent = dashboard.submit("parent", "delegates work", "orchestrator")
    child = dashboard.submit("child", "sub task", "worker", parent_task_id=parent.task_id)

    snapshot = dashboard.snapshot()

    assert snapshot["edges"] == [
        {"from": parent.task_id, "to": child.task_id, "type": "delegates_to"}
    ]


def test_submit_without_parent_task_id_has_no_edges():
    dashboard = _fresh_dashboard()
    dashboard.submit("solo", "no delegation", "worker")

    assert dashboard.snapshot()["edges"] == []


def test_to_session_view_only_includes_edges_within_scope():
    dashboard = _fresh_dashboard()
    parent = dashboard.submit("parent", "in session", "orchestrator", metadata={"session_id": "s1"})
    child = dashboard.submit(
        "child", "in session", "worker", metadata={"session_id": "s1"}, parent_task_id=parent.task_id
    )
    dashboard.submit(
        "other", "different session", "worker", metadata={"session_id": "s2"}, parent_task_id=parent.task_id
    )

    view = dashboard.to_session_view("s1")

    assert len(view["tasks"]) == 2
    assert view["edges"] == [{"from": parent.task_id, "to": child.task_id, "type": "delegates_to"}]


def test_snapshot_edges_include_multi_level_delegation():
    dashboard = _fresh_dashboard()
    grandparent = dashboard.submit("grandparent", "top of chain", "orchestrator")
    parent = dashboard.submit("parent", "middle", "sub-orchestrator", parent_task_id=grandparent.task_id)
    child = dashboard.submit("child", "leaf", "worker", parent_task_id=parent.task_id)

    edges = dashboard.snapshot()["edges"]

    assert {"from": grandparent.task_id, "to": parent.task_id, "type": "delegates_to"} in edges
    assert {"from": parent.task_id, "to": child.task_id, "type": "delegates_to"} in edges
    assert len(edges) == 2


def test_render_tree_indents_children_under_parent():
    dashboard = _fresh_dashboard()
    parent = dashboard.submit("parent", "delegates work", "orchestrator")
    dashboard.submit("child", "sub task", "worker", parent_task_id=parent.task_id)

    tree = dashboard.render_tree()

    assert "- [queued] parent (orchestrator)" in tree
    assert "  - [queued] child (worker)" in tree


def test_render_tree_lists_multiple_roots_independently():
    dashboard = _fresh_dashboard()
    dashboard.submit("root-a", "first", "worker-a")
    dashboard.submit("root-b", "second", "worker-b")

    tree = dashboard.render_tree()

    assert "- [queued] root-a (worker-a)" in tree
    assert "- [queued] root-b (worker-b)" in tree
    assert "  -" not in tree  # neither root is nested under the other


def test_render_tree_skips_a_child_id_that_no_longer_exists():
    dashboard = _fresh_dashboard()
    parent = dashboard.submit("parent", "delegates work", "orchestrator")
    child = dashboard.submit("child", "sub task", "worker", parent_task_id=parent.task_id)
    # Simulate the child having been removed after the parent->child edge
    # was recorded (e.g. a reset/cleanup path) -- children_task_ids can
    # reference an id no longer present in _tasks.
    del dashboard._tasks[child.task_id]

    tree = dashboard.render_tree()  # must not raise

    assert "- [queued] parent (orchestrator)" in tree
    assert "child" not in tree

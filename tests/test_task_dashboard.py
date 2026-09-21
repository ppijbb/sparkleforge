"""Tests for the live multi-agent delegation graph in TaskDashboard."""

from src.core.surface.task_dashboard import TaskDashboard


def _fresh_dashboard() -> TaskDashboard:
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
    dashboard.submit("child", "in session", "worker", metadata={"session_id": "s1"}, parent_task_id=parent.task_id)
    dashboard.submit("other", "different session", "worker", metadata={"session_id": "s2"}, parent_task_id=parent.task_id)

    view = dashboard.to_session_view("s1")

    assert len(view["tasks"]) == 2
    assert view["edges"] == [{"from": parent.task_id, "to": [
        t["task_id"] for t in view["tasks"] if t["name"] == "child"
    ][0], "type": "delegates_to"}]


def test_render_tree_indents_children_under_parent():
    dashboard = _fresh_dashboard()
    parent = dashboard.submit("parent", "delegates work", "orchestrator")
    dashboard.submit("child", "sub task", "worker", parent_task_id=parent.task_id)

    tree = dashboard.render_tree()

    assert "- [queued] parent (orchestrator)" in tree
    assert "  - [queued] child (worker)" in tree

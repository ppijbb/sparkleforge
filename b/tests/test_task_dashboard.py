from __future__ import annotations

import pytest
from src.core.surface.task_dashboard import TaskDashboard, TaskRecord, TaskStatus


def test_to_session_view_excludes_orphan_edges():
    dashboard = TaskDashboard()
    dashboard.reset()

    # Create parent in session B
    parent = dashboard.submit(
        name="Parent Task", description="Parent", agent_id="agent-1",
        metadata={"session_id": "session-B"}
    )
    # Manually assign child to session A with parent_task_id pointing to parent in session B
    child = dashboard.submit(
        name="Child Task", description="Child", agent_id="agent-2",
        parent_task_id=parent.task_id,
        metadata={"session_id": "session-A"}
    )

    view = dashboard.to_session_view("session-A")
    assert len(view["tasks"]) == 1
    assert len(view["edges"]) == 0

"""
tests/test_surface_plane.py — Unit tests for Phase H: Surface (User Boundary)
"""
import pytest
import time

from src.core.surface.nl_shell import NLShell, ShellIntent
from src.core.surface.task_dashboard import TaskDashboard, TaskStatus
from src.core.surface.notification_channel import NotificationChannel, Notification, NotificationLevel
from src.core.surface.explainability import ExplainabilityEngine


@pytest.fixture(autouse=True)
def reset_singletons():
    TaskDashboard._instance = None
    NotificationChannel._instance = None
    yield
    TaskDashboard._instance = None
    NotificationChannel._instance = None


# --- NLShell Tests ---

def test_nl_shell_direct_passthrough():
    shell = NLShell()
    intent = shell.parse_intent("$ echo hello")
    assert intent.intent_type == "command"
    assert intent.command == "echo hello"
    assert intent.confidence == 1.0


def test_nl_shell_pattern_match_list_files():
    shell = NLShell()
    intent = shell.parse_intent("list files in the current directory")
    assert intent.intent_type == "command"
    assert "ls" in intent.command
    assert intent.confidence > 0.5


def test_nl_shell_pattern_show_memory():
    shell = NLShell()
    intent = shell.parse_intent("show memory usage")
    assert intent.command == "free -h"


def test_nl_shell_unknown():
    shell = NLShell()
    intent = shell.parse_intent("bake me a pizza with extra cheese")
    assert intent.intent_type == "unknown"
    assert intent.command is None
    assert intent.confidence == 0.0


@pytest.mark.asyncio
async def test_nl_shell_single_run():
    shell = NLShell()
    result = await shell.single_run("$ echo anvil_surface_test")
    assert result["ok"] is True
    assert "anvil_surface_test" in result["stdout"]


@pytest.mark.asyncio
async def test_nl_shell_unknown_returns_error():
    shell = NLShell()
    result = await shell.single_run("do something completely undefined and weird")
    assert result["ok"] is False
    assert "error" in result


# --- TaskDashboard Tests ---

def test_task_dashboard_submit_and_complete():
    dashboard = TaskDashboard()
    task = dashboard.submit("test_task", "Run a test", "agent_1")
    assert task.status == TaskStatus.QUEUED

    dashboard.start(task.task_id)
    assert dashboard.get(task.task_id).status == TaskStatus.RUNNING

    dashboard.complete(task.task_id, result={"data": "output"})
    assert dashboard.get(task.task_id).status == TaskStatus.SUCCESS
    assert dashboard.get(task.task_id).result == {"data": "output"}


def test_task_dashboard_progress():
    dashboard = TaskDashboard()
    task = dashboard.submit("progress_task", "Track progress", "agent_1")
    dashboard.start(task.task_id)
    dashboard.update_progress(task.task_id, 0.5)
    assert dashboard.get(task.task_id).progress == 0.5


def test_task_dashboard_failure():
    dashboard = TaskDashboard()
    task = dashboard.submit("failing_task", "A task that fails", "agent_1")
    dashboard.start(task.task_id)
    dashboard.complete(task.task_id, error="Something went wrong")
    assert dashboard.get(task.task_id).status == TaskStatus.FAILED
    assert dashboard.get(task.task_id).error == "Something went wrong"


def test_task_dashboard_cancel():
    dashboard = TaskDashboard()
    task = dashboard.submit("cancel_task", "Cancel this", "agent_1")
    dashboard.cancel(task.task_id)
    assert dashboard.get(task.task_id).status == TaskStatus.CANCELLED


def test_task_dashboard_summary():
    dashboard = TaskDashboard()
    t1 = dashboard.submit("t1", "Task 1", "agent_1")
    t2 = dashboard.submit("t2", "Task 2", "agent_1")
    dashboard.start(t1.task_id)
    dashboard.complete(t1.task_id)
    summary = dashboard.summary()
    assert summary["total"] == 2
    assert summary["success"] == 1
    assert summary["queued"] == 1


def test_task_dashboard_update_callback():
    dashboard = TaskDashboard()
    updates = []
    dashboard.register_update_callback(lambda t: updates.append(t.status))

    task = dashboard.submit("cb_task", "Callback task", "agent_1")
    dashboard.start(task.task_id)
    dashboard.complete(task.task_id)

    assert TaskStatus.QUEUED in updates
    assert TaskStatus.RUNNING in updates
    assert TaskStatus.SUCCESS in updates


# --- NotificationChannel Tests ---

def test_notification_channel_send():
    channel = NotificationChannel()
    n = Notification(title="Test", message="Hello", level=NotificationLevel.INFO)
    result = channel.send(n)
    assert result is True
    history = channel.get_history()
    assert any(h.title == "Test" for h in history)


def test_notification_approval_request():
    channel = NotificationChannel()
    result = channel.notify_approval_needed(
        action="execute_shell",
        agent_id="agent_1",
        risk_level="high",
        request_id="req-1234",
    )
    assert result is True
    history = channel.get_history()
    assert any(h.level == NotificationLevel.APPROVAL for h in history)


def test_notification_anomaly_alert():
    channel = NotificationChannel()
    result = channel.notify_anomaly(
        agent_id="agent_1",
        reason="Rate limit exceeded",
        severity="high",
    )
    assert result is True
    history = channel.get_history()
    assert any(h.level == NotificationLevel.CRITICAL for h in history)


# --- Explainability Tests ---

def test_explainability_explain_recent():
    import tempfile, os
    from src.core.guard.action_journal import ActionJournal

    ActionJournal._instance = None
    with tempfile.TemporaryDirectory() as tmpdir:
        journal = ActionJournal(journal_path=os.path.join(tmpdir, "journal.jsonl"))
        entry = journal.record(
            agent_id="agent_1",
            action="ls /tmp",
            description="List temp dir",
            risk_level="low",
        )
        journal.update_outcome(entry.entry_id, "success")

        engine = ExplainabilityEngine(journal=journal)
        reports = engine.explain_recent(agent_id="agent_1", limit=5)
        assert len(reports) >= 1
        assert reports[0].action == "ls /tmp"
        assert "low-risk" in reports[0].reasoning


def test_explainability_why_text():
    import tempfile, os
    from src.core.guard.action_journal import ActionJournal

    ActionJournal._instance = None
    with tempfile.TemporaryDirectory() as tmpdir:
        journal = ActionJournal(journal_path=os.path.join(tmpdir, "journal.jsonl"))
        entry = journal.record(
            agent_id="agent_2",
            action="write_file /etc/config",
            description="Update system config",
            risk_level="high",
        )

        engine = ExplainabilityEngine(journal=journal)
        text = engine.why(entry.entry_id)
        assert "high-risk" in text
        assert "agent_2" in text
    ActionJournal._instance = None


@pytest.mark.asyncio
async def test_bootstrap_surface_plane():
    from src.core.bootstrap_graph import BootstrapGraph
    from src.core.surface.surface_plane import SurfacePlane
    from src.core.guard.guard_plane import GuardPlane

    # Reset guard plane singleton since bootstrap creates it fresh
    GuardPlane._instance = None

    graph = BootstrapGraph()
    res = await graph.run()
    assert res.ok

    stages = [s.name for s in res.stages]
    assert "surface_plane" in stages
    assert "guard_plane" in stages

    sp_stage = next(s for s in res.stages if s.name == "surface_plane")
    assert sp_stage.ok
    assert sp_stage.payload["initialized"] is True
    assert isinstance(sp_stage.payload["surface_plane"], SurfacePlane)

    GuardPlane._instance = None

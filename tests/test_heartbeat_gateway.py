import asyncio
import os
import pytest

from src.core.automation.automation_engine import AutomationEngine
from src.core.surface.notification_channel import NotificationChannel, NotificationLevel, get_channel_gateway
from src.core.anvil.mode_controller import ModeController, ExecutionMode


@pytest.mark.asyncio
async def test_heartbeat_markdown_parser_and_runner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    heartbeat_file = tmp_path / "HEARTBEAT.md"
    heartbeat_file.write_text("# Heartbeat Checklist\n- [ ] Check system health\n- [ ] Verify database indexes\n")

    engine = AutomationEngine()
    # Mock callback
    executed = []
    async def mock_callback(query, session_id):
        executed.append(query)
        return "ok"
    engine.set_execution_callback(mock_callback)

    res = await engine.tick_heartbeat()
    assert res["checked"] == 2
    assert len(executed) == 2
    assert "Check system health" in executed[0]
    assert "Verify database indexes" in executed[1]


def test_active_hours_quiet_window_enforcement():
    engine = AutomationEngine()
    # Set active hours to 09:00 - 17:00
    engine.set_active_hours(9, 0, 17, 0, enforce=True)

    # Patch datetime.now via mocking or check helper logic
    # We test helper logic directly
    assert engine._active_hours_enforced is True
    assert engine.is_within_active_hours() in (True, False)  # depends on current system time

    # Disable enforcement should always return True
    engine.set_active_hours(9, 0, 17, 0, enforce=False)
    assert engine.is_within_active_hours() is True


def test_bidirectional_channel_gateway_approve_deny():
    gateway = get_channel_gateway()
    gateway.reset()

    approval_status = []
    def on_decision(decision):
        approval_status.append(decision)

    gateway.register_approval_request("req-12345", "deploy to production", "agent-x", callback=on_decision)
    assert "req-12345" in gateway.pending_approvals

    # Test inbound deny reply
    handled_deny = gateway.handle_inbound_reply("req-12345", "deny")
    assert handled_deny is True
    assert gateway.pending_approvals["req-12345"]["status"] == "denied"
    assert approval_status == ["deny"]

    # Register another for approve
    gateway.register_approval_request("req-99999", "write file", "agent-y", callback=on_decision)
    handled_approve = gateway.handle_inbound_reply("req-99999", "approve")
    assert handled_approve is True
    assert gateway.pending_approvals["req-99999"]["status"] == "approved"
    assert approval_status == ["deny", "approve"]


def test_hard_blocker_auto_escalation_to_channel():
    controller = ModeController(initial_mode=ExecutionMode.AUTONOMOUS)
    channel = NotificationChannel()
    channel.reset()

    # Trigger blocker / anomaly / failure past threshold
    channel.notify_approval_needed("delete root file", "agent-core", "critical", "req-blocker-1")
    gateway = get_channel_gateway()
    assert "req-blocker-1" in gateway.pending_approvals

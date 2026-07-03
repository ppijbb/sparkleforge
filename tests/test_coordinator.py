import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.core.trust_gate import TrustContext, TrustLevel
from src.core.session.remote_session import RemoteSession
from src.core.session.coordinator import CoordinatorNode, WorkerNode, NodeStatus
from src.core.scheduler import Scheduler, ScheduleConfig


# --- 1. Worker Pairing & Register/Deregister Tests ---

def test_coordinator_worker_registration():
    coordinator = CoordinatorNode()
    mock_session = MagicMock(spec=RemoteSession)
    
    coordinator.register_worker("worker-1", mock_session)
    
    assert "worker-1" in coordinator.active_workers
    assert coordinator.worker_statuses["worker-1"] == NodeStatus.ONLINE
    assert coordinator.worker_loads["worker-1"] == 0
    assert coordinator.heartbeat_failures["worker-1"] == 0
    
    coordinator.deregister_worker("worker-1")
    assert "worker-1" not in coordinator.active_workers
    assert "worker-1" not in coordinator.worker_statuses


# --- 2. Heartbeat & Failover Recovery Tests ---

@pytest.mark.asyncio
async def test_coordinator_heartbeat_success():
    coordinator = CoordinatorNode()
    mock_session = AsyncMock(spec=RemoteSession)
    mock_session.execute.return_value = {"status": "success", "stdout": "pong"}
    
    coordinator.register_worker("worker-1", mock_session)
    
    # Run check once
    await coordinator._check_heartbeats(max_failures=3)
    
    assert coordinator.worker_statuses["worker-1"] == NodeStatus.ONLINE
    assert coordinator.heartbeat_failures["worker-1"] == 0
    mock_session.execute.assert_called_once_with("ping")


@pytest.mark.asyncio
async def test_coordinator_heartbeat_failure_and_failover():
    coordinator = CoordinatorNode()
    
    # Worker 1 (Failing node)
    mock_session_1 = AsyncMock(spec=RemoteSession)
    mock_session_1.execute.side_effect = Exception("Connection lost")
    
    # Worker 2 (Healthy backup node)
    mock_session_2 = AsyncMock(spec=RemoteSession)
    mock_session_2.execute.return_value = {"status": "success", "stdout": "task completed"}
    
    coordinator.register_worker("worker-1", mock_session_1)
    coordinator.register_worker("worker-2", mock_session_2)
    
    # Assign a task to worker-1 initially
    task_id = "test-task-1"
    task_payload = {"command": "echo 'heavy task'", "timeout": 10.0}
    coordinator.task_assignments[task_id] = "worker-1"
    coordinator.task_payloads[task_id] = task_payload
    coordinator.worker_loads["worker-1"] = 1
    
    # Run check heartbeat which fails 3 times
    await coordinator._check_heartbeats(max_failures=3)
    await coordinator._check_heartbeats(max_failures=3)
    await coordinator._check_heartbeats(max_failures=3)
    
    # Wait briefly for background failover task
    await asyncio.sleep(0.05)
    
    # Worker 1 should be marked OFFLINE
    assert coordinator.worker_statuses["worker-1"] == NodeStatus.OFFLINE
    
    # The task should have been rescheduled to worker-2
    assert coordinator.task_assignments[task_id] == "worker-2"
    mock_session_2.execute.assert_any_call("echo 'heavy task'", timeout=10.0)


# --- 3. Policy Sync Tests ---

@pytest.mark.asyncio
async def test_policy_sync_propagation():
    coordinator = CoordinatorNode()
    
    mock_session_1 = AsyncMock(spec=RemoteSession)
    mock_session_1.send_trust_context.return_value = True
    
    mock_session_2 = AsyncMock(spec=RemoteSession)
    mock_session_2.send_trust_context.return_value = True
    
    coordinator.register_worker("worker-1", mock_session_1)
    coordinator.register_worker("worker-2", mock_session_2)
    
    # Sync a deny list
    deny_names = ["rm", "chmod"]
    deny_prefixes = ["sudo"]
    
    success = await coordinator.sync_policy(deny_names, deny_prefixes)
    assert success is True
    
    # Assert trust context was sent to both online sessions
    mock_session_1.send_trust_context.assert_called_once()
    mock_session_2.send_trust_context.assert_called_once()
    
    sent_trust_1 = mock_session_1.send_trust_context.call_args[0][0]
    assert "rm" in sent_trust_1.deny_names
    assert "sudo" in sent_trust_1.deny_prefixes


# --- 4. Distributed Scheduler Task Routing Tests ---

@pytest.mark.asyncio
async def test_scheduler_routes_via_coordinator():
    coordinator = CoordinatorNode()
    mock_worker_session = AsyncMock(spec=RemoteSession)
    mock_worker_session.execute.return_value = {
        "status": "success",
        "stdout": "delegated schedule executed successfully"
    }
    
    coordinator.register_worker("remote-worker", mock_worker_session)
    
    # Instantiate scheduler with coordinator
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = Scheduler(storage_path=Path(tmpdir), coordinator=coordinator)
        
        schedule = ScheduleConfig(
            schedule_id="sched-123",
            name="test-schedule",
            cron_expression="* * * * *",
            user_query="df -h",
            timeout_seconds=15,
        )
        scheduler.schedules[schedule.schedule_id] = schedule
        
        # Execute the schedule
        await scheduler._execute_schedule(schedule)
        
        # Verify history status
        assert len(scheduler.executions) == 1
        execution = scheduler.executions[0]
        assert execution.status == "completed"
        assert execution.result == {"status": "success", "message": "Delegated execution succeeded."}
        
        # Verify execution actually bypassed local and hit remote worker via coordinator delegation
        mock_worker_session.execute.assert_called_once_with("df -h", timeout=15.0)


# --- 5. WorkerNode Local Handling Tests ---

@pytest.mark.asyncio
async def test_worker_node_blocks_unauthorized_command_locally():
    worker = WorkerNode(worker_id="local-worker")
    
    # Deny tool "rm"
    trust = TrustContext(deny_names=frozenset(["rm"]))
    worker.trust_context = trust
    
    # Execute unauthorized command
    res = await worker.handle_execute("rm -rf /")
    assert res["status"] == "failed"
    assert "Blocked" in res["stderr"]

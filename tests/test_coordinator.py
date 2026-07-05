import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.core.trust_gate import TrustContext, TrustLevel
from src.core.session.remote_session import RemoteSession
from src.core.session.coordinator import CoordinatorNode, WorkerNode, NodeStatus
from src.core.session.secure_envelope import (
    decrypt_credential_envelope,
    encrypt_credential_envelope,
)
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


# --- 5. Node Discovery Tests ---

@pytest.mark.asyncio
async def test_discover_workers_pairs_responsive_nodes():
    coordinator = CoordinatorNode()

    responsive = AsyncMock(spec=RemoteSession)
    responsive.connect.return_value = True

    unresponsive = AsyncMock(spec=RemoteSession)
    unresponsive.connect.side_effect = Exception("Connection refused")

    silent = AsyncMock(spec=RemoteSession)
    silent.connect.return_value = False

    discovered = await coordinator.discover_workers({
        "node-a": responsive,
        "node-b": unresponsive,
        "node-c": silent,
    })

    assert discovered == ["node-a"]
    assert "node-a" in coordinator.active_workers
    assert coordinator.worker_statuses["node-a"] == NodeStatus.ONLINE
    assert "node-b" not in coordinator.active_workers
    assert "node-c" not in coordinator.active_workers


@pytest.mark.asyncio
async def test_discover_workers_skips_already_registered():
    coordinator = CoordinatorNode()
    existing = AsyncMock(spec=RemoteSession)
    coordinator.register_worker("node-a", existing)

    candidate = AsyncMock(spec=RemoteSession)
    discovered = await coordinator.discover_workers({"node-a": candidate})

    assert discovered == []
    candidate.connect.assert_not_called()
    assert coordinator.active_workers["node-a"] is existing


# --- 6. Memory & Credential Delegation Tests ---

@pytest.mark.asyncio
async def test_delegate_memory_to_worker():
    coordinator = CoordinatorNode()
    mock_session = AsyncMock(spec=RemoteSession)
    mock_session.send_payload.return_value = True
    coordinator.register_worker("worker-1", mock_session)

    entries = {"project": "sparkleforge", "phase": "Z"}
    ok = await coordinator.delegate_memory("worker-1", "anvil", entries)

    assert ok is True
    mock_session.send_payload.assert_called_once_with(
        "sync_memory", {"namespace": "anvil", "entries": entries}
    )


@pytest.mark.asyncio
async def test_delegate_memory_refused_for_offline_worker():
    coordinator = CoordinatorNode()
    mock_session = AsyncMock(spec=RemoteSession)
    coordinator.register_worker("worker-1", mock_session)
    coordinator.worker_statuses["worker-1"] = NodeStatus.OFFLINE

    ok = await coordinator.delegate_memory("worker-1", "anvil", {"k": "v"})

    assert ok is False
    mock_session.send_payload.assert_not_called()


@pytest.mark.asyncio
async def test_delegate_credential_with_ttl():
    coordinator = CoordinatorNode()
    mock_session = AsyncMock(spec=RemoteSession)
    mock_session.send_payload.return_value = True
    coordinator.register_worker("worker-1", mock_session, shared_secret="pairing-secret")

    mock_vault = MagicMock()
    mock_vault.retrieve.return_value = "s3cret-token"

    ok = await coordinator.delegate_credential(
        "worker-1", "api_key", ttl_seconds=60.0, vault=mock_vault
    )

    assert ok is True
    action, payload = mock_session.send_payload.call_args[0]
    assert action == "receive_credential"
    assert payload["key"] == "api_key"
    # The plaintext secret must never appear anywhere in the wire payload
    assert "value" not in payload
    assert "s3cret-token" not in json.dumps(payload)
    # The receiving side can open the envelope with the same pairing secret
    opened = decrypt_credential_envelope("pairing-secret", "api_key", payload["envelope"])
    assert opened["value"] == "s3cret-token"
    import time as time_module
    assert opened["expires_at"] > time_module.time()


@pytest.mark.asyncio
async def test_delegate_credential_missing_from_vault():
    coordinator = CoordinatorNode()
    mock_session = AsyncMock(spec=RemoteSession)
    coordinator.register_worker("worker-1", mock_session, shared_secret="pairing-secret")

    mock_vault = MagicMock()
    mock_vault.retrieve.return_value = None

    ok = await coordinator.delegate_credential("worker-1", "missing_key", vault=mock_vault)

    assert ok is False
    mock_session.send_payload.assert_not_called()


@pytest.mark.asyncio
async def test_delegate_credential_uses_shared_vault_from_init():
    mock_vault = MagicMock()
    mock_vault.retrieve.return_value = "s3cret-token"
    coordinator = CoordinatorNode(vault=mock_vault)

    mock_session = AsyncMock(spec=RemoteSession)
    mock_session.send_payload.return_value = True
    coordinator.register_worker("worker-1", mock_session, shared_secret="pairing-secret")

    ok = await coordinator.delegate_credential("worker-1", "api_key")

    assert ok is True
    mock_vault.retrieve.assert_called_once_with("api_key")


@pytest.mark.asyncio
async def test_delegate_credential_fails_without_configured_vault():
    coordinator = CoordinatorNode()
    mock_session = AsyncMock(spec=RemoteSession)
    coordinator.register_worker("worker-1", mock_session, shared_secret="pairing-secret")

    ok = await coordinator.delegate_credential("worker-1", "api_key")

    assert ok is False
    mock_session.send_payload.assert_not_called()


@pytest.mark.asyncio
async def test_delegate_credential_refused_without_pairing_secret():
    mock_vault = MagicMock()
    mock_vault.retrieve.return_value = "s3cret-token"
    coordinator = CoordinatorNode(vault=mock_vault)

    mock_session = AsyncMock(spec=RemoteSession)
    coordinator.register_worker("worker-1", mock_session)

    ok = await coordinator.delegate_credential("worker-1", "api_key")

    assert ok is False
    mock_session.send_payload.assert_not_called()


@pytest.mark.asyncio
async def test_worker_receives_memory_and_credential():
    worker = WorkerNode(worker_id="worker-1", shared_secret="pairing-secret")

    assert await worker.handle_sync_memory("anvil", {"k1": "v1"}) is True
    assert await worker.handle_sync_memory("anvil", {"k2": "v2"}) is True
    assert worker.shared_memory["anvil"] == {"k1": "v1", "k2": "v2"}

    import time as time_module
    envelope = encrypt_credential_envelope(
        "pairing-secret", "api_key", "s3cret", time_module.time() + 60.0
    )
    assert await worker.handle_receive_credential("api_key", envelope) is True
    assert worker.get_delegated_credential("api_key") == "s3cret"


@pytest.mark.asyncio
async def test_worker_discards_expired_credential():
    worker = WorkerNode(worker_id="worker-1", shared_secret="pairing-secret")

    import time as time_module
    # Already-expired handoff is rejected outright
    stale = encrypt_credential_envelope(
        "pairing-secret", "stale_key", "old", time_module.time() - 1.0
    )
    assert await worker.handle_receive_credential("stale_key", stale) is False
    assert worker.get_delegated_credential("stale_key") is None

    # Valid handoff expires after its TTL passes
    short_lived = encrypt_credential_envelope(
        "pairing-secret", "api_key", "s3cret", time_module.time() + 0.05
    )
    assert await worker.handle_receive_credential("api_key", short_lived) is True
    await asyncio.sleep(0.06)
    assert worker.get_delegated_credential("api_key") is None


@pytest.mark.asyncio
async def test_worker_rejects_envelope_with_wrong_secret():
    worker = WorkerNode(worker_id="worker-1", shared_secret="pairing-secret")

    import time as time_module
    envelope = encrypt_credential_envelope(
        "different-secret", "api_key", "s3cret", time_module.time() + 60.0
    )
    assert await worker.handle_receive_credential("api_key", envelope) is False
    assert worker.get_delegated_credential("api_key") is None

    # A worker without a pairing secret rejects any envelope
    unpaired = WorkerNode(worker_id="worker-2")
    valid = encrypt_credential_envelope(
        "pairing-secret", "api_key", "s3cret", time_module.time() + 60.0
    )
    assert await unpaired.handle_receive_credential("api_key", valid) is False


# --- 7. WorkerNode Local Handling Tests ---

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

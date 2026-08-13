"""
tests/test_guard_plane.py — Unit tests for Phase G: Guard (Security & Trust)
"""
import os
import tempfile

import pytest

from src.core.guard.capability_manager import CapabilityManager
from src.core.guard.action_journal import ActionJournal
from src.core.guard.anomaly_detector import AnomalyDetector
from src.core.guard.credential_vault import CredentialVault
from src.core.guard.sandbox_executor import SandboxExecutor


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singletons before each test."""
    CapabilityManager._instance = None
    ActionJournal._instance = None
    AnomalyDetector._instance = None
    CredentialVault._instance = None
    yield
    CapabilityManager._instance = None
    ActionJournal._instance = None
    AnomalyDetector._instance = None
    CredentialVault._instance = None


def test_capability_grant_and_check():
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CapabilityManager(state_path=os.path.join(tmpdir, "caps.json"))
        
        assert not cm.agent_has("agent_1", "read_file")
        cm.grant_agent("agent_1", "read_file")
        assert cm.agent_has("agent_1", "read_file")
        
        cm.revoke_agent("agent_1", "read_file")
        assert not cm.agent_has("agent_1", "read_file")


def test_capability_unknown_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CapabilityManager(state_path=os.path.join(tmpdir, "caps.json"))
        result = cm.grant_agent("agent_1", "nonexistent_capability")
        assert result is False


def test_capability_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CapabilityManager(state_path=os.path.join(tmpdir, "caps.json"))
        cm.grant_agent("agent_1", "read_file")
        cm.grant_agent("agent_1", "write_file")
        caps = cm.get_agent_capabilities("agent_1")
        names = [c.name for c in caps]
        assert "read_file" in names
        assert "write_file" in names


@pytest.mark.asyncio
async def test_check_and_execute_fails_closed_for_hitl_capability():
    """Issue #776: HITLGate.resolve() was never called outside tests, so any
    HIGH/CRITICAL capability requiring HITL always timed out and denied in
    production after a multi-minute hang. HITLGate is removed; the same
    capability must still be denied, just immediately instead of after a wait."""
    from src.core.guard.guard_plane import GuardPlane

    guard = GuardPlane()
    guard.capability_manager.reset()
    guard.capability_manager.grant_agent("agent_1", "execute_shell")

    result = await guard.check_and_execute(
        agent_id="agent_1",
        capability_name="execute_shell",
        command="echo hi",
        description="regression test command",
        dry_run=True,
    )

    assert result["ok"] is False
    assert "approval" in result["error"].lower()


@pytest.mark.asyncio
async def test_check_and_execute_allows_capability_without_hitl():
    from src.core.guard.guard_plane import GuardPlane

    guard = GuardPlane()
    guard.capability_manager.reset()
    guard.capability_manager.grant_agent("agent_1", "read_file")

    result = await guard.check_and_execute(
        agent_id="agent_1",
        capability_name="read_file",
        command="echo hi",
        description="no-hitl capability",
        dry_run=True,
    )

    assert result["ok"] is True


def test_action_journal_record_and_update():
    with tempfile.TemporaryDirectory() as tmpdir:
        journal = ActionJournal(journal_path=os.path.join(tmpdir, "journal.jsonl"))
        
        entry = journal.record(
            agent_id="agent_1",
            action="ls /tmp",
            description="List temp directory",
            risk_level="low",
        )
        assert entry.outcome == "pending"
        
        journal.update_outcome(entry.entry_id, "success")
        recent = journal.recent(limit=5)
        assert any(e.entry_id == entry.entry_id and e.outcome == "success" for e in recent)


def test_action_journal_snapshot_and_rollback():
    with tempfile.TemporaryDirectory() as tmpdir:
        journal = ActionJournal(journal_path=os.path.join(tmpdir, "journal.jsonl"))
        
        pre_state = {"file_content": "original content"}
        entry = journal.record(
            agent_id="agent_1",
            action="write_file",
            description="Overwrite important file",
            risk_level="medium",
            pre_state=pre_state,
        )
        
        # Rollback should return the pre-state
        restored = journal.rollback(entry.entry_id)
        assert restored == pre_state


def test_anomaly_detector_rate_limit():
    detector = AnomalyDetector()
    detector.rate_limit_max_actions = 5
    detector.rate_limit_window_s = 60.0
    
    anomalies = []
    for i in range(6):
        result = detector.observe("agent_1", f"action_{i}")
        anomalies.extend(result)
    
    # Should detect rate limit on the 6th action
    assert len(anomalies) >= 1
    assert any("Rate limit" in a.reason for a in anomalies)


def test_anomaly_detector_forbidden_pattern():
    detector = AnomalyDetector()
    
    anomalies = detector.observe("agent_1", "rm -rf /important/data")
    assert len(anomalies) >= 1
    assert any("Forbidden" in a.reason for a in anomalies)
    assert anomalies[0].severity == "critical"


def test_credential_vault_store_retrieve():
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = CredentialVault(fallback_path=os.path.join(tmpdir, ".creds"))
        
        vault.store("api_key", "supersecret123")
        retrieved = vault.retrieve("api_key")
        assert retrieved == "supersecret123"


def test_credential_vault_delete():
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = CredentialVault(fallback_path=os.path.join(tmpdir, ".creds"))
        vault.store("temp_key", "value123")
        assert vault.retrieve("temp_key") == "value123"
        vault.delete("temp_key")
        vault.reset()  # Clear cache
        assert vault.retrieve("temp_key") is None


def test_sandbox_executor_dry_run():
    executor = SandboxExecutor(timeout_seconds=5.0)
    result = executor.execute("echo hello", dry_run=True)
    assert result.ok
    assert result.sandbox_type == "dry-run"
    assert result.stdout == "[dry-run]"


def test_sandbox_executor_real_command():
    # Generous timeout: on a cold CI runner the docker/firejail backend may need
    # to pull an image before it can run, and there's no unsandboxed fallback
    # to mask that latency anymore (removing that fallback closed a sandbox-escape hole).
    executor = SandboxExecutor(timeout_seconds=60.0)
    result = executor.execute("echo anvil_guard_test")
    assert result.ok
    assert "anvil_guard_test" in result.stdout.strip()


def test_sandbox_executor_env_strategy_subprocess():
    os.environ["SPARKLEFORGE_SANDBOX_STRATEGY"] = "subprocess"
    try:
        executor = SandboxExecutor(timeout_seconds=5.0)
        result = executor.execute("echo env_subprocess_test")
        assert result.ok
        assert result.sandbox_type == "subprocess"
        assert "env_subprocess_test" in result.stdout.strip()
    finally:
        os.environ.pop("SPARKLEFORGE_SANDBOX_STRATEGY", None)


@pytest.mark.asyncio
async def test_bootstrap_guard_plane():
    from src.core.bootstrap_graph import BootstrapGraph
    from src.core.guard.guard_plane import GuardPlane
    
    graph = BootstrapGraph()
    res = await graph.run()
    assert res.ok
    
    stages = [s.name for s in res.stages]
    assert "guard_plane" in stages
    
    stage = next(s for s in res.stages if s.name == "guard_plane")
    assert stage.ok
    assert stage.payload["initialized"] is True
    assert isinstance(stage.payload["guard_plane"], GuardPlane)

"""
tests/test_security_tools.py — GuardPlane/TrustGate agent tools (Anvil Phase Σ-3, #419/#510).
"""
import os
import stat

import pytest

from src.core.guard.action_journal import ActionJournal
from src.core.guard.capability_manager import CapabilityManager
from src.core.guard.security_tools import (
    QUARANTINE_FILE_PARAMETERS,
    REVOKE_CAPABILITY_PARAMETERS,
    quarantine_file,
    register_security_tools,
    revoke_capability,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset guard singletons before/after each test for isolation."""
    ActionJournal._instance = None
    CapabilityManager._instance = None
    yield
    ActionJournal._instance = None
    CapabilityManager._instance = None


@pytest.fixture
def journal(tmp_path):
    return ActionJournal(journal_path=os.path.join(tmp_path, "action_journal.jsonl"))


@pytest.fixture
def capability_manager(tmp_path):
    return CapabilityManager(state_path=os.path.join(tmp_path, "caps.json"))


def test_quarantine_moves_and_strips_exec_bit(tmp_path, journal):
    risky = tmp_path / "risky.sh"
    risky.write_text("#!/bin/bash\nrm -rf /\n")
    risky.chmod(risky.stat().st_mode | stat.S_IEXEC)

    quarantine_dir = tmp_path / "quarantine"
    result = quarantine_file(
        file_path=str(risky),
        reason="dangerous rm -rf command",
        agent_id="test_agent",
        quarantine_dir=str(quarantine_dir),
    )

    assert result["success"] is True
    assert not risky.exists()
    quarantined = result["quarantined_path"]
    assert os.path.exists(quarantined)
    assert not (os.stat(quarantined).st_mode & stat.S_IEXEC)

    entries = journal.recent(limit=10)
    assert any(e.action == "quarantine_file" and e.outcome == "success" for e in entries)
    # A snapshot must exist so the quarantine can be rolled back.
    quarantine_entry = next(e for e in entries if e.action == "quarantine_file")
    assert quarantine_entry.snapshot_id is not None
    snapshot = journal.get_snapshot(quarantine_entry.snapshot_id)
    assert snapshot is not None
    assert snapshot.state["original_path"] == str(risky)


def test_quarantine_missing_file_fails_cleanly(tmp_path, journal):
    result = quarantine_file(file_path=str(tmp_path / "does_not_exist.sh"))
    assert result["success"] is False
    assert result["quarantined_path"] is None


def test_quarantine_leaves_benign_file_alone(tmp_path, journal):
    benign = tmp_path / "app.py"
    benign.write_text("print('hello world')\n")
    original_mode = benign.stat().st_mode

    # Only risky.sh is quarantined; app.py must never be touched by this call.
    other = tmp_path / "risky.sh"
    other.write_text("echo hi\n")
    quarantine_file(file_path=str(other), quarantine_dir=str(tmp_path / "q"))

    assert benign.exists()
    assert benign.stat().st_mode == original_mode


def test_revoke_capability_actually_revokes(journal, capability_manager):
    capability_manager.grant_agent("agent_1", "execute_shell")
    assert capability_manager.agent_has("agent_1", "execute_shell")

    result = revoke_capability("agent_1", "execute_shell", reason="suspicious behavior detected")

    assert result["success"] is True
    assert not capability_manager.agent_has("agent_1", "execute_shell")

    entries = journal.recent(limit=10)
    assert any(e.action == "revoke_capability" for e in entries)


def test_register_security_tools_adds_both_tools():
    from src.core.tools.registry import ToolRegistry

    import src.core.tools.registry as registry_module

    test_registry = ToolRegistry()
    original = registry_module.registry
    registry_module.registry = test_registry
    try:
        register_security_tools()
        assert "quarantine_file" in test_registry.tools
        assert "revoke_capability" in test_registry.tools
        assert test_registry.tools["quarantine_file"].parameters == QUARANTINE_FILE_PARAMETERS
        assert test_registry.tools["revoke_capability"].parameters == REVOKE_CAPABILITY_PARAMETERS
    finally:
        registry_module.registry = original

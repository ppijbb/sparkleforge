"""
tests/test_security_tools.py — GuardPlane/TrustGate agent tools (Anvil Phase Σ-3, #419/#510).
"""
import asyncio
import os
import stat
from pathlib import Path

import pytest

import src.core.guard.security_tools as security_tools
from src.core.guard.action_journal import ActionJournal
from src.core.guard.capability_manager import CapabilityManager
from src.core.guard.security_tools import (
    QUARANTINE_FILE_PARAMETERS,
    REVOKE_CAPABILITY_PARAMETERS,
    _quarantine_file_tool,
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


@pytest.fixture(autouse=True)
def quarantine_base(tmp_path, monkeypatch):
    """Confine QUARANTINE_BASE to a per-test tmp dir instead of the real repo's data/quarantine/."""
    base = tmp_path / "quarantine_root"
    monkeypatch.setattr(security_tools, "QUARANTINE_BASE", base)
    return base


@pytest.fixture
def journal(tmp_path):
    return ActionJournal(journal_path=os.path.join(tmp_path, "action_journal.jsonl"))


@pytest.fixture
def capability_manager(tmp_path):
    return CapabilityManager(state_path=os.path.join(tmp_path, "caps.json"))


def test_quarantine_moves_and_strips_exec_bit(tmp_path, journal, quarantine_base):
    risky = tmp_path / "risky.sh"
    risky.write_text("#!/bin/bash\nrm -rf /\n")
    risky.chmod(risky.stat().st_mode | stat.S_IEXEC)

    result = quarantine_file(
        file_path=str(risky),
        reason="dangerous rm -rf command",
        agent_id="test_agent",
    )

    assert result["success"] is True
    assert not risky.exists()
    quarantined = result["quarantined_path"]
    assert os.path.exists(quarantined)
    assert quarantine_base in Path(quarantined).parents
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
    quarantine_file(file_path=str(other))

    assert benign.exists()
    assert benign.stat().st_mode == original_mode


def test_quarantine_dir_cannot_escape_quarantine_base(tmp_path, journal, quarantine_base):
    risky = tmp_path / "risky.sh"
    risky.write_text("echo hi\n")

    result = quarantine_file(
        file_path=str(risky),
        quarantine_dir="../../../etc",
    )

    assert result["success"] is False
    assert risky.exists()


def test_quarantine_refuses_symlinks(tmp_path, journal, quarantine_base):
    target = tmp_path / "real_target.txt"
    target.write_text("do not touch")
    link = tmp_path / "link_to_target"
    link.symlink_to(target)

    result = quarantine_file(file_path=str(link))

    assert result["success"] is False
    assert target.exists()
    assert link.exists()


def test_quarantine_skips_content_snapshot_for_oversized_file(tmp_path, journal, monkeypatch):
    monkeypatch.setattr(security_tools, "MAX_SNAPSHOT_BYTES", 10)
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 100)

    result = quarantine_file(file_path=str(big))

    assert result["success"] is True
    quarantine_entry = next(e for e in journal.recent(limit=10) if e.action == "quarantine_file")
    snapshot = journal.get_snapshot(quarantine_entry.snapshot_id)
    assert snapshot.state["content_b64"] is None
    assert snapshot.state["size"] == 100


def test_quarantine_file_tool_offloads_blocking_io(tmp_path, journal, quarantine_base):
    risky = tmp_path / "risky.sh"
    risky.write_text("echo hi\n")

    result = asyncio.run(_quarantine_file_tool(file_path=str(risky)))

    assert result["success"] is True


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

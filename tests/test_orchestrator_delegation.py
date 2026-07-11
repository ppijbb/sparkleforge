"""
tests/test_orchestrator_delegation.py — Runtime sub-agent delegation (Anvil Phase Σ-2, #495/#509).
"""
import os
import tempfile

import pytest

from src.core.guard.action_journal import ActionJournal
from src.core.orchestrator.delegation import (
    DELEGATION_REGISTRY,
    DelegationDepthExceeded,
    delegate_to_agent,
)


@pytest.fixture(autouse=True)
def reset_action_journal():
    """Reset the ActionJournal singleton so each test gets an isolated journal file."""
    ActionJournal._instance = None
    yield
    ActionJournal._instance = None


@pytest.fixture
def journal(tmp_path):
    return ActionJournal(journal_path=os.path.join(tmp_path, "action_journal.jsonl"))


async def _noop_adapter(task, context):
    return {"echo": task, "context": context}


def test_registry_covers_expected_roles():
    for role in (
        "research_agent",
        "validation_agent",
        "verifier_agent",
        "evaluation_agent",
        "codebase_agent",
        "document_organizer_agent",
    ):
        assert role in DELEGATION_REGISTRY


@pytest.mark.asyncio
async def test_unknown_role_rejected_and_journaled(journal):
    state = {"delegation_depth": 0, "max_delegation_depth": 3}
    with pytest.raises(ValueError):
        await delegate_to_agent(state, "not_a_real_role", {"id": "t1"})

    entries = journal.recent(limit=10)
    assert any(e.action == "delegate_to_agent_rejected" for e in entries)


@pytest.mark.asyncio
async def test_depth_guard_blocks_and_journals_denial(journal, monkeypatch):
    monkeypatch.setitem(DELEGATION_REGISTRY, "fake_role", _noop_adapter)
    state = {"delegation_depth": 3, "max_delegation_depth": 3}

    with pytest.raises(DelegationDepthExceeded):
        await delegate_to_agent(state, "fake_role", {"id": "t1"})

    entries = journal.recent(limit=10)
    assert any(e.action == "delegate_to_agent_denied" for e in entries)
    # A denied delegation must not mutate the caller's depth counter.
    assert state["delegation_depth"] == 3


@pytest.mark.asyncio
async def test_successful_delegation_increments_then_restores_depth(journal, monkeypatch):
    monkeypatch.setitem(DELEGATION_REGISTRY, "fake_role", _noop_adapter)
    state = {"delegation_depth": 0, "max_delegation_depth": 3}

    result = await delegate_to_agent(state, "fake_role", {"id": "t1"}, context={"foo": "bar"})

    assert result["success"] is True
    assert result["result"]["echo"] == {"id": "t1"}
    assert result["result"]["context"]["foo"] == "bar"
    # depth is restored on the caller's state once the delegated call returns
    assert state["delegation_depth"] == 0

    entries = journal.recent(limit=10)
    success_entries = [e for e in entries if e.action == "delegate_to_agent"]
    assert success_entries and success_entries[0].outcome == "success"


@pytest.mark.asyncio
async def test_failed_delegation_is_journaled_as_failure(journal, monkeypatch):
    async def _boom(task, context):
        raise RuntimeError("adapter exploded")

    monkeypatch.setitem(DELEGATION_REGISTRY, "fake_role", _boom)
    state = {"delegation_depth": 0, "max_delegation_depth": 3}

    result = await delegate_to_agent(state, "fake_role", {"id": "t1"})

    assert result["success"] is False
    assert "adapter exploded" in result["error"]
    assert state["delegation_depth"] == 0

    entries = journal.recent(limit=10)
    failed = [e for e in entries if e.action == "delegate_to_agent" and e.outcome == "failure"]
    assert failed

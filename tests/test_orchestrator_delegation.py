"""
tests/test_orchestrator_delegation.py — Runtime sub-agent delegation (Anvil Phase Σ-2, #495/#509).
"""
import asyncio
import os

import pytest

from src.core.guard.action_journal import ActionJournal
from src.core.guard.invocation_gateway import InvocationDecision
from src.core.orchestrator.delegation import (
    DELEGATION_REGISTRY,
    DelegationDenied,
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
async def test_invocation_gateway_denial_blocks_delegation_before_dispatch(monkeypatch):
    """Issue #568: delegate_to_agent must route through InvocationGateway,
    and a denial must stop execution before the adapter ever runs."""
    adapter_called = False

    async def _adapter(task, context):
        nonlocal adapter_called
        adapter_called = True
        return {"ok": True}

    monkeypatch.setitem(DELEGATION_REGISTRY, "gated_role", _adapter)

    import src.core.orchestrator.delegation as delegation_module

    fake_gateway = type(
        "FakeGateway",
        (),
        {"authorize": lambda self, **kwargs: InvocationDecision(allowed=False, reason="test denial")},
    )()
    monkeypatch.setattr(delegation_module, "get_invocation_gateway", lambda: fake_gateway)

    state = {"delegation_depth": 0, "max_delegation_depth": 3}
    with pytest.raises(DelegationDenied):
        await delegate_to_agent(state, "gated_role", {"id": "t1"})

    assert adapter_called is False


@pytest.mark.asyncio
async def test_depth_guard_blocks_and_journals_denial(journal, monkeypatch):
    monkeypatch.setitem(DELEGATION_REGISTRY, "fake_role", _noop_adapter)
    state = {}

    with pytest.raises(DelegationDepthExceeded):
        await delegate_to_agent(state, "fake_role", {"id": "t1"}, context={"delegation_depth": 3, "max_delegation_depth": 3})

    entries = journal.recent(limit=10)
    assert any(e.action == "delegate_to_agent_denied" for e in entries)


@pytest.mark.asyncio
async def test_successful_delegation_never_mutates_shared_state(journal, monkeypatch):
    monkeypatch.setitem(DELEGATION_REGISTRY, "fake_role", _noop_adapter)
    state = {"max_delegation_depth": 3}

    result = await delegate_to_agent(state, "fake_role", {"id": "t1"}, context={"foo": "bar"})

    assert result["success"] is True
    assert result["result"]["echo"] == {"id": "t1"}
    assert result["result"]["context"]["foo"] == "bar"
    assert result["result"]["context"]["delegation_depth"] == 1
    # issue #516: delegate_to_agent must not touch the shared state object at
    # all -- depth lives entirely in the per-call context now.
    assert state == {"max_delegation_depth": 3}

    entries = journal.recent(limit=10)
    success_entries = [e for e in entries if e.action == "delegate_to_agent"]
    assert success_entries and success_entries[0].outcome == "success"


@pytest.mark.asyncio
async def test_failed_delegation_is_journaled_as_failure(journal, monkeypatch):
    async def _boom(task, context):
        raise RuntimeError("adapter exploded")

    monkeypatch.setitem(DELEGATION_REGISTRY, "fake_role", _boom)
    state = {}

    result = await delegate_to_agent(state, "fake_role", {"id": "t1"})

    assert result["success"] is False
    assert "adapter exploded" in result["error"]

    entries = journal.recent(limit=10)
    failed = [e for e in entries if e.action == "delegate_to_agent" and e.outcome == "failure"]
    assert failed


@pytest.mark.asyncio
async def test_nested_delegation_chain_is_actually_bounded(journal, monkeypatch):
    """Regression test for issue #516.

    The bug: depth was read from `state`, but nested calls only received the
    incremented value via `context`, and the parent's `finally` block
    restored `state` before the nested call could observe it -- so a chain
    of nested delegations always saw depth==0 and the guard never triggered.
    This drives an actual 3-hop chain, passing each call's received context
    into the next, and asserts the 4th hop is refused.
    """
    captured_contexts = []

    async def _recording_adapter(task, context):
        captured_contexts.append(context)
        return "ok"

    monkeypatch.setitem(DELEGATION_REGISTRY, "fake_role", _recording_adapter)
    state = {"max_delegation_depth": 3}

    context = None
    for expected_depth in (1, 2, 3):
        result = await delegate_to_agent(state, "fake_role", {"id": "t"}, context=context)
        assert result["success"] is True
        assert result["delegation_depth"] == expected_depth
        # Simulate the delegated agent itself delegating further by handing
        # the context it received straight back in as the next call's context.
        context = captured_contexts[-1]

    with pytest.raises(DelegationDepthExceeded):
        await delegate_to_agent(state, "fake_role", {"id": "t"}, context=context)

    # The shared state object was never touched across the whole chain.
    assert state == {"max_delegation_depth": 3}


@pytest.mark.asyncio
async def test_concurrent_sibling_delegations_do_not_race_on_shared_state(journal, monkeypatch):
    """Regression test for issue #516's race-condition half of the bug.

    Two independent top-level delegations sharing the same `state` object
    (as they would under asyncio.gather from execute_research) must each see
    their own depth==1, not clobber each other via a shared counter.
    """
    async def _recording_adapter(task, context):
        return context["delegation_depth"]

    monkeypatch.setitem(DELEGATION_REGISTRY, "fake_role", _recording_adapter)
    state = {"max_delegation_depth": 3}

    results = await asyncio.gather(
        delegate_to_agent(state, "fake_role", {"id": "a"}),
        delegate_to_agent(state, "fake_role", {"id": "b"}),
    )

    assert [r["result"] for r in results] == [1, 1]

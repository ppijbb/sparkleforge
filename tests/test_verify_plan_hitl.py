"""Tests for HITL plan feedback checkpoint provider.

These tests ensure the monkeypatched ``plan_feedback_provider`` mocks match the
real async contract defined in ``hitl_feedback.py``. The real provider is an
``async def`` function, so synchronous lambdas would return a coroutine object
instead of a ``tuple[CheckpointDecision, str]`` when awaited, masking real
production failures behind false-positive green tests.
"""

from __future__ import annotations

import pytest

from src.core.anvil.hitl_checkpoint import CheckpointDecision, HITLCheckpointManager
from src.core.anvil import hitl_feedback
from src.core.anvil.hitl_feedback import plan_feedback_provider


pytestmark = pytest.mark.asyncio


async def _async_approve_provider(stage, context):
    return (CheckpointDecision.APPROVE, "")


async def _async_reject_provider(stage, context):
    return (CheckpointDecision.REJECT, "plan needs more detail")


async def test_checkpoint_approve_uses_async_provider(monkeypatch):
    """The checkpoint manager must await an async provider and return APPROVE."""
    manager = HITLCheckpointManager()

    async def _provider(stage, context):
        return (CheckpointDecision.APPROVE, "")

    monkeypatch.setattr(hitl_feedback, "plan_feedback_provider", _provider)

    decision, message = await manager.checkpoint("plan", {"plan": "stub"})

    assert decision is CheckpointDecision.APPROVE
    assert message == ""


async def test_checkpoint_reject_uses_async_provider(monkeypatch):
    """The checkpoint manager must await an async provider and return REJECT."""
    manager = HITLCheckpointManager()

    async def _provider(stage, context):
        return (CheckpointDecision.REJECT, "plan needs more detail")

    monkeypatch.setattr(hitl_feedback, "plan_feedback_provider", _provider)

    decision, message = await manager.checkpoint("plan", {"plan": "stub"})

    assert decision is CheckpointDecision.REJECT
    assert message == "plan needs more detail"


async def test_checkpoint_provider_is_awaited(monkeypatch):
    """Regression guard: the provider must be awaited, not called synchronously."""
    manager = HITLCheckpointManager()

    call_count = {"count": 0}

    async def _provider(stage, context):
        call_count["count"] += 1
        return (CheckpointDecision.APPROVE, "")

    monkeypatch.setattr(hitl_feedback, "plan_feedback_provider", _provider)

    await manager.checkpoint("plan", {"plan": "stub"})

    assert call_count["count"] == 1


async def test_checkpoint_rejects_synchronous_lambda(monkeypatch):
    """A synchronous lambda must not silently pass through the await path.

    If someone reintroduces a synchronous lambda mock, awaiting it returns a
    coroutine object (or raises ``TypeError``), not a tuple. This test fails
    loudly so the false-positive coverage cannot return.
    """
    manager = HITLCheckpointManager()

    monkeypatch.setattr(
        hitl_feedback,
        "plan_feedback_provider",
        lambda stage, context: (CheckpointDecision.APPROVE, ""),
    )

    with pytest.raises((TypeError, AttributeError)):
        await manager.checkpoint("plan", {"plan": "stub"})


async def test_real_plan_feedback_provider_is_async():
    """The real provider must be a coroutine function (async def)."""
    import inspect

    assert inspect.iscoroutinefunction(plan_feedback_provider), (
        "plan_feedback_provider must be async def so awaiting it returns a tuple, "
        "not a coroutine object."
    )


async def test_checkpoint_with_real_provider_approves(monkeypatch):
    """End-to-end validation through the real async provider call path.

    We monkeypatch only the underlying feedback source so the real
    ``plan_feedback_provider`` coroutine runs through ``checkpoint()``.
    """
    manager = HITLCheckpointManager()

    async def _fake_source(stage, context):
        return (CheckpointDecision.APPROVE, "")

    monkeypatch.setattr(
        hitl_feedback,
        "_collect_plan_feedback",
        _fake_source,
        raising=False,
    )

    decision, message = await manager.checkpoint("plan", {"plan": "stub"})

    assert decision is CheckpointDecision.APPROVE
    assert isinstance(message, str)

"""Anvil M4 follow-up (issue #612): HITLCheckpointManager wired into verify_plan.

Covers that the AFTER_PLANNING checkpoint is skipped for headless/autopilot
runs (zero behavior change for automation) and drives plan_approved/
current_step/should_continue correctly for APPROVE/REVISE/ABORT when a human
is actually attached.
"""

import json

import pytest

from src.core.anvil.hitl_checkpoint import CheckpointDecision
from src.core.orchestrator.verification import VerificationNode


async def _approved_llm_result(**kwargs):
    from types import SimpleNamespace

    return SimpleNamespace(
        content=json.dumps({"approved": True, "confidence": 0.9, "feedback": "looks good"})
    )


def _base_state(**overrides):
    state = {
        "user_request": "do the thing",
        "planned_tasks": [{"task_id": "task_1", "name": "Do the thing"}],
        "execution_plan": {"strategy": "sequential"},
        "plan_iteration": 0,
        "plan_feedback": None,
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_autopilot_mode_skips_human_checkpoint(monkeypatch):
    monkeypatch.setattr(
        "src.core.orchestrator.verification.execute_llm_task",
        _approved_llm_result,
    )
    monkeypatch.setattr("src.core.orchestrator.verification.is_interactive", lambda: True)

    node = VerificationNode()

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("HITL checkpoint must not run in autopilot mode")

    monkeypatch.setattr(node, "_run_human_plan_checkpoint", fail_if_called)

    state = _base_state(autopilot_mode=True)
    result = await node.verify_plan(state)

    assert result["plan_approved"] is True
    assert result["current_step"] == "overseer_initial_review"


@pytest.mark.asyncio
async def test_non_interactive_session_skips_human_checkpoint(monkeypatch):
    monkeypatch.setattr(
        "src.core.orchestrator.verification.execute_llm_task",
        _approved_llm_result,
    )
    monkeypatch.setattr("src.core.orchestrator.verification.is_interactive", lambda: False)

    node = VerificationNode()

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("HITL checkpoint must not run without a TTY")

    monkeypatch.setattr(node, "_run_human_plan_checkpoint", fail_if_called)

    state = _base_state(autopilot_mode=False)
    result = await node.verify_plan(state)

    assert result["plan_approved"] is True
    assert result["current_step"] == "overseer_initial_review"


@pytest.mark.asyncio
async def test_interactive_human_approves(monkeypatch):
    monkeypatch.setattr(
        "src.core.orchestrator.verification.execute_llm_task",
        _approved_llm_result,
    )
    monkeypatch.setattr("src.core.orchestrator.verification.is_interactive", lambda: True)
    monkeypatch.setattr(
        "src.core.orchestrator.verification.plan_feedback_provider",
        lambda stage, context: (CheckpointDecision.APPROVE, ""),
    )

    node = VerificationNode()
    state = _base_state(autopilot_mode=False)
    result = await node.verify_plan(state)

    assert result["plan_approved"] is True
    assert result["current_step"] == "overseer_initial_review"
    assert result.get("should_continue", True) is not False


@pytest.mark.asyncio
async def test_interactive_human_revises_routes_back_to_planning(monkeypatch):
    monkeypatch.setattr(
        "src.core.orchestrator.verification.execute_llm_task",
        _approved_llm_result,
    )
    monkeypatch.setattr("src.core.orchestrator.verification.is_interactive", lambda: True)
    monkeypatch.setattr(
        "src.core.orchestrator.verification.plan_feedback_provider",
        lambda stage, context: (CheckpointDecision.REVISE, "Add requirement: cover edge cases"),
    )

    node = VerificationNode()
    state = _base_state(autopilot_mode=False)
    result = await node.verify_plan(state)

    assert result["plan_approved"] is False
    assert result["current_step"] == "planning_agent"
    assert result["plan_feedback"] == "Add requirement: cover edge cases"


@pytest.mark.asyncio
async def test_interactive_human_aborts_halts_the_run(monkeypatch):
    monkeypatch.setattr(
        "src.core.orchestrator.verification.execute_llm_task",
        _approved_llm_result,
    )
    monkeypatch.setattr("src.core.orchestrator.verification.is_interactive", lambda: True)
    monkeypatch.setattr(
        "src.core.orchestrator.verification.plan_feedback_provider",
        lambda stage, context: (CheckpointDecision.ABORT, "user said stop"),
    )

    node = VerificationNode()
    state = _base_state(autopilot_mode=False)
    result = await node.verify_plan(state)

    assert result["plan_approved"] is False
    assert result["should_continue"] is False
    assert result["current_step"] == "aborted_by_user"
    assert result["error_message"] == "user said stop"


@pytest.mark.asyncio
async def test_checkpoint_context_carries_task_names(monkeypatch):
    monkeypatch.setattr(
        "src.core.orchestrator.verification.execute_llm_task",
        _approved_llm_result,
    )
    monkeypatch.setattr("src.core.orchestrator.verification.is_interactive", lambda: True)

    seen_context = {}

    def capturing_provider(stage, context):
        seen_context.update(context)
        return CheckpointDecision.APPROVE, ""

    monkeypatch.setattr(
        "src.core.orchestrator.verification.plan_feedback_provider", capturing_provider
    )

    node = VerificationNode()
    state = _base_state(
        autopilot_mode=False,
        planned_tasks=[
            {"task_id": "task_1", "name": "Collect sources"},
            {"task_id": "task_2", "name": "Draft summary"},
        ],
    )
    await node.verify_plan(state)

    assert seen_context["task_count"] == 2
    assert seen_context["strategy"] == "sequential"
    assert seen_context["task_names"] == ["Collect sources", "Draft summary"]

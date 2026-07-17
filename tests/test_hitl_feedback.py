"""Direct coverage for the interactive AFTER_PLANNING feedback provider."""

import pytest

from src.core.anvil.hitl_checkpoint import CheckpointDecision, CheckpointStage
from src.core.orchestrator import hitl_feedback


def _patch_prompts(monkeypatch, answers):
    calls = iter(answers)
    monkeypatch.setattr(hitl_feedback.Prompt, "ask", lambda *a, **k: next(calls))


@pytest.mark.asyncio
async def test_approve_choice(monkeypatch):
    _patch_prompts(monkeypatch, ["1"])
    decision, feedback = await hitl_feedback.plan_feedback_provider(
        CheckpointStage.AFTER_PLANNING, {"task_count": 1}
    )
    assert decision == CheckpointDecision.APPROVE
    assert feedback == ""


@pytest.mark.asyncio
async def test_abort_choice(monkeypatch):
    _patch_prompts(monkeypatch, ["4"])
    decision, feedback = await hitl_feedback.plan_feedback_provider(
        CheckpointStage.AFTER_PLANNING, {}
    )
    assert decision == CheckpointDecision.ABORT
    assert feedback


@pytest.mark.asyncio
async def test_add_requirement_choice(monkeypatch):
    _patch_prompts(monkeypatch, ["2", "cover the auth flow"])
    decision, feedback = await hitl_feedback.plan_feedback_provider(
        CheckpointStage.AFTER_PLANNING, {}
    )
    assert decision == CheckpointDecision.REVISE
    assert feedback == "Add requirement: cover the auth flow"


@pytest.mark.asyncio
async def test_flag_task_choice(monkeypatch):
    _patch_prompts(monkeypatch, ["3", "task_2 duplicates task_1"])
    decision, feedback = await hitl_feedback.plan_feedback_provider(
        CheckpointStage.AFTER_PLANNING, {}
    )
    assert decision == CheckpointDecision.REVISE
    assert feedback == "Fix/remove task: task_2 duplicates task_1"


def test_is_interactive_reflects_stdin(monkeypatch):
    monkeypatch.setattr(hitl_feedback.sys.stdin, "isatty", lambda: True)
    assert hitl_feedback.is_interactive() is True

    monkeypatch.setattr(hitl_feedback.sys.stdin, "isatty", lambda: False)
    assert hitl_feedback.is_interactive() is False

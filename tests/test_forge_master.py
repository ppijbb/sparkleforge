"""Integration tests for Forge Master subsystem (Router, SessionManager, Controller, Delegation)."""

from unittest.mock import AsyncMock, patch

import pytest
from src.core.forge_master import (
    ForgeMasterController,
    ForgeMasterRouter,
    ForgeMasterSessionManager,
)
from src.core.orchestrator.delegation import DELEGATION_REGISTRY, delegate_to_agent


def test_forge_master_router_capability_matching():
    router = ForgeMasterRouter()

    # Refactoring task -> Claude Code
    assignment1 = router.route_task("Perform architectural refactoring of backend modules")
    assert assignment1.agent_name == "claude_code"
    assert "Claude Code" in assignment1.assigned_goal

    # Code gen task -> Codex
    assignment2 = router.route_task("Generate helper functions and fix syntax")
    assert assignment2.agent_name == "codex"

    # Search task -> Gemini CLI
    assignment3 = router.route_task("Search large context documentation and explain setup")
    assert assignment3.agent_name == "gemini_cli"


def test_router_fallbacks_are_relevance_gated_not_the_whole_pool():
    """Fallback allocation must reflect actual task fit, not just static baseline
    score, or every failure ends up cascading into claude_code/codex regardless
    of whether the task ever needed them."""
    router = ForgeMasterRouter()

    # A gemini-shaped task shouldn't fall back to claude_code/codex just because
    # they happen to have the highest static capability scores in the matrix.
    search_assignment = router.route_task(
        "Search large context documentation and explain setup"
    )
    assert search_assignment.agent_name == "gemini_cli"
    assert "claude_code" not in search_assignment.fallback_agents
    assert "codex" not in search_assignment.fallback_agents

    # Same when the agent is pinned explicitly rather than auto-selected.
    pinned_assignment = router.route_task(
        "In one short paragraph, explain what a binary search algorithm does.",
        preferred_agent="gemini_cli",
    )
    assert pinned_assignment.agent_name == "gemini_cli"
    assert pinned_assignment.fallback_agents == []

    # A refactor-shaped task shouldn't get codex/gemini as noise fallbacks either.
    refactor_assignment = router.route_task(
        "Perform architectural refactoring of backend modules"
    )
    assert refactor_assignment.agent_name == "claude_code"
    assert refactor_assignment.fallback_agents == []


@pytest.mark.asyncio
async def test_router_async_uses_llm_decision_when_available():
    """route_task_async must let the LLM actually decide, not just run the
    keyword heuristic under a new name."""
    router = ForgeMasterRouter()

    mock_result = type(
        "Result",
        (),
        {
            "content": '{"agent": "gemini_cli", "fallback_agents": [], "reason": "best fit for doc search"}'
        },
    )()
    mock_orchestrator = type("Orch", (), {})()
    mock_orchestrator.execute_with_model = AsyncMock(return_value=mock_result)

    with patch("src.core.llm_manager.get_llm_orchestrator", return_value=mock_orchestrator):
        assignment = await router.route_task_async("Some deliberately ambiguous task text")

    assert assignment.agent_name == "gemini_cli"
    assert assignment.fallback_agents == []
    assert "LLM routing" in assignment.capability_reason
    mock_orchestrator.execute_with_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_router_async_falls_back_to_heuristic_when_llm_fails():
    """LLM routing failure (all retries) must fall back to the deterministic
    heuristic, not raise or return a bad assignment."""
    router = ForgeMasterRouter()

    mock_orchestrator = type("Orch", (), {})()
    mock_orchestrator.execute_with_model = AsyncMock(side_effect=RuntimeError("network down"))

    with patch("src.core.llm_manager.get_llm_orchestrator", return_value=mock_orchestrator), \
         patch("src.core.forge_master.router.asyncio.sleep", new=AsyncMock()):
        assignment = await router.route_task_async(
            "Perform architectural refactoring of backend modules"
        )

    assert assignment.agent_name == "claude_code"
    assert "Heuristic fallback" in assignment.capability_reason
    assert mock_orchestrator.execute_with_model.await_count == ForgeMasterRouter.MAX_ROUTE_RETRIES


@pytest.mark.asyncio
async def test_router_async_respects_preferred_agent_without_llm_call():
    """An explicit preferred_agent is the caller's own judgment call - honor it
    without spending an LLM call on a decision that's already been made."""
    router = ForgeMasterRouter()

    mock_orchestrator = type("Orch", (), {})()
    mock_orchestrator.execute_with_model = AsyncMock()

    with patch("src.core.llm_manager.get_llm_orchestrator", return_value=mock_orchestrator):
        assignment = await router.route_task_async(
            "Some task", preferred_agent="gemini_cli"
        )

    assert assignment.agent_name == "gemini_cli"
    mock_orchestrator.execute_with_model.assert_not_awaited()


def test_session_manager_lifecycle():
    mgr = ForgeMasterSessionManager()

    # Create persistent 24/7 session
    sess = mgr.create_session("codex", is_persistent=True)
    assert sess.is_persistent is True
    assert sess.session_id in mgr.sessions

    # Close session
    closed = mgr.close_session(sess.session_id)
    assert closed is True
    assert sess.session_id not in mgr.sessions


@pytest.mark.asyncio
async def test_controller_execution_with_mocked_cli():
    controller = ForgeMasterController()

    mock_cli_result = {
        "success": True,
        "response": "```python\ndef calculate(a: int, b: int) -> int:\n    return a * b\n```\nSuccessfully calculated.",
        "confidence": 0.90,
    }

    with patch.object(
        controller.session_manager.cli_manager,
        "execute_with_agent",
        new=AsyncMock(return_value=mock_cli_result),
    ):
        result = await controller.execute_task_with_master_control(
            task_query="Write calculate function",
            preferred_agent="codex",
        )

        assert result["success"] is True
        assert result["master_verdict"] == "PASSED"
        assert result["agent_used"] == "codex"
        assert result["adversarial_audit"]["passed"] is True


@pytest.mark.asyncio
async def test_delegation_registry_contains_forge_master():
    assert "forge_master" in DELEGATION_REGISTRY

    mock_task = {"description": "Perform meta-orchestration audit"}
    mock_state = {"max_delegation_depth": 3}

    mock_controller_result = {
        "success": True,
        "master_verdict": "PASSED",
        "agent_used": "claude_code",
        "response": "Audit passed.",
    }

    with patch(
        "src.core.forge_master.controller.ForgeMasterController.execute_task_with_master_control",
        new=AsyncMock(return_value=mock_controller_result),
    ):
        result = await delegate_to_agent(
            state=mock_state,
            role="forge_master",
            task=mock_task,
            context={},
        )

        assert result["success"] is True
        assert result["role"] == "forge_master"
        assert result["result"]["master_verdict"] == "PASSED"

"""Integration tests for Forge Master subsystem (Router, SessionManager, Controller, Delegation)."""

from unittest.mock import AsyncMock, patch

import pytest
from src.core.forge_master import (
    EXECUTION_PERSONAS,
    ForgeMasterController,
    ForgeMasterRouter,
    ForgeMasterSessionManager,
    apply_persona,
)
from src.core.forge_master.tools import register_forge_master_dispatch_tool
from src.core.orchestrator.delegation import DELEGATION_REGISTRY, delegate_to_agent
from src.core.tools.registry import registry


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
async def test_controller_does_not_auto_switch_agents_on_critical_failure():
    """A crash (ESCALATE_TO_FALLBACK) must not make the controller silently
    call a different CLI agent - which agent to use next is a decision for
    whoever calls execute_task_with_master_control (e.g. the dispatch_to_cli_agent
    tool), not something this code should do on its own."""
    controller = ForgeMasterController()

    crash_result = {"success": False, "error": "boom", "response": ""}

    with patch.object(
        controller.session_manager.cli_manager,
        "execute_with_agent",
        new=AsyncMock(return_value=crash_result),
    ) as mock_execute:
        result = await controller.execute_task_with_master_control(
            task_query="Perform architectural refactoring of backend modules",
            preferred_agent="gemini_cli",
        )

    assert result["success"] is False
    assert result["last_agent_used"] == "gemini_cli"
    assert result["attempts"] == 1

    # Every attempted call must target the one explicitly chosen agent - no
    # silent switch to claude_code/codex even though this refactor-shaped
    # query would score them as relevant fallback_candidates.
    assert mock_execute.await_count == 1
    called_agent = mock_execute.await_args.kwargs.get("agent_name")
    assert called_agent == "gemini_cli"
    assert "claude_code" in result["fallback_candidates"]


def test_dispatch_to_cli_agent_tool_is_registered():
    """The agent's own reasoning loop must be able to discover and call this
    tool to pick a CLI agent itself - it must not only exist as internal
    Python code nobody outside forge_master can reach."""
    register_forge_master_dispatch_tool()

    assert "dispatch_to_cli_agent" in registry.get_all_tool_names()
    assert registry.tool_sources["dispatch_to_cli_agent"] == "local"

    schema = registry.tools["dispatch_to_cli_agent"].parameters
    assert schema["required"] == ["agent_name", "task_query"]
    assert set(ForgeMasterRouter.CAPABILITY_MATRIX.keys()) == set(
        schema["properties"]["agent_name"]["enum"]
    )


@pytest.mark.asyncio
async def test_dispatch_to_cli_agent_tool_executes_with_the_chosen_agent():
    """Calling the tool must actually run the agent_name the caller picked,
    with no routing/ranking substituted in behind it."""
    register_forge_master_dispatch_tool()

    controller = ForgeMasterController()
    mock_cli_result = {
        "success": True,
        "response": "def add(a, b): return a + b",
        "confidence": 0.9,
    }

    with patch(
        "src.core.forge_master.tools.ForgeMasterController",
        return_value=controller,
    ), patch.object(
        controller.session_manager.cli_manager,
        "execute_with_agent",
        new=AsyncMock(return_value=mock_cli_result),
    ) as mock_execute:
        result = await registry.execute(
            "dispatch_to_cli_agent",
            {"agent_name": "codex", "task_query": "Write an add function"},
        )

    assert result["success"] is True
    assert result["agent_used"] == "codex"
    assert mock_execute.await_args.kwargs.get("agent_name") == "codex"


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

"""Issue #776: deny_command was a stub -- it computed `reason` and discarded
it, never writing to user_responses or re-invoking the orchestrator, unlike
approve_command right above it. These tests cover the fixed deny_command
mirrors approve_command's behavior (just with a "denied" response)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.cli.commands import work


class FakeConsole:
    def __init__(self):
        self.messages = []

    def print(self, msg):
        self.messages.append(msg)

    def status(self, *args, **kwargs):
        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()


class FakeSessionManager:
    def __init__(self, state):
        self._state = state
        self.context_engineer = object()

    def restore_session(self, session_id, context_engineer, shared_memory):
        return self._state


def make_cli(state):
    console = FakeConsole()
    session_control = SimpleNamespace(current_session_id="sess_1")
    cli = SimpleNamespace(console=console, session_control=session_control)
    return cli


def make_orchestrator(state, execute_result=None):
    orchestrator = SimpleNamespace(
        session_manager=FakeSessionManager(state),
        shared_memory={},
        execute=AsyncMock(return_value=execute_result or {"detailed_results": {}}),
    )
    return orchestrator


@pytest.mark.asyncio
async def test_deny_command_marks_action_denied_and_reinvokes_orchestrator(monkeypatch):
    state = {
        "user_query": "do the thing",
        "pending_questions": [{"id": "action_42"}],
        "user_responses": {},
    }
    orchestrator = make_orchestrator(state)
    monkeypatch.setattr(work, "get_orchestrator", lambda: orchestrator)

    cli = make_cli(state)
    await work.deny_command(cli, ["42", "too risky"])

    orchestrator.execute.assert_awaited_once()
    call_kwargs = orchestrator.execute.await_args.kwargs
    assert call_kwargs["custom_state"]["user_responses"]["action_42"] == {
        "response": "denied",
        "reason": "too risky",
    }
    assert any("Denied action 42" in m for m in cli.console.messages)


@pytest.mark.asyncio
async def test_deny_command_defaults_reason_when_not_provided(monkeypatch):
    state = {
        "user_query": "do the thing",
        "pending_questions": [{"id": "action_7"}],
        "user_responses": {},
    }
    orchestrator = make_orchestrator(state)
    monkeypatch.setattr(work, "get_orchestrator", lambda: orchestrator)

    cli = make_cli(state)
    await work.deny_command(cli, ["7"])

    call_kwargs = orchestrator.execute.await_args.kwargs
    assert call_kwargs["custom_state"]["user_responses"]["action_7"]["reason"] == "Denied by user"


@pytest.mark.asyncio
async def test_deny_command_no_pending_actions(monkeypatch):
    state = {"user_query": "do the thing"}
    orchestrator = make_orchestrator(state)
    monkeypatch.setattr(work, "get_orchestrator", lambda: orchestrator)

    cli = make_cli(state)
    await work.deny_command(cli, ["1"])

    orchestrator.execute.assert_not_awaited()
    assert any("No pending actions to deny" in m for m in cli.console.messages)


@pytest.mark.asyncio
async def test_deny_command_requires_args(monkeypatch):
    cli = make_cli({})
    await work.deny_command(cli, [])

    assert any("Usage: deny" in m for m in cli.console.messages)


@pytest.mark.asyncio
async def test_deny_command_no_active_session(monkeypatch):
    orchestrator = make_orchestrator({})
    monkeypatch.setattr(work, "get_orchestrator", lambda: orchestrator)

    cli = SimpleNamespace(console=FakeConsole(), session_control=None)
    await work.deny_command(cli, ["1"])

    orchestrator.execute.assert_not_awaited()
    assert any("No active session" in m for m in cli.console.messages)


@pytest.mark.asyncio
async def test_deny_command_no_matching_action_skips_orchestrator(monkeypatch):
    """A typo'd or already-resolved action_id must not silently re-invoke the
    orchestrator with an unmodified user_responses -- it should tell the user
    nothing matched and stop."""
    state = {
        "user_query": "do the thing",
        "pending_questions": [{"id": "action_42"}],
        "user_responses": {},
    }
    orchestrator = make_orchestrator(state)
    monkeypatch.setattr(work, "get_orchestrator", lambda: orchestrator)

    cli = make_cli(state)
    await work.deny_command(cli, ["99"])

    orchestrator.execute.assert_not_awaited()
    assert any("No matching pending action" in m for m in cli.console.messages)


@pytest.mark.asyncio
async def test_approve_command_no_matching_action_skips_orchestrator(monkeypatch):
    state = {
        "user_query": "do the thing",
        "pending_questions": [{"id": "action_42"}],
        "user_responses": {},
    }
    orchestrator = make_orchestrator(state)
    monkeypatch.setattr(work, "get_orchestrator", lambda: orchestrator)

    cli = make_cli(state)
    await work.approve_command(cli, ["99"])

    orchestrator.execute.assert_not_awaited()
    assert any("No matching pending action" in m for m in cli.console.messages)


@pytest.mark.asyncio
async def test_work_command_with_real_session_control(monkeypatch):
    from src.core.session_control import SessionControl

    orchestrator = make_orchestrator({})
    monkeypatch.setattr(work, "get_orchestrator", lambda: orchestrator)

    session_control = SessionControl()
    cli = SimpleNamespace(console=FakeConsole(), session_control=session_control)

    await work.work_command(cli, ["test", "goal"])

    orchestrator.execute.assert_awaited_once()
    assert session_control.current_session_id is None


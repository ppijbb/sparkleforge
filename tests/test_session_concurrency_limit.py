"""Issue #685: enforce the concurrent-session quota that already existed but
was never actually wired into a session-bootstrap entry point.

SessionControl.register_active_session()/_get_max_active_sessions() already
raised RuntimeError on overrun, but nothing called register_active_session
from the CLI run path, so the quota was pure dead code. These tests cover
the register/release cycle directly, and that main_commands wires it in
(and always releases the slot, on every exit path) for the `run` command.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import src.cli.main_commands as main_commands
from src.core.session_control import SessionControl, SessionStatus


def _fresh_control(max_sessions: int = 2) -> SessionControl:
    control = SessionControl()
    control.default_quota.max_concurrent_sessions = max_sessions
    control._get_max_active_sessions = lambda: max_sessions
    return control


def test_register_active_session_enforces_quota():
    control = _fresh_control(max_sessions=2)
    control.register_active_session("s1", "query one")
    control.register_active_session("s2", "query two")

    with pytest.raises(RuntimeError, match="quota"):
        control.register_active_session("s3", "query three")


def test_release_active_session_frees_a_slot():
    control = _fresh_control(max_sessions=1)
    control.register_active_session("s1", "query one")

    with pytest.raises(RuntimeError):
        control.register_active_session("s2", "query two")

    control.release_active_session("s1")
    control.register_active_session("s2", "query two")  # must not raise now

    assert control.active_sessions["s1"]["status"] == SessionStatus.COMPLETED
    assert control.active_sessions["s2"]["status"] == SessionStatus.ACTIVE


def test_release_active_session_is_a_safe_no_op_for_unknown_session():
    control = SessionControl()
    control.release_active_session("never-registered")  # must not raise


def test_resolve_run_session_surfaces_quota_error(monkeypatch):
    control = _fresh_control(max_sessions=0)
    monkeypatch.setattr("src.core.session_control.get_session_control", lambda: control)

    args = SimpleNamespace(session_id=None, continue_session=False, query="do a thing")
    session_id, error = asyncio.run(main_commands._resolve_run_session(args))

    assert session_id == ""
    assert error is not None and "quota" in error.lower()


def test_resolve_run_session_registers_new_session(monkeypatch):
    control = _fresh_control(max_sessions=2)
    monkeypatch.setattr("src.core.session_control.get_session_control", lambda: control)

    args = SimpleNamespace(session_id=None, continue_session=False, query="do a thing")
    session_id, error = asyncio.run(main_commands._resolve_run_session(args))

    assert error is None
    assert session_id in control.active_sessions
    assert control.active_sessions[session_id]["status"] == SessionStatus.ACTIVE


def test_handle_run_command_releases_session_on_disk_check_failure(monkeypatch):
    control = _fresh_control(max_sessions=2)
    monkeypatch.setattr("src.core.session_control.get_session_control", lambda: control)
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_disk_space_safety",
        lambda: (False, "no space"),
    )

    args = SimpleNamespace(
        mode="research",
        query="test query",
        model=None,
        max_tokens=None,
        task=None,
        session_id=None,
        continue_session=False,
    )
    config = SimpleNamespace(llm=SimpleNamespace(provider="openrouter"))

    result = asyncio.run(main_commands.handle_run_command(args, config))

    assert result == 1
    registered_ids = list(control.active_sessions.keys())
    assert len(registered_ids) == 1
    assert control.active_sessions[registered_ids[0]]["status"] == SessionStatus.COMPLETED


def test_handle_run_command_releases_session_when_runtime_overrides_raise(monkeypatch):
    """Issue #763: an exception between the disk check and the old try block
    (e.g. _apply_runtime_overrides() touching a config shape it doesn't
    expect) must not leak the session slot -- the try/finally now starts
    before that call, not after it."""
    control = _fresh_control(max_sessions=2)
    monkeypatch.setattr("src.core.session_control.get_session_control", lambda: control)
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_disk_space_safety",
        lambda: (True, ""),
    )
    monkeypatch.setattr(
        "src.core.observe.system_collector.check_network_connectivity",
        lambda: (True, ""),
    )

    args = SimpleNamespace(
        mode="research",
        query="test query",
        model="custom-model",  # forces _apply_runtime_overrides to read config.llm.provider
        max_tokens=None,
        task=None,
        session_id=None,
        continue_session=False,
    )
    # config.llm has no `provider` attribute, so _apply_runtime_overrides()
    # raises AttributeError as soon as args.model is set; handle_run_command's
    # own except Exception turns that into a `return 1` rather than a raise.
    config = SimpleNamespace(llm=SimpleNamespace())

    result = asyncio.run(main_commands.handle_run_command(args, config))

    assert result == 1
    registered_ids = list(control.active_sessions.keys())
    assert len(registered_ids) == 1
    assert control.active_sessions[registered_ids[0]]["status"] == SessionStatus.COMPLETED

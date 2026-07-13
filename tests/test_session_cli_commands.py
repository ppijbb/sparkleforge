import io

import pytest
from rich.console import Console

from src.cli.commands.session import session_quota_command, session_show_command
from src.core.session_control import SessionControl


class _CliShim:
    def __init__(self, session_control: SessionControl):
        self.session_control = session_control
        self.buffer = io.StringIO()
        self.console = Console(file=self.buffer, force_terminal=False, width=120)

    def output(self) -> str:
        return self.buffer.getvalue()


@pytest.mark.asyncio
async def test_session_quota_command_shows_usage_without_flags() -> None:
    controller = SessionControl()
    session_id = "cli-quota-show"
    controller.register_active_session(session_id, user_query="q")
    cli = _CliShim(controller)

    await session_quota_command(cli, [session_id])

    out = cli.output()
    assert "Quota Usage" in out
    assert "Tokens:" in out
    assert "Cost:" in out
    assert "Time:" in out


@pytest.mark.asyncio
async def test_session_quota_command_updates_then_shows_new_limits() -> None:
    controller = SessionControl()
    session_id = "cli-quota-update"
    controller.register_active_session(session_id, user_query="q")
    cli = _CliShim(controller)

    await session_quota_command(cli, [session_id, "--max-tokens", "42", "--budget", "3.5"])

    out = cli.output()
    assert "Quota updated" in out
    assert "42" in out
    usage = controller.get_quota_usage(session_id)
    assert usage["tokens"]["limit"] == 42
    assert usage["cost"]["limit"] == 3.5


@pytest.mark.asyncio
async def test_session_quota_command_reports_missing_session() -> None:
    controller = SessionControl()
    cli = _CliShim(controller)

    await session_quota_command(cli, ["does-not-exist"])

    assert "No tracked quota" in cli.output()


@pytest.mark.asyncio
async def test_session_show_command_includes_quota_usage() -> None:
    controller = SessionControl()
    session_id = "cli-show-quota"
    controller.register_active_session(session_id, user_query="q")
    cli = _CliShim(controller)

    await session_show_command(cli, [session_id])

    assert "Quota Usage" in cli.output()

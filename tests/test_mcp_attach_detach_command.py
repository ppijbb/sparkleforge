import io

import pytest
from rich.console import Console

from src.cli.commands.mcp import (
    mcp_attach_command,
    mcp_detach_command,
    mcp_list_command,
)


class _CliShim:
    def __init__(self):
        self.buffer = io.StringIO()
        self.console = Console(file=self.buffer, force_terminal=False, width=120)

    def output(self) -> str:
        return self.buffer.getvalue()


class _FakeHub:
    def __init__(self):
        self.mcp_server_configs = {}
        self.mcp_sessions = {}
        self.fastmcp_clients = {}
        self.registered_with = None
        self.disconnected = None

    async def _register_dynamic_server(self, name, path):
        self.registered_with = (name, path)
        self.mcp_server_configs[name] = {"path": str(path)}
        self.mcp_sessions[name] = object()
        return True

    async def _disconnect_from_mcp_server(self, name):
        self.disconnected = name
        self.mcp_sessions.pop(name, None)
        self.fastmcp_clients.pop(name, None)


@pytest.fixture
def fake_hub(monkeypatch):
    hub = _FakeHub()
    monkeypatch.setattr("src.core.mcp_integration.get_mcp_hub", lambda: hub)
    return hub


async def test_attach_with_explicit_path_registers_and_connects(fake_hub, tmp_path):
    server_file = tmp_path / "server.py"
    server_file.write_text("# fake fastmcp server\n")
    cli = _CliShim()

    await mcp_attach_command(cli, ["demo", str(server_file)])

    assert fake_hub.registered_with == ("demo", server_file)
    assert "Attached MCP server: demo" in cli.output()


async def test_attach_missing_file_does_not_touch_hub(fake_hub, tmp_path):
    cli = _CliShim()

    await mcp_attach_command(cli, ["demo", str(tmp_path / "nope.py")])

    assert fake_hub.registered_with is None
    assert "not found" in cli.output()


async def test_detach_disconnects_known_server(fake_hub):
    fake_hub.mcp_sessions["demo"] = object()
    cli = _CliShim()

    await mcp_detach_command(cli, ["demo"])

    assert fake_hub.disconnected == "demo"
    assert "Detached MCP server: demo" in cli.output()


async def test_detach_unknown_server_is_a_noop(fake_hub):
    cli = _CliShim()

    await mcp_detach_command(cli, ["ghost"])

    assert fake_hub.disconnected is None
    assert "not currently attached" in cli.output()


async def test_list_reflects_attach_then_detach(fake_hub, tmp_path):
    server_file = tmp_path / "server.py"
    server_file.write_text("# fake fastmcp server\n")
    cli = _CliShim()

    await mcp_attach_command(cli, ["demo", str(server_file)])
    await mcp_list_command(cli, [])
    assert "demo" in cli.output()
    assert "connected" in cli.output()

    cli2 = _CliShim()
    await mcp_detach_command(cli2, ["demo"])
    cli3 = _CliShim()
    await mcp_list_command(cli3, [])
    assert "registered" in cli3.output()

"""MCP 서버 attach/detach 명령어.

UniversalMCPHub은 REPL 세션이 살아있는 동안 get_mcp_hub()가 반환하는
프로세스 전역 싱글턴이고, hub._register_dynamic_server/_disconnect_from_mcp_server는
이미 MCPServerBuilder가 "도구 없음 -> 자동 생성 -> 즉시 연결" 흐름에서 쓰고 있는
실제 동작 코드다 (재시작 없이 즉시 반영됨). 여기서는 그 두 메서드를 사용자가
직접 호출할 수 있는 REPL 명령으로만 노출한다.
"""

import logging
from pathlib import Path
from typing import List

from rich.table import Table

logger = logging.getLogger(__name__)


async def mcp_attach_command(cli, args: List[str]):
    """MCP 서버를 실행 중인 세션에 즉시 연결 (재시작 불필요)."""
    if not args:
        cli.console.print("[red]Usage: mcp attach <name> [path/to/server.py][/red]")
        return

    name = args[0]

    if len(args) >= 2:
        server_path = Path(args[1])
    else:
        # 경로 생략 시 MCPServerBuilder가 실제로 서버를 저장하는 위치에서 찾는다
        # (auto-build로 만들어졌다가 detach된 서버를 이름만으로 재연결하는 경우).
        from src.core.mcp_server_builder import MCPServerBuilder

        server_path = MCPServerBuilder().server_dir / name / "server.py"

    if not server_path.exists():
        cli.console.print(f"[red]❌ Server file not found: {server_path}[/red]")
        return

    try:
        from src.core.mcp_integration import get_mcp_hub

        hub = get_mcp_hub()
        connected = await hub._register_dynamic_server(name, server_path)
        if connected:
            cli.console.print(f"[green]✅ Attached MCP server: {name}[/green]")
        else:
            cli.console.print(
                f"[yellow]⚠️ Registered {name} but connection failed — check logs[/yellow]"
            )
    except Exception as e:
        logger.error(f"Failed to attach MCP server {name}: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to attach {name}: {e}[/red]")


async def mcp_detach_command(cli, args: List[str]):
    """실행 중인 MCP 서버 연결을 즉시 해제 (재시작 불필요)."""
    if not args:
        cli.console.print("[red]Usage: mcp detach <name>[/red]")
        return

    name = args[0]

    try:
        from src.core.mcp_integration import get_mcp_hub

        hub = get_mcp_hub()
        known = set(hub.mcp_server_configs) | set(hub.mcp_sessions) | set(hub.fastmcp_clients)
        if name not in known:
            cli.console.print(f"[yellow]⚠️ {name} is not currently attached[/yellow]")
            return

        await hub._disconnect_from_mcp_server(name)
        cli.console.print(f"[green]✅ Detached MCP server: {name}[/green]")
    except Exception as e:
        logger.error(f"Failed to detach MCP server {name}: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to detach {name}: {e}[/red]")


async def mcp_list_command(cli, args: List[str]):
    """현재 등록/연결된 MCP 서버 목록."""
    from src.core.mcp_integration import get_mcp_hub

    hub = get_mcp_hub()
    configured = set(hub.mcp_server_configs)
    connected = set(hub.mcp_sessions) | set(hub.fastmcp_clients)

    if not configured and not connected:
        cli.console.print("[yellow]No MCP servers registered[/yellow]")
        return

    table = Table(title="MCP Servers", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Status")

    for name in sorted(configured | connected):
        status = "[green]connected[/green]" if name in connected else "[dim]registered[/dim]"
        table.add_row(name, status)

    cli.console.print(table)

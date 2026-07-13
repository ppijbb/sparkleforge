"""세션 관리 명령어"""

import logging
from typing import List

from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)


def _format_quota_usage(usage: dict) -> str:
    """쿼터 사용량 딕셔너리(get_quota_usage 반환값)를 사람이 읽을 수 있는 텍스트로 변환."""
    tokens = usage["tokens"]
    cost = usage["cost"]
    time_usage = usage["time"]
    return (
        "[bold]Quota Usage:[/bold]\n"
        f"  Tokens: {tokens['used']:.0f}/{tokens['limit']:.0f} "
        f"({tokens['pct_used']:.1f}%, {tokens['remaining']:.0f} remaining)\n"
        f"  Cost: ${cost['used']:.4f}/${cost['limit']:.4f} "
        f"({cost['pct_used']:.1f}%, ${cost['remaining']:.4f} remaining)\n"
        f"  Time: {time_usage['used']:.0f}s/{time_usage['limit']:.0f}s "
        f"({time_usage['pct_used']:.1f}%, {time_usage['remaining']:.0f}s remaining)"
    )


async def session_list_command(cli, args: List[str]):
    """세션 목록 표시."""
    if not cli.session_control:
        cli.console.print("[red]❌ Session control not available[/red]")
        return

    try:
        from src.core.session_control import SessionStatus

        limit = 20
        if args and args[0].isdigit():
            limit = int(args[0])

        sessions = await cli.session_control.search_sessions(limit=limit)

        if not sessions:
            cli.console.print("[yellow]No sessions found[/yellow]")
            return

        table = Table(title="Sessions", show_header=True, header_style="bold cyan")
        table.add_column("Status", style="cyan", width=10)
        table.add_column("Session ID", style="green", width=30)
        table.add_column("Progress", justify="right", width=10)
        table.add_column("Last Activity", width=20)
        table.add_column("Query", style="dim", width=40)

        status_icons = {
            SessionStatus.ACTIVE: "[green]🟢 ACTIVE[/green]",
            SessionStatus.PAUSED: "[yellow]🟡 PAUSED[/yellow]",
            SessionStatus.COMPLETED: "[green]✅ COMPLETED[/green]",
            SessionStatus.FAILED: "[red]❌ FAILED[/red]",
            SessionStatus.CANCELLED: "[red]🚫 CANCELLED[/red]",
            SessionStatus.WAITING: "[yellow]⏳ WAITING[/yellow]",
        }

        for s in sessions:
            status = status_icons.get(s.status, "[dim]⚪ UNKNOWN[/dim]")
            session_id = s.session_id[:28] + "..." if len(s.session_id) > 28 else s.session_id
            progress = f"{s.progress_percentage:.1f}%"
            last_activity = s.last_activity.strftime("%Y-%m-%d %H:%M:%S")
            query = (
                (s.user_query[:37] + "...")
                if s.user_query and len(s.user_query) > 40
                else (s.user_query or "N/A")
            )

            table.add_row(status, session_id, progress, last_activity, query)

        cli.console.print(table)

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to list sessions: {e}[/red]")


async def session_show_command(cli, args: List[str]):
    """세션 상세 정보 표시."""
    if not cli.session_control:
        cli.console.print("[red]❌ Session control not available[/red]")
        return

    if not args:
        cli.console.print("[red]Usage: session show <session_id>[/red]")
        return

    session_id = args[0]

    try:
        session_info = await cli.session_control.get_session(session_id)

        if not session_info:
            cli.console.print(f"[red]Session not found: {session_id}[/red]")
            return

        from src.core.session_control import SessionStatus

        status_icons = {
            SessionStatus.ACTIVE: "[green]🟢 ACTIVE[/green]",
            SessionStatus.PAUSED: "[yellow]🟡 PAUSED[/yellow]",
            SessionStatus.COMPLETED: "[green]✅ COMPLETED[/green]",
            SessionStatus.FAILED: "[red]❌ FAILED[/red]",
            SessionStatus.CANCELLED: "[red]🚫 CANCELLED[/red]",
            SessionStatus.WAITING: "[yellow]⏳ WAITING[/yellow]",
        }

        status = status_icons.get(session_info.status, "[dim]⚪ UNKNOWN[/dim]")

        info_text = f"""
[bold]Session ID:[/bold] {session_info.session_id}
[bold]Status:[/bold] {status}
[bold]Created:[/bold] {session_info.created_at.strftime("%Y-%m-%d %H:%M:%S")}
[bold]Last Activity:[/bold] {session_info.last_activity.strftime("%Y-%m-%d %H:%M:%S")}
[bold]Progress:[/bold] {session_info.progress_percentage:.1f}%
[bold]Errors:[/bold] {session_info.error_count}
[bold]Warnings:[/bold] {session_info.warning_count}
"""

        if session_info.user_query:
            info_text += f"[bold]Query:[/bold] {session_info.user_query}\n"

        if session_info.current_task:
            info_text += f"[bold]Current Task:[/bold] {session_info.current_task}\n"

        usage = cli.session_control.get_quota_usage(session_id)
        if usage:
            info_text += "\n" + _format_quota_usage(usage)

        cli.console.print(Panel(info_text.strip(), title="Session Details", border_style="cyan"))

    except Exception as e:
        logger.error(f"Failed to show session: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to show session: {e}[/red]")


async def session_pause_command(cli, args: List[str]):
    """세션 일시정지."""
    if not cli.session_control:
        cli.console.print("[red]❌ Session control not available[/red]")
        return

    if not args:
        cli.console.print("[red]Usage: session pause <session_id>[/red]")
        return

    session_id = args[0]

    try:
        success = await cli.session_control.pause_session(session_id)
        if success:
            cli.console.print(f"[green]✅ Session paused: {session_id}[/green]")
        else:
            cli.console.print(f"[red]❌ Failed to pause session: {session_id}[/red]")
    except Exception as e:
        logger.error(f"Failed to pause session: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to pause session: {e}[/red]")


async def session_resume_command(cli, args: List[str]):
    """세션 재개."""
    if not cli.session_control:
        cli.console.print("[red]❌ Session control not available[/red]")
        return

    if not args:
        cli.console.print("[red]Usage: session resume <session_id>[/red]")
        return

    session_id = args[0]

    try:
        success = await cli.session_control.resume_session(session_id)
        if success:
            cli.console.print(f"[green]✅ Session resumed: {session_id}[/green]")
        else:
            cli.console.print(f"[red]❌ Failed to resume session: {session_id}[/red]")
    except Exception as e:
        logger.error(f"Failed to resume session: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to resume session: {e}[/red]")


async def session_cancel_command(cli, args: List[str]):
    """세션 취소."""
    if not cli.session_control:
        cli.console.print("[red]❌ Session control not available[/red]")
        return

    if not args:
        cli.console.print("[red]Usage: session cancel <session_id>[/red]")
        return

    session_id = args[0]

    try:
        success = await cli.session_control.cancel_session(session_id)
        if success:
            cli.console.print(f"[green]✅ Session cancelled: {session_id}[/green]")
        else:
            cli.console.print(f"[red]❌ Failed to cancel session: {session_id}[/red]")
    except Exception as e:
        logger.error(f"Failed to cancel session: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to cancel session: {e}[/red]")


async def session_delete_command(cli, args: List[str]):
    """세션 삭제."""
    if not cli.session_control:
        cli.console.print("[red]❌ Session control not available[/red]")
        return

    if not args:
        cli.console.print("[red]Usage: session delete <session_id>[/red]")
        return

    session_id = args[0]

    try:
        success = await cli.session_control.delete_session(session_id)
        if success:
            cli.console.print(f"[green]✅ Session deleted: {session_id}[/green]")
        else:
            cli.console.print(f"[red]❌ Failed to delete session: {session_id}[/red]")
    except Exception as e:
        logger.error(f"Failed to delete session: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to delete session: {e}[/red]")


async def session_search_command(cli, args: List[str]):
    """세션 검색."""
    if not cli.session_control:
        cli.console.print("[red]❌ Session control not available[/red]")
        return

    query = " ".join(args) if args else None

    try:
        sessions = await cli.session_control.search_sessions(query=query, limit=20)

        if not sessions:
            cli.console.print("[yellow]No sessions found[/yellow]")
            return

        cli.console.print(f"[green]Found {len(sessions)} sessions:[/green]\n")

        for s in sessions:
            cli.console.print(
                f"  [cyan]{s.session_id}[/cyan] | [dim]{s.status.value}[/dim] | {s.user_query or 'N/A'}"
            )

    except Exception as e:
        logger.error(f"Failed to search sessions: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to search sessions: {e}[/red]")


async def session_stats_command(cli, args: List[str]):
    """세션 통계 표시."""
    if not cli.session_control:
        cli.console.print("[red]❌ Session control not available[/red]")
        return

    try:
        sessions = await cli.session_control.search_sessions(limit=1000)

        if not sessions:
            cli.console.print("[yellow]No sessions found[/yellow]")
            return

        from collections import Counter

        from src.core.session_control import SessionStatus

        status_counts = Counter(s.status for s in sessions)
        total = len(sessions)

        stats_text = f"""
[bold]Total Sessions:[/bold] {total}
[bold]Active:[/bold] {status_counts.get(SessionStatus.ACTIVE, 0)}
[bold]Paused:[/bold] {status_counts.get(SessionStatus.PAUSED, 0)}
[bold]Completed:[/bold] {status_counts.get(SessionStatus.COMPLETED, 0)}
[bold]Failed:[/bold] {status_counts.get(SessionStatus.FAILED, 0)}
[bold]Cancelled:[/bold] {status_counts.get(SessionStatus.CANCELLED, 0)}
[bold]Waiting:[/bold] {status_counts.get(SessionStatus.WAITING, 0)}
"""

        session_stats = cli.session_control.get_session_statistics()
        quotas = session_stats.get("quotas", {})
        if quotas:
            max_concurrent = quotas.get("max_concurrent", 0)
            active_count = quotas.get("active_count", 0)
            pct = (active_count / max_concurrent * 100.0) if max_concurrent else 0.0
            limit_reached = "[red]YES[/red]" if quotas.get("limit_reached") else "[green]no[/green]"
            stats_text += (
                f"\n[bold]Concurrent Session Quota:[/bold] {active_count}/{max_concurrent} "
                f"({pct:.1f}%) — limit reached: {limit_reached}\n"
            )

        cli.console.print(
            Panel(stats_text.strip(), title="Session Statistics", border_style="cyan")
        )

    except Exception as e:
        logger.error(f"Failed to get session stats: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to get session stats: {e}[/red]")


async def session_tasks_command(cli, args: List[str]):
    """세션의 작업 목록 표시."""
    if not cli.session_control:
        cli.console.print("[red]❌ Session control not available[/red]")
        return

    if not args:
        cli.console.print("[red]Usage: session tasks <session_id>[/red]")
        return

    session_id = args[0]

    try:
        from src.core.surface.task_dashboard import TaskDashboard

        tasks = await cli.session_control.get_session_tasks(session_id)

        if not tasks:
            cli.console.print(f"[yellow]No tasks found for session: {session_id}[/yellow]")
            return

        table = Table(
            title=f"Tasks for {session_id[:20]}...",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Task ID", style="green", width=30)
        table.add_column("Status", style="cyan", width=15)
        table.add_column("Progress", justify="right", width=10)
        table.add_column("Description", style="dim", width=40)

        for task in tasks:
            task_id = (
                task.get("task_id", "N/A")[:28] + "..."
                if len(task.get("task_id", "")) > 28
                else task.get("task_id", "N/A")
            )
            status = task.get("status", "N/A")
            progress = f"{task.get('progress', 0):.1f}%"
            description = (
                (task.get("description", "N/A")[:37] + "...")
                if len(task.get("description", "")) > 40
                else (task.get("description", "N/A") or "N/A")
            )

            table.add_row(task_id, status, progress, description)

        cli.console.print(table)

    except Exception as e:
        logger.error(f"Failed to get session tasks: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to get session tasks: {e}[/red]")


def _parse_quota_flags(args: List[str]) -> dict:
    """`--max-tokens N --budget N --timeout N` 형태의 플래그를 파싱."""
    flags = {"max_tokens": None, "budget": None, "timeout": None}
    parsers = {
        "--max-tokens": ("max_tokens", int),
        "--budget": ("budget", float),
        "--timeout": ("timeout", int),
    }
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in parsers and i + 1 < len(args):
            key, cast = parsers[arg]
            flags[key] = cast(args[i + 1])
            i += 2
        else:
            i += 1
    return flags


async def session_quota_command(cli, args: List[str]):
    """세션 쿼터 사용량 표시 또는 변경.

    Usage: session quota <session_id> [--max-tokens N] [--budget N] [--timeout SECONDS]
    """
    if not cli.session_control:
        cli.console.print("[red]❌ Session control not available[/red]")
        return

    if not args:
        cli.console.print(
            "[red]Usage: session quota <session_id> [--max-tokens N] [--budget N] "
            "[--timeout SECONDS][/red]"
        )
        return

    session_id = args[0]
    flags = _parse_quota_flags(args[1:])

    try:
        if any(v is not None for v in flags.values()):
            updated = cli.session_control.update_session_quota(session_id, **flags)
            if not updated:
                cli.console.print(f"[red]❌ No tracked quota for session: {session_id}[/red]")
                return
            cli.console.print(f"[green]✅ Quota updated for session: {session_id}[/green]")

        usage = cli.session_control.get_quota_usage(session_id)
        if not usage:
            cli.console.print(f"[red]❌ No tracked quota for session: {session_id}[/red]")
            return

        cli.console.print(
            Panel(_format_quota_usage(usage), title=f"Quota — {session_id}", border_style="cyan")
        )

    except Exception as e:
        logger.error(f"Failed to handle session quota: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to handle session quota: {e}[/red]")

"""세션 관리 명령어"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from typing import List

from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)


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

        info_text += _format_quota_section(cli, session_info)

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
        stats_text += _format_quota_section(cli, None)
        stats_text += _format_concurrency_section(cli, sessions)

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


def _get_quota_manager(cli) -> Optional[Any]:
    """Return the quota manager exposed by the CLI, if any."""
    quota_manager = getattr(cli, "quota_manager", None)
    if quota_manager is not None:
        return quota_manager
    session_control = getattr(cli, "session_control", None)
    if session_control is None:
        return None
    return getattr(session_control, "quota_manager", None)


def _format_tokens(value: Any) -> str:
    """Format a token count into a human readable string."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if numeric >= 1_000_000:
        return f"{numeric / 1_000_000:.2f}M"
    if numeric >= 1_000:
        return f"{numeric / 1_000:.1f}K"
    return f"{int(numeric)}"


def _format_cost(value: Any) -> str:
    """Format a cost value into a human readable currency string."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if numeric >= 1.0:
        return f"${numeric:.2f}"
    if numeric > 0:
        return f"${numeric:.4f}"
    return "$0.00"


def _format_duration(seconds: Any) -> str:
    """Format a duration in seconds into a human readable string."""
    try:
        numeric = float(seconds)
    except (TypeError, ValueError):
        return "N/A"
    if numeric < 0:
        numeric = 0.0
    hours = int(numeric // 3600)
    minutes = int((numeric % 3600) // 60)
    secs = int(numeric % 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_percent(used: Any, limit: Any) -> str:
    """Format a usage percentage from used/limit values."""
    try:
        used_numeric = float(used)
        limit_numeric = float(limit)
    except (TypeError, ValueError):
        return "N/A"
    if limit_numeric <= 0:
        return "N/A"
    return f"{(used_numeric / limit_numeric) * 100:.1f}%"


def _quota_snapshot(quota_manager: Any, session_info: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Collect a quota snapshot dict from the quota manager, if available."""
    snapshot: Optional[Dict[str, Any]] = None
    try:
        if hasattr(quota_manager, "get_quota_snapshot"):
            snapshot = quota_manager.get_quota_snapshot()
        elif hasattr(quota_manager, "snapshot"):
            snapshot = quota_manager.snapshot()
    except Exception as exc:
        logger.debug(f"Quota snapshot unavailable: {exc}")
        snapshot = None

    if snapshot is None and session_info is not None:
        snapshot = getattr(session_info, "quota_snapshot", None)
        if snapshot is not None and not isinstance(snapshot, dict):
            try:
                snapshot = dict(snapshot)
            except Exception:
                snapshot = None

    if snapshot is None:
        return None
    if not isinstance(snapshot, dict):
        try:
            snapshot = dict(snapshot)
        except Exception:
            return None
    return snapshot


def _format_quota_section(cli, session_info: Optional[Any]) -> str:
    """Render a human readable quota usage section."""
    quota_manager = _get_quota_manager(cli)
    if quota_manager is None:
        return ""

    snapshot = _quota_snapshot(quota_manager, session_info)
    if not snapshot:
        return "\n[bold]Quota:[/bold] N/A\n"

    tokens_used = snapshot.get("tokens_used", snapshot.get("token_used", 0))
    tokens_limit = snapshot.get(
        "tokens_limit", snapshot.get("token_limit", snapshot.get("max_tokens", 0))
    )
    tokens_remaining = snapshot.get(
        "tokens_remaining", snapshot.get("token_remaining", 0)
    )
    cost_used = snapshot.get("cost_used", snapshot.get("spent", 0.0))
    cost_limit = snapshot.get("cost_limit", snapshot.get("budget", 0.0))
    cost_remaining = snapshot.get("cost_remaining", 0.0)
    if cost_remaining == 0.0 and cost_limit:
        try:
            cost_remaining = float(cost_limit) - float(cost_used)
        except (TypeError, ValueError):
            cost_remaining = 0.0
    elapsed_seconds = snapshot.get("elapsed_seconds", snapshot.get("elapsed", 0))
    timeout_seconds = snapshot.get(
        "timeout_seconds", snapshot.get("max_duration_seconds", 0)
    )
    remaining_seconds = snapshot.get("remaining_seconds", 0)
    if not remaining_seconds and timeout_seconds:
        try:
            remaining_seconds = float(timeout_seconds) - float(elapsed_seconds)
        except (TypeError, ValueError):
            remaining_seconds = 0

    tokens_pct = _format_percent(tokens_used, tokens_limit) if tokens_limit else "N/A"
    cost_pct = _format_percent(cost_used, cost_limit) if cost_limit else "N/A"
    time_pct = _format_percent(elapsed_seconds, timeout_seconds) if timeout_seconds else "N/A"

    section = "\n[bold]Quota Usage:[/bold]\n"
    section += f"  [bold]Tokens:[/bold] {_format_tokens(tokens_used)} / {_format_tokens(tokens_limit)} used ({tokens_pct})\n"
    section += f"  [bold]Remaining tokens:[/bold] {_format_tokens(tokens_remaining)}\n"
    section += f"  [bold]Cost:[/bold] {_format_cost(cost_used)} / {_format_cost(cost_limit)} used ({cost_pct})\n"
    section += f"  [bold]Remaining budget:[/bold] {_format_cost(cost_remaining)}\n"
    section += f"  [bold]Elapsed time:[/bold] {_format_duration(elapsed_seconds)} / {_format_duration(timeout_seconds)} ({time_pct})\n"
    section += f"  [bold]Remaining time:[/bold] {_format_duration(remaining_seconds)}\n"
    return section


def _format_concurrency_section(cli, sessions: List[Any]) -> str:
    """Render concurrent session usage against the configured limit."""
    quota_manager = _get_quota_manager(cli)
    if quota_manager is None:
        return ""

    max_concurrent = 0
    try:
        max_concurrent = getattr(quota_manager, "max_concurrent_sessions", 0) or 0
        if not max_concurrent and hasattr(quota_manager, "get_max_concurrent_sessions"):
            max_concurrent = quota_manager.get_max_concurrent_sessions() or 0
    except Exception:
        max_concurrent = 0

    if not max_concurrent:
        return ""

    from src.core.session_control import SessionStatus

    active = sum(1 for s in sessions if s.status == SessionStatus.ACTIVE)
    usage_pct = _format_percent(active, max_concurrent)
    section = "\n[bold]Concurrent Sessions:[/bold]\n"
    section += f"  [bold]Active:[/bold] {active} / {int(max_concurrent)} ({usage_pct})\n"
    return section


def _parse_quota_set_args(args: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """Parse key=value pairs for the quota set command."""
    settings: Dict[str, Any] = {}
    errors: List[str] = []
    numeric_keys = {
        "max_tokens",
        "tokens_limit",
        "token_limit",
        "budget",
        "cost_limit",
        "timeout_seconds",
        "max_duration_seconds",
        "max_concurrent_sessions",
    }
    for pair in args:
        if "=" not in pair:
            errors.append(f"Invalid setting '{pair}' (expected key=value)")
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            errors.append(f"Invalid setting '{pair}' (expected key=value)")
            continue
        if key in numeric_keys:
            try:
                settings[key] = float(value)
            except ValueError:
                errors.append(f"Invalid numeric value for '{key}': {value}")
        else:
            settings[key] = value
    return settings, errors


async def session_quota_command(cli, args: List[str]):
    """Show or update session quota settings."""
    quota_manager = _get_quota_manager(cli)
    if quota_manager is None:
        cli.console.print("[red]❌ Quota manager not available[/red]")
        return

    if not args or args[0] in {"show", "status"}:
        cli.console.print(
            Panel(
                _format_quota_section(cli, None).strip() or "Quota: N/A",
                title="Session Quota",
                border_style="cyan",
            )
        )
        return

    action = args[0]
    if action in {"set", "update", "config"}:
        settings, errors = _parse_quota_set_args(args[1:])
        if errors:
            for err in errors:
                cli.console.print(f"[red]❌ {err}[/red]")
            cli.console.print(
                "[dim]Usage: session quota set <key=value> ... "
 "(keys: max_tokens, budget, timeout_seconds, max_concurrent_sessions)[/dim]"
            )
            return
        if not settings:
            cli.console.print(
                "[red]Usage: session quota set <key=value> ... "
 "(keys: max_tokens, budget, timeout_seconds, max_concurrent_sessions)[/red]"
            )
            return
        try:
            if hasattr(quota_manager, "update_limits"):
                quota_manager.update_limits(settings)
            elif hasattr(quota_manager, "configure"):
                quota_manager.configure(settings)
            else:
                for key, value in settings.items():
                    setattr(quota_manager, key, value)
            cli.console.print(f"[green]✅ Quota settings updated: {settings}[/green]")
        except Exception as exc:
            logger.error(f"Failed to update quota settings: {exc}", exc_info=True)
            cli.console.print(f"[red]❌ Failed to update quota settings: {exc}[/red]")
        return

    cli.console.print(
        "[red]Usage: session quota [show|set <key=value> ...][/red]"
    )

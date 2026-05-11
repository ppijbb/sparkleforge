"""Schedule commands shared by the REPL and interactive CLI."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List

from rich.panel import Panel
from rich.table import Table

from src.core.scheduler import (
    ScheduleConfig,
    ScheduleExecution,
    configure_scheduler_execution,
)

logger = logging.getLogger(__name__)


def _fmt_dt(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _schedule_status_markup(schedule: ScheduleConfig) -> str:
    if schedule.status.value == "active":
        return "[green]active[/green]"
    if schedule.status.value == "paused":
        return "[yellow]paused[/yellow]"
    if schedule.status.value == "disabled":
        return "[red]disabled[/red]"
    return f"[cyan]{schedule.status.value}[/cyan]"


def _execution_status_markup(execution: ScheduleExecution) -> str:
    if execution.status == "completed":
        return "[green]completed[/green]"
    if execution.status == "running":
        return "[cyan]running[/cyan]"
    if execution.status == "cancelled":
        return "[yellow]cancelled[/yellow]"
    return f"[red]{execution.status}[/red]"


def _schedule_detail_text(schedule: ScheduleConfig) -> str:
    tags = ", ".join(schedule.tags) if schedule.tags else "-"
    return "\n".join(
        [
            f"[bold]ID:[/bold] {schedule.schedule_id}",
            f"[bold]Name:[/bold] {schedule.name}",
            f"[bold]Cron:[/bold] {schedule.cron_expression}",
            f"[bold]Status:[/bold] {schedule.status.value}",
            f"[bold]Enabled:[/bold] {schedule.enabled}",
            f"[bold]Created:[/bold] {_fmt_dt(schedule.created_at)}",
            f"[bold]Next Run:[/bold] {_fmt_dt(schedule.next_run)}",
            f"[bold]Last Run:[/bold] {_fmt_dt(schedule.last_run)}",
            f"[bold]Run Counts:[/bold] total={schedule.run_count}, success={schedule.success_count}, failure={schedule.failure_count}",
            f"[bold]Tags:[/bold] {tags}",
            f"[bold]Query:[/bold] {schedule.user_query}",
        ]
    )


def _resolve_history_args(args: List[str]) -> tuple[str | None, int]:
    schedule_id = None
    limit = 20

    if not args:
        return schedule_id, limit

    if len(args) == 1:
        if args[0].isdigit():
            return None, int(args[0])
        return args[0], limit

    schedule_id = args[0]
    if args[1].isdigit():
        limit = int(args[1])
    return schedule_id, limit


async def schedule_list_command(cli, args: List[str]):
    """List schedules."""
    try:
        from src.core.scheduler import get_scheduler

        scheduler = get_scheduler()
        schedules = scheduler.list_schedules()

        if not schedules:
            cli.console.print("[yellow]No schedules found[/yellow]")
            return

        table = Table(title="Schedules", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="green", width=34)
        table.add_column("Name", width=22)
        table.add_column("Cron", width=18)
        table.add_column("Status", width=12)
        table.add_column("Next Run", width=20)

        for schedule in schedules:
            table.add_row(
                schedule.schedule_id,
                schedule.name,
                schedule.cron_expression,
                _schedule_status_markup(schedule),
                _fmt_dt(schedule.next_run),
            )

        cli.console.print(table)
    except Exception as e:
        logger.error("Failed to list schedules: %s", e, exc_info=True)
        cli.console.print(f"[red]Failed to list schedules: {e}[/red]")


async def schedule_create_command(cli, args: List[str]):
    """Create a schedule."""
    if len(args) < 3:
        cli.console.print("[red]Usage: schedule create <name> <cron> <query>[/red]")
        cli.console.print(
            "[dim]Example: schedule create daily-report '0 9 * * *' 'Generate daily report'[/dim]"
        )
        return

    name = args[0]
    cron = args[1]
    query = " ".join(args[2:])

    try:
        from src.core.scheduler import get_scheduler

        scheduler = get_scheduler()
        schedule = scheduler.create_schedule(name, cron, query)
        cli.console.print(
            f"[green]Created schedule[/green] {schedule.schedule_id} [dim]({schedule.name})[/dim]"
        )
    except Exception as e:
        logger.error("Failed to create schedule: %s", e, exc_info=True)
        cli.console.print(f"[red]Failed to create schedule: {e}[/red]")


async def schedule_show_command(cli, args: List[str]):
    """Show schedule details."""
    if not args:
        cli.console.print("[red]Usage: schedule show <schedule_id>[/red]")
        return

    try:
        from src.core.scheduler import get_scheduler

        scheduler = get_scheduler()
        schedule = scheduler.get_schedule(args[0])
        if not schedule:
            cli.console.print(f"[red]Schedule not found: {args[0]}[/red]")
            return

        cli.console.print(
            Panel(
                _schedule_detail_text(schedule),
                title=f"Schedule {schedule.schedule_id}",
                border_style="cyan",
            )
        )
    except Exception as e:
        logger.error("Failed to show schedule: %s", e, exc_info=True)
        cli.console.print(f"[red]Failed to show schedule: {e}[/red]")


async def schedule_pause_command(cli, args: List[str]):
    """Pause a schedule."""
    if not args:
        cli.console.print("[red]Usage: schedule pause <schedule_id>[/red]")
        return

    try:
        from src.core.scheduler import get_scheduler

        scheduler = get_scheduler()
        success = scheduler.pause_schedule(args[0])
        if success:
            cli.console.print(f"[green]Paused schedule:[/green] {args[0]}")
        else:
            cli.console.print(f"[red]Schedule not found: {args[0]}[/red]")
    except Exception as e:
        logger.error("Failed to pause schedule: %s", e, exc_info=True)
        cli.console.print(f"[red]Failed to pause schedule: {e}[/red]")


async def schedule_resume_command(cli, args: List[str]):
    """Resume a schedule."""
    if not args:
        cli.console.print("[red]Usage: schedule resume <schedule_id>[/red]")
        return

    try:
        from src.core.scheduler import get_scheduler

        scheduler = get_scheduler()
        success = scheduler.resume_schedule(args[0])
        if success:
            cli.console.print(f"[green]Resumed schedule:[/green] {args[0]}")
        else:
            cli.console.print(f"[red]Schedule not found: {args[0]}[/red]")
    except Exception as e:
        logger.error("Failed to resume schedule: %s", e, exc_info=True)
        cli.console.print(f"[red]Failed to resume schedule: {e}[/red]")


async def schedule_delete_command(cli, args: List[str]):
    """Delete a schedule."""
    if not args:
        cli.console.print("[red]Usage: schedule delete <schedule_id>[/red]")
        return

    try:
        from src.core.scheduler import get_scheduler

        scheduler = get_scheduler()
        success = scheduler.delete_schedule(args[0])
        if success:
            cli.console.print(f"[green]Deleted schedule:[/green] {args[0]}")
        else:
            cli.console.print(f"[red]Schedule not found: {args[0]}[/red]")
    except Exception as e:
        logger.error("Failed to delete schedule: %s", e, exc_info=True)
        cli.console.print(f"[red]Failed to delete schedule: {e}[/red]")


async def schedule_history_command(cli, args: List[str]):
    """Show execution history."""
    try:
        from src.core.scheduler import get_scheduler

        scheduler = get_scheduler()
        schedule_id, limit = _resolve_history_args(args)
        history = scheduler.get_execution_history(schedule_id=schedule_id, limit=limit)

        if not history:
            cli.console.print("[yellow]No schedule execution history found[/yellow]")
            return

        table = Table(
            title="Schedule Execution History", show_header=True, header_style="bold cyan"
        )
        table.add_column("Execution ID", width=26)
        table.add_column("Schedule ID", width=34)
        table.add_column("Status", width=12)
        table.add_column("Started", width=20)
        table.add_column("Duration", width=10)

        for execution in history:
            duration = (
                f"{execution.duration_seconds:.1f}s"
                if execution.duration_seconds is not None
                else "-"
            )
            table.add_row(
                execution.execution_id,
                execution.schedule_id,
                _execution_status_markup(execution),
                _fmt_dt(execution.started_at),
                duration,
            )

        cli.console.print(table)
    except Exception as e:
        logger.error("Failed to show schedule history: %s", e, exc_info=True)
        cli.console.print(f"[red]Failed to show schedule history: {e}[/red]")


async def schedule_stats_command(cli, args: List[str]):
    """Show scheduler statistics."""
    try:
        from src.core.scheduler import get_scheduler

        scheduler = get_scheduler()
        stats = scheduler.get_schedule_statistics()
        body = "\n".join(
            [
                f"[bold]Total schedules:[/bold] {stats['total_schedules']}",
                f"[bold]Active:[/bold] {stats['active_schedules']}",
                f"[bold]Paused:[/bold] {stats['paused_schedules']}",
                f"[bold]Disabled:[/bold] {stats['disabled_schedules']}",
                f"[bold]Total runs:[/bold] {stats['total_runs']}",
                f"[bold]Success:[/bold] {stats['total_success']}",
                f"[bold]Failure:[/bold] {stats['total_failure']}",
                f"[bold]Success rate:[/bold] {stats['success_rate']:.1%}",
            ]
        )
        cli.console.print(Panel(body, title="Scheduler Stats", border_style="cyan"))
    except Exception as e:
        logger.error("Failed to show schedule stats: %s", e, exc_info=True)
        cli.console.print(f"[red]Failed to show schedule stats: {e}[/red]")


async def schedule_run_command(cli, args: List[str]):
    """Run a schedule immediately."""
    if not args:
        cli.console.print("[red]Usage: schedule run <schedule_id>[/red]")
        return

    try:
        from src.core.scheduler import get_scheduler

        scheduler = configure_scheduler_execution(get_scheduler())
        execution = await scheduler.run_now(args[0])
        cli.console.print(
            f"[green]Ran schedule:[/green] {execution.schedule_id} [dim]status={execution.status}[/dim]"
        )
    except Exception as e:
        logger.error("Failed to run schedule now: %s", e, exc_info=True)
        cli.console.print(f"[red]Failed to run schedule: {e}[/red]")


async def schedule_add_command(cli, args: List[str]):
    """Legacy alias for schedule create."""
    await schedule_create_command(cli, args)


async def schedule_remove_command(cli, args: List[str]):
    """Legacy alias for schedule delete."""
    await schedule_delete_command(cli, args)


async def schedule_enable_command(cli, args: List[str]):
    """Legacy alias for schedule resume."""
    await schedule_resume_command(cli, args)


async def schedule_disable_command(cli, args: List[str]):
    """Legacy alias for schedule pause."""
    await schedule_pause_command(cli, args)

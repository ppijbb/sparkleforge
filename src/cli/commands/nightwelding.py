"""Nightwelding (reproduce-first autonomous issue fixer) commands for the REPL.

Mirrors handle_nightwelding_command's behavior (src/cli/main_commands.py),
but reports through cli.console instead of `logger.info`/`logger.error` --
the REPL suppresses non-ERROR logging by default, so a straight reuse of
the argparse-CLI handler would run silently from inside the REPL.
"""

from __future__ import annotations

from typing import List

from rich.table import Table


def _status_style(status_value: str) -> str:
    if status_value == "draft_opened":
        return "green"
    if status_value == "failed":
        return "red"
    return "cyan"


async def nightwelding_run_command(cli, args: List[str]):
    """Usage: nightwelding run [issue_number] [--label <backlog_label>] [--max-iterations N] [--max-per-run N]"""
    from src.core.nightwelding.runner import run_nightwelding_issue, run_nightwelding_sweep

    issue_number = None
    backlog_label = "nightwelding"
    max_iterations = 4
    max_per_run = 3

    positional = [a for a in args if not a.startswith("--")]
    if positional and positional[0].isdigit():
        issue_number = int(positional[0])
    for i, a in enumerate(args):
        if a == "--label" and i + 1 < len(args):
            backlog_label = args[i + 1]
        elif a == "--max-iterations" and i + 1 < len(args) and args[i + 1].isdigit():
            max_iterations = int(args[i + 1])
        elif a == "--max-per-run" and i + 1 < len(args) and args[i + 1].isdigit():
            max_per_run = int(args[i + 1])

    label = f"issue #{issue_number}" if issue_number else f"backlog '{backlog_label}'"
    with cli.console.status(f"[bold cyan]Nightwelding: running {label}...", spinner="dots"):
        try:
            if issue_number:
                items = [await run_nightwelding_issue(issue_number, max_iterations=max_iterations)]
            else:
                items = await run_nightwelding_sweep(
                    backlog_label=backlog_label,
                    max_per_run=max_per_run,
                    max_iterations=max_iterations,
                )
        except Exception as e:
            cli.console.print(f"[red]❌ Nightwelding run failed: {e}[/red]")
            return

    if not items:
        cli.console.print("[yellow]Nightwelding: no eligible issues found.[/yellow]")
        return

    for item in items:
        style = _status_style(item.status.value)
        if item.status.value == "draft_opened":
            cli.console.print(
                f"[green]✅ Issue #{item.issue_number}[/green]: Draft PR opened -> {item.pr_url}"
            )
        else:
            cli.console.print(
                f"[{style}]❌ Issue #{item.issue_number}[/{style}]: {item.status.value} — {item.failure_reason}"
            )


async def nightwelding_status_command(cli, args: List[str]):
    """Usage: nightwelding status"""
    from src.core.nightwelding.models import NightweldingQueue

    queue = NightweldingQueue()
    items = queue.list()
    if not items:
        cli.console.print("[yellow]Nightwelding: queue is empty.[/yellow]")
        return

    table = Table(title="Nightwelding Queue", show_header=True, header_style="bold cyan")
    table.add_column("Issue", style="green", width=8)
    table.add_column("Status", width=16)
    table.add_column("Updated", width=20)
    table.add_column("Detail", style="dim")

    for item in items[:20]:
        detail = ""
        if item.status.value == "failed" and item.failure_reason:
            detail = item.failure_reason.splitlines()[0]
        style = _status_style(item.status.value)
        table.add_row(
            f"#{item.issue_number}",
            f"[{style}]{item.status.value}[/{style}]",
            str(item.updated_at),
            detail,
        )

    cli.console.print(table)


async def nightwelding_list_command(cli, args: List[str]):
    """Usage: nightwelding list [--verbose]"""
    from src.core.nightwelding.models import NightweldingQueue

    verbose = "--verbose" in args
    queue = NightweldingQueue()
    items = queue.list()
    if not items:
        cli.console.print("[yellow]Nightwelding: queue is empty.[/yellow]")
        return

    for item in items:
        style = _status_style(item.status.value)
        line = f"[{style}]#{item.issue_number}: {item.status.value}[/{style}] pr={item.pr_url or '-'}"
        if item.status.value == "failed" and item.failure_reason:
            line += f" | reason: {item.failure_reason.splitlines()[0]}"
        cli.console.print(line)
        log_val = getattr(item, "log", None)
        if verbose and log_val:
            cli.console.print(f"  [dim]log: {log_val}[/dim]")

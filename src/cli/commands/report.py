"""Report management commands for SparkleForge CLI."""

import logging
from typing import List
from pathlib import Path
from rich.table import Table
from rich.panel import Panel

logger = logging.getLogger(__name__)


async def report_command(cli, args: List[str]):
    """Report management command."""
    if not args:
        cli.console.print("[red]Usage: report <generate|history>[/red]")
        return
        
    subcommand = args[0]
    
    if subcommand == "generate":
        cli.console.print("[cyan]📊 Generating Daily Agent Metric Evaluation Report...[/cyan]")
        try:
            from src.core.monitoring.report_generator import generate_daily_report
            
            with cli.console.status("[bold cyan]Analyzing logs and generating critique...", spinner="dots"):
                res = await generate_daily_report(Path.cwd())
                
            cli.console.print(f"[green]✅ Report successfully generated![/green]")
            cli.console.print(f"[bold]Strict Score:[/bold] [yellow]{res['strict_score']:.1f} / 100[/yellow]")
            cli.console.print(f"[bold]Total Attempts:[/bold] {res['metrics']['total_attempts']}")
            cli.console.print(f"[bold]Success Rate:[/bold] {res['metrics']['success_rate']:.1f}%")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)
            cli.console.print(f"[red]❌ Failed to generate report: {e}[/red]")
            
    elif subcommand == "history":
        cli.console.print("[cyan]📈 Showing Agent Metric Evaluation History:[/cyan]")
        try:
            import json
            history_file = Path.cwd() / "results" / "agent_reports" / "history.json"
            if not history_file.exists():
                cli.console.print("[yellow]No report history found.[/yellow]")
                return
                
            history = json.loads(history_file.read_text(encoding="utf-8"))
            if not history:
                cli.console.print("[yellow]History is empty.[/yellow]")
                return
                
            table = Table(title="Agent Metric History", show_header=True, header_style="bold magenta")
            table.add_column("Date", style="dim")
            table.add_column("Strict Score", justify="right", style="yellow")
            table.add_column("Attempts", justify="right")
            table.add_column("Success Rate", justify="right", style="green")
            table.add_column("Maker's Marks", justify="right")
            
            for entry in sorted(history, key=lambda x: x.get("date", ""), reverse=True):
                table.add_row(
                    entry.get("date", "N/A"),
                    f"{entry.get('strict_score', 0.0):.1f}",
                    str(entry.get("total_attempts", 0)),
                    f"{entry.get('success_rate', 0.0):.1f}%",
                    str(entry.get("total_marks", 0))
                )
            cli.console.print(table)
        except Exception as e:
            logger.error(f"Failed to view history: {e}", exc_info=True)
            cli.console.print(f"[red]❌ Failed to read history: {e}[/red]")

    elif subcommand == "aggregate":
        cli.console.print("[cyan]📦 Aggregating Release Metrics...[/cyan]")
        try:
            import json
            from src.core.monitoring.report_generator import aggregate_release_metrics

            history_file = Path.cwd() / "results" / "agent_reports" / "history.json"
            history = []
            if history_file.exists():
                history = json.loads(history_file.read_text(encoding="utf-8"))

            summary = aggregate_release_metrics(history)
            if summary["entry_count"] == 0:
                cli.console.print("[yellow]No report history found.[/yellow]")
                return

            start_date, end_date = summary["date_range"]
            panel_body = (
                f"[bold]Date Range:[/bold] {start_date} → {end_date}\n"
                f"[bold]Days Covered:[/bold] {summary['entry_count']}\n"
                f"[bold]Total Attempts:[/bold] {summary['total_attempts']}\n"
                f"[bold]Total Maker's Marks:[/bold] {summary['total_marks']}\n"
                f"[bold]Average Strict Score:[/bold] [yellow]{summary['average_strict_score']:.1f} / 100[/yellow]\n"
                f"[bold]Weighted Success Rate:[/bold] [green]{summary['weighted_success_rate']:.1f}%[/green]"
            )
            cli.console.print(Panel(panel_body, title="Release Metrics Summary"))
        except Exception as e:
            logger.error(f"Failed to aggregate release metrics: {e}", exc_info=True)
            cli.console.print(f"[red]❌ Failed to aggregate release metrics: {e}[/red]")

    else:
        cli.console.print(f"[red]❌ Unknown subcommand: {subcommand}[/red]")
        cli.console.print("Available subcommands: generate, history, aggregate")

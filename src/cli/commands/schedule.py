"""
스케줄 관리 명령어
"""

import logging
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logger = logging.getLogger(__name__)


async def schedule_list_command(cli, args: List[str]):
    """스케줄 목록 표시."""
    try:
        from src.core.scheduler import get_scheduler
        
        scheduler = get_scheduler()
        schedules = await scheduler.list_schedules()
        
        if not schedules:
            cli.console.print("[yellow]No schedules found[/yellow]")
            return
        
        table = Table(title="Schedules", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="green", width=30)
        table.add_column("Name", width=20)
        table.add_column("Cron", width=20)
        table.add_column("Status", style="cyan", width=10)
        table.add_column("Next Run", width=20)
        
        for sched in schedules:
            sched_id = sched.get("schedule_id", "N/A")[:28] + "..." if len(sched.get("schedule_id", "")) > 28 else sched.get("schedule_id", "N/A")
            name = sched.get("name", "N/A")
            cron = sched.get("cron_expression", "N/A")
            status = "[green]ENABLED[/green]" if sched.get("enabled", False) else "[red]DISABLED[/red]"
            next_run = sched.get("next_run", "N/A")
            
            table.add_row(sched_id, name, cron, status, str(next_run))
        
        cli.console.print(table)
        
    except Exception as e:
        logger.error(f"Failed to list schedules: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to list schedules: {e}[/red]")


async def schedule_add_command(cli, args: List[str]):
    """스케줄 추가."""
    if len(args) < 3:
        cli.console.print("[red]Usage: schedule add <name> <cron> <query>[/red]")
        cli.console.print("[dim]Example: schedule add daily-report '0 9 * * *' 'Generate daily report'[/dim]")
        return
    
    name = args[0]
    cron = args[1]
    query = " ".join(args[2:])
    
    try:
        from src.core.scheduler import get_scheduler
        
        scheduler = get_scheduler()
        schedule_id = await scheduler.add_schedule(name, cron, query)
        
        cli.console.print(f"[green]✅ Schedule added: {schedule_id}[/green]")
        cli.console.print(f"[dim]Name: {name}, Cron: {cron}[/dim]")
        
    except Exception as e:
        logger.error(f"Failed to add schedule: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to add schedule: {e}[/red]")


async def schedule_remove_command(cli, args: List[str]):
    """스케줄 제거."""
    if not args:
        cli.console.print("[red]Usage: schedule remove <schedule_id>[/red]")
        return
    
    schedule_id = args[0]
    
    try:
        from src.core.scheduler import get_scheduler
        
        scheduler = get_scheduler()
        success = await scheduler.remove_schedule(schedule_id)
        
        if success:
            cli.console.print(f"[green]✅ Schedule removed: {schedule_id}[/green]")
        else:
            cli.console.print(f"[red]❌ Failed to remove schedule: {schedule_id}[/red]")
            
    except Exception as e:
        logger.error(f"Failed to remove schedule: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to remove schedule: {e}[/red]")


async def schedule_enable_command(cli, args: List[str]):
    """스케줄 활성화."""
    if not args:
        cli.console.print("[red]Usage: schedule enable <schedule_id>[/red]")
        return
    
    schedule_id = args[0]
    
    try:
        from src.core.scheduler import get_scheduler
        
        scheduler = get_scheduler()
        success = await scheduler.enable_schedule(schedule_id)
        
        if success:
            cli.console.print(f"[green]✅ Schedule enabled: {schedule_id}[/green]")
        else:
            cli.console.print(f"[red]❌ Failed to enable schedule: {schedule_id}[/red]")
            
    except Exception as e:
        logger.error(f"Failed to enable schedule: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to enable schedule: {e}[/red]")


async def schedule_disable_command(cli, args: List[str]):
    """스케줄 비활성화."""
    if not args:
        cli.console.print("[red]Usage: schedule disable <schedule_id>[/red]")
        return
    
    schedule_id = args[0]
    
    try:
        from src.core.scheduler import get_scheduler
        
        scheduler = get_scheduler()
        success = await scheduler.disable_schedule(schedule_id)
        
        if success:
            cli.console.print(f"[green]✅ Schedule disabled: {schedule_id}[/green]")
        else:
            cli.console.print(f"[red]❌ Failed to disable schedule: {schedule_id}[/red]")
            
    except Exception as e:
        logger.error(f"Failed to disable schedule: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to disable schedule: {e}[/red]")

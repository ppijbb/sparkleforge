"""
체크포인트 관리 명령어
"""

import logging
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logger = logging.getLogger(__name__)


async def checkpoint_save_command(cli, args: List[str]):
    """체크포인트 저장."""
    if not cli.checkpoint_manager:
        cli.console.print("[red]❌ Checkpoint manager not available[/red]")
        return
    
    try:
        checkpoint_name = args[0] if args else None
        
        checkpoint_id = await cli.checkpoint_manager.save_checkpoint(
            state={"manual": True},
            metadata={"name": checkpoint_name} if checkpoint_name else {}
        )
        
        cli.console.print(f"[green]✅ Checkpoint saved: {checkpoint_id}[/green]")
        if checkpoint_name:
            cli.console.print(f"[dim]Name: {checkpoint_name}[/dim]")
            
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to save checkpoint: {e}[/red]")


async def checkpoint_list_command(cli, args: List[str]):
    """체크포인트 목록 표시."""
    if not cli.checkpoint_manager:
        cli.console.print("[red]❌ Checkpoint manager not available[/red]")
        return
    
    try:
        checkpoints = await cli.checkpoint_manager.list_checkpoints()
        
        if not checkpoints:
            cli.console.print("[yellow]No checkpoints found[/yellow]")
            return
        
        table = Table(title="Checkpoints", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="green", width=30)
        table.add_column("Created", width=20)
        table.add_column("Metadata", style="dim", width=40)
        
        for cp in checkpoints:
            cp_id = cp.get("checkpoint_id", "N/A")[:28] + "..." if len(cp.get("checkpoint_id", "")) > 28 else cp.get("checkpoint_id", "N/A")
            created = cp.get("created_at", "N/A")
            if isinstance(created, str):
                created = created[:19]  # ISO format에서 날짜 부분만
            metadata = str(cp.get("metadata", {}))[:37] + "..." if len(str(cp.get("metadata", {}))) > 40 else str(cp.get("metadata", {}))
            
            table.add_row(cp_id, created, metadata)
        
        cli.console.print(table)
        
    except Exception as e:
        logger.error(f"Failed to list checkpoints: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to list checkpoints: {e}[/red]")


async def checkpoint_restore_command(cli, args: List[str]):
    """체크포인트 복원."""
    if not cli.checkpoint_manager:
        cli.console.print("[red]❌ Checkpoint manager not available[/red]")
        return
    
    if not args:
        cli.console.print("[red]Usage: checkpoint restore <checkpoint_id>[/red]")
        return
    
    checkpoint_id = args[0]
    
    try:
        state = await cli.checkpoint_manager.restore_checkpoint(checkpoint_id)
        
        if state:
            cli.console.print(f"[green]✅ Checkpoint restored: {checkpoint_id}[/green]")
        else:
            cli.console.print(f"[red]❌ Checkpoint not found: {checkpoint_id}[/red]")
            
    except Exception as e:
        logger.error(f"Failed to restore checkpoint: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to restore checkpoint: {e}[/red]")


async def checkpoint_delete_command(cli, args: List[str]):
    """체크포인트 삭제."""
    if not cli.checkpoint_manager:
        cli.console.print("[red]❌ Checkpoint manager not available[/red]")
        return
    
    if not args:
        cli.console.print("[red]Usage: checkpoint delete <checkpoint_id>[/red]")
        return
    
    checkpoint_id = args[0]
    
    try:
        success = await cli.checkpoint_manager.delete_checkpoint(checkpoint_id)
        
        if success:
            cli.console.print(f"[green]✅ Checkpoint deleted: {checkpoint_id}[/green]")
        else:
            cli.console.print(f"[red]❌ Failed to delete checkpoint: {checkpoint_id}[/red]")
            
    except Exception as e:
        logger.error(f"Failed to delete checkpoint: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to delete checkpoint: {e}[/red]")

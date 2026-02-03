"""
컨텍스트 관리 명령어
"""

import logging
from typing import List
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)


async def context_show_command(cli, args: List[str]):
    """컨텍스트 표시."""
    if not cli.context_loader:
        cli.console.print("[red]❌ Context loader not available[/red]")
        return
    
    try:
        context = await cli.context_loader.load_context()
        
        if context:
            preview = context[:2000]  # 처음 2000자만
            if len(context) > 2000:
                preview += f"\n\n... ({len(context) - 2000} more characters)"
            
            cli.console.print(Panel(preview, title="Project Context (SPARKLEFORGE.md)", border_style="cyan"))
        else:
            cli.console.print("[yellow]No context file found (SPARKLEFORGE.md)[/yellow]")
            
    except Exception as e:
        logger.error(f"Failed to show context: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to show context: {e}[/red]")


async def context_reload_command(cli, args: List[str]):
    """컨텍스트 재로드."""
    if not cli.context_loader:
        cli.console.print("[red]❌ Context loader not available[/red]")
        return
    
    try:
        cli.context_loader.clear_cache()
        context = await cli.context_loader.load_context()
        
        if context:
            cli.console.print("[green]✅ Context reloaded[/green]")
        else:
            cli.console.print("[yellow]No context file found (SPARKLEFORGE.md)[/yellow]")
            
    except Exception as e:
        logger.error(f"Failed to reload context: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Failed to reload context: {e}[/red]")

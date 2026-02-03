"""
도움말 명령어
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


async def help_command(console: Console):
    """도움말 표시."""
    help_text = """
[bold cyan]SparkleForge CLI Commands[/bold cyan]

[bold]Research:[/bold]
  research <query>              Execute research request
  <query>                       Direct query (same as research)

[bold]Session Management:[/bold]
  session list [limit]          List all sessions
  session show <id>             Show session details
  session pause <id>            Pause session
  session resume <id>           Resume session
  session cancel <id>           Cancel session
  session delete <id>           Delete session
  session search <query>        Search sessions
  session stats                 Show session statistics
  session tasks <id>            List tasks for session

[bold]Context:[/bold]
  context show                  Show loaded project context
  context reload                Reload project context

[bold]Checkpoint:[/bold]
  checkpoint save [name]        Save checkpoint
  checkpoint list               List all checkpoints
  checkpoint restore <id>       Restore checkpoint
  checkpoint delete <id>       Delete checkpoint

[bold]Schedule:[/bold]
  schedule list                 List all schedules
  schedule add <name> <cron> <query>  Add schedule
  schedule remove <id>          Remove schedule
  schedule enable <id>          Enable schedule
  schedule disable <id>         Disable schedule

[bold]Config:[/bold]
  config show                   Show configuration
  config set <key> <value>      Set configuration (not implemented)
  config get <key>              Get configuration value

[bold]Other:[/bold]
  help                          Show this help
  clear                         Clear screen
  exit / quit                   Exit REPL
"""
    
    console.print(Panel(help_text.strip(), title="Help", border_style="cyan"))

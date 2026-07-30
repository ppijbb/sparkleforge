"""Forge Master CLI command for SparkleForge REPL."""

import logging
from typing import List

from src.core.forge_master import ForgeMasterController

logger = logging.getLogger(__name__)


async def forge_master_command(cli, args: List[str]):
    """forge-master <query> [--agent AGENT] [--persistent] [--persona PERSONA]

    Execute meta-orchestrated task using external local AI CLI tools.
    """
    if not args:
        cli.console.print(
            "[red]Usage: forge-master <query> [--agent AGENT] [--persistent] [--persona PERSONA][/red]"
        )
        cli.console.print("Available: claude_code, codex, gemini_cli, hermes, open_code, cline_cli")
        cli.console.print("Personas: ponytail, caveman, blacksmith")
        return

    query_parts = []
    preferred_agent = None
    is_persistent = False
    persona = None

    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--agent" and idx + 1 < len(args):
            preferred_agent = args[idx + 1]
            idx += 2
        elif arg == "--persona" and idx + 1 < len(args):
            persona = args[idx + 1]
            idx += 2
        elif arg in ("--persistent", "-p"):
            is_persistent = True
            idx += 1
        else:
            query_parts.append(arg)
            idx += 1

    query = " ".join(query_parts).strip()
    if not query:
        cli.console.print("[red]Error: Query cannot be empty.[/red]")
        return

    cli.console.print(
        f"[cyan]⚡ [Forge Master] Task: '{query}' (Agent: {preferred_agent or 'Auto'})[/cyan]"
    )

    controller = ForgeMasterController()

    with cli.console.status("[bold cyan]Forge Master orchestrating & auditing...", spinner="dots"):
        result = await controller.execute_task_with_master_control(
            task_query=query,
            preferred_agent=preferred_agent,
            is_persistent_session=is_persistent,
            persona=persona,
        )

    if result.get("success"):
        agent_used = result.get("agent_used")
        verdict = result.get("master_verdict")
        cli.console.print(f"\n[green]✅ Master Verdict: {verdict} (Agent: {agent_used})[/green]")
        audit_fb = result.get("adversarial_audit", {}).get("feedback")
        cli.console.print(f"[bold]Adversarial Audit Feedback:[/bold] {audit_fb}")
        cli.console.print(f"[bold]Token Metrics:[/bold] {result.get('token_metrics')}")
        cli.console.print(f"\n[bold]Output Response:[/bold]\n{result.get('response')}")
    else:
        cli.console.print(f"\n[red]❌ Forge Master Failed: {result.get('error')}[/red]")

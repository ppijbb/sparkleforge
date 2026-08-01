"""Forge Master CLI command for SparkleForge REPL."""

import logging
from typing import List

from src.core.forge_master import ForgeMasterController, ForgeMasterRouter

logger = logging.getLogger(__name__)

BATCH_TASK_DELIMITER = " ||| "


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


async def forge_master_batch_command(cli, args: List[str]):
    """forge-batch <query1> ||| <query2> ||| ... [--agent AGENT]

    Dispatch multiple tasks to the CLI agent fleet at once via ForgeMaster.
    Each task is routed, executed, and audited independently and concurrently.
    Prints one summary line per task - full detail goes to the log, not the
    terminal, so a large batch doesn't flood the console.
    """
    from src.core.forge_master.tools import _dispatch_batch_to_forge_master_tool

    if not args:
        cli.console.print(
            f"[red]Usage: forge-batch <query1>{BATCH_TASK_DELIMITER}<query2>... [--agent AGENT][/red]"
        )
        return

    forced_agent = None
    if "--agent" in args:
        idx = args.index("--agent")
        if idx + 1 < len(args):
            forced_agent = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]

    raw = " ".join(args).strip()
    queries = [q.strip() for q in raw.split(BATCH_TASK_DELIMITER) if q.strip()]
    if not queries:
        cli.console.print("[red]Error: No tasks found.[/red]")
        return

    router = ForgeMasterRouter()
    tasks = [
        {
            "agent_name": forced_agent or router.route_task(q).agent_name,
            "task_query": q,
        }
        for q in queries
    ]

    cli.console.print(f"[cyan]⚡ [Forge Master] Dispatching {len(tasks)} task(s)...[/cyan]")

    with cli.console.status("[bold cyan]Forge Master orchestrating & auditing...", spinner="dots"):
        batch_result = await _dispatch_batch_to_forge_master_tool(tasks)

    for i, (task, result) in enumerate(zip(tasks, batch_result["results"])):
        if result.get("success"):
            cli.console.print(
                f"[green]✅ [{i}] {task['agent_name']}: {result.get('master_verdict')}[/green]"
            )
        else:
            cli.console.print(
                f"[red]❌ [{i}] {task['agent_name']}: {result.get('error', 'failed')}[/red]"
            )

    cli.console.print(
        f"\n[bold]{batch_result['succeeded']}/{batch_result['total']} succeeded.[/bold] "
        "(full detail logged, not printed)"
    )

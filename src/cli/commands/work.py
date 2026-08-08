import argparse
import logging
import os
from contextlib import contextmanager
from typing import List

from src.core.agent_orchestrator import get_orchestrator
from src.core.env_configurator import ConfigurationError, verify_environment

logger = logging.getLogger(__name__)

# Maps substrings of AgentHarness's "[Harness] ... Node" log lines (research
# mode) and AgentLoop's "[AgentLoop] ..." log lines (autonomous/coworker mode)
# to a human-readable spinner label, so the status text moves as execution
# actually moves instead of sitting on a static "Working...".
_STAGE_LABELS = [
    ("Classify Node", "🔍 Classifying request..."),
    ("Planner Node", "📋 Planning tasks..."),
    ("Single Agent Node", "🤖 Running agent..."),
    ("Executor Node", "⚙️  Executing tasks..."),
    ("assigned to:", "🧑‍💻 Assigning tasks..."),
    ("Anvil engine processed", "🔨 Running local tasks..."),
    ("SubAgent Delegate Node", "🤝 Delegating to sub-agent..."),
    ("Document Processor Node", "📄 Processing documents..."),
    ("Synthesize Node", "📝 Synthesizing results..."),
    ("Retrying in", "⏳ Retrying..."),
    ("Context limit hit", "🗜️  Compressing context..."),
]

# AgentLoop's iteration/tool-call lines carry their own useful detail (which
# tool, which step) so we echo them instead of collapsing to a static label.
_STAGE_ECHO_NEEDLES = ("[AgentLoop] Executing tool:", "[AgentLoop] Iteration")


class _StageStatusHandler(logging.Handler):
    """Updates a rich Status spinner's label as the harness/loop moves through stages."""

    def __init__(self, status):
        super().__init__()
        self.status = status

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        for needle in _STAGE_ECHO_NEEDLES:
            if needle in msg:
                self.status.update(f"[bold cyan]⚙️  {msg.split(']', 1)[-1].strip()}")
                return
        for needle, label in _STAGE_LABELS:
            if needle in msg:
                self.status.update(f"[bold cyan]{label}")
                return


@contextmanager
def _stage_status(cli, initial_label: str):
    """Like cli.console.status(), but its label tracks execution progress."""
    with cli.console.status(f"[bold cyan]{initial_label}", spinner="dots") as status:
        handler = _StageStatusHandler(status)
        stage_loggers = [
            logging.getLogger("src.core.agent_harness"),
            logging.getLogger("src.core.agent_loop"),
        ]
        for stage_logger in stage_loggers:
            stage_logger.addHandler(handler)
        try:
            yield status
        finally:
            for stage_logger in stage_loggers:
                stage_logger.removeHandler(handler)


async def work_command(cli, args: List[str]):
    """Work <goal> - Start or continue a coworker session."""
    os.environ.setdefault("SPARKLEFORGE_ENV", "development")
    # Security check before execution
    try:
        environment_ok = verify_environment()
    except ConfigurationError as exc:
        logger.error("Security check failed: %s", exc)
        cli.console.print(f"[red]Security check failed: {exc}[/red]")
        return

    if not environment_ok:
        logger.error("Security check failed. Aborting.")
        cli.console.print("[red]Security check failed.[/red]")
        return

    if not args:
        cli.console.print("[red]Usage: work <goal>[/red]")
        return

    goal = " ".join(args)
    session_id = getattr(cli.session_control, "current_session_id", None) if cli.session_control else None

    cli.console.print(f"[cyan]🤝 Starting coworker session for: {goal}[/cyan]")

    orchestrator = get_orchestrator()
    custom_state = {"mode": "coworker", "current_goal": goal}

    with _stage_status(cli, "Working..."):
        result = await orchestrator.execute(
            user_query=goal, session_id=session_id, restore_session=True, custom_state=custom_state
        )

    _display_action_proposals(cli, result)


async def actions_command(cli, args: List[str]):
    """Actions - List pending action proposals."""
    session_id = getattr(cli.session_control, "current_session_id", None) if cli.session_control else None
    if not session_id:
        cli.console.print("[yellow]No active session.[/yellow]")
        return

    # Load state from memory
    orchestrator = get_orchestrator()
    state = orchestrator.session_manager.restore_session(
        session_id, orchestrator.session_manager.context_engineer, orchestrator.shared_memory
    )
    if not state:
        cli.console.print("[yellow]Could not load session state.[/yellow]")
        return

    _display_action_proposals(cli, state)


async def approve_command(cli, args: List[str]):
    """Approve <action_id|all> - Approve pending actions."""
    if not args:
        cli.console.print("[red]Usage: approve <action_id|all>[/red]")
        return

    action_id = args[0]
    session_id = getattr(cli.session_control, "current_session_id", None) if cli.session_control else None
    if not session_id:
        cli.console.print("[yellow]No active session.[/yellow]")
        return

    orchestrator = get_orchestrator()
    state = orchestrator.session_manager.restore_session(
        session_id, orchestrator.session_manager.context_engineer, orchestrator.shared_memory
    )

    if not state or not state.get("pending_questions"):
        cli.console.print("[yellow]No pending actions to approve.[/yellow]")
        return

    user_responses = state.get("user_responses", {})
    matched = False
    for q in state.get("pending_questions", []):
        qid = q.get("id", "")
        if qid.startswith("action_") and (action_id == "all" or qid == f"action_{action_id}"):
            user_responses[qid] = {"response": "approved"}
            cli.console.print(f"[green]✅ Approved action {qid[7:]}[/green]")
            matched = True

    if not matched:
        cli.console.print(f"[yellow]No matching pending action '{action_id}'.[/yellow]")
        return

    with _stage_status(cli, "Executing approved actions..."):
        result = await orchestrator.execute(
            user_query=state.get("user_query", ""),
            session_id=session_id,
            restore_session=True,
            custom_state={"user_responses": user_responses},
        )

    _display_action_proposals(cli, result)


async def deny_command(cli, args: List[str]):
    """Deny <action_id|all> [reason] - Deny an action."""
    if not args:
        cli.console.print("[red]Usage: deny <action_id|all> [reason][/red]")
        return

    action_id = args[0]
    reason = " ".join(args[1:]) if len(args) > 1 else "Denied by user"

    session_id = getattr(cli.session_control, "current_session_id", None) if cli.session_control else None
    if not session_id:
        cli.console.print("[yellow]No active session.[/yellow]")
        return

    orchestrator = get_orchestrator()
    state = orchestrator.session_manager.restore_session(
        session_id, orchestrator.session_manager.context_engineer, orchestrator.shared_memory
    )

    if not state or not state.get("pending_questions"):
        cli.console.print("[yellow]No pending actions to deny.[/yellow]")
        return

    user_responses = state.get("user_responses", {})
    matched = False
    for q in state.get("pending_questions", []):
        qid = q.get("id", "")
        if qid.startswith("action_") and (action_id == "all" or qid == f"action_{action_id}"):
            user_responses[qid] = {"response": "denied", "reason": reason}
            cli.console.print(f"[red]❌ Denied action {qid[7:]}[/red]")
            matched = True

    if not matched:
        cli.console.print(f"[yellow]No matching pending action '{action_id}'.[/yellow]")
        return

    with _stage_status(cli, "Applying denial..."):
        result = await orchestrator.execute(
            user_query=state.get("user_query", ""),
            session_id=session_id,
            restore_session=True,
            custom_state={"user_responses": user_responses},
        )

    _display_action_proposals(cli, result)


def _display_action_proposals(cli, state):
    proposals = state.get("action_proposals", [])
    if not proposals:
        # content > final_report > results 순서로 실제 응답 필드를 탐색
        output = (
            state.get("content")
            or state.get("final_report")
            or state.get("results")
        )
        if output:
            cli.console.print(f"\n[bold]Result:[/bold]\n{output}")
        else:
            cli.console.print("[green]No pending actions.[/green]")
        return

    cli.console.print("\n[bold]Action Proposals:[/bold]")
    for ap in proposals:
        status_color = (
            "yellow"
            if ap["status"] == "pending"
            else "green" if ap["status"] == "approved" else "red"
        )
        cli.console.print(
            f"  [{ap['id']}] {ap['title']} - [{status_color}]{ap['status']}[/{status_color}]"
        )
        if ap["status"] == "pending":
            cli.console.print(f"      Type: {ap['tool_type']}, Preview: {ap['preview']}")

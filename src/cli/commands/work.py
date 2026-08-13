import argparse
import logging
import os
from typing import List

from src.cli.ui.spinner import stage_status
from src.core.agent_orchestrator import get_orchestrator
from src.core.env_configurator import ConfigurationError, verify_environment
from src.core.session_manager import get_session_manager

logger = logging.getLogger(__name__)

_STAGE_LOGGERS = ("src.core.agent_harness", "src.core.agent_loop")


def _stage_status(cli, initial_label: str):
    """Like cli.console.status(), but its label tracks execution progress."""
    return stage_status(cli.console, initial_label, _STAGE_LOGGERS)


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

    custom_state = {"mode": "coworker", "current_goal": goal}

    # get_orchestrator() inside the spinner scope, not before it: on the
    # first call in a process it lazily constructs AgentHarness (tool
    # registration, LLM client init, ...), which logs a burst of one-time
    # setup chatter that should be caught by the same chat-mode noise filter
    # as everything else this turn does, not leak before the spinner starts.
    with _stage_status(cli, "Working..."):
        orchestrator = get_orchestrator()
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
    session_manager = get_session_manager()
    state = session_manager.restore_session(
        session_id, session_manager.context_engineer, session_manager.shared_memory
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
    session_manager = get_session_manager()
    state = session_manager.restore_session(
        session_id, session_manager.context_engineer, session_manager.shared_memory
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
    session_manager = get_session_manager()
    state = session_manager.restore_session(
        session_id, session_manager.context_engineer, session_manager.shared_memory
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
            from src.cli.ui.markdown_stream import render_markdown_result

            render_markdown_result(cli.console, output, title="Result")
        else:
            cli.console.print("[green]No pending actions.[/green]")
        return

    from src.cli.ui.todo_panel import TodoItem, render_todo_panel

    items = [
        TodoItem(
            id=ap["id"],
            title=ap["title"],
            status=ap["status"],
            detail=f"Type: {ap['tool_type']}, Preview: {ap['preview']}" if ap["status"] == "pending" else None,
        )
        for ap in proposals
    ]
    render_todo_panel(cli.console, "Action Proposals", items)

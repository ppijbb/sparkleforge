"""Interactive human feedback provider for the AFTER_PLANNING HITL checkpoint.

Wires src.core.anvil.hitl_checkpoint.HITLCheckpointManager into the real
plan-approval path (see verification.py:verify_plan) instead of leaving it
unused. Only engaged when a human can actually respond (a real TTY on
stdin); autopilot/headless runs never touch this module.
"""

import asyncio
import sys
from typing import Any, Dict, Tuple

from rich.console import Console
from rich.prompt import Prompt

from src.core.anvil.hitl_checkpoint import CheckpointDecision, CheckpointStage

_MENU = (
    "[bold]1[/bold]) Approve the plan as-is\n"
    "[bold]2[/bold]) Revise — add a missing requirement\n"
    "[bold]3[/bold]) Revise — flag a task as wrong/unnecessary\n"
    "[bold]4[/bold]) Abort this research run"
)


def is_interactive() -> bool:
    """True only when stdin is a real TTY a human can type into."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


async def plan_feedback_provider(
    stage: CheckpointStage, context: Dict[str, Any]
) -> Tuple[CheckpointDecision, str]:
    """Console-based feedback provider for the AFTER_PLANNING checkpoint.

    Offers a context-specific menu (approve / add requirement / flag a task /
    abort) rather than a bare APPROVE-REVISE-ABORT prompt, then maps the
    choice down to the (decision, feedback) contract HITLCheckpointManager
    expects.

    ``Prompt.ask`` blocks on stdin, so it runs in a worker thread — this
    function is awaited from HITLCheckpointManager.checkpoint(), and blocking
    the event loop there would stall every other concurrent async task.
    """
    console = Console()
    console.print(f"\n[bold cyan]HITL checkpoint: {stage.value}[/bold cyan]")
    console.print(f"Tasks planned: {context.get('task_count', 0)}")
    console.print(f"Strategy: {context.get('strategy', 'unknown')}")
    for name in context.get("task_names", [])[:5]:
        console.print(f"  - {name}")
    console.print(_MENU)

    choice = await asyncio.to_thread(
        Prompt.ask, "Choice", choices=["1", "2", "3", "4"], default="1"
    )

    if choice == "1":
        return CheckpointDecision.APPROVE, ""
    if choice == "4":
        return CheckpointDecision.ABORT, "Aborted by human reviewer at plan checkpoint"

    detail = await asyncio.to_thread(
        Prompt.ask,
        "Describe the requirement to add" if choice == "2" else "Describe which task is wrong and why",
    )
    prefix = "Add requirement" if choice == "2" else "Fix/remove task"
    return CheckpointDecision.REVISE, f"{prefix}: {detail}"

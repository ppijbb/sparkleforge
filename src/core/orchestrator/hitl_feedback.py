"""Interactive human feedback provider for the AFTER_PLANNING HITL checkpoint.

Wires src.core.anvil.hitl_checkpoint.HITLCheckpointManager into the real
plan-approval path (see verification.py:verify_plan) instead of leaving it
unused. Only engaged when a human can actually respond (a real TTY on
stdin); autopilot/headless runs never touch this module.
"""

import sys
from typing import Any, Dict, Tuple

from src.cli.ui.confirm import MenuOption, free_text, menu_choice
from src.core.anvil.hitl_checkpoint import CheckpointDecision, CheckpointStage

_OPTIONS = [
    MenuOption("1", "Approve the plan as-is"),
    MenuOption("2", "Revise — add a missing requirement"),
    MenuOption("3", "Revise — flag a task as wrong/unnecessary"),
    MenuOption("4", "Abort this research run"),
]


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
    """
    context_lines = [
        f"Tasks planned: {context.get('task_count', 0)}",
        f"Strategy: {context.get('strategy', 'unknown')}",
        *[f"  - {name}" for name in context.get("task_names", [])[:5]],
    ]
    choice = await menu_choice(
        f"HITL checkpoint: {stage.value}", _OPTIONS, default="1", context_lines=context_lines
    )

    if choice == "1":
        return CheckpointDecision.APPROVE, ""
    if choice == "4":
        return CheckpointDecision.ABORT, "Aborted by human reviewer at plan checkpoint"

    detail = await free_text(
        "Describe the requirement to add" if choice == "2" else "Describe which task is wrong and why",
    )
    prefix = "Add requirement" if choice == "2" else "Fix/remove task"
    return CheckpointDecision.REVISE, f"{prefix}: {detail}"

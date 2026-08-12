"""Shared menu-style confirmation prompt for human-in-the-loop checkpoints.

Before this existed, `src/core/orchestrator/hitl_feedback.py` built its own
throwaway `Console()` instance and rolled its own numbered-menu prompt,
disconnected from every other approval surface in the CLI (REPL
`approve`/`deny`, the top-level `approve`/`deny` subcommands). This is the
one place that pattern lives now, styled with the same vocabulary as the
rest of the CLI (`src/cli/ui/theme.py`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from rich import get_console
from rich.prompt import Prompt


@dataclass
class MenuOption:
    key: str
    label: str


async def menu_choice(
    title: str,
    options: list[MenuOption],
    default: str,
    context_lines: list[str] | None = None,
) -> str:
    """Show a numbered menu and block (off the event loop) for the user's pick.

    `Prompt.ask` blocks on stdin, so it runs in a worker thread -- callers
    awaiting this from inside an async checkpoint won't stall the event loop.
    """
    console = get_console()
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    for line in context_lines or []:
        console.print(line)
    for option in options:
        console.print(f"[bold]{option.key}[/bold]) {option.label}")

    return await asyncio.to_thread(
        Prompt.ask, "Choice", choices=[o.key for o in options], default=default
    )


async def free_text(prompt: str) -> str:
    """Prompt for a line of free text, off the event loop."""
    return await asyncio.to_thread(Prompt.ask, prompt)

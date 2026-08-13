"""Live-style checklist panel for a set of tracked items.

Matches cline's FocusChain / opencode's todo-item pattern: a `[✓]/[●]/[✗]/[•]/[ ]`
status glyph per item, with overall progress folded into the panel title,
instead of a flat bullet-point list. Used for anything shaped like a small
list of named things each with a status -- action proposals today
(`src/cli/commands/work.py`), potentially `TaskGraph`/`UnifiedTask` nodes
(`src/core/task_graph.py`) elsewhere later since they share the same
id/title/status shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

_GLYPH = {
    "completed": ("✓", "green"),
    "approved": ("✓", "green"),
    "done": ("✓", "green"),
    "failed": ("✗", "red"),
    "denied": ("✗", "red"),
    "running": ("●", "cyan"),
    "ready": ("•", "yellow"),
    "pending": (" ", "dim"),
}
_DONE_STATUSES = {"completed", "approved", "done"}


@dataclass
class TodoItem:
    id: str
    title: str
    status: str
    detail: str | None = None


def render_todo_panel(console: Console, title: str, items: list[TodoItem]) -> None:
    """Print a checklist panel; does nothing for an empty list."""
    if not items:
        return

    done = sum(1 for item in items if item.status in _DONE_STATUSES)
    body = Text()
    for item in items:
        glyph, style = _GLYPH.get(item.status, ("•", "white"))
        body.append(f"[{glyph}] ", style=style)
        body.append(f"{item.id}  ", style="dim")
        body.append(f"{item.title}", style=style)
        body.append(f"  ({item.status})\n", style="dim")
        if item.detail:
            body.append(f"      {item.detail}\n", style="dim italic")

    console.print(Panel(body, title=f"{title} — {done}/{len(items)} done", border_style="cyan"))


def _demo() -> None:
    """`python -m src.cli.ui.todo_panel` -- smallest runnable check."""
    console = Console()
    items = [
        TodoItem("a1", "Write report", "completed"),
        TodoItem("a2", "Run tests", "running"),
        TodoItem("a3", "Deploy", "pending", detail="Type: docker, Preview: up -d"),
    ]
    render_todo_panel(console, "Action Proposals", items)
    render_todo_panel(console, "Empty", [])  # must not raise or print anything
    print("todo_panel self-check OK")


if __name__ == "__main__":
    _demo()

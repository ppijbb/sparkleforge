"""Shared stage-aware spinner for long-running agent operations.

Wraps `console.status()` and rewrites its caption as execution moves
through stages (instead of sitting on a static "Working..." the whole
run), by tailing specific loggers for lines that indicate a stage change.

`work_command`/`approve_command`/`deny_command` (agent_harness/agent_loop
node names) and `research_command` (autonomous_orchestrator's own stage
announcements) used to each hand-roll an independent logging.Handler
subclass with a different keyword heuristic to do this; this is the one
component both now drive.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterable

from rich.markup import escape

# (substring to look for in a log message, spinner caption to show when found).
# First match wins, so more specific node names are listed ahead of the
# generic stage words research_command's orchestrator also logs.
DEFAULT_STAGE_LABELS: list[tuple[str, str]] = [
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
    ("Planning", "📋 Planning tasks..."),
    ("Analyzing", "🔍 Analyzing..."),
    ("Compressing", "🗜️  Compressing context..."),
    ("Verifying", "✅ Verifying..."),
    ("Evaluating", "📊 Evaluating..."),
    ("Synthesizing", "📝 Synthesizing results..."),
    ("Searching", "⚙️  Searching..."),
    ("Researching", "⚙️  Researching..."),
]

# Lines that carry their own useful detail (which tool, which step) get
# echoed verbatim instead of collapsed to a static label.
DEFAULT_ECHO_NEEDLES: tuple[str, ...] = (
    "[AgentLoop] Executing tool:",
    "[AgentLoop] Iteration",
)


class _StageStatusHandler(logging.Handler):
    """Updates a rich Status spinner's label as tailed loggers move through stages."""

    def __init__(
        self,
        status,
        stage_labels: list[tuple[str, str]],
        echo_needles: tuple[str, ...],
    ):
        super().__init__()
        self.status = status
        self.stage_labels = stage_labels
        self.echo_needles = echo_needles

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        for needle in self.echo_needles:
            if needle in msg:
                detail = escape(msg.split("]", 1)[-1].strip())
                self.status.update(f"[bold cyan]⚙️  {detail}")
                return
        for needle, label in self.stage_labels:
            if needle in msg:
                self.status.update(f"[bold cyan]{label}")
                return


@contextmanager
def stage_status(
    console,
    initial_label: str,
    logger_names: Iterable[str],
    stage_labels: list[tuple[str, str]] | None = None,
    echo_needles: tuple[str, ...] = DEFAULT_ECHO_NEEDLES,
):
    """Like `console.status()`, but its caption tracks execution progress.

    Attaches a logging.Handler to `logger_names` for the duration of the
    `with` block that rewrites the spinner's label as matching log lines
    come through.
    """
    with console.status(f"[bold cyan]{initial_label}", spinner="dots") as status:
        handler = _StageStatusHandler(status, stage_labels or DEFAULT_STAGE_LABELS, echo_needles)
        loggers = [logging.getLogger(name) for name in logger_names]
        for lg in loggers:
            lg.addHandler(handler)
        try:
            yield status
        finally:
            for lg in loggers:
                lg.removeHandler(handler)

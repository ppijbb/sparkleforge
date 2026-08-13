"""Markdown-to-terminal rendering for agent/research output.

Final research/work results are markdown-formatted text (the same content
that gets written to `output/*.md`), but the REPL was dumping it into a
plain `rich.panel.Panel` -- headers, bold, code fences, etc. all showed up
as raw `#`/`**`/`` ``` `` characters instead of being rendered. This renders
it through rich's own `Markdown` (mdflow's terminal-markdown pattern,
minus a token-level streaming layer: nothing in the LLM call path here
emits partial tokens to the CLI today, so there's nothing to stream).
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


def render_markdown_result(console: Console, content: str, title: str = "Result") -> None:
    """Print `content` as rendered markdown inside a titled panel."""
    if not content:
        return
    console.print(Panel(Markdown(content), title=title, border_style="green"))


def _demo() -> None:
    """`python -m src.cli.ui.markdown_stream` -- smallest runnable check."""
    console = Console()
    render_markdown_result(
        console,
        "# Heading\n\nSome **bold** text and a list:\n\n- one\n- two\n\n```python\nprint('hi')\n```",
    )
    render_markdown_result(console, "", title="Empty")  # must not raise or print anything
    print("markdown_stream self-check OK")


if __name__ == "__main__":
    _demo()

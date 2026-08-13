"""Diff rendering for file-edit tool results.

Renders a diff with a line-number gutter and colored add/remove (matching
cline's DiffView experience) instead of a bare "파일 작업 완료" one-liner.
Built on stdlib difflib.SequenceMatcher; unchanged runs longer than
`context_lines` at each end are elided so a one-line change in a long file
doesn't dump the whole file to the terminal.
"""

from __future__ import annotations

import difflib

from rich.console import Console
from rich.text import Text


def render_diff(
    console: Console,
    file_path: str,
    old_text: str,
    new_text: str,
    context_lines: int = 3,
) -> bool:
    """Print a colored, line-numbered diff of `old_text` -> `new_text`.

    Returns False (and prints nothing) if the two texts are identical.
    """
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    opcodes = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes()

    if all(tag == "equal" for tag, *_ in opcodes):
        return False

    body = Text()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            block = list(range(i1, i2))
            if len(block) > 2 * context_lines:
                for idx in block[:context_lines]:
                    body.append(f"  {idx + 1:>5} │ {old_lines[idx]}\n", style="dim")
                skipped = len(block) - 2 * context_lines
                body.append(f"        … {skipped} unchanged line(s) …\n", style="dim italic")
                for idx in block[-context_lines:]:
                    body.append(f"  {idx + 1:>5} │ {old_lines[idx]}\n", style="dim")
            else:
                for idx in block:
                    body.append(f"  {idx + 1:>5} │ {old_lines[idx]}\n", style="dim")
        elif tag in ("delete", "replace"):
            for idx in range(i1, i2):
                body.append(f"- {idx + 1:>5} │ {old_lines[idx]}\n", style="red")
            if tag == "replace":
                for idx in range(j1, j2):
                    body.append(f"+ {idx + 1:>5} │ {new_lines[idx]}\n", style="green")
        elif tag == "insert":
            for idx in range(j1, j2):
                body.append(f"+ {idx + 1:>5} │ {new_lines[idx]}\n", style="green")

    console.print(Text(file_path, style="bold"))
    console.print(body)
    return True


def _demo() -> None:
    """`python -m src.cli.ui.diff_view` -- smallest runnable check for the collapsing logic."""
    old = "\n".join(f"line {i}" for i in range(1, 21))
    new = old.replace("line 10", "line ten (edited)")
    console = Console()
    changed = render_diff(console, "example.txt", old, new, context_lines=2)
    assert changed, "expected a diff to be detected"
    assert render_diff(console, "example.txt", old, old) is False, "identical text must report no diff"
    print("diff_view self-check OK")


if __name__ == "__main__":
    _demo()

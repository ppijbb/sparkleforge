"""CLI entry point for the installed sparkleforge command."""

import os
import sys
from pathlib import Path


_RUN_OPTIONS_WITH_VALUES = {
    "--output",
    "-o",
    "--format",
    "--max-tokens",
    "--model",
}

def _run_command_has_query(argv: list[str]) -> bool:
    """Return True when argv contains a positional query for the run command."""
    if len(argv) < 2 or argv[1] != "run":
        return True

    skip_next = False
    for arg in argv[2:]:
        if skip_next:
            skip_next = False
            continue
        if arg in _RUN_OPTIONS_WITH_VALUES:
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        return True
    return False


def _inject_stdin_query_for_run() -> None:
    """Support automation that pipes the run query through stdin."""
    if _run_command_has_query(sys.argv) or sys.stdin.isatty():
        return

    query = sys.stdin.read().strip()
    if query:
        sys.argv.insert(2, query)


def main_entry() -> None:
    """Run the repository-level CLI entry point from an installed script."""
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    _inject_stdin_query_for_run()

    from src.cli.commands.run import main_entry as repository_main_entry

    repository_main_entry()


if __name__ == "__main__":
    main_entry()

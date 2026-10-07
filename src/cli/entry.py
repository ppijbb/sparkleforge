<<<<<<< ours
import sys
import logging

logger = logging.getLogger("sparkleforge.cli")

def main():
    # Check for --help or help command without printing banners upfront
    if "--help" in sys.argv or "-h" in sys.argv or (len(sys.argv) > 1 and sys.argv[1] == "help"):
        # Print clean help or defer banner
        pass
    else:
        # Only log/print startup info if verbose or debug is enabled
        if any(arg in sys.argv for arg in ("--verbose", "-v", "--debug")):
            logger.debug("SparkleForge: Where Ideas Sparkle and Get Forged ⚒️✨")

    # Import main entry/command handlers lazily or after arg checks if needed
    from src.cli.main_commands import cli_main
    cli_main()

if __name__ == "__main__":
    main()
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
    "--task",
    "--session",
    "--mode",
    "--depth",
    "--autopilot",
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
    os.environ.setdefault("SPARKLEFORGE_ENV", "development")

    # Load .env file before any other logic to ensure consistent config
    from dotenv import load_dotenv
    load_dotenv(override=True)

    project_root = Path(__file__).resolve().parent.parent.parent
    # Do NOT chdir here (issue #1222): the installed sparkleforge/sparkle
    # command must operate on the caller's cwd, like git/npx/eslint do.
    # SparkleForge's own state (db/logs/.env/etc.) is already resolved via
    # the `project_root` computed from __file__ in main.py and
    # autonomous_research_system.py, so it stays package-relative regardless
    # of where this is invoked from.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    _inject_stdin_query_for_run()

    from main import main_entry as repository_main_entry

    repository_main_entry()


if __name__ == "__main__":
    main_entry()
=======
import sys
import logging

logger = logging.getLogger("sparkleforge.cli")

def main():
    # Check for --help or help command without printing banners upfront
    if "--help" in sys.argv or "-h" in sys.argv or (len(sys.argv) > 1 and sys.argv[1] == "help"):
        # Print clean help or defer banner
        pass
    else:
        # Only log/print startup info if verbose or debug is enabled
        if any(arg in sys.argv for arg in ("--verbose", "-v", "--debug")):
            logger.debug("SparkleForge: Where Ideas Sparkle and Get Forged ⚒️✨")

    # Import main entry/command handlers lazily or after arg checks if needed
    from src.cli.main_commands import cli_main
    cli_main()

if __name__ == "__main__":
    main()
>>>>>>> theirs

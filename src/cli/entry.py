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
    # Runtime Isolation & CWD Workspace / Session Binding (Issue #1908)
    # Isolate sys.path so external PYTHONPATH pollution from other projects (e.g. RTM)
    # does not leak into SparkleForge, while anchoring the project runtime root and
    # ensuring local session/history storage binds to the current working directory (CWD).
    cwd = Path.cwd().resolve()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    # Clean up foreign 'src' or package entries from sys.path if they belong to other roots
    sys.path = [p for p in sys.path if not p or Path(p).resolve() == cwd or Path(p).resolve() == project_root or not (Path(p).resolve() / "src").exists() or Path(p).resolve().name == "sparkleforge"]

    os.environ["SPARKLEFORGE_WORKSPACE_ROOT"] = str(cwd)
    os.environ.setdefault("SPARKLEFORGE_SESSION_DB", str(cwd / ".sparkleforge" / "session.db"))

    _inject_stdin_query_for_run()

    from main import main_entry as repository_main_entry

    repository_main_entry()


if __name__ == "__main__":
    main_entry()

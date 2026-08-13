"""Shared logger-suppression policy for the REPL's "quiet mode" and for
actively-watching-an-agent-work console output.

Both main.py's REPL entry path and REPLCLI's constructor need the exact
same suppression applied; keeping it in one place instead of two hardcoded
copies is what stops it from silently drifting out of sync (checkpoint
manager / session storage / session control / context loader all leaked
raw init chatter above the REPL banner because neither of the old copies
had been updated when those modules added their own __init__ logging).
"""

from __future__ import annotations

import logging

# Infra/init loggers whose own status text isn't meant for the chat surface --
# real chat output goes through src.cli.* or is printed directly via rich,
# never through these. A new "src.core.*" module that logs at __init__ time
# should be added here rather than silently leaking through.
QUIET_LOGGER_NAMES = (
    "__main__",
    "src.core.agent_orchestrator",
    "src.core.mcp_integration",
    "src.core.shared_memory",
    "src.core.skills_manager",
    "src.core.prompt_refiner_wrapper",
    "src.core.checkpoint_manager",
    "src.core.session_storage",
    "src.core.session_control",
    "src.core.context_loader",
    "src.core.scheduler",
    "streamlit",
    "streamlit.runtime",
    "local_researcher",
)


def apply_repl_quiet_mode() -> None:
    """Suppress infra/init logger chatter so only chat-facing output reaches the console."""
    for logger_name in QUIET_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


class ChatModeFilter(logging.Filter):
    """Console-only allowlist for while an agent turn is actively running.

    QUIET_LOGGER_NAMES is a blocklist and it keeps drifting: every internal
    module that starts logging at INFO (capability_manager's "Granted
    capability ...", mode_controller's "Execution mode switched: ...",
    llm_manager's "Executing with NVIDIA NIM model: ...", agent_loop's
    "Iteration N/30", ...) leaks to the console as raw internal-state text
    until someone notices and adds it to the list one name at a time. None
    of that is what a user watching an agent work wants to see -- it's
    orchestration plumbing, not "here's what the agent just did". The
    curated version of that (a spinner caption, a per-tool-call trace line)
    already exists elsewhere; this just stops the raw duplicate from also
    reaching the console.

    Flipped to an allowlist instead: only src.cli.* (actual chat-facing
    output) and WARNING+ (real problems) get through. Everything else still
    goes to the log file -- this only changes what's visible on screen.
    Scoped to attach/detach around the specific span where an agent is
    actively executing (`stage_status()`), not applied globally, so
    diagnostic commands like `health`/`tools` that print non-src.cli INFO
    lines as their actual payload are unaffected.
    """

    ALLOWED_PREFIXES = ("src.cli",)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return record.name.startswith(self.ALLOWED_PREFIXES)


def find_console_handler():
    """Return the RichConsoleHandler on the root logger, if any."""
    from src.cli.ui.console_log_handler import RichConsoleHandler

    for handler in logging.getLogger().handlers:
        if isinstance(handler, RichConsoleHandler):
            return handler
    return None

"""Shared logger-suppression policy for the REPL's "quiet mode".

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


# agent_loop/agent_harness's routine progress lines ("[AgentLoop] Executing
# tool: X", "[AgentLoop] Iteration N/30") already have dedicated UI:
# src/cli/ui/spinner.py's _StageStatusHandler turns them into the spinner's
# caption, and src/utils/output_manager.py's output_tool_execution renders a
# proper trace line per completed tool call. Printing them AGAIN as raw log
# lines is pure duplication -- but these loggers can't just be added to
# QUIET_LOGGER_NAMES (that suppresses the logger itself below INFO, which
# would also stop _StageStatusHandler -- attached directly to the same
# logger -- from ever seeing the records it needs for the spinner caption).
# A console-handler-only Filter blocks console rendering without touching
# the logger's level, so both mechanisms keep working. WARNING+ still gets
# through: routine progress is what's duplicated, not real problems.
CONSOLE_FILTERED_LOGGER_PREFIXES = (
    "src.core.agent_loop",
    "src.core.agent_harness",
)


class ChatModeFilter(logging.Filter):
    """Allowlist filter for chat turns: only src.cli.* INFO and WARNING+ reach console.

    The existing QUIET_LOGGER_NAMES blocklist keeps drifting -- every new
    src.core.* module that logs at __init__ time leaks internal state-machine
    chatter (capability grants, mode switches, model routing) to the console
    until someone remembers to add it. This is the inverse: during an active
    chat turn (scoped to stage_status()'s span), only actual chat-facing
    output (src.cli.*) and real problems (WARNING+) are rendered to the
    console handler; everything else still goes to the log file. Diagnostic
    subcommands (health/tools) that print non-src.cli INFO as their payload
    are unaffected because the filter is attached/detached around the span,
    not applied globally.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return record.name == "src.cli" or record.name.startswith("src.cli.")


def attach_chat_mode_filter(handler: logging.Handler) -> ChatModeFilter:
    """Attach a ChatModeFilter to ``handler`` and return it for later removal."""
    filt = ChatModeFilter()
    handler.addFilter(filt)
    return filt


def detach_chat_mode_filter(handler: logging.Handler, filt: ChatModeFilter) -> None:
    """Remove a previously-attached ChatModeFilter from ``handler``."""
    try:
        handler.removeFilter(filt)
    except Exception:
        pass


class ConsoleInternalsFilter(logging.Filter):
    """Keeps routine agent_loop/agent_harness progress lines off the console (still logged to file)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return not record.name.startswith(CONSOLE_FILTERED_LOGGER_PREFIXES)

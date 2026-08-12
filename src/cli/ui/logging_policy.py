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

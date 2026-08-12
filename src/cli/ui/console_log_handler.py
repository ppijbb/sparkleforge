"""A logging.Handler that renders records through rich instead of raw text.

Replaces the `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"` console
formatter, which is what made every non-REPL subcommand (`health`, `tools`,
`docker`, ...) print raw timestamped log lines instead of styled output. File
logging (full detail, every level) is untouched -- this only changes what
reaches the terminal.
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console, get_console
from rich.markup import escape

from src.cli.ui import theme
from src.cli.ui.console_log_handler import ConsoleInternalsFilter


class RichConsoleHandler(logging.Handler):
    """Prints log records via rich, styled/iconified by level (or by a
    leading known icon already in the message, e.g. "✅ Done").

    Looks up `sys.stderr` fresh on every emit rather than binding it once at
    construction, so it keeps working after rich's own Live/Status redirects
    stderr for the duration of a spinner (this replaces the old
    `_LiveAwareStderr` proxy class main.py maintained for the same reason).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addFilter(ConsoleInternalsFilter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            console = get_console()
            style = theme.style_for_levelname(record.levelname)
            console.print(theme.markup_for(message, style), highlight=False)
            if record.exc_info:
                formatter = logging.Formatter()
                traceback_text = escape(formatter.formatException(record.exc_info))
                console.print(f"[dim]{traceback_text}[/dim]")
        except Exception:
            self.handleError(record)


class ConsoleInternalsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno <= logging.INFO and record.name in ("agent_loop", "agent_harness"):
            return False
        return True

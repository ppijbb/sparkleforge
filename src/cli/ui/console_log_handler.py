"""A logging.Handler that renders records through rich instead of raw text.

Replaces the `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"` console
formatter, which is what made every non-REPL subcommand (`health`, `tools`,
`docker`, ...) print raw timestamped log lines instead of styled output. File
logging (full detail, every level) is untouched -- this only changes what
reaches the terminal.
"""

from __future__ import annotations

import logging

from rich import get_console
from rich.markup import escape

from src.cli.ui import theme


class RichConsoleHandler(logging.Handler):
    """Prints log records via rich, styled/iconified by level (or by a
    leading known icon already in the message, e.g. "✅ Done").

    Renders through rich's process-wide get_console() singleton -- the same
    one REPLCLI/output_manager/spinners share -- rather than a private
    Console(file=sys.stderr). A log line during an active spinner (e.g.
    "[AgentLoop] Executing tool: ...") used to render through its own fresh
    Console bound to stderr: confirmed live in a real tty, that write isn't
    coordinated with the *other* console's active Live region, so the two
    fight over the same terminal and the log line ends up concatenated onto
    the spinner's current frame with no newline instead of appearing as its
    own line. A print from the same Console object an active Status/Live
    belongs to is something rich already handles correctly (that's the
    entire point of Live's "print above the live region" support) -- a print
    from a *different* Console object is not.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Renders log records through the global console singleton.

        Note: We explicitly set soft_wrap=True on the print call to ensure
        long log lines (JSON, paths) do not hard-wrap at the terminal width.
        """
        try:
            message = record.getMessage()
            console = get_console()
            style = theme.style_for_levelname(record.levelname)
            console.print(theme.markup_for(message, style), highlight=False, soft_wrap=True)
            if record.exc_info:
                formatter = logging.Formatter()
                traceback_text = escape(formatter.formatException(record.exc_info))
                console.print(f"[dim]{traceback_text}[/dim]", soft_wrap=True)
        except Exception:
            self.handleError(record)

"""Shared icon/color vocabulary for SparkleForge's terminal output.

Every place that used to hand-roll its own ``f"[red]❌ {msg}[/red]"`` or
``f"[green]✅ {msg}[/green]"`` markup string should go through
:func:`success`, :func:`error`, :func:`warning`, or :func:`info` here
instead, so the whole CLI shares one visual language.
"""

from __future__ import annotations

from rich.markup import escape

# Icon already carries the semantic meaning (success/error/...), so callers
# that pass a pre-iconified message (many existing "✅ ..."/"❌ ..." strings
# scattered across the codebase) are recognized here and not double-iconified.
_KNOWN_ICON_STYLES: dict[str, str] = {
    "✅": "green",
    "❌": "red",
    "⚠️": "yellow",
    "🛑": "red",
    "ℹ️": "cyan",
}

STYLE_SUCCESS = "green"
STYLE_ERROR = "red"
STYLE_WARNING = "yellow"
STYLE_INFO = "cyan"
STYLE_DIM = "dim"

ICON_SUCCESS = "✅"
ICON_ERROR = "❌"
ICON_WARNING = "⚠️"
ICON_INFO = "ℹ️"


def _strip_known_icon(message: str) -> tuple[str, str | None]:
    """If `message` already starts with a known status icon, split it off."""
    stripped = message.lstrip()
    for icon, style in _KNOWN_ICON_STYLES.items():
        if stripped.startswith(icon):
            return stripped[len(icon):].strip(), style
    return message, None


def markup_for(message: str, default_style: str) -> str:
    """Build a rich markup string for `message`, styled by its icon or `default_style`.

    `message` is arbitrary dynamic text (file paths, tool output, exception
    text, ...) that may itself contain `[...]`-shaped substrings -- escaped
    here so it can't be parsed as (and potentially crash on, e.g. a stray
    `[/bold]`) Rich console markup.
    """
    body, detected_style = _strip_known_icon(message)
    style = detected_style or default_style
    icon = {
        STYLE_SUCCESS: ICON_SUCCESS,
        STYLE_ERROR: ICON_ERROR,
        STYLE_WARNING: ICON_WARNING,
        STYLE_INFO: ICON_INFO,
    }.get(style, "")
    prefix = f"{icon} " if icon else ""
    return f"[{style}]{prefix}{escape(body)}[/{style}]"


def style_for_levelname(levelname: str) -> str:
    """Map a stdlib logging levelname to a rich style, for the console log handler."""
    return {
        "DEBUG": STYLE_DIM,
        "INFO": "default",
        "WARNING": STYLE_WARNING,
        "ERROR": STYLE_ERROR,
        "CRITICAL": "bold red",
    }.get(levelname, "default")

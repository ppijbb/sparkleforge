"""Live elapsed-time indicator for long-running provider calls.

A primary-model LLM call can take up to ~180s per attempt with no output in
between (see model_registry.py's NVIDIA client timeout) -- the terminal just
goes silent until it either returns or times out. `with_progress` wraps an
awaitable with a 1s-tick "{label}... {elapsed}s" status line, matching the
elapsed-time-indicator pattern used by claude-code/opencode/openclaw for the
same problem. Skipped entirely when stdout isn't a real terminal (CI logs,
piped output) to avoid spamming captured logs with carriage-return noise.
"""

import asyncio
import sys
import time
from typing import Awaitable, TypeVar

from rich.console import Console
from rich.markup import escape

T = TypeVar("T")


async def with_progress(coro: Awaitable[T], label: str, interval: float = 1.0) -> T:
    """Await `coro`, printing a live elapsed-time status line while it runs."""
    # sys.stdout may be swapped for a wrapper without isatty() (e.g.
    # SupabaseStdoutRedirector during a research run) -- treat that as
    # "not a real terminal" rather than crashing the call it's wrapping.
    isatty = getattr(sys.stdout, "isatty", None)
    if not callable(isatty) or not isatty():
        return await coro

    console = Console()
    start = time.time()
    safe_label = escape(label)

    async def _tick(status) -> None:
        while True:
            elapsed = time.time() - start
            status.update(f"[bold cyan]⏳ {safe_label}... {elapsed:.0f}s")
            await asyncio.sleep(interval)

    with console.status(f"[bold cyan]⏳ {safe_label}...", spinner="dots") as status:
        ticker = asyncio.create_task(_tick(status))
        try:
            return await coro
        finally:
            ticker.cancel()
            try:
                await ticker
            except asyncio.CancelledError:
                pass

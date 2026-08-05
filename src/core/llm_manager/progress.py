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

T = TypeVar("T")

_CLEAR_LINE = "\r\033[K"


async def with_progress(coro: Awaitable[T], label: str, interval: float = 1.0) -> T:
    """Await `coro`, printing a live elapsed-time status line while it runs."""
    # sys.stdout may be swapped for a wrapper without isatty() (e.g.
    # SupabaseStdoutRedirector during a research run) -- treat that as
    # "not a real terminal" rather than crashing the call it's wrapping.
    isatty = getattr(sys.stdout, "isatty", None)
    if not callable(isatty) or not isatty():
        return await coro

    start = time.time()

    async def _tick() -> None:
        while True:
            elapsed = time.time() - start
            try:
                print(f"\r⏳ {label}... {elapsed:.0f}s", end="", flush=True)
            except UnicodeEncodeError:
                # Terminal encoding can't represent the emoji (e.g. legacy
                # Windows code pages, LC_ALL=C) -- stop ticking rather than
                # crash the call this is just decorating.
                return
            await asyncio.sleep(interval)

    ticker = asyncio.create_task(_tick())
    try:
        return await coro
    finally:
        ticker.cancel()
        try:
            await ticker
        except asyncio.CancelledError:
            pass
        print(_CLEAR_LINE, end="", flush=True)

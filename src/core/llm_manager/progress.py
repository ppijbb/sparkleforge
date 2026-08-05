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


async def with_progress(coro: Awaitable[T], label: str) -> T:
    """Await `coro`, printing a live elapsed-time status line while it runs."""
    if not sys.stdout.isatty():
        return await coro

    start = time.time()
    done = asyncio.Event()

    async def _tick() -> None:
        while not done.is_set():
            elapsed = time.time() - start
            print(f"\r⏳ {label}... {elapsed:.0f}s", end="", flush=True)
            try:
                await asyncio.wait_for(done.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    ticker = asyncio.create_task(_tick())
    try:
        return await coro
    finally:
        done.set()
        await ticker
        print("\r" + " " * (len(label) + 20) + "\r", end="", flush=True)

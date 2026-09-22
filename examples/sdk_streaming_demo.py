#!/usr/bin/env python3
"""Demo: subscribe to intermediate progress while src.sdk.run() executes (#1621).

    python examples/sdk_streaming_demo.py "Latest AI trends in 2025"

on_progress is awaited once per orchestrator event streamed for this
specific run (today: a handful of events from analysis.py -- most nodes
don't call stream_event() yet, so expect a sparse stream, not a full
per-agent play-by-play).
"""

import asyncio
import sys

from src.sdk import ProgressEvent, run


async def _print_progress(event: ProgressEvent) -> None:
    pct = f"{event.percentage:.0f}%" if event.percentage is not None else "?"
    print(f"[{event.timestamp}] ({pct}) {event.phase} <{event.agent_name}>: {event.message}")


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Latest AI trends in 2025"
    result = await run(prompt, on_progress=_print_progress)
    print("\n--- Final result ---")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

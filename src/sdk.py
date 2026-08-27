"""Programmatic entrypoint for running SparkleForge research in-process.

Equivalent to headless CLI mode (``sparkleforge --prompt "..."``,
``main.py:700``/``main.py:1019``), for callers that want to embed SparkleForge
in their own pipeline instead of spawning a subprocess for every request.

    from src.sdk import run
    result = await run("Latest AI trends in 2025")

``AutonomousOrchestrator`` (``src/core/autonomous_orchestrator.py``) reads its
settings from ``src.core.researcher_config``'s module-level ``config``, which
starts as ``None`` and is only populated by calling ``load_config_from_env()``
-- normally done implicitly by the CLI's bootstrap sequence before any command
runs. A bare import of ``AutonomousOrchestrator`` skips that, so ``run()``
loads it first if nothing else already has.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

_config_load_lock = asyncio.Lock()


async def _ensure_config_loaded() -> None:
    from src.core import researcher_config

    async with _config_load_lock:
        if researcher_config.config is None:
            researcher_config.load_config_from_env()


async def run(prompt: str) -> Dict[str, Any]:
    """Run one research request and return its result dict.

    Exceptions from the underlying orchestrator (including provider/config
    errors) propagate as-is rather than being wrapped -- the same behavior
    main.py's own `--prompt` headless path already has.
    """
    await _ensure_config_loaded()

    # Local import: avoids loading AutonomousOrchestrator's (and its
    # dependencies') full import graph for callers who only need e.g. the
    # status API. Patch target for tests is src.sdk.AutonomousOrchestrator
    # only if imported at module level here; as-is, patch
    # src.core.autonomous_orchestrator.AutonomousOrchestrator instead.
    from src.core.autonomous_orchestrator import AutonomousOrchestrator

    orchestrator = AutonomousOrchestrator()
    return await orchestrator.run_research(prompt)

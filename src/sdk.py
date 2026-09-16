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
import uuid
from datetime import datetime, timezone

_jobs: Dict[str, Dict[str, Any]] = {}
_job_tasks: Dict[str, asyncio.Task] = {}

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


async def _execute_job(job_id: str, prompt: str) -> None:
    try:
        result = await run(prompt)
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
    finally:
        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        if job_id in _job_tasks:
            del _job_tasks[job_id]


async def submit_job(topic: str, **kwargs: Any) -> str:
    """Submit a research job asynchronously and return its unique job_id."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "running",
        "prompt": topic,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    task = asyncio.create_task(_execute_job(job_id, topic))
    _job_tasks[job_id] = task
    return job_id


async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get the current progress status, error, and completion status of a job."""
    job = _jobs.get(job_id)
    if job is None:
        from src.utils.supabase_exporter import get_supabase_client, get_job_status as sb_get_job_status, SupabaseQueryError
        if get_supabase_client() is not None:
            try:
                sb_job = await sb_get_job_status(job_id)
                if sb_job is not None:
                    return sb_job
            except SupabaseQueryError:
                pass
        raise ValueError(f"Job not found: {job_id}")
    return {"job_id": job_id, "status": job["status"], "submitted_at": job["submitted_at"], "error": job.get("error")}


async def get_report(job_id: str) -> Dict[str, Any] | None:
    """Get the finished report result for a completed job."""
    job = _jobs.get(job_id)
    if job is None:
        from src.utils.supabase_exporter import get_supabase_client, get_report as sb_get_report, SupabaseQueryError
        if get_supabase_client() is not None:
            try:
                return await sb_get_report(job_id)
            except SupabaseQueryError:
                pass
        return None
    return job.get("result")

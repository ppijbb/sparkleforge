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
import os
from datetime import datetime, timezone
from typing import Optional, TypedDict, Literal

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


class JobStatus(TypedDict):
    job_id: str
    status: Literal["running", "completed", "failed"]
    prompt: str
    submitted_at: str
    completed_at: Optional[str]
    error: Optional[str]


# Local fallback store for jobs when Supabase is not configured or for in-process mode
_local_jobs: Dict[str, Dict[str, Any]] = {}
_local_reports: Dict[str, Dict[str, Any]] = {}


async def _execute_bg_job(job_id: str, prompt: str) -> None:
    try:
        result = await run(prompt)
        _local_jobs[job_id]["status"] = "completed"
        _local_jobs[job_id]["result"] = result
        _local_reports[job_id] = result
    except Exception as e:
        _local_jobs[job_id]["status"] = "failed"
        _local_jobs[job_id]["error"] = str(e)
    finally:
        _local_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()


async def submit_job(prompt: str) -> str:
    """Submit a research task in the background and return a unique job_id."""
    job_id = str(uuid.uuid4())
    now_str = datetime.now(timezone.utc).isoformat()
    
    # Check if Supabase client is available and configured
    from src.utils.supabase_exporter import get_supabase_client
    if get_supabase_client() is not None:
        # Supabase-backed jobs could be inserted here if table exists,
        # but fallback or hybrid storage is also supported. Let's register in local or supabase.
        pass

    _local_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "prompt": prompt,
        "submitted_at": now_str,
        "completed_at": None,
        "error": None,
    }
    asyncio.create_task(_execute_bg_job(job_id, prompt))
    return job_id


async def get_job_status(job_id: str) -> Optional[JobStatus]:
    """Get the current status of an asynchronous job by job_id."""
    # First check local store
    if job_id in _local_jobs:
        j = _local_jobs[job_id]
        return {
            "job_id": j["job_id"],
            "status": j["status"],
            "prompt": j["prompt"],
            "submitted_at": j["submitted_at"],
            "completed_at": j.get("completed_at"),
            "error": j.get("error"),
        }

    # Try Supabase if configured
    from src.utils.supabase_exporter import get_supabase_client
    if get_supabase_client() is not None:
        from src.utils.supabase_exporter import get_job_status as sb_get_job_status
        try:
            sb_job = await sb_get_job_status(job_id)
            if sb_job:
                return sb_job
        except Exception:
            pass

    return None


async def get_report(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the final research report for a completed job."""
    if job_id in _local_reports:
        return _local_reports[job_id]
    
    from src.utils.supabase_exporter import get_supabase_client, get_report as sb_get_report
    if get_supabase_client() is not None:
        return await sb_get_report(job_id)
    return None

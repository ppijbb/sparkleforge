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

Also provides asynchronous background job submission and status polling:

    from src.sdk import submit_job, get_job_status, get_report

    job_id = await submit_job("Latest AI trends in 2025")
    status = await get_job_status(job_id)
    report = await get_report(job_id)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid

from src.utils.logger import get_logger

logger = get_logger(__name__)

_config_load_lock = asyncio.Lock()

# Global in-memory registry for jobs submitted in-process.
# Serves as local fallback when Supabase is not configured or in local mode,
# and shares state with status_api.
# Backed by HybridStorage for durability across restarts when local.
from src.storage.hybrid_storage import HybridStorage
_hybrid_storage = HybridStorage()
_jobs: Dict[str, Dict[str, Any]] = {}
_job_tasks: Dict[str, asyncio.Task[Any]] = {}


@dataclass
class JobStatus:
    """Represents the execution and progress status of a research job."""

    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    topic: Optional[str] = None
    prompt: Optional[str] = None
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job status to a serializable dictionary."""
        data: Dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
        }
        if self.topic is not None:
            data["topic"] = self.topic
        if self.prompt is not None:
            data["prompt"] = self.prompt
        if self.submitted_at is not None:
            data["submitted_at"] = self.submitted_at
        if self.completed_at is not None:
            data["completed_at"] = self.completed_at
        if self.error is not None:
            data["error"] = self.error
        if self.result is not None:
            data["result"] = self.result
        if self.user_id is not None:
            data["user_id"] = self.user_id
        return data

    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.to_dict().get(item, default)


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


async def _execute_job(job_id: str, topic: str, user_id: Optional[str] = None) -> None:
    """Execute research job asynchronously and update status in local and remote stores."""
    from src.utils.supabase_exporter import (
        SupabaseExporter,
        get_supabase_client,
        update_job_status,
    )

    _jobs[job_id]["status"] = "running"
    if get_supabase_client() is not None:
        try:
            await update_job_status(job_id, "running")
        except Exception as e:
            logger.warning(f"Failed to update running status in Supabase for job {job_id}: {e}")

    try:
        result = await run(topic)
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result

        if get_supabase_client() is not None:
            try:
                await update_job_status(job_id, "completed")
                if isinstance(result, dict):
                    report_payload = dict(result)
                    report_payload.setdefault("id", job_id)
                    report_payload.setdefault("topic", topic)
                    await _hybrid_storage.save_research_result({
                        "objective_id": job_id,
                        "user_request": topic,
                        "final_synthesis": {"content": result.get("content", str(result))},
                        "execution_results": result.get("execution_results", []),
                        "validation_score": result.get("confidence", 0.8),
                    })
                    if user_id:
                        report_payload.setdefault("user_id", user_id)
                    exporter = SupabaseExporter()
                    await exporter.publish_report(report_payload)
            except Exception as e:
                logger.warning(f"Failed to export completed job {job_id} to Supabase: {e}")
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)

        if get_supabase_client() is not None:
            try:
                await update_job_status(job_id, "failed", error_message=str(e))
            except Exception as err:
                logger.warning(
                    f"Failed to update failed status in Supabase for job {job_id}: {err}"
                )
    finally:
        completed_at = datetime.now(timezone.utc).isoformat()
        _jobs[job_id]["completed_at"] = completed_at
        _job_tasks.pop(job_id, None)
        # Save state to local hybrid storage for durability
        try:
            status_val = _jobs[job_id].get("status")
            res_val = _jobs[job_id].get("result") or {}
            if isinstance(res_val, str):
                res_val = {"content": res_val}
            content_val = res_val.get("content", _jobs[job_id].get("error", ""))
            await _hybrid_storage.store_research(
                research_id=job_id,
                user_id=_jobs[job_id].get("user_id", "local_user"),
                topic=_jobs[job_id].get("topic", ""),
                content=content_val,
                results=res_val,
                metadata={"status": status_val, "error": _jobs[job_id].get("error")},
                summary=content_val[:500] if content_val else "",
            )
        except Exception as ex:
            logger.debug("Failed to persist job to hybrid storage: %s", ex)


async def submit_job(topic: str, *, user_id: Optional[str] = None, **kwargs: Any) -> str:
    """Submit a research job asynchronously and return its unique job_id."""
    job_id: Optional[str] = None

    from src.utils.supabase_exporter import create_job, get_supabase_client

    if get_supabase_client() is not None:
        try:
            created = await create_job(topic, user_id=user_id)
            if created and "id" in created:
                job_id = str(created["id"])
        except Exception as e:
            logger.warning(f"Failed to create job in Supabase, falling back to local: {e}")

    if not job_id:
        job_id = str(uuid.uuid4())

    now = datetime.now(timezone.utc).isoformat()
    _jobs[job_id] = {
        "job_id": job_id,
        "topic": topic,
        "prompt": topic,
        "status": "running",
        "submitted_at": now,
        "user_id": user_id,
    }

    task = asyncio.create_task(_execute_job(job_id, topic, user_id=user_id))
    _job_tasks[job_id] = task
    
    # Persist initial running state to hybrid storage
    try:
        await _hybrid_storage.store_research(
            research_id=job_id,
            user_id=user_id or "local_user",
            topic=topic,
            content="",
            results={},
            metadata={"status": "running"},
            summary="",
        )
    except Exception as ex:
        logger.debug("Failed to persist initial job state: %s", ex)
        
    return job_id


async def get_job_status(job_id: str) -> JobStatus:
    """Get the current progress status, error, and completion status of a job."""
    job = _jobs.get(job_id)
    if job is not None:
        return JobStatus(
            job_id=job_id,
            status=job.get("status", "pending"),
            topic=job.get("topic"),
            prompt=job.get("prompt"),
            submitted_at=job.get("submitted_at"),
            completed_at=job.get("completed_at"),
            error=job.get("error"),
            result=job.get("result"),
            user_id=job.get("user_id"),
        )

    # Fallback to local HybridStorage if not in memory
    try:
        history = await _hybrid_storage.get_user_research_history("local_user", limit=100)
        for mem in history:
            if mem.research_id == job_id:
                meta = mem.metadata or {}
                st = meta.get("status", "completed" if mem.content else "pending")
                err = meta.get("error")
                return JobStatus(
                    job_id=job_id,
                    status=st,
                    topic=mem.topic,
                    prompt=mem.topic,
                    submitted_at=mem.timestamp.isoformat() if mem.timestamp else None,
                    completed_at=mem.timestamp.isoformat() if st in ("completed", "failed") else None,
                    error=err,
                    result=mem.results,
                    user_id=mem.user_id,
                )
    except Exception as e:
        logger.debug("Failed to query hybrid storage for job status: %s", e)

    from src.utils.supabase_exporter import (

    from src.utils.supabase_exporter import (
        SupabaseQueryError,
        get_job_status as sb_get_job_status,
        get_supabase_client,
    )

    if get_supabase_client() is not None:
        try:
            sb_row = await sb_get_job_status(job_id)
            if sb_row is not None:
                return JobStatus(
                    job_id=str(sb_row.get("id", job_id)),
                    status=sb_row.get("status", "pending"),
                    topic=sb_row.get("topic"),
                    prompt=sb_row.get("topic"),
                    submitted_at=sb_row.get("created_at"),
                    completed_at=sb_row.get("updated_at")
                    if sb_row.get("status") in ("completed", "failed")
                    else None,
                    error=sb_row.get("error_message"),
                    user_id=sb_row.get("user_id"),
                )
        except SupabaseQueryError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to query job status: {e}") from e

    raise ValueError(f"Job not found: {job_id}")


async def get_report(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the finished report result for a completed job, or None if not found/unfinished."""
    job = _jobs.get(job_id)
    if job is not None and job.get("result") is not None:
        return job.get("result")

    # Fallback to hybrid storage index/files
    try:
        history = await _hybrid_storage.get_user_research_history("local_user", limit=100)
        for mem in history:
            if mem.research_id == job_id:
                if mem.results:
                    return mem.results
                if mem.content:
                    return {"content": mem.content, "summary": mem.summary}
    except Exception as e:
        logger.debug("Failed to query hybrid storage for report: %s", e)

    from src.utils.supabase_exporter import (
        SupabaseQueryError,
        get_report as sb_get_report,
        get_supabase_client,
    )

    if get_supabase_client() is not None:
        try:
            sb_report = await sb_get_report(job_id)
            if sb_report is not None:
                return sb_report
        except SupabaseQueryError:
            raise
        except Exception:
            pass

    return None

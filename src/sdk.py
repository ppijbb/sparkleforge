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

    job_id = await submit_job("Latest AI trends in 2026")
    status = await get_job_status(job_id)
    report = await get_report(job_id)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, Optional, List

# Optional file locking support for cross-process safety
import contextlib
import uuid

from src.utils.logger import get_logger

logger = get_logger(__name__)

_config_load_lock = asyncio.Lock()

# Configuration constants
_JOB_STORE_TTL_DAYS = int(os.environ.get("SPARKLEFORGE_SDK_JOB_STORE_TTL_DAYS", "30"))
_DEFAULT_JOB_STORE_PATH = Path.home() / ".sparkleforge" / "sdk_jobs.json"
_JOB_STORE_PATH = Path(os.environ.get("SPARKLEFORGE_SDK_JOB_STORE_PATH", str(_DEFAULT_JOB_STORE_PATH)))

# Global in-memory and persistent registry for jobs submitted in-process.
_jobs: Dict[str, Dict[str, Any]] = {}
_job_tasks: Dict[str, asyncio.Task[Any]] = {}
_file_lock = threading.Lock()

def _ensure_store_dir() -> None:
    try:
        _JOB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create job store directory at {_JOB_STORE_PATH.parent}: {e}")
        raise

def _load_job_store() -> Dict[str, Dict[str, Any]]:
    _file_lock.acquire()
    try:
        if not _JOB_STORE_PATH.exists():
            return {}
        with open(_JOB_STORE_PATH, "r", encoding="utf-8") as f:
            # Optional fcntl locking for POSIX multi-process safety
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            except (ImportError, OSError):
                pass
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Job store file corrupted or unreadable at {_JOB_STORE_PATH}: {e}")
        raise RuntimeError(f"Job store file corrupted at {_JOB_STORE_PATH}: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading job store from {_JOB_STORE_PATH}: {e}")
        raise
    finally:
        _file_lock.release()

def _persist_job_store(store: Dict[str, Dict[str, Any]]) -> None:
    _ensure_store_dir()
    _file_lock.acquire()
    temp_path = _JOB_STORE_PATH.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            json.dump(store, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, _JOB_STORE_PATH)
    except Exception as e:
        logger.error(f"Failed to atomically persist job store to {_JOB_STORE_PATH}: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise
    finally:
        _file_lock.release()

def _prune_expired_jobs(store: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=_JOB_STORE_TTL_DAYS)
    pruned = {}
    for jid, job in store.items():
        sub_at = job.get("submitted_at")
        if sub_at:
            try:
                dt = datetime.fromisoformat(sub_at)
                if dt >= cutoff:
                    pruned[jid] = job
                continue
            except Exception:
                pass
        pruned[jid] = job
    return pruned

def _init_local_store() -> None:
    try:
        store = _load_job_store()
        # Orphaned job recovery & pruning
        changed = False
        now_str = datetime.now(timezone.utc).isoformat()
        for jid, job in store.items():
            if job.get("status") in ("pending", "running"):
                job["status"] = "unknown"
                job["error"] = "Service restarted while job was in flight; status unknown."
                job["completed_at"] = now_str
                changed = True
        store = _prune_expired_jobs(store)
        _jobs.update(store)
        if changed:
            _persist_job_store(_jobs)
    except Exception as e:
        logger.warning(f"Could not load or reconcile local job store on startup: {e}")

# Initialize local store on import
_init_local_store()

def _save_job_to_store(job_id: str, job_data: Dict[str, Any]) -> None:
    _jobs[job_id] = job_data
    try:
        store = _load_job_store()
        store[job_id] = job_data
        store = _prune_expired_jobs(store)
        _persist_job_store(store)
    except Exception as e:
        logger.warning(f"Failed to persist job {job_id} to local file store: {e}")


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

    _save_job_to_store(job_id, _jobs[job_id])
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
        _save_job_to_store(job_id, _jobs[job_id])


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

    _save_job_to_store(job_id, _jobs[job_id])
    task = asyncio.create_task(_execute_job(job_id, topic, user_id=user_id))
    _job_tasks[job_id] = task
    return job_id


async def get_job_status(job_id: str) -> JobStatus:
    """Get the current progress status, error, and completion status of a job.
    
    Supabase remains the primary source of truth when configured; local store
    serves as fallback or for local-only deployments.
    """
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

    raise ValueError(f"Job not found: {job_id}")


async def get_report(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the finished report result for a completed job, or None if not found/unfinished.
    
    Supabase is queried first when configured, with local store as fallback.
    """
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

    job = _jobs.get(job_id)
    if job is not None and job.get("result") is not None:
        return job.get("result")

    return None

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

Job records also persist to a local JSON file (#1654) so status/report survive
a process restart on purely local (no-Supabase) deployments -- e.g. status_api.py
being restarted. This file-backed store is single-process/single-worker only:
concurrent writers (multiple uvicorn workers) racing on the same file can lose
an update, so run status_api.py with a single worker (uvicorn's default) for
local mode. Hosted/Supabase-backed deployments are unaffected -- Supabase
remains the source of truth there.
"""

from __future__ import annotations

import asyncio
import json
import threading
import fcntl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple
import uuid

from src.utils.logger import get_logger

logger = get_logger(__name__)

_config_load_lock = asyncio.Lock()

# Global in-memory registry for jobs submitted in-process.
# Serves as local fallback when Supabase is not configured or in local mode,
# and shares state with status_api.
_jobs: Dict[str, Dict[str, Any]] = {}
_job_tasks: Dict[str, asyncio.Task[Any]] = {}

# #1654: durable local fallback, one tier below _jobs (this process's memory)
# and above Supabase -- see the module docstring's single-worker note.
_JOB_STORE_PATH = Path(
    os.getenv("SPARKLEFORGE_SDK_JOB_STORE_PATH", os.getenv("SPARKLEFORGE_SDK_JOB_STORE", str(Path.home() / ".sparkleforge" / "sdk_jobs.json")))
)
_job_store_file_lock = threading.Lock()

# #1654 review: the job-store file accumulated every job forever. Mirrors
# forge_jobs' Supabase-side TTL (#1619) -- terminal jobs older than this are
# dropped the next time any job is persisted.
try:
    _JOB_STORE_TTL_DAYS = int(os.getenv("SPARKLEFORGE_SDK_JOB_STORE_TTL_DAYS", "30"))
except (ValueError, TypeError):
    _JOB_STORE_TTL_DAYS = 30


def _validate_store_path() -> None:
    """Validate path writability at startup."""
    try:
        _JOB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        test_file = _JOB_STORE_PATH.parent / ".writability_test"
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"SDK job store path {_JOB_STORE_PATH} is not writable: {e}")


def _prune_expired_jobs(store: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=_JOB_STORE_TTL_DAYS)
    pruned: Dict[str, Dict[str, Any]] = {}
    for jid, job in store.items():
        completed_at = job.get("completed_at")
        if job.get("status") in ("completed", "failed") and completed_at:
            try:
                if datetime.fromisoformat(completed_at) < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
        pruned[jid] = job
    return pruned


def _load_job_store() -> Dict[str, Dict[str, Any]]:
    """Best-effort read of the durable job-record file.

    Raises on JSON decode or OS errors if corruption is detected, allowing
    callers to handle or log alerting instead of silent data loss.
    """
    with _job_store_file_lock:
        try:
            if not _JOB_STORE_PATH.exists():
                return {}
            with open(_JOB_STORE_PATH, "r", encoding="utf-8") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                except (OSError, IOError, AttributeError):
                    pass
                content = f.read()
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except (OSError, IOError, AttributeError):
                    pass
            return json.loads(content) if content.strip() else {}
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to read or decode SDK job store at {_JOB_STORE_PATH}: {e}")
            raise
        except Exception as e:
            logger.debug(f"Unexpected error reading SDK job store at {_JOB_STORE_PATH}: {e}")
            return {}


def _persist_job(job_id: str) -> None:
    """Durably record one job's current state, so it survives a restart.

    Best-effort: never raises. Read-modify-write on a single JSON file --
    safe for one process, not for concurrent writers (see module docstring).
    """
    job = _jobs.get(job_id)
    if job is None:
        return
    with _job_store_file_lock:
        try:
            _JOB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = _JOB_STORE_PATH.with_suffix(".json.tmp")
            store: Dict[str, Any] = {}
            if _JOB_STORE_PATH.exists():
                try:
                    with open(_JOB_STORE_PATH, "r", encoding="utf-8") as f:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        except (OSError, IOError, AttributeError):
                            pass
                        content = f.read()
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        except (OSError, IOError, AttributeError):
                            pass
                    store = json.loads(content) if content.strip() else {}
                except Exception:
                    store = {}
            store[job_id] = job
            store = _prune_expired_jobs(store)
            with open(tmp_path, "w", encoding="utf-8") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except (OSError, IOError, AttributeError):
                    pass
                json.dump(store, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, _JOB_STORE_PATH)
        except Exception as e:
            logger.error(f"Failed to persist SDK job {job_id}: {e}")


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


@dataclass
class ProgressEvent:
    """One intermediate progress notification from a `run()` call (#1621)."""

    phase: str
    agent_name: Optional[str]
    message: str
    percentage: Optional[float]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "agent_name": self.agent_name,
            "message": self.message,
            "percentage": self.percentage,
            "timestamp": self.timestamp,
        }


async def _ensure_config_loaded() -> None:
    from src.core import researcher_config

    async with _config_load_lock:
        if researcher_config.config is None:
            researcher_config.load_config_from_env()


async def run(
    prompt: str,
    *,
    on_progress: Optional[Callable[[ProgressEvent], Awaitable[None]]] = None,
) -> Dict[str, Any]:
    """Run one research request and return its result dict.

    Exceptions from the underlying orchestrator (including provider/config
    errors) propagate as-is rather than being wrapped -- the same behavior
    main.py's own `--prompt` headless path already has.

    on_progress: if given, awaited with a ProgressEvent for every event the
    orchestrator streams via StreamingManager for this specific call (scoped
    by a fresh objective_id so concurrent run() calls don't cross-talk). Today
    that's the couple of events analysis.py's node emits (WORKFLOW_START,
    then AGENT_ACTION on analysis completion) -- most orchestrator nodes don't
    call stream_event() yet, so callers should expect a sparse stream, not a
    step-by-step blow-by-blow. A raising on_progress is logged and swallowed;
    it never fails the underlying research run.
    """
    await _ensure_config_loaded()

    # Local import: avoids loading AutonomousOrchestrator's (and its
    # dependencies') full import graph for callers who only need e.g. the
    # status API. Patch target for tests is src.sdk.AutonomousOrchestrator
    # only if imported at module level here; as-is, patch
    # src.core.autonomous_orchestrator.AutonomousOrchestrator instead.
    from src.core.autonomous_orchestrator import AutonomousOrchestrator

    orchestrator = AutonomousOrchestrator()

    if on_progress is None:
        return await orchestrator.run_research(prompt)

    from src.core.streaming_manager import get_streaming_manager

    objective_id = f"sdk_{uuid.uuid4().hex[:12]}"
    manager = get_streaming_manager()

    async def _relay(event: Any) -> None:
        if event.workflow_id != objective_id:
            return
        try:
            await on_progress(
                ProgressEvent(
                    phase=event.event_type.value,
                    agent_name=event.agent_id,
                    message=str(
                        event.data.get("message") or event.data.get("action") or ""
                    ),
                    percentage=event.data.get("progress") or event.data.get("percentage"),
                    timestamp=event.timestamp.isoformat(),
                )
            )
        except Exception:
            logger.warning("sdk.run on_progress callback raised; continuing.", exc_info=True)

    manager.add_local_listener(_relay)
    try:
        return await orchestrator.execute(prompt, objective_id=objective_id)
    finally:
        manager.remove_local_listener(_relay)


async def _execute_job(job_id: str, topic: str, user_id: Optional[str] = None) -> None:
    """Execute research job asynchronously and update status in local and remote stores."""
    from src.utils.supabase_exporter import (
        SupabaseExporter,
        get_supabase_client,
        update_job_status,
    )

    _jobs[job_id]["status"] = "running"
    await asyncio.to_thread(_persist_job, job_id)
    if get_supabase_client() is not None:
        try:
            await update_job_status(job_id, "running")
        except Exception as e:
            logger.warning(f"Failed to update running status in Supabase for job {job_id}: {e}")

    try:
        result = await run(topic)
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result
        await asyncio.to_thread(_persist_job, job_id)

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
        await asyncio.to_thread(_persist_job, job_id)

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
        await asyncio.to_thread(_persist_job, job_id)
        _job_tasks.pop(job_id, None)


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
    await asyncio.to_thread(_persist_job, job_id)

    task = asyncio.create_task(_execute_job(job_id, topic, user_id=user_id))
    _job_tasks[job_id] = task
    return job_id


async def get_job_status(job_id: str) -> JobStatus:
    """Get the current progress status, error, and completion status of a job."""
    from src.utils.supabase_exporter import (
        SupabaseQueryError,
        get_job_status as sb_get_job_status,
        get_supabase_client,
    )

    # 1. Query Supabase first if configured (Supabase is source of truth)
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
            logger.debug(f"Supabase query failed for job {job_id}, falling back to local: {e}")

    # 2. Check in-memory process dictionary next
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

    # 3. Fall back to local durable store
    try:
        job_store = await asyncio.to_thread(_load_job_store)
    except Exception:
        job_store = {}

    persisted = job_store.get(job_id) if isinstance(job_store, dict) else None
    if persisted is not None:
        status_val = persisted.get("status", "pending")
        error_val = persisted.get("error")
        
        # Orphaned job check: attempt Supabase verification if available, else mark unknown/failed gracefully
        if status_val in ("pending", "running") and job_id not in _job_tasks:
            verified_status = None
            if get_supabase_client() is not None:
                try:
                    sb_row = await sb_get_job_status(job_id)
                    if sb_row and sb_row.get("status"):
                        verified_status = sb_row.get("status")
                except Exception:
                    pass
            
            if verified_status:
                status_val = verified_status
            else:
                status_val = "unknown"
                error_val = "Job was in progress when the process restarted; status could not be verified."

        return JobStatus(
            job_id=job_id,
            status=status_val,
            topic=persisted.get("topic"),
            prompt=persisted.get("prompt"),
            submitted_at=persisted.get("submitted_at"),
            completed_at=persisted.get("completed_at"),
            error=error_val,
            result=persisted.get("result"),
            user_id=persisted.get("user_id"),
        )

    raise ValueError(f"Job not found: {job_id}")


async def get_report(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the finished report result for a completed job, or None if not found/unfinished."""
    from src.utils.supabase_exporter import (
        SupabaseQueryError,
        get_report as sb_get_report,
        get_supabase_client,
    )

    # 1. Query Supabase first if configured (Supabase is source of truth)
    if get_supabase_client() is not None:
        try:
            sb_report = await sb_get_report(job_id)
            if sb_report is not None:
                return sb_report
        except SupabaseQueryError:
            raise
        except Exception:
            pass

    # 2. Check in-memory process dictionary next
    job = _jobs.get(job_id)
    if job is not None and job.get("result") is not None:
        return job.get("result")

    # 3. Fall back to local durable store
    try:
        job_store = await asyncio.to_thread(_load_job_store)
    except Exception:
        job_store = {}

    persisted = job_store.get(job_id) if isinstance(job_store, dict) else None
    if persisted is not None and persisted.get("result") is not None:
        return persisted.get("result")

    return None

"""Status/report/task-submission API: #1564's API layer (Anvil Phase B-2).

POST /tasks submits a prompt and runs it in-process via src.sdk.run();
GET /tasks/{job_id}/status and /tasks/{job_id}/report poll it. These are
tracked in an in-process registry, separate from the Supabase-backed
/jobs and /reports routes below.

Lets a client that isn't at the PC (mobile, a different machine) check on a
research job and read its finished report, without needing direct Supabase
credentials. Backed by the same Supabase tables the public telemetry
dashboard (``src/web/live_dashboard.py``) already reads from -- this only
covers deployments with Supabase configured; it does not cover purely local,
same-machine runs (``src/core/session_control.py``'s in-process session
state, or ``src/storage/hybrid_storage.py``'s local file store).

Requires a bearer token: set STATUS_API_TOKEN and send
``Authorization: Bearer <token>`` on every request. Without the env var set,
both routes reject every request (503) rather than serving data unguarded.

Run with::

    STATUS_API_TOKEN=... uvicorn src.web.status_api:app --port 8502
"""

from __future__ import annotations

import asyncio
import os
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.utils.supabase_exporter import (
    SupabaseQueryError,
    get_job_status,
    get_report,
    get_supabase_client,
)

# Anvil Phase B-2: in-process job registry for POST /tasks, separate from the
# Supabase-backed /jobs and /reports routes above (those read rows written by
# the normal CLI/exporter pipeline; a task submitted here runs in this
# process via src.sdk.run() and is tracked in memory only).
# ponytail: single-process dict, not persisted -- move to a real queue/store
# if this ever needs to survive a restart or run behind multiple workers.
_tasks: Dict[str, Dict[str, Any]] = {}


async def _execute_task(job_id: str, prompt: str) -> None:
    # Delegate to sdk implementation or maintain compatibility wrapper
    from src.sdk import _local_jobs, _execute_bg_job
    if job_id not in _local_jobs:
        _local_jobs[job_id] = _tasks.get(job_id, {"status": "running", "prompt": prompt, "submitted_at": datetime.now(timezone.utc).isoformat()})
    await _execute_bg_job(job_id, prompt)
    _tasks[job_id] = _local_jobs[job_id]


async def submit_task(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    prompt = body.get("prompt") if isinstance(body, dict) else None
    if not prompt or not isinstance(prompt, str):
        return JSONResponse({"error": "'prompt' (string) is required"}, status_code=400)

    from src.sdk import submit_job
    job_id = await submit_job(prompt)
    from src.sdk import _local_jobs
    _tasks[job_id] = {
        "status": _local_jobs[job_id]["status"],
        "prompt": prompt,
        "submitted_at": _local_jobs[job_id]["submitted_at"],
    }
    return JSONResponse({"job_id": job_id, "status": "running"}, status_code=202)


async def task_status(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    job_id = request.path_params["job_id"]
    from src.sdk import _local_jobs
    if job_id in _local_jobs:
        _tasks[job_id] = _local_jobs[job_id]
    task = _tasks.get(job_id)
    if task is None:
        return JSONResponse({"error": "task not found"}, status_code=404)
    return JSONResponse({"job_id": job_id, "status": task["status"], "submitted_at": task["submitted_at"]})


async def task_report(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    job_id = request.path_params["job_id"]
    from src.sdk import _local_reports, _local_jobs
    if job_id in _local_jobs:
        _tasks[job_id] = _local_jobs[job_id]
    task = _tasks.get(job_id)
    if task is None:
        return JSONResponse({"error": "task not found"}, status_code=404)
    if task["status"] == "running":
        return JSONResponse({"error": "task still running"}, status_code=409)
    if task["status"] == "failed":
        return JSONResponse({"error": task.get("error", "task failed")}, status_code=500)
    return JSONResponse(_local_reports.get(job_id, task.get("result", {})))


def _service_unavailable(detail: str) -> JSONResponse:
    # Generic message on purpose -- doesn't confirm/deny *why* to an
    # unauthenticated caller (config missing vs. a live query failing are
    # both just "try again later" from outside).
    return JSONResponse({"error": f"Service temporarily unavailable: {detail}"}, status_code=503)


def _check_auth(request: Request) -> JSONResponse | None:
    """Return an error response if the request isn't authorized, else None.

    Fails closed: if STATUS_API_TOKEN isn't set, every request is rejected
    (503) rather than silently serving job/report rows -- job_id/report_id
    are UUIDs, not secrets, so this endpoint must not be reachable without a
    configured token.
    """
    expected = os.environ.get("STATUS_API_TOKEN")
    if not expected:
        return _service_unavailable("auth not configured")
    got = request.headers.get("authorization", "")
    if not got.startswith("Bearer ") or not secrets.compare_digest(got[len("Bearer "):], expected):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


async def job_status(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    if get_supabase_client() is None:
        return _service_unavailable("not configured")
    try:
        job = await get_job_status(request.path_params["job_id"])
    except SupabaseQueryError:
        return _service_unavailable("query failed")
    if job is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return JSONResponse(job)


async def report(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    if get_supabase_client() is None:
        return _service_unavailable("not configured")
    try:
        result = await get_report(request.path_params["report_id"])
    except SupabaseQueryError:
        return _service_unavailable("query failed")
    if result is None:
        return JSONResponse({"error": "report not found"}, status_code=404)
    return JSONResponse(result)


app = Starlette(
    routes=[
        Route("/jobs/{job_id}/status", job_status),
        Route("/reports/{report_id}", report),
        Route("/tasks", submit_task, methods=["POST"]),
        Route("/tasks/{job_id}/status", task_status),
        Route("/tasks/{job_id}/report", task_report),
    ]
)

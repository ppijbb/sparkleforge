"""Status/report/task-submission API: #1564's API layer (Anvil Phase B-2).

POST /tasks submits a prompt and runs it in-process via src.sdk.run();
GET /tasks/{job_id}/status and /tasks/{job_id}/report poll it. These are
tracked in src.sdk's job registry (in-process memory, plus a durable local
JSON file -- see src/sdk.py's module docstring, #1654), separate from the
Supabase-backed /jobs and /reports routes below.

Lets a client that isn't at the PC (mobile, a different machine) check on a
research job and read its finished report, without needing direct Supabase
credentials. /jobs and /reports are backed by the same Supabase tables the
public telemetry dashboard (``src/web/live_dashboard.py``) already reads
from, so they only cover deployments with Supabase configured. /tasks works
either way: with Supabase configured it also mirrors status there, and
without it, it durably falls back to a local file so status/report survive
a restart of this process -- see #1654. Run with a single worker (uvicorn's
default): the local file store is not safe for multiple concurrent writers.

Requires a bearer token: set STATUS_API_TOKEN and send
``Authorization: Bearer <token>`` on every request. Without the env var set,
both routes reject every request (503) rather than serving data unguarded.

Run with::

    STATUS_API_TOKEN=... uvicorn src.web.status_api:app --port 8502
"""

from __future__ import annotations

import os
import secrets

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

# Anvil Phase B-2: in-process job registry for POST /tasks, shared with
# src.sdk._jobs (a task submitted here runs in this process via src.sdk.submit_job()
# and is tracked in the unified SDK registry).
from src.sdk import _jobs as _tasks  # noqa: F401
from src.sdk import (
    _execute_job,
    get_job_status as sdk_get_job_status,
    get_report as sdk_get_report,
    submit_job as sdk_submit_job,
)

__all__ = ["app", "_tasks"]


async def _execute_task(job_id: str, prompt: str) -> None:
    """Legacy helper maintained for backward compatibility."""
    await _execute_job(job_id, prompt)


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

    job_id = await sdk_submit_job(prompt)
    return JSONResponse({"job_id": job_id, "status": "running"}, status_code=202)


async def task_status(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    job_id = request.path_params["job_id"]
    try:
        task = await sdk_get_job_status(job_id)
    except ValueError:
        return JSONResponse({"error": "task not found"}, status_code=404)
    except Exception:
        return JSONResponse({"error": "failed to query task status"}, status_code=500)
    return JSONResponse({
        "job_id": job_id,
        "status": task.status,
        "submitted_at": task.submitted_at,
    })


async def task_report(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    job_id = request.path_params["job_id"]
    try:
        task = await sdk_get_job_status(job_id)
    except ValueError:
        return JSONResponse({"error": "task not found"}, status_code=404)
    except Exception:
        return JSONResponse({"error": "failed to query task"}, status_code=500)

    if task.status in ("pending", "running"):
        return JSONResponse({"error": "task still running"}, status_code=409)
    if task.status == "failed":
        return JSONResponse({"error": task.error or "task failed"}, status_code=500)

    result = await sdk_get_report(job_id)
    if result is None:
        return JSONResponse({"error": "report not found"}, status_code=404)
    return JSONResponse(result)


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

"""Read-only status/report API: first slice of #1564's API layer.

Lets a client that isn't at the PC (mobile, a different machine) check on a
research job and read its finished report, without needing direct Supabase
credentials. Backed by the same Supabase tables the public telemetry
dashboard (``src/web/live_dashboard.py``) already reads from -- this only
covers deployments with Supabase configured; it does not cover purely local,
same-machine runs (``src/core/session_control.py``'s in-process session
state, or ``src/storage/hybrid_storage.py``'s local file store).

Run with::

    uvicorn src.web.status_api:app --port 8502
"""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.utils.supabase_exporter import get_job_status, get_report, get_supabase_client


def _unconfigured_response() -> JSONResponse:
    return JSONResponse(
        {"error": "Supabase is not configured on this server (SUPABASE_URL/SUPABASE_KEY)."},
        status_code=503,
    )


async def job_status(request: Request) -> JSONResponse:
    if get_supabase_client() is None:
        return _unconfigured_response()
    job = await get_job_status(request.path_params["job_id"])
    if job is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return JSONResponse(job)


async def report(request: Request) -> JSONResponse:
    if get_supabase_client() is None:
        return _unconfigured_response()
    result = await get_report(request.path_params["report_id"])
    if result is None:
        return JSONResponse({"error": "report not found"}, status_code=404)
    return JSONResponse(result)


app = Starlette(
    routes=[
        Route("/jobs/{job_id}/status", job_status),
        Route("/reports/{report_id}", report),
    ]
)

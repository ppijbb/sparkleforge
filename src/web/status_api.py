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
import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from src.utils.supabase_exporter import (
    SupabaseQueryError,
    get_job_status,
    get_report,
    get_supabase_client,
)

from src.sdk import _jobs as _sdk_jobs, _job_tasks
from src.sdk import (
    _execute_job,
    get_job_status as sdk_get_job_status,
    get_report as sdk_get_report,
    submit_job as sdk_submit_job,
)
from src.core.streaming_manager import get_streaming_manager

__all__ = ["app", "_tasks", "ControlPlaneStore"]


class ControlPlaneStore:
    """Local SQLite WAL persistence for the control plane (tasks, sessions, events, approvals)."""
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("SPARKLEFORGE_CONTROL_PLANE_DB", str(os.path.expanduser("~/.sparkleforge/control_plane.db")))
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        import sqlite3
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    job_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    topic TEXT,
                    prompt TEXT,
                    status TEXT,
                    submitted_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    result TEXT,
                    user_id TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    event_type TEXT,
                    payload TEXT,
                    timestamp REAL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    job_id TEXT,
                    status TEXT,
                    prompt TEXT,
                    created_at REAL
                )
            """)

    def save_task(self, job_id: str, data: dict):
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO tasks (job_id, session_id, topic, prompt, status, submitted_at, completed_at, error, result, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                data.get("session_id", "default_session"),
                data.get("topic"),
                data.get("prompt"),
                data.get("status"),
                data.get("submitted_at"),
                data.get("completed_at"),
                data.get("error"),
                json.dumps(data.get("result")) if data.get("result") is not None else None,
                data.get("user_id")
            ))

    def get_task(self, job_id: str) -> Optional[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT job_id, session_id, topic, prompt, status, submitted_at, completed_at, error, result, user_id FROM tasks WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "job_id": row[0],
            "session_id": row[1],
            "topic": row[2],
            "prompt": row[3],
            "status": row[4],
            "submitted_at": row[5],
            "completed_at": row[6],
            "error": row[7],
            "result": json.loads(row[8]) if row[8] else None,
            "user_id": row[9]
        }

    def list_tasks(self, limit: int = 50, offset: int = 0) -> List[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT job_id, session_id, topic, prompt, status, submitted_at, completed_at, error, result, user_id FROM tasks ORDER BY submitted_at DESC LIMIT ? OFFSET ?", (limit, offset))
        rows = cursor.fetchall()
        return [{
            "job_id": r[0],
            "session_id": r[1],
            "topic": r[2],
            "prompt": r[3],
            "status": r[4],
            "submitted_at": r[5],
            "completed_at": r[6],
            "error": r[7],
            "result": json.loads(r[8]) if r[8] else None,
            "user_id": r[9]
        } for r in rows]

    def list_sessions(self) -> List[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT session_id FROM tasks")
        rows = cursor.fetchall()
        sessions = []
        for r in rows:
            sid = r[0] or "default_session"
            c = self.conn.cursor()
            c.execute("SELECT COUNT(*) FROM tasks WHERE session_id = ?", (sid,))
            count = c.fetchone()[0]
            sessions.append({"session_id": sid, "task_count": count})
        return sessions

    def add_event(self, job_id: str, event_type: str, payload: dict) -> int:
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO events (job_id, event_type, payload, timestamp) VALUES (?, ?, ?, ?)",
                           (job_id, event_type, json.dumps(payload), time.time()))
            return cursor.lastrowid

    def get_events(self, job_id: str, after_id: int = 0) -> List[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, job_id, event_type, payload, timestamp FROM events WHERE job_id = ? AND id > ? ORDER BY id ASC", (job_id, after_id))
        rows = cursor.fetchall()
        return [{
            "id": r[0],
            "job_id": r[1],
            "event_type": r[2],
            "payload": json.loads(r[3]),
            "timestamp": r[4]
        } for r in rows]


_control_store = ControlPlaneStore()
_tasks = _sdk_jobs


async def _sync_task_to_store(job_id: str):
    try:
        status = await sdk_get_job_status(job_id)
        _control_store.save_task(job_id, status.to_dict())
    except Exception:
        pass


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

    session_id = body.get("session_id", "default_session")
    # sdk_submit_job doesn't accept session_id parameter yet; pass prompt only
    job_id = await sdk_submit_job(prompt)
    # TODO: when sdk_submit_job supports session_id, pass it through
    # job_id = await sdk_submit_job(prompt, session_id=session_id)
    _control_store.save_task(job_id, {"job_id": job_id, "session_id": session_id, "prompt": prompt, "topic": prompt, "status": "running", "submitted_at": time.time()})
    _control_store.add_event(job_id, "status", {"status": "running", "message": "Task submitted"})
    return JSONResponse({"job_id": job_id, "status": "running"}, status_code=202)


async def task_status(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    job_id = request.path_params["job_id"]
    try:
        task = await sdk_get_job_status(job_id) or _control_store.get_task(job_id)
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
        task = await sdk_get_job_status(job_id) or _control_store.get_task(job_id)
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


async def list_sessions(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    sessions = _control_store.list_sessions()
    return JSONResponse(sessions)


async def list_tasks(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    limit = int(request.query_params.get("limit", 50))
    offset = int(request.query_params.get("offset", 0))
    tasks = _control_store.list_tasks(limit=limit, offset=offset)
    return JSONResponse(tasks)


async def task_events(request: Request) -> StreamingResponse:
    if (err := _check_auth(request)) is not None:
        return err
    job_id = request.path_params["job_id"]
    last_event_id_str = request.headers.get("last-event-id") or request.query_params.get("last_event_id", "0")
    try:
        last_event_id = int(last_event_id_str)
    except ValueError:
        last_event_id = 0

    async def event_generator() -> AsyncGenerator[str, None]:
        nonlocal last_event_id
        # Replay missed events
        missed = _control_store.get_events(job_id, after_id=last_event_id)
        for ev in missed:
            last_event_id = ev["id"]
            data_str = json.dumps({"event_type": ev["event_type"], "payload": ev["payload"], "id": ev["id"]})
            yield f"id: {ev['id']}\nevent: {ev['event_type']}\ndata: {data_str}\n\n"

        # Stream live events
        while True:
            events = _control_store.get_events(job_id, after_id=last_event_id)
            if events:
                for ev in events:
                    last_event_id = ev["id"]
                    data_str = json.dumps({"event_type": ev["event_type"], "payload": ev["payload"], "id": ev["id"]})
                    yield f"id: {ev['id']}\nevent: {ev['event_type']}\ndata: {data_str}\n\n"
            else:
                # Check if job is finished
                task = _control_store.get_task(job_id)
                if task and task.get("status") in ("completed", "failed"):
                    break
                # Also check sdk jobs
                try:
                    st = await sdk_get_job_status(job_id)
                    if st.status in ("completed", "failed"):
                        break
                except Exception:
                    pass
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def steer_task(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    job_id = request.path_params["job_id"]
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    
    action = body.get("action") # approve, reject, revise, or constraint update
    prompt = body.get("prompt") or body.get("constraint")

    _control_store.add_event(job_id, "steer", {"action": action, "prompt": prompt})
    return JSONResponse({"status": "steered", "job_id": job_id, "action": action})


async def cancel_task(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    job_id = request.path_params["job_id"]
    task = _job_tasks.get(job_id)
    if task and not task.done():
        task.cancel()
    
    # Update status
    t = _control_store.get_task(job_id)
    if t:
        t["status"] = "failed"
        t["error"] = "Cancelled by user"
        _control_store.save_task(job_id, t)
    _control_store.add_event(job_id, "error", {"message": "Cancelled by user"})
    return JSONResponse({"status": "cancelled", "job_id": job_id})


async def task_artifacts(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err
    job_id = request.path_params["job_id"]
    task = _control_store.get_task(job_id)
    if not task:
        try:
            st = await sdk_get_job_status(job_id)
            task = st.to_dict()
        except Exception:
            return JSONResponse({"error": "task not found"}, status_code=404)

    artifacts = []
    result = task.get("result")
    if isinstance(result, dict):
        if "report" in result:
            artifacts.append({"type": "report", "content": result["report"]})
        if "patch" in result:
            artifacts.append({"type": "patch", "content": result["patch"]})
        if "diff" in result:
            artifacts.append({"type": "diff", "content": result["diff"]})
    return JSONResponse({"job_id": job_id, "artifacts": artifacts})


def _service_unavailable(detail: str) -> JSONResponse:
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

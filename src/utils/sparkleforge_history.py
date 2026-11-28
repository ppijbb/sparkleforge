"""SparkleForge-wide work/conversation history persistence (Supabase).

Records what happened across every subsystem that produces work worth
remembering -- autofix CI, REPL sessions, forge_master sessions -- as a
first-class session + event stream in Supabase's sparkleforge_sessions /
sparkleforge_history_events tables (see supabase_schema.sql).

This is deliberately a separate, dedicated schema/module from
supabase_realtime_logger.py's agent_logs/logs tables: those exist as a live
broadcast backup for the research orchestrator specifically, whereas this
module exists so any subsystem can persist "what happened" as a defined
session with a start/end, independent of real-time streaming.

Callers never block on network I/O and never see an exception from this
module: every write is enqueued to a single background worker thread, and
if Supabase isn't configured, entries are silently dropped (the same
fail-open pattern supabase_exporter/supabase_realtime_logger already use).
start_history_session() always returns a usable session id so callers don't
need to branch on whether persistence is actually active.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.utils.supabase_exporter import get_supabase_client

import sqlite3

logger = logging.getLogger(__name__)

_queue: "queue.Queue" = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_stop_event = threading.Event()

_BATCH_MAX_SIZE = 25
_BATCH_MAX_WAIT_S = 2.0


def get_local_error_ledger_path() -> Path:
    path = Path("storage/error_contexts.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_local_connection() -> sqlite3.Connection:
    db_path = get_local_error_ledger_path()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_error_contexts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            scenario_name TEXT,
            error_type TEXT,
            error_message TEXT,
            execution_context TEXT,
            remediation_status TEXT DEFAULT 'pending',
            resolution_summary TEXT,
            commit_or_patch_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def _write_local_error_context(payload: Dict[str, Any]) -> int:
    conn = get_local_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO agent_error_contexts 
        (session_id, scenario_name, error_type, error_message, execution_context, remediation_status)
        VALUES (?, ?, ?, ?, ?, 'pending')
        """,
        (
            payload.get("session_id"),
            payload.get("scenario_name"),
            payload.get("error_type"),
            payload.get("error_message"),
            json.dumps(payload.get("execution_context", {})),
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def mark_error_resolved(error_id: int | str, resolution_summary: str, commit_or_patch_hash: Optional[str] = None) -> None:
    """Transition remediation_status to 'resolved' and link fix locally and in queue for Supabase."""
    conn = get_local_connection()
    try:
        conn.execute(
            """
            UPDATE agent_error_contexts
            SET remediation_status = 'resolved', resolution_summary = ?, commit_or_patch_hash = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (resolution_summary, commit_or_patch_hash, error_id),
        )
        conn.commit()
    except Exception as e:
        logger.debug("Failed to mark local error resolved: %s", e)
    finally:
        conn.close()

    client = get_supabase_client()
    if client:
        try:
            client.table("agent_error_contexts").update({
                "remediation_status": "resolved",
                "resolution_summary": resolution_summary,
                "commit_or_patch_hash": commit_or_patch_hash,
            }).eq("id", error_id).execute()
        except Exception as e:
            logger.debug("Failed to update error status in Supabase: %s", e)


def compute_ledger_metrics() -> Dict[str, Any]:
    conn = get_local_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM agent_error_contexts")
        total = cursor.fetchone()[0] or 0
        cursor.execute("SELECT COUNT(*) FROM agent_error_contexts WHERE remediation_status = 'resolved'")
        resolved = cursor.fetchone()[0] or 0
        recovery_rate = (resolved / total * 100.0) if total > 0 else 0.0
        
        cursor.execute("SELECT COUNT(*) FROM agent_error_contexts WHERE resolution_summary IS NOT NULL AND resolution_summary != ''")
        reused = cursor.fetchone()[0] or 0
        reuse_rate = (reused / resolved * 100.0) if resolved > 0 else (0.0 if total == 0 else 0.0)
        
        return {
            "total_errors": total,
            "resolved_errors": resolved,
            "recovery_rate": recovery_rate,
            "reuse_rate": reuse_rate,
        }
    except Exception as e:
        logger.debug("Failed to compute ledger metrics: %s", e)
        return {"total_errors": 0, "resolved_errors": 0, "recovery_rate": 0.0, "reuse_rate": 0.0}
    finally:
        conn.close()


def _ensure_worker() -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread and _worker_thread.is_alive():
            return
        _stop_event.clear()
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()


def _worker_loop() -> None:
    client = get_supabase_client()
    if not client:
        # Not configured: drain the queue so callers never block on a full
        # queue, but do no network I/O.
        while not _stop_event.is_set():
            try:
                _queue.get(timeout=0.1)
                _queue.task_done()
            except queue.Empty:
                continue
        return

    while not _stop_event.is_set():
        batch = []
        deadline = time.monotonic() + _BATCH_MAX_WAIT_S
        while len(batch) < _BATCH_MAX_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = _queue.get(timeout=remaining)
            except queue.Empty:
                break
            batch.append(item)
            _queue.task_done()

        for kind, payload in batch:
            try:
                if kind == "session_start":
                    client.table("sparkleforge_sessions").insert(payload).execute()
                elif kind == "session_end":
                    client.table("sparkleforge_sessions").update(payload["fields"]).eq(
                        "id", payload["id"]
                    ).execute()
                elif kind == "event":
                    client.table("sparkleforge_history_events").insert(payload).execute()
                elif kind == "error_context":
                    client.table("agent_error_contexts").insert(payload).execute()
                    # Also ensure local ledger has it
                    _write_local_error_context(payload)
            except Exception as e:
                logger.debug("Failed to write sparkleforge history (%s): %s", kind, e)


def start_history_session(
    source: str,
    *,
    external_ref: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Begin a new history session and return its id.

    Args:
        source: which subsystem this session belongs to, e.g. "autofix",
            "repl", "forge_master".
        external_ref: an external identifier for correlation, e.g. a GitHub
            issue number/URL or a REPL session id.
    """
    session_id = str(uuid.uuid4())
    _ensure_worker()
    _queue.put(
        (
            "session_start",
            {
                "id": session_id,
                "source": source,
                "external_ref": external_ref,
                "title": title,
                "status": "running",
                "metadata": metadata or {},
            },
        )
    )
    return session_id


def log_history_event(
    session_id: str,
    event_type: str,
    content: str = "",
    *,
    role: Optional[str] = None,
    backend: Optional[str] = None,
    level: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one event to a session's history.

    Args:
        event_type: "message" | "log" | "llm_call" | "tool_call" | "commit"
            | "error" (free-form; not enforced).
        role: "user" | "assistant" | "system" | "tool" for conversational
            content; omit for non-conversational events.
        backend: which LLM backend actually served this event, if any, e.g.
            "nvidia:nemotron-3-ultra-550b-a55b" or "openrouter:z-ai/glm-5.2:free".
    """
    _ensure_worker()
    _queue.put(
        (
            "event",
            {
                "session_id": session_id,
                "event_type": event_type,
                "role": role,
                "backend": backend,
                "level": level,
                "content": content,
                "metadata": metadata or {},
            },
        )
    )


def log_error_context(
    error_type: str,
    error_message: str,
    *,
    session_id: Optional[str] = None,
    scenario_name: Optional[str] = None,
    execution_context: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a structured failure to agent_error_contexts (see supabase_schema.sql).

    Same fire-and-forget, fail-open contract as log_history_event: callers
    never block on network I/O and never see an exception from this module.
    """
    _ensure_worker()
    payload_dict = {
        "session_id": session_id,
        "scenario_name": scenario_name,
        "error_type": error_type,
        "error_message": error_message,
        "execution_context": execution_context or {},
    }
    _write_local_error_context(payload_dict)

    _queue.put(
        (
            "error_context",
            payload_dict,
        )
    )


def fetch_recent_patch_failures(issue_number: str, limit: int = 1) -> list:
    """Best-effort read of this issue's most recently recorded PatchApplyFailure rows.

    Synchronous (unlike the write path above) since callers need the result
    before building the next prompt. Fails open -- returns [] if Supabase
    isn't configured or the query errors -- because this is a diagnostic
    nice-to-have, never a hard dependency for fix_issue().
    """
    client = get_supabase_client()
    if not client:
        # Fallback to local SQLite ledger
        conn = get_local_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT error_message, execution_context, created_at FROM agent_error_contexts
                WHERE error_type = 'PatchApplyFailure' AND json_extract(execution_context, '$.issue_number') = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (str(issue_number), limit),
            )
            rows = cursor.fetchall()
            return [{"error_message": r[0], "execution_context": json.loads(r[1] or "{}"), "created_at": r[2]} for r in rows]
        except Exception as e:
            logger.debug("Failed to fetch local patch failures: %s", e)
            return []
        finally:
            conn.close()
    try:
        resp = (
            client.table("agent_error_contexts")
            .select("error_message,execution_context,created_at")
            .eq("error_type", "PatchApplyFailure")
            .filter("execution_context->>issue_number", "eq", str(issue_number))
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logger.debug("Failed to fetch prior patch-failure context: %s", e)
        return []


def end_history_session(
    session_id: str,
    status: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Close a session. status is typically "succeeded" or "failed"."""
    _ensure_worker()
    fields: Dict[str, Any] = {
        "status": status,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata is not None:
        fields["metadata"] = metadata
    _queue.put(("session_end", {"id": session_id, "fields": fields}))


def stop_history_worker() -> None:
    """Stop the background worker thread.

    Waits for everything already queued to be processed first (via
    queue.join(), matched by _worker_loop calling task_done() on every item
    it dequeues) so a stop doesn't strand pending history unwritten. For
    tests / graceful shutdown.
    """
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        _queue.join()
    _stop_event.set()
    if _worker_thread:
        _worker_thread.join(timeout=_BATCH_MAX_WAIT_S + 1.0)
        _worker_thread = None

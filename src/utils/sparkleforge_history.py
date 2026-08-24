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

logger = logging.getLogger(__name__)

_queue: "queue.Queue" = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_stop_event = threading.Event()

_BATCH_MAX_SIZE = 25
_BATCH_MAX_WAIT_S = 2.0


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
                _queue.get(timeout=1.0)
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
    """Stop the background worker thread. For tests / graceful shutdown."""
    global _worker_thread
    _stop_event.set()
    if _worker_thread:
        _worker_thread.join(timeout=1.0)
        _worker_thread = None

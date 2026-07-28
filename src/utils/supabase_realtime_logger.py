"""Supabase Real-time Logging Utility for SparkleForge.

Streams agent execution events and logs to Supabase Realtime channels.
Runs safely and non-blockingly so it doesn't introduce latency to the main execution flow.
Includes both global functions (legacy/compatibility) and per-session logger classes.
"""

import logging
import threading
import queue
import sys
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Any, Dict, Optional

from src.utils.supabase_exporter import get_supabase_client

logger = logging.getLogger(__name__)

# Global queue for non-blocking log transmission
_log_queue: queue.Queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()

# Batch the DB insert side (not the realtime broadcast, which stays per-event
# for low latency) to cut Supabase API traffic on chatty continuous-mode runs.
_BATCH_MAX_SIZE = 25
_BATCH_MAX_WAIT_S = 2.0
# Lets external monitoring tell "no logs yet" apart from "worker is dead".

# Heartbeat payload constants (extracted for maintainability).
_HEARTBEAT_SESSION_ID = "system"
_HEARTBEAT_AGENT_NAME = "supabase_logger_worker"
_HEARTBEAT_MESSAGE = "heartbeat"
_HEARTBEAT_LEVEL = "heartbeat"
_HEARTBEAT_INTERVAL_S = 30.0


class SupabaseRealtimeHandler(logging.Handler):
    """Logging handler that queues logs for real-time broadcast via Supabase."""

    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id

    def emit(self, record):
        try:
            log_message = self.format(record)
            agent = "system"
            
            # Determine agent name based on record name or message tags
            if hasattr(record, "name") and record.name:
                name_lower = record.name.lower()
                if "planner" in name_lower:
                    agent = "planner"
                elif "executor" in name_lower:
                    agent = "executor"
                elif "verifier" in name_lower:
                    agent = "verifier"
                elif "generator" in name_lower:
                    agent = "generator"

            if "[PLANNER]" in log_message or "[planner]" in log_message:
                agent = "planner"
            elif "[EXECUTOR]" in log_message or "[executor]" in log_message:
                agent = "executor"
            elif "[VERIFIER]" in log_message or "[verifier]" in log_message:
                agent = "verifier"
            elif "[GENERATOR]" in log_message or "[generator]" in log_message:
                agent = "generator"

            # Enqueue the log event
            enqueue_log_event(
                session_id=self.session_id,
                agent_name=agent,
                message=log_message,
                level=record.levelname.lower()
            )
        except Exception:
            pass  # Fail silently to avoid breaking the application logging


def start_supabase_logger_worker():
    """Start the background worker thread that sends queued logs to Supabase."""
    global _worker_thread, _stop_event
    
    if _worker_thread and _worker_thread.is_alive():
        return

    _stop_event.clear()
    _worker_thread = threading.Thread(target=_supabase_logger_worker_loop, daemon=True)
    _worker_thread.start()
    logger.debug("Supabase realtime logger worker thread started.")


def stop_supabase_logger_worker():
    """Stop the background worker thread."""
    global _worker_thread, _stop_event
    if _worker_thread:
        _stop_event.set()
        _worker_thread.join(timeout=1.0)
        _worker_thread = None
        logger.debug("Supabase realtime logger worker thread stopped.")


def enqueue_log_event(session_id: str, agent_name: str, message: str, level: str = "info"):
    """Queue a log event to be broadcast to Supabase."""
    start_supabase_logger_worker()
    
    event = {
        "session_id": session_id,
        "agent_name": agent_name,
        "message": message,
        "level": level,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    _log_queue.put(event)


def _supabase_logger_worker_loop():
    """Worker loop that fetches items from the queue and sends them to Supabase."""
    client = get_supabase_client()
    if not client:
        # If client not configured, drain the queue and exit
        while not _stop_event.is_set():
            try:
                _log_queue.get_nowait()
                _log_queue.task_done()
            except queue.Empty:
                time.sleep(1.0)
        return

    try:
        channel = client.channel("agent_telemetry")
        channel.subscribe()
    except Exception as e:
        logger.debug(f"Failed to subscribe to realtime channel: {e}")
        channel = None

    last_heartbeat = time.monotonic()
    next_heartbeat = last_heartbeat + _HEARTBEAT_INTERVAL_S

    while not _stop_event.is_set():
        batch = []
        deadline = time.monotonic() + _BATCH_MAX_WAIT_S

        while len(batch) < _BATCH_MAX_SIZE:
            now = time.monotonic()
            remaining_batch = deadline - now
            remaining_heartbeat = next_heartbeat - now
            wait_timeout = min(_BATCH_MAX_WAIT_S, max(0.0, remaining_heartbeat), max(0.0, remaining_batch))
            if wait_timeout <= 0:
                break
            try:
                event = _log_queue.get(timeout=wait_timeout)
            except queue.Empty:
                break

            if channel:
                try:
                    channel.send_broadcast("agent_log", event)
                except Exception as e:
                    logger.debug(f"Error broadcasting log to Supabase realtime channel: {e}")
            batch.append(event)
            _log_queue.task_done()

        if batch:
            try:
                client.table("agent_logs").insert(batch).execute()
            except Exception as e:
                logger.debug(f"Error sending log batch to Supabase: {e}")

        if time.monotonic() >= next_heartbeat:
            last_heartbeat = time.monotonic()
            next_heartbeat = last_heartbeat + _HEARTBEAT_INTERVAL_S
            try:
                client.table("agent_logs").insert(
                    {
                        "session_id": _HEARTBEAT_SESSION_ID,
                        "agent_name": _HEARTBEAT_AGENT_NAME,
                        "message": _HEARTBEAT_MESSAGE,
                        "level": _HEARTBEAT_LEVEL,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ).execute()
            except Exception as e:
                logger.debug(f"Error sending heartbeat to Supabase: {e}")


class SupabaseStdoutRedirector:
    """Redirects stdout to both original stdout and Supabase realtime queue."""
    def __init__(self, session_id: str, original_stream):
        self.session_id = session_id
        self.original_stream = original_stream
        self.buffer = ""

    def write(self, text: str):
        self.original_stream.write(text)
        self.original_stream.flush()

        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.replace("\r", "")
            if line.strip():
                enqueue_log_event(self.session_id, "terminal", line, "info")

    def flush(self):
        self.original_stream.flush()


@contextmanager
def redirect_stdout_to_supabase(session_id: str):
    """Context manager to temporarily redirect sys.stdout to Supabase Realtime."""
    original_stdout = sys.stdout
    redirector = SupabaseStdoutRedirector(session_id, original_stdout)
    sys.stdout = redirector
    try:
        yield
    finally:
        sys.stdout = original_stdout


# --- Per-session Logger classes from main (PR 609) ---

class SupabaseRealtimeLogger:
    """Per-session logger instance to isolate concurrent logging streams."""

    def __init__(self, session_id: str, supabase_client: Optional[Any] = None):
        self.session_id = session_id
        self.client = supabase_client
        self.queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        """Background worker to drain this session's queue and send to Supabase."""
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                log_entry = self.queue.get(timeout=1.0)
                log_entry["session_id"] = self.session_id
                if self.client is not None:
                    try:
                        self.client.table("logs").insert(log_entry).execute()
                    except Exception:
                        pass
                self.queue.task_done()
            except queue.Empty:
                continue

    def log(self, message: str, level: str = "INFO"):
        """Enqueue a log entry for this session."""
        self.queue.put({"message": message, "level": level})

    def stop(self, timeout: float = 5.0):
        """Signal the worker to stop and drain remaining items before joining."""
        self.stop_event.set()
        deadline = timeout
        while not self.queue.empty() and deadline > 0:
            try:
                self.queue.get(timeout=0.1)
                self.queue.task_done()
            except queue.Empty:
                break
            deadline -= 0.1
        self.thread.join(timeout=timeout)


_stdout_lock = threading.Lock()
_global_original_stdout: Optional[Any] = None
_active_redirect_count = 0


class SessionStdoutRedirector:
    """Context manager that routes stdout to a specific session logger."""

    def __init__(self, logger: SupabaseRealtimeLogger):
        self.logger = logger
        self._original_stdout: Optional[Any] = None

    def __enter__(self):
        global _global_original_stdout, _active_redirect_count
        with _stdout_lock:
            if _active_redirect_count == 0:
                _global_original_stdout = sys.stdout
            _active_redirect_count += 1
            self._original_stdout = _global_original_stdout
            sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _global_original_stdout, _active_redirect_count
        with _stdout_lock:
            _active_redirect_count -= 1
            if _active_redirect_count == 0:
                sys.stdout = _global_original_stdout
                _global_original_stdout = None
            elif self._original_stdout is not None:
                sys.stdout = self._original_stdout
        return False

    def write(self, text):
        if text.strip():
            self.logger.log(text.strip(), level="STDOUT")
        if self._original_stdout is not None:
            self._original_stdout.write(text)

    def flush(self):
        if self._original_stdout is not None:
            self._original_stdout.flush()


class SupabaseLoggingHandler(logging.Handler):
    """Logging handler that routes records to a session logger."""

    def __init__(self, logger: SupabaseRealtimeLogger):
        super().__init__()
        self.logger = logger

    def emit(self, record):
        try:
            self.logger.log(self.format(record), level=record.levelname)
        except Exception:
            self.handleError(record)


def test_concurrent_logging_isolation():
    """Regression test: stopping one session must not affect another."""
    client = None  # Mock
    logger_a = SupabaseRealtimeLogger("A", client)
    logger_b = SupabaseRealtimeLogger("B", client)

    logger_a.log("Msg A")
    logger_b.log("Msg B")

    assert logger_a.session_id == "A"
    assert logger_b.session_id == "B"

    logger_a.stop()
    logger_b.log("Msg B2")
    logger_b.stop()

    assert logger_a.queue.empty()
    assert logger_b.queue.empty()
    print("Isolation test passed")

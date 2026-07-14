"""Supabase Real-time Logging Utility for SparkleForge.

Streams agent execution events and logs to Supabase Realtime channels.
Runs safely and non-blockingly so it doesn't introduce latency to the main execution flow.
"""

import logging
import threading
import queue
import time
from datetime import datetime
from typing import Any, Dict, Optional

from src.utils.supabase_exporter import get_supabase_client

logger = logging.getLogger(__name__)

# Global queue for non-blocking log transmission
_log_queue: queue.Queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


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
    # Start the worker thread if it is not running
    start_supabase_logger_worker()
    
    event = {
        "session_id": session_id,
        "agent_name": agent_name,
        "message": message,
        "level": level,
        "timestamp": datetime.utcnow().isoformat() + "Z"
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

    # Set up realtime channel for logging
    # In Supabase-py, we broadcast to a channel. We'll use 'agent_telemetry' as the channel name.
    try:
        channel = client.channel("agent_telemetry")
        channel.subscribe()
    except Exception as e:
        logger.debug(f"Failed to subscribe to realtime channel: {e}")
        channel = None

    while not _stop_event.is_set():
        try:
            # Block for a short time to avoid busy looping
            event = _log_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            # Send broadcast via Supabase Realtime
            if channel:
                channel.send_broadcast("agent_log", event)
            
            # Also optionally mirror to an agent_logs table for persistent history
            # (only if database connectivity works)
            try:
                # We do this asynchronously/non-blocking via the background thread
                client.table("agent_logs").insert(event).execute()
            except Exception:
                pass # Fail silently if table doesn't exist or insert fails
                
        except Exception as e:
            logger.debug(f"Error sending log to Supabase: {e}")
        finally:
            _log_queue.task_done()


import sys
from contextlib import contextmanager

class SupabaseStdoutRedirector:
    """Redirects stdout to both original stdout and Supabase realtime queue."""
    def __init__(self, session_id: str, original_stream):
        self.session_id = session_id
        self.original_stream = original_stream
        self.buffer = ""

    def write(self, text: str):
        # Write to original stdout
        self.original_stream.write(text)
        self.original_stream.flush()

        # Send lines to Supabase
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

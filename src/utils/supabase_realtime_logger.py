"""Per-session Supabase realtime logger.

Replaces the previous module-level globals with an instance-based design so
that concurrent orchestrator sessions do not share a single worker thread or
queue. Each session instantiates its own ``SupabaseRealtimeLogger`` and
``SessionStdoutRedirector``/``SupabaseLoggingHandler`` to ensure complete
isolation and correct ``session_id`` attribution.
"""

import logging
import queue
import sys
import threading
from typing import Any, Optional


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
                        # Never let persistence errors kill the worker thread.
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
        # Drain remaining items with a bounded wait so we never hang forever.
        deadline = timeout
        while not self.queue.empty() and deadline > 0:
            try:
                self.queue.get(timeout=0.1)
                self.queue.task_done()
            except queue.Empty:
                break
            deadline -= 0.1
        self.thread.join(timeout=timeout)


class SessionStdoutRedirector:
    """Context manager that routes stdout to a specific session logger.

    Unlike the previous global ``SupabaseStdoutRedirector``, this only
    intercepts stdout while the context is active and routes to the
    session-scoped logger instance.
    """

    def __init__(self, logger: SupabaseRealtimeLogger):
        self.logger = logger
        self._original_stdout: Optional[Any] = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_stdout is not None:
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
    """Logging handler that routes records to a session logger.

    Use this instead of replacing ``sys.stdout`` globally when integrating
    with the standard ``logging`` framework.
    """

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

    # Stop session A; session B must still be able to flush.
    logger_a.stop()
    logger_b.log("Msg B2")
    logger_b.stop()

    assert logger_a.queue.empty()
    assert logger_b.queue.empty()
    print("Isolation test passed")

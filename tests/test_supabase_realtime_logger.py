"""Regression tests for per-session Supabase realtime logger isolation."""

import sys
import threading

from src.utils.supabase_realtime_logger import (
    SupabaseLoggingHandler,
    SessionStdoutRedirector,
    SupabaseRealtimeLogger,
)


def test_logger_isolation_between_sessions():
    logger_a = SupabaseRealtimeLogger("session-a", None)
    logger_b = SupabaseRealtimeLogger("session-b", None)

    logger_a.log("message A")
    logger_b.log("message B")

    assert logger_a.session_id == "session-a"
    assert logger_b.session_id == "session-b"

    # Stopping session A must not terminate session B's worker.
    logger_a.stop()

    logger_b.log("message B2")
    logger_b.stop()

    assert logger_a.queue.empty()
    assert logger_b.queue.empty()


def test_concurrent_sessions_do_not_lose_logs():
    logger = SupabaseRealtimeLogger("concurrent", None)
    errors = []

    def producer():
        try:
            for i in range(50):
                logger.log(f"event-{i}")
        except Exception as exc:  # pragma: no cover - regression guard
            errors.append(exc)

    threads = [threading.Thread(target=producer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.stop()
    assert not errors
    assert logger.queue.empty()


def test_session_stdout_redirector_routes_to_session_logger():
    logger = SupabaseRealtimeLogger("redirect-session", None)
    with SessionStdoutRedirector(logger):
        print("captured line")
    logger.stop()
    assert logger.queue.empty()


def test_concurrent_orchestrator_sessions_attribute_logs_correctly():
    """Two concurrent sessions must not clobber each other's stdout or worker.

    Regression for issue #634: stopping one session must not kill the other
    session's worker, and stdout redirection must be scoped per session.
    """
    logger_a = SupabaseRealtimeLogger("orch-a", None)
    logger_b = SupabaseRealtimeLogger("orch-b", None)

    captured = {"a": [], "b": []}

    original_stdout = sys.stdout

    def run_session(logger, key):
        with SessionStdoutRedirector(logger):
            print(f"{key}-line-1")
            print(f"{key}-line-2")
        # After exiting the context, global stdout must be restored to the
        # original stream, not to the other session's redirector.
        assert sys.stdout is original_stdout, (
            f"stdout was not restored for session {key}"
        )
        captured[key].append("done")

    thread_a = threading.Thread(target=run_session, args=(logger_a, "a"))
    thread_b = threading.Thread(target=run_session, args=(logger_b, "b"))

    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    # Stopping session A must not affect session B's worker.
    logger_a.stop()
    logger_b.log("after-a-stop")
    logger_b.stop()

    assert sys.stdout is original_stdout
    assert logger_a.queue.empty()
    assert logger_b.queue.empty()
    assert captured == {"a": ["done"], "b": ["done"]}


def test_supabase_logging_handler_routes_to_session_logger():
    """SupabaseLoggingHandler must attribute records to its session logger."""
    import logging

    logger = SupabaseRealtimeLogger("handler-session", None)
    handler = SupabaseLoggingHandler(logger)

    std_logger = logging.getLogger("test-supabase-handler")
    std_logger.handlers.clear()
    std_logger.addHandler(handler)
    std_logger.setLevel(logging.INFO)
    std_logger.propagate = False

    std_logger.info("structured record")

    logger.stop()
    assert logger.queue.empty()


def test_stop_is_idempotent():
    """Calling stop() multiple times must not raise or hang."""
    logger = SupabaseRealtimeLogger("idempotent-session", None)
    logger.log("one")
    logger.stop()
    logger.stop()  # second stop must be a no-op
    assert logger.queue.empty()

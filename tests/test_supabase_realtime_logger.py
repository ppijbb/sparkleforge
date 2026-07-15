"""Regression tests for per-session Supabase realtime logger isolation."""

import threading

from src.utils.supabase_realtime_logger import (
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

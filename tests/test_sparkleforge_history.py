"""Regression tests for the SparkleForge-wide history persistence module."""

import queue
import threading
import time

import src.utils.sparkleforge_history as history


class _FakeTable:
    def __init__(self, recorder, name):
        self.recorder = recorder
        self.name = name
        self._op = None
        self._payload = None
        self._filters = {}

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def execute(self):
        self.recorder.append((self.name, self._op, self._payload, dict(self._filters)))
        return self


class _FakeClient:
    def __init__(self):
        self.calls = []

    def table(self, name):
        return _FakeTable(self.calls, name)


def _run_loop_until_idle(monkeypatch, fake_client, prefill_items, idle_wait=0.4):
    """Reset module globals, prefill the queue, run the loop briefly, then stop it."""
    history._queue = queue.Queue()
    history._stop_event = threading.Event()
    for item in prefill_items:
        history._queue.put(item)

    monkeypatch.setattr(history, "get_supabase_client", lambda: fake_client)

    thread = threading.Thread(target=history._worker_loop, daemon=True)
    thread.start()
    time.sleep(idle_wait)
    history._stop_event.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_start_history_session_returns_id_even_without_supabase_configured(monkeypatch):
    monkeypatch.setattr(history, "get_supabase_client", lambda: None)
    history._worker_thread = None
    history._stop_event = threading.Event()

    session_id = history.start_history_session("autofix", external_ref="1548")

    assert session_id
    history.log_history_event(session_id, "log", "hello")
    history.end_history_session(session_id, "succeeded")
    history.stop_history_worker()


def test_session_start_event_and_end_are_written_to_expected_tables(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(history, "_BATCH_MAX_WAIT_S", 0.1)

    items = [
        (
            "session_start",
            {
                "id": "s1",
                "source": "autofix",
                "external_ref": "1548",
                "title": None,
                "status": "running",
                "metadata": {},
            },
        ),
        (
            "event",
            {
                "session_id": "s1",
                "event_type": "llm_call",
                "role": None,
                "backend": "nvidia:some-model",
                "level": None,
                "content": "patch applied",
                "metadata": {},
            },
        ),
        (
            "session_end",
            {"id": "s1", "fields": {"status": "succeeded", "ended_at": "2026-08-24T00:00:00Z"}},
        ),
    ]

    _run_loop_until_idle(monkeypatch, client, items, idle_wait=0.4)

    tables_hit = {name for name, _op, _payload, _filters in client.calls}
    assert tables_hit == {"sparkleforge_sessions", "sparkleforge_history_events"}

    session_calls = [c for c in client.calls if c[0] == "sparkleforge_sessions"]
    assert any(op == "insert" and payload["id"] == "s1" for _n, op, payload, _f in session_calls)
    assert any(
        op == "update" and payload["status"] == "succeeded" and filters == {"id": "s1"}
        for _n, op, payload, filters in session_calls
    )

    event_calls = [c for c in client.calls if c[0] == "sparkleforge_history_events"]
    assert any(
        op == "insert" and payload["backend"] == "nvidia:some-model" for _n, op, payload, _f in event_calls
    )


def test_end_to_end_helpers_enqueue_expected_shapes(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(history, "_BATCH_MAX_WAIT_S", 0.1)
    history._queue = queue.Queue()
    history._stop_event = threading.Event()
    monkeypatch.setattr(history, "get_supabase_client", lambda: client)

    thread = threading.Thread(target=history._worker_loop, daemon=True)
    thread.start()

    session_id = history.start_history_session("repl", title="interactive session")
    history.log_history_event(session_id, "message", "hi", role="user")
    history.end_history_session(session_id, "succeeded")

    history._queue.join()
    time.sleep(0.3)
    history._stop_event.set()
    thread.join(timeout=2.0)

    assert any(
        name == "sparkleforge_sessions" and op == "insert" and payload["source"] == "repl"
        for name, op, payload, _f in client.calls
    )
    assert any(
        name == "sparkleforge_history_events" and op == "insert" and payload["role"] == "user"
        for name, op, payload, _f in client.calls
    )

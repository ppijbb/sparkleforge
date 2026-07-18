"""Issues #666/#667: batch the Supabase log-worker DB inserts and add a heartbeat.

The global _supabase_logger_worker_loop previously issued one Supabase
insert per queued log line. These tests verify multiple queued events are
flushed as a single batched insert, and that a heartbeat row is written
when the worker has been idle for longer than the heartbeat interval.
"""

import queue
import threading
import time

import src.utils.supabase_realtime_logger as sr_logger


class _FakeTable:
    def __init__(self, recorder, name):
        self.recorder = recorder
        self.name = name
        self._payload = None

    def insert(self, payload):
        self._payload = payload
        return self

    def execute(self):
        self.recorder.append((self.name, self._payload))
        return self


class _FakeClient:
    def __init__(self):
        self.inserts = []

    def table(self, name):
        return _FakeTable(self.inserts, name)

    def channel(self, name):
        return self

    def subscribe(self):
        return self

    def send_broadcast(self, event_type, payload):
        pass


def _run_loop_until_idle(monkeypatch, fake_client, prefill_events, idle_wait=0.3):
    """Reset module globals, prefill the queue, run the loop briefly, then stop it."""
    sr_logger._log_queue = queue.Queue()
    sr_logger._stop_event = threading.Event()
    for event in prefill_events:
        sr_logger._log_queue.put(event)

    monkeypatch.setattr(sr_logger, "get_supabase_client", lambda: fake_client)

    thread = threading.Thread(target=sr_logger._supabase_logger_worker_loop, daemon=True)
    thread.start()
    time.sleep(idle_wait)
    sr_logger._stop_event.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_multiple_events_are_flushed_as_a_single_batched_insert(monkeypatch):
    monkeypatch.setattr(sr_logger, "_BATCH_MAX_WAIT_S", 0.2)
    client = _FakeClient()
    events = [
        {"session_id": "s1", "agent_name": "system", "message": f"line-{i}", "level": "info"}
        for i in range(5)
    ]

    _run_loop_until_idle(monkeypatch, client, events, idle_wait=0.5)

    agent_log_inserts = [payload for name, payload in client.inserts if name == "agent_logs"]
    # First flush must contain all 5 queued events in one call, not 5 separate calls.
    assert any(isinstance(p, list) and len(p) == 5 for p in agent_log_inserts)


def test_heartbeat_is_sent_when_worker_is_idle(monkeypatch):
    monkeypatch.setattr(sr_logger, "_HEARTBEAT_INTERVAL_S", 0.1)
    monkeypatch.setattr(sr_logger, "_BATCH_MAX_WAIT_S", 0.05)
    client = _FakeClient()

    _run_loop_until_idle(monkeypatch, client, prefill_events=[], idle_wait=0.4)

    heartbeats = [
        payload
        for name, payload in client.inserts
        if name == "agent_logs" and isinstance(payload, dict) and payload.get("level") == "heartbeat"
    ]
    assert heartbeats, "expected at least one heartbeat insert while the worker was idle"
    assert heartbeats[0]["agent_name"] == "supabase_logger_worker"

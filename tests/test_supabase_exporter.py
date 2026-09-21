"""Regression tests for SupabaseExporter async behavior."""

import asyncio
import time

import src.utils.supabase_exporter as supabase_exporter
from src.utils.supabase_exporter import (
    SupabaseExporter,
    frontier_equivalent_cost_usd,
    update_job_status,
)


class _FakeExecute:
    def execute(self):
        time.sleep(0.05)
        return None


class _FakeTable:
    def insert(self, report):
        return _FakeExecute()


class _FakeClient:
    def table(self, name):
        return _FakeTable()


def test_publish_report_does_not_block_event_loop():
    exporter = SupabaseExporter(client=_FakeClient())

    async def main():
        sleep_done = False

        async def background():
            nonlocal sleep_done
            await asyncio.sleep(0.01)
            sleep_done = True

        publish_task = asyncio.create_task(exporter.publish_report({"id": 1}))
        background_task = asyncio.create_task(background())

        start = time.monotonic()
        await asyncio.wait_for(asyncio.gather(publish_task, background_task), timeout=2.0)
        elapsed = time.monotonic() - start

        assert sleep_done
        assert elapsed < 1.0

    asyncio.run(main())


def test_frontier_equivalent_cost_usd_prices_input_and_output_separately():
    cost = frontier_equivalent_cost_usd(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == 15.00 + 75.00


def test_frontier_equivalent_cost_usd_zero_tokens_is_zero():
    assert frontier_equivalent_cost_usd(0, 0) == 0.0


def test_frontier_equivalent_cost_usd_unknown_model_falls_back_to_default():
    cost = frontier_equivalent_cost_usd(1_000_000, 1_000_000, frontier_model="not-a-real-model")
    assert cost == 15.00 + 75.00


class _FakeUpdateQuery:
    def __init__(self, recorder, data):
        self.recorder = recorder
        self.data = data

    def eq(self, *_a, **_kw):
        return self

    def execute(self):
        self.recorder.append(self.data)

        class _Result:
            data = [self.data]

        return _Result()


class _FakeJobsTable:
    def __init__(self, recorder):
        self.recorder = recorder

    def update(self, data):
        return _FakeUpdateQuery(self.recorder, data)


class _FakeJobsClient:
    def __init__(self):
        self.updates = []

    def table(self, _name):
        return _FakeJobsTable(self.updates)


def test_update_job_status_sets_expires_at_on_completed(monkeypatch):
    client = _FakeJobsClient()
    monkeypatch.setattr(supabase_exporter, "get_supabase_client", lambda: client)

    result = asyncio.run(update_job_status("job1", "completed"))

    assert result is True
    assert "expires_at" in client.updates[0]


def test_update_job_status_sets_expires_at_on_failed(monkeypatch):
    client = _FakeJobsClient()
    monkeypatch.setattr(supabase_exporter, "get_supabase_client", lambda: client)

    asyncio.run(update_job_status("job2", "failed", error_message="boom"))

    assert "expires_at" in client.updates[0]
    assert client.updates[0]["error_message"] == "boom"


def test_update_job_status_leaves_expires_at_unset_for_non_terminal_status(monkeypatch):
    client = _FakeJobsClient()
    monkeypatch.setattr(supabase_exporter, "get_supabase_client", lambda: client)

    asyncio.run(update_job_status("job3", "running"))

    assert "expires_at" not in client.updates[0]

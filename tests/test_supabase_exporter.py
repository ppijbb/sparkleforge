"""Regression tests for SupabaseExporter async behavior."""

import asyncio
import time

from src.utils.supabase_exporter import SupabaseExporter, frontier_equivalent_cost_usd


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

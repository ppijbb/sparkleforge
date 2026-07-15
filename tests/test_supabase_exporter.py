"""Regression tests for SupabaseExporter async behavior."""

import asyncio
import time

from src.utils.supabase_exporter import SupabaseExporter


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

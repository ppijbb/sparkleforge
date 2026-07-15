"""Supabase report exporter.

Exposes an async-friendly API while offloading the blocking supabase-py
client calls to a thread pool so the asyncio event loop is never stalled.
"""

import asyncio
import os


class SupabaseExporter:
    """Publishes reports to Supabase without blocking the event loop."""

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            from supabase import create_client, Client

            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            self._client = create_client(url, key)
        return self._client

    async def publish_report(self, report: dict) -> None:
        await asyncio.to_thread(self._publish_report_sync, report)

    def _publish_report_sync(self, report: dict) -> None:
        client = self._get_client()
        client.table("reports").insert(report).execute()

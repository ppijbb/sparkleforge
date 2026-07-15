"""Supabase exporter for SparkleForge.

Handles publishing completed research results and managing job states in Supabase.
Designed to be completely decoupled: if Supabase credentials are not found,
it gracefully logs a warning/debug message and skips the operation without throwing errors.
"""

import logging
import os
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import supabase client
try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
    logger.debug("Supabase python package not installed or import failed.")


def get_supabase_client() -> Optional["Client"]:
    """Initialize and return the Supabase client if credentials are configured."""
    if not HAS_SUPABASE:
        return None

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        logger.debug("Supabase credentials not configured in environment variables.")
        return None

    try:
        return create_client(url, key)
    except Exception as e:
        logger.warning(f"Failed to create Supabase client: {e}")
        return None


async def publish_report(
    topic: str,
    full_report: str,
    confidence_score: float = 0.0,
    source_count: int = 0,
    sources: Optional[List[Dict[str, Any]]] = None,
    keywords: Optional[List[str]] = None,
    user_id: Optional[str] = None,
    summary: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Publish a completed research report to the Supabase database.

    Returns the inserted record if successful, or None.
    """
    client = get_supabase_client()
    if not client:
        logger.debug("Skipping Supabase report publish (not configured).")
        return None

    sources = sources or []
    keywords = keywords or []
    
    # Generate simple summary from report if not provided
    if not summary and full_report:
        lines = full_report.split("\n")
        summary_lines = []
        for line in lines:
            if line.startswith("#"):
                continue
            if line.strip():
                summary_lines.append(line.strip())
            if len(summary_lines) >= 3:
                break
        summary = " ".join(summary_lines)

    data = {
        "topic": topic,
        "summary": summary or topic,
        "full_report": full_report,
        "confidence_score": float(confidence_score),
        "source_count": int(source_count),
        "sources": sources,
        "keywords": keywords,
    }

    if user_id:
        data["user_id"] = user_id

    try:
        logger.info(f"Publishing report to Supabase: {topic[:30]}...")
        # Offload sync client calls to a thread pool to avoid blocking the event loop
        response = await asyncio.to_thread(
            lambda: client.table("reports").insert(data).execute()
        )
        
        if response.data and len(response.data) > 0:
            logger.info("Successfully published report to Supabase.")
            return response.data[0]
        
        logger.warning("Supabase insertion completed but returned no data.")
        return None
    except Exception as e:
        logger.error(f"Failed to publish report to Supabase: {e}")
        return None


async def create_job(topic: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Create a new pending research job in Supabase."""
    client = get_supabase_client()
    if not client:
        return None

    data = {
        "topic": topic,
        "status": "pending",
    }
    if user_id:
        data["user_id"] = user_id

    try:
        response = await asyncio.to_thread(
            lambda: client.table("forge_jobs").insert(data).execute()
        )
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Failed to create job in Supabase: {e}")
        return None


async def update_job_status(
    job_id: str, status: str, error_message: Optional[str] = None
) -> bool:
    """Update status of a research job in Supabase."""
    client = get_supabase_client()
    if not client:
        return False

    data = {
        "status": status,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    if error_message:
        data["error_message"] = error_message

    try:
        response = await asyncio.to_thread(
            lambda: client.table("forge_jobs").update(data).eq("id", job_id).execute()
        )
        return len(response.data) > 0
    except Exception as e:
        logger.error(f"Failed to update job status in Supabase: {e}")
        return False


class SupabaseExporter:
    """Publishes reports to Supabase without blocking the event loop."""

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            self._client = get_supabase_client()
        return self._client

    async def publish_report(self, report: dict) -> None:
        await asyncio.to_thread(self._publish_report_sync, report)

    def _publish_report_sync(self, report: dict) -> None:
        client = self._get_client()
        if client:
            client.table("reports").insert(report).execute()

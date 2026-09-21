#!/usr/bin/env python3
"""Cleanup expired forge_jobs based on expires_at timestamp."""

import os
import sys
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cleanup_expired_jobs")

try:
    from supabase import create_client
except ImportError:
    create_client = None

def main():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logger.warning("Supabase credentials not configured. Skipping expired jobs cleanup.")
        sys.exit(0)

    if not create_client:
        logger.error("Supabase package not installed.")
        sys.exit(1)

    client = create_client(url, key)
    now_iso = datetime.now(timezone.utc).isoformat()
    
    total_deleted = 0
    while True:
        try:
            response = client.table("forge_jobs").select("id").lt("expires_at", now_iso).limit(1000).execute()
        except Exception as e:
            logger.error(f"Failed to query expired forge_jobs: {e}")
            sys.exit(1)
            
        records = response.data or []
        if not records:
            break
            
        ids = [r["id"] for r in records]
        try:
            del_resp = client.table("forge_jobs").delete().in_("id", ids).execute()
            # Check if there is an error attribute or if response handling indicates failure
            if hasattr(del_resp, "error") and del_resp.error:
                logger.error(f"Failed to delete expired jobs batch: {del_resp.error}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Exception raised while deleting expired jobs batch: {e}")
            sys.exit(1)
            
        count = len(ids)
        total_deleted += count
        logger.info(f"✅ Deleted {count} expired jobs (Total so far: {total_deleted})")
        
    logger.info(f"Finished cleaning up expired forge_jobs. Total deleted: {total_deleted}")

if __name__ == "__main__":
    main()

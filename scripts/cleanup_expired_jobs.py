#!/usr/bin/env python3
"""Delete forge_jobs rows past their expires_at TTL (see issue #1619).

Only rows that reached a terminal status (completed/failed) ever get an
expires_at set -- see update_job_status() in src/utils/supabase_exporter.py.
Pending/running jobs have expires_at = NULL and are never touched.
Deleting a forge_jobs row cascades to its agent_logs rows (ON DELETE CASCADE,
see supabase_schema.sql).
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.supabase_exporter import get_supabase_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete. Without this flag, only reports how many rows would be deleted.",
    )
    args = parser.parse_args()

    client = get_supabase_client()
    if not client:
        print("Supabase not configured; nothing to clean up.", file=sys.stderr)
        return 1

    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        # PostgREST caps a single response at its configured max rows
        # (commonly 1000); a run with more expired rows than that only
        # deletes the first batch. This script is meant to run periodically
        # (cron/scheduled workflow), so leftovers get caught by the next
        # run rather than being silently lost -- not implementing full
        # offset pagination here to keep this a simple batch utility.
        expired = (
            client.table("forge_jobs")
            .select("id,status,expires_at")
            .lt("expires_at", now_iso)
            .execute()
        )
    except Exception as e:
        print(f"Failed to query expired forge_jobs rows: {e}", file=sys.stderr)
        return 1
    rows = expired.data or []
    if not rows:
        print("No expired forge_jobs rows.")
        return 0

    if not args.yes:
        print(
            f"{len(rows)} expired forge_jobs row(s) would be deleted "
            "(dry run; pass --yes to delete)."
        )
        return 0

    ids = [row["id"] for row in rows]
    try:
        client.table("forge_jobs").delete().in_("id", ids).execute()
    except Exception as e:
        print(f"Failed to delete expired forge_jobs rows: {e}", file=sys.stderr)
        return 1
    print(f"Deleted {len(ids)} expired forge_jobs row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

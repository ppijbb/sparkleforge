import json
import asyncio
import pytest

from src.core.mcp_servers.sparkleforge_server import mcp, open_board, board_snapshot, steer_task, cancel_task
from src.sdk import submit_job, get_job_status

@pytest.mark.asyncio
async def test_mcp_app_board_tools():
    if mcp is None:
        pytest.skip("FastMCP not available")

    # Submit a dummy job
    job_id = await submit_job("Test MCP board snapshot job")

    # Test board_snapshot
    snapshot_str = await board_snapshot()
    snapshot = json.loads(snapshot_str)
    assert "workers" in snapshot
    assert "jobs" in snapshot
    job_ids = [j.get("job_id") for j in snapshot["jobs"]]
    assert job_id in job_ids

    # Test steer_task
    steer_res = json.loads(await steer_task(job_id, "approve", "Looks good"))
    assert steer_res["success"] is True
    assert steer_res["job_id"] == job_id

    # Test cancel_task
    cancel_res = json.loads(await cancel_task(job_id))
    assert cancel_res["success"] is True

    # Verify status changed
    status = await get_job_status(job_id)
    assert status.status in ("failed", "completed")

    # Test open_board resource / tool
    board_res = await open_board()
    assert "structuredContent" in board_res
    assert board_res["structuredContent"]["status"] == "ok"
    
    res_html = mcp._resources["ui://sparkleforge/board.html"].fn() if hasattr(mcp, "_resources") else ""
    assert "SparkleForge Swarm Control Board" in res_html

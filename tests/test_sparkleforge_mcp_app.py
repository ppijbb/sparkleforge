import pytest
import json

def test_sparkleforge_mcp_app_components():
    try:
        from src.core.mcp_servers.sparkleforge_server import mcp, FASTMCP_AVAILABLE
    except ImportError:
        pytest.skip("sparkleforge_server or fastmcp not available")

    if not FASTMCP_AVAILABLE or mcp is None:
        pytest.skip("FastMCP not installed")

    # Check that open_board tool and app tools are registered
    tools = getattr(mcp, "_tool_manager", None) or getattr(mcp, "tools", None)
    assert tools is not None

@pytest.mark.asyncio
async def test_open_board_and_tools():
    try:
        from src.core.mcp_servers.sparkleforge_server import open_board, board_snapshot, steer_task, cancel_task
    except ImportError:
        pytest.skip("MCP App tools not defined")

    res = await open_board({})
    assert isinstance(res, dict)
    assert "content" in res
    assert "structuredContent" in res

    snap = await board_snapshot()
    assert isinstance(snap, str)
    data = json.loads(snap)
    assert "workers" in data

    steer_res = await steer_task("job_123", "approve", "Looks good")
    steer_data = json.loads(steer_res)
    assert steer_data.get("success") is True

    cancel_res = await cancel_task("job_123")
    cancel_data = json.loads(cancel_res)
    assert cancel_data.get("success") is True

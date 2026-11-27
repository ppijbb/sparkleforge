import asyncio
import json

import pytest


def test_sparkleforge_mcp_app_components():
    try:
        from src.core.mcp_servers.sparkleforge_server import FASTMCP_AVAILABLE, mcp
    except ImportError:
        pytest.skip("sparkleforge_server or fastmcp not available")

    if not FASTMCP_AVAILABLE or mcp is None:
        pytest.skip("FastMCP not installed")

    components = mcp._local_provider._components
    for key in (
        "resource:ui://sparkleforge/board.html@",
        "tool:open_board@",
        "tool:board_snapshot@",
        "tool:steer_task@",
        "tool:cancel_task@",
    ):
        assert key in components, f"Missing MCP App component: {key}"

    model_visible_names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"open_board", "cancel_task"}.issubset(model_visible_names)
    assert "board_snapshot" not in model_visible_names
    assert "steer_task" not in model_visible_names


@pytest.mark.asyncio
async def test_open_board_and_tools():
    try:
        from src.core.mcp_servers.sparkleforge_server import (
            board_snapshot,
            cancel_task,
            open_board,
            steer_task,
        )
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

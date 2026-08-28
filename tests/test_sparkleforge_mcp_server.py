"""Anvil Phase B-1: SparkleForge exposed as an MCP server via src/sdk.py."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.core.mcp_servers import sparkleforge_server


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_run_task_wraps_sdk_run_success():
    fake_result = {"success": True, "content": "42"}
    with patch("src.sdk.run", new=AsyncMock(return_value=fake_result)):
        raw = await sparkleforge_server.run_task(prompt="what is the answer?")

    assert json.loads(raw) == fake_result


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_run_task_returns_structured_error_instead_of_raising():
    with patch("src.sdk.run", new=AsyncMock(side_effect=RuntimeError("boom"))):
        raw = await sparkleforge_server.run_task(prompt="anything")

    parsed = json.loads(raw)
    assert parsed["success"] is False
    assert "boom" in parsed["error"]


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_run_task_wraps_sdk_run_success())
    asyncio.run(test_run_task_returns_structured_error_instead_of_raising())
    print("ok")

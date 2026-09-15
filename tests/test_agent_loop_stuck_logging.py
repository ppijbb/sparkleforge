import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.agent_loop import AgentLoop, MAX_STUCK_TOOL_REPEATS

@pytest.mark.asyncio
async def test_stuck_loop_logging_format_string(caplog):
    loop = AgentLoop.__new__(AgentLoop)
    loop.orchestrator = AsyncMock()
    loop.mcp_hub = AsyncMock()
    loop.compressor = MagicMock()
    loop.compressor.compress_if_needed_background = AsyncMock(side_effect=lambda h: h)
    loop.compressor.prune_tool_output = lambda x: x
    loop.memory = MagicMock()
    loop.memory.get_context_block.return_value = ""
    loop.mode_controller = None
    loop.method_resolver = None
    loop.overseer = None

    # Mock model response returning the same tool call repeatedly
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "test_tool", "arguments": '{"path": "foo.py"}'}
    }
    mock_result = MagicMock()
    mock_result.content = "I am calling the tool."
    mock_result.metadata = {"tool_calls": [tool_call]}
    loop.orchestrator.execute_with_model = AsyncMock(return_value=mock_result)
    loop.mcp_hub.execute_tool = AsyncMock(return_value={"success": True, "output": "ok"})

    messages = [{"role": "user", "content": "do something"}]

    with caplog.at_level(logging.WARNING):
        res = await loop.run_conversation(messages, max_iterations=MAX_STUCK_TOOL_REPEATS + 2)

    assert res["success"] is False
    assert "Stuck loop detected" in res["error"]
    # Verify the warning was logged successfully without raising a TypeError
    assert any("Stuck loop detected" in record.message for record in caplog.records)

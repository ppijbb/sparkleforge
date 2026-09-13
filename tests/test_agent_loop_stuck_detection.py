import pytest
from unittest.mock import AsyncMock, MagicMock
from src.core.agent_loop import AgentLoop, MAX_STUCK_TOOL_REPEATS
from src.core.llm_manager import ModelResult


@pytest.mark.asyncio
async def test_stuck_loop_logging_and_termination(caplog):
    loop = AgentLoop()
    loop.mcp_hub = AsyncMock()
    loop.mcp_hub.initialize_mcp = AsyncMock()
    loop.compressor = MagicMock()
    loop.compressor.compress_if_needed_background = AsyncMock(side_effect=lambda h: h)
    loop.compressor.prune_tool_output = lambda x: x

    # Return the same tool call repeatedly to trigger stuck loop
    tool_call_payload = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": '{"path": "test.txt"}'
        }
    }
    
    loop.orchestrator = AsyncMock()
    loop.orchestrator.execute_with_model = AsyncMock(
        return_value=ModelResult(
            content="",
            model_used="mock-model",
            metadata={"tool_calls": [tool_call_payload]}
        )
    )
    loop.mcp_hub.execute_tool = AsyncMock(return_value={"success": True, "output": "content"})

    result = await loop.run_conversation([{"role": "user", "content": "read test.txt"}], max_iterations=10)

    assert result["success"] is False
    assert result["error"] == "stuck_loop"
    assert any("Stuck loop detected" in record.message for record in caplog.records)
    assert not any(isinstance(record.exc_info, tuple) and record.exc_info[0] is not None for record in caplog.records)

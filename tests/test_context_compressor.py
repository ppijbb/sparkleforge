"""Issue #1333: ContextCompressor compressed on message *count* (10), not the
advertised token budget -- a normal tool-use loop crosses 10 messages within
3-4 iterations regardless of how little of the real token budget is used,
forcing compression (and the memory loss it causes) almost every iteration.
"""

import pytest

from src.core.context_compressor import ContextCompressor
from src.core.llm_manager import ModelResult


class FakeOrchestrator:
    def __init__(self):
        self.calls = 0

    async def execute_with_model(self, **kwargs):
        self.calls += 1
        return ModelResult("summary", "test-model", 0.0, 0.0, 0.0, {})


def _short_messages(n: int) -> list[dict]:
    return [{"role": "user", "content": "ok"} for _ in range(n)]


@pytest.mark.asyncio
async def test_short_history_under_new_threshold_is_not_compressed():
    orchestrator = FakeOrchestrator()
    compressor = ContextCompressor(orchestrator)

    messages = _short_messages(15)  # over the old limit(10), under the new one
    result = await compressor.compress_if_needed(messages)

    assert result == messages
    assert orchestrator.calls == 0


@pytest.mark.asyncio
async def test_history_past_the_new_threshold_still_compresses():
    orchestrator = FakeOrchestrator()
    compressor = ContextCompressor(orchestrator)

    messages = _short_messages(45)
    result = await compressor.compress_if_needed(messages)

    assert orchestrator.calls == 1
    assert len(result) < len(messages)

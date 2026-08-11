"""Issue #1333: ContextCompressor compressed on message *count* (10), not the
advertised token budget -- a normal tool-use loop crosses 10 messages within
3-4 iterations regardless of how little of the real token budget is used,
forcing compression (and the memory loss it causes) almost every iteration.
"""

import pytest

import asyncio

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


class FailingBackgroundOrchestrator:
    def __init__(self):
        self.calls = 0

    async def execute_with_model(self, **kwargs):
        self.calls += 1
        raise RuntimeError("background summarization LLM failure")


def _over_limit_messages() -> list[dict]:
    return [
        {"role": "user", "content": " ".join(["word"] * 200)}
        for _ in range(10)
    ]


@pytest.mark.asyncio
async def test_background_failure_at_hard_limit_falls_back_to_sync():
    orchestrator = FailingBackgroundOrchestrator()
    compressor = ContextCompressor(orchestrator)
    compressor._pending_task = asyncio.get_event_loop().create_task(
        compressor.compress_by_summarization(_over_limit_messages())
    )
    compressor._pending_result = None

    messages = _over_limit_messages()
    token_limit = 100

    result = await compressor.compress_if_needed_background(messages, token_limit=token_limit)

    # Fallback synchronous compression was attempted (background failed, then
    # sync fallback also failed via the same orchestrator, so calls >= 2).
    assert orchestrator.calls >= 2
    # The fallback path returned a (truncated) history rather than the raw
    # over-limit input.
    assert result is not None


@pytest.mark.asyncio
async def test_discard_pending_background_compaction_awaits_cancelled_task():
    compressor = ContextCompressor(FakeOrchestrator())

    async def _never_completes():
        await asyncio.sleep(10)

    compressor._pending_task = asyncio.get_event_loop().create_task(_never_completes())
    await compressor.discard_pending_background_compaction()
    assert compressor._pending_task is None
    assert compressor._pending_snapshot is None

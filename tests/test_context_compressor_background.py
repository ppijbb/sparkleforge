"""Issue #1335: agent_loop awaited context compaction synchronously, so every
LLM summarization call blocked the loop for its full duration -- one
measured session lost 12% of its heat budget doing nothing but waiting on
it. compress_if_needed_background() must start compaction as a background
task and let the caller keep going, splicing the result in once it's ready,
and only block outright once history is past the hard token ceiling.
"""

import asyncio

import pytest

from src.core.context_compressor import ContextCompressor
from src.core.llm_manager import ModelResult


class GatedOrchestrator:
    """Like FakeOrchestrator, but execute_with_model blocks until released --
    lets tests prove a call returns without waiting for it."""

    def __init__(self):
        self.calls = 0
        self.release = asyncio.Event()

    async def execute_with_model(self, **kwargs):
        self.calls += 1
        await self.release.wait()
        return ModelResult("summary", "test-model", 0.0, 0.0, 0.0, {})


def _short_messages(n: int) -> list[dict]:
    return [{"role": "user", "content": "ok"} for _ in range(n)]


@pytest.mark.asyncio
async def test_under_threshold_is_a_no_op():
    orchestrator = GatedOrchestrator()
    compressor = ContextCompressor(orchestrator)

    messages = _short_messages(15)
    result = await compressor.compress_if_needed_background(messages)

    assert result == messages
    assert orchestrator.calls == 0
    assert compressor._pending_task is None


@pytest.mark.asyncio
async def test_over_threshold_starts_background_task_without_blocking():
    orchestrator = GatedOrchestrator()
    compressor = ContextCompressor(orchestrator)
    messages = _short_messages(45)

    # The orchestrator never releases in this test -- if compress_if_needed_
    # background awaited it directly, this would hang past the timeout.
    result = await asyncio.wait_for(
        compressor.compress_if_needed_background(messages), timeout=1.0
    )

    assert result == messages, "history must be returned unchanged while compaction is in flight"
    assert compressor._pending_task is not None
    assert not compressor._pending_task.done()

    await asyncio.sleep(0)  # let the scheduled background task actually start running
    assert orchestrator.calls == 1

    orchestrator.release.set()
    await compressor._pending_task


@pytest.mark.asyncio
async def test_finished_background_task_is_spliced_in_on_next_call():
    orchestrator = GatedOrchestrator()
    compressor = ContextCompressor(orchestrator)
    messages = _short_messages(45)

    await compressor.compress_if_needed_background(messages)
    orchestrator.release.set()
    await compressor._pending_task  # let the background task actually finish

    # The loop kept appending while compaction ran in the background.
    grown_messages = messages + _short_messages(2)
    result = await compressor.compress_if_needed_background(grown_messages)

    assert compressor._pending_task is None
    assert len(result) < len(grown_messages), "prefix should have been compacted"
    assert result[-2:] == grown_messages[-2:], "tail appended after the snapshot must survive"


@pytest.mark.asyncio
async def test_hard_limit_blocks_until_pending_compaction_finishes():
    orchestrator = GatedOrchestrator()
    compressor = ContextCompressor(orchestrator)
    messages = _short_messages(45)

    async def release_shortly():
        await asyncio.sleep(0.05)
        orchestrator.release.set()

    asyncio.ensure_future(release_shortly())

    # token_limit small enough that these messages are already over it, so
    # the call must wait for compaction instead of returning it untouched.
    result = await asyncio.wait_for(
        compressor.compress_if_needed_background(messages, token_limit=10),
        timeout=1.0,
    )

    assert len(result) < len(messages)
    assert compressor._pending_task is None


@pytest.mark.asyncio
async def test_discard_pending_background_compaction_cancels_and_clears():
    orchestrator = GatedOrchestrator()
    compressor = ContextCompressor(orchestrator)
    messages = _short_messages(45)

    await compressor.compress_if_needed_background(messages)
    assert compressor._pending_task is not None

    compressor.discard_pending_background_compaction()

    assert compressor._pending_task is None
    assert compressor._pending_snapshot_len == 0
    # #1354: the cancelled task must actually be drained so it doesn't leak
    # a "Task exception was never retrieved" warning.
    await asyncio.sleep(0)


class FailingOrchestrator:
    """Always raises -- simulates a background compaction that fails."""

    async def execute_with_model(self, **kwargs):
        raise RuntimeError("summarization backend unavailable")


@pytest.mark.asyncio
async def test_hard_limit_falls_back_to_sync_compression_when_background_task_fails():
    """#1354: if the background task fails, the hard-limit path must not
    hand the caller an uncompacted, over-limit history -- it must fall back
    to a synchronous compression so the result is actually under the limit.
    """
    orchestrator = FailingOrchestrator()
    compressor = ContextCompressor(orchestrator)
    messages = _short_messages(45)

    result = await asyncio.wait_for(
        compressor.compress_if_needed_background(messages, token_limit=10),
        timeout=1.0,
    )

    assert len(result) < len(messages), "must fall back to sync compression, not return raw history"


@pytest.mark.asyncio
async def test_hard_limit_preserves_concurrently_appended_tail():
    """#1350: messages appended to the shared history list while the
    hard-limit path awaits the background task must survive in the tail,
    not be silently dropped.
    """
    orchestrator = GatedOrchestrator()
    compressor = ContextCompressor(orchestrator)
    messages = _short_messages(45)

    async def append_during_wait():
        await asyncio.sleep(0.02)
        messages.append({"role": "user", "content": "SENTINEL_APPENDED_DURING_WAIT"})
        orchestrator.release.set()

    asyncio.ensure_future(append_during_wait())

    result = await asyncio.wait_for(
        compressor.compress_if_needed_background(messages, token_limit=10),
        timeout=1.0,
    )

    assert result[-1]["content"] == "SENTINEL_APPENDED_DURING_WAIT"

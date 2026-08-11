"""Issue #1354: background compaction failure at the hard token limit must fall
back to synchronous compression instead of returning uncompacted over-limit
history that the LLM API would reject.
"""

import asyncio

import pytest

from src.core.context_compressor import ContextCompressor
from src.core.llm_manager import ModelResult


class FailingBackgroundOrchestrator:
    """Orchestrator whose first call raises (simulating a failed background
    summarization) and whose subsequent calls succeed (synchronous fallback)."""

    def __init__(self):
        self.calls = 0

    async def execute_with_model(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("background summarization LLM error")
        return ModelResult("fallback summary", "test-model", 0.0, 0.0, 0.0, {})


def _over_limit_messages(token_limit: int) -> list[dict]:
    """Build a message list whose heuristic token count exceeds token_limit."""
    needed_words = int(token_limit / 1.3) + 10
    big = "word " * needed_words
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": "tail"},
        {"role": "assistant", "content": "tail"},
    ]


def _estimate_tokens(messages: list[dict]) -> float:
    return sum(len(str(m.get("content", "")).split()) * 1.3 for m in messages)


@pytest.mark.asyncio
async def test_background_failure_at_hard_limit_falls_back_to_sync():
    orchestrator = FailingBackgroundOrchestrator()
    compressor = ContextCompressor(orchestrator)

    token_limit = 100
    messages = _over_limit_messages(token_limit)
    assert _estimate_tokens(messages) >= token_limit

    async def _failing_background():
        raise RuntimeError("background task failed")

    compressor._pending_task = asyncio.ensure_future(_failing_background())

    result = await compressor.compress_if_needed_background(messages, token_limit=token_limit)

    assert orchestrator.calls >= 2
    assert _estimate_tokens(result) < _estimate_tokens(messages)


@pytest.mark.asyncio
async def test_background_success_returns_compacted_without_fallback():
    class OkOrchestrator:
        def __init__(self):
            self.calls = 0

        async def execute_with_model(self, **kwargs):
            self.calls += 1
            return ModelResult("summary", "test-model", 0.0, 0.0, 0.0, {})

    orchestrator = OkOrchestrator()
    compressor = ContextCompressor(orchestrator)

    token_limit = 100
    messages = _over_limit_messages(token_limit)

    async def _ok_background():
        return [
            {"role": "system", "content": "system"},
            {"role": "system", "content": "[Previous History Summary]: summary"},
            {"role": "user", "content": "tail"},
            {"role": "assistant", "content": "tail"},
        ]

    compressor._pending_task = asyncio.ensure_future(_ok_background())

    result = await compressor.compress_if_needed_background(messages, token_limit=token_limit)
    assert orchestrator.calls == 0
    assert _estimate_tokens(result) < token_limit


@pytest.mark.asyncio
async def test_discard_pending_background_compaction_awaits_cancelled_task():
    compressor = ContextCompressor(None)

    async def _long_running():
        await asyncio.sleep(10)

    compressor._pending_task = asyncio.ensure_future(_long_running())
    compressor._pending_snapshot = {"sentinel": True}

    await compressor.discard_pending_background_compaction()

    assert compressor._pending_task is None
    assert compressor._pending_snapshot is None
"""Issue #1354: background compaction failure at the hard token limit must fall
back to synchronous compression instead of returning uncompacted over-limit
history that the LLM API would reject.
"""

import asyncio

import pytest

from src.core.context_compressor import ContextCompressor
from src.core.llm_manager import ModelResult


class FailingBackgroundOrchestrator:
    """Orchestrator whose first call raises (simulating a failed background
    summarization) and whose subsequent calls succeed (synchronous fallback)."""

    def __init__(self):
        self.calls = 0

    async def execute_with_model(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("background summarization LLM error")
        return ModelResult("fallback summary", "test-model", 0.0, 0.0, 0.0, {})


def _over_limit_messages(token_limit: int) -> list[dict]:
    """Build a message list whose heuristic token count exceeds token_limit."""
    # Each word contributes ~1.3 tokens; make content large enough to exceed.
    needed_words = int(token_limit / 1.3) + 10
    big = "word " * needed_words
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": "tail"},
        {"role": "assistant", "content": "tail"},
    ]


def _estimate_tokens(messages: list[dict]) -> float:
    return sum(len(str(m.get("content", "")).split()) * 1.3 for m in messages)


@pytest.mark.asyncio
async def test_background_failure_at_hard_limit_falls_back_to_sync(monkeypatch):
    orchestrator = FailingBackgroundOrchestrator()
    compressor = ContextCompressor(orchestrator)

    token_limit = 100
    messages = _over_limit_messages(token_limit)
    assert _estimate_tokens(messages) >= token_limit

    # Seed a pending background task that will fail when awaited.
    async def _failing_background():
        raise RuntimeError("background task failed")

    compressor._pending_task = asyncio.ensure_future(_failing_background())

    result = await compressor.compress_if_needed_background(messages, token_limit=token_limit)

    # Fallback synchronous compression must have run (second orchestrator call).
    assert orchestrator.calls >= 2
    # Result must be below the hard limit (or at least not the raw input).
    assert _estimate_tokens(result) < _estimate_tokens(messages)


@pytest.mark.asyncio
async def test_background_success_returns_compacted_without_fallback():
    class OkOrchestrator:
        def __init__(self):
            self.calls = 0

        async def execute_with_model(self, **kwargs):
            self.calls += 1
            return ModelResult("summary", "test-model", 0.0, 0.0, 0.0, {})

    orchestrator = OkOrchestrator()
    compressor = ContextCompressor(orchestrator)

    token_limit = 100
    messages = _over_limit_messages(token_limit)

    async def _ok_background():
        return [
            {"role": "system", "content": "system"},
            {"role": "system", "content": "[Previous History Summary]: summary"},
            {"role": "user", "content": "tail"},
            {"role": "assistant", "content": "tail"},
        ]

    compressor._pending_task = asyncio.ensure_future(_ok_background())

    result = await compressor.compress_if_needed_background(messages, token_limit=token_limit)
    # No synchronous fallback needed.
    assert orchestrator.calls == 0
    assert _estimate_tokens(result) < token_limit


@pytest.mark.asyncio
async def test_discard_pending_background_compaction_awaits_cancelled_task():
    compressor = ContextCompressor(None)

    async def _long_running():
        await asyncio.sleep(10)

    compressor._pending_task = asyncio.ensure_future(_long_running())
    compressor._pending_snapshot = {"sentinel": True}

    await compressor.discard_pending_background_compaction()

    assert compressor._pending_task is None
    assert compressor._pending_snapshot is None
    assert compressor._pending_task is None

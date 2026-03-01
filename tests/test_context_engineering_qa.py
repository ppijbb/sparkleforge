"""End-to-end QA for Context Engineering (Phase 4).

Production-style tests: compaction integrity, session save/restore,
token savings. No mocks.
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

# Add project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(project_root))

from src.core.context_compaction import CompactionConfig, PruneStrategy
from src.core.session_manager import SessionManager
from src.core.session_storage import SessionStorage


def _make_messages(n: int, with_tool_outputs: int = 0) -> list[dict]:
    """Build a list of message dicts (type + content). Optionally add tool outputs for prune."""
    out = [
        {
            "type": "human" if i % 2 == 0 else "ai",
            "content": f"Message {i} with some content to estimate tokens. " * 5,
        }
        for i in range(n)
    ]
    for i in range(with_tool_outputs):
        out.insert(
            len(out) - 1,
            {
                "role": "tool",
                "content": f"Tool result {i} with content. " * 10,
                "metadata": {"tool_result": True},
            },
        )
    return out


def _estimate_tokens(messages: list[dict]) -> int:
    """Simple token estimate (match manager logic)."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += int(len(content.split()) * 1.3)
    return total


@pytest.mark.asyncio
async def test_compaction_integrity():
    """Compaction preserves structure and reduces message count (prune strategy)."""
    config = CompactionConfig(
        preserve_last_n=10,
        strategy="prune",
        auto_compact=True,
    )
    strategy = PruneStrategy(llm_client=None)

    # Include tool outputs so prune actually removes some messages
    messages = _make_messages(20, with_tool_outputs=15)
    compressed = await strategy.compact(messages, config)

    assert isinstance(compressed, list)
    assert len(compressed) < len(messages)
    for msg in compressed:
        assert isinstance(msg, dict)
        assert "type" in msg or "role" in msg
        assert "content" in msg or any(
            k in msg for k in ("content", "text")
        )


@pytest.mark.asyncio
async def test_compaction_token_savings():
    """Compaction reduces estimated token count (prune removes tool outputs)."""
    config = CompactionConfig(
        preserve_last_n=5,
        strategy="prune",
        auto_compact=True,
    )
    strategy = PruneStrategy(llm_client=None)

    messages = _make_messages(15, with_tool_outputs=12)
    original_tokens = _estimate_tokens(messages)

    compressed = await strategy.compact(messages, config)
    compressed_tokens = _estimate_tokens(compressed)

    assert compressed_tokens < original_tokens
    assert len(compressed) < len(messages)


def test_session_save_restore_integrity():
    """Session save and restore return equivalent essential state."""
    with tempfile.TemporaryDirectory(prefix="sparkleforge_qa_sessions_") as tmp:
        storage = SessionStorage(storage_path=tmp, use_database=False)
        sm = SessionManager(storage=storage)

        session_id = "qa_save_restore_1"
        agent_state = {
            "user_query": "Test query",
            "session_id": session_id,
            "research_plan": "Plan text",
            "research_tasks": [{"task_id": "t1", "query": "q1"}],
            "messages": [{"type": "human", "content": "Hello"}],
        }

        success = sm.save_session(
            session_id=session_id,
            agent_state=agent_state,
            context_engineer=None,
            shared_memory=None,
        )
        assert success is True

        restored = sm.restore_session(
            session_id=session_id,
            context_engineer=None,
            shared_memory=None,
        )
        assert restored is not None
        assert isinstance(restored, dict)
        assert restored.get("user_query") == agent_state["user_query"]
        assert restored.get("session_id") == session_id
        assert restored.get("research_plan") == agent_state["research_plan"]
        assert len(restored.get("research_tasks", [])) == len(
            agent_state["research_tasks"]
        )
        assert len(restored.get("messages", [])) == len(
            agent_state["messages"]
        )


def test_session_restore_missing_returns_none():
    """Restore of non-existent session returns None."""
    with tempfile.TemporaryDirectory(prefix="sparkleforge_qa_sessions_") as tmp:
        storage = SessionStorage(storage_path=tmp, use_database=False)
        sm = SessionManager(storage=storage)
        restored = sm.restore_session(
            session_id="nonexistent_session_xyz",
            context_engineer=None,
            shared_memory=None,
        )
        assert restored is None

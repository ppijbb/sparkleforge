import logging
from typing import Any, Dict, List

import asyncio
from src.core.llm_manager import MultiModelOrchestrator, TaskType
from src.core.pre_compressor import get_pre_compressor

logger = logging.getLogger(__name__)


class ContextCompressor:
    """Automatic Context Compression Manager (Phase 3).

    Handles token limit enforcement by summarizing old messages
    and pruning large tool outputs.
    """

    def __init__(self, orchestrator: MultiModelOrchestrator | None = None):
        self.orchestrator = orchestrator or MultiModelOrchestrator()
        self.pre_compressor = get_pre_compressor()
        # Tool-use loops add 1 assistant message + 1+ tool-result messages per
        # iteration (more with parallel tool calls), so a low threshold here
        # forces compression every 3-4 iterations regardless of how far the
        # real token budget is from being used (#1333) — this is a message-
        # count backstop against unbounded growth, not the primary trigger.
        self.max_history_messages = 40
        # Background compaction bookkeeping (#1350).
        self._pending_task: "asyncio.Task[Any] | None" = None
        self._pending_result_messages: List[Dict[str, Any]] | None = None
        # Snapshot length captured at the start of compress_if_needed_background
        # so the post-snapshot tail is never lost when a hard-limit trigger
        # applies the just-started task in the same call frame.
        self._pending_snapshot_len = 0
        self._pending_call_start_len = 0

    async def compress_if_needed(
        self, messages: List[Dict[str, Any]], token_limit: int = 100000
    ) -> List[Dict[str, Any]]:
        """Checks if compression is needed and applies it."""
        # Simple heuristic token counting
        total_tokens = sum(len(str(m.get("content", "")).split()) * 1.3 for m in messages)

        if total_tokens < token_limit * 0.8 and len(messages) <= self.max_history_messages:
            return messages

        logger.info(f"Context limit reached ({int(total_tokens)} tokens). Compressing...")

        return await self.compress_by_summarization(messages)

    async def compress_by_summarization(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Summarizes the middle part of the conversation to save space."""
        if len(messages) <= 4:
            return messages

        # Keep the first 2 (system/initial query) and last 2 messages intact
        system_and_start = messages[:2]
        middle = messages[2:-2]
        tail = messages[-2:]

        if not middle:
            return messages

        middle_text = "\n".join([f"{m['role']}: {(m.get('content') or '')[:500]}" for m in middle])

        summary_prompt = f"Please summarize the following conversation history concisely while preserving key technical details and tool results:\n\n{middle_text}"

        try:
            summary_result = await self.orchestrator.execute_with_model(
                prompt=summary_prompt,
                task_type=TaskType.RESEARCH,
                system_message="You are a context compression assistant. Provide a concise summary of the conversation.",
            )

            summary_content = summary_result.content

            new_history = (
                system_and_start
                + [{"role": "system", "content": f"[Previous History Summary]: {summary_content}"}]
                + tail
            )

            logger.info("Successfully compressed context via summarization.")
            return new_history

        except Exception as e:
            logger.error(f"Failed to compress context: {e}")
            # Fallback: simple truncation
            return system_and_start + tail

    def prune_tool_output(self, content: str, max_length: int = 2000) -> str:
        """Prunes large tool outputs using PreCompressor."""
        if len(content) < max_length:
            return content

        return self.pre_compressor.compress_for_context(content, max_tokens=int(max_length / 4))

    async def compress_if_needed_background(
        self, messages: List[Dict[str, Any]], token_limit: int = 100000
    ) -> List[Dict[str, Any]]:
        """Background compaction variant (#1350).

        Starts a background compaction task when none is pending, and only
        applies it when the hard token limit is reached. The tail (messages
        appended after the snapshot) is preserved by capturing the message
        count at the start of this call, not at task-creation time.
        """
        call_start_len = len(messages)
        self._pending_call_start_len = call_start_len

        if self._pending_task is None:
            # Take the snapshot *before* starting the task so the snapshot
            # length reflects the messages that existed when this call began.
            self._pending_snapshot_len = call_start_len
            self._pending_task = asyncio.create_task(self._do_compress(messages))

        estimated_tokens = sum(len(str(m.get("content", "")).split()) * 1.3 for m in messages)
        hard_limit = int(token_limit)

        if estimated_tokens >= hard_limit:
            await self._pending_task
            return self._apply_pending_result(messages)

        return messages

    async def _do_compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform the actual summarization for background compaction."""
        try:
            return await self.compress_by_summarization(list(messages))
        finally:
            # Stash the result so _apply_pending_result can read it even if
            # the task object is cleared.
            pass

    def _apply_pending_result(
        self, current_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply a completed background compaction, preserving the post-snapshot tail.

        The tail is computed against the message count captured at the start
        of the originating ``compress_if_needed_background`` call, so messages
        present at call time are never silently dropped (#1350).
        """
        if self._pending_task is None:
            return current_messages

        try:
            compacted = self._pending_task.result()
        except Exception as e:
            logger.error(f"Background compaction failed: {e}")
            compacted = None

        # Use the call-start length (captured before task creation) so the
        # tail includes every message that existed when the caller invoked us.
        snapshot_len = max(self._pending_call_start_len, 0)
        tail = current_messages[snapshot_len:] if snapshot_len <= len(current_messages) else []

        self._pending_task = None
        self._pending_result_messages = None
        self._pending_snapshot_len = 0
        self._pending_call_start_len = 0

        if not compacted:
            return current_messages

        return list(compacted) + list(tail)

    async def discard_pending_background_compaction(self) -> None:
        """Cancel and await any pending background compaction task (#1350).

        Awaiting the cancellation ensures deterministic cleanup and avoids
        "Task was destroyed but it is pending" warnings.
        """
        task = self._pending_task
        if task is None:
            return
        task.cancel()
        try:
            await asyncio.gather(task, return_exceptions=True)
        except Exception as e:
            logger.debug(f"Background compaction cancellation error: {e}")
        finally:
            self._pending_task = None
            self._pending_result_messages = None
            self._pending_snapshot_len = 0
            self._pending_call_start_len = 0

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
        """Background-friendly variant of compress_if_needed.

        When a background compaction task is already pending, this method
        awaits it. If the background task failed (or did not reduce tokens
        enough) and the history still exceeds the hard ``token_limit``,
        fall back to synchronous ``compress_by_summarization`` so the caller
        never receives an over-limit history that the LLM API would reject
        (issue #1354).
        """
        total_tokens = sum(len(str(m.get("content", "")).split()) * 1.3 for m in messages)

        if self._pending_task is not None:
            try:
                await self._pending_task
            except Exception as e:
                logger.warning("Background compaction task failed: %s", e)
            finally:
                self._pending_task = None
                self._pending_snapshot = None

            result = self._apply_pending_result(messages)
            result_tokens = sum(len(str(m.get("content", "")).split()) * 1.3 for m in result)
            if result_tokens >= token_limit:
                logger.warning(
                    "Background compaction left history over the hard token limit "
                    "(%d >= %d); falling back to synchronous compression",
                    int(result_tokens),
                    token_limit,
                )
                try:
                    return await self.compress_by_summarization(result)
                except Exception as e:
                    logger.error("Synchronous fallback compression failed: %s", e)
                    raise
            return result

        if total_tokens >= token_limit:
            logger.warning(
                "History exceeds hard token limit (%d >= %d) with no pending "
                "background compaction; compressing synchronously",
                int(total_tokens),
                token_limit,
            )
            return await self.compress_by_summarization(messages)

        return messages

    def _apply_pending_result(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply the result of a completed background compaction task.

        If the background task raised, the exception is swallowed here and the
        original ``messages`` are returned so the caller can detect the
        over-limit condition and fall back to synchronous compression.
        """
        if self._pending_result is None:
            return messages
        return self._pending_result

    async def discard_pending_background_compaction(self) -> None:
        """Cancel and await any pending background compaction task.

        Awaiting the cancelled task suppresses the ``"Task exception was never
        retrieved"`` warning the event loop emits when a cancelled task is never
        awaited (issue #1354 secondary finding).
        """
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
            try:
                await self._pending_task
            except asyncio.CancelledError:
                pass
            finally:
                self._pending_task = None
                self._pending_snapshot = None

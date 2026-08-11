import logging
import asyncio
from typing import Any, Dict, List

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
        # Background compaction lifecycle state (issue #1354).
        self._pending_task: asyncio.Task | None = None
        self._pending_snapshot: List[Dict[str, Any]] | None = None
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
        """Background-aware variant of ``compress_if_needed`` (issue #1354).

        If a background compaction task is already in flight, await it. When
        the hard token limit is reached and the background task either failed
        (returning uncompacted history) or did not reduce enough, fall back to
        synchronous ``compress_by_summarization`` so the caller never sends
        over-limit context to the LLM API.
        """
        total_tokens = sum(len(str(m.get("content", "")).split()) * 1.3 for m in messages)

        if self._pending_task is not None and not self._pending_task.done():
            try:
                compacted = await self._pending_task
            except Exception as e:
                logger.warning(
                    "Background compaction task failed: %s; attempting synchronous fallback",
                    e,
                )
                compacted = messages
            finally:
                self._pending_task = None
                self._pending_snapshot = None

            if total_tokens >= token_limit:
                compacted_tokens = sum(
                    len(str(m.get("content", "")).split()) * 1.3 for m in compacted
                )
                if compacted_tokens >= token_limit:
                    logger.info(
                        "Background compaction insufficient (%d tokens >= %d); "
                        "falling back to synchronous compression",
                        int(compacted_tokens),
                        token_limit,
                    )
                    try:
                        compacted = await self.compress_by_summarization(messages)
                    except Exception as e:
                        logger.error(
                            "Synchronous fallback compression failed at hard limit: %s", e
                        )
                        raise
            return compacted

        if total_tokens < token_limit * 0.8 and len(messages) <= self.max_history_messages:
            return messages

        if total_tokens >= token_limit:
            logger.info(
                "Hard token limit reached (%d >= %d); compressing synchronously",
                int(total_tokens),
                token_limit,
            )
            return await self.compress_by_summarization(messages)

        # Soft limit: kick off a background compaction task.
        self._pending_snapshot = list(messages)
        self._pending_task = asyncio.create_task(self.compress_by_summarization(messages))
        return messages

    async def discard_pending_background_compaction(self) -> None:
        """Cancel and await any in-flight background compaction task (issue #1354).

        Awaiting the cancelled task suppresses the ``"Task exception was never
        retrieved"`` warning and keeps the lifecycle state consistent.
        """
        if self._pending_task is not None and not self._pending_task.done():
            self._pending_task.cancel()
            try:
                await self._pending_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug("Discarded background compaction task raised: %s", e)
            finally:
                self._pending_task = None
                self._pending_snapshot = None

import asyncio
import logging
from typing import Any, Dict, List, Optional

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

        # Background compaction state (#1335): a summarization LLM call
        # blocking the loop for its full duration wasted up to 12% of a
        # session's heat budget doing nothing but waiting. `_pending_task`
        # holds an in-flight compress_by_summarization() run over a snapshot
        # taken at `_pending_snapshot_len`; the loop keeps appending to the
        # real history list while it runs, and the result is spliced back in
        # (snapshot summary + whatever was appended since) once it's done.
        self._pending_task: Optional[asyncio.Task] = None
        self._pending_snapshot_len: int = 0

    @staticmethod
    def _estimate_tokens(messages: List[Dict[str, Any]]) -> float:
        """Simple heuristic token counting."""
        return sum(len(str(m.get("content", "")).split()) * 1.3 for m in messages)

    async def compress_if_needed(
        self, messages: List[Dict[str, Any]], token_limit: int = 100000
    ) -> List[Dict[str, Any]]:
        """Checks if compression is needed and applies it (blocking)."""
        total_tokens = self._estimate_tokens(messages)

        if total_tokens < token_limit * 0.8 and len(messages) <= self.max_history_messages:
            return messages

        logger.info(f"Context limit reached ({int(total_tokens)} tokens). Compressing...")

        return await self.compress_by_summarization(messages)

    async def compress_if_needed_background(
        self, messages: List[Dict[str, Any]], token_limit: int = 100000
    ) -> List[Dict[str, Any]]:
        """Non-blocking variant of compress_if_needed() (#1335).

        Past the soft threshold (same trigger as compress_if_needed), starts
        summarization as a background task over a snapshot and returns
        `messages` unchanged so the caller's loop keeps making progress
        instead of waiting out the LLM summarization call. The result is
        spliced back in on a later call once the task finishes. Only blocks
        if history has grown past `token_limit` itself (the hard ceiling)
        while a compaction is still in flight -- at that point waiting is
        the only safe option.
        """
        if self._pending_task is not None and self._pending_task.done():
            messages = self._apply_pending_result(messages)

        total_tokens = self._estimate_tokens(messages)
        under_soft_threshold = (
            total_tokens < token_limit * 0.8 and len(messages) <= self.max_history_messages
        )
        if under_soft_threshold:
            return messages

        if self._pending_task is None:
            snapshot = list(messages)
            self._pending_snapshot_len = len(snapshot)
            logger.info(
                f"Context limit reached ({int(total_tokens)} tokens). "
                f"Starting background compaction ({len(snapshot)} messages)..."
            )
            self._pending_task = asyncio.create_task(self.compress_by_summarization(snapshot))

        if total_tokens >= token_limit:
            logger.warning(
                "Hard context limit reached (%.0f >= %d tokens) with compaction still "
                "in flight; blocking until it finishes.",
                total_tokens,
                token_limit,
            )
            await self._pending_task
            messages = self._apply_pending_result(messages)

            # #1354: the background task may have failed (network/LLM error)
            # and _apply_pending_result falls back to the uncompacted
            # history in that case -- sending that straight to the LLM API
            # would hit a hard context_length_exceeded error. If we're still
            # over the hard limit here, force a synchronous compression as
            # a last resort rather than let an over-limit history through.
            if self._estimate_tokens(messages) >= token_limit:
                logger.warning(
                    "Still over hard limit (%d) after background compaction; "
                    "falling back to synchronous compression.",
                    token_limit,
                )
                try:
                    messages = await self.compress_by_summarization(messages)
                except Exception as e:
                    logger.error(f"Synchronous fallback compression failed: {e}")

        return messages

    def _apply_pending_result(self, current_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splice a finished background compaction into the live history.

        `current_messages` may have grown past the snapshot the background
        task summarized (the loop kept appending while it ran) -- that tail
        is preserved verbatim on top of the compacted prefix.
        """
        task, self._pending_task = self._pending_task, None
        # #1361: capture (and immediately clear) the snapshot offset that
        # belongs to *this* task atomically with extracting the task itself.
        # Nothing awaits between these two lines, so a re-entrant call
        # starting a new compaction can't overwrite the offset out from
        # under us before it's used below.
        snapshot_len = self._pending_snapshot_len
        self._pending_snapshot_len = 0
        try:
            compressed = task.result()
        except Exception as e:
            logger.error(f"Background compaction failed, keeping uncompacted history: {e}")
            return current_messages

        new_tail = current_messages[snapshot_len:]
        logger.info(
            "Swapped in background-compacted history (%d -> %d messages)",
            len(current_messages),
            len(compressed) + len(new_tail),
        )
        return compressed + new_tail

    def discard_pending_background_compaction(self) -> None:
        """Drop any in-flight background compaction.

        Needed before a synchronous emergency compression (e.g. a
        CONTEXT_LIMIT retry replacing history outright) so a background
        task's result can't later be spliced in against a snapshot offset
        that no longer matches the (now-replaced) history list.
        """
        if self._pending_task is not None and not self._pending_task.done():
            task = self._pending_task
            task.cancel()

            # #1354: cancelling without ever awaiting the task leaves it in
            # limbo -- the event loop logs "Task exception was never
            # retrieved" and cleanup isn't deterministic. This method is
            # called from synchronous call sites, so it can't await the
            # cancellation itself; schedule a drain task that does instead.
            async def _drain_cancelled_task() -> None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Error while cancelling background compaction")

            asyncio.ensure_future(_drain_cancelled_task())
        self._pending_task = None
        self._pending_snapshot_len = 0

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

import logging
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

import logging
from datetime import datetime
from typing import Any, Dict, List

from src.core.orchestrator.state import ResearchState
from src.core.orchestrator.base_node import BaseNode
from src.core.hierarchical_compression import compress_data

logger = logging.getLogger(__name__)

class CompressionNode(BaseNode):
    """Handler for hierarchical data compression."""

    def __init__(self, context_manager):
        self.context_manager = context_manager

    async def hierarchical_compression(self, state: ResearchState) -> ResearchState:
        """Hierarchical Compression (혁신 2)."""
        logger.info("🗜️ Applying Hierarchical Compression")

        execution_results = state.get("execution_results", [])
        compression_results = []

        if not execution_results:
            state["current_step"] = "continuous_verification"
            return state

        for result in execution_results:
            task_id = result.get("task_id", "unknown")
            result_data = result.get("result")

            if not result_data:
                continue

            try:
                compressed = await compress_data(result_data)
                compression_results.append({
                    "task_id": task_id,
                    "original_size": len(str(result_data)),
                    "compressed_size": len(str(compressed.data)),
                    "compression_ratio": compressed.compression_ratio,
                    "compressed_data": compressed.data,
                    "status": "compressed",
                })
            except Exception as e:
                logger.warning(f"Compression failed for {task_id}: {e}")
                compression_results.append({"task_id": task_id, "compressed_data": result_data, "status": "failed"})

        # Summary stats
        total_orig = sum(c.get("original_size", 0) for c in compression_results)
        total_comp = sum(c.get("compressed_size", 0) for c in compression_results)
        ratio = total_comp / max(total_orig, 1)

        state.update({
            "compression_results": compression_results,
            "compression_metadata": {"overall_compression_ratio": ratio},
            "current_step": "continuous_verification",
        })
        return state

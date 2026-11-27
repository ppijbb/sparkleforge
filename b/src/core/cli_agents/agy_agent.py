"""Google Antigravity (agy) CLI Agent adapter.

Supports Google's Antigravity CLI agent for high-velocity multi-file code synthesis and reasoning.
"""

import json
import logging
import re
from typing import Any, Dict

from .base_cli_agent import BaseCLIAgent, CLIAgentConfig, CLIExecutionResult

logger = logging.getLogger(__name__)


class AgyAgent(BaseCLIAgent):
    """Google Antigravity (agy) CLI Agent."""

    def __init__(self, api_key: str | None = None, command: str = "agy"):
        config = CLIAgentConfig(
            name="agy",
            command=command,
            args=["--format", "json"],
            env={"GEMINI_API_KEY": api_key} if api_key else {},
            timeout=400,
            output_format="json",
        )
        super().__init__(config)

    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute a query using the agy CLI."""
        files = kwargs.get("files", [])
        context = kwargs.get("context", "")

        prompt_parts = [query]
        if context:
            prompt_parts.append(f"[Context]\n{context}")
        if files:
            prompt_parts.append(f"[Files]: {', '.join(files)}")
        full_prompt = "\n\n".join(prompt_parts)

        result = await self._execute_command([self.config.command, full_prompt])
        parsed = self.parse_output(result)

        return {
            "success": result.success and parsed.get("success", True),
            "response": parsed.get("response", ""),
            "confidence": parsed.get("confidence", 0.9),
            "metadata": {
                "agent": "agy",
                "files": files,
                "execution_time": result.execution_time,
            },
            "usage": parsed.get("usage", {}),
        }

    def parse_output(self, result: CLIExecutionResult) -> Dict[str, Any]:
        """Parse agy CLI output."""
        if not result.success:
            return {
                "success": False,
                "error": result.error or "agy execution failed",
                "response": "",
                "confidence": 0.0,
            }

        try:
            if self.config.output_format == "json":
                try:
                    data = json.loads(result.output.strip())
                except json.JSONDecodeError:
                    # Fallback if wrapped in markdown blocks
                    match = re.search(r"

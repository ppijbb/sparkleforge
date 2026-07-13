"""
nl_shell.py — Natural Language Shell: interactive and single-run execution modes.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ShellIntent:
    """Parsed intent from natural language input."""
    raw: str
    intent_type: str         # command | query | config | unknown
    command: Optional[str]   # Resolved shell command if applicable
    confidence: float        # 0.0 - 1.0
    explanation: str         # Why this resolution was chosen
    metadata: Dict[str, Any] = field(default_factory=dict)


# Simple keyword-to-command mapping for offline NL resolution
_NL_PATTERNS: List[tuple[str, str, str]] = [
    # (pattern, resolved_command, intent_type)
    # Surface integration (#570): route session/dashboard NL queries through
    # the same CLI dispatch path instead of a separate implementation.
    (r"session\s+list",            "session list",     "cli"),
    (r"session\s+stats?",          "session stats",    "cli"),
    (r"show\s+tasks?",             "session tasks",    "cli"),
    (r"list\s+files?",           "ls -la",           "command"),
    (r"show\s+disk\s+usage",     "df -h",             "command"),
    (r"show\s+memory",           "free -h",           "command"),
    (r"show\s+processes?",       "ps aux",            "command"),
    (r"show\s+network",          "ip addr",           "command"),
    (r"current\s+directory",     "pwd",               "command"),
    (r"who\s+am\s+i",            "whoami",            "command"),
    (r"show\s+env",              "env",               "command"),
    (r"show\s+cpu",              "lscpu",             "command"),
    (r"uptime",                  "uptime",            "command"),
    (r"date\s+and\s+time",       "date",              "command"),
    (r"show\s+ports?",           "ss -tulnp",         "command"),
    (r"running\s+services?",     "systemctl list-units --type=service --state=running", "command"),
]


class NLShell:
    """
    Natural Language Shell that resolves plain-text instructions to executable commands.

    Modes:
    - interactive: REPL loop (async generator)
    - single_run:  One-shot execution and return

    In production, the LLM agent would handle ambiguous queries.
    Here we use deterministic pattern matching for offline capability.
    """

    def __init__(
        self,
        executor: Optional[Any] = None,  # SandboxExecutor or SecureShellExecutor
        require_approval: bool = True,
    ) -> None:
        self._executor = executor
        self._require_approval = require_approval
        self._history: List[Dict[str, Any]] = []

    def parse_intent(self, text: str) -> ShellIntent:
        """Parse natural language text into a ShellIntent."""
        text_lower = text.lower().strip()

        # CLI passthrough (starts with /): route through REPL dispatch
        if text_lower.startswith("/"):
            cmd = text[1:].strip()
            return ShellIntent(
                raw=text, intent_type="cli", command=cmd,
                confidence=1.0, explanation="CLI passthrough via REPL dispatch",
            )

        # Direct shell command passthrough (starts with $ or !)
        if text_lower.startswith(("$", "!")):
            cmd = text[1:].strip()
            return ShellIntent(
                raw=text, intent_type="command", command=cmd,
                confidence=1.0, explanation="Direct command passthrough",
            )

        # Pattern-based NL resolution
        for pattern, resolved, itype in _NL_PATTERNS:
            if re.search(pattern, text_lower):
                return ShellIntent(
                    raw=text, intent_type=itype, command=resolved,
                    confidence=0.85,
                    explanation=f"Matched pattern '{pattern}' → '{resolved}'",
                )

        # Unknown - cannot resolve offline
        return ShellIntent(
            raw=text, intent_type="unknown", command=None,
            confidence=0.0,
            explanation="No pattern match found. Requires LLM resolution.",
        )

    async def execute_intent(self, intent: ShellIntent) -> Dict[str, Any]:
        """Execute a resolved intent, optionally via sandbox executor."""
        if intent.intent_type == "cli":
            return await self._dispatch_cli(intent)
        if not intent.command:
            return {
                "ok": False,
                "error": f"Cannot resolve: {intent.raw}",
                "explanation": intent.explanation,
            }

        if self._executor is not None:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._executor.execute, intent.command
            )
            output = {
                "ok":          result.ok,
                "stdout":      result.stdout,
                "stderr":      result.stderr,
                "returncode":  result.returncode,
                "duration_ms": result.duration_ms,
                "command":     intent.command,
            }
        else:
            # Simple subprocess fallback
            import subprocess
            try:
                proc = subprocess.run(
                    intent.command, shell=True, capture_output=True, text=True, timeout=15
                )
                output = {
                    "ok":         proc.returncode == 0,
                    "stdout":     proc.stdout,
                    "stderr":     proc.stderr,
                    "returncode": proc.returncode,
                    "command":    intent.command,
                }
            except subprocess.TimeoutExpired:
                output = {"ok": False, "error": "Command timed out", "command": intent.command}

        self._history.append({"input": intent.raw, "resolved": intent.command, **output})
        return output

    async def _dispatch_cli(self, intent: ShellIntent) -> Dict[str, Any]:
        """Route a CLI intent through the REPL command_handlers dispatch path.

        This keeps NL-resolved session/dashboard commands on the same backend
        functions as the structured CLI (Surface integration, #570).
        """
        try:
            from src.cli.repl_cli import REPLCLI

            cli = REPLCLI()
            await cli.handle_command(intent.command or "")
            output = {
                "ok": True,
                "command": intent.command,
                "explanation": intent.explanation,
            }
        except Exception as e:
            logger.error("NLShell CLI dispatch failed: %s", e, exc_info=True)
            output = {
                "ok": False,
                "error": str(e),
                "command": intent.command,
                "explanation": intent.explanation,
            }
        self._history.append({"input": intent.raw, "resolved": intent.command, **output})
        return output

    async def single_run(self, text: str) -> Dict[str, Any]:
        """Execute a single NL command and return results."""
        intent = self.parse_intent(text)
        logger.info("NLShell single_run: '%s' → '%s' (%.0f%%)", text, intent.command, intent.confidence * 100)
        return await self.execute_intent(intent)

    async def interactive(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator for interactive REPL mode.
        Yields result dicts for each input line.
        Exit keywords: 'exit', 'quit', 'bye'
        """
        EXIT_WORDS = {"exit", "quit", "bye", "q"}
        print("NLShell — Natural Language Shell (type 'exit' to quit)")
        while True:
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, input, "nl> "
                )
                if text.strip().lower() in EXIT_WORDS:
                    break
                result = await self.single_run(text)
                yield result
            except (EOFError, KeyboardInterrupt):
                break

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._history)

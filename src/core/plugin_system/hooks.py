"""Lifecycle hooks (hooks.json) and HookRunner.

Hooks run at PreTaskRun, PostTaskRun, PreToolUse, PostToolUse.
Command-type hooks run a script with env SPARKLEFORGE_PLUGIN_ROOT and optional stdin JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HookPhase(str, Enum):
    """Hook lifecycle phases."""

    PreTaskRun = "PreTaskRun"
    PostTaskRun = "PostTaskRun"
    PreToolUse = "PreToolUse"
    PostToolUse = "PostToolUse"


@dataclass
class HookDef:
    """Single hook definition (command or inline)."""

    type: str
    command: Optional[str] = None
    matcher: Optional[str] = None


@dataclass
class HooksConfig:
    """hooks.json content: description + hooks per phase."""

    description: str = ""
    hooks: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def load(cls, hooks_path: Path) -> Optional["HooksConfig"]:
        """Load hooks.json from path."""
        if not hooks_path.is_file():
            return None
        try:
            data = json.loads(hooks_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load hooks.json %s: %s", hooks_path, e)
            return None
        desc = (data.get("description") or "").strip()
        hooks_raw = data.get("hooks") or {}
        return cls(description=desc, hooks=dict(hooks_raw))


class HookRunner:
    """Runs lifecycle hooks from registered plugins."""

    def __init__(self, plugin_roots: Optional[List[Path]] = None) -> None:
        self.plugin_roots = list(plugin_roots or [])
        self._hooks_cache: Dict[str, List[tuple[Path, HooksConfig]]] = {}

    def register_plugin_root(self, root: Path) -> None:
        """Register a plugin root for hook discovery."""
        if root not in self.plugin_roots:
            self.plugin_roots.append(root)
        self._hooks_cache.clear()

    def _get_hooks_for_phase(self, phase: HookPhase) -> List[tuple[Path, HooksConfig, Dict[str, Any]]]:
        """Discover and return (plugin_root, config, hook_entry) for phase."""
        if phase.value not in self._hooks_cache:
            entries: List[tuple[Path, HooksConfig]] = []
            for proot in self.plugin_roots:
                hooks_path = proot / ".sparkleforge-plugin" / "hooks" / "hooks.json"
                if not hooks_path.is_file():
                    hooks_path = proot / "hooks" / "hooks.json"
                cfg = HooksConfig.load(hooks_path)
                if cfg and phase.value in cfg.hooks:
                    entries.append((proot, cfg))
            self._hooks_cache[phase.value] = entries
        result = []
        for proot, cfg in self._hooks_cache.get(phase.value, []):
            for entry in cfg.hooks.get(phase.value, []):
                result.append((proot, cfg, entry))
        return result

    def _expand_command(self, command: str, plugin_root: Path) -> str:
        """Replace ${SPARKLEFORGE_PLUGIN_ROOT} in command."""
        root_str = str(plugin_root.resolve())
        return command.replace("${SPARKLEFORGE_PLUGIN_ROOT}", root_str).replace(
            "${CLAUDE_PLUGIN_ROOT}", root_str
        )

    def _matches_matcher(self, matcher: Optional[str], context: Dict[str, Any]) -> bool:
        """Return True if context matches matcher (e.g. tool name pattern)."""
        if not matcher:
            return True
        tool_name = (context.get("tool_name") or "").strip()
        if not tool_name:
            return True
        parts = [p.strip() for p in matcher.split("|") if p.strip()]
        return any(tool_name == p or (p in tool_name) for p in parts)

    async def run_pre_task_run(
        self, session_id: str, task_description: Optional[str] = None
    ) -> None:
        """Run all PreTaskRun hooks."""
        context = {
            "session_id": session_id,
            "task_description": task_description or "",
            "phase": HookPhase.PreTaskRun.value,
        }
        for proot, _cfg, entry in self._get_hooks_for_phase(HookPhase.PreTaskRun):
            hooks_list = entry.get("hooks") or []
            for h in hooks_list:
                if h.get("type") == "command" and h.get("command"):
                    cmd = self._expand_command(h["command"], proot)
                    await self._run_command_hook(cmd, proot, context)

    async def run_post_task_run(
        self,
        session_id: str,
        task_description: Optional[str] = None,
        success: bool = True,
        result_summary: Optional[str] = None,
    ) -> None:
        """Run all PostTaskRun hooks."""
        context = {
            "session_id": session_id,
            "task_description": task_description or "",
            "success": success,
            "result_summary": result_summary or "",
            "phase": HookPhase.PostTaskRun.value,
        }
        for proot, _cfg, entry in self._get_hooks_for_phase(HookPhase.PostTaskRun):
            hooks_list = entry.get("hooks") or []
            for h in hooks_list:
                if h.get("type") == "command" and h.get("command"):
                    cmd = self._expand_command(h["command"], proot)
                    await self._run_command_hook(cmd, proot, context)

    async def run_pre_tool_use(
        self, tool_name: str, tool_input: Dict[str, Any], session_id: str = ""
    ) -> bool:
        """Run PreToolUse hooks that match tool_name. Return False to block tool execution."""
        context = {
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "phase": HookPhase.PreToolUse.value,
        }
        for proot, _cfg, entry in self._get_hooks_for_phase(HookPhase.PreToolUse):
            matcher = entry.get("matcher")
            if not self._matches_matcher(matcher, context):
                continue
            hooks_list = entry.get("hooks") or []
            for h in hooks_list:
                if h.get("type") == "command" and h.get("command"):
                    cmd = self._expand_command(h["command"], proot)
                    exit_code = await self._run_command_hook(cmd, proot, context, wait=True)
                    if exit_code == 2:
                        return False
        return True

    async def run_post_tool_use(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: Any,
        session_id: str = "",
    ) -> None:
        """Run PostToolUse hooks that match tool_name."""
        context = {
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_result": tool_result,
            "phase": HookPhase.PostToolUse.value,
        }
        for proot, _cfg, entry in self._get_hooks_for_phase(HookPhase.PostToolUse):
            matcher = entry.get("matcher")
            if not self._matches_matcher(matcher, context):
                continue
            hooks_list = entry.get("hooks") or []
            for h in hooks_list:
                if h.get("type") == "command" and h.get("command"):
                    cmd = self._expand_command(h["command"], proot)
                    await self._run_command_hook(cmd, proot, context)

    async def _run_command_hook(
        self,
        command: str,
        plugin_root: Path,
        context: Dict[str, Any],
        wait: bool = True,
    ) -> Optional[int]:
        """Run a shell command hook with env and optional stdin JSON. Returns exit code if wait."""
        env = os.environ.copy()
        env["SPARKLEFORGE_PLUGIN_ROOT"] = str(plugin_root.resolve())
        stdin_json = json.dumps(context, ensure_ascii=False)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(plugin_root),
            )
            stdout_data, stderr_data = await proc.communicate(stdin_json.encode("utf-8"))
            if stderr_data and proc.returncode != 0:
                logger.debug("Hook stderr: %s", stderr_data.decode("utf-8", errors="replace"))
            return proc.returncode if wait else None
        except Exception as e:
            logger.warning("Hook command failed: %s - %s", command, e)
            return 0

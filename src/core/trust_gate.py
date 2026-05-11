"""Startup-time trust policy assembly and tool filtering."""

from __future__ import annotations

import json
import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class TrustLevel(str, Enum):
    """Trust level for the current runtime."""

    FULL = "full"
    PARTIAL = "partial"
    READ_ONLY = "read_only"
    SANDBOXED = "sandboxed"


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class TrustContext:
    """Resolved trust policy for one runtime."""

    level: TrustLevel = TrustLevel.FULL
    deny_names: frozenset[str] = frozenset()
    deny_prefixes: tuple[str, ...] = ()
    allowed_mcp_servers: frozenset[str] | None = None

    @classmethod
    def default(cls) -> TrustContext:
        return cls()

    def allows_tool(self, tool_name: str, mcp_server: str | None = None) -> bool:
        """Return True if the tool is allowed under this trust context."""
        normalized_name = (tool_name or "").strip().lower()
        if not normalized_name:
            return False

        if normalized_name in self.deny_names:
            return False

        if any(normalized_name.startswith(prefix) for prefix in self.deny_prefixes):
            return False

        if self.allowed_mcp_servers is not None and mcp_server:
            return mcp_server.lower() in self.allowed_mcp_servers

        return True


_CURRENT_TRUST_CONTEXT: ContextVar[TrustContext] = ContextVar(
    "sparkleforge_trust_context",
    default=TrustContext.default(),
)


def set_current_trust_context(trust: TrustContext) -> Token:
    """Set the current runtime trust context."""
    return _CURRENT_TRUST_CONTEXT.set(trust)


def get_current_trust_context() -> TrustContext:
    """Return the current runtime trust context."""
    return _CURRENT_TRUST_CONTEXT.get()


def reset_current_trust_context(token: Token) -> None:
    """Restore the previous trust context."""
    _CURRENT_TRUST_CONTEXT.reset(token)


class TrustGate:
    """Assemble a trust context from environment and local project policy."""

    def __init__(
        self,
        project_root: Path | None = None,
        runtime_mode: str = "local",
    ) -> None:
        self.project_root = Path(project_root or Path.cwd())
        self.runtime_mode = runtime_mode

    def _read_policy_file(self) -> dict[str, Any]:
        path = self.project_root / ".sparkleforge-trust"
        if not path.is_file():
            return {}

        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}

        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            parsed: dict[str, Any] = {}
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                parsed[key.strip()] = value.strip()
            return parsed

    def _resolve_level(self, file_config: dict[str, Any]) -> TrustLevel:
        raw_level = (
            os.getenv("SPARKLEFORGE_TRUST_LEVEL")
            or str(file_config.get("level") or "").strip()
            or TrustLevel.FULL.value
        )
        try:
            level = TrustLevel(raw_level.lower())
        except ValueError:
            level = TrustLevel.FULL

        if self.runtime_mode != "local" and level == TrustLevel.FULL:
            return TrustLevel.SANDBOXED
        return level

    async def evaluate(self) -> TrustContext:
        """Evaluate the current runtime trust policy."""
        file_config = self._read_policy_file()
        level = self._resolve_level(file_config)

        deny_names = {
            name.strip().lower() for name in _split_csv(os.getenv("SPARKLEFORGE_DENY_TOOLS"))
        }
        if not deny_names and file_config.get("deny_names"):
            value = file_config["deny_names"]
            if isinstance(value, list):
                deny_names = {str(name).strip().lower() for name in value if str(name).strip()}
            else:
                deny_names = {name.strip().lower() for name in _split_csv(str(value))}

        deny_prefixes = tuple(
            prefix.strip().lower() for prefix in _split_csv(os.getenv("SPARKLEFORGE_DENY_PREFIXES"))
        )
        if not deny_prefixes and file_config.get("deny_prefixes"):
            value = file_config["deny_prefixes"]
            if isinstance(value, list):
                deny_prefixes = tuple(
                    str(prefix).strip().lower() for prefix in value if str(prefix).strip()
                )
            else:
                deny_prefixes = tuple(prefix.strip().lower() for prefix in _split_csv(str(value)))

        allowed_servers_raw = os.getenv("SPARKLEFORGE_ALLOWED_MCP_SERVERS")
        allowed_mcp_servers: frozenset[str] | None
        if allowed_servers_raw:
            allowed_mcp_servers = frozenset(
                server.strip().lower() for server in _split_csv(allowed_servers_raw)
            )
        else:
            value = file_config.get("allowed_mcp_servers")
            if isinstance(value, list):
                allowed_mcp_servers = frozenset(
                    str(server).strip().lower() for server in value if str(server).strip()
                )
            elif value:
                allowed_mcp_servers = frozenset(
                    server.strip().lower() for server in _split_csv(str(value))
                )
            else:
                allowed_mcp_servers = None

        trust = TrustContext(
            level=level,
            deny_names=frozenset(deny_names),
            deny_prefixes=deny_prefixes,
            allowed_mcp_servers=allowed_mcp_servers,
        )
        set_current_trust_context(trust)
        return trust

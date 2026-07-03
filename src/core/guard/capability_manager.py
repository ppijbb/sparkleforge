"""
capability_manager.py — Capability-based permission granting/revoking at tool and agent levels.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    LOW = "low"           # Read-only, no side effects
    MEDIUM = "medium"     # Local writes, reversible
    HIGH = "high"         # Irreversible or system-wide changes
    CRITICAL = "critical" # Destructive / security-sensitive


@dataclass
class Capability:
    name: str
    description: str
    risk_level: RiskLevel
    requires_hitl: bool = False  # Whether human approval is required

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Capability) and self.name == other.name


# Built-in capability registry
BUILTIN_CAPABILITIES: Dict[str, Capability] = {
    "read_file":        Capability("read_file",        "Read files from filesystem",            RiskLevel.LOW),
    "write_file":       Capability("write_file",       "Write files to filesystem",             RiskLevel.MEDIUM),
    "execute_shell":    Capability("execute_shell",    "Execute shell commands",                RiskLevel.HIGH,    requires_hitl=True),
    "install_package":  Capability("install_package",  "Install system packages",               RiskLevel.HIGH,    requires_hitl=True),
    "network_request":  Capability("network_request",  "Make outbound network requests",        RiskLevel.MEDIUM),
    "credential_read":  Capability("credential_read",  "Read stored credentials",               RiskLevel.HIGH),
    "credential_write": Capability("credential_write", "Store or update credentials",           RiskLevel.CRITICAL, requires_hitl=True),
    "system_config":    Capability("system_config",    "Modify system configuration",           RiskLevel.CRITICAL, requires_hitl=True),
    "process_control":  Capability("process_control",  "Start, stop, or kill processes",        RiskLevel.HIGH,    requires_hitl=True),
    "memory_read":      Capability("memory_read",      "Read agent memory and context",         RiskLevel.LOW),
    "memory_write":     Capability("memory_write",     "Write to agent memory and context",     RiskLevel.MEDIUM),
    "iot_read":         Capability("iot_read",         "Read telemetry or status from IoT devices", RiskLevel.LOW),
    "iot_control":      Capability("iot_control",      "Send control commands to physical IoT devices", RiskLevel.HIGH, requires_hitl=True),
}


class CapabilityManager:
    """
    Manages capability grants and revocations for agents and tools.
    Thread-safe singleton with persistent state support.
    """

    _instance: Optional["CapabilityManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, state_path: Optional[str] = None) -> "CapabilityManager":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self, state_path: Optional[str] = None) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._state_path = state_path or os.path.join("data", "capability_grants.json")
        self._agent_grants: Dict[str, Set[str]] = {}   # agent_id -> set of capability names
        self._tool_grants: Dict[str, Set[str]] = {}    # tool_name -> set of capability names
        self._revocations: Dict[str, Set[str]] = {}    # id -> revoked capabilities
        self._lock_data = threading.RLock()
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted grants from disk."""
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r") as f:
                    data = json.load(f)
                    self._agent_grants = {k: set(v) for k, v in data.get("agents", {}).items()}
                    self._tool_grants  = {k: set(v) for k, v in data.get("tools", {}).items()}
                    self._revocations  = {k: set(v) for k, v in data.get("revocations", {}).items()}
                    logger.info("Loaded capability grants from %s", self._state_path)
            except Exception as e:
                logger.warning("Failed to load capability state: %s", e)

    def _save_state(self) -> None:
        """Persist current grants to disk."""
        os.makedirs(os.path.dirname(self._state_path) if os.path.dirname(self._state_path) else ".", exist_ok=True)
        try:
            with open(self._state_path, "w") as f:
                json.dump({
                    "agents": {k: list(v) for k, v in self._agent_grants.items()},
                    "tools":  {k: list(v) for k, v in self._tool_grants.items()},
                    "revocations": {k: list(v) for k, v in self._revocations.items()},
                }, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save capability state: %s", e)

    def grant_agent(self, agent_id: str, capability_name: str) -> bool:
        """Grant a capability to an agent."""
        if capability_name not in BUILTIN_CAPABILITIES:
            logger.error("Unknown capability: %s", capability_name)
            return False
        with self._lock_data:
            self._agent_grants.setdefault(agent_id, set()).add(capability_name)
            self._revocations.get(agent_id, set()).discard(capability_name)
            self._save_state()
        logger.info("Granted capability '%s' to agent '%s'", capability_name, agent_id)
        return True

    def revoke_agent(self, agent_id: str, capability_name: str) -> bool:
        """Revoke a capability from an agent."""
        with self._lock_data:
            self._agent_grants.get(agent_id, set()).discard(capability_name)
            self._revocations.setdefault(agent_id, set()).add(capability_name)
            self._save_state()
        logger.info("Revoked capability '%s' from agent '%s'", capability_name, agent_id)
        return True

    def grant_tool(self, tool_name: str, capability_name: str) -> bool:
        """Grant a capability to a tool."""
        if capability_name not in BUILTIN_CAPABILITIES:
            return False
        with self._lock_data:
            self._tool_grants.setdefault(tool_name, set()).add(capability_name)
            self._save_state()
        return True

    def agent_has(self, agent_id: str, capability_name: str) -> bool:
        """Check whether an agent has a specific capability."""
        with self._lock_data:
            if capability_name in self._revocations.get(agent_id, set()):
                return False
            return capability_name in self._agent_grants.get(agent_id, set())

    def tool_has(self, tool_name: str, capability_name: str) -> bool:
        """Check whether a tool has a specific capability."""
        with self._lock_data:
            return capability_name in self._tool_grants.get(tool_name, set())

    def get_agent_capabilities(self, agent_id: str) -> List[Capability]:
        """Return the list of capabilities granted to an agent."""
        with self._lock_data:
            caps = self._agent_grants.get(agent_id, set()) - self._revocations.get(agent_id, set())
            return [BUILTIN_CAPABILITIES[c] for c in caps if c in BUILTIN_CAPABILITIES]

    def get_capability(self, name: str) -> Optional[Capability]:
        return BUILTIN_CAPABILITIES.get(name)

    def reset(self) -> None:
        """Clear all grants (for testing)."""
        with self._lock_data:
            self._agent_grants.clear()
            self._tool_grants.clear()
            self._revocations.clear()

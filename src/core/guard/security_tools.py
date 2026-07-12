"""Agent-callable GuardPlane/TrustGate tools (Anvil Phase Σ-3, issue #419/#510).

Exposes `quarantine_file` (isolate a suspect file) and `revoke_capability`
(strip a previously granted capability) as tools registered into the shared
`src.core.tools.registry`, so agents that identify a threat via GuardPlane's
scanning can actually act on it instead of only reporting it.

Every call is recorded in `ActionJournal`; `quarantine_file` records a
pre-action snapshot so a quarantine can be rolled back via
`ActionJournal.rollback()` if it turns out to be a false positive.
"""
from __future__ import annotations

import logging
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any, Dict

from src.core.guard.action_journal import ActionJournal
from src.core.guard.capability_manager import CapabilityManager

logger = logging.getLogger(__name__)

DEFAULT_QUARANTINE_DIR = os.path.join("data", "quarantine")


def quarantine_file(
    file_path: str,
    reason: str = "",
    agent_id: str = "agent",
    quarantine_dir: str | None = None,
) -> Dict[str, Any]:
    """Isolate a suspect file so it can no longer be executed or read in place.

    Strips the executable bit and moves the file into `quarantine_dir`
    (default `data/quarantine/`), tagging it with a timestamp so repeated
    quarantines of the same filename don't collide. Returns success=False
    without raising if the path doesn't exist or can't be moved, since
    "target already gone" is a normal outcome for a security tool.
    """
    src = Path(file_path).expanduser().resolve()
    journal = ActionJournal()

    if not src.exists():
        return {"success": False, "error": f"File not found: {file_path}", "quarantined_path": None}

    if not src.is_file():
        return {"success": False, "error": f"Not a regular file: {file_path}", "quarantined_path": None}

    quarantine_root = Path(quarantine_dir or DEFAULT_QUARANTINE_DIR).expanduser().resolve()
    quarantine_root.mkdir(parents=True, exist_ok=True)

    pre_state = {
        "original_path": str(src),
        "mode": src.stat().st_mode,
        "content_b64": None,
    }
    try:
        import base64

        pre_state["content_b64"] = base64.b64encode(src.read_bytes()).decode("ascii")
    except OSError as e:
        logger.warning("Could not snapshot %s before quarantine: %s", src, e)

    entry = journal.record(
        agent_id=agent_id,
        action="quarantine_file",
        description=reason or f"Quarantining suspicious file: {src}",
        risk_level="high",
        metadata={"original_path": str(src)},
        pre_state=pre_state,
    )

    dest = quarantine_root / f"{int(time.time())}_{src.name}"
    try:
        os.chmod(src, src.stat().st_mode & ~stat.S_IEXEC & ~stat.S_IXGRP & ~stat.S_IXOTH)
        shutil.move(str(src), str(dest))
    except OSError as e:
        logger.error("Failed to quarantine %s: %s", src, e)
        journal.update_outcome(entry.entry_id, outcome="failure", error=str(e))
        return {"success": False, "error": str(e), "quarantined_path": None}

    journal.update_outcome(entry.entry_id, outcome="success")
    logger.info("Quarantined %s -> %s", src, dest)
    return {
        "success": True,
        "original_path": str(src),
        "quarantined_path": str(dest),
        "entry_id": entry.entry_id,
    }


def revoke_capability(agent_id: str, capability_name: str, reason: str = "") -> Dict[str, Any]:
    """Revoke a previously granted capability from an agent, journaled for traceability."""
    journal = ActionJournal()
    manager = CapabilityManager()

    revoked = manager.revoke_agent(agent_id, capability_name)
    journal.record(
        agent_id=agent_id,
        action="revoke_capability",
        description=reason or f"Revoking capability '{capability_name}' from '{agent_id}'",
        risk_level="medium",
        metadata={"capability_name": capability_name, "revoked": revoked},
    )
    return {"success": revoked, "agent_id": agent_id, "capability_name": capability_name}


async def _quarantine_file_tool(
    file_path: str,
    reason: str = "",
    agent_id: str = "agent",
) -> Dict[str, Any]:
    return quarantine_file(file_path=file_path, reason=reason, agent_id=agent_id)


async def _revoke_capability_tool(
    agent_id: str,
    capability_name: str,
    reason: str = "",
) -> Dict[str, Any]:
    return revoke_capability(agent_id=agent_id, capability_name=capability_name, reason=reason)


QUARANTINE_FILE_PARAMETERS = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Path to the file to isolate (e.g. a malicious script or executable).",
        },
        "reason": {
            "type": "string",
            "description": "Why this file is being quarantined.",
            "default": "",
        },
        "agent_id": {
            "type": "string",
            "description": "Identifier of the agent taking this action, for the audit journal.",
            "default": "agent",
        },
    },
    "required": ["file_path"],
}

REVOKE_CAPABILITY_PARAMETERS = {
    "type": "object",
    "properties": {
        "agent_id": {
            "type": "string",
            "description": "Agent whose capability should be revoked.",
        },
        "capability_name": {
            "type": "string",
            "description": (
                "Capability to revoke, e.g. 'execute_shell', 'network_request', "
                "'credential_read', 'process_control'."
            ),
        },
        "reason": {
            "type": "string",
            "description": "Why this capability is being revoked.",
            "default": "",
        },
    },
    "required": ["agent_id", "capability_name"],
}


def register_security_tools() -> None:
    """Register `quarantine_file`/`revoke_capability` into the shared tool registry."""
    from src.core.tools.registry import ToolCategory, ToolMetadata, registry

    registry.register(
        ToolMetadata(
            name="quarantine_file",
            description=(
                "Isolate a suspicious or malicious file by stripping its executable bit "
                "and moving it out of the project into a quarantine directory, so it can "
                "no longer run or be read in place. Use this instead of deleting files "
                "you suspect are dangerous, since quarantined files can be rolled back."
            ),
            parameters=QUARANTINE_FILE_PARAMETERS,
            category=ToolCategory.UTILITY,
            tags=["security", "guard", "quarantine", "trustgate"],
            source="local",
        ),
        _quarantine_file_tool,
        _quarantine_file_tool,
    )
    registry.register(
        ToolMetadata(
            name="revoke_capability",
            description=(
                "Revoke a previously granted capability (e.g. execute_shell, "
                "network_request, credential_read) from an agent that is behaving "
                "suspiciously or no longer needs elevated access."
            ),
            parameters=REVOKE_CAPABILITY_PARAMETERS,
            category=ToolCategory.UTILITY,
            tags=["security", "guard", "capability", "trustgate"],
            source="local",
        ),
        _revoke_capability_tool,
        _revoke_capability_tool,
    )

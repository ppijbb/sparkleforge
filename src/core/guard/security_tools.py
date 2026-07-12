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

import asyncio
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
QUARANTINE_BASE = Path(DEFAULT_QUARANTINE_DIR).expanduser().resolve()

# Above this size, don't load the file into memory for the pre-quarantine
# journal snapshot -- record metadata only. 100 MiB.
MAX_SNAPSHOT_BYTES = 100 * 1024 * 1024


def _resolve_quarantine_root(quarantine_dir: str | None) -> Path:
    """Resolve the quarantine destination directory, confined to QUARANTINE_BASE.

    `quarantine_dir`, if given, is treated as a subdirectory name under
    `QUARANTINE_BASE` -- never an arbitrary absolute path -- so a caller can't
    redirect quarantined files into `/etc` or any other system location.
    """
    if not quarantine_dir:
        return QUARANTINE_BASE
    candidate = (QUARANTINE_BASE / quarantine_dir).expanduser().resolve()
    if QUARANTINE_BASE not in candidate.parents:
        raise ValueError(
            f"quarantine_dir must resolve inside {QUARANTINE_BASE}, got {candidate}"
        )
    return candidate


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
    raw = Path(file_path).expanduser()
    journal = ActionJournal()

    if raw.is_symlink():
        return {
            "success": False,
            "error": f"Refusing to quarantine a symlink: {file_path}",
            "quarantined_path": None,
        }

    src = raw.resolve()

    if not src.exists():
        return {"success": False, "error": f"File not found: {file_path}", "quarantined_path": None}

    if not src.is_file():
        return {"success": False, "error": f"Not a regular file: {file_path}", "quarantined_path": None}

    try:
        quarantine_root = _resolve_quarantine_root(quarantine_dir)
    except ValueError as e:
        return {"success": False, "error": str(e), "quarantined_path": None}
    quarantine_root.mkdir(parents=True, exist_ok=True)

    file_size = src.stat().st_size
    pre_state = {
        "original_path": str(src),
        "mode": src.stat().st_mode,
        "size": file_size,
        "content_b64": None,
    }
    if file_size > MAX_SNAPSHOT_BYTES:
        logger.warning(
            "Skipping content snapshot for %s (%d bytes exceeds %d byte limit); "
            "recording metadata only",
            src,
            file_size,
            MAX_SNAPSHOT_BYTES,
        )
    else:
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
    """Revoke a previously granted capability from an agent, journaled for traceability.

    The journal entry is recorded *before* the capability is actually
    mutated, so a journaling failure (disk full, permissions) aborts the
    revocation instead of leaving state mutated with no audit trail.
    """
    journal = ActionJournal()
    manager = CapabilityManager()

    entry = journal.record(
        agent_id=agent_id,
        action="revoke_capability",
        description=reason or f"Revoking capability '{capability_name}' from '{agent_id}'",
        risk_level="medium",
        metadata={"capability_name": capability_name},
    )

    try:
        revoked = manager.revoke_agent(agent_id, capability_name)
    except Exception as e:
        journal.update_outcome(entry.entry_id, outcome="failure", error=str(e))
        raise

    journal.update_outcome(entry.entry_id, outcome="success" if revoked else "failure")
    return {"success": revoked, "agent_id": agent_id, "capability_name": capability_name}


async def _quarantine_file_tool(
    file_path: str,
    reason: str = "",
    agent_id: str = "agent",
) -> Dict[str, Any]:
    return await asyncio.to_thread(
        quarantine_file, file_path=file_path, reason=reason, agent_id=agent_id
    )


async def _revoke_capability_tool(
    agent_id: str,
    capability_name: str,
    reason: str = "",
) -> Dict[str, Any]:
    return await asyncio.to_thread(
        revoke_capability, agent_id=agent_id, capability_name=capability_name, reason=reason
    )


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

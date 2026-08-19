"""
action_journal.py — Immutable action journaling with snapshot-based rollback.
"""
from __future__ import annotations

import json
import logging
import fcntl
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Anchored to the SparkleForge install location, not cwd, so a coworker
# session run against a target repo doesn't leak its audit journal into it
# (#1331).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class JournalEntry:
    entry_id: str
    agent_id: str
    action: str
    description: str
    metadata: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    risk_level: str = "low"
    snapshot_id: Optional[str] = None  # Reference to pre-action snapshot
    outcome: str = "pending"           # pending | success | failure | rolled_back
    error: Optional[str] = None


@dataclass
class Snapshot:
    snapshot_id: str
    entry_id: str
    description: str
    state: Dict[str, Any]              # Captured state before action
    timestamp: float = field(default_factory=time.time)


class ActionJournal:
    """
    Append-only action journal with optional snapshot support for rollback.
    Persists to JSONL on disk.
    """

    _instance: Optional["ActionJournal"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, journal_path: Optional[str] = None, _force_new: bool = False) -> "ActionJournal":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self, journal_path: Optional[str] = None, _force_new: bool = False) -> None:
        if self._initialized and not _force_new:
            return
        if _force_new:
            self._entries = []
            self._snapshots = {}
        self._initialized = True
        self._journal_path = journal_path or str(_PROJECT_ROOT / "data" / "action_journal.jsonl")
        self._snapshots_path = self._journal_path.replace(".jsonl", "_snapshots.json")
        self._entries:   List[JournalEntry] = []
        self._snapshots: Dict[str, Snapshot] = {}
        self._lock_data = threading.RLock()
        self._load()

    def _load(self) -> None:
        """Load existing journal entries from disk."""
        os.makedirs(os.path.dirname(self._journal_path) if os.path.dirname(self._journal_path) else ".", exist_ok=True)
        if os.path.exists(self._journal_path):
            try:
                with open(self._journal_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                self._entries.append(JournalEntry(**data))
                            except json.JSONDecodeError as e:
                                logger.warning("Skipping corrupt journal line: %s", e)
                logger.info("Loaded %d journal entries", len(self._entries))
            except Exception as e:
                logger.warning("Failed to load journal: %s", e)

        if os.path.exists(self._snapshots_path):
            try:
                with open(self._snapshots_path, "r") as f:
                    raw = json.load(f)
                    for k, v in raw.items():
                        self._snapshots[k] = Snapshot(**v)
            except Exception as e:
                logger.warning("Failed to load snapshots: %s", e)

    def _append_entry(self, entry: JournalEntry) -> None:
        """Append entry to JSONL file."""
        try:
            with open(self._journal_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(asdict(entry)) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.error("Failed to append journal entry: %s", e)

    def _save_snapshots(self) -> None:
        """Persist snapshot registry."""
        try:
            with open(self._snapshots_path, "w") as f:
                json.dump({k: asdict(v) for k, v in self._snapshots.items()}, f, indent=2)
        except Exception as e:
            logger.error("Failed to save snapshots: %s", e)

    def record(
        self,
        agent_id: str,
        action: str,
        description: str,
        risk_level: str = "low",
        metadata: Optional[Dict[str, Any]] = None,
        pre_state: Optional[Dict[str, Any]] = None,
    ) -> JournalEntry:
        """
        Record an action. If pre_state is provided, create a snapshot for rollback.
        Returns the JournalEntry (use entry_id to later update outcome).
        """
        snapshot_id: Optional[str] = None
        if pre_state is not None:
            snap = Snapshot(
                snapshot_id=str(uuid.uuid4()),
                entry_id="",  # Will be filled below
                description=f"Pre-action snapshot for: {action}",
                state=pre_state,
            )
            snapshot_id = snap.snapshot_id

        entry = JournalEntry(
            entry_id=str(uuid.uuid4()),
            agent_id=agent_id,
            action=action,
            description=description,
            metadata=metadata or {},
            risk_level=risk_level,
            snapshot_id=snapshot_id,
        )

        if pre_state is not None:
            snap.entry_id = entry.entry_id
            with self._lock_data:
                self._snapshots[snapshot_id] = snap  # type: ignore[index]
            self._save_snapshots()

        with self._lock_data:
            self._entries.append(entry)
        self._append_entry(entry)
        return entry

    def update_outcome(self, entry_id: str, outcome: str, error: Optional[str] = None) -> bool:
        """Mark an entry as success/failure. Rewrites the last matching JSONL line."""
        with self._lock_data:
            for e in reversed(self._entries):
                if e.entry_id == entry_id:
                    e.outcome = outcome
                    e.error   = error
                    self._rewrite_journal()
                    return True
        return False

    def _rewrite_journal(self) -> None:
        """Full rewrite of JSONL file (used after outcome updates)."""
        try:
            with open(self._journal_path, "w") as f:
                for entry in self._entries:
                    f.write(json.dumps(asdict(entry)) + "\n")
        except Exception as e:
            logger.error("Failed to rewrite journal: %s", e)

    def get_snapshot(self, snapshot_id: str) -> Optional[Snapshot]:
        """Retrieve a pre-action snapshot by ID."""
        with self._lock_data:
            return self._snapshots.get(snapshot_id)

    def rollback(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Return the pre-action state for rollback. Callers are responsible
        for actually restoring the state.
        """
        with self._lock_data:
            for e in self._entries:
                if e.entry_id == entry_id and e.snapshot_id:
                    snap = self._snapshots.get(e.snapshot_id)
                    if snap:
                        e.outcome = "rolled_back"
                        self._rewrite_journal()
                        logger.info("Rolled back entry %s", entry_id)
                        return snap.state
        logger.warning("No snapshot found for entry %s", entry_id)
        return None

    def recent(self, limit: int = 20, agent_id: Optional[str] = None) -> List[JournalEntry]:
        """Return recent journal entries, optionally filtered by agent."""
        with self._lock_data:
            entries = list(self._entries)
        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]

    def reset(self) -> None:
        """Clear all entries (for testing)."""
        with self._lock_data:
            self._entries.clear()
            self._snapshots.clear()

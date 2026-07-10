"""Data model and JSON-backed store for Nightshift work items.

Mirrors src/core/scheduler.py's storage convention: a JSON file under
~/.sparkleforge/nightshift/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class NightshiftStatus(str, Enum):
    QUEUED = "queued"
    WRITING_TEST = "writing_test"
    RED = "red"
    IMPLEMENTING = "implementing"
    GREEN = "green"
    DRAFT_OPENED = "draft_opened"
    FAILED = "failed"


@dataclass
class NightshiftItem:
    issue_number: int
    status: NightshiftStatus = NightshiftStatus.QUEUED
    repro_test_files: List[str] = field(default_factory=list)
    branch: Optional[str] = None
    pr_url: Optional[str] = None
    failure_reason: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NightshiftItem":
        data = dict(data)
        if isinstance(data.get("status"), str):
            data["status"] = NightshiftStatus(data["status"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class NightshiftQueue:
    """JSON-backed store of NightshiftItem history, keyed by issue number."""

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path.home() / ".sparkleforge" / "nightshift"
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.queue_file = self.storage_path / "queue.json"
        self.items: Dict[int, NightshiftItem] = {}
        self._load()

    def _load(self) -> None:
        if not self.queue_file.exists():
            return
        try:
            data = json.loads(self.queue_file.read_text(encoding="utf-8"))
            for item_data in data.get("items", []):
                item = NightshiftItem.from_dict(item_data)
                self.items[item.issue_number] = item
        except Exception:
            pass

    def _save(self) -> None:
        data = {
            "items": [item.to_dict() for item in self.items.values()],
            "updated_at": datetime.now().isoformat(),
        }
        self.queue_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def upsert(self, item: NightshiftItem) -> None:
        item.updated_at = datetime.now()
        self.items[item.issue_number] = item
        self._save()

    def get(self, issue_number: int) -> Optional[NightshiftItem]:
        return self.items.get(issue_number)

    def remove(self, issue_number: int) -> bool:
        if self.items.pop(issue_number, None) is not None:
            self._save()
            return True
        return False

    def list(self) -> List[NightshiftItem]:
        return sorted(self.items.values(), key=lambda i: i.updated_at, reverse=True)

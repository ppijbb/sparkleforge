"""Data model and JSON-backed store for Nightwelding work items.

Mirrors src/core/scheduler.py's storage convention: a JSON file under
~/.sparkleforge/nightwelding/. Also tracks Maker's Mark records for
standout-successful auto-fix outcomes (see issue #586).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import time


class NightweldingStatus(str, Enum):
    QUEUED = "queued"
    WRITING_TEST = "writing_test"
    RED = "red"
    IMPLEMENTING = "implementing"
    GREEN = "green"
    DRAFT_OPENED = "draft_opened"
    FAILED = "failed"


@dataclass
class NightweldingItem:
    issue_number: int
    status: NightweldingStatus = NightweldingStatus.QUEUED
    repro_test_files: List[str] = field(default_factory=list)
    branch: str | None = None
    pr_url: str | None = None
    failure_reason: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NightweldingItem:
        data = dict(data)
        if isinstance(data.get("status"), str):
            data["status"] = NightweldingStatus(data["status"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class NightweldingQueue:
    """JSON-backed store of NightweldingItem history, keyed by issue number."""

    def __init__(self, storage_path: Path | None = None):
        if storage_path is None:
            storage_path = Path.home() / ".sparkleforge" / "nightwelding"
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.queue_file = self.storage_path / "queue.json"
        self.items: Dict[int, NightweldingItem] = {}
        self._load()

    def _load(self) -> None:
        if not self.queue_file.exists():
            return
        try:
            data = json.loads(self.queue_file.read_text(encoding="utf-8"))
            for item_data in data.get("items", []):
                item = NightweldingItem.from_dict(item_data)
                self.items[item.issue_number] = item
        except Exception:
            pass

    def _save(self) -> None:
        data = {
            "items": [item.to_dict() for item in self.items.values()],
            "updated_at": datetime.now().isoformat(),
        }
        self.queue_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def upsert(self, item: NightweldingItem) -> None:
        item.updated_at = datetime.now()
        self.items[item.issue_number] = item
        self._save()

    def get(self, issue_number: int) -> NightweldingItem | None:
        return self.items.get(issue_number)

    def remove(self, issue_number: int) -> bool:
        if self.items.pop(issue_number, None) is not None:
            self._save()
            return True
        return False

    def list(self) -> List[NightweldingItem]:
        return sorted(self.items.values(), key=lambda i: i.updated_at, reverse=True)


@dataclass
class MakerMark:
    """A standout-successful nightwelding/auto-fix outcome (issue #586).

    Mirrors the proven pattern of skill_performance.json: accumulate
    per-pattern success signals so future runs can prefer patterns with
    strong mark history instead of starting from a blank slate.
    """

    mark_id: str
    issue_number: int
    issue_type: str = ""
    approach_pattern: str = ""
    iterations: int = 1
    duration_seconds: float = 0.0
    human_intervention_required: bool = False
    first_attempt_success: bool = True
    merged: bool = False
    pr_url: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MakerMark":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class MakerMarkLedger:
    """JSON-backed ledger of Maker's Mark records.

    Stored at ~/.sparkleforge/nightwelding/maker_marks.json. Provides
    eligibility signals for #579 (nightwelding pre-qualification) and
    high-priority candidates for SkillDistiller.
    """

    def __init__(self, storage_path: Path | None = None):
        if storage_path is None:
            storage_path = Path.home() / ".sparkleforge" / "nightwelding"
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.ledger_file = self.storage_path / "maker_marks.json"
        self.marks: List[MakerMark] = []
        self._load()

    def _load(self) -> None:
        if not self.ledger_file.exists():
            return
        try:
            data = json.loads(self.ledger_file.read_text(encoding="utf-8"))
            for mark_data in data.get("marks", []):
                self.marks.append(MakerMark.from_dict(mark_data))
        except Exception:
            pass

    def _save(self) -> None:
        data = {
            "marks": [mark.to_dict() for mark in self.marks],
            "updated_at": datetime.now().isoformat(),
        }
        self.ledger_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def record(self, mark: MakerMark) -> MakerMark:
        self.marks.append(mark)
        self._save()
        return mark

    def list(self) -> List[MakerMark]:
        return sorted(self.marks, key=lambda m: m.created_at, reverse=True)

    def for_issue_type(self, issue_type: str) -> List[MakerMark]:
        return [m for m in self.marks if m.issue_type == issue_type]

    def eligibility_signal(self, issue_type: str) -> Dict[str, Any]:
        """Return a signal for #579 pre-qualification checks.

        Patterns with strong mark history favor automatic attempts; patterns
        with poor or no history favor routing to a human first.
        """
        marks = self.for_issue_type(issue_type)
        if not marks:
            return {
                "known": False,
                "total_marks": 0,
                "success_rate": 0.0,
                "prefer_automatic": False,
            }
        first_attempt = sum(1 for m in marks if m.first_attempt_success)
        no_intervention = sum(1 for m in marks if not m.human_intervention_required)
        success_rate = (no_intervention / len(marks)) if marks else 0.0
        return {
            "known": True,
            "total_marks": len(marks),
            "first_attempt_success_rate": first_attempt / len(marks),
            "success_rate": success_rate,
            "prefer_automatic": success_rate >= 0.5,
        }

    def top_distillation_candidates(self, top_k: int = 10) -> List[MakerMark]:
        """Return marked successes prioritized for SkillDistiller.

        Marked successes are treated as higher-priority skill candidates than
        ordinary distilled workflows (issue #586).
        """
        scored = []
        for mark in self.marks:
            score = 1.0 if mark.first_attempt_success else 0.5
            if not mark.human_intervention_required:
                score += 0.5
            if mark.merged:
                score += 0.5
            scored.append((mark, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [mark for mark, _ in scored[:top_k]]


def is_mark_eligible(
    *,
    first_attempt_success: bool,
    human_intervention_required: bool,
    merged: bool = False,
    explicitly_tagged: bool = False,
) -> bool:
    """Define the conditions under which a mark is stamped (issue #586).

    A mark is recorded when a PR passes all CI checks on the first attempt
    without rework caused by human review comments, or when a human explicitly
    tags the outcome as a standout success.
    """
    if explicitly_tagged:
        return True
    return bool(first_attempt_success and not human_intervention_required and merged)

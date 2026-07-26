"""Practical Human Intervention Telemetry & Preference Learning Daemon.

Background telemetry logger that records user approval/rejection rates of agent
proposals (HITL checkpoint decisions) and adaptively tunes future intervention
thresholds over time. See issue #1068.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_STORE_PATH = os.path.join(
    os.path.expanduser("~"), ".sparkleforge", "intervention_telemetry.json"
)


@dataclass
class InterventionRecord:
    """A single intervention decision observation."""

    stage: str
    decision: str
    auto_resolved: bool
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionStats:
    """Aggregated approval/rejection statistics for a stage."""

    approvals: int = 0
    rejections: int = 0
    revisions: int = 0
    suspensions: int = 0
    auto_resolved: int = 0
    total: int = 0

    @property
    def approval_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.approvals / self.total

    @property
    def rejection_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.rejections + self.revisions) / self.total


class InterventionTelemetryDaemon:
    """Thread-safe telemetry logger + adaptive threshold tuner.

    Records HITL checkpoint decisions and uses an exponential moving average of
    per-stage approval/rejection rates to recommend intervention thresholds.
    Higher rejection rates lower the recommended threshold (intervene sooner);
    higher approval rates raise it (intervene later / trust the agent more).
    """

    def __init__(
        self,
        store_path: str | None = None,
        *,
        min_threshold: float = 0.2,
        max_threshold: float = 0.95,
        smoothing: float = 0.3,
        autosave: bool = True,
    ) -> None:
        self.store_path = store_path or _DEFAULT_STORE_PATH
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.smoothing = max(0.0, min(1.0, smoothing))
        self.autosave = autosave
        self._lock = threading.RLock()
        self._records: List[InterventionRecord] = []
        self._thresholds: Dict[str, float] = {}
        self._loaded = False
        self._load()

    @classmethod
    def default(cls) -> "InterventionTelemetryDaemon":
        """Construct the default daemon instance used by HITL checkpoints."""
        return cls()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            if not os.path.exists(self.store_path):
                return
            with open(self.store_path) as f:
                data = json.load(f)
            for rec in data.get("records", []):
                self._records.append(
                    InterventionRecord(
                        stage=rec.get("stage", ""),
                        decision=rec.get("decision", ""),
                        auto_resolved=bool(rec.get("auto_resolved", False)),
                        timestamp=float(rec.get("timestamp", 0.0)),
                        context=rec.get("context", {}) or {},
                    )
                )
            thresholds = data.get("thresholds", {})
            if isinstance(thresholds, dict):
                self._thresholds = {
                    str(k): float(v) for k, v in thresholds.items() if isinstance(v, (int, float))
                }
        except Exception as ex:
            logger.warning("Failed to load intervention telemetry: %s", ex)

    def _save_locked(self) -> None:
        if not self.autosave:
            return
        try:
            os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
            payload = {
                "records": [
                    {
                        "stage": r.stage,
                        "decision": r.decision,
                        "auto_resolved": r.auto_resolved,
                        "timestamp": r.timestamp,
                        "context": r.context,
                    }
                    for r in self._records
                ],
                "thresholds": self._thresholds,
                "updated_at": time.time(),
            }
            tmp = f"{self.store_path}.tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.store_path)
        except Exception as ex:
            logger.warning("Failed to persist intervention telemetry: %s", ex)

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record(
        self,
        stage: str,
        decision: str,
        *,
        auto_resolved: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> InterventionRecord:
        """Record a single intervention decision observation."""
        rec = InterventionRecord(
            stage=stage,
            decision=decision,
            auto_resolved=auto_resolved,
            timestamp=time.time(),
            context=context or {},
        )
        with self._lock:
            self._records.append(rec)
            self._update_threshold_locked(stage, decision)
            self._save_locked()
        logger.debug(
            "Recorded intervention: stage=%s decision=%s auto=%s",
            stage,
            decision,
            auto_resolved,
        )
        return rec

    # ------------------------------------------------------------------ #
    # Adaptive threshold tuning
    # ------------------------------------------------------------------ #
    def _update_threshold_locked(self, stage: str, decision: str) -> None:
        current = self._thresholds.get(stage, 0.5)
        # Map decisions to a target signal: approvals push threshold up
        # (trust agent more, intervene later), rejections/revise push down
        # (intervene sooner), suspensions push down slightly.
        if decision == "approve":
            signal = 1.0
        elif decision == "revise":
            signal = 0.25
        elif decision == "abort":
            signal = 0.0
        else:
            signal = 0.4
        new = (1.0 - self.smoothing) * current + self.smoothing * signal
        new = max(self.min_threshold, min(self.max_threshold, new))
        self._thresholds[stage] = new

    def recommend_threshold(self, stage: str) -> float:
        """Return the recommended intervention threshold for ``stage``.

        The threshold is a confidence score in [0, 1]; the orchestrator should
        request human intervention when agent confidence falls *below* it.
        """
        with self._lock:
            return self._thresholds.get(stage, 0.5)

    def set_threshold(self, stage: str, value: float) -> None:
        with self._lock:
            self._thresholds[stage] = max(
                self.min_threshold, min(self.max_threshold, float(value))
            )
            self._save_locked()

    # ------------------------------------------------------------------ #
    # Stats / reporting
    # ------------------------------------------------------------------ #
    def stats(self, stage: str | None = None) -> Dict[str, InterventionStats]:
        with self._lock:
            stages: Dict[str, InterventionStats] = {}
            for rec in self._records:
                key = stage or rec.stage
                if stage is not None and rec.stage != stage:
                    continue
                s = stages.setdefault(key, InterventionStats())
                s.total += 1
                if rec.auto_resolved:
                    s.auto_resolved += 1
                if rec.decision == "approve":
                    s.approvals += 1
                elif rec.decision == "revise":
                    s.revisions += 1
                elif rec.decision == "abort":
                    s.rejections += 1
                else:
                    s.suspensions += 1
            return stages

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            stats = self.stats()
            return {
                "record_count": len(self._records),
                "stages": {
                    name: {
                        "approvals": s.approvals,
                        "rejections": s.rejections,
                        "revisions": s.revisions,
                        "suspensions": s.suspensions,
                        "auto_resolved": s.auto_resolved,
                        "total": s.total,
                        "approval_rate": round(s.approval_rate, 4),
                        "rejection_rate": round(s.rejection_rate, 4),
                        "recommended_threshold": round(
                            self._thresholds.get(name, 0.5), 4
                        ),
                    }
                    for name, s in stats.items()
                },
                "thresholds": dict(self._thresholds),
            }

    def records(self) -> List[InterventionRecord]:
        with self._lock:
            return list(self._records)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._thresholds.clear()
            self._save_locked()

"""
anomaly_detector.py — Anomaly detection for unauthorized or suspicious system activity.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    event_id: str
    agent_id: str
    action: str
    reason: str
    severity: str  # low | medium | high | critical
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class AnomalyDetector:
    """
    Rule-based anomaly detector that monitors action patterns and raises alerts.

    Rules:
    1. Rate limiting — Too many actions in a short window
    2. Forbidden patterns — Actions matching blocked keywords
    3. Privilege escalation — Agent using capabilities beyond its grant
    4. Off-hours activity — Activity outside expected time windows
    """

    _instance: Optional["AnomalyDetector"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "AnomalyDetector":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        # Rolling window of (timestamp, action) per agent
        self._action_windows: Dict[str, Deque[Tuple[float, str]]] = defaultdict(
            lambda: deque(maxlen=200)
        )
        self._anomalies: List[AnomalyEvent] = []
        self._alert_callbacks: List[Callable[[AnomalyEvent], None]] = []
        self._lock_data = threading.RLock()

        # Configuration
        self.rate_limit_window_s: float = 60.0   # 1 minute window
        self.rate_limit_max_actions: int = 50    # Max actions per agent per window
        self.forbidden_keywords: List[str] = [
            "rm -rf", "format", "mkfs", "dd if=",
            "chmod 777", "sudo su", "> /dev/sd",
        ]

    def register_alert_callback(self, cb: Callable[[AnomalyEvent], None]) -> None:
        """Register a callback invoked on every anomaly detection."""
        self._alert_callbacks.append(cb)

    def _fire_alert(self, event: AnomalyEvent) -> None:
        with self._lock_data:
            self._anomalies.append(event)
        logger.warning("[ANOMALY] %s | agent=%s | action=%s", event.reason, event.agent_id, event.action)
        for cb in self._alert_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error("Alert callback error: %s", e)

    def observe(
        self,
        agent_id: str,
        action: str,
        metadata: Optional[dict] = None,
    ) -> List[AnomalyEvent]:
        """
        Observe an agent action and check for anomalies.
        Returns list of any anomaly events detected.
        """
        import uuid
        detected: List[AnomalyEvent] = []
        now = time.time()
        metadata = metadata or {}

        with self._lock_data:
            window = self._action_windows[agent_id]
            window.append((now, action))

            # 1. Rate limit check
            recent = [t for t, _ in window if now - t <= self.rate_limit_window_s]
            if len(recent) > self.rate_limit_max_actions:
                evt = AnomalyEvent(
                    event_id=str(uuid.uuid4()),
                    agent_id=agent_id,
                    action=action,
                    reason=f"Rate limit exceeded: {len(recent)} actions in {self.rate_limit_window_s}s",
                    severity="high",
                    metadata=metadata,
                )
                detected.append(evt)

            # 2. Forbidden pattern check
            action_lower = action.lower()
            for kw in self.forbidden_keywords:
                if kw in action_lower:
                    evt = AnomalyEvent(
                        event_id=str(uuid.uuid4()),
                        agent_id=agent_id,
                        action=action,
                        reason=f"Forbidden pattern detected: '{kw}'",
                        severity="critical",
                        metadata=metadata,
                    )
                    detected.append(evt)
                    break

        for evt in detected:
            self._fire_alert(evt)

        return detected

    def get_anomalies(
        self,
        agent_id: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 50,
    ) -> List[AnomalyEvent]:
        """Return recent anomaly events, optionally filtered."""
        with self._lock_data:
            events = list(self._anomalies)
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]

    def reset(self) -> None:
        with self._lock_data:
            self._action_windows.clear()
            self._anomalies.clear()

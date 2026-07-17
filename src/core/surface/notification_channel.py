"""
notification_channel.py — User approval notification channels (system tray, desktop, fallback log).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
import os
from enum import Enum
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

try:
    import plyer
    _PLYER_AVAILABLE = True
except ImportError:
    _PLYER_AVAILABLE = False

try:
    import pystray
    _PYSTRAY_AVAILABLE = True
except ImportError:
    _PYSTRAY_AVAILABLE = False

try:
    import aiohttp
    _WEBHOOK_URL = os.getenv("SPARKLEFORGE_ALERT_WEBHOOK")
except ImportError:
    _WEBHOOK_URL = None


class NotificationLevel(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"
    APPROVAL = "approval"   # Requires user action


@dataclass
class Notification:
    title: str
    message: str
    level: NotificationLevel = NotificationLevel.INFO
    action_id: Optional[str] = None    # ID for approval/rejection tracking
    callback: Optional[Callable[[], None]] = None  # On-click callback


class NotificationChannel:
    """
    Multi-backend notification dispatcher.
    
    Priority:
    1. OS desktop notification (via plyer if available)
    2. System tray icon (via pystray if available)
    3. Fallback: structured logging output
    """

    _instance: Optional["NotificationChannel"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "NotificationChannel":
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
        self._history: List[Notification] = []
        self._lock_data = threading.RLock()

        backend = "plyer" if _PLYER_AVAILABLE else ("pystray" if _PYSTRAY_AVAILABLE else "log")
        logger.info("NotificationChannel initialized with backend: %s", backend)

    def send(self, notification: Notification) -> bool:
        """Send a notification through available backends."""
        with self._lock_data:
            self._history.append(notification)

        # Try webhook for critical alerts
        if _WEBHOOK_URL and notification.level == NotificationLevel.CRITICAL:
            try:
                import requests
                requests.post(_WEBHOOK_URL, json={
                    "title": notification.title,
                    "message": notification.message,
                    "level": notification.level.value
                }, timeout=5)
                logger.debug("Notification sent via webhook: %s", notification.title)
            except Exception as e:
                logger.warning("Webhook notification failed: %s", e)

        # Try plyer (cross-platform desktop notifications)
        if _PLYER_AVAILABLE:
            try:
                import plyer
                plyer.notification.notify(
                    title=notification.title,
                    message=notification.message,
                    app_name="SparkleForge Anvil",
                    timeout=10,
                )
                logger.debug("Notification sent via plyer: %s", notification.title)
                return True
            except Exception as e:
                logger.warning("plyer notification failed: %s", e)

        # Fallback: structured log (always works)
        level_map = {
            NotificationLevel.INFO:     logger.info,
            NotificationLevel.WARNING:  logger.warning,
            NotificationLevel.CRITICAL: logger.critical,
            NotificationLevel.APPROVAL: logger.warning,
        }
        log_fn = level_map.get(notification.level, logger.info)
        log_fn(
            "[NOTIFICATION][%s] %s — %s",
            notification.level.value.upper(),
            notification.title,
            notification.message,
        )
        return True

    def notify_approval_needed(
        self,
        action: str,
        agent_id: str,
        risk_level: str,
        request_id: str,
    ) -> bool:
        """Send a user-facing approval request notification."""
        return self.send(Notification(
            title=f"⚠️ Approval Required [{risk_level.upper()}]",
            message=f"Agent '{agent_id}' wants to: {action}\n\nRequest ID: {request_id[:8]}",
            level=NotificationLevel.APPROVAL,
            action_id=request_id,
        ))

    def notify_anomaly(self, agent_id: str, reason: str, severity: str) -> bool:
        """Send a security anomaly alert notification."""
        return self.send(Notification(
            title=f"🚨 Security Alert [{severity.upper()}]",
            message=f"Anomaly detected from agent '{agent_id}':\n{reason}",
            level=NotificationLevel.CRITICAL,
        ))

    def get_history(self, limit: int = 50) -> List[Notification]:
        with self._lock_data:
            return list(self._history[-limit:])

    def reset(self) -> None:
        with self._lock_data:
            self._history.clear()

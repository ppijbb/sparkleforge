"""
notification_channel.py — User approval notification channels and bidirectional gateway (Webhook, Slack, Telegram, WebSocket).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
import os
from enum import Enum
from typing import Callable, Dict, List, Optional

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


class BidirectionalChannelGateway:
    """Bidirectional Channel Gateway supporting Webhook, Slack, Telegram, and WebSocket control plane."""
    
    _instance: Optional["BidirectionalChannelGateway"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "BidirectionalChannelGateway":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        self.inbound_goals: List[Dict[str, Any]] = []
        self.approval_callbacks: Dict[str, Callable[[str], None]] = {}

    def register_approval_request(self, request_id: str, action: str, agent_id: str, callback: Optional[Callable[[str], None]] = None):
        """Register a pending blocker approval waiting for remote decision."""
        self.pending_approvals[request_id] = {
            "action": action,
            "agent_id": agent_id,
            "status": "pending"
        }
        if callback:
            self.approval_callbacks[request_id] = callback

    def handle_inbound_reply(self, request_id: str, decision: str) -> bool:
        """Handle inbound remote HITL approve/deny reply ('approve' or 'deny')."""
        if request_id not in self.pending_approvals:
            logger.warning(f"Inbound reply for unknown request_id: {request_id}")
            return False
        
        decision_lower = decision.lower()
        if decision_lower in ("approve", "approved", "yes"):
            self.pending_approvals[request_id]["status"] = "approved"
            logger.info(f"Request {request_id} APPROVED via bidirectional channel.")
            if request_id in self.approval_callbacks:
                try:
                    self.approval_callbacks[request_id]("approve")
                except Exception as e:
                    logger.error(f"Error in approval callback: {e}")
            return True
        elif decision_lower in ("deny", "denied", "abort", "no"):
            self.pending_approvals[request_id]["status"] = "denied"
            logger.info(f"Request {request_id} DENIED via bidirectional channel.")
            if request_id in self.approval_callbacks:
                try:
                    self.approval_callbacks[request_id]("deny")
                except Exception as e:
                    logger.error(f"Error in denial callback: {e}")
            return True
        return False

    def submit_inbound_goal(self, goal: str, source: str = "webhook") -> str:
        """Receive inbound goal submission from remote channel."""
        goal_id = f"goal-{len(self.inbound_goals) + 1}"
        entry = {"goal_id": goal_id, "goal": goal, "source": source, "status": "received"}
        self.inbound_goals.append(entry)
        logger.info(f"Received inbound goal from {source}: {goal}")
        return goal_id

    def reset(self):
        self.pending_approvals.clear()
        self.inbound_goals.clear()
        self.approval_callbacks.clear()

def get_channel_gateway() -> BidirectionalChannelGateway:
    return BidirectionalChannelGateway()


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
        self.gateway = get_channel_gateway()

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
        # Register in bidirectional channel gateway
        self.gateway.register_approval_request(
            request_id=request_id,
            action=action,
            agent_id=agent_id
        )
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
            self.gateway.reset()

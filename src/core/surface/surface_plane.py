"""
surface_plane.py — Unified entry point for all Phase H surface/UI components.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.core.surface.nl_shell import NLShell
from src.core.surface.task_dashboard import TaskDashboard
from src.core.surface.notification_channel import NotificationChannel
from src.core.surface.explainability import ExplainabilityEngine

logger = logging.getLogger(__name__)


class SurfacePlane:
    """
    Aggregates all user-facing surface components:
    - Natural Language Shell
    - Task Queue Dashboard
    - Notification Channel
    - Action Explainability Engine

    Optionally wires the notification channel to guard plane alerts.
    """

    def __init__(self, guard_plane: Optional[Any] = None) -> None:
        self.nl_shell             = NLShell()
        self.task_dashboard       = TaskDashboard()
        self.notification_channel = NotificationChannel()
        self.explainability       = ExplainabilityEngine()

        # Wire guard plane alerts → notification channel
        if guard_plane is not None:
            self._wire_guard(guard_plane)

        logger.info("SurfacePlane initialized with all user-facing components")

    def _wire_guard(self, guard_plane: Any) -> None:
        """Connect guard plane anomaly detector to notification channel."""
        try:
            guard_plane.anomaly_detector.register_alert_callback(
                lambda evt: self.notification_channel.notify_anomaly(
                    agent_id=evt.agent_id,
                    reason=evt.reason,
                    severity=evt.severity,
                )
            )
            guard_plane.hitl_gate.set_approval_callback(
                lambda req: self.notification_channel.notify_approval_needed(
                    action=req.action,
                    agent_id=req.agent_id,
                    risk_level=str(req.risk_level),
                    request_id=req.request_id,
                )
            )
            logger.info("SurfacePlane wired to GuardPlane alerts")
        except Exception as e:
            logger.warning("Could not wire guard plane to surface: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Return a status summary of all surface components."""
        dashboard_summary = self.task_dashboard.summary()
        notifications_count = len(self.notification_channel.get_history())

        return {
            "task_summary":         dashboard_summary,
            "notifications_sent":   notifications_count,
            "nl_shell_history":     len(self.nl_shell.get_history()),
            "initialized":          True,
        }

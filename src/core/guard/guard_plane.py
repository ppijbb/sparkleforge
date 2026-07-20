"""
guard_plane.py — Unified entry point for all Phase G security/guard components.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from src.core.guard.capability_manager import CapabilityManager
from src.core.guard.sandbox_executor import SandboxExecutor
from src.core.guard.action_journal import ActionJournal
from src.core.guard.anomaly_detector import AnomalyDetector
from src.core.guard.credential_vault import CredentialVault

logger = logging.getLogger(__name__)


class GuardPlane:
    """
    Aggregates all guard subsystems: capability management, HITL gates,
    sandboxed execution, action journaling, anomaly detection, and credential vault.

    Instantiated once by BootstrapGraph and injected into other planes that need it.
    """

    def __init__(self) -> None:
        self.capability_manager = CapabilityManager()
        self.sandbox_executor   = SandboxExecutor()
        self.action_journal     = ActionJournal()
        self.anomaly_detector   = AnomalyDetector()
        self.credential_vault   = CredentialVault()

        # Wire anomaly alerts to journal
        self.anomaly_detector.register_alert_callback(self._on_anomaly)

        logger.info("GuardPlane initialized with all security subsystems")

    def _on_anomaly(self, event: Any) -> None:
        """Log anomaly events to the action journal."""
        self.action_journal.record(
            agent_id=event.agent_id,
            action="anomaly_detected",
            description=event.reason,
            risk_level="critical",
            metadata={"severity": event.severity, "original_action": event.action},
        )

    async def check_and_execute(
        self,
        agent_id: str,
        capability_name: str,
        command: str,
        description: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Full guard pipeline:
        1. Check capability grant
        2. Detect anomalies
        3. Request HITL approval if needed
        4. Execute in sandbox
        5. Journal the result
        """
        from src.core.guard.capability_manager import BUILTIN_CAPABILITIES

        # 1. Capability check
        if not self.capability_manager.agent_has(agent_id, capability_name):
            logger.warning("Agent '%s' lacks capability '%s'", agent_id, capability_name)
            return {"ok": False, "error": f"Missing capability: {capability_name}"}

        cap = BUILTIN_CAPABILITIES.get(capability_name)
        risk_level = cap.risk_level if cap else "low"

        # 2. Anomaly detection
        anomalies = self.anomaly_detector.observe(agent_id, command)
        if any(a.severity == "critical" for a in anomalies):
            return {"ok": False, "error": "Critical anomaly detected — action blocked"}

        # 3. HITL approval if needed -- fail closed. There is no live
        # human-approval channel wired to this path in production (issue
        # #776): it used to block on HITLGate.request_approval(), but nothing
        # outside tests ever called HITLGate.resolve(), so every HIGH/CRITICAL
        # request already timed out and denied after a multi-minute hang.
        # Denying immediately preserves that outcome without the pointless wait.
        if cap and cap.requires_hitl:
            self.action_journal.record(
                agent_id=agent_id,
                action=command,
                description=description,
                risk_level=str(risk_level),
                metadata={"blocked_by": "hitl_unavailable"},
            )
            return {"ok": False, "error": "Action requires human approval, which is not available"}

        # 4. Journal pre-execution
        entry = self.action_journal.record(
            agent_id=agent_id,
            action=command,
            description=description,
            risk_level=str(risk_level),
        )

        # 5. Execute in sandbox
        result = await asyncio.to_thread(
            self.sandbox_executor.execute,
            command,
            dry_run=dry_run,
        )

        # 6. Update journal with outcome
        self.action_journal.update_outcome(
            entry.entry_id,
            outcome="success" if result.ok else "failure",
            error=result.stderr if not result.ok else None,
        )

        return {
            "ok":          result.ok,
            "stdout":      result.stdout,
            "stderr":      result.stderr,
            "returncode":  result.returncode,
            "duration_ms": result.duration_ms,
            "sandbox":     result.sandbox_type,
            "entry_id":    entry.entry_id,
        }

    async def check_and_control_device(
        self,
        agent_id: str,
        device_id: str,
        command: str,
        description: str,
        is_write: bool = True,
    ) -> Dict[str, Any]:
        """
        IoT Device Guard pipeline:
        1. Determine capability based on action type (read or write)
        2. Check capability grant (iot_read or iot_control)
        3. Human-in-the-loop (HITL) check if required
        4. Journal the execution
        5. Execute control via ActuationPlane
        """
        from src.core.guard.capability_manager import BUILTIN_CAPABILITIES
        from src.core.actuate.actuation_plane import ActuationPlane
        
        capability_name = "iot_control" if is_write else "iot_read"
        
        # 1. Capability check
        if not self.capability_manager.agent_has(agent_id, capability_name):
            logger.warning("Agent '%s' lacks capability '%s' to control IoT device '%s'", agent_id, capability_name, device_id)
            return {"ok": False, "error": f"Missing capability: {capability_name}"}

        cap = BUILTIN_CAPABILITIES.get(capability_name)
        risk_level = cap.risk_level if cap else "low"

        # 2. HITL approval check -- fail closed (see check_and_execute's HITL
        # block above for why: no live approval channel is wired in
        # production, so this always denied after a timeout anyway).
        if cap and cap.requires_hitl:
            self.action_journal.record(
                agent_id=agent_id,
                action=f"iot_control:{device_id}",
                description=description,
                risk_level=str(risk_level),
                metadata={"blocked_by": "hitl_unavailable"},
            )
            return {"ok": False, "error": "Action requires human approval, which is not available"}

        # 3. Journal pre-execution
        entry = self.action_journal.record(
            agent_id=agent_id,
            action=f"iot:{device_id}:{command}",
            description=description,
            risk_level=str(risk_level),
        )

        # 4. Execute via ActuationPlane
        actuator = ActuationPlane()
        try:
            if hasattr(actuator, "control_device"):
                # If command is write action
                if is_write:
                    result_data = actuator.control_device(device_id, command)
                else:
                    result_data = actuator.read_device(device_id, command)
                ok = result_data.get("status") == "success"
                stdout = result_data.get("stdout", "")
                stderr = result_data.get("stderr", "")
                returncode = result_data.get("returncode", 0)
            else:
                ok = False
                stdout = ""
                stderr = "ActuationPlane lacks control_device interface."
                returncode = -1
        except Exception as e:
            ok = False
            stdout = ""
            stderr = str(e)
            returncode = -1

        # 5. Journal post-execution outcome
        self.action_journal.update_outcome(
            entry.entry_id,
            outcome="success" if ok else "failure",
            error=stderr if not ok else None,
        )

        return {
            "ok": ok,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "entry_id": entry.entry_id,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return a status summary of all guard subsystems."""
        recent_anomalies  = self.anomaly_detector.get_anomalies(limit=5)
        recent_actions    = self.action_journal.recent(limit=5)

        return {
            "recent_anomalies":  len(recent_anomalies),
            "recent_actions":    len(recent_actions),
            "credential_keys":   len(self.credential_vault.list_keys()),
            "initialized":       True,
        }

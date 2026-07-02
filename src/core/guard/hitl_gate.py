"""
hitl_gate.py — Risk-tiered Human-in-the-loop (HITL) approval gates.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from src.core.guard.capability_manager import RiskLevel

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    PENDING  = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT  = "timeout"
    AUTO     = "auto"     # Auto-approved (low-risk)


@dataclass
class ApprovalRequest:
    request_id: str
    agent_id: str
    action: str
    description: str
    risk_level: RiskLevel
    metadata: dict = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    resolver: Optional[str] = None  # Who approved/rejected (human or "auto")


# Thresholds: risk levels that require human approval
HITL_REQUIRED_LEVELS = {RiskLevel.HIGH, RiskLevel.CRITICAL}
# Auto-approved levels
AUTO_APPROVE_LEVELS   = {RiskLevel.LOW, RiskLevel.MEDIUM}

# Default timeout (seconds) per risk level
APPROVAL_TIMEOUTS: Dict[RiskLevel, float] = {
    RiskLevel.LOW:      0.0,    # Instant auto-approve
    RiskLevel.MEDIUM:   0.0,    # Instant auto-approve
    RiskLevel.HIGH:     120.0,  # 2 minute timeout
    RiskLevel.CRITICAL: 300.0,  # 5 minute timeout
}


class HITLGate:
    """
    Manages Human-in-the-loop approval gates for risky actions.

    In production, approval callbacks notify the user via the notification
    channel. In test/headless mode, a default_approver function can be
    injected that auto-approves or auto-rejects.
    """

    _instance: Optional["HITLGate"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "HITLGate":
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
        self._requests: Dict[str, ApprovalRequest] = {}
        self._events:   Dict[str, threading.Event] = {}
        self._lock_data = threading.RLock()
        # Callback invoked when approval is needed (UI/notification)
        self._approval_callback: Optional[Callable[[ApprovalRequest], None]] = None
        # Inject a resolver for headless/test operation
        self._default_approver: Optional[Callable[[ApprovalRequest], bool]] = None

    def set_approval_callback(self, cb: Callable[[ApprovalRequest], None]) -> None:
        """Set callback fired when a new approval request arrives."""
        self._approval_callback = cb

    def set_default_approver(self, fn: Callable[[ApprovalRequest], bool]) -> None:
        """Inject a headless approver function (returns True=approve, False=reject)."""
        self._default_approver = fn

    def request_approval(
        self,
        agent_id: str,
        action: str,
        description: str,
        risk_level: RiskLevel,
        metadata: Optional[dict] = None,
    ) -> ApprovalRequest:
        """
        Submit an action for approval.  Returns the resolved ApprovalRequest.
        Blocks until approved, rejected, or timeout.
        """
        req = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            agent_id=agent_id,
            action=action,
            description=description,
            risk_level=risk_level,
            metadata=metadata or {},
        )

        # Auto-approve low/medium risk immediately
        if risk_level in AUTO_APPROVE_LEVELS:
            req.status   = ApprovalStatus.AUTO
            req.resolver = "auto"
            req.resolved_at = time.time()
            logger.debug("Auto-approved action '%s' (risk=%s)", action, risk_level)
            return req

        with self._lock_data:
            self._requests[req.request_id] = req
            evt = threading.Event()
            self._events[req.request_id] = evt

        # Fire UI callback if registered
        if self._approval_callback:
            try:
                self._approval_callback(req)
            except Exception as e:
                logger.warning("Approval callback error: %s", e)

        # Headless auto-approver
        if self._default_approver:
            try:
                approved = self._default_approver(req)
                self.resolve(req.request_id, approved=approved, resolver="auto")
                evt.set()
            except Exception as e:
                logger.warning("Default approver error: %s", e)

        # Wait for resolution with timeout
        timeout = APPROVAL_TIMEOUTS.get(risk_level, 120.0)
        granted = evt.wait(timeout=timeout)

        with self._lock_data:
            resolved = self._requests.get(req.request_id, req)
            if not granted or resolved.status == ApprovalStatus.PENDING:
                resolved.status      = ApprovalStatus.TIMEOUT
                resolved.resolved_at = time.time()
                logger.warning("Approval timed out for action '%s'", action)

        return resolved

    def resolve(self, request_id: str, approved: bool, resolver: str = "human") -> bool:
        """Resolve a pending approval request."""
        with self._lock_data:
            req = self._requests.get(request_id)
            if not req or req.status != ApprovalStatus.PENDING:
                return False
            req.status      = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
            req.resolved_at = time.time()
            req.resolver    = resolver
            evt = self._events.get(request_id)
            if evt:
                evt.set()
        logger.info(
            "Request %s %s by %s",
            request_id,
            "APPROVED" if approved else "REJECTED",
            resolver,
        )
        return True

    def get_pending(self) -> List[ApprovalRequest]:
        """Return all pending approval requests."""
        with self._lock_data:
            return [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]

    def get_history(self, limit: int = 50) -> List[ApprovalRequest]:
        """Return recent resolved requests."""
        with self._lock_data:
            resolved = [r for r in self._requests.values() if r.status != ApprovalStatus.PENDING]
            return sorted(resolved, key=lambda r: r.created_at, reverse=True)[:limit]

    def is_approved(self, request: ApprovalRequest) -> bool:
        return request.status in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO)

    def reset(self) -> None:
        """Clear all state (for testing)."""
        with self._lock_data:
            self._requests.clear()
            self._events.clear()

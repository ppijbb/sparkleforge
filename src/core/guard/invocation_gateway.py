"""InvocationGateway: single mandatory pass-through for agent-to-agent invocation.

Issue #568. `docs/ANVIL_PLAN.md` §4 identified that agent delegation
(`src/core/orchestrator/delegation.py::delegate_to_agent`) and MCP tool
execution (`src/core/mcp_integration/hub_mixins/execution.py::execute_tool`)
each had to remember to add their own IntentGuardrail/CapabilityManager
checks by hand -- and neither one actually had, before this. That's the
pattern behind #516/#519/#312: a new call path gets added, nobody re-adds
the guard checks, and the same class of defect resurfaces.

This module gives both real call paths one place to route an authorization
decision through, so the check can't be silently skipped by a new call site
forgetting to wire it in, and every decision (allowed or denied) lands in
`ActionJournal` regardless of outcome -- so `action_journal.jsonl` alone is
enough to reconstruct any invocation's history.

Credential delegation (the third path #568 names) has no real implementation
yet -- `CredentialVault` is local secret storage with no delegation concept.
`InvocationKind.CREDENTIAL_DELEGATION` is reserved here for #614 (agent
identity & mandates) to route through once that call path actually exists;
retrofitting a path that doesn't exist would be premature.

CapabilityManager's existing model is coarse action-risk capabilities
(execute_shell, write_file, network_request, ...), not "may X delegate to
role Y" -- so `required_capability` is optional per call and only checked
when the caller can name a specific BUILTIN_CAPABILITIES entry that applies;
forcing an invented capability name here would be incoherent with the rest
of the guard plane, not a real fix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

from src.core.guard.action_journal import ActionJournal
from src.core.guard.capability_manager import CapabilityManager

if TYPE_CHECKING:
    from src.core.anvil.intent_guardrail import IntentGuardrail

logger = logging.getLogger(__name__)

SYSTEM_ACTOR = "system"

# Coarse action-risk capabilities the "system" actor is granted by default --
# get_current_agent_name() (src/core/agent_security.py) has no real caller
# for agent_security_context() anywhere in the codebase today, so it always
# returns None and every MCP tool call's actor resolves to SYSTEM_ACTOR.
# CapabilityManager grants were never issued anywhere in production before
# this gateway existed (issue #777); without this bootstrap, turning
# required_capability checks on here would silently deny every real
# shell/file/browser tool call the moment this gateway went live.
_SYSTEM_ACTOR_DEFAULT_CAPABILITIES = ["execute_shell", "write_file", "read_file", "network_request"]


class InvocationKind(Enum):
    AGENT_DELEGATION = "agent_delegation"
    MCP_TOOL = "mcp_tool"
    CREDENTIAL_DELEGATION = "credential_delegation"  # reserved for #614; no real caller yet


@dataclass
class InvocationDecision:
    allowed: bool
    reason: str


class InvocationGateway:
    """Single mandatory choke point for agent delegation and MCP tool execution."""

    def __init__(
        self,
        capability_manager: CapabilityManager | None = None,
        action_journal: ActionJournal | None = None,
    ):
        self.capability_manager = capability_manager or CapabilityManager()
        self.action_journal = action_journal or ActionJournal()
        self.capability_manager.grant_default(SYSTEM_ACTOR, _SYSTEM_ACTOR_DEFAULT_CAPABILITIES)

    def authorize(
        self,
        *,
        kind: InvocationKind,
        actor: str,
        target: str,
        description: str,
        required_capability: Optional[str] = None,
        intent_guardrail: Optional["IntentGuardrail"] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InvocationDecision:
        """Authorize a single agent-delegation or MCP-tool invocation.

        actor: who is making the call (delegator_id, or the current agent
            identity for a tool call).
        target: what's being invoked (the delegated role, or the tool name).
        required_capability: name of a BUILTIN_CAPABILITIES entry this
            invocation maps to, if any -- pass None when no existing
            capability cleanly applies (see module docstring).
        intent_guardrail: the caller's active IntentGuardrail, if it has one
            attached to the current session/workflow -- checked directly via
            evaluate() rather than its periodic should_check() sampling,
            since this is a discrete per-call authorization decision, not a
            periodic health check.

        Every call is journaled to ActionJournal regardless of outcome.
        """
        reasons: list[str] = []
        allowed = True

        if required_capability is not None:
            if not self.capability_manager.agent_has(actor, required_capability):
                allowed = False
                reasons.append(f"missing capability '{required_capability}'")

        if intent_guardrail is not None:
            try:
                assessment = intent_guardrail.evaluate(description)
                if not assessment.aligned:
                    allowed = False
                    reasons.append(f"intent drift detected (score={assessment.drift_score:.2f})")
            except Exception as e:
                logger.warning("InvocationGateway: IntentGuardrail evaluation failed: %s", e)

        decision = InvocationDecision(
            allowed=allowed,
            reason="; ".join(reasons) if reasons else "authorized",
        )

        self.action_journal.record(
            agent_id=actor,
            action=f"invocation_gateway:{kind.value}",
            description=(
                f"{'ALLOWED' if allowed else 'DENIED'}: {kind.value} -> {target} ({description})"
            ),
            risk_level="low" if allowed else "medium",
            metadata={
                "target": target,
                "required_capability": required_capability,
                **(metadata or {}),
            },
        )
        return decision


_invocation_gateway: InvocationGateway | None = None


def get_invocation_gateway() -> InvocationGateway:
    """Global InvocationGateway accessor, mirroring get_llm_orchestrator()'s singleton pattern."""
    global _invocation_gateway
    if _invocation_gateway is None:
        _invocation_gateway = InvocationGateway()
    return _invocation_gateway

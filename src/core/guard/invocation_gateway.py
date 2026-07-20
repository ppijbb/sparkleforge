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
`InvocationKind.CREDENTIAL_DELEGATION` is reserved here for whenever that
call path is actually built; retrofitting a path that doesn't exist would
be premature. What issue #614 (agent identity & signed mandates) *did* add
is `authorize()`'s optional `mandate`/`issuer_public_key_b64` parameters --
any invocation kind (agent delegation, MCP tool, or credential delegation
once it exists) can present a cryptographically signed `Mandate`
(`src/core/guard/agent_identity.py`) instead of relying on a pre-existing
local capability grant for `actor`.

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
    from src.core.guard.agent_identity import Mandate

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
        mandate: Optional["Mandate"] = None,
        issuer_public_key_b64: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InvocationDecision:
        """Authorize a single agent-delegation, MCP-tool, or credential-delegation invocation.

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
        mandate / issuer_public_key_b64: a signed Mandate (issue #614) and
            the issuer's public key to verify it against. When present, this
            supersedes the plain actor-based capability check: the identity
            being authorized is mandate.subject, not `actor` (actor is still
            who placed the call, for the journal). A valid mandate that
            covers required_capability grants that capability to
            mandate.subject via CapabilityManager -- so the grant persists
            for the mandate's remaining validity window, not just this call.

        Every call is journaled to ActionJournal regardless of outcome,
        including the mandate verification result when one was presented.
        """
        reasons: list[str] = []
        allowed = True
        mandate_info: Optional[Dict[str, Any]] = None

        if mandate is not None:
            from src.core.guard.agent_identity import mandate_covers_capability, verify_mandate

            if issuer_public_key_b64 is None:
                allowed = False
                reasons.append("mandate presented without an issuer public key to verify against")
                mandate_info = {"subject": mandate.subject, "issuer": mandate.issuer, "valid": False}
            else:
                valid, verify_reason = verify_mandate(mandate, issuer_public_key_b64)
                mandate_info = {
                    "issuer": mandate.issuer,
                    "subject": mandate.subject,
                    "scope": mandate.scope,
                    "valid": valid,
                    "reason": verify_reason,
                }
                if not valid:
                    allowed = False
                    reasons.append(f"mandate invalid: {verify_reason}")
                elif required_capability is not None and not mandate_covers_capability(
                    mandate, required_capability
                ):
                    allowed = False
                    reasons.append(
                        f"mandate scope {mandate.scope} does not cover required capability "
                        f"'{required_capability}'"
                    )
                elif required_capability is not None:
                    # Reflect the verified delegation as a real (persisted) capability
                    # grant for the mandate's subject, so agent_has() checks elsewhere
                    # in the codebase also see it -- not just this one gateway decision.
                    self.capability_manager.grant_agent(mandate.subject, required_capability)
        elif required_capability is not None:
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
                # Fail closed: a guardrail that cannot render a verdict must not be
                # treated as a pass. With SYSTEM_ACTOR pre-granted every dangerous
                # capability and get_current_agent_name() always resolving to it
                # today, this guardrail is the only real check in front of
                # execute_shell/write_file -- allowing on error would make it a
                # no-op exactly when it errors.
                logger.error("InvocationGateway: IntentGuardrail evaluation failed: %s", e)
                allowed = False
                reasons.append(f"guardrail_evaluation_failed: {e}")

        decision = InvocationDecision(
            allowed=allowed,
            reason="; ".join(reasons) if reasons else "authorized",
        )

        journal_metadata = {
            "target": target,
            "required_capability": required_capability,
            **(metadata or {}),
        }
        if mandate_info is not None:
            journal_metadata["mandate"] = mandate_info

        self.action_journal.record(
            agent_id=actor,
            action=f"invocation_gateway:{kind.value}",
            description=(
                f"{'ALLOWED' if allowed else 'DENIED'}: {kind.value} -> {target} ({description})"
            ),
            risk_level="low" if allowed else "medium",
            metadata=journal_metadata,
        )
        return decision


_invocation_gateway: InvocationGateway | None = None


def get_invocation_gateway() -> InvocationGateway:
    """Global InvocationGateway accessor, mirroring get_llm_orchestrator()'s singleton pattern."""
    global _invocation_gateway
    if _invocation_gateway is None:
        _invocation_gateway = InvocationGateway()
    return _invocation_gateway

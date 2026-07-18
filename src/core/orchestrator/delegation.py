"""Runtime sub-agent delegation for the orchestrator graph (issue #495, #509).

The orchestrator StateGraph (`src/core/orchestrator/graph.py`) is a fixed DAG:
Planner -> Verify -> Adaptive Supervisor -> Execute -> Compress -> Continuous
Verification -> Overseer -> Synthesis. Reaching a role with no static edge
(e.g. asking `validation_agent` to spot-check a result mid-execution) used to
require adding a new hardcoded node and redeploying.

`delegate_to_agent` lets `execute_research` / `adaptive_supervisor` invoke any
registered agent role at runtime instead. Depth is bounded the same way the
existing `overseer_iterations` / `max_iterations` guard bounds Overseer
retries, and every call is journaled via `ActionJournal` for traceability.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict

from src.core.guard.action_journal import ActionJournal
from src.core.guard.invocation_gateway import InvocationKind, get_invocation_gateway
from src.core.orchestrator.state import ResearchState

logger = logging.getLogger(__name__)

DEFAULT_MAX_DELEGATION_DEPTH = 3

DelegationAdapter = Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[Any]]


class DelegationDepthExceeded(RuntimeError):
    """Raised when a delegation chain would exceed the configured depth limit."""


class DelegationDenied(RuntimeError):
    """Raised when InvocationGateway denies a delegation (issue #568)."""


async def _delegate_research_agent(task: Dict[str, Any], context: Dict[str, Any]) -> Any:
    from src.agents.research_agent import ResearchAgent

    agent = ResearchAgent()
    return await agent.execute_task(
        task=task,
        objective_id=str(context.get("objective_id", "delegated")),
        is_refinement=bool(context.get("is_refinement", False)),
        context=context,
    )


async def _delegate_validation_agent(task: Dict[str, Any], context: Dict[str, Any]) -> Any:
    from src.agents.validation_agent import ValidationAgent

    agent = ValidationAgent()
    return await agent.validate_results(
        execution_results=context.get("execution_results") or [task],
        original_objectives=context.get("objectives", []),
        user_request=context.get("user_request", task.get("description", "")),
        context=context,
        objective_id=context.get("objective_id"),
    )


async def _delegate_verifier_agent(task: Dict[str, Any], context: Dict[str, Any]) -> Any:
    from src.agents.verifier_agent import VerifierAgent

    agent = VerifierAgent()
    return await agent.verify_results(
        state=context.get("harness_state", {}),
        completed_tasks=context.get("completed_tasks") or [task],
    )


async def _delegate_evaluation_agent(task: Dict[str, Any], context: Dict[str, Any]) -> Any:
    from src.agents.evaluation_agent import EvaluationAgent

    agent = EvaluationAgent()
    return await agent.evaluate_results(
        execution_results=context.get("execution_results") or [task],
        original_objectives=context.get("objectives", []),
        context=context,
        objective_id=context.get("objective_id"),
    )


async def _delegate_codebase_agent(task: Dict[str, Any], context: Dict[str, Any]) -> Any:
    from pathlib import Path

    from src.agents.codebase_agent import CodebaseAgent

    path = task.get("path") or context.get("path")
    agent = CodebaseAgent(Path(path) if path else None)
    return await agent.analyze_codebase(
        include_patterns=context.get("include_patterns"),
        exclude_patterns=context.get("exclude_patterns"),
    )


async def _delegate_document_organizer_agent(task: Dict[str, Any], context: Dict[str, Any]) -> Any:
    from pathlib import Path

    from src.agents.document_organizer_agent import DocumentOrganizerAgent

    path = task.get("path") or context.get("path")
    agent = DocumentOrganizerAgent(Path(path) if path else None)
    return await agent.analyze_documents(
        include_patterns=context.get("include_patterns"),
        exclude_patterns=context.get("exclude_patterns"),
    )


# Roles reachable via runtime delegation even though the static graph has no
# edge to them. Each adapter normalizes the generic (task, context) call into
# that agent's real signature and returns its native result unchanged.
DELEGATION_REGISTRY: Dict[str, DelegationAdapter] = {
    "research_agent": _delegate_research_agent,
    "validation_agent": _delegate_validation_agent,
    "verifier_agent": _delegate_verifier_agent,
    "evaluation_agent": _delegate_evaluation_agent,
    "codebase_agent": _delegate_codebase_agent,
    "document_organizer_agent": _delegate_document_organizer_agent,
}


async def delegate_to_agent(
    state: ResearchState,
    role: str,
    task: Dict[str, Any],
    context: Dict[str, Any] | None = None,
    delegator_id: str = "orchestrator",
) -> Dict[str, Any]:
    """Invoke `role` at runtime, bounded by delegation depth and journaled.

    Depth is tracked exclusively through `context`, not `state` (issue #516):
    `state` is a single mutable object shared across the whole orchestrator
    run, so a nested call (a delegated agent that itself delegates further)
    would always observe the pre-increment depth once the parent's cleanup
    restored it, making the guard ineffective for chains longer than one hop.
    It also meant concurrent delegations dispatched via `asyncio.gather`
    would race on the same `state["delegation_depth"]` key. `context` is a
    fresh dict per call (each adapter receives its own `{**context,
    "delegation_depth": depth + 1}`), so both problems disappear: a nested
    `delegate_to_agent` call just needs to pass its received `context` back
    in, and concurrent siblings never share a mutable counter.

    `state` is only consulted for `max_delegation_depth` on the outermost
    call (when `context` doesn't already carry one forward).
    """
    context = dict(context or {})
    depth = int(context.get("delegation_depth") or 0)
    max_depth = int(
        context.get("max_delegation_depth")
        or state.get("max_delegation_depth")
        or DEFAULT_MAX_DELEGATION_DEPTH
    )

    journal = ActionJournal()

    if depth >= max_depth:
        journal.record(
            agent_id=delegator_id,
            action="delegate_to_agent_denied",
            description=f"Delegation depth limit ({max_depth}) reached; refusing to delegate to '{role}'",
            risk_level="medium",
            metadata={"role": role, "depth": depth, "max_depth": max_depth},
        )
        raise DelegationDepthExceeded(
            f"Delegation depth limit ({max_depth}) reached; cannot delegate to '{role}'"
        )

    adapter = DELEGATION_REGISTRY.get(role)
    if adapter is None:
        journal.record(
            agent_id=delegator_id,
            action="delegate_to_agent_rejected",
            description=f"Unknown delegation role '{role}'",
            risk_level="low",
            metadata={"role": role, "available_roles": sorted(DELEGATION_REGISTRY)},
        )
        raise ValueError(
            f"Unknown delegation role '{role}'. Available roles: {sorted(DELEGATION_REGISTRY)}"
        )

    decision = get_invocation_gateway().authorize(
        kind=InvocationKind.AGENT_DELEGATION,
        actor=delegator_id,
        target=role,
        description=f"delegate task {task.get('id') or task.get('task_id')} to '{role}'",
        intent_guardrail=context.get("intent_guardrail"),
        metadata={"depth": depth + 1, "task_id": task.get("id") or task.get("task_id")},
    )
    if not decision.allowed:
        raise DelegationDenied(
            f"InvocationGateway denied delegation to '{role}': {decision.reason}"
        )

    entry = journal.record(
        agent_id=delegator_id,
        action="delegate_to_agent",
        description=f"Delegating task to '{role}' at depth {depth + 1}/{max_depth}",
        risk_level="low",
        metadata={"role": role, "depth": depth + 1, "task_id": task.get("id") or task.get("task_id")},
    )

    child_context = {**context, "delegation_depth": depth + 1, "max_delegation_depth": max_depth}
    try:
        result = await adapter(task, child_context)
    except Exception as e:  # noqa: BLE001 - surfaced to caller, not swallowed
        logger.warning("Delegation to '%s' failed: %s", role, e)
        journal.update_outcome(entry.entry_id, outcome="failure", error=str(e))
        return {"role": role, "success": False, "error": str(e), "delegation_depth": depth + 1}
    else:
        journal.update_outcome(entry.entry_id, outcome="success")
        return {"role": role, "success": True, "result": result, "delegation_depth": depth + 1}

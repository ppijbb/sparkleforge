"""Input router: normalize multi-source inputs into InputEnvelope (OpenClaw-style).

Supports: message (user_query), heartbeat, cron, webhook, agent_to_agent.
Trace context (request_id, turn_id, entrypoint, source_type) is ensured at lane boundary.
"""

import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict

InputType = str  # "message" | "heartbeat" | "cron" | "webhook" | "agent_to_agent"

# Standard trace context keys for correlation across turn/agent/task/tool/llm.
TRACE_REQUEST_ID = "request_id"
TRACE_TURN_ID = "turn_id"
TRACE_ENTRYPOINT = "entrypoint"
TRACE_SOURCE_TYPE = "source_type"
TRACE_TASK_ID = "task_id"
TRACE_AGENT_ID = "agent_id"

_current_trace_context: ContextVar[Dict[str, Any] | None] = ContextVar(
    "trace_context", default=None
)


def get_trace_context() -> Dict[str, Any] | None:
    """Return current trace context (set by session lane / orchestrator)."""
    return _current_trace_context.get()


def set_trace_context(ctx: Dict[str, Any] | None) -> None:
    """Set current trace context for this async context."""
    _current_trace_context.set(ctx)


@dataclass
class InputEnvelope:
    """Normalized input for session lane / orchestrator."""

    type: InputType
    session_id: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


def normalize_message(session_id: str, user_query: str, **kwargs: Any) -> InputEnvelope:
    """Normalize user message (e.g. chat/API) into InputEnvelope."""
    return InputEnvelope(
        type="message",
        session_id=session_id,
        payload={"user_query": user_query, **kwargs},
        metadata=kwargs,
    )


def normalize_heartbeat(session_id: str, **kwargs: Any) -> InputEnvelope:
    """Normalize heartbeat (periodic 'check for work') into InputEnvelope."""
    return InputEnvelope(
        type="heartbeat",
        session_id=session_id,
        payload=dict(kwargs),
        metadata=kwargs,
    )


def normalize_cron(session_id: str, instruction: str, **kwargs: Any) -> InputEnvelope:
    """Normalize cron/scheduled job into InputEnvelope."""
    return InputEnvelope(
        type="cron",
        session_id=session_id,
        payload={"user_query": instruction, "cron": True, **kwargs},
        metadata=kwargs,
    )


def normalize_webhook(session_id: str, body: Dict[str, Any], **kwargs: Any) -> InputEnvelope:
    """Normalize webhook HTTP body into InputEnvelope."""
    return InputEnvelope(
        type="webhook",
        session_id=session_id,
        payload={"body": body, **kwargs},
        metadata=kwargs,
    )


def normalize_agent_to_agent(
    session_id: str,
    target_agent: str | None,
    target_session: str | None,
    instruction: str,
    **kwargs: Any,
) -> InputEnvelope:
    """Normalize agent-to-agent task request into InputEnvelope."""
    return InputEnvelope(
        type="agent_to_agent",
        session_id=target_session or session_id,
        payload={
            "user_query": instruction,
            "target_agent": target_agent,
            "source_session": session_id,
            **kwargs,
        },
        metadata=kwargs,
    )


def ensure_trace_context(envelope: InputEnvelope) -> None:
    """Ensure envelope.metadata has request_id, turn_id, entrypoint, source_type.
    Call at lane boundary (enqueue or worker start). Mutates envelope.metadata.
    """
    if envelope.metadata is None:
        envelope.metadata = {}
    req = envelope.metadata.get(TRACE_REQUEST_ID)
    if not req:
        req = str(uuid.uuid4())
        envelope.metadata[TRACE_REQUEST_ID] = req
    envelope.metadata.setdefault(TRACE_TURN_ID, req)
    envelope.metadata.setdefault(TRACE_ENTRYPOINT, envelope.type)
    envelope.metadata.setdefault(TRACE_SOURCE_TYPE, envelope.type)


def envelope_to_user_query(envelope: InputEnvelope) -> str:
    """Extract user_query string from envelope for orchestrator.execute."""
    p = envelope.payload
    if envelope.type == "message":
        return (p.get("user_query") or "").strip()
    if envelope.type in ("cron", "agent_to_agent"):
        return (p.get("user_query") or "").strip()
    if envelope.type == "webhook":
        body = p.get("body") or {}
        return (body.get("user_query") or body.get("message") or "").strip()
    if envelope.type == "heartbeat":
        return (p.get("user_query") or "").strip()
    return ""

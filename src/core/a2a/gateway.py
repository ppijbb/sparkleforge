"""A2A protocol gateway exposing agent card, JSON-RPC, and SSE multi-turn task streams over CoordinatorNode."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from typing import Any, AsyncGenerator, Dict, List, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from sse_starlette.sse import EventSourceResponse

from src import __version__
from src.core.a2a.models import AgentCard, TaskArtifactUpdateEvent, TaskStatusUpdateEvent
from src.core.guard.invocation_gateway import InvocationGateway, InvocationKind, get_invocation_gateway
from src.core.session.coordinator import CoordinatorNode, get_coordinator_node

__all__ = ["app", "create_a2a_app"]


def _get_agent_card() -> AgentCard:
    return AgentCard(
        name="SparkleForge",
        description="Autonomous multi-agent research and engineering orchestration engine.",
        version=__version__,
        url=os.getenv("SPARKLEFORGE_A2A_URL", "http://localhost:8503"),
        capabilities=["research", "code_review", "system_cleanup", "semantic_search", "security_scan"],
        authentication={"scheme": "Bearer"},
        streaming=True,
        pushNotifications=False,
    )


def _check_auth(request: Request) -> JSONResponse | None:
    expected = os.environ.get("STATUS_API_TOKEN") or os.environ.get("A2A_API_TOKEN")
    if not expected:
        return JSONResponse({"error": "Service temporarily unavailable: auth not configured"}, status_code=503)
    got = request.headers.get("authorization", "")
    if not got.startswith("Bearer ") or not secrets.compare_digest(got[len("Bearer "):], expected):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


async def agent_card_endpoint(request: Request) -> JSONResponse:
    card = _get_agent_card()
    return JSONResponse(card.to_dict())


async def message_send(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    context_id = body.get("contextId") or body.get("context_id")
    task_id = body.get("taskId") or body.get("task_id")
    message = body.get("message") or body.get("prompt")
    actor = body.get("actor", "external-peer")

    if not message or not isinstance(message, str):
        return JSONResponse({"error": "'message' or 'prompt' (string) is required"}, status_code=400)

    gateway: InvocationGateway = get_invocation_gateway()
    decision = gateway.authorize(
        kind=InvocationKind.AGENT_DELEGATION,
        actor=actor,
        target="CoordinatorNode.delegate_task",
        description=message,
        required_capability="execute_shell",
    )
    if not decision.allowed:
        return JSONResponse({"error": f"forbidden by guard policy: {decision.reason}"}, status_code=403)

    coordinator: CoordinatorNode = get_coordinator_node()
    try:
        res = await coordinator.delegate_task(
            task_id=task_id,
            context_id=context_id,
            payload={"message": message, "actor": actor},
        )
    except Exception as e:
        return JSONResponse({"error": f"task delegation failed: {e}"}, status_code=500)

    return JSONResponse(res)


async def message_stream(request: Request) -> EventSourceResponse | JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err

    try:
        body = await request.json()
    except Exception:
        # Allow GET query params or fallback for SSE
        body = dict(request.query_params)

    context_id = body.get("contextId") or body.get("context_id") or "ctx_default"
    task_id = body.get("taskId") or body.get("task_id") or f"task_{os.urandom(4).hex()}"
    message = body.get("message") or body.get("prompt") or "Run task stream"
    actor = body.get("actor", "external-peer")

    gateway: InvocationGateway = get_invocation_gateway()
    decision = gateway.authorize(
        kind=InvocationKind.AGENT_DELEGATION,
        actor=actor,
        target="CoordinatorNode.delegate_task_stream",
        description=message,
        required_capability="execute_shell",
    )
    if not decision.allowed:
        return JSONResponse({"error": f"forbidden by guard policy: {decision.reason}"}, status_code=403)

    coordinator: CoordinatorNode = get_coordinator_node()

    async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        yield {
            "event": "status",
            "data": json.dumps(
                TaskStatusUpdateEvent(
                    taskId=task_id,
                    contextId=context_id,
                    status="submitted",
                    message="Task received and submitted to CoordinatorNode.",
                ).to_dict()
            ),
        }
        await asyncio.sleep(0.1)
        yield {
            "event": "status",
            "data": json.dumps(
                TaskStatusUpdateEvent(
                    taskId=task_id,
                    contextId=context_id,
                    status="working",
                    message="Worker agent active and processing.",
                ).to_dict()
            ),
        }

        try:
            res = await coordinator.delegate_task(
                task_id=task_id,
                context_id=context_id,
                payload={"message": message, "actor": actor},
            )
            yield {
                "event": "artifact",
                "data": json.dumps(
                    TaskArtifactUpdateEvent(
                        taskId=task_id,
                        contextId=context_id,
                        artifact={"result": res},
                    ).to_dict()
                ),
            }
            yield {
                "event": "status",
                "data": json.dumps(
                    TaskStatusUpdateEvent(
                        taskId=task_id,
                        contextId=context_id,
                        status="completed",
                        message="Task execution completed successfully.",
                    ).to_dict()
                ),
            }
        except Exception as e:
            yield {
                "event": "status",
                "data": json.dumps(
                    TaskStatusUpdateEvent(
                        taskId=task_id,
                        contextId=context_id,
                        status="failed",
                        message=str(e),
                    ).to_dict()
                ),
            }

    return EventSourceResponse(event_generator())


async def get_task_endpoint(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err

    task_id = request.path_params["task_id"]
    coordinator: CoordinatorNode = get_coordinator_node()
    status = coordinator.get_task_status(task_id)
    if not status:
        return JSONResponse({"error": "task not found"}, status_code=404)
    return JSONResponse(status)


async def cancel_task_endpoint(request: Request) -> JSONResponse:
    if (err := _check_auth(request)) is not None:
        return err

    task_id = request.path_params["task_id"]
    coordinator: CoordinatorNode = get_coordinator_node()
    success = await coordinator.cancel_task(task_id)
    if not success:
        return JSONResponse({"error": "task not found or unable to cancel"}, status_code=404)
    return JSONResponse({"taskId": task_id, "status": "cancelled"})


def create_a2a_app() -> Starlette:
    return Starlette(
        routes=[
            Route("/.well-known/agent.json", agent_card_endpoint, methods=["GET"]),
            Route("/message/send", message_send, methods=["POST"]),
            Route("/message/stream", message_stream, methods=["POST", "GET"]),
            Route("/tasks/{task_id}", get_task_endpoint, methods=["GET"]),
            Route("/tasks/{task_id}/cancel", cancel_task_endpoint, methods=["POST"]),
        ]
    )


app = create_a2a_app()

"""Data classes and schemas for the A2A protocol agent cards and SSE event payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentCard:
    name: str
    description: str
    version: str
    url: str
    capabilities: List[str] = field(default_factory=list)
    authentication: Dict[str, Any] = field(default_factory=lambda: {"scheme": "Bearer"})
    streaming: bool = True
    pushNotifications: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "url": self.url,
            "capabilities": self.capabilities,
            "authentication": self.authentication,
            "streaming": self.streaming,
            "pushNotifications": self.pushNotifications,
        }


@dataclass
class TaskStatusUpdateEvent:
    taskId: str
    contextId: str
    status: str  # "submitted" | "working" | "input-required" | "completed" | "failed"
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taskId": self.taskId,
            "contextId": self.contextId,
            "status": self.status,
            "message": self.message,
        }


@dataclass
class TaskArtifactUpdateEvent:
    taskId: str
    contextId: str
    artifact: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taskId": self.taskId,
            "contextId": self.contextId,
            "artifact": self.artifact,
        }

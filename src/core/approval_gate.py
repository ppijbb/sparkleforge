"""Approval gate for risky tools (OAC/Cline-style: propose then execute with user approval)."""

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def get_approval_gate_config():
    """Return ApprovalGateConfig from researcher config."""
    from src.core.researcher_config import get_researcher_system_config

    config = get_researcher_system_config()
    return config.approval_gate


def is_risky_tool(tool_name: str, parameters: Dict[str, Any] | None = None) -> bool:
    """True if tool name matches risky patterns (file write, shell, fetch, etc.)."""
    gate = get_approval_gate_config()
    name_lower = (tool_name or "").lower()
    for pattern in gate.risky_tool_patterns:
        if pattern.lower() in name_lower:
            return True
    return False


def requires_approval(
    tool_name: str, parameters: Dict[str, Any] | None = None
) -> bool:
    """True if approval gate is enabled and this tool is risky."""
    # Autonomy mode에서는 사용자 승인 질문을 만들지 않음.
    # 안전성은 allowlist/샌드박스/보안 정책에서 담당.
    if (
        os.getenv("APPROVAL_GATE_DISABLED_IN_AUTONOMY", "false")
        .strip()
        .lower()
        in {"1", "true", "yes", "y", "on"}
    ):
        return False
    gate = get_approval_gate_config()
    if not gate.enabled:
        return False
    return is_risky_tool(tool_name, parameters or {})

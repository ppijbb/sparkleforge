"""Forge Master Dispatch Tool - Agent-Callable CLI Agent Selection

`route_task`(휴리스틱)이나 숨겨진 LLM 호출이 대신 골라주는 게 아니라, 실제로
작업을 수행하는 에이전트(agent_loop의 tool-call 턴)가 `dispatch_to_cli_agent`
도구를 직접 호출해 어떤 외부 CLI 에이전트를 쓸지 스스로 판단하도록 노출하는
모듈. 이 실행기는 라우팅/랭킹을 하지 않고, 골라준 agent_name 그대로 실행한
결과만 돌려준다 (실패해도 다른 에이전트로 자동 전환하지 않음 - 그 판단도
호출한 에이전트의 몫).
"""

from typing import Any, Dict

from .controller import ForgeMasterController
from .router import ForgeMasterRouter


def _build_agent_strengths_summary() -> str:
    parts = []
    for agent, info in ForgeMasterRouter.CAPABILITY_MATRIX.items():
        strengths = ", ".join(info.get("strengths", []))
        parts.append(f"{agent} ({info.get('cost_tier', 'unknown')} cost): {strengths}")
    return "; ".join(parts)


DISPATCH_TO_CLI_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "agent_name": {
            "type": "string",
            "enum": list(ForgeMasterRouter.CAPABILITY_MATRIX.keys()),
            "description": (
                "Which external CLI coding agent to dispatch this task to. Choose "
                "based on actual fit for THIS specific task, not habit or cost - "
                "each agent's strengths and relative cost tier: "
                + _build_agent_strengths_summary()
            ),
        },
        "task_query": {
            "type": "string",
            "description": "The task to hand off to the chosen CLI agent.",
        },
        "context": {
            "type": "string",
            "description": "Optional extra context (e.g. relevant diff, prior findings).",
            "default": "",
        },
    },
    "required": ["agent_name", "task_query"],
}


async def _dispatch_to_cli_agent_tool(
    agent_name: str,
    task_query: str,
    context: str = "",
) -> Dict[str, Any]:
    """Execute a task with the specific CLI agent the caller chose.

    No routing, ranking, or cross-agent fallback happens here - if this
    fails, the caller decides (via its own reasoning) whether to call this
    tool again with a different agent_name.
    """
    controller = ForgeMasterController()
    return await controller.execute_task_with_master_control(
        task_query=task_query,
        context=context or None,
        preferred_agent=agent_name,
    )


def register_forge_master_dispatch_tool() -> None:
    """Register `dispatch_to_cli_agent` into the shared tool registry."""
    from src.core.tools.registry import ToolCategory, ToolMetadata, registry

    registry.register(
        ToolMetadata(
            name="dispatch_to_cli_agent",
            description=(
                "Hand off a coding task to one specific external CLI agent "
                "(claude_code, codex, gemini_cli, hermes, open_code, cline_cli). "
                "You must choose agent_name yourself based on the task and each "
                "agent's described strengths - this tool does not pick or switch "
                "agents for you, and will not silently retry with a different "
                "agent on failure. If it fails, decide whether to call it again "
                "with a different agent_name."
            ),
            parameters=DISPATCH_TO_CLI_AGENT_PARAMETERS,
            category=ToolCategory.CODE,
            tags=["forge_master", "cli_agent", "dispatch"],
            source="local",
        ),
        _dispatch_to_cli_agent_tool,
        _dispatch_to_cli_agent_tool,
    )

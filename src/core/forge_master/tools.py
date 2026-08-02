"""Forge Master Dispatch Tool - Agent-Callable CLI Agent Batch Dispatch

`route_task`(휴리스틱)이나 숨겨진 LLM 호출이 대신 골라주는 게 아니라, 실제로
작업을 수행하는 에이전트(agent_loop의 tool-call 턴)가 `dispatch_batch_to_forge_master`
도구를 직접 호출해 태스크 묶음을 넘기고, 태스크별로 어떤 외부 CLI 에이전트를
쓸지 스스로 판단하도록 노출하는 모듈. 각 태스크는 `ForgeMasterController`의
기존 라우팅/세션/적대적 검증 파이프라인을 그대로 거치며, asyncio.gather로만
동시 실행된다 (실패해도 다른 에이전트로 자동 전환하지 않음 - 그 판단도
호출한 에이전트의 몫). 서비스/에이전트 계층엔 이 tool 하나만 노출되고,
개별 CLI 에이전트(claude_code, codex, gemini_cli, ...)는 forgemaster 뒤에 숨는다.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

from .controller import ForgeMasterController
from .router import ForgeMasterRouter

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENCY = 5


def _build_agent_strengths_summary() -> str:
    parts = []
    for agent, info in ForgeMasterRouter.CAPABILITY_MATRIX.items():
        strengths = ", ".join(info.get("strengths", []))
        parts.append(f"{agent} ({info.get('cost_tier', 'unknown')} cost): {strengths}")
    return "; ".join(parts)


DISPATCH_BATCH_TO_FORGE_MASTER_PARAMETERS = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "enum": list(ForgeMasterRouter.CAPABILITY_MATRIX.keys()),
                        "description": (
                            "Which external CLI coding agent to dispatch this task to. "
                            "Choose based on actual fit for THIS specific task, not habit "
                            "or cost - each agent's strengths and relative cost tier: "
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
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "0-based indices of other tasks IN THIS SAME BATCH that must "
                            "finish first (e.g. a task that fixes review feedback on an "
                            "earlier task's output). Omit for tasks with no ordering "
                            "requirement - most tasks don't need this."
                        ),
                        "default": [],
                    },
                },
                "required": ["agent_name", "task_query"],
            },
            "description": (
                "Batch of tasks. Each task is routed, executed, and adversarially "
                "audited independently (own retries), all running concurrently up "
                "to max_concurrency. Tasks sharing the same agent_name share one "
                "session for the batch (context continuity), closed when the batch "
                "finishes. Tasks with no 'dependencies' all run at once (gated only "
                "by max_concurrency); tasks that declare dependencies run in "
                "dependency-ordered waves instead."
            ),
        },
        "max_concurrency": {
            "type": "integer",
            "description": "Cap on how many tasks run at once against the CLI agent fleet.",
            "default": DEFAULT_MAX_CONCURRENCY,
        },
    },
    "required": ["tasks"],
}


async def _run_batch_in_waves(
    tasks: List[Dict[str, Any]], run_one
) -> List[Any]:
    """Execute tasks in dependency-ordered waves via the existing TaskQueue.

    Only called when at least one task declares `dependencies` - reuses
    TaskQueue's dependency graph / parallel-group logic instead of
    reinventing wave scheduling. Each wave still runs through `run_one`,
    which holds the same concurrency semaphore as the no-dependency path.
    """
    from src.core.task_queue import TaskQueue

    queue = TaskQueue()
    queue.add_tasks(
        [
            {
                "task_id": str(i),
                "dependencies": [str(d) for d in (task.get("dependencies") or [])],
            }
            for i, task in enumerate(tasks)
        ]
    )

    results: List[Any] = [None] * len(tasks)
    while queue.has_pending_tasks():
        wave = queue.get_next_task_group()
        if not wave:
            # Circular or out-of-range dependency - nothing left is resolvable.
            for i in range(len(tasks)):
                if results[i] is None:
                    results[i] = RuntimeError(
                        "unresolved dependency in batch (cycle or bad index)"
                    )
            break

        wave_results = await asyncio.gather(
            *(run_one(tasks[int(task_id)]) for task_id in wave), return_exceptions=True
        )
        for task_id, result in zip(wave, wave_results):
            results[int(task_id)] = result
            queue.mark_completed(task_id)

    return results


async def _dispatch_batch_to_forge_master_tool(
    tasks: List[Dict[str, Any]],
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> Dict[str, Any]:
    """Execute a batch of tasks, each through the full ForgeMaster pipeline.

    No cross-task coordination or cross-agent fallback happens here - each
    task keeps the same "no auto-switch on failure" contract as a single
    dispatch; only concurrency (and, when declared, dependency ordering)
    is new.
    """
    controller = ForgeMasterController()
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    # Batch-scoped session grouping: tasks sharing an agent_name reuse one
    # session instead of each spinning up its own, so same-agent tasks in a
    # batch keep context continuity. Persistent for the batch's duration
    # only - closed in the finally below, not left to leak past this call.
    session_ids_by_agent: Dict[str, str] = {}
    for task in tasks:
        agent_name = task.get("agent_name")
        if agent_name and agent_name not in session_ids_by_agent:
            session = controller.session_manager.create_session(
                agent_name=agent_name, is_persistent=True
            )
            session_ids_by_agent[agent_name] = session.session_id

    async def _run_one(task: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            return await controller.execute_task_with_master_control(
                task_query=task["task_query"],
                context=task.get("context") or None,
                preferred_agent=task.get("agent_name"),
                session_id=session_ids_by_agent.get(task.get("agent_name")),
            )

    try:
        if any(task.get("dependencies") for task in tasks):
            raw_results = await _run_batch_in_waves(tasks, _run_one)
        else:
            raw_results = await asyncio.gather(
                *(_run_one(task) for task in tasks), return_exceptions=True
            )
    finally:
        for session_id in session_ids_by_agent.values():
            controller.session_manager.close_session(session_id)

    results = []
    for task, result in zip(tasks, raw_results):
        if isinstance(result, Exception):
            results.append(
                {
                    "success": False,
                    "error": str(result),
                    "task_query": task.get("task_query", ""),
                    "agent_used": task.get("agent_name"),
                }
            )
        else:
            results.append(result)

    _log_batch_manifest(results)

    return {
        "success": all(r.get("success") for r in results),
        "total": len(results),
        "succeeded": sum(1 for r in results if r.get("success")),
        "results": results,
    }


def _log_batch_manifest(results: List[Dict[str, Any]]) -> None:
    """One structured summary line per batch call, to the log file only.

    Without this, a batch's outcome is scattered across interleaved
    per-task retry/audit log lines from controller.py - nothing surfaces
    a single "what happened in this batch" view. Never printed to stdout;
    callers (REPL, agent loop) decide what the user actually sees.
    """
    manifest = [
        {
            "agent_used": r.get("agent_used") or r.get("last_agent_used"),
            "success": r.get("success", False),
            "master_verdict": r.get("master_verdict"),
            "skepticism_score": (r.get("adversarial_audit") or {}).get("skepticism_score"),
        }
        for r in results
    ]
    logger.info("forge_master batch manifest: %s", json.dumps(manifest, ensure_ascii=False))


def register_forge_master_dispatch_tool() -> None:
    """Register `dispatch_batch_to_forge_master` into the shared tool registry."""
    from src.core.tools.registry import ToolCategory, ToolMetadata, registry

    registry.register(
        ToolMetadata(
            name="dispatch_batch_to_forge_master",
            description=(
                "Hand off a batch of coding tasks to ForgeMaster, which routes "
                "each one to an external CLI agent (claude_code, codex, "
                "gemini_cli, hermes, open_code, cline_cli) and runs them "
                "concurrently. You must choose agent_name yourself per task "
                "based on the task and each agent's described strengths - this "
                "tool does not pick or switch agents for you, and will not "
                "silently retry a failed task with a different agent. If a "
                "task fails, decide whether to re-dispatch it with a different "
                "agent_name."
            ),
            parameters=DISPATCH_BATCH_TO_FORGE_MASTER_PARAMETERS,
            # UTILITY (not CODE): the hub's local-tool dispatcher special-cases
            # CODE to always run through the generic _execute_code_tool sandbox
            # (expects code/language params), never through this tool's own
            # registered executor. UTILITY is the category scheduler/security's
            # local pass-through tools already use to get routed straight to
            # registry.execute() instead.
            category=ToolCategory.UTILITY,
            tags=["forge_master", "cli_agent", "dispatch", "batch"],
            source="local",
        ),
        _dispatch_batch_to_forge_master_tool,
        _dispatch_batch_to_forge_master_tool,
    )

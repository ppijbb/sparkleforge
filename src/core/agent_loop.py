import asyncio
import json
import re
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.core.llm_manager import ModelResult, MultiModelOrchestrator, TaskType
from src.core.mcp_integration import UniversalMCPHub, get_mcp_hub

logger = logging.getLogger(__name__)


HEAT_SOFT_DEADLINE_RATIO = 0.85


def _summarize_tool_result(tool_result: Dict[str, Any]) -> str:
    """Best-effort one-line description of a successful tool_results entry for the Heat report."""
    for key in ("summary", "message", "output", "content"):
        value = tool_result.get(key)
        if isinstance(value, str) and value:
            return value[:200]
    data = tool_result.get("data")
    if data is not None:
        return str(data)[:200]
    return "completed"


@dataclass
class IterationBudget:
    """Hermes-style iteration budget tracker.

    heat_seconds is an optional wall-clock time budget ("Heat", issue #585)
    layered on top of the iteration count. At HEAT_SOFT_DEADLINE_RATIO of the
    budget, the loop stops starting new tool-calling iterations and returns a
    wrap-up report instead of either an abrupt iteration-budget cutoff or
    risking overshooting heat_seconds mid-iteration.
    """

    max_iterations: int = 90
    current_iteration: int = 0
    start_time: float = field(default_factory=time.time)
    heat_seconds: float | None = None

    def consume(self):
        self.current_iteration += 1
        if self.current_iteration > self.max_iterations:
            raise RuntimeError(
                f"Iteration budget exceeded: {self.current_iteration}/{self.max_iterations}"
            )

    @property
    def remaining(self):
        return self.max_iterations - self.current_iteration

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def heat_soft_expired(self) -> bool:
        return self.heat_seconds is not None and self.elapsed >= self.heat_seconds * HEAT_SOFT_DEADLINE_RATIO

    @property
    def heat_hard_expired(self) -> bool:
        return self.heat_seconds is not None and self.elapsed >= self.heat_seconds


class AgentLoop:
    """SparkleForge Autonomous Agent Loop (Phase 1).

    This class implements the 'Tool Calling Until Completion' pattern from Hermes.
    """

    # Class-level defaults so instances built via AgentLoop.__new__() (as some
    # lightweight unit tests do, to skip the heavy real __init__) still have
    # these attributes — accessed as "may be None" throughout, same as
    # intent_guardrail already was before this class existed.
    mode_controller = None
    method_resolver = None
    intent_guardrail = None

    AUTONOMOUS_PROBLEM_SOLVING_CONTRACT = """
Autonomous problem-solving contract:
- Operate as a self-directed execution agent, not a conversational assistant.
- Do not ask the user for clarification while tools or reasonable assumptions can move the task forward.
- If inputs are ambiguous, choose the most conservative useful interpretation and proceed.
- Use tools iteratively until the task is solved, a validated partial result is available, or a hard blocker is reached.
- If a tool fails, record the failure, try a different viable tool or approach, and continue.
- Final output must be the best available answer/result, with assumptions and hard blockers stated briefly.
"""

    def __init__(self, orchestrator: MultiModelOrchestrator | None = None):
        from src.core.llm_manager import MultiModelOrchestrator

        self.orchestrator = orchestrator or MultiModelOrchestrator()
        self._plan_first = False
        self.mcp_hub: UniversalMCPHub = get_mcp_hub()

        # Phase 3: Context Compression
        from src.core.context_compressor import ContextCompressor

        self.compressor = ContextCompressor(self.orchestrator)

        # Phase 6: Persistent Memory
        from src.core.memory.persistent import PersistentMemory

        from src.core.anvil.method_resolver import MethodResolver
        from src.core.anvil.mode_controller import ExecutionMode, ModeController

        self.memory = PersistentMemory()
        self.mode_controller = ModeController(plan_first=self._plan_first)
        self.method_resolver = MethodResolver()
        self.intent_guardrail = None

    async def run_conversation(
        self,
        messages: List[Dict[str, Any]],
        task_type: TaskType = TaskType.RESEARCH,
        max_iterations: int = 20,
        system_message: str | None = None,
        heat_seconds: float | None = None,
    ) -> Dict[str, Any]:
        """Runs the autonomous loop.

        heat_seconds: optional wall-clock time budget ("Heat", issue #585).
        When set, the loop stops starting new iterations once
        HEAT_SOFT_DEADLINE_RATIO of the budget has elapsed and returns a
        wrap-up report summarizing what was completed/failed/remaining,
        instead of running until max_iterations or being cut off mid-task.
        """
        from src.core.error_classifier import ErrorCategory, ErrorClassifier

        budget = IterationBudget(max_iterations=max_iterations, heat_seconds=heat_seconds)
        tool_results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        tool_calls_count = 0

        # Ensure MCP is initialized
        try:
            await self.mcp_hub.initialize_mcp()
        except Exception as e:
            logger.warning("[AgentLoop] MCP initialization failed: %s", e)
            errors.append({"type": "mcp_initialization_failed", "message": str(e)})

        # Get tool schemas in OpenAI format
        tools = self._get_openai_tools()

        system_message = self._build_autonomous_system_message(system_message)

        # Phase 6: Load persistent memory
        mem_context = self.memory.get_context_block()
        if mem_context:
            system_message = (system_message or "") + "\n\n" + mem_context

        # Prepare history
        history = list(messages)
        retry_count = 0
        max_retries = 3
        tool_capable_waits = 0

        while budget.remaining > 0 and not budget.heat_hard_expired:
            if budget.heat_soft_expired:
                logger.info(
                    "[AgentLoop] Heat soft deadline reached (%.0fs elapsed of %.0fs budget); "
                    "wrapping up instead of starting a new iteration",
                    budget.elapsed,
                    budget.heat_seconds,
                )
                return self._build_result(
                    success=True,
                    content=self._build_heat_wrap_up_content(tool_results, errors),
                    iterations=budget.current_iteration,
                    history=history,
                    metadata={
                        "heat_expired": True,
                        "heat_report": self._build_heat_report(budget, tool_results, errors),
                    },
                    tool_calls_count=tool_calls_count,
                    tool_results=tool_results,
                    errors=errors,
                )

            budget.consume()
            logger.info(f"[AgentLoop] Iteration {budget.current_iteration}/{max_iterations}")
            self._apply_mode_to_messages(history)
            await self._guard_intent(history)

            # Phase 3: Compress context if needed
            history = await self.compressor.compress_if_needed(history)

            # Step 1: Call LLM with Resilience (Phase 4)
            try:
                last_msg = history[-1]["content"] if history else ""
                prev_msgs = history[:-1] if len(history) > 1 else []

                result: ModelResult = await self.orchestrator.execute_with_model(
                    prompt=last_msg,
                    task_type=task_type,
                    model_name=self._select_tool_capable_model(task_type) if tools else None,
                    system_message=system_message,
                    history_messages=prev_msgs,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                )

                # Success - reset retry count
                retry_count = 0

            except Exception as e:
                category = ErrorClassifier.classify(e)
                logger.warning(f"[AgentLoop] Error detected: {category.value} - {e}")

                if category == ErrorCategory.RETRYABLE and retry_count < max_retries:
                    retry_count += 1
                    wait_time = 2**retry_count
                    logger.info(f"Retrying in {wait_time}s... ({retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    budget.current_iteration -= 1  # Don't count retry as iteration
                    continue
                elif category == ErrorCategory.CONTEXT_LIMIT:
                    # Force compression and retry
                    logger.info("Context limit hit. Forcing EXTREME compression.")
                    history = await self.compressor.compress_by_summarization(history)
                    continue
                else:
                    # Fatal or max retries
                    errors.append(
                        {
                            "type": "model_execution_failed",
                            "category": category.value,
                            "message": str(e),
                        }
                    )
                    return self._build_result(
                        success=False,
                        content=f"Execution failed due to {category.value}: {str(e)}",
                        iterations=budget.current_iteration,
                        history=history,
                        metadata={"error_category": category.value},
                        tool_calls_count=tool_calls_count,
                        tool_results=tool_results,
                        errors=errors,
                        error=str(e),
                    )

            content = result.content
            metadata = result.metadata or {}
            tool_calls = self._normalize_tool_calls(metadata.get("tool_calls", []))
            tool_calls_count += len(tool_calls)

            # Add Assistant response to history
            # (잘린/불량 JSON 인자를 그대로 넣으면 이후 모든 API 호출이 400으로 오염됨)
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = self._sanitize_tool_calls_for_history(tool_calls)
            history.append(assistant_msg)

            if not tool_calls:
                # Tool 불가 provider로 폴백된 응답은 '완료'로 인정하지 않는다.
                # (예: NIM 429 → Gemini 폴백 시 파일 수정 없이 완료 보고하는 환각 방지)
                if tools and not self._is_tool_capable_result(result):
                    if tool_capable_waits < 3 and budget.remaining > 0:
                        tool_capable_waits += 1
                        history.pop()  # 폴백 응답은 히스토리에서 제거
                        logger.warning(
                            "[AgentLoop] Non-tool-capable model %s answered without tool calls; "
                            "waiting 35s for tool-capable provider (%d/3)",
                            result.model_used,
                            tool_capable_waits,
                        )
                        await asyncio.sleep(35)
                        continue
                    errors.append(
                        {
                            "type": "tool_capable_model_unavailable",
                            "message": f"Final answer produced by non-tool-capable model {result.model_used}",
                        }
                    )
                    metadata["tool_capable_model_unavailable"] = True
                # No tool calls, we are done
                if tools:
                    metadata.setdefault("tool_calling_disabled_reason", "model_returned_no_tool_calls")
                return self._build_result(
                    success=True,
                    content=content,
                    iterations=budget.current_iteration,
                    history=history,
                    metadata=metadata,
                    tool_calls_count=tool_calls_count,
                    tool_results=tool_results,
                    errors=errors,
                )

            # Step 3: Execute tools
            for tool_call in tool_calls:
                model_tool_name = tool_call.get("function", {}).get("name")
                tool_name = self._tool_alias_map().get(model_tool_name, model_tool_name)
                arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                arguments: Dict[str, Any]
                try:
                    arguments = (
                        json.loads(arguments_str)
                        if isinstance(arguments_str, str)
                        else arguments_str
                    )
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must decode to a JSON object")
                except json.JSONDecodeError:
                    tool_exec_result = {
                        "success": False,
                        "error": f"Invalid JSON tool arguments: {arguments_str}",
                        "tool_name": tool_name,
                    }
                    errors.append(
                        {
                            "type": "invalid_tool_arguments",
                            "tool_name": tool_name,
                            "message": tool_exec_result["error"],
                        }
                    )
                    self._append_tool_result(
                        history, tool_call, tool_name, tool_exec_result, tool_results
                    )
                    continue
                except ValueError as e:
                    tool_exec_result = {
                        "success": False,
                        "error": str(e),
                        "tool_name": tool_name,
                    }
                    errors.append(
                        {
                            "type": "invalid_tool_arguments",
                            "tool_name": tool_name,
                            "message": str(e),
                        }
                    )
                    self._append_tool_result(
                        history, tool_call, tool_name, tool_exec_result, tool_results
                    )
                    continue

                if not tool_name:
                    tool_exec_result = {
                        "success": False,
                        "error": "Tool call missing function.name",
                        "tool_name": None,
                    }
                    errors.append(
                        {"type": "missing_tool_name", "message": tool_exec_result["error"]}
                    )
                    self._append_tool_result(
                        history, tool_call, tool_name, tool_exec_result, tool_results
                    )
                    continue

                logger.info(f"[AgentLoop] Executing tool: {tool_name}")
                try:
                    tool_exec_result = await self.mcp_hub.execute_tool(tool_name, arguments)
                    if self.mode_controller:
                        self.mode_controller.record_success()
                    await self._record_resolved_capability(tool_name)
                except Exception as e:
                    logger.error(f"Tool execution failed: {tool_name} - {e}")
                    tool_exec_result = {"success": False, "error": str(e)}
                    errors.append(
                        {
                            "type": "tool_execution_failed",
                            "tool_name": tool_name,
                            "message": str(e),
                        }
                    )
                    if self.mode_controller:
                        self.mode_controller.record_failure()

                self._append_tool_result(history, tool_call, tool_name, tool_exec_result, tool_results)

        if budget.heat_hard_expired:
            # Safety net: a single iteration ran long enough to cross the hard
            # heat deadline before the soft-deadline check could catch it.
            logger.warning(
                "[AgentLoop] Heat hard deadline reached (%.0fs elapsed of %.0fs budget)",
                budget.elapsed,
                budget.heat_seconds,
            )
            return self._build_result(
                success=True,
                content=self._build_heat_wrap_up_content(tool_results, errors),
                iterations=budget.current_iteration,
                history=history,
                metadata={
                    "heat_expired": True,
                    "heat_hard_cutoff": True,
                    "error_category": "heat_hard_deadline_exceeded",
                    "heat_report": self._build_heat_report(budget, tool_results, errors),
                },
                tool_calls_count=tool_calls_count,
                tool_results=tool_results,
                errors=errors,
            )

        errors.append({"type": "iteration_budget_exceeded", "message": "Max iterations reached"})
        return self._build_result(
            success=False,
            content="Iteration budget exceeded.",
            iterations=budget.current_iteration,
            history=history,
            metadata={
                "error_category": "iteration_budget_exceeded",
                "iteration_exhausted": True,
            },
            tool_calls_count=tool_calls_count,
            tool_results=tool_results,
            errors=errors,
            error="Max iterations reached",
        )

    def _build_heat_report(
        self,
        budget: "IterationBudget",
        tool_results: List[Dict[str, Any]],
        errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Summarize what was completed/failed/remaining when a Heat time budget expires.

        Built from tool_results/errors -- the loop's own real, live-tracked
        execution record -- rather than any of the separate (and currently
        unpopulated) session/task-dashboard tracking systems elsewhere in the
        codebase.
        """
        completed = [
            {"tool": r.get("tool_name"), "summary": _summarize_tool_result(r)}
            for r in tool_results
            if r.get("success", True)
        ]
        failed = [
            {"tool": r.get("tool_name"), "error": r.get("error", "unknown error")}
            for r in tool_results
            if not r.get("success", True)
        ]

        if failed:
            next_action = (
                f"Investigate {len(failed)} failed tool call(s) before continuing "
                f"(most recent: {failed[-1]['tool']} -- {failed[-1]['error']})."
            )
        elif completed:
            next_action = (
                "Resume this goal with additional Heat time to continue past the "
                f"{len(completed)} step(s) already completed."
            )
        else:
            next_action = "No tool calls completed in this Heat window; resume with more time or a narrower goal."

        return {
            "elapsed_seconds": round(budget.elapsed, 1),
            "heat_budget_seconds": budget.heat_seconds,
            "iterations_used": budget.current_iteration,
            "completed": completed,
            "failed": failed,
            "errors": errors,
            "next_recommended_action": next_action,
        }

    def _build_heat_wrap_up_content(
        self, tool_results: List[Dict[str, Any]], errors: List[Dict[str, Any]]
    ) -> str:
        """Human-readable summary line used as the result's `content` when Heat expires."""
        completed_count = sum(1 for r in tool_results if r.get("success", True))
        failed_count = len(tool_results) - completed_count
        return (
            f"Heat time budget reached. Completed {completed_count} tool call(s)"
            + (f", {failed_count} failed" if failed_count else "")
            + ". See metadata.heat_report for a full breakdown."
        )

    def _append_tool_result(
        self,
        history: List[Dict[str, Any]],
        tool_call: Dict[str, Any],
        tool_name: str | None,
        tool_exec_result: Dict[str, Any],
        tool_results: List[Dict[str, Any]],
    ) -> None:
        """Record a tool execution in both model history and structured output."""
        if not isinstance(tool_exec_result, dict):
            tool_exec_result = {"success": True, "data": tool_exec_result}

        tool_exec_result.setdefault("tool_name", tool_name)
        tool_results.append(tool_exec_result)

        tool_result_str = json.dumps(tool_exec_result, ensure_ascii=False)
        pruned_result = self.compressor.prune_tool_output(tool_result_str)

        history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "name": tool_name,
                "content": pruned_result,
            }
        )

    def _build_result(
        self,
        *,
        success: bool,
        content: str,
        iterations: int,
        history: List[Dict[str, Any]],
        metadata: Dict[str, Any] | None = None,
        tool_calls_count: int = 0,
        tool_results: List[Dict[str, Any]] | None = None,
        errors: List[Dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> Dict[str, Any]:
        """Return the stable public result shape used by harness/orchestrator nodes."""
        result = {
            "success": success,
            "content": content,
            "metadata": metadata or {},
            "iterations": iterations,
            "history": history,
            "tool_calls_count": tool_calls_count,
            "tool_results": tool_results or [],
            "errors": errors or [],
        }
        if error:
            result["error"] = error
        return result

    @staticmethod
    def _sanitize_tool_calls_for_history(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """히스토리에 저장할 tool_calls의 arguments가 유효한 JSON 문자열임을 보장."""
        sanitized = []
        for tc in tool_calls:
            tc_copy = json.loads(json.dumps(tc))
            fn = tc_copy.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    fn["arguments"] = "{}"
            else:
                try:
                    fn["arguments"] = json.dumps(args)
                except (TypeError, ValueError):
                    fn["arguments"] = "{}"
            sanitized.append(tc_copy)
        return sanitized

    def _normalize_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """Convert provider-specific tool call objects into OpenAI-like dictionaries."""
        if not tool_calls:
            return []
        normalized = []
        for call in tool_calls:
            if isinstance(call, dict):
                normalized.append(call)
                continue

            function = getattr(call, "function", None)
            if function is not None:
                normalized.append(
                    {
                        "id": getattr(call, "id", None),
                        "type": getattr(call, "type", "function"),
                        "function": {
                            "name": getattr(function, "name", None),
                            "arguments": getattr(function, "arguments", "{}"),
                        },
                    }
                )
                continue

            name = getattr(call, "name", None)
            if name:
                normalized.append(
                    {
                        "id": getattr(call, "id", None),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": getattr(call, "arguments", "{}"),
                        },
                    }
                )
        return normalized


    TOOL_CAPABLE_PROVIDERS = ("nvidia", "openrouter", "groq", "openai")

    def _is_tool_capable_result(self, result: Any) -> bool:
        """응답을 생성한 모델이 tool_calls를 반환할 수 있는 provider인지 확인."""
        models = getattr(self.orchestrator, "models", {}) or {}
        config = models.get(getattr(result, "model_used", None))
        provider = getattr(config, "provider", None)
        if provider is None:
            provider = ((getattr(result, "metadata", None) or {}).get("provider"))
        return provider in self.TOOL_CAPABLE_PROVIDERS

    def _select_tool_capable_model(self, task_type: TaskType) -> str | None:
        """Prefer providers that can return OpenAI-compatible tool_calls (rate-limited 제외)."""
        models = getattr(self.orchestrator, "models", {}) or {}
        limited = getattr(self.orchestrator, "_is_provider_rate_limited", lambda _p: False)
        for provider in self.TOOL_CAPABLE_PROVIDERS:
            if limited(provider):
                continue
            for name, config in models.items():
                if getattr(config, "provider", None) != provider:
                    continue
                capabilities = getattr(config, "capabilities", []) or []
                if task_type in capabilities:
                    return name
        return None

    def _build_autonomous_system_message(self, system_message: str | None) -> str:
        if system_message:
            return f"{system_message.strip()}\n\n{self.AUTONOMOUS_PROBLEM_SOLVING_CONTRACT.strip()}"
        return self.AUTONOMOUS_PROBLEM_SOLVING_CONTRACT.strip()

    # Docker 샌드박스가 필요한 도구 (샌드박스 불가 시 노출 제외)
    SANDBOX_TOOLS = ("python_coder", "code_interpreter")

    def _sandbox_available(self) -> bool:
        """Docker 샌드박스 실행 가능 여부 (결과 캐시)."""
        if not hasattr(self, "_sandbox_ok"):
            try:
                from src.core.sandbox.docker_sandbox import get_sandbox

                sandbox = get_sandbox()
                runtime = getattr(sandbox.config, "runtime", None)
                allow_fallback = getattr(sandbox.config, "allow_default_runtime_fallback", False)
                import docker as _docker

                client = _docker.from_env()
                client.ping()
                runtimes = (client.info().get("Runtimes") or {}).keys()
                self._sandbox_ok = (not runtime) or (runtime in runtimes) or allow_fallback
            except Exception:
                self._sandbox_ok = False
            if not self._sandbox_ok:
                logger.warning(
                    "[AgentLoop] Docker sandbox unavailable; hiding sandbox tools %s",
                    self.SANDBOX_TOOLS,
                )
        return self._sandbox_ok

    def _get_openai_tools(self) -> List[Dict[str, Any]]:
        """Converts MCP tools to OpenAI tool format."""
        openai_tools = []
        alias_map: Dict[str, str] = {}
        used_aliases = set()
        registry_tools = getattr(getattr(self.mcp_hub, "registry", None), "tools", {}) or {}
        sandbox_ok = self._sandbox_available()

        for name, info in registry_tools.items():
            if not sandbox_ok and name in self.SANDBOX_TOOLS:
                continue
            alias = self._openai_tool_name(name)
            if alias in used_aliases:
                suffix = 2
                base = alias[:58]
                while f"{base}_{suffix}" in used_aliases:
                    suffix += 1
                alias = f"{base}_{suffix}"
            used_aliases.add(alias)
            alias_map[alias] = name

            parameters = getattr(info, "parameters", None) or {"type": "object", "properties": {}}
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            else:
                if "properties" not in parameters and parameters:
                    required_fields = [k for k, v in parameters.items() if isinstance(v, dict) and v.get("required")]
                    properties = {}
                    for k, v in parameters.items():
                        if isinstance(v, dict):
                            v_copy = dict(v)
                            v_copy.pop("required", None)
                            properties[k] = v_copy
                        else:
                            properties[k] = v
                    parameters = {
                        "type": "object",
                        "properties": properties,
                    }
                    if required_fields:
                        parameters["required"] = required_fields

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": alias,
                        "description": getattr(info, "description", ""),
                        "parameters": parameters,
                    },
                }
            )

        self._openai_tool_alias_map = alias_map
        return openai_tools

    def _tool_alias_map(self) -> Dict[str, str]:
        return getattr(self, "_openai_tool_alias_map", {})

    def _openai_tool_name(self, name: str) -> str:
        alias = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64].strip("_")
        return alias or "tool"

    # --- Anvil core wiring helpers ---

    def _apply_mode_to_messages(self, history: List[Dict[str, Any]]) -> None:
        """ModeController의 쓰기 차단 상태를 루프 컨텍스트에 반영."""
        if self.mode_controller is None or not self.mode_controller.is_write_blocked():
            return
        for msg in history:
            if isinstance(msg, dict) and msg.get("role") == "system":
                msg["content"] = (msg.get("content", "") or "") + (
                    "\n\n[ModeController] PLAN_FIRST 모드 - 계획 승인 전까지 쓰기 액션 차단됨."
                )
                break

    async def _guard_intent(self, history: List[Dict[str, Any]]) -> None:
        """IntentGuardrail로 최근 작업 요약의 의도 정렬을 주기적 진단."""
        if self.intent_guardrail is None:
            return
        if self.mode_controller is None:
            return
        step_index = len([m for m in history if m.get("role") == "tool"])
        if not self.intent_guardrail.should_check(step_index):
            return
        summary = " ".join(
            str(m.get("content", "")) for m in history[-3:] if isinstance(m, dict)
        )
        try:
            assessment = self.intent_guardrail.evaluate(summary)
        except Exception as e:
            logger.warning("[AgentLoop] IntentGuardrail evaluation failed: %s", e)
            return
        if self.intent_guardrail.needs_human_review() and self.mode_controller:
            self.mode_controller.on_intent_review_needed()

    async def _record_resolved_capability(self, capability: str) -> None:
        """MethodResolver를 통해 도구 capability 해결 시도를 기록."""
        if self.method_resolver is None:
            return
        resolved = await self.method_resolver.resolve(capability)
        if not resolved.resolved and self.mode_controller:
            self.mode_controller.on_unresolved_capability(capability)

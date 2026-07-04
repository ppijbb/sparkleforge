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


@dataclass
class IterationBudget:
    """Hermes-style iteration budget tracker."""

    max_iterations: int = 90
    current_iteration: int = 0
    start_time: float = field(default_factory=time.time)

    def consume(self):
        self.current_iteration += 1
        if self.current_iteration > self.max_iterations:
            raise RuntimeError(
                f"Iteration budget exceeded: {self.current_iteration}/{self.max_iterations}"
            )

    @property
    def remaining(self):
        return self.max_iterations - self.current_iteration


class AgentLoop:
    """SparkleForge Autonomous Agent Loop (Phase 1).

    This class implements the 'Tool Calling Until Completion' pattern from Hermes.
    """

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
        self.mcp_hub: UniversalMCPHub = get_mcp_hub()

        # Phase 3: Context Compression
        from src.core.context_compressor import ContextCompressor

        self.compressor = ContextCompressor(self.orchestrator)

        # Phase 6: Persistent Memory
        from src.core.memory.persistent import PersistentMemory

        self.memory = PersistentMemory()

    async def run_conversation(
        self,
        messages: List[Dict[str, Any]],
        task_type: TaskType = TaskType.RESEARCH,
        max_iterations: int = 20,
        system_message: str | None = None,
    ) -> Dict[str, Any]:
        """Runs the autonomous loop."""
        from src.core.error_classifier import ErrorCategory, ErrorClassifier

        budget = IterationBudget(max_iterations=max_iterations)
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

        while budget.remaining > 0:
            budget.consume()
            logger.info(f"[AgentLoop] Iteration {budget.current_iteration}/{max_iterations}")

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
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
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

                self._append_tool_result(history, tool_call, tool_name, tool_exec_result, tool_results)

        errors.append({"type": "iteration_budget_exceeded", "message": "Max iterations reached"})
        return self._build_result(
            success=False,
            content="Iteration budget exceeded.",
            iterations=budget.current_iteration,
            history=history,
            metadata={"error_category": "iteration_budget_exceeded"},
            tool_calls_count=tool_calls_count,
            tool_results=tool_results,
            errors=errors,
            error="Max iterations reached",
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
        """Prefer providers that can return OpenAI-compatible tool_calls."""
        models = getattr(self.orchestrator, "models", {}) or {}
        preferred = ("nvidia", "openrouter", "groq", "openai")
        for provider in preferred:
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

    def _get_openai_tools(self) -> List[Dict[str, Any]]:
        """Converts MCP tools to OpenAI tool format."""
        openai_tools = []
        alias_map: Dict[str, str] = {}
        used_aliases = set()
        registry_tools = getattr(getattr(self.mcp_hub, "registry", None), "tools", {}) or {}

        for name, info in registry_tools.items():
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

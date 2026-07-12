import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

from src.core.mcp_integration import ToolCategory, execute_tool
from src.core.orchestrator.base_node import BaseNode
from src.core.orchestrator.delegation import delegate_to_agent
from src.core.orchestrator.state import ResearchState

logger = logging.getLogger(__name__)

# Playwright controller lazy import
_playwright_controller = None


def _get_browser_controller():
    """PlaywrightController 인스턴스를 lazy하게 가져옵니다."""
    global _playwright_controller
    if _playwright_controller is None:
        from src.automation.browser_manager import get_playwright_controller

        _playwright_controller = get_playwright_controller()
    return _playwright_controller


class ExecutionNode(BaseNode):
    """Handler for research execution and resource allocation."""

    def __init__(self, llm_config, agent_config, research_depth, streaming_manager):
        self.llm_config = llm_config
        self.agent_config = agent_config
        self.research_depth = research_depth
        self.streaming_manager = streaming_manager

    async def adaptive_supervisor(self, state: ResearchState) -> ResearchState:
        """Adaptive Supervisor (혁신 1)."""
        logger.info("🎯 Adaptive Supervisor allocating resources")

        complexity_raw = state.get("complexity_score", 5.0)
        if isinstance(complexity_raw, dict):
            complexity = complexity_raw.get("score", complexity_raw.get("value", 5.0))
        else:
            complexity = float(complexity_raw)

        available_budget = self.llm_config.budget_limit

        allocated_researchers = min(
            max(int(complexity), self.agent_config.min_researchers),
            self.agent_config.max_researchers,
            int(available_budget / 10),
        )

        priority_queue = self._create_priority_queue(state)
        quality_threshold = self.agent_config.quality_threshold

        logger.info(f"🧠 Complexity Score: {complexity}")
        logger.info(f"👥 Allocated Researchers: {allocated_researchers}")
        logger.info(f"📋 Priority Queue Size: {len(priority_queue)}")

        state.update(
            {
                "allocated_researchers": allocated_researchers,
                "priority_queue": priority_queue,
                "quality_threshold": quality_threshold,
                "current_step": "execute_research",
                "innovation_stats": {
                    **state.get("innovation_stats", {}),
                    "allocated_researchers": allocated_researchers,
                    "complexity_score": complexity,
                    "priority_queue_size": len(priority_queue),
                },
            }
        )
        return state

    async def delegate_to_agent(
        self,
        state: ResearchState,
        role: str,
        task: Dict[str, Any],
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Runtime delegation to an agent role outside the static graph DAG.

        Exposed to `adaptive_supervisor`/`execute_research` so either node can
        reach a role (e.g. `validation_agent`, `codebase_agent`) that has no
        static edge, instead of requiring a new hardcoded node. See
        `src.core.orchestrator.delegation` for the depth guard, journaling,
        and per-role adapters (`DELEGATION_REGISTRY`).
        """
        return await delegate_to_agent(state, role, task, context, delegator_id="execution_node")

    async def execute_research(self, state: ResearchState) -> ResearchState:
        """Run planned research tasks through the Hermes-style autonomous tool loop."""
        self._log_node_input("execute_research", state)
        logger.info("⚙️ Thinking: Executing research tasks with Hermes tool loop")

        tasks = state.get("planned_tasks", []) or [
            {
                "id": "task_1",
                "name": "Research request",
                "description": state.get("user_request", ""),
                "type": "research",
            }
        ]
        execution_plan = state.get("execution_plan", {})

        use_parallel = (
            execution_plan.get("strategy") in ["parallel", "hybrid"]
            and len(tasks) > 1
            and self.agent_config.max_concurrent_research_units > 1
        )

        max_iterations = self._get_hermes_max_iterations(state)
        concurrency = self.agent_config.max_concurrent_research_units if use_parallel else 1
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self._execute_task_with_hermes(task, state, max_iterations)

        if use_parallel:
            logger.info("🚀 Using Hermes parallel execution")
            execution_results = await asyncio.gather(*(run_task(task) for task in tasks))
        else:
            logger.info("📝 Using Hermes sequential execution")
            execution_results = []
            for task in tasks:
                execution_results.append(await run_task(task))

        streaming_data = [self._streaming_event_from_result(result) for result in execution_results]

        # Depth Adjustment (Progressive Deepening)
        self._adjust_depth_if_needed(state, tasks, execution_results)

        tool_calls_count = sum(r.get("agent_loop_metadata", {}).get("tool_calls_count", 0) for r in execution_results)
        hermes_completed = len([r for r in execution_results if r.get("status") == "completed"])

        state.update(
            {
                "execution_results": execution_results,
                "streaming_data": streaming_data,
                "current_step": "hierarchical_compression",
                "research_iteration": state.get("research_iteration", 0) + 1,
                "innovation_stats": {
                    **state.get("innovation_stats", {}),
                    "tasks_executed": len(execution_results),
                    "parallel_execution_used": use_parallel,
                    "hermes_execution_used": True,
                    "hermes_tasks_completed": hermes_completed,
                    "tool_calls_count": tool_calls_count,
                },
            }
        )
        self._log_node_output("execute_research", state, {"tasks_executed": len(execution_results)})
        return state

    async def _execute_task_with_hermes(
        self, task: Dict[str, Any], state: ResearchState, max_iterations: int
    ) -> Dict[str, Any]:
        from src.core.agent_loop import AgentLoop
        from src.core.llm_manager import TaskType
        from src.core.prompt_builder import get_system_prompt

        task_id = self._task_id(task)
        task_name = task.get("name") or task.get("title") or task_id
        prompt = self._build_hermes_task_prompt(task, state)
        loop = AgentLoop()

        try:
            result = await loop.run_conversation(
                messages=[{"role": "user", "content": prompt}],
                task_type=TaskType.RESEARCH,
                max_iterations=max_iterations,
                system_message=get_system_prompt(
                    "researcher",
                    (
                        "Use available tools aggressively when useful. Do not ask the user for "
                        "clarification; make conservative assumptions, solve the task, and return "
                        "a concise result with evidence, assumptions, and limitations."
                    ),
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Hermes task execution failed for %s: %s", task_id, e)
            result = {
                "success": False,
                "content": "",
                "error": str(e),
                "iterations": 0,
                "tool_calls_count": 0,
                "tool_results": [],
                "errors": [{"type": "hermes_exception", "message": str(e)}],
                "metadata": {},
            }

        if result.get("success") and result.get("content"):
            return {
                "task_id": task_id,
                "task_name": task_name,
                "tool_used": "hermes_agent_loop",
                "result": result.get("content"),
                "status": "completed",
                "iterations": result.get("iterations", 0),
                "agent_loop_metadata": self._agent_loop_metadata(result),
            }

        legacy_result = await self._execute_task_with_legacy_tools(task)
        if not legacy_result.get("task_id"):
            legacy_result["task_id"] = task_id
        if not legacy_result.get("task_name"):
            legacy_result["task_name"] = task_name
        legacy_result["agent_loop_metadata"] = self._agent_loop_metadata(result)
        if result.get("error"):
            legacy_result.setdefault("hermes_error", result.get("error"))
        return legacy_result

    async def _execute_task_with_legacy_tools(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback for environments where the model/provider cannot drive tool calls."""
        task_success = False
        tool_attempts = []
        task_id = self._task_id(task)
        task_name = task.get("name") or task.get("title") or task_id

        try:
            tool_category = self._get_tool_category_for_task(task)

            if tool_category == ToolCategory.BROWSER:
                browser_result = await self._execute_browser_task(task)
                if browser_result:
                    return browser_result

            available_tools = self._get_available_tools_for_category(tool_category)

            for tool_name in available_tools:
                try:
                    params = self._generate_tool_parameters(task, tool_name)
                    if "__missing_required__" in params:
                        tool_attempts.append(
                            {"tool": tool_name, "success": False, "error": "Missing params"}
                        )
                        continue

                    tool_result = await execute_tool(tool_name, params)
                    tool_attempts.append(
                        {
                            "tool": tool_name,
                            "success": tool_result.get("success", False),
                            "execution_time": tool_result.get("execution_time", 0.0),
                        }
                    )

                    if tool_result.get("success", False) and self._validate_tool_result(
                        tool_result, task
                    ):
                        task_success = True
                        return {
                            "task_id": task_id,
                            "task_name": task_name,
                            "tool_used": tool_name,
                            "result": tool_result.get("data"),
                            "status": "completed",
                            "legacy_fallback_used": True,
                        }
                except Exception as e:
                    if isinstance(e, asyncio.CancelledError):
                        raise
                    logger.warning("Tool %s failed: %s", tool_name, e)
                    tool_attempts.append({"tool": tool_name, "success": False, "error": str(e)})

            if not task_success:
                return {
                    "task_id": task_id,
                    "task_name": task_name,
                    "status": "failed",
                    "error": f"All {len(available_tools)} tools failed",
                    "attempts": tool_attempts,
                    "legacy_fallback_used": True,
                }
        except Exception as e:
            if isinstance(e, asyncio.CancelledError):
                raise
            logger.error("Task execution error: %s", e)
            return {
                "task_id": task_id,
                "task_name": task_name,
                "status": "failed",
                "error": str(e),
                "attempts": tool_attempts,
                "legacy_fallback_used": True,
            }

    def _task_id(self, task: Dict[str, Any]) -> str:
        return str(task.get("id") or task.get("task_id") or task.get("name") or "task")

    def _build_hermes_task_prompt(self, task: Dict[str, Any], state: ResearchState) -> str:
        request = state.get("user_request", "")
        objectives = state.get("analyzed_objectives", [])
        preliminary = state.get("preliminary_research", {})
        return (
            "Execute this research task to completion as an autonomous problem-solving agent.\n"
            "Do not stop to ask for clarification. If details are missing, make the most "
            "conservative useful assumption, use tools to reduce uncertainty, and continue.\n"
            "Cite evidence from tool outputs when available.\n\n"
            f"Original request:\n{request}\n\n"
            f"Task:\n{task}\n\n"
            f"Known objectives:\n{objectives}\n\n"
            f"Preliminary research context:\n{preliminary}\n\n"
            "Return a direct result for this task, including useful findings, source/tool evidence, assumptions, and any hard blockers."
        )

    def _get_hermes_max_iterations(self, state: ResearchState) -> int:
        researching = state.get("research_depth", {}).get("researching", {})
        configured = researching.get("max_iterations") if isinstance(researching, dict) else None
        return int(configured or state.get("max_iterations", 10) or 10)

    def _agent_loop_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": result.get("success", False),
            "iterations": result.get("iterations", 0),
            "tool_calls_count": result.get("tool_calls_count", 0),
            "tool_results": result.get("tool_results", []),
            "errors": result.get("errors", []),
            "metadata": result.get("metadata", {}),
        }

    def _streaming_event_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        metadata = result.get("agent_loop_metadata", {})
        return {
            "timestamp": datetime.now().isoformat(),
            "task_id": result.get("task_id", ""),
            "status": result.get("status", "completed"),
            "data": result.get("result") or result.get("error"),
            "tool_used": result.get("tool_used", "hermes_agent_loop"),
            "iterations": result.get("iterations") or metadata.get("iterations", 0),
            "tool_calls_count": metadata.get("tool_calls_count", 0),
        }

    async def _save_executions(self, results: List[Dict[str, Any]]):
        """Save execution state during cleanup."""
        try:
            # Implementation of persistence logic
            pass
        except Exception as e:
            logger.error(f"Failed to save executions: {e}")

    def _adjust_depth_if_needed(self, state, tasks, execution_results):
        current_depth = state.get("research_depth", {})
        if not current_depth or not hasattr(self, "research_depth"):
            return

        from src.core.adaptive_research_depth import DepthConfig, ResearchPreset

        try:
            progress = {
                "iteration_count": state.get("research_iteration", 0) + 1,
                "completion_rate": len(
                    [r for r in execution_results if r.get("status") == "completed"]
                )
                / max(len(tasks), 1),
                "tasks_total": len(tasks),
            }
            preset = ResearchPreset(current_depth.get("preset", "medium"))
            current_depth_config = DepthConfig(
                preset=preset,
                planning=current_depth.get("planning", {}),
                researching=current_depth.get("researching", {}),
                reporting=current_depth.get("reporting", {}),
                complexity_score=current_depth.get("complexity_score", 0.5),
            )
            adjusted = self.research_depth.adjust_depth_progressively(
                current_depth_config, progress
            )
            if adjusted:
                state["research_depth"] = {
                    "preset": adjusted.preset.value,
                    "planning": adjusted.planning,
                    "researching": adjusted.researching,
                    "reporting": adjusted.reporting,
                }
        except:
            pass

    def decide_next_step_based_on_context(self, state: ResearchState) -> str:
        """컨텍스트 기반 다음 단계 결정."""
        if state.get("error_message"):
            return "verify"
        iteration = state.get("research_iteration", 0)
        if iteration < 1:
            return "continue_research"
        return "compress"

    def _create_priority_queue(self, state: ResearchState) -> List[Dict[str, Any]]:
        tasks = state.get("planned_tasks", [])
        queue = []
        for task in tasks:
            p = (
                1
                if task.get("priority") == "high"
                else (2 if task.get("priority") == "medium" else 3)
            )
            queue.append(
                {
                    "task_id": task.get("task_id", ""),
                    "priority": p,
                    "complexity": task.get("estimated_complexity", 5),
                }
            )
        queue.sort(key=lambda x: (x["priority"], x["complexity"]))
        return queue

    def _get_tool_category_for_task(self, task: Dict[str, Any]) -> ToolCategory:
        """태스크 타입에서 도구 카테고리를 결정합니다."""
        tt = task.get("type", "research").lower()
        desc = task.get("description", "").lower()

        # BROWSER: URL 접근, 웹 스크래핑, 폼 제출 등
        if "browser" in tt or "web" in tt or "scrape" in tt:
            return ToolCategory.BROWSER
        if any(kw in desc for kw in ["navigate", "browse", "screenshot", "scrape", "fill form"]):
            return ToolCategory.BROWSER

        if "search" in tt:
            return ToolCategory.SEARCH
        if "academic" in tt:
            return ToolCategory.ACADEMIC
        if "data" in tt:
            return ToolCategory.DATA
        if "code" in tt:
            return ToolCategory.CODE

        return ToolCategory.SEARCH

    def _get_available_tools_for_category(self, category: ToolCategory) -> List[str]:
        """카테고리별 사용 가능한 도구 목록을 반환합니다."""
        mapping = {
            ToolCategory.SEARCH: ["g-search", "ddg_search::search", "tavily-mcp::tavily-search"],
            ToolCategory.ACADEMIC: ["arxiv", "scholar"],
            ToolCategory.DATA: ["fetch::fetch_url", "fetch::extract_elements"],
            ToolCategory.CODE: ["python_coder", "code_interpreter"],
            ToolCategory.BROWSER: [
                "cdp_navigate",
                "cdp_click",
                "cdp_type_text",
                "cdp_screenshot",
                "cdp_extract_text",
                "cdp_js",
                "cdp_page_info",
            ],
        }
        tools = mapping.get(category)
        if tools is None:
            raise ValueError(f"No tools configured for category: {category.value}")
        return tools

    async def _execute_browser_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """PlaywrightController를 사용하여 브라우저 태스크를 실행합니다."""
        controller = _get_browser_controller()
        task_desc = task.get("description", "")
        task_url = task.get("url", "")
        task_actions = task.get("actions", [])

        # 동적으로 컨트롤러 종류 확인
        is_cdp = hasattr(controller, "cdp")

        try:
            # 컨트롤러 초기화 시도
            init_success = await controller.initialize()
            if not init_success:
                raise RuntimeError("Browser backend initialization failed.")

            # URL이 있으면 navigate
            if task_url:
                await controller.navigate(task_url)
            elif "http" in task_desc:
                import re

                url_match = re.search(r"https?://\S+", task_desc)
                if url_match:
                    await controller.navigate(url_match.group())

            # actions가 있으면 interact
            if task_actions:
                await controller.interact(task_actions)

            # 콘텐츠 추출
            extraction_spec = task.get("extraction", {"full_text": True, "metadata": True})
            extracted = await controller.extract(extraction_spec)

            # 검증
            expectations = task.get("verify", [])
            if expectations:
                verification = await controller.verify(expectations)
                extracted["verification"] = {
                    "verified": verification.verified,
                    "confidence": verification.confidence,
                    "details": verification.details,
                }

            return {
                "task_id": task.get("id"),
                "task_name": task.get("name"),
                "tool_used": "cdp" if is_cdp else "playwright",
                "result": extracted,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Browser task execution failed: {e}")
            # Do NOT make fallback throughpasses. Return as failed state.
            return {
                "task_id": task.get("id"),
                "task_name": task.get("name"),
                "tool_used": "cdp" if hasattr(controller, "cdp") else "playwright",
                "status": "failed",
                "error": str(e),
                "is_fallback_failure": True,
            }

    def _generate_tool_parameters(self, task: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        params = (task.get("parameters") or {}).copy()
        text = f"{task.get('name', '')} {task.get('description', '')}"
        if "query" not in params:
            params["query"] = text[:200]
        return params

    def _validate_tool_result(self, tool_result: Dict[str, Any], task: Dict[str, Any]) -> bool:
        if not tool_result.get("success"):
            return False
        data = tool_result.get("data")
        if not data:
            return False
        if isinstance(data, (str, list, dict)) and len(data) == 0:
            return False
        return True

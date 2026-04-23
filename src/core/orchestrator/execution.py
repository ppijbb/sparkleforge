import logging
from datetime import datetime
from typing import Any, Dict, List

from src.core.orchestrator.state import ResearchState
from src.core.orchestrator.base_node import BaseNode
from src.core.mcp_integration import ToolCategory, execute_tool

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

        state.update({
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
        })
        return state

    async def execute_research(self, state: ResearchState) -> ResearchState:
        """연구 실행 (Universal MCP Hub + Streaming Pipeline + Parallel Execution)."""
        self._log_node_input("execute_research", state)
        logger.info("⚙️ Thinking: Executing research tasks and gathering information")

        tasks = state.get("planned_tasks", [])
        agent_assignments = state.get("agent_assignments", {})
        execution_plan = state.get("execution_plan", {})
        objective_id = state.get("objective_id", "default")

        use_parallel = (
            execution_plan.get("strategy") in ["parallel", "hybrid"]
            and len(tasks) > 1
            and self.agent_config.max_concurrent_research_units > 1
        )

        execution_results = []
        streaming_data = []

        if use_parallel:
            logger.info("🚀 Using parallel execution with ParallelAgentExecutor")
            from src.core.parallel_agent_executor import ParallelAgentExecutor
            executor = ParallelAgentExecutor()
            parallel_results = await executor.execute_parallel_tasks(
                tasks=tasks,
                agent_assignments=agent_assignments,
                execution_plan=execution_plan,
                objective_id=objective_id,
            )
            execution_results = parallel_results.get("execution_results", [])
            streaming_data = [
                {
                    "timestamp": datetime.now().isoformat(),
                    "task_id": r.get("task_id", ""),
                    "status": r.get("status", "completed"),
                    "data": r.get("result"),
                    "tool_used": r.get("tool_used", ""),
                }
                for r in execution_results
            ]
        else:
            logger.info("📝 Using sequential execution")
            for task in tasks:
                task_success = False
                tool_attempts = []
                try:
                    tool_category = self._get_tool_category_for_task(task)

                    # BROWSER 카테고리: PlaywrightController로 직접 처리
                    if tool_category == ToolCategory.BROWSER:
                        browser_result = await self._execute_browser_task(task)
                        if browser_result:
                            execution_results.append(browser_result)
                            streaming_data.append({
                                "timestamp": datetime.now().isoformat(),
                                "task_id": task.get("id"),
                                "status": browser_result.get("status", "completed"),
                                "data": browser_result.get("result"),
                                "tool_used": "playwright",
                            })
                            continue

                    available_tools = self._get_available_tools_for_category(tool_category)

                    for tool_name in available_tools:
                        try:
                            params = self._generate_tool_parameters(task, tool_name)
                            if "__missing_required__" in params:
                                tool_attempts.append({"tool": tool_name, "success": False, "error": "Missing params"})
                                continue
                            
                            tool_result = await execute_tool(tool_name, params)
                            tool_attempts.append({
                                "tool": tool_name, 
                                "success": tool_result.get("success", False),
                                "execution_time": tool_result.get("execution_time", 0.0)
                            })

                            if tool_result.get("success", False) and self._validate_tool_result(tool_result, task):
                                res_item = {
                                    "task_id": task.get("id"),
                                    "task_name": task.get("name"),
                                    "tool_used": tool_name,
                                    "result": tool_result.get("data"),
                                    "status": "completed",
                                }
                                execution_results.append(res_item)
                                streaming_data.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "task_id": task.get("id"),
                                    "status": "completed",
                                    "data": tool_result.get("data"),
                                    "tool_used": tool_name,
                                })
                                task_success = True
                                break
                        except Exception as e:
                            logger.warning(f"Tool {tool_name} failed: {e}")
                    
                    if not task_success:
                        execution_results.append({
                            "task_id": task.get("id"),
                            "status": "failed",
                            "error": f"All {len(available_tools)} tools failed",
                            "attempts": tool_attempts,
                        })
                except Exception as e:
                    logger.error(f"Task execution error: {e}")

        # Depth Adjustment (Progressive Deepening)
        self._adjust_depth_if_needed(state, tasks, execution_results)

        state.update({
            "execution_results": execution_results,
            "streaming_data": streaming_data,
            "current_step": "hierarchical_compression",
            "research_iteration": state.get("research_iteration", 0) + 1,
            "innovation_stats": {
                **state.get("innovation_stats", {}),
                "tasks_executed": len(execution_results),
                "parallel_execution_used": use_parallel,
            },
        })
        self._log_node_output("execute_research", state, {"tasks_executed": len(execution_results)})
        return state

    def _adjust_depth_if_needed(self, state, tasks, execution_results):
        current_depth = state.get("research_depth", {})
        if not current_depth or not hasattr(self, "research_depth"):
            return

        from src.core.adaptive_research_depth import DepthConfig, ResearchPreset
        try:
            progress = {
                "iteration_count": state.get("research_iteration", 0) + 1,
                "completion_rate": len([r for r in execution_results if r.get("status") == "completed"]) / max(len(tasks), 1),
                "tasks_total": len(tasks),
            }
            preset = ResearchPreset(current_depth.get("preset", "medium"))
            current_depth_config = DepthConfig(
                preset=preset, 
                planning=current_depth.get("planning", {}),
                researching=current_depth.get("researching", {}),
                reporting=current_depth.get("reporting", {}),
                complexity_score=current_depth.get("complexity_score", 0.5)
            )
            adjusted = self.research_depth.adjust_depth_progressively(current_depth_config, progress)
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
            p = 1 if task.get("priority") == "high" else (2 if task.get("priority") == "medium" else 3)
            queue.append({"task_id": task.get("task_id", ""), "priority": p, "complexity": task.get("estimated_complexity", 5)})
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
            ToolCategory.ACADEMIC: ["semantic_scholar::papers-search-basic", "arxiv::arxiv_search"],
            ToolCategory.DATA: ["fetch::fetch_url", "fetch::extract_elements"],
            ToolCategory.CODE: ["python_coder", "code_interpreter"],
            ToolCategory.BROWSER: ["cdp_navigate", "cdp_click", "cdp_type_text", "cdp_screenshot", "cdp_extract_text", "cdp_js", "cdp_page_info"],
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
                raise RuntimeError(f"Browser backend initialization failed.")

            # URL이 있으면 navigate
            page_state = None
            if task_url:
                page_state = await controller.navigate(task_url)
            elif "http" in task_desc:
                import re
                url_match = re.search(r'https?://\S+', task_desc)
                if url_match:
                    page_state = await controller.navigate(url_match.group())

            # actions가 있으면 interact
            if task_actions:
                action_results = await controller.interact(task_actions)

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
        if not tool_result.get("success"): return False
        data = tool_result.get("data")
        if not data: return False
        if isinstance(data, (str, list, dict)) and len(data) == 0: return False
        return True

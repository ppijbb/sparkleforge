import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List

from src.core.llm_manager import TaskType, execute_llm_task
from src.core.mcp_integration import execute_tool
from src.core.orchestrator.base_node import BaseNode
from src.core.orchestrator.state import ResearchState

logger = logging.getLogger(__name__)


def _extract_tool_result_items(result_data: Any) -> List[Any]:
    """Return result items from common local/MCP tool payload shapes."""
    if isinstance(result_data, list):
        return result_data
    if not isinstance(result_data, dict):
        return []

    for key in ("results", "items", "data"):
        value = result_data.get(key)
        if isinstance(value, list):
            return value

    value = result_data.get("result")
    if isinstance(value, list):
        return value
    if value:
        return [value]

    return []


class PlanningNode(BaseNode):
    """Handler for research planning and task decomposition."""

    def __init__(
        self, context_manager, context_loader, research_depth, hybrid_storage, streaming_manager
    ):
        self.context_manager = context_manager
        self.context_loader = context_loader
        self.research_depth = research_depth
        self.hybrid_storage = hybrid_storage
        self.streaming_manager = streaming_manager

    async def planning_agent(self, state: ResearchState) -> ResearchState:
        """Planning Agent: MCP 기반 사전 조사 → Task 분해 → Agent 동적 할당 (재귀적 컨텍스트 사용)."""
        # 입력 로깅
        self._log_node_input("planning_agent", state)

        logger.info("📋 Thinking: Creating research plan and task breakdown")
        logger.info(f"📊 Complexity Score: {state.get('complexity_score', 5.0)}")
        logger.info(f"🎯 Objectives: {len(state.get('analyzed_objectives', []))}")

        # 현재 컨텍스트 가져오기
        current_context = self.context_manager.get_current_context()
        if not current_context:
            initial_context_data = {
                "user_request": state.get("user_request", ""),
                "context": state.get("context", {}),
                "objective_id": state.get("objective_id", ""),
                "stage": "planning",
            }
            current_context_id = self.context_manager.push_context(
                context_data=initial_context_data, depth=0
            )
            current_context = self.context_manager.get_current_context()

        if current_context:
            analysis_context = {
                "intent_analysis": state.get("intent_analysis", {}),
                "domain_analysis": state.get("domain_analysis", {}),
                "scope_analysis": state.get("scope_analysis", {}),
                "analyzed_objectives": state.get("analyzed_objectives", []),
                "complexity_score": state.get("complexity_score", 5.0),
                "stage": "planning",
            }

            extended_context = self.context_manager.extend_context(
                current_context.context_id,
                analysis_context,
                metadata={
                    "node": "planning_agent",
                    "timestamp": datetime.now().isoformat(),
                },
            )

            if extended_context:
                logger.debug(f"Context extended for planning: {extended_context.context_id}")

        # 사용자 응답 대기 중이면 응답 처리
        if state.get("waiting_for_user", False):
            user_responses = state.get("user_responses", {})
            if user_responses:
                from src.core.human_clarification_handler import (
                    get_clarification_handler,
                )

                clarification_handler = get_clarification_handler()

                for question_id, response_data in user_responses.items():
                    clarification = response_data.get("clarification", {})
                    state["clarification_context"] = state.get("clarification_context", {})
                    state["clarification_context"][question_id] = clarification

                state["waiting_for_user"] = False
                state["pending_questions"] = []
                logger.info("✅ User responses processed, continuing planning")

        try:
            # 컨텍스트 로드 (SPARKLEFORGE.md)
            try:
                project_context = await self.context_loader.load_context()
                if project_context:
                    logger.info("📄 Loaded project context from SPARKLEFORGE.md")
                    state["context"] = state.get("context", {})
                    state["context"]["project_context"] = project_context
            except Exception as e:
                logger.debug(f"Failed to load context: {e}")

            # 불명확한 부분 감지 (항상 수행)
            if not state.get("clarification_context"):
                from src.core.human_clarification_handler import (
                    get_clarification_handler,
                )

                clarification_handler = get_clarification_handler()
                try:
                    ambiguities = await asyncio.wait_for(
                        clarification_handler.detect_ambiguities(
                            state.get("user_request", ""),
                            {
                                "objectives": state.get("analyzed_objectives", []),
                                "domain": state.get("domain_analysis", {}),
                                "scope": state.get("scope_analysis", {}),
                            },
                        ),
                        timeout=10.0,
                    )
                except TimeoutError:
                    logger.warning("detect_ambiguities timeout, proceeding without clarification")
                    ambiguities = []
                except Exception as e:
                    logger.warning(f"detect_ambiguities failed: {e}")
                    ambiguities = []
            else:
                ambiguities = []

            if ambiguities:
                # Non-interactive automation must not wait for user input inside the
                # graph. Proceed with explicit best-effort assumptions so scheduled
                # workflows produce a deliverable or a real runtime error.
                if state.get("autopilot_mode", False):
                    logger.info(
                        "🤖 Autopilot mode — proceeding without interactive clarification"
                    )
                    state["clarification_context"] = state.get("clarification_context", {})
                    state["autopilot_assumptions"] = [
                        {
                            "ambiguity": ambiguity,
                            "resolution": (
                                "Proceed with the most conservative interpretation "
                                "that satisfies the original request."
                            ),
                        }
                        for ambiguity in ambiguities
                    ]
                    state["waiting_for_user"] = False
                    state["pending_questions"] = []
                    state["autopilot_mode"] = True
                else:
                    # interactive mode: 사용자에게 질문 전달
                    questions = []
                    for ambiguity in ambiguities:
                        question = await clarification_handler.generate_question(
                            ambiguity, {"user_request": state.get("user_request", "")}
                        )
                        questions.append(question)

                    state["pending_questions"] = questions
                    state["waiting_for_user"] = True
                    state["current_step"] = "waiting_for_clarification"
                    state["user_responses"] = {}

                    key_changes = {
                        "pending_questions_count": len(questions),
                        "waiting_for_user": True,
                        "current_step": "waiting_for_clarification",
                    }
                    self._log_node_output("planning_agent", state, key_changes)
                    return state

            # Adaptive Research Depth
            from src.core.adaptive_research_depth import ResearchPreset

            user_request = state.get("user_request", "")
            preset_str = state.get("research_preset")
            preset = None
            if preset_str:
                try:
                    preset = ResearchPreset(preset_str)
                except ValueError:
                    preset = None

            depth_config = self.research_depth.determine_depth(
                user_request, preset=preset, context=state.get("context")
            )

            state["research_depth"] = {
                "preset": depth_config.preset.value,
                "planning": depth_config.planning,
                "researching": depth_config.researching,
                "reporting": depth_config.reporting,
                "complexity_score": depth_config.complexity_score,
            }

            preliminary_research = await self._conduct_preliminary_research(state)
            tasks = await self._decompose_into_tasks(state, preliminary_research, depth_config)

            clarification_context = state.get("clarification_context", {})
            if clarification_context:
                from src.core.human_clarification_handler import (
                    get_clarification_handler,
                )

                clarification_handler = get_clarification_handler()
                for task in tasks:
                    for question_id, clarification in clarification_context.items():
                        task = clarification_handler.apply_clarification(clarification, task)

            agent_assignments = await self._assign_agents_dynamically(tasks, state)
            execution_plan = await self._create_execution_plan(tasks, agent_assignments)

            state.update(
                {
                    "preliminary_research": preliminary_research,
                    "planned_tasks": tasks,
                    "agent_assignments": agent_assignments,
                    "execution_plan": execution_plan,
                    "plan_approved": False,
                    "plan_feedback": None,
                    "plan_iteration": state.get("plan_iteration", 0) + 1,
                    "current_step": "verify_plan",
                }
            )

        except Exception as e:
            logger.error(f"❌ Planning failed: {e}")
            state["error_message"] = str(e)
            state["should_continue"] = False
            raise

        self._log_node_output(
            "planning_agent",
            state,
            {"tasks_count": len(tasks), "agents_count": len(agent_assignments)},
        )
        return state

    async def _conduct_preliminary_research(self, state: ResearchState) -> Dict[str, Any]:
        """MCP 도구로 사전 조사 수행."""
        objectives = state.get("analyzed_objectives", [])
        domain = state.get("domain_analysis", {})
        keywords = self._extract_keywords(objectives, domain)

        search_results = []
        search_tools = ["g-search", "tavily", "exa"]
        # Distribute keywords across search tools via modulo so a single tool
        # failure or low-relevance return doesn't waste every other tool on the
        # exact same query. An empty keywords list must not reach any tool.
        if not keywords or not search_tools:
            return {
                "keywords": keywords,
                "search_results": [],
                "academic_results": [],
                "sources_count": 0,
                "total_results": 0,
            }

        for i, kw in enumerate(keywords[:4]):
            if not kw:
                continue
            tool_name = search_tools[i % len(search_tools)]
            try:
                result = await execute_tool(
                    tool_name=tool_name, parameters={"query": kw, "max_results": 5}
                )
                if result.get("success", False):
                    result_data = result.get("data", {})
                    data_list = _extract_tool_result_items(result_data)
                    search_results.append(
                        {
                            "keyword": kw,
                            "tool": tool_name,
                            "data": data_list,
                            "sources_count": len(data_list),
                        }
                    )
            except Exception as e:
                logger.warning(f"⚠️ {tool_name} search error: {e}")

        academic_results = []
        academic_query = " ".join(keywords[:2])
        if academic_query:
            try:
                result = await execute_tool(
                    tool_name="arxiv",
                    parameters={"query": academic_query, "max_results": 3},
                )
                if result.get("success", False):
                    result_data = result.get("data", {})
                    data_list = _extract_tool_result_items(result_data)
                    academic_results.append(
                        {"tool": "arxiv", "data": data_list, "sources_count": len(data_list)}
                    )
            except Exception as e:
                logger.warning(f"⚠️ academic search error: {e}")

        return {
            "keywords": keywords,
            "search_results": search_results,
            "academic_results": academic_results,
            "sources_count": len(search_results) + len(academic_results),
            "total_results": sum(
                r.get("sources_count", 0) for r in search_results + academic_results
            ),
        }

    def _extract_keywords(
        self, objectives: List[Dict[str, Any]], domain: Dict[str, Any]
    ) -> List[str]:
        keywords = []
        for obj in objectives:
            words = (obj.get("description", "")).lower().split()
            keywords.extend(
                [w for w in words if len(w) > 3 and w not in ["the", "and", "for", "with", "from"]]
            )
        keywords.extend(domain.get("fields", []))
        from collections import Counter

        return [kw for kw, _ in Counter(keywords).most_common(10)]

    async def _decompose_into_tasks(
        self,
        state: ResearchState,
        preliminary_research: Dict[str, Any],
        depth_config: Any | None = None,
    ) -> List[Dict[str, Any]]:
        complexity_raw = state.get("complexity_score", 5.0)
        complexity = (
            float(complexity_raw.get("score", 5.0))
            if isinstance(complexity_raw, dict)
            else float(complexity_raw)
        )

        num_tasks = 5
        if depth_config:
            p_cfg = depth_config.planning.get("decompose", {})
            if p_cfg.get("mode") == "auto":
                num_tasks = min(max(3, int(complexity) + 3), p_cfg.get("auto_max_subtopics", 8))
            else:
                num_tasks = p_cfg.get("initial_subtopics", 5)

        initial_tasks = await self._create_initial_tasks(
            state, preliminary_research, num_tasks, complexity
        )
        final_tasks = []
        max_rec = depth_config.planning.get("max_recursion_depth", 3) if depth_config else 3

        for task in initial_tasks:
            if await self._is_atomic_task(task, depth_config, complexity):
                final_tasks.append(task)
            else:
                subtasks = await self._recursive_decompose(
                    task, state, preliminary_research, depth_config, 0, max_rec
                )
                final_tasks.extend(subtasks)
        return final_tasks

    async def _create_initial_tasks(
        self,
        state: ResearchState,
        preliminary_research: Dict[str, Any],
        num_tasks: int,
        complexity: float,
    ) -> List[Dict[str, Any]]:
        plan_iteration = state.get("plan_iteration", 0)
        plan_feedback = state.get("plan_feedback") or ""
        feedback_block = (
            f"\n[PREVIOUS PLAN REJECTED]\n{plan_feedback}\n"
            if plan_iteration > 0 and plan_feedback
            else ""
        )

        decomposition_prompt = f"""
        Based on preliminary research, decompose into {num_tasks} tasks:
        {feedback_block}
        (Apply non-linear structural mutation: prioritize high-surprise leaps over incremental tweaks while maintaining physical feasibility)
        Request: {state.get("user_request", "")}
        Complexity: {complexity}
        Preliminary Research: {preliminary_research.get("keywords", [])}
        Return as JSON array of task objects.
        {{ "task_id": "task_1", "name": "...", "description": "...", ... }}
        """
        result = await execute_llm_task(prompt=decomposition_prompt, task_type=TaskType.PLANNING)
        return self._parse_tasks_result(result.content)

    async def _is_atomic_task(
        self, task: Dict[str, Any], depth_config: Any | None, complexity: float
    ) -> bool:
        t_comp = task.get("estimated_complexity", 5)
        if isinstance(t_comp, dict):
            t_comp = t_comp.get("score", 5)
        if float(t_comp) >= 8:
            return False
        if len(task.get("dependencies", [])) >= 2:
            return False
        if len(task.get("required_tools", [])) >= 3:
            return False
        return float(t_comp) <= 5

    async def _recursive_decompose(
        self,
        task: Dict[str, Any],
        state: ResearchState,
        preliminary_research: Dict[str, Any],
        depth_config: Any | None,
        current_depth: int,
        max_depth: int,
    ) -> List[Dict[str, Any]]:
        if current_depth >= max_depth:
            return [task]
        t_comp = task.get("estimated_complexity", 5)
        if isinstance(t_comp, dict):
            t_comp = t_comp.get("score", 5)
        num_sub = min(3 + int(float(t_comp) / 2), 5)
        parent_id = task.get("task_id", "unknown")

        prompt = f"Decompose task {parent_id} into {num_sub} subtasks. Result as JSON array."
        result = await execute_llm_task(prompt=prompt, task_type=TaskType.PLANNING)
        subtasks = self._parse_tasks_result(result.content)

        final = []
        for st in subtasks:
            st["parent_task_id"] = parent_id
            st["decomposition_depth"] = current_depth + 1
            if await self._is_atomic_task(st, depth_config, float(t_comp)):
                final.append(st)
            else:
                final.extend(
                    await self._recursive_decompose(
                        st, state, preliminary_research, depth_config, current_depth + 1, max_depth
                    )
                )
        return final

    async def _assign_agents_dynamically(
        self, tasks: List[Dict[str, Any]], state: ResearchState
    ) -> Dict[str, List[str]]:
        assignments = {}
        avail = state.get("allocated_researchers", 1)
        for task in tasks:
            tid = task.get("task_id", "any")
            comp = task.get("estimated_complexity", 5)
            if isinstance(comp, dict):
                comp = comp.get("score", 5)
            num = 1 if float(comp) <= 3 else (2 if float(comp) <= 7 else 3)
            num = min(num, avail)
            assignments[tid] = self._select_agent_types(task.get("type", "research"), num)
        return assignments

    def _select_agent_types(self, task_type: str, num: int) -> List[str]:
        mapping = {
            "academic": ["academic_researcher"],
            "market": ["market_analyst"],
            "technical": ["technical_researcher"],
            "data": ["data_collector"],
            "synthesis": ["synthesis_specialist"],
        }
        base = mapping.get(task_type, ["academic_researcher"])
        if num <= len(base):
            return base[:num]
        res = base.copy()
        for t in ["market_analyst", "technical_researcher", "data_collector"]:
            if len(res) >= num:
                break
            if t not in res:
                res.append(t)
        return res[:num]

    async def _create_execution_plan(
        self, tasks: List[Dict[str, Any]], agent_assignments: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        graph = {t.get("task_id"): t.get("dependencies", []) for t in tasks}
        parallel = self._identify_parallel_groups(graph)
        order = self._determine_execution_order(tasks, graph)
        return {
            "strategy": "hybrid" if parallel else "sequential",
            "parallel_groups": parallel,
            "execution_order": order,
            "task_count": len(tasks),
            "agent_count": len(set(a for al in agent_assignments.values() for a in al)),
        }

    def _identify_parallel_groups(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        res = []
        proc = set()
        for tid, deps in graph.items():
            if tid not in proc and not deps:
                group = [tid]
                proc.add(tid)
                for ot, od in graph.items():
                    if ot not in proc and not od:
                        group.append(ot)
                        proc.add(ot)
                if len(group) > 1:
                    res.append(group)
        return res

    def _determine_execution_order(
        self, tasks: List[Dict[str, Any]], graph: Dict[str, List[str]]
    ) -> List[str]:
        in_degree = {tid: 0 for tid in graph.keys()}
        for tid, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[tid] += 1
        queue = [tid for tid, d in in_degree.items() if d == 0]
        res = []
        while queue:
            curr = queue.pop(0)
            res.append(curr)
            for tid, deps in graph.items():
                if curr in deps:
                    in_degree[tid] -= 1
                    if in_degree[tid] == 0:
                        queue.append(tid)
        return res

    def _parse_tasks_result(self, content: str) -> List[Dict[str, Any]]:
        for attempt in range(3):
            try:
                cleaned = content.strip()
                if "```json" in cleaned:
                    match = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
                    if match:
                        cleaned = match.group(1).strip()
                elif "```" in cleaned:
                    match = re.search(r"```\s*(.*?)\s*```", cleaned, re.DOTALL)
                    if match:
                        cleaned = match.group(1).strip()

                if cleaned.startswith("["):
                    return json.loads(cleaned)
            except:
                if attempt == 2:
                    raise
        return []

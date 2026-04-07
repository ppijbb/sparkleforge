"""SparkleForge Agent Harness.

이 모듈은 시스템의 최상위 실행 셸이며, LangGraph 기반의 상태 머신을 정의하고 구동합니다.
"""

import logging
from typing import Dict, Any, Literal
import time

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.core.harness_state import HarnessState, create_initial_harness_state
from src.core.task_router import TaskRouter, RoutePath

logger = logging.getLogger(__name__)

class AgentHarness:
    """Agent Harness - 2026 표준 상태 머신 오케스트레이터"""
    
    def __init__(self):
        self.router = TaskRouter()
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """LangGraph 상태 머신 구축"""
        workflow = StateGraph(HarnessState)
        
        # 노드 추가
        workflow.add_node("classify", self._node_classify)
        workflow.add_node("single_agent", self._node_single_agent)
        workflow.add_node("planner", self._node_planner)
        workflow.add_node("executor", self._node_executor)
        workflow.add_node("subagent_delegate", self._node_subagent_delegate)
        workflow.add_node("synthesize", self._node_synthesize)
        
        # 기본 라우팅 그래프 
        workflow.set_entry_point("classify")
        
        # 조건부 엣지 - classify 이후 라우팅
        workflow.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "single_agent": "single_agent",
                "planner_parallel": "planner",
                "financial_pipeline": "planner",  # 일단 플래너로 라우팅 후 처리
                "codebase_agent": "single_agent", # 향후 codebase 폴백
                "creativity_agent": "single_agent" # 향후 창의성 워크플로 풀백
            }
        )
        
        # Planner 종료 후 실행 
        workflow.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {
                "execute_parallel": "executor",
                "delegate_subagents": "subagent_delegate",
                "synthesize": "synthesize"
            }
        )
        
        # 작업 완료 후는 무조건 합성
        workflow.add_edge("single_agent", "synthesize")
        workflow.add_edge("executor", "synthesize")
        workflow.add_edge("subagent_delegate", "synthesize")
        
        # 합성이 끝나면 종료
        workflow.add_edge("synthesize", END)
        
        return workflow.compile(checkpointer=self.memory)
        
    # --- Nodes ---
    
    async def _node_classify(self, state: HarnessState) -> Dict[str, Any]:
        """[Node] 입력 분류 및 파이프라인 판단"""
        query = state["workflow"]["user_query"]
        logger.info(f"[Harness] Classify Node: Analyzing query '{query[:50]}...'")
        
        # TaskRouter를 통해 경로 결정
        route = self.router.determine_route(query)
        logger.info(f"[Harness] Route determined: {route.name}")
        
        # 라우트에 따른 초기 상태 세팅
        updated_state = self.router.update_state_for_route(state, route)
        
        # 내부 라우팅 저장을 위해 workflow phase를 변경하여 전달
        updated_state["workflow"]["phase"] = route.value
        
        return updated_state

    async def _node_single_agent(self, state: HarnessState) -> Dict[str, Any]:
        """[Node] 단일 에이전트로 단순 태스크 해결"""
        logger.info("[Harness] Single Agent Node")
        # Agent Orchestrator 내부의 agent 클래스를 활용하거나 여기서 직접 호출
        # 지금은 스켈레톤, 이후 실제 agent 연결
        state["workflow"]["phase"] = "execute"
        # 더미 결과 추가
        # state["workflow"]["final_output"] = f"Result for {state['workflow']['user_query']}"
        return state

    async def _node_planner(self, state: HarnessState) -> Dict[str, Any]:
        """[Node] 복합 태스크 기획 및 분할"""
        logger.info("[Harness] Planner Node")
        state["workflow"]["phase"] = "plan"
        
        # Planner 에이전트 호출
        from src.agents.planner_agent import PlannerAgent
        planner = PlannerAgent()
        result = await planner.create_plan(state)
        
        state["workflow"]["plan"] = result.get("plan", "")
        # 새로운 TaskState 리스트 할당
        state["workflow"]["tasks"] = result.get("tasks", []) 
        
        return state

    async def _node_executor(self, state: HarnessState) -> Dict[str, Any]:
        """[Node] 병렬 에이전트로 태스크 처리 (기존 Executor)"""
        logger.info("[Harness] Executor Node")
        state["workflow"]["phase"] = "execute"
        
        from src.core.parallel_agent_executor import ParallelAgentExecutor
        executor = ParallelAgentExecutor()
        
        # Execute tasks using parallel execution engine
        tasks = state["workflow"]["tasks"]
        session_id = state["workflow"]["session_id"]
        
        # In a real environment, wait for execution results
        results = await executor.execute_parallel_tasks(
            tasks=tasks,
            agent_assignments={},
            execution_plan={"strategy": "parallel_groups"},
            objective_id=session_id
        )
        
        state["workflow"]["final_output"] = f"Executed {len(tasks)} tasks via parallel agent. See individual results."
        return state

    async def _node_subagent_delegate(self, state: HarnessState) -> Dict[str, Any]:
        """[Node] 서브에이전트 군단에 태스크 위임 (Context quarantined)"""
        logger.info("[Harness] SubAgent Delegate Node")
        state["workflow"]["phase"] = "execute_isolated"
        return state

    async def _node_synthesize(self, state: HarnessState) -> Dict[str, Any]:
        """[Node] 결과 총합 및 최종 응답 생성"""
        logger.info("[Harness] Synthesize Node")
        state["workflow"]["phase"] = "synthesize"
        
        # 더미 결과 추가
        final_output = state["workflow"].get("final_output")
        if not final_output:
            state["workflow"]["final_output"] = "Placeholder logic completed"
            
        return state

    # --- Conditional Edges ---
    
    def _route_after_classify(self, state: HarnessState) -> str:
        """라우팅 선택 엣지"""
        phase = state["workflow"]["phase"]
        
        if phase == RoutePath.SINGLE_AGENT.value:
            return "single_agent"
        elif phase == RoutePath.PLANNER_PARALLEL.value:
            return "planner_parallel"
        elif phase == RoutePath.FINANCIAL_PIPELINE.value:
            return "financial_pipeline"
        elif phase == RoutePath.CODEBASE_AGENT.value:
            return "codebase_agent"
        elif phase == RoutePath.CREATIVITY_AGENT.value:
            return "creativity_agent"
            
        return "single_agent" # 기본 폴백

    def _route_after_planner(self, state: HarnessState) -> str:
        """기획 후 실행 방식 선택 엣지"""
        # 현재는 기본적으로 executor(병렬 일반 실행)로 전송
        # Phase 2(SubAgent) 연동 시 delegate로 라우팅되도록 설정 가능
        if len(state["workflow"]["tasks"]) == 0:
            # Task가 없으면 바로 합성 단계로
            return "synthesize"
            
        return "execute_parallel"

    async def execute(self, session_id: str, request: str, max_iterations: int = 10) -> Dict[str, Any]:
        """하네스 실행 (오케스트레이터의 주 진입점)"""
        
        start_time = time.time()
        logger.info(f"🚀 Harness starting session {session_id} for request: '{request[:20]}...'")
        
        # 1. 초기 상태 생성
        initial_state = create_initial_harness_state(session_id, request, max_iterations)
        
        # 2. 실행 설정 (LangGraph 스레드)
        config = {"configurable": {"thread_id": session_id}}
        
        # 3. 그래프 실행
        try:
            final_state = await self.graph.ainvoke(initial_state, config)
            logger.info(f"✅ Harness completed in {time.time() - start_time:.2f}s")
            
            # API 호환을 위해 결과 반환
            return {
                "success": True,
                "session_id": session_id,
                "plan": final_state["workflow"].get("plan", ""),
                "tasks": final_state["workflow"].get("tasks", []),
                "results": final_state["workflow"].get("final_output", ""),
            }
        except Exception as e:
            logger.error(f"❌ Harness execution failed: {e}")
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e)
            }

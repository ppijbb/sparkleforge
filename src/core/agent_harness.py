"""SparkleForge Agent Harness.

이 모듈은 시스템의 최상위 실행 셸이며, LangGraph 기반의 상태 머신을 정의하고 구동합니다.
"""

import asyncio
import logging
from datetime import datetime
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
        workflow.add_node("document_processor", self._node_document_processor)
        workflow.add_node("synthesize", self._node_synthesize)
        
        # 기본 라우팅 그래프 
        workflow.set_entry_point("classify")
        
        # 파이프라인 상관없이 복잡한 태스크는 모두 Planner-TaskGraph 엔진으로 연결
        workflow.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "single_agent": "single_agent",
                "planner_parallel": "planner",
                "financial_pipeline": "planner", 
                "codebase_agent": "planner",  
                "creativity_agent": "planner",
                "document_pipeline": "document_processor"
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
        workflow.add_edge("document_processor", "planner") # 문서 처리 후 추가 계획 수립 가능
        workflow.add_edge("synthesize", END)
        
        return workflow.compile(checkpointer=self.memory)
        
    # --- Nodes ---
    
    async def _node_classify(self, state: HarnessState) -> Dict[str, Any]:
        """[Node] 입력 분류 및 파이프라인 판단"""
        query = state["workflow"]["user_query"]
        logger.info(f"[Harness] Classify Node: Analyzing query '{query[:50]}...'")
        
        # TaskRouter를 통해 경로 결정 (LLM 기반 — await 필요)
        route = await self.router.determine_route(query)
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
        """[Node] 복합 태스크 기획 및 분할 + HIL 통합"""
        logger.info("[Harness] Planner Node")
        state["workflow"]["phase"] = "plan"
        
        # HIL: ambiguity 감지 및 해소
        state = await self._resolve_hil(state)
        
        # Planner 에이전트 호출
        from src.agents.planner_agent import PlannerAgent
        planner = PlannerAgent()
        result = await planner.create_plan(state)
        
        state["workflow"]["plan"] = result.get("plan", "")
        # 새로운 TaskState 리스트 할당
        state["workflow"]["tasks"] = result.get("tasks", []) 
        
        return state

    async def _resolve_hil(self, state: HarnessState) -> HarnessState:
        """HIL: ambiguity 감지 + interaction_mode에 따른 해소.

        interaction_mode:
            - 'autonomous': LLM이 자율 추론으로 ambiguity를 해소. 결과는 hil.resolved_clarifications에만 기록.
            - 'interactive': pending_questions에 추가하여 사용자 응답 대기.

        HIL 결과는 hil state에만 기록되며, 최종 output에는 포함되지 않습니다.
        """
        hil = state.get("hil", {})
        interaction_mode = hil.get("interaction_mode", "autonomous")
        
        # 이미 해소된 clarification이 있으면 skip
        if hil.get("resolved_clarifications"):
            logger.info("[Harness/HIL] Clarifications already resolved, continuing")
            return state
        
        # interactive 모드에서 사용자 응답 대기 중이면 응답 처리
        if hil.get("waiting_for_user", False) and hil.get("resolved_clarifications"):
            hil["waiting_for_user"] = False
            hil["pending_questions"] = []
            state["hil"] = hil
            logger.info("[Harness/HIL] User responses processed")
            return state
        
        # Ambiguity 감지
        from src.core.human_clarification_handler import get_clarification_handler
        clarification_handler = get_clarification_handler()
        
        user_query = state["workflow"]["user_query"]
        try:
            ambiguities = await asyncio.wait_for(
                clarification_handler.detect_ambiguities(
                    user_query,
                    {
                        "objectives": [],
                        "domain": state["context"].get("domain_analysis", {}),
                        "scope": {},
                    },
                ),
                timeout=10.0,
            )
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("[Harness/HIL] Ambiguity detection timeout, proceeding without")
            ambiguities = []
        except Exception as e:
            logger.warning(f"[Harness/HIL] Ambiguity detection failed: {e}")
            ambiguities = []
        
        if not ambiguities:
            logger.info("[Harness/HIL] No ambiguities detected")
            return state
        
        logger.info(f"[Harness/HIL] Detected {len(ambiguities)} ambiguities (mode={interaction_mode})")
        
        if interaction_mode == "autonomous":
            # LLM 자율 추론으로 해소
            resolved = {}
            inference_log = []
            
            for ambiguity in ambiguities:
                question = await clarification_handler.generate_question(
                    ambiguity, {"user_request": user_query}
                )
                auto_response = await clarification_handler.auto_select_response(
                    question,
                    {"user_request": user_query},
                    None,  # shared_memory
                )
                processed = await clarification_handler.process_user_response(
                    question["id"], auto_response, {"question": question}
                )
                
                if processed.get("validated", False):
                    resolved[question["id"]] = processed.get("clarification", {})
                    inference_log.append({
                        "question_id": question["id"],
                        "question_text": question.get("text", ""),
                        "inferred_response": auto_response,
                        "timestamp": datetime.now().isoformat(),
                    })
            
            hil["resolved_clarifications"] = resolved
            hil["inference_log"] = inference_log
            hil["waiting_for_user"] = False
            state["hil"] = hil
            logger.info(f"[Harness/HIL] Autonomous: resolved {len(resolved)} clarifications")
        
        else:
            # interactive 모드: 질문 생성하여 대기
            questions = []
            for ambiguity in ambiguities:
                question = await clarification_handler.generate_question(
                    ambiguity, {"user_request": user_query}
                )
                questions.append(question)
            
            hil["pending_questions"] = questions
            hil["waiting_for_user"] = True
            state["hil"] = hil
            logger.info(f"[Harness/HIL] Interactive: {len(questions)} questions pending")
        
        return state

    async def _node_executor(self, state: HarnessState) -> Dict[str, Any]:
        """[Node] 병렬 에이전트로 태스크 처리 (기존 Executor)"""
        logger.info("[Harness] Executor Node")
        state["workflow"]["phase"] = "execute"
        
        from src.core.parallel_agent_executor import ParallelAgentExecutor
        from src.core.mcp_integration import get_mcp_hub
        
        # Ensure MCP Hub is initialized before execution
        try:
            mcp_hub = get_mcp_hub()
            await mcp_hub.initialize_mcp()
            logger.info("[Harness] MCP Hub initialized")
        except Exception as e:
            logger.warning(f"[Harness] MCP Hub initialization failed: {e}")

        # 에이전트 동적 할당 (TaskRouter 활용)
        tasks = state["workflow"]["tasks"]
        agent_assignments = {}
        for task in tasks:
            agent_id = task.get("task_id")
            # LLM이 태스크 성격에 맞는 전문가 에이전트 선정 (async)
            assigned_agent = await self.router.assign_agent_for_task(task)
            agent_assignments[agent_id] = assigned_agent
            logger.info(f"[Harness] Task {agent_id} assigned to: {assigned_agent}")

        executor = ParallelAgentExecutor()
        session_id = state["workflow"]["session_id"]
        
        # Execute tasks using parallel execution engine with dynamic assignments
        results = await executor.execute_parallel_tasks(
            tasks=tasks,
            agent_assignments=agent_assignments,
            execution_plan={"strategy": "parallel_groups"},
            objective_id=session_id
        )
        
        # Update tasks with results
        execution_results = results.get("execution_results", [])
        if execution_results:
            for i, res in enumerate(execution_results):
                if i < len(tasks):
                    tasks[i]["result"] = res.get("result")
                    tasks[i]["status"] = res.get("status")
                    
        state["workflow"]["tasks"] = tasks
        state["workflow"]["final_output"] = f"Executed {len(tasks)} tasks via dynamic agent army. See individual results."
        return state

    async def _node_subagent_delegate(self, state: HarnessState) -> Dict[str, Any]:
        """[Node] 서브에이전트 군단에 태스크 위임 (Context quarantined)"""
        logger.info("[Harness] SubAgent Delegate Node")
        state["workflow"]["phase"] = "execute_isolated"
        return state

    async def _node_document_processor(self, state: HarnessState) -> Dict[str, Any]:
        """[Node] Docling을 이용한 지능형 문서 처리 및 리소스 추출"""
        logger.info("[Harness] Document Processor Node")
        query = state["workflow"]["user_query"]
        
        from src.core.document_processing.docling_processor import DoclingProcessor
        from src.storage.hybrid_storage import HybridStorage
        
        processor = DoclingProcessor()
        storage = HybridStorage()
        
        # URL 또는 파일 경로 추출 (단순 정규식 또는 LLM 전략)
        import re
        urls = re.findall(r'https?://\S+', query)
        paths = re.findall(r'/[^/\s]+\.[^/\s]+', query)
        
        source = urls[0] if urls else (paths[0] if paths else None)
        
        if source:
            logger.info(f"[Harness] Processing document: {source}")
            result = await processor.process(source, user_id="default_user", instruction=query)
            
            if result.get("success"):
                await processor.store_to_history(storage, result)
                # 컨텍스트에 마크다운 결과 추가
                state["workflow"]["user_query"] += f"\n\n[PROCESSED DOCUMENT CONTENT]\n{result['markdown'][:2000]}..."
                logger.info(f"[Harness] Document processed and stored: {result['doc_id']}")
            else:
                logger.warning(f"[Harness] Document processing failed: {result.get('error')}")
        
        state["workflow"]["phase"] = "plan" # 문서 처리 후 다시 기획 단계로 유도 가능
        return state

    async def _node_synthesize(self, state: HarnessState) -> Dict[str, Any]:
        """[Node] 결과 총합 및 최종 응답 생성 (HIL 데이터는 output에 포함하지 않음)"""
        logger.info("[Harness] Synthesize Node")
        state["workflow"]["phase"] = "synthesize"
        
        from src.agents.generator_agent import GeneratorAgent
        generator = GeneratorAgent()
        result = await generator.synthesize(state)
        
        state["workflow"]["final_output"] = result.get("final_output", "")
            
        return state

    # --- Conditional Edges ---
    
    def _route_after_classify(self, state: HarnessState) -> str:
        """라우팅 선택 엣지"""
        phase = state["workflow"]["phase"]
        
        route_map = {
            RoutePath.SINGLE_AGENT.value: "single_agent",
            RoutePath.PLANNER_PARALLEL.value: "planner_parallel",
            RoutePath.FINANCIAL_PIPELINE.value: "financial_pipeline",
            RoutePath.CODEBASE_AGENT.value: "codebase_agent",
            RoutePath.CREATIVITY_AGENT.value: "creativity_agent",
            RoutePath.DOCUMENT_PIPELINE.value: "document_pipeline",
        }
        
        route = route_map.get(phase)
        if route is None:
            logger.error(f"Unknown route phase: '{phase}'. Valid: {list(route_map.keys())}")
            raise ValueError(f"TaskRouter returned unknown phase: '{phase}'")
        return route

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
            
            # API 호환을 위해 결과 반환 (HIL 데이터는 output에 포함하지 않음)
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

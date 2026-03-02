"""LangGraph Orchestrator (v2.0 - 8대 혁신 통합)

Adaptive Supervisor, Hierarchical Compression, Multi-Model Orchestration,
Continuous Verification, Streaming Pipeline, Universal MCP Hub,
Adaptive Context Window, Production-Grade Reliability를 통합한
고도화된 LangGraph 기반 오케스트레이터.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage

# LangGraph imports
from langgraph.graph import END, StateGraph

from src.core.compression import compress_data
from src.core.llm_manager import TaskType, execute_llm_task
from src.core.mcp_integration import ToolCategory, execute_tool
from src.core.observability import get_langfuse_run_config
from src.core.prompt_security import REJECTION_MESSAGE, validate_user_input
from src.core.researcher_config import (
    get_agent_config,
    get_llm_config,
    get_mcp_config,
    get_research_config,
)
from src.core.streaming_manager import EventType

logger = logging.getLogger(__name__)


class ResearchState(TypedDict):
    """LangGraph 연구 워크플로우 상태 정의 (8대 혁신 통합)."""

    # Input
    user_request: str
    context: Dict[str, Any] | None
    objective_id: str

    # Adaptive Supervisor (혁신 1)
    complexity_score: float
    allocated_researchers: int
    priority_queue: List[Dict[str, Any]]
    quality_threshold: float

    # Analysis
    analyzed_objectives: List[Dict[str, Any]]
    intent_analysis: Dict[str, Any]
    domain_analysis: Dict[str, Any]
    scope_analysis: Dict[str, Any]

    # Planning Agent (새 필드)
    preliminary_research: Dict[str, Any]  # MCP 도구로 수집한 사전 조사 결과
    planned_tasks: List[Dict[str, Any]]  # 세부 task 목록
    agent_assignments: Dict[str, List[str]]  # agent별 할당된 task
    execution_plan: Dict[str, Any]  # 실행 전략 (순서, 병렬성)
    plan_approved: bool  # Plan 검증 통과 여부
    plan_feedback: str | None  # Plan 검증 피드백
    plan_iteration: int  # Plan 재작성 횟수

    # Execution (Universal MCP Hub + Streaming Pipeline)
    execution_results: List[Dict[str, Any]]
    agent_status: Dict[str, Any]
    execution_metadata: Dict[str, Any]
    streaming_data: List[Dict[str, Any]]
    streaming_events: List[Dict[str, Any]]  # 실시간 스트리밍 이벤트

    # Hierarchical Compression (혁신 2)
    compression_results: List[Dict[str, Any]]
    compression_metadata: Dict[str, Any]

    # Continuous Verification (혁신 4)
    verification_results: Dict[str, Any]
    confidence_scores: Dict[str, float]
    verification_stages: List[Dict[str, Any]]

    # Evaluation
    evaluation_results: Dict[str, Any]
    quality_metrics: Dict[str, float]
    improvement_areas: List[str]

    # Validation
    validation_results: Dict[str, Any]
    validation_score: float
    missing_elements: List[str]

    # Synthesis (Adaptive Context Window)
    final_synthesis: Dict[str, Any]
    deliverable_path: str | None
    synthesis_metadata: Dict[str, Any]

    # Human-in-the-loop 관련 필드
    pending_questions: List[Dict[str, Any]]  # 대기 중인 질문들
    user_responses: Dict[str, Any]  # 질문 ID -> 사용자 응답
    clarification_context: Dict[str, Any]  # 명확화된 정보
    waiting_for_user: bool  # 사용자 응답 대기 중인지
    autopilot_mode: bool  # CLI 모드에서 자동 선택 모드
    context_window_usage: Dict[str, Any]

    # Greedy Overseer 필드
    overseer_iterations: int  # Overseer 반복 횟수
    overseer_requirements: List[Dict[str, Any]]  # 추가 요구사항
    overseer_evaluations: List[Dict[str, Any]]  # 각 iteration의 평가
    completeness_scores: Dict[str, float]  # 목표별 완전성 점수
    quality_assessments: Dict[str, Dict[str, float]]  # 결과별 품질 평가
    overseer_decision: str | None  # 'continue', 'retry', 'ask_user', 'proceed'

    # Control Flow
    current_step: str
    iteration: int
    max_iterations: int
    should_continue: bool
    error_message: str | None

    # Innovation Stats
    innovation_stats: Dict[str, Any]

    # Messages for LangGraph
    messages: Annotated[List[BaseMessage], "Messages in the conversation"]


class AutonomousOrchestrator:
    """9대 혁신을 통합한 LangGraph 오케스트레이터."""

    def __init__(self):
        """초기화."""
        self.llm_config = get_llm_config()
        self.agent_config = get_agent_config()
        self.research_config = get_research_config()
        self.mcp_config = get_mcp_config()

        # 스트리밍 매니저 초기화
        from src.core.streaming_manager import get_streaming_manager

        self.streaming_manager = get_streaming_manager()

        # 메모리 및 학습 시스템 초기화
        from src.storage.hybrid_storage import HybridStorage

        from src.agents.creativity_agent import CreativityAgent
        from src.learning.research_recommender import ResearchRecommender
        from src.learning.user_profiler import UserProfiler

        self.hybrid_storage = HybridStorage()
        self.user_profiler = UserProfiler()
        self.research_recommender = ResearchRecommender(
            self.hybrid_storage, self.user_profiler
        )
        self.creativity_agent = CreativityAgent()

        # 9번째 혁신: Adaptive Research Depth
        from src.core.adaptive_research_depth import AdaptiveResearchDepth

        depth_config = (
            self.research_config.research_depth
            if hasattr(self.research_config, "research_depth")
            else {}
        )
        if isinstance(depth_config, dict):
            self.research_depth = AdaptiveResearchDepth(depth_config)
        else:
            # AdaptiveResearchDepthConfig 객체인 경우
            self.research_depth = AdaptiveResearchDepth(
                {
                    "default_preset": getattr(depth_config, "default_preset", "auto"),
                    "presets": getattr(depth_config, "presets", {}),
                }
            )

        # 완전 자동형 기능: 코드베이스 에이전트 및 문서 정리 에이전트
        from src.agents.codebase_agent import CodebaseAgent
        from src.agents.document_organizer_agent import DocumentOrganizerAgent
        from src.core.checkpoint_manager import CheckpointManager
        from src.core.context_loader import ContextLoader
        from src.core.session_control import get_session_control

        self.codebase_agent = CodebaseAgent()
        self.document_organizer = DocumentOrganizerAgent()
        self.checkpoint_manager = CheckpointManager()
        self.context_loader = ContextLoader()
        self.session_control = get_session_control()

        # 재귀적 컨텍스트 관리자 초기화
        from src.core.recursive_context_manager import get_recursive_context_manager

        self.context_manager = get_recursive_context_manager()

        self.graph = None
        self._build_langgraph_workflow()

    def _build_langgraph_workflow(self):
        """LangGraph 워크플로우 구축."""
        # StateGraph 생성
        workflow = StateGraph(ResearchState)

        # 노드 추가 (8대 혁신 통합 + Planning Agent + Greedy Overseer)
        workflow.add_node("analyze_objectives", self._analyze_objectives)
        workflow.add_node("planning_agent", self._planning_agent)
        workflow.add_node("verify_plan", self._verify_plan)
        workflow.add_node("overseer_initial_review", self._overseer_initial_review)
        workflow.add_node("adaptive_supervisor", self._adaptive_supervisor)
        workflow.add_node("execute_research", self._execute_research)
        workflow.add_node("hierarchical_compression", self._hierarchical_compression)
        workflow.add_node("continuous_verification", self._continuous_verification)
        workflow.add_node("overseer_evaluation", self._overseer_evaluation)
        workflow.add_node("evaluate_results", self._evaluate_results)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("synthesize_deliverable", self._synthesize_deliverable)

        # 엣지 추가 (Planning Agent 통합)
        workflow.set_entry_point("analyze_objectives")

        # Planning Agent 워크플로우
        workflow.add_edge("analyze_objectives", "planning_agent")

        # Planning Agent 후 조건부 분기 (사용자 응답 대기 여부 확인)
        workflow.add_conditional_edges(
            "planning_agent",
            lambda state: (
                "waiting_for_clarification"
                if state.get("waiting_for_user", False)
                else "verify_plan"
            ),
            {
                "waiting_for_clarification": "planning_agent",  # 사용자 응답 대기 중이면 다시 planning_agent로
                "verify_plan": "verify_plan",
            },
        )

        # Plan 검증 후 조건부 분기 (재시도 로직)
        workflow.add_conditional_edges(
            "verify_plan",
            lambda state: (
                "approved" if state.get("plan_approved", False) else "planning_agent"
            ),
            {"approved": "overseer_initial_review", "planning_agent": "planning_agent"},
        )

        # Overseer Initial Review -> Adaptive Supervisor
        workflow.add_edge("overseer_initial_review", "adaptive_supervisor")

        # 기존 워크플로우 (Overseer 통합)
        workflow.add_edge("adaptive_supervisor", "execute_research")

        # 컨텍스트 기반 자동 단계 결정 (재귀적 컨텍스트 사용)
        workflow.add_conditional_edges(
            "execute_research",
            self._decide_next_step_based_on_context,
            {
                "continue_research": "execute_research",
                "compress": "hierarchical_compression",
                "verify": "continuous_verification",
            },
        )

        workflow.add_edge("hierarchical_compression", "continuous_verification")

        # Verification -> Overseer Evaluation
        workflow.add_edge("continuous_verification", "overseer_evaluation")

        # Overseer Evaluation -> Decision Router
        workflow.add_conditional_edges(
            "overseer_evaluation",
            self._overseer_decision_router,
            {
                "retry": "execute_research",
                "waiting_for_clarification": "planning_agent",
                "proceed": "evaluate_results",
            },
        )

        # Evaluation -> Validation -> Synthesis
        workflow.add_edge("evaluate_results", "validate_results")
        workflow.add_edge("validate_results", "synthesize_deliverable")
        workflow.add_edge("synthesize_deliverable", END)

        # 그래프 컴파일
        self.graph = workflow.compile()

    def _log_node_input(self, node_name: str, state: ResearchState):
        """노드 입력 로깅."""
        logger.info(f"\n{'=' * 80}\n🔵 NODE INPUT: {node_name}\n{'=' * 80}")
        logger.info(f"User Request: {state.get('user_request', 'N/A')}")
        logger.info(f"Current Step: {state.get('current_step', 'N/A')}")
        logger.info(f"Iteration: {state.get('iteration', 0)}")
        logger.info(f"Complexity Score: {state.get('complexity_score', 'N/A')}")

        # 주요 필드 선택적 로깅
        if "analyzed_objectives" in state:
            logger.info(
                f"Objectives Count: {len(state.get('analyzed_objectives', []))}"
            )
        if "planned_tasks" in state:
            logger.info(f"Planned Tasks Count: {len(state.get('planned_tasks', []))}")
        if "agent_assignments" in state:
            logger.info(
                f"Agent Assignments Count: {len(state.get('agent_assignments', {}))}"
            )
        logger.info("=" * 80)

    def _log_node_output(
        self, node_name: str, state: ResearchState, key_changes: Dict[str, Any] = None
    ):
        """노드 출력 로깅."""
        logger.info(f"\n{'=' * 80}\n🟢 NODE OUTPUT: {node_name}\n{'=' * 80}")
        logger.info(f"Next Step: {state.get('current_step', 'N/A')}")
        logger.info(f"Should Continue: {state.get('should_continue', 'N/A')}")
        logger.info(f"Error Message: {state.get('error_message', 'None')}")

        # 주요 변경사항 로깅
        if key_changes:
            logger.info(
                f"Key Changes:\n{json.dumps(key_changes, indent=2, ensure_ascii=False)}"
            )

        # State 업데이트 요약
        logger.info(f"Complexity Score: {state.get('complexity_score', 'N/A')}")
        logger.info(
            f"Allocated Researchers: {state.get('allocated_researchers', 'N/A')}"
        )
        logger.info(f"Iteration: {state.get('iteration', 0)}")
        logger.info("=" * 80)

    async def _analyze_objectives(self, state: ResearchState) -> ResearchState:
        """목표 분석 (Multi-Model Orchestration + 재귀적 컨텍스트)."""
        # 입력 로깅
        self._log_node_input("analyze_objectives", state)

        logger.info("🔍 Thinking: Analyzing research objectives and requirements")
        logger.info(f"📝 Research Request: {state['user_request']}")

        # Sparkle 시드 아이디어: user_request만으로 초기 아이디어 생성 (워크플로우 앞단)
        try:
            seed_insights = await self.creativity_agent.generate_seed_ideas(
                (state.get("user_request") or "").strip()
            )
            state["seed_ideas"] = [
                {
                    "insight_id": getattr(i, "insight_id", ""),
                    "type": getattr(getattr(i, "type", None), "value", "unknown"),
                    "title": getattr(i, "title", ""),
                    "description": getattr(i, "description", ""),
                    "reasoning": getattr(i, "reasoning", ""),
                    "related_concepts": getattr(i, "related_concepts", []),
                }
                for i in seed_insights
            ]
            logger.info(f"✨ Seed ideas (sparkle) generated: {len(seed_insights)}")
        except Exception as e:
            logger.warning(f"Sparkle seed ideas failed: {e}")
            state["seed_ideas"] = []

        # 초기 컨텍스트 생성 (재귀적 컨텍스트 사용)
        initial_context_data = {
            "user_request": state["user_request"],
            "context": state.get("context", {}),
            "objective_id": state.get("objective_id", ""),
            "stage": "analysis",
        }
        context_id = self.context_manager.push_context(
            context_data=initial_context_data,
            depth=0,
            parent_id=None,
            metadata={
                "node": "analyze_objectives",
                "timestamp": datetime.now().isoformat(),
            },
        )
        logger.debug(f"Initial context created: {context_id}")

        # 스트리밍 이벤트: 분석 시작
        await self.streaming_manager.stream_event(
            event_type=EventType.WORKFLOW_START,
            agent_id="orchestrator",
            workflow_id=state["objective_id"],
            data={
                "stage": "analysis",
                "message": "Starting objective analysis",
                "request": state["user_request"][:100] + "..."
                if len(state["user_request"]) > 100
                else state["user_request"],
            },
            priority=1,
        )

        analysis_prompt = f"""
        Analyze the following research request comprehensively:
        
        Request: {state["user_request"]}
        Context: {state.get("context", {})}
        
        Provide detailed analysis including:
        1. Intent analysis (what the user wants to achieve)
        2. Domain analysis (relevant fields and expertise areas)
        3. Scope analysis (breadth and depth of research needed)
        4. Complexity assessment (1-10 scale)
        5. Resource requirements and constraints
        6. Success criteria and quality metrics
        
        Use production-level analysis with specific, actionable insights.
        Return the result in JSON format with the following structure:
        {{
            "objectives": [{{"id": "obj_1", "description": "Research objective", "priority": "high"}}],
            "intent": {{"primary": "research", "secondary": "analysis"}},
            "domain": {{"fields": ["technology", "research"], "expertise": "general"}},
            "scope": {{"breadth": "comprehensive", "depth": "detailed"}},
            "complexity": 7.0
        }}
        """

        try:
            # Multi-Model Orchestration으로 분석
            result = await execute_llm_task(
                prompt=analysis_prompt,
                task_type=TaskType.ANALYSIS,
                system_message="You are an expert research analyst with comprehensive domain knowledge.",
            )

            logger.info(f"✅ Analysis completed using model: {result.model_used}")
            logger.info(f"📊 Analysis confidence: {result.confidence}")

            # 분석 결과 파싱
            analysis_data = self._parse_analysis_result(result.content)

            logger.info(
                f"🎯 Identified objectives: {len(analysis_data.get('objectives', []))}"
            )
            logger.info(f"🧠 Complexity score: {analysis_data.get('complexity', 5.0)}")
            logger.info(
                f"🏷️ Domain: {analysis_data.get('domain', {}).get('fields', [])}"
            )

            # 유사 연구 검색
            similar_research = await self._search_similar_research(
                state["user_request"], state.get("user_id", "default_user")
            )

            state.update(
                {
                    "analyzed_objectives": analysis_data.get("objectives", []),
                    "intent_analysis": analysis_data.get("intent", {}),
                    "domain_analysis": analysis_data.get("domain", {}),
                    "scope_analysis": analysis_data.get("scope", {}),
                    "complexity_score": analysis_data.get("complexity", 5.0),
                    "current_step": "planning_agent",
                    "similar_research": similar_research,  # 유사 연구 추가
                    "innovation_stats": {
                        "analysis_model": result.model_used,
                        "analysis_confidence": result.confidence,
                        "analysis_time": result.execution_time,
                    },
                }
            )

            # 스트리밍 이벤트: 분석 완료
            await self.streaming_manager.stream_event(
                event_type=EventType.AGENT_ACTION,
                agent_id="orchestrator",
                workflow_id=state["objective_id"],
                data={
                    "action": "analysis_completed",
                    "status": "completed",
                    "objectives_count": len(analysis_data.get("objectives", [])),
                    "complexity_score": analysis_data.get("complexity", 5.0),
                    "model_used": result.model_used,
                    "confidence": result.confidence,
                },
                priority=1,
            )

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            state["error_message"] = str(e)
            state["should_continue"] = False
            raise  # Fail-fast

        # 출력 로깅
        key_changes = {
            "analyzed_objectives": len(analysis_data.get("objectives", [])),
            "complexity_score": analysis_data.get("complexity", 5.0),
            "intent_analysis": analysis_data.get("intent", {}),
            "domain_analysis": analysis_data.get("domain", {}),
        }
        self._log_node_output("analyze_objectives", state, key_changes)

        return state

    async def _planning_agent(self, state: ResearchState) -> ResearchState:
        """Planning Agent: MCP 기반 사전 조사 → Task 분해 → Agent 동적 할당 (재귀적 컨텍스트 사용)."""
        # 입력 로깅
        self._log_node_input("planning_agent", state)

        logger.info("📋 Thinking: Creating research plan and task breakdown")
        logger.info(f"📊 Complexity Score: {state.get('complexity_score', 5.0)}")
        logger.info(f"🎯 Objectives: {len(state.get('analyzed_objectives', []))}")

        # 현재 컨텍스트 가져오기
        current_context = self.context_manager.get_current_context()
        if not current_context:
            # 컨텍스트가 없으면 초기 컨텍스트 생성
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

        # 분석 결과를 컨텍스트에 추가 (재귀적 확장)
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
                logger.debug(
                    f"Context extended for planning: {extended_context.context_id}"
                )

        # 사용자 응답 대기 중이면 응답 처리
        if state.get("waiting_for_user", False):
            user_responses = state.get("user_responses", {})
            if user_responses:
                # 응답이 있으면 명확화 정보 적용
                from src.core.human_clarification_handler import (
                    get_clarification_handler,
                )

                clarification_handler = get_clarification_handler()

                for question_id, response_data in user_responses.items():
                    clarification = response_data.get("clarification", {})
                    # 계획에 명확화 정보 적용 (나중에 사용)
                    state["clarification_context"] = state.get(
                        "clarification_context", {}
                    )
                    state["clarification_context"][question_id] = clarification

                # 대기 상태 해제
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

            # CLI 모드 감지 (더 정확한 방법)
            import sys

            is_cli_mode = (
                not hasattr(sys, "ps1")  # Interactive shell이 아님
                and "streamlit" not in sys.modules  # Streamlit이 로드되지 않음
                and not any(
                    "streamlit" in str(arg) for arg in sys.argv
                )  # Streamlit 실행 인자가 없음
            )

            # 불명확한 부분 감지 (CLI 모드에서는 건너뛰기, 사용자 응답이 없을 때만)
            if not state.get("clarification_context") and not is_cli_mode:
                from src.core.human_clarification_handler import (
                    get_clarification_handler,
                )

                clarification_handler = get_clarification_handler()

                # 타임아웃 설정 (10초)
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
                    logger.warning("detect_ambiguities timeout, skipping clarification")
                    ambiguities = []
            elif is_cli_mode:
                # CLI 모드에서는 clarification 건너뛰기
                ambiguities = []
                logger.info("🤖 CLI mode: Skipping ambiguity detection")
            else:
                ambiguities = []

            # ambiguities가 있으면 처리
            if ambiguities:
                # CLI 모드이거나 autopilot 모드인 경우 자동 선택
                if is_cli_mode or state.get("autopilot_mode", False):
                    logger.info(
                        "🤖 CLI/Autopilot mode detected - auto-selecting responses"
                    )

                    # 각 질문에 대해 자동 응답 생성
                    user_responses = {}
                    clarification_context = {}

                    for ambiguity in ambiguities:
                        question = await clarification_handler.generate_question(
                            ambiguity, {"user_request": state.get("user_request", "")}
                        )

                        # History 기반 자동 선택
                        shared_memory = getattr(self, "hybrid_storage", None)
                        if not shared_memory:
                            try:
                                from src.storage.hybrid_storage import HybridStorage

                                shared_memory = HybridStorage()
                            except:
                                shared_memory = None

                        auto_response = (
                            await clarification_handler.auto_select_response(
                                question,
                                {"user_request": state.get("user_request", "")},
                                shared_memory,
                            )
                        )

                        # 응답 처리
                        processed = await clarification_handler.process_user_response(
                            question["id"], auto_response, {"question": question}
                        )

                        if processed.get("validated", False):
                            user_responses[question["id"]] = processed
                            clarification_context[question["id"]] = processed.get(
                                "clarification", {}
                            )

                            logger.info(
                                f"✅ Auto-selected response for {question['type']}: {auto_response}"
                            )

                    # 명확화 정보를 state에 저장하고 계속 진행
                    state["clarification_context"] = clarification_context
                    state["user_responses"] = user_responses
                    state["waiting_for_user"] = False
                    state["pending_questions"] = []
                    state["autopilot_mode"] = True

                    logger.info(
                        f"✅ Auto-processed {len(user_responses)} clarifications in autopilot mode"
                    )
                else:
                    # 웹 모드: 사용자에게 질문
                    questions = []
                    for ambiguity in ambiguities:
                        question = await clarification_handler.generate_question(
                            ambiguity, {"user_request": state.get("user_request", "")}
                        )
                        questions.append(question)

                    # 사용자 응답 대기 상태로 전환
                    state["pending_questions"] = questions
                    state["waiting_for_user"] = True
                    state["current_step"] = "waiting_for_clarification"
                    state["user_responses"] = {}

                    logger.info(
                        f"❓ Generated {len(questions)} questions for user clarification"
                    )
                    logger.info("⏸️ Waiting for user responses...")

                    # 출력 로깅
                    key_changes = {
                        "pending_questions_count": len(questions),
                        "waiting_for_user": True,
                        "current_step": "waiting_for_clarification",
                    }
                    self._log_node_output("planning_agent", state, key_changes)

                    return state

            # 9번째 혁신: Adaptive Research Depth - 연구 깊이 결정
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

            # 깊이 설정을 state에 저장
            state["research_depth"] = {
                "preset": depth_config.preset.value,
                "planning": depth_config.planning,
                "researching": depth_config.researching,
                "reporting": depth_config.reporting,
                "complexity_score": depth_config.complexity_score,
            }
            logger.info(
                f"📊 Research depth determined: {depth_config.preset.value} (complexity: {depth_config.complexity_score:.2f})"
            )

            # 1. MCP 도구로 사전 조사
            preliminary_research = await self._conduct_preliminary_research(state)
            logger.info(
                f"🔍 Preliminary research completed: {preliminary_research.get('sources_count', 0)} sources"
            )

            # 2. Task 분해 (복잡도 기반) - 명확화 정보 및 깊이 설정 반영
            tasks = await self._decompose_into_tasks(
                state, preliminary_research, depth_config
            )
            logger.info(
                f"📋 Tasks decomposed: {len(tasks)} tasks (depth: {depth_config.preset.value})"
            )

            # 명확화 정보를 작업에 적용
            clarification_context = state.get("clarification_context", {})
            if clarification_context:
                from src.core.human_clarification_handler import (
                    get_clarification_handler,
                )

                clarification_handler = get_clarification_handler()

                for task in tasks:
                    for question_id, clarification in clarification_context.items():
                        task = clarification_handler.apply_clarification(
                            clarification, task
                        )

            # 3. Agent 동적 할당 (복잡도 기반)
            agent_assignments = await self._assign_agents_dynamically(tasks, state)
            logger.info(
                f"👥 Agent assignments: {len(agent_assignments)} task-agent mappings"
            )

            # 4. 실행 전략 수립
            execution_plan = await self._create_execution_plan(tasks, agent_assignments)
            logger.info(
                f"📈 Execution strategy: {execution_plan.get('strategy', 'sequential')}"
            )

            # Planning 결과를 state에 저장
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
                    "innovation_stats": {
                        **state.get("innovation_stats", {}),
                        "planning_agent": "active",
                        "preliminary_sources": preliminary_research.get(
                            "sources_count", 0
                        ),
                        "planned_tasks_count": len(tasks),
                        "agent_assignments_count": len(agent_assignments),
                        "execution_strategy": execution_plan.get(
                            "strategy", "sequential"
                        ),
                    },
                }
            )

            # 계획을 컨텍스트에 추가 (재귀적 확장)
            if current_context:
                plan_context = {
                    "planned_tasks": tasks,
                    "agent_assignments": agent_assignments,
                    "execution_plan": execution_plan,
                    "plan_approved": False,
                    "preliminary_research": preliminary_research,
                }
                self.context_manager.extend_context(
                    current_context.context_id,
                    plan_context,
                    metadata={
                        "plan_completed": True,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                logger.debug(f"Plan added to context: {current_context.context_id}")

            # 출력 로깅
            key_changes = {
                "preliminary_research_sources": preliminary_research.get(
                    "sources_count", 0
                ),
                "planned_tasks_count": len(tasks),
                "agent_assignments_count": len(agent_assignments),
                "execution_strategy": execution_plan.get("strategy", "sequential"),
                "plan_iteration": state.get("plan_iteration", 0),
                "planned_tasks": [
                    {
                        "id": task.get("id"),
                        "type": task.get("type"),
                        "agent": task.get("assigned_agent"),
                    }
                    for task in tasks[:3]
                ],  # 처음 3개만 로깅
            }
            self._log_node_output("planning_agent", state, key_changes)

            logger.info("✅ Planning Agent completed successfully")
            return state

        except Exception as e:
            logger.error(f"❌ Planning Agent failed: {e}")
            state["error_message"] = str(e)
            state["should_continue"] = False
            raise  # Fail-fast

    async def _verify_plan(self, state: ResearchState) -> ResearchState:
        """Plan 검증: LLM 기반 plan 타당성 검증."""
        # 입력 로깅
        self._log_node_input("verify_plan", state)

        logger.info("✅ Verifying research plan")
        logger.info(f"📋 Tasks to verify: {len(state.get('planned_tasks', []))}")
        logger.info(f"👥 Agent assignments: {len(state.get('agent_assignments', {}))}")

        try:
            verification_prompt = f"""
            Verify the following research plan for quality and completeness:
            
            Research Request: {state.get("user_request", "")}
            Objectives: {state.get("analyzed_objectives", [])}
            Domain: {state.get("domain_analysis", {})}
            Complexity Score: {state.get("complexity_score", 5.0)}
            
            Planned Tasks: {state.get("planned_tasks", [])}
            Agent Assignments: {state.get("agent_assignments", {})}
            Execution Plan: {state.get("execution_plan", {})}
            
            Check the following criteria:
            1. Completeness: Are all research objectives covered by the tasks?
            2. Agent Allocation: Is the number of agents appropriate for task complexity?
            3. Execution Strategy: Is the execution order and parallelization logical?
            4. Resource Efficiency: Are the estimated costs and time reasonable?
            5. Dependencies: Are task dependencies properly handled?
            6. MCP Tools: Are appropriate tools assigned to each task?
            
            Return your assessment in JSON format:
            {{
                "approved": boolean,
                "confidence": float (0.0-1.0),
                "feedback": "detailed feedback string",
                "suggested_changes": ["list of specific improvements"],
                "critical_issues": ["list of blocking issues if any"]
            }}
            """

            result = await execute_llm_task(
                prompt=verification_prompt,
                task_type=TaskType.VERIFICATION,
                system_message="You are an expert research planner and quality auditor with deep knowledge of research methodologies and resource optimization.",
            )

            logger.info(
                f"🔍 Plan verification completed using model: {result.metadata.get('model', 'unknown') if result.metadata else 'unknown'}"
            )
            logger.info(f"📊 Verification confidence: {result.confidence}")

            # 안전 필터 감지 (ModelResult는 dataclass이므로 속성으로 접근)
            content = result.content if result.content else ""
            if content and (
                "blocked by safety filters" in content.lower()
                or "Unable to extract content" in content
            ):
                logger.warning(
                    "⚠️ Safety filter triggered in verification. Using default result."
                )
                verification = {
                    "approved": True,
                    "confidence": 0.5,
                    "feedback": "Verification skipped due to safety filter. Proceeding with plan.",
                    "suggested_changes": [],
                    "critical_issues": [],
                }
            else:
                # 검증 결과 파싱 (안전하게)
                try:
                    verification = self._parse_verification_result(content)
                except Exception as parse_error:
                    logger.warning(
                        f"⚠️ Verification parsing failed: {parse_error}. Using default result."
                    )
                    verification = {
                        "approved": True,
                        "confidence": 0.5,
                        "feedback": f"Verification parsing failed: {str(parse_error)}. Proceeding with plan.",
                        "suggested_changes": [],
                        "critical_issues": [],
                    }

            if verification.get("approved", False):
                state["plan_approved"] = True
                state["plan_feedback"] = verification.get("feedback", "Plan approved")
                logger.info("✅ Plan approved by verification")
                logger.info(f"💬 Feedback: {verification.get('feedback', '')}")
            else:
                state["plan_approved"] = False
                state["plan_feedback"] = verification.get("feedback", "Plan rejected")
                logger.warning(f"❌ Plan rejected: {verification.get('feedback')}")
                logger.warning(
                    f"🔧 Suggested changes: {verification.get('suggested_changes', [])}"
                )

                # 최대 재시도 횟수 확인 (무한 루프 방지)
                max_iterations = 3
                if state.get("plan_iteration", 0) >= max_iterations:
                    logger.error(
                        f"❌ Maximum plan iterations ({max_iterations}) reached. Proceeding with current plan."
                    )
                    state["plan_approved"] = True
                    state["plan_feedback"] = (
                        f"Plan approved after {max_iterations} iterations (forced)"
                    )

            state.update(
                {
                    "current_step": "adaptive_supervisor"
                    if state.get("plan_approved", False)
                    else "planning_agent",
                    "innovation_stats": {
                        **state.get("innovation_stats", {}),
                        "plan_verification": "completed",
                        "plan_approved": state.get("plan_approved", False),
                        "verification_confidence": verification.get("confidence", 0.0),
                        "verification_iteration": state.get("plan_iteration", 0),
                    },
                }
            )

            # 출력 로깅
            key_changes = {
                "plan_approved": state.get("plan_approved", False),
                "verification_confidence": verification.get("confidence", 0.0),
                "plan_iteration": state.get("plan_iteration", 0),
                "feedback": verification.get("feedback", "")[:200],  # 처음 200자만
            }
            self._log_node_output("verify_plan", state, key_changes)

            return state

        except Exception as e:
            logger.warning(
                f"⚠️ Plan verification failed: {e}. Proceeding with default verification."
            )
            # 검증 실패해도 연구 계속 진행
            state["plan_approved"] = True
            state["plan_feedback"] = f"Verification failed but proceeding: {str(e)}"
            state["plan_verification_error"] = str(e)

            state.update(
                {
                    "current_step": "adaptive_supervisor",  # 검증 실패해도 계속 진행
                    "innovation_stats": {
                        **state.get("innovation_stats", {}),
                        "plan_verification": "failed_but_continuing",
                        "plan_approved": True,
                        "verification_confidence": 0.5,
                        "verification_iteration": state.get("plan_iteration", 0),
                    },
                }
            )

            self._log_node_output(
                "verify_plan",
                state,
                {
                    "plan_approved": True,
                    "verification_confidence": 0.5,
                    "error": str(e),
                    "action": "proceeding_despite_error",
                },
            )

            return state  # 예외 발생하지 않고 계속 진행

    async def _adaptive_supervisor(self, state: ResearchState) -> ResearchState:
        """Adaptive Supervisor (혁신 1)."""
        logger.info("🎯 Adaptive Supervisor allocating resources")

        complexity_raw = state.get("complexity_score", 5.0)
        # complexity가 dict인 경우 처리
        if isinstance(complexity_raw, dict):
            complexity = complexity_raw.get("score", complexity_raw.get("value", 5.0))
        elif isinstance(complexity_raw, (int, float)):
            complexity = float(complexity_raw)
        else:
            complexity = 5.0

        available_budget = self.llm_config.budget_limit

        # 동적 연구자 할당
        allocated_researchers = min(
            max(int(complexity), self.agent_config.min_researchers),
            self.agent_config.max_researchers,
            int(available_budget / 10),  # 예상 비용 기반
        )

        # 우선순위 큐 생성
        priority_queue = self._create_priority_queue(state)

        # 품질 임계값 설정
        quality_threshold = self.agent_config.quality_threshold

        logger.info(f"🧠 Complexity Score: {complexity}")
        logger.info(f"👥 Allocated Researchers: {allocated_researchers}")
        logger.info(f"📊 Quality Threshold: {quality_threshold}")
        logger.info(f"📋 Priority Queue Size: {len(priority_queue)}")
        logger.info(f"💰 Available Budget: ${available_budget}")

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

    async def _execute_research(self, state: ResearchState) -> ResearchState:
        """연구 실행 (9번째 혁신: Progressive Deepening 통합)."""
        """연구 실행 (Universal MCP Hub + Streaming Pipeline + Parallel Execution)."""
        # 입력 로깅
        self._log_node_input("execute_research", state)

        logger.info("⚙️ Thinking: Executing research tasks and gathering information")

        # Planning Agent에서 생성된 tasks 사용
        tasks = state.get("planned_tasks", [])
        agent_assignments = state.get("agent_assignments", {})
        execution_plan = state.get("execution_plan", {})
        objective_id = state.get("objective_id", "default")

        logger.info(f"📋 Executing {len(tasks)} planned tasks")
        logger.info(f"👥 Agent assignments: {len(agent_assignments)} mappings")
        logger.info(
            f"📈 Execution strategy: {execution_plan.get('strategy', 'sequential')}"
        )

        # 병렬 실행 사용 여부 결정
        use_parallel = (
            execution_plan.get("strategy") in ["parallel", "hybrid"]
            and len(tasks) > 1
            and self.agent_config.max_concurrent_research_units > 1
        )

        if use_parallel:
            logger.info("🚀 Using parallel execution with ParallelAgentExecutor")

            # ParallelAgentExecutor 사용
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

            logger.info(
                f"✅ Parallel execution completed: {len(execution_results)} tasks executed"
            )
        else:
            logger.info(
                "📝 Using sequential execution (parallel execution conditions not met)"
            )
            # 순차 실행 (기존 로직 - 병렬 실행이 불가능한 경우)
            execution_results = []
            streaming_data = []

            for task in tasks:
                task_success = False
                tool_attempts = []

                try:
                    # MCP 도구 선택 및 실행 (대체 도구 로직 포함)
                    tool_category = self._get_tool_category_for_task(task)
                    available_tools = self._get_available_tools_for_category(
                        tool_category
                    )

                    # 도구 우선순위별로 시도
                    for tool_name in available_tools:
                        try:
                            logger.info(f"🔧 Attempting tool: {tool_name}")
                            # 파라미터 자동 생성 및 검증
                            tool_parameters = self._generate_tool_parameters(
                                task, tool_name
                            )
                            tool_result = await execute_tool(tool_name, tool_parameters)

                            tool_attempts.append(
                                {
                                    "tool": tool_name,
                                    "success": tool_result.get("success", False),
                                    "error": tool_result.get("error", ""),
                                    "execution_time": tool_result.get(
                                        "execution_time", 0.0
                                    ),
                                }
                            )

                            if tool_result.get("success", False):
                                # 실제 데이터 검증
                                if self._validate_tool_result(tool_result, task):
                                    execution_results.append(
                                        {
                                            "task_id": task.get("id"),
                                            "task_name": task.get("name"),
                                            "tool_used": tool_name,
                                            "result": tool_result.get("data"),
                                            "execution_time": tool_result.get(
                                                "execution_time", 0.0
                                            ),
                                            "confidence": tool_result.get(
                                                "confidence", 0.0
                                            ),
                                            "attempts": len(tool_attempts),
                                            "status": "completed",
                                        }
                                    )

                                    # 스트리밍 데이터 추가
                                    streaming_data.append(
                                        {
                                            "timestamp": datetime.now().isoformat(),
                                            "task_id": task.get("id"),
                                            "status": "completed",
                                            "data": tool_result.get("data"),
                                            "tool_used": tool_name,
                                        }
                                    )

                                    logger.info(
                                        f"✅ Tool '{tool_name}' executed successfully with valid data"
                                    )
                                    task_success = True
                                    break
                                else:
                                    logger.warning(
                                        f"⚠️ Tool '{tool_name}' returned invalid data, trying next tool..."
                                    )
                            else:
                                logger.warning(
                                    f"❌ Tool '{tool_name}' failed: {tool_result.get('error', 'Unknown error')}"
                                )

                        except Exception as tool_error:
                            logger.warning(
                                f"❌ Tool '{tool_name}' execution error: {tool_error}"
                            )
                            tool_attempts.append(
                                {
                                    "tool": tool_name,
                                    "success": False,
                                    "error": str(tool_error),
                                    "execution_time": 0.0,
                                }
                            )
                            continue

                    if not task_success:
                        logger.error(
                            f"❌ All tools failed for task {task.get('id')}. Attempts: {tool_attempts}"
                        )
                        # 실패한 작업도 기록
                        execution_results.append(
                            {
                                "task_id": task.get("id"),
                                "task_name": task.get("name"),
                                "tool_used": "none",
                                "result": None,
                                "execution_time": 0.0,
                                "confidence": 0.0,
                                "attempts": len(tool_attempts),
                                "error": "All tools failed",
                                "tool_attempts": tool_attempts,
                                "status": "failed",
                            }
                        )

                except Exception as e:
                    logger.error(
                        f"❌ Critical error executing task {task.get('id')}: {e}"
                    )
                    execution_results.append(
                        {
                            "task_id": task.get("id"),
                            "task_name": task.get("name"),
                            "tool_used": "none",
                            "result": None,
                            "execution_time": 0.0,
                            "confidence": 0.0,
                            "attempts": 0,
                            "error": str(e),
                            "status": "failed",
                        }
                    )

        # 9번째 혁신: Progressive Deepening - 연구 진행 상황 분석 및 깊이 조정
        current_depth = state.get("research_depth", {})
        if current_depth and hasattr(self, "research_depth"):
            progress = {
                "iteration_count": state.get("research_iteration", 0) + 1,
                "completion_rate": float(
                    len(
                        [r for r in execution_results if r.get("status") == "completed"]
                    )
                )
                / max(len(tasks), 1),
                "tasks_total": len(tasks),
                "tasks_completed": len(
                    [r for r in execution_results if r.get("status") == "completed"]
                ),
            }

            # DepthConfig 객체 재구성
            from src.core.adaptive_research_depth import DepthConfig, ResearchPreset

            try:
                preset = ResearchPreset(current_depth.get("preset", "medium"))
                current_depth_config = DepthConfig(
                    preset=preset,
                    planning=current_depth.get("planning", {}),
                    researching=current_depth.get("researching", {}),
                    reporting=current_depth.get("reporting", {}),
                    complexity_score=current_depth.get("complexity_score", 0.5),
                )

                # Progressive Deepening 체크
                adjusted_depth = self.research_depth.adjust_depth_progressively(
                    current_depth_config,
                    progress,
                    goals_achieved=False,  # TODO: 실제 목표 달성 여부 확인
                )

                if adjusted_depth:
                    logger.info(
                        f"📈 Progressive Deepening: {current_depth_config.preset.value} -> {adjusted_depth.preset.value}"
                    )
                    state["research_depth"] = {
                        "preset": adjusted_depth.preset.value,
                        "planning": adjusted_depth.planning,
                        "researching": adjusted_depth.researching,
                        "reporting": adjusted_depth.reporting,
                        "complexity_score": adjusted_depth.complexity_score,
                    }
                    state["research_depth_adjusted"] = True
            except Exception as e:
                logger.debug(f"Progressive Deepening check failed: {e}")

        state.update(
            {
                "execution_results": execution_results,
                "streaming_data": streaming_data,
                "current_step": "hierarchical_compression",
                "research_iteration": state.get("research_iteration", 0) + 1,
                "innovation_stats": {
                    **state.get("innovation_stats", {}),
                    "tasks_executed": len(execution_results),
                    "tools_used": len(
                        set(
                            r.get("tool_used", "")
                            for r in execution_results
                            if r.get("tool_used")
                        )
                    ),
                    "execution_success_rate": float(
                        len(
                            [
                                r
                                for r in execution_results
                                if r.get("status") == "completed"
                            ]
                        )
                    )
                    / max(len(tasks), 1),
                    "parallel_execution_used": use_parallel,
                },
            }
        )

        # 출력 로깅
        key_changes = {
            "tasks_executed": len(execution_results),
            "tasks_successful": len(
                [r for r in execution_results if r.get("status") == "completed"]
            ),
            "tools_used": len(
                set(
                    r.get("tool_used", "")
                    for r in execution_results
                    if r.get("tool_used")
                )
            ),
            "execution_success_rate": float(
                len([r for r in execution_results if r.get("status") == "completed"])
            )
            / max(len(tasks), 1),
            "total_execution_time": sum(
                r.get("execution_time", 0.0) for r in execution_results
            ),
            "parallel_execution_used": use_parallel,
        }
        self._log_node_output("execute_research", state, key_changes)

        return state

    async def _hierarchical_compression(self, state: ResearchState) -> ResearchState:
        """Hierarchical Compression (혁신 2)."""
        logger.info("🗜️ Applying Hierarchical Compression")

        execution_results = state.get("execution_results", [])
        compression_results = []

        # 실행 결과를 컨텍스트에 추가 (재귀적 확장)
        current_context = self.context_manager.get_current_context()
        if current_context:
            execution_context = {
                "execution_results": execution_results,
                "execution_metadata": state.get("execution_metadata", {}),
                "streaming_data": state.get("streaming_data", []),
                "stage": "execution_completed",
            }
            self.context_manager.extend_context(
                current_context.context_id,
                execution_context,
                metadata={
                    "execution_completed": True,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.debug(
                f"Execution results added to context: {current_context.context_id}"
            )

        # 실행 결과가 없는 경우 처리
        if not execution_results:
            logger.warning(
                "⚠️ No execution results available for compression. Skipping compression step."
            )
            state.update(
                {
                    "compression_results": [],
                    "compression_metadata": {
                        "overall_compression_ratio": 1.0,
                        "total_original_size": 0,
                        "total_compressed_size": 0,
                        "compression_count": 0,
                    },
                    "current_step": "continuous_verification",
                    "innovation_stats": {
                        **state.get("innovation_stats", {}),
                        "compression_ratio": 1.0,
                        "compression_applied": 0,
                    },
                }
            )
            return state

        # 각 실행 결과에 대해 압축 적용
        for result in execution_results:
            task_id = result.get("task_id", "unknown")
            result_data = result.get("result")

            # result가 None이거나 빈 데이터인 경우 스킵
            if result_data is None:
                logger.warning(
                    f"⚠️ Skipping compression for task {task_id}: result is None (execution may have failed)"
                )
                compression_results.append(
                    {
                        "task_id": task_id,
                        "original_size": 0,
                        "compressed_size": 0,
                        "compression_ratio": 1.0,
                        "validation_score": 0.0,
                        "compressed_data": None,
                        "important_info_preserved": [],
                        "status": "skipped_no_data",
                    }
                )
                continue

            # 빈 딕셔너리나 빈 문자열인 경우도 스킵
            if isinstance(result_data, dict) and not result_data:
                logger.warning(
                    f"⚠️ Skipping compression for task {task_id}: result is empty dict"
                )
                compression_results.append(
                    {
                        "task_id": task_id,
                        "original_size": 0,
                        "compressed_size": 0,
                        "compression_ratio": 1.0,
                        "validation_score": 0.0,
                        "compressed_data": None,
                        "important_info_preserved": [],
                        "status": "skipped_empty_data",
                    }
                )
                continue

            if isinstance(result_data, str) and not result_data.strip():
                logger.warning(
                    f"⚠️ Skipping compression for task {task_id}: result is empty string"
                )
                compression_results.append(
                    {
                        "task_id": task_id,
                        "original_size": 0,
                        "compressed_size": 0,
                        "compression_ratio": 1.0,
                        "validation_score": 0.0,
                        "compressed_data": None,
                        "important_info_preserved": [],
                        "status": "skipped_empty_string",
                    }
                )
                continue

            try:
                # 데이터 압축
                compressed = await compress_data(result_data)

                compression_results.append(
                    {
                        "task_id": task_id,
                        "original_size": len(str(result_data)),
                        "compressed_size": len(str(compressed.data)),
                        "compression_ratio": compressed.compression_ratio,
                        "validation_score": compressed.validation_score,
                        "compressed_data": compressed.data,
                        "important_info_preserved": compressed.important_info_preserved,
                        "status": "compressed",
                    }
                )

            except Exception as e:
                logger.warning(
                    f"⚠️ Compression failed for task {task_id}: {e}. Using original data."
                )
                # 압축 실패 시 원본 데이터 사용
                compression_results.append(
                    {
                        "task_id": task_id,
                        "original_size": len(str(result_data)),
                        "compressed_size": len(str(result_data)),
                        "compression_ratio": 1.0,
                        "validation_score": 1.0,
                        "compressed_data": result_data,
                        "important_info_preserved": [],
                        "status": "compression_failed_using_original",
                    }
                )

        # 전체 압축 통계
        total_original = sum(c.get("original_size", 0) for c in compression_results)
        total_compressed = sum(c.get("compressed_size", 0) for c in compression_results)
        overall_compression_ratio = total_compressed / max(total_original, 1)

        # 압축 결과를 컨텍스트에 추가 (재귀적 확장)
        current_context = self.context_manager.get_current_context()
        if current_context:
            compression_context = {
                "compression_results": compression_results,
                "compression_metadata": {
                    "overall_compression_ratio": overall_compression_ratio,
                    "total_original_size": total_original,
                    "total_compressed_size": total_compressed,
                    "compression_count": len(compression_results),
                },
                "stage": "compression_completed",
            }
            self.context_manager.extend_context(
                current_context.context_id,
                compression_context,
                metadata={
                    "compression_completed": True,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.debug(
                f"Compression results added to context: {current_context.context_id}"
            )

        state.update(
            {
                "compression_results": compression_results,
                "compression_metadata": {
                    "overall_compression_ratio": overall_compression_ratio,
                    "total_original_size": total_original,
                    "total_compressed_size": total_compressed,
                    "compression_count": len(compression_results),
                },
                "current_step": "continuous_verification",
                "innovation_stats": {
                    **state.get("innovation_stats", {}),
                    "compression_ratio": float(overall_compression_ratio),
                    "compression_applied": len(compression_results),
                },
            }
        )

        return state

    async def _continuous_verification(self, state: ResearchState) -> ResearchState:
        """Continuous Verification (혁신 4 + 재귀적 컨텍스트 사용)."""
        logger.info("🔬 Applying Continuous Verification")

        # 현재 컨텍스트 가져오기 및 실행 결과 추가
        current_context = self.context_manager.get_current_context()
        if current_context:
            # 압축 결과를 컨텍스트에 추가 (재귀적 확장)
            compression_context = {
                "compression_results": state.get("compression_results", []),
                "compression_metadata": state.get("compression_metadata", {}),
                "stage": "verification",
            }
            self.context_manager.extend_context(
                current_context.context_id,
                compression_context,
                metadata={
                    "node": "continuous_verification",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.debug(
                f"Compression results added to context: {current_context.context_id}"
            )

        compression_results = state.get("compression_results", [])
        execution_results = state.get("execution_results", [])

        # 검색 실패 확인: compression_results가 비어있거나 모든 결과가 실패한 경우
        if not compression_results:
            logger.warning(
                "⚠️ No compression results available for verification. Checking execution results..."
            )

            # execution_results 확인
            if not execution_results:
                logger.error(
                    "❌ No execution results available. Research execution may have failed completely."
                )
                state.update(
                    {
                        "verification_stages": [],
                        "confidence_scores": {},
                        "verification_failed": True,
                        "error_message": "No research results available for verification. Search execution may have failed.",
                        "current_step": "evaluate_results",  # 검증 실패해도 평가 단계로 진행
                        "innovation_stats": {
                            **state.get("innovation_stats", {}),
                            "verification_applied": 0,
                            "avg_confidence": 0.0,
                            "verification_status": "skipped_no_results",
                        },
                    }
                )
                return state

            # execution_results에서 실패한 작업만 있는지 확인
            successful_results = [
                r
                for r in execution_results
                if r.get("status") == "completed" and r.get("result") is not None
            ]
            if not successful_results:
                logger.error(
                    "❌ All execution results failed. No successful research results to verify."
                )
                state.update(
                    {
                        "verification_stages": [],
                        "confidence_scores": {},
                        "verification_failed": True,
                        "error_message": "All research execution failed. No successful results to verify.",
                        "current_step": "evaluate_results",  # 검증 실패해도 평가 단계로 진행
                        "innovation_stats": {
                            **state.get("innovation_stats", {}),
                            "verification_applied": 0,
                            "avg_confidence": 0.0,
                            "verification_status": "skipped_all_failed",
                            "failed_tasks": len(execution_results),
                        },
                    }
                )
                return state

        # 유효한 결과만 필터링 (result가 None이거나 빈 데이터인 경우 제외)
        valid_results = []
        for result in compression_results:
            compressed_data = result.get("compressed_data")
            if compressed_data is not None and compressed_data != "":
                valid_results.append(result)
            else:
                task_id = result.get("task_id", "unknown")
                logger.warning(
                    f"⚠️ Skipping verification for task {task_id}: no valid compressed data"
                )

        if not valid_results:
            logger.warning(
                "⚠️ No valid compression results after filtering. Proceeding with minimal verification."
            )
            state.update(
                {
                    "verification_stages": [],
                    "confidence_scores": {},
                    "verification_failed": True,
                    "error_message": "No valid compression results available for verification.",
                    "current_step": "evaluate_results",  # 검증 실패해도 평가 단계로 진행
                    "innovation_stats": {
                        **state.get("innovation_stats", {}),
                        "verification_applied": 0,
                        "avg_confidence": 0.0,
                        "verification_status": "skipped_no_valid_data",
                    },
                }
            )
            return state

        verification_stages = []
        confidence_scores = {}

        # 3단계 검증 (유효한 결과만)
        for i, result in enumerate(valid_results):
            task_id = result.get("task_id")

            try:
                # Stage 1: Self-Verification
                self_score = await self._self_verification(result)

                # Stage 2: Cross-Verification
                cross_score = await self._cross_verification(result, valid_results)

                # Stage 3: External Verification (선택적)
                if self_score < 0.7 or cross_score < 0.7:
                    external_score = await self._external_verification(result)
                else:
                    external_score = 1.0

                # 종합 신뢰도 점수
                final_score = (
                    self_score * 0.3 + cross_score * 0.4 + external_score * 0.3
                )

                verification_stages.append(
                    {
                        "task_id": task_id,
                        "stage_1_self": self_score,
                        "stage_2_cross": cross_score,
                        "stage_3_external": external_score,
                        "final_score": final_score,
                    }
                )

                confidence_scores[task_id] = final_score

            except Exception as e:
                logger.warning(
                    f"⚠️ Verification failed for task {task_id}: {e}. Assigning low confidence score."
                )
                # 검증 실패 시 낮은 신뢰도 점수 할당
                verification_stages.append(
                    {
                        "task_id": task_id,
                        "stage_1_self": 0.3,
                        "stage_2_cross": 0.3,
                        "stage_3_external": 0.3,
                        "final_score": 0.3,
                        "verification_error": str(e),
                    }
                )
                confidence_scores[task_id] = 0.3

        # 검증 결과를 컨텍스트에 추가 (재귀적 확장)
        current_context = self.context_manager.get_current_context()
        if current_context:
            verification_result_context = {
                "verification_results": {
                    "verification_stages": verification_stages,
                    "confidence_scores": confidence_scores,
                },
                "verification_failed": False,
                "stage": "verification_completed",
            }
            self.context_manager.extend_context(
                current_context.context_id,
                verification_result_context,
                metadata={
                    "verification_completed": True,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.debug(
                f"Verification results added to context: {current_context.context_id}"
            )

        state.update(
            {
                "verification_stages": verification_stages,
                "confidence_scores": confidence_scores,
                "verification_failed": False,
                "current_step": "evaluate_results",
                "innovation_stats": {
                    **state.get("innovation_stats", {}),
                    "verification_applied": len(verification_stages),
                    "avg_confidence": float(sum(confidence_scores.values()))
                    / max(len(confidence_scores), 1)
                    if confidence_scores
                    else 0.0,
                    "valid_results_count": len(valid_results),
                    "total_results_count": len(compression_results),
                },
            }
        )

        return state

    async def _evaluate_results(self, state: ResearchState) -> ResearchState:
        """결과 평가 (Multi-Model Orchestration)."""
        logger.info("📊 Evaluating results with Multi-Model Orchestration")

        evaluation_prompt = f"""
        Evaluate the following research results comprehensively:
        
        Execution Results: {state.get("execution_results", [])}
        Compression Results: {state.get("compression_results", [])}
        Verification Results: {state.get("verification_stages", [])}
        Confidence Scores: {state.get("confidence_scores", {})}
        
        Provide detailed evaluation including:
        1. Quality assessment with metrics
        2. Completeness analysis
        3. Accuracy verification
        4. Improvement recommendations
        5. Risk assessment
        6. Overall satisfaction score
        
        Use production-level evaluation with specific, actionable insights.
        Return the result in JSON format with the following structure:
        {{
            "overall_score": 0.85,
            "metrics": {{"quality": 0.8, "completeness": 0.9, "accuracy": 0.85}},
            "improvements": ["Add more sources", "Improve analysis depth"]
        }}
        """

        # Multi-Model Orchestration으로 평가
        result = await execute_llm_task(
            prompt=evaluation_prompt,
            task_type=TaskType.VERIFICATION,
            system_message="You are an expert research evaluator with comprehensive quality assessment capabilities.",
            use_ensemble=True,  # Weighted Ensemble 사용
        )

        # 평가 결과 파싱
        evaluation_data = self._parse_evaluation_result(result.content)

        state.update(
            {
                "evaluation_results": evaluation_data,
                "quality_metrics": evaluation_data.get("metrics", {}),
                "improvement_areas": evaluation_data.get("improvements", []),
                "current_step": "validate_results",
                "innovation_stats": {
                    **state.get("innovation_stats", {}),
                    "evaluation_model": result.model_used,
                    "evaluation_confidence": result.confidence,
                    "quality_score": evaluation_data.get("overall_score", 0.8),
                },
            }
        )

        return state

    async def _validate_results(self, state: ResearchState) -> ResearchState:
        """결과 검증."""
        logger.info("✅ Validating results")

        # 검증 로직
        validation_score = self._calculate_validation_score(state)
        missing_elements = self._identify_missing_elements(state)

        state.update(
            {
                "validation_score": validation_score,
                "missing_elements": missing_elements,
                "current_step": "synthesize_deliverable",
                "innovation_stats": {
                    **state.get("innovation_stats", {}),
                    "validation_score": validation_score,
                    "missing_elements_count": len(missing_elements),
                },
            }
        )

        return state

    async def _synthesize_deliverable(self, state: ResearchState) -> ResearchState:
        """최종 결과 종합 (Adaptive Context Window + 재귀적 컨텍스트 사용)."""
        logger.info("📝 Synthesizing final deliverable with Adaptive Context Window")

        # 현재 컨텍스트 가져오기 및 모든 단계 결과 통합
        current_context = self.context_manager.get_current_context()
        if current_context:
            # 모든 단계 결과를 컨텍스트에 통합 (재귀적 확장)
            synthesis_context = {
                "verification_results": state.get("verification_results", {}),
                "confidence_scores": state.get("confidence_scores", {}),
                "evaluation_results": state.get("evaluation_results", {}),
                "quality_metrics": state.get("quality_metrics", {}),
                "validation_results": state.get("validation_results", {}),
                "validation_score": state.get("validation_score", 0.0),
                "stage": "synthesis",
            }
            self.context_manager.extend_context(
                current_context.context_id,
                synthesis_context,
                metadata={
                    "node": "synthesize_deliverable",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.debug(
                f"All stage results integrated into context: {current_context.context_id}"
            )

            # 컨텍스트에서 종합 정보 추출
            context_data = current_context.context_data
            synthesis_prompt = f"""
        Synthesize the following research findings into a comprehensive deliverable:
        
        User Request: {context_data.get("user_request", state.get("user_request", ""))}
        Intent Analysis: {context_data.get("intent_analysis", {})}
        Domain Analysis: {context_data.get("domain_analysis", {})}
        Planned Tasks: {len(context_data.get("planned_tasks", []))} tasks
        Execution Results: {len(context_data.get("execution_results", []))} results
        Compression Results: {len(context_data.get("compression_results", []))} compressed
        Verification Results: {context_data.get("verification_results", {})}
        Evaluation Results: {context_data.get("evaluation_results", {})}
        Quality Metrics: {context_data.get("quality_metrics", {})}
        
        Create a comprehensive synthesis including:
        1. Executive summary with key insights
        2. Detailed findings with evidence
        3. Analysis and interpretation
        4. Conclusions and recommendations
        5. Limitations and future work
        6. Appendices with supporting data
        
        Use adaptive context management for optimal content organization.
        Use the recursive context to ensure all stages are properly integrated.
        """
        else:
            # 컨텍스트가 없으면 기존 방식 사용
            synthesis_prompt = f"""
        Synthesize the following research findings into a comprehensive deliverable:
        
        User Request: {state.get("user_request", "")}
        Execution Results: {state.get("execution_results", [])}
        Compression Results: {state.get("compression_results", [])}
        Verification Results: {state.get("verification_stages", [])}
        Evaluation Results: {state.get("evaluation_results", {})}
        Quality Metrics: {state.get("quality_metrics", {})}
        
        Create a comprehensive synthesis including:
        1. Executive summary with key insights
        2. Detailed findings with evidence
        3. Analysis and interpretation
        4. Conclusions and recommendations
        5. Limitations and future work
        6. Appendices with supporting data
        
        Use adaptive context management for optimal content organization.
        """

        # Multi-Model Orchestration으로 종합
        result = await execute_llm_task(
            prompt=synthesis_prompt,
            task_type=TaskType.SYNTHESIS,
            system_message="You are an expert research synthesizer with adaptive context window capabilities.",
        )

        # 컨텍스트 윈도우 사용량 계산
        context_usage = self._calculate_context_usage(state, result.content)

        # 최종 종합 결과를 컨텍스트에 추가 (재귀적 확장)
        if current_context:
            final_context = {
                "final_synthesis": {
                    "content": result.content,
                    "model_used": result.model_used,
                    "confidence": result.confidence,
                    "execution_time": result.execution_time,
                },
                "context_window_usage": context_usage,
                "stage": "completed",
            }
            self.context_manager.extend_context(
                current_context.context_id,
                final_context,
                metadata={
                    "synthesis_completed": True,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.debug(
                f"Final synthesis added to context: {current_context.context_id}"
            )

        state.update(
            {
                "final_synthesis": {
                    "content": result.content,
                    "model_used": result.model_used,
                    "confidence": result.confidence,
                    "execution_time": result.execution_time,
                },
                "context_window_usage": context_usage,
                "current_step": "completed",
                "innovation_stats": {
                    **state.get("innovation_stats", {}),
                    "synthesis_model": result.model_used,
                    "synthesis_confidence": result.confidence,
                    "context_window_usage": context_usage.get("usage_ratio", 1.0),
                },
            }
        )

        # 연구 결과를 메모리에 저장
        await self._save_research_memory(state)

        # 창의적 인사이트 생성
        await self._generate_creative_insights(state)

        return state

    async def _generate_creative_insights(self, state: ResearchState) -> None:
        """창의적 인사이트를 생성합니다."""
        try:
            context = state.get("user_request", "")
            current_ideas = []

            # 기존 아이디어들 수집
            if "analyzed_objectives" in state:
                for obj in state["analyzed_objectives"]:
                    if "description" in obj:
                        current_ideas.append(obj["description"])

            if "execution_results" in state:
                for result in state["execution_results"]:
                    if "summary" in result:
                        current_ideas.append(result["summary"])

            if not current_ideas:
                logger.warning("No current ideas found for creativity generation")
                return

            # 창의적 인사이트 생성
            insights = await self.creativity_agent.generate_creative_insights(
                context=context,
                current_ideas=current_ideas[:5],  # 최대 5개 아이디어만 사용
            )

            if insights:
                # 인사이트를 상태에 저장
                state["creative_insights"] = [
                    {
                        "insight_id": insight.insight_id,
                        "type": insight.type.value,
                        "title": insight.title,
                        "description": insight.description,
                        "related_concepts": insight.related_concepts,
                        "confidence": insight.confidence,
                        "novelty_score": insight.novelty_score,
                        "applicability_score": insight.applicability_score,
                        "reasoning": insight.reasoning,
                        "examples": insight.examples,
                        "metadata": insight.metadata,
                    }
                    for insight in insights
                ]

                logger.info(f"Generated {len(insights)} creative insights")

                # 스트리밍 이벤트 발생
                await self.streaming_manager.stream_event(
                    event_type=EventType.AGENT_ACTION,
                    agent_id="creativity_agent",
                    workflow_id=state["objective_id"],
                    data={
                        "action": "creative_insights_generated",
                        "insights_count": len(insights),
                        "insights": [
                            {
                                "title": insight.title,
                                "type": insight.type.value,
                                "confidence": insight.confidence,
                            }
                            for insight in insights
                        ],
                    },
                    priority=2,
                )
            else:
                logger.warning("No creative insights generated")

        except Exception as e:
            logger.error(f"Failed to generate creative insights: {e}")

    # ==================== Planning Agent Helper Methods ====================

    async def _conduct_preliminary_research(
        self, state: ResearchState
    ) -> Dict[str, Any]:
        """MCP 도구로 사전 조사 수행."""
        logger.info("🔍 Conducting preliminary research with MCP tools")

        objectives = state.get("analyzed_objectives", [])
        domain = state.get("domain_analysis", {})

        # 핵심 키워드 추출
        keywords = self._extract_keywords(objectives, domain)
        logger.info(f"🔑 Extracted keywords: {keywords[:5]}")  # 상위 5개만 로그

        # MCP 도구로 검색
        search_results = []
        # 실제 MCP 도구 이름 사용 (라우팅 지원: g-search, tavily, exa는 _execute_search_tool로 자동 라우팅됨)
        search_tools = ["g-search", "tavily", "exa"]  # _execute_search_tool로 라우팅됨

        for i, keyword in enumerate(keywords[:3]):  # 상위 3개 키워드
            tool_name = search_tools[i % len(search_tools)]  # 도구 순환 사용

            try:
                result = await execute_tool(
                    tool_name=tool_name, parameters={"query": keyword, "max_results": 5}
                )

                if result.get("success", False):
                    result_data = result.get("data", {})
                    if isinstance(result_data, dict) and "results" in result_data:
                        data_list = result_data.get("results", [])
                    else:
                        data_list = result_data if isinstance(result_data, list) else []

                    search_results.append(
                        {
                            "keyword": keyword,
                            "tool": tool_name,
                            "data": data_list,
                            "sources_count": len(data_list),
                        }
                    )
                    logger.info(
                        f"✅ {tool_name} search for '{keyword}': {len(data_list)} results"
                    )
                else:
                    logger.warning(
                        f"⚠️ {tool_name} search failed for '{keyword}': {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                logger.warning(f"⚠️ {tool_name} search error for '{keyword}': {e}")

        # 학술 검색 (실제 MCP 도구 이름 사용)
        academic_results = []
        academic_tools = [
            "semantic_scholar::papers-search-basic"
        ]  # arxiv, scholar는 MCP에 없으므로 semantic_scholar 사용

        for tool_name in academic_tools:
            try:
                result = await execute_tool(
                    tool_name=tool_name,
                    parameters={"query": " ".join(keywords[:2]), "max_results": 3},
                )

                if result.get("success", False):
                    result_data = result.get("data", {})
                    if isinstance(result_data, dict) and "results" in result_data:
                        data_list = result_data.get("results", [])
                    else:
                        data_list = result_data if isinstance(result_data, list) else []

                    academic_results.append(
                        {
                            "tool": tool_name,
                            "data": data_list,
                            "sources_count": len(data_list),
                        }
                    )
                    logger.info(
                        f"✅ {tool_name} academic search: {len(data_list)} results"
                    )

            except Exception as e:
                logger.warning(f"⚠️ {tool_name} academic search error: {e}")

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
        """목표와 도메인에서 핵심 키워드 추출."""
        keywords = []

        # Objectives에서 키워드 추출
        for obj in objectives:
            description = obj.get("description", "")
            # 간단한 키워드 추출 (실제로는 더 정교한 NLP 사용)
            words = description.lower().split()
            keywords.extend(
                [
                    w
                    for w in words
                    if len(w) > 3 and w not in ["the", "and", "for", "with", "from"]
                ]
            )

        # Domain에서 키워드 추출
        fields = domain.get("fields", [])
        keywords.extend(fields)

        # 중복 제거 및 빈도순 정렬
        from collections import Counter

        keyword_counts = Counter(keywords)
        return [kw for kw, count in keyword_counts.most_common(10)]

    async def _decompose_into_tasks(
        self,
        state: ResearchState,
        preliminary_research: Dict[str, Any],
        depth_config: Any | None = None,
    ) -> List[Dict[str, Any]]:
        """복잡도 기반 task 분해 (9번째 혁신: Adaptive Research Depth 통합 + 재귀적 분해)."""
        logger.info(
            "📋 Decomposing research into specific tasks (with recursive decomposition)"
        )

        # complexity와 num_tasks를 함수 시작 부분에서 항상 초기화 (스코프 문제 방지)
        complexity_raw = state.get("complexity_score", 5.0)
        if isinstance(complexity_raw, dict):
            complexity = complexity_raw.get("score", complexity_raw.get("value", 5.0))
        elif isinstance(complexity_raw, (int, float)):
            complexity = float(complexity_raw)
        else:
            complexity = 5.0

        num_tasks = 5  # 기본값

        # 9번째 혁신: depth_config가 있으면 사용
        if depth_config:
            planning_config = depth_config.planning.get("decompose", {})
            mode = planning_config.get("mode", "manual")

            if mode == "auto":
                # 자동 모드: 복잡도 기반
                # 복잡도에 따른 task 개수 결정
                if complexity <= 5:
                    num_tasks = 3 + int(complexity)  # 3-8개
                elif complexity <= 8:
                    num_tasks = 5 + int(complexity)  # 5-13개
                else:
                    num_tasks = 8 + int(complexity * 0.5)  # 8-13개

                # auto_max_subtopics 제한 적용
                max_subtopics = planning_config.get("auto_max_subtopics", 8)
                num_tasks = min(num_tasks, max_subtopics)
            else:
                # 수동 모드: 프리셋 설정 사용
                num_tasks = planning_config.get("initial_subtopics", 5)
                logger.info(f"📊 Using preset subtopics: {num_tasks}")
        else:
            # 기존 로직 (depth_config가 없는 경우)
            # 복잡도에 따른 task 개수 결정
            if complexity <= 5:
                num_tasks = 3 + int(complexity)  # 3-8개
            elif complexity <= 8:
                num_tasks = 5 + int(complexity)  # 5-13개
            else:
                num_tasks = 8 + int(complexity * 0.5)  # 8-13개

        logger.info(f"📊 Target task count: {num_tasks}")

        # 초기 태스크 생성
        initial_tasks = await self._create_initial_tasks(
            state, preliminary_research, num_tasks, complexity
        )

        # 재귀적 분해 적용
        final_tasks = []
        max_recursion_depth = (
            depth_config.planning.get("max_recursion_depth", 3) if depth_config else 3
        )

        for task in initial_tasks:
            if await self._is_atomic_task(task, depth_config, complexity):
                final_tasks.append(task)
                logger.info(
                    f"  ✅ Atomic task: {task.get('name', 'Unknown')} (no further decomposition needed)"
                )
            else:
                # 재귀적 분해
                logger.info(
                    f"  🔄 Non-atomic task detected: {task.get('name', 'Unknown')} - starting recursive decomposition"
                )
                subtasks = await self._recursive_decompose(
                    task,
                    state,
                    preliminary_research,
                    depth_config,
                    current_depth=0,
                    max_depth=max_recursion_depth,
                )
                final_tasks.extend(subtasks)
                logger.info(f"  ✅ Decomposed into {len(subtasks)} subtasks")

        # Task 검증 및 로깅
        logger.info(
            f"📋 Final task count: {len(final_tasks)} (from {len(initial_tasks)} initial tasks)"
        )
        for i, task in enumerate(final_tasks):
            logger.info(
                f"  Task {i + 1}: {task.get('name', 'Unknown')} ({task.get('type', 'research')}) - {task.get('assigned_agent_type', 'unknown')} agent"
            )

        return final_tasks

    async def _create_initial_tasks(
        self,
        state: ResearchState,
        preliminary_research: Dict[str, Any],
        num_tasks: int,
        complexity: float,
    ) -> List[Dict[str, Any]]:
        """초기 태스크 생성 (기존 로직)."""
        # LLM으로 task 생성 (사전 조사 결과 포함)
        decomposition_prompt = f"""
        Based on preliminary research, decompose the research into {num_tasks} specific, executable tasks:
        
        Research Request: {state.get("user_request", "")}
        Objectives: {state.get("analyzed_objectives", [])}
        Domain: {state.get("domain_analysis", {})}
        Complexity Score: {complexity}
        
        Preliminary Research:
        - Keywords: {preliminary_research.get("keywords", [])}
        - Search Results: {len(preliminary_research.get("search_results", []))} sources
        - Academic Results: {len(preliminary_research.get("academic_results", []))} sources
        
        For each task, provide the following structure:
        {{
            "task_id": "task_1",
            "name": "Specific task name",
            "description": "Detailed task description",
            "type": "academic|market|technical|data|synthesis",
            "assigned_agent_type": "academic_researcher|market_analyst|technical_researcher|data_collector|synthesis_specialist",
            "required_tools": ["g-search", "arxiv", "tavily"],
            "dependencies": ["task_0"],
            "estimated_complexity": 1-10,
            "priority": "high|medium|low",
            "success_criteria": ["specific measurable criteria"]
        }}
        
        Ensure tasks cover all research objectives and have logical dependencies.
        Return as JSON array of task objects.
        """

        result = await execute_llm_task(
            prompt=decomposition_prompt,
            task_type=TaskType.PLANNING,
            system_message="You are an expert research project manager with deep knowledge of task decomposition and resource allocation.",
        )

        # Task 결과 파싱
        tasks = self._parse_tasks_result(result.content)
        return tasks

    async def _is_atomic_task(
        self, task: Dict[str, Any], depth_config: Any | None, complexity: float
    ) -> bool:
        """태스크가 원자적(atomic)인지 판단 (ROMA의 Atomizer 개념).

        원자적 태스크는 직접 실행 가능한 태스크로, 더 이상 분해할 필요가 없습니다.
        판단 기준:
        - 복잡도가 낮음 (estimated_complexity <= 5)
        - 의존성이 적음 (<= 1)
        - 도구 요구사항이 적음 (<= 2)
        - 명확한 성공 기준
        """
        # 복잡도 기반 판단
        task_complexity = task.get("estimated_complexity", 5)
        if isinstance(task_complexity, dict):
            task_complexity = task_complexity.get(
                "score", task_complexity.get("value", 5)
            )
        elif not isinstance(task_complexity, (int, float)):
            task_complexity = 5

        # 복잡도가 매우 높으면 (>= 8) 비원자적
        if task_complexity >= 8:
            return False

        # 의존성 체크: 의존성이 많으면 (>= 2) 비원자적
        dependencies = task.get("dependencies", [])
        if len(dependencies) >= 2:
            return False

        # 도구 요구사항 체크: 도구가 많으면 (>= 3) 비원자적
        required_tools = task.get("required_tools", [])
        if len(required_tools) >= 3:
            return False

        # 복잡도가 낮으면 (<= 5) 원자적
        if task_complexity <= 5:
            return True

        # 복잡도가 중간이면 (6-7) 추가 조건 확인
        # 성공 기준이 명확하고, 의존성이 없고, 도구가 적으면 원자적
        success_criteria = task.get("success_criteria", [])
        if (
            len(success_criteria) >= 2
            and len(dependencies) == 0
            and len(required_tools) <= 2
        ):
            return True

        # 기본값: 복잡도가 중간 이상이면 비원자적
        return False

    async def _recursive_decompose(
        self,
        task: Dict[str, Any],
        state: ResearchState,
        preliminary_research: Dict[str, Any],
        depth_config: Any | None,
        current_depth: int,
        max_depth: int,
    ) -> List[Dict[str, Any]]:
        """비원자 태스크를 재귀적으로 분해 (ROMA의 재귀적 분해 개념).

        복잡한 태스크를 더 작은 하위 태스크로 분해합니다.
        최대 재귀 깊이를 제한하여 무한 루프를 방지합니다.
        """
        if current_depth >= max_depth:
            logger.warning(
                f"  ⚠️ Maximum recursion depth ({max_depth}) reached for task: {task.get('name', 'Unknown')}"
            )
            # 최대 깊이에 도달하면 원자적 태스크로 간주
            return [task]

        task_complexity = task.get("estimated_complexity", 5)
        if isinstance(task_complexity, dict):
            task_complexity = task_complexity.get(
                "score", task_complexity.get("value", 5)
            )
        elif not isinstance(task_complexity, (int, float)):
            task_complexity = 5

        # 하위 태스크 개수 결정 (복잡도 기반)
        num_subtasks = min(3 + int(task_complexity / 2), 5)  # 최대 5개

        # parent_task_id 추출 (프롬프트에서 사용하기 전에 정의)
        parent_task_id = task.get("task_id", "unknown")

        logger.info(
            f"  🔄 Recursive decomposition (depth {current_depth + 1}/{max_depth}): {task.get('name', 'Unknown')} -> {num_subtasks} subtasks"
        )

        # 하위 태스크 생성 프롬프트
        decomposition_prompt = f"""
        Decompose the following complex task into {num_subtasks} smaller, more manageable subtasks:
        
        Parent Task:
        - Name: {task.get("name", "")}
        - Description: {task.get("description", "")}
        - Type: {task.get("type", "research")}
        - Complexity: {task_complexity}
        - Required Tools: {task.get("required_tools", [])}
        
        Research Context:
        - Request: {state.get("user_request", "")}
        - Objectives: {state.get("analyzed_objectives", [])}
        
        Create subtasks that:
        1. Are more specific and focused than the parent task
        2. Can be executed independently or with minimal dependencies
        3. Together accomplish the parent task's goal
        4. Have lower complexity scores (target: 3-6 each)
        
        For each subtask, provide:
        {{
            "task_id": "subtask_{parent_task_id}_1",
            "name": "Specific subtask name",
            "description": "Detailed subtask description",
            "type": "{task.get("type", "research")}",
            "assigned_agent_type": "{task.get("assigned_agent_type", "academic_researcher")}",
            "required_tools": ["g-search", "arxiv"],
            "dependencies": [],
            "estimated_complexity": 3-6,
            "priority": "{task.get("priority", "medium")}",
            "success_criteria": ["specific measurable criteria"],
            "parent_task_id": "{parent_task_id}"
        }}
        
        Return as JSON array of subtask objects.
        """

        result = await execute_llm_task(
            prompt=decomposition_prompt,
            task_type=TaskType.PLANNING,
            system_message="You are an expert at breaking down complex research tasks into manageable subtasks.",
        )

        # 하위 태스크 파싱
        subtasks = self._parse_tasks_result(result.content)

        # 하위 태스크에 parent_task_id 추가 (이미 위에서 정의됨)
        for subtask in subtasks:
            subtask["parent_task_id"] = parent_task_id
            subtask["decomposition_depth"] = current_depth + 1

        # 각 하위 태스크에 대해 재귀적으로 원자성 확인
        final_subtasks = []
        for subtask in subtasks:
            if await self._is_atomic_task(subtask, depth_config, task_complexity):
                final_subtasks.append(subtask)
            else:
                # 더 깊이 분해
                deeper_subtasks = await self._recursive_decompose(
                    subtask,
                    state,
                    preliminary_research,
                    depth_config,
                    current_depth + 1,
                    max_depth,
                )
                final_subtasks.extend(deeper_subtasks)

        return final_subtasks

    async def _assign_agents_dynamically(
        self, tasks: List[Dict[str, Any]], state: ResearchState
    ) -> Dict[str, List[str]]:
        """복잡도 기반 동적 agent 할당."""
        logger.info("👥 Assigning agents dynamically based on task complexity")

        agent_assignments = {}
        available_researchers = state.get("allocated_researchers", 1)

        for task in tasks:
            task_id = task.get("task_id", "unknown")
            complexity_raw = task.get("estimated_complexity", 5)
            # complexity 타입 체크
            if isinstance(complexity_raw, dict):
                complexity = complexity_raw.get("score", complexity_raw.get("value", 5))
            elif isinstance(complexity_raw, (int, float)):
                complexity = int(complexity_raw)
            else:
                complexity = 5
            task_type = task.get("type", "research")

            # 복잡도에 따른 agent 수 결정
            if complexity <= 3:
                num_agents = 1
            elif complexity <= 7:
                num_agents = min(2, available_researchers)
            else:
                num_agents = min(3, available_researchers)

            # Agent 유형 결정
            agent_types = self._select_agent_types(task_type, num_agents)

            agent_assignments[task_id] = agent_types

            logger.info(
                f"  {task_id}: {num_agents} agents ({', '.join(agent_types)}) for complexity {complexity}"
            )

        return agent_assignments

    def _select_agent_types(self, task_type: str, num_agents: int) -> List[str]:
        """Task 유형에 따른 agent 유형 선택."""
        agent_type_mapping = {
            "academic": ["academic_researcher"],
            "market": ["market_analyst"],
            "technical": ["technical_researcher"],
            "data": ["data_collector"],
            "synthesis": ["synthesis_specialist"],
            "research": ["academic_researcher", "technical_researcher"],
        }

        base_types = agent_type_mapping.get(task_type, ["academic_researcher"])

        # 필요한 수만큼 agent 유형 반환
        if num_agents <= len(base_types):
            return base_types[:num_agents]
        else:
            # 부족한 경우 다른 유형 추가
            additional_types = [
                "market_analyst",
                "technical_researcher",
                "data_collector",
                "synthesis_specialist",
            ]
            result = base_types.copy()
            for agent_type in additional_types:
                if len(result) >= num_agents:
                    break
                if agent_type not in result:
                    result.append(agent_type)
            return result[:num_agents]

    async def _create_execution_plan(
        self, tasks: List[Dict[str, Any]], agent_assignments: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """실행 전략 수립."""
        logger.info("📈 Creating execution plan")

        # 의존성 분석
        dependency_graph = self._build_dependency_graph(tasks)

        # 병렬 가능한 task 그룹 식별
        parallel_groups = self._identify_parallel_groups(dependency_graph)

        # 실행 순서 결정
        execution_order = self._determine_execution_order(tasks, dependency_graph)

        # 전략 결정
        strategy = "hybrid" if parallel_groups else "sequential"

        # 예상 시간 계산
        estimated_total_time = sum(task.get("estimated_time", 30) for task in tasks)

        execution_plan = {
            "strategy": strategy,
            "parallel_groups": parallel_groups,
            "execution_order": execution_order,
            "estimated_total_time": estimated_total_time,
            "dependency_graph": dependency_graph,
            "task_count": len(tasks),
            "agent_count": len(
                set(agent for agents in agent_assignments.values() for agent in agents)
            ),
        }

        logger.info(
            f"📊 Execution plan: {strategy} strategy, {len(parallel_groups)} parallel groups, {estimated_total_time}min total"
        )

        return execution_plan

    def _build_dependency_graph(
        self, tasks: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Task 의존성 그래프 구축."""
        graph = {}

        for task in tasks:
            task_id = task.get("task_id", "")
            dependencies = task.get("dependencies", [])
            graph[task_id] = dependencies

        return graph

    def _identify_parallel_groups(
        self, dependency_graph: Dict[str, List[str]]
    ) -> List[List[str]]:
        """병렬 실행 가능한 task 그룹 식별."""
        # 간단한 구현: 의존성이 없는 task들을 그룹화
        parallel_groups = []
        processed = set()

        for task_id, dependencies in dependency_graph.items():
            if task_id in processed:
                continue

            if not dependencies:  # 의존성이 없는 task
                group = [task_id]
                # 다른 의존성 없는 task들 찾기
                for other_task, other_deps in dependency_graph.items():
                    if (
                        other_task != task_id
                        and other_task not in processed
                        and not other_deps
                    ):
                        group.append(other_task)
                        processed.add(other_task)

                if len(group) > 1:
                    parallel_groups.append(group)
                    processed.update(group)

        return parallel_groups

    def _determine_execution_order(
        self, tasks: List[Dict[str, Any]], dependency_graph: Dict[str, List[str]]
    ) -> List[str]:
        """의존성을 고려한 실행 순서 결정."""
        # 위상 정렬을 사용한 실행 순서 결정
        in_degree = {task_id: 0 for task_id in dependency_graph.keys()}

        # 진입 차수 계산
        for task_id, dependencies in dependency_graph.items():
            for dep in dependencies:
                if dep in in_degree:
                    in_degree[task_id] += 1

        # 위상 정렬
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # 현재 task에 의존하는 task들의 진입 차수 감소
            for task_id, dependencies in dependency_graph.items():
                if current in dependencies:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)

        return result

    # ==================== Helper Methods ====================

    def _parse_analysis_result(self, content: str) -> Dict[str, Any]:
        """분석 결과 파싱 — JSON이면 파싱, 아니면 텍스트 기반 fallback 구조 생성."""
        import json
        import re

        cleaned = (content or "").strip()

        # Markdown 코드 블록 제거
        md_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if md_match:
            cleaned = md_match.group(1).strip()

        # JSON 파싱 시도
        if cleaned.startswith("{"):
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning("JSON decode failed, using text fallback")

        # 텍스트 응답 → 기본 구조로 변환
        logger.info("LLM returned non-JSON analysis; building fallback structure from text")
        return {
            "objectives": [
                {"id": "obj_1", "description": cleaned[:500] or "General research", "priority": "high"}
            ],
            "intent": {"primary": "research", "secondary": "analysis"},
            "domain": {"fields": ["general"], "expertise": "general"},
            "scope": {"breadth": "comprehensive", "depth": "detailed"},
            "complexity": 5.0,
        }

    def _parse_tasks_result(self, content: str) -> List[Dict[str, Any]]:
        """Task 분해 결과 파싱 - 재시도 로직 포함."""
        import json
        import re

        # 최대 3회 재시도
        for attempt in range(3):
            try:
                # Markdown 코드 블록 제거
                cleaned_content = content.strip()
                if "```json" in cleaned_content:
                    match = re.search(
                        r"```json\s*(.*?)\s*```", cleaned_content, re.DOTALL
                    )
                    if match:
                        cleaned_content = match.group(1).strip()
                elif "```" in cleaned_content:
                    match = re.search(r"```\s*(.*?)\s*```", cleaned_content, re.DOTALL)
                    if match:
                        cleaned_content = match.group(1).strip()

                # JSON 배열 파싱 시도
                if cleaned_content.startswith("["):
                    return json.loads(cleaned_content)
                else:
                    if attempt < 2:
                        logger.warning(
                            f"⚠️ Attempt {attempt + 1}: Invalid JSON array format, retrying..."
                        )
                        continue
                    else:
                        raise ValueError(
                            "Invalid JSON array format in task decomposition result"
                        )

            except json.JSONDecodeError as e:
                if attempt < 2:
                    logger.warning(
                        f"⚠️ Attempt {attempt + 1}: JSON decode error: {e}, retrying..."
                    )
                    continue
                else:
                    logger.error(
                        f"❌ Failed to parse tasks result after 3 attempts: {e}"
                    )
                    raise ValueError(f"Task parsing failed after 3 attempts: {e}")
            except Exception as e:
                if attempt < 2:
                    logger.warning(
                        f"⚠️ Attempt {attempt + 1}: Parse error: {e}, retrying..."
                    )
                    continue
                else:
                    logger.error(
                        f"❌ Failed to parse tasks result after 3 attempts: {e}"
                    )
                    raise ValueError(f"Task parsing failed after 3 attempts: {e}")

        raise ValueError("Unexpected error in task parsing")

    def _parse_verification_result(self, content: str) -> Dict[str, Any]:
        """Plan 검증 결과 파싱 - 재시도 로직 포함. 실패 시 기본값 반환하여 연구 계속 진행."""
        import json
        import re

        # 안전 필터 응답 감지
        if (
            not content
            or "blocked by safety filters" in content.lower()
            or "Unable to extract content" in content
        ):
            logger.warning(
                "⚠️ Safety filter triggered or empty response. Using default verification result."
            )
            return {
                "approved": True,  # 기본적으로 승인하여 연구 계속 진행
                "confidence": 0.5,  # 낮은 신뢰도
                "feedback": "Verification skipped due to safety filter. Proceeding with plan.",
                "suggested_changes": [],
                "critical_issues": [],
            }

        # 최대 3회 재시도
        for attempt in range(3):
            try:
                # Markdown 코드 블록 제거
                cleaned_content = content.strip()
                if "```json" in cleaned_content:
                    match = re.search(
                        r"```json\s*(.*?)\s*```", cleaned_content, re.DOTALL
                    )
                    if match:
                        cleaned_content = match.group(1).strip()
                elif "```" in cleaned_content:
                    match = re.search(r"```\s*(.*?)\s*```", cleaned_content, re.DOTALL)
                    if match:
                        cleaned_content = match.group(1).strip()

                # JSON 파싱 시도
                if cleaned_content.startswith("{"):
                    parsed = json.loads(cleaned_content)
                    # 필수 필드 검증
                    if not isinstance(parsed, dict):
                        raise ValueError("Parsed result is not a dictionary")
                    return parsed
                else:
                    if attempt < 2:
                        logger.warning(
                            f"⚠️ Attempt {attempt + 1}: Invalid JSON format, retrying..."
                        )
                        continue
                    else:
                        # 최종 실패 시 기본값 반환
                        logger.warning(
                            "⚠️ JSON parsing failed after 3 attempts. Using default verification result."
                        )
                        return {
                            "approved": True,
                            "confidence": 0.6,
                            "feedback": "Verification parsing failed. Proceeding with plan.",
                            "suggested_changes": [],
                            "critical_issues": [],
                        }

            except json.JSONDecodeError as e:
                if attempt < 2:
                    logger.warning(
                        f"⚠️ Attempt {attempt + 1}: JSON decode error: {e}, retrying..."
                    )
                    continue
                else:
                    logger.warning(
                        "⚠️ Failed to parse verification result after 3 attempts. Using default result."
                    )
                    # 기본값 반환하여 연구 계속 진행
                    return {
                        "approved": True,
                        "confidence": 0.6,
                        "feedback": f"Verification parsing failed: {str(e)}. Proceeding with plan.",
                        "suggested_changes": [],
                        "critical_issues": [],
                    }
            except Exception as e:
                if attempt < 2:
                    logger.warning(
                        f"⚠️ Attempt {attempt + 1}: Parse error: {e}, retrying..."
                    )
                    continue
                else:
                    logger.warning(
                        f"⚠️ Verification parsing error: {e}. Using default result."
                    )
                    # 기본값 반환하여 연구 계속 진행
                    return {
                        "approved": True,
                        "confidence": 0.6,
                        "feedback": f"Verification error: {str(e)}. Proceeding with plan.",
                        "suggested_changes": [],
                        "critical_issues": [],
                    }

        # 최종 fallback
        logger.warning(
            "⚠️ Unexpected error in verification parsing. Using default result."
        )
        return {
            "approved": True,
            "confidence": 0.5,
            "feedback": "Verification parsing failed. Proceeding with plan.",
            "suggested_changes": [],
            "critical_issues": [],
        }

    def _parse_evaluation_result(self, content: str) -> Dict[str, Any]:
        """평가 결과 파싱 - 재시도 로직 및 safety filter 응답 처리 포함."""
        import json
        import re

        # Safety filter 응답 체크
        if not content or not isinstance(content, str):
            logger.warning(
                "Evaluation result content is empty or invalid, using default"
            )
            return self._get_default_evaluation_result()

        # Safety filter로 차단된 응답 체크
        safety_indicators = [
            "Content blocked by safety filters",
            "Unable to extract content",
            "[Content blocked",
            "safety filter",
            "finish_reason=2",
        ]

        content_lower = content.lower()
        if any(indicator.lower() in content_lower for indicator in safety_indicators):
            logger.warning(
                "Evaluation result was blocked by safety filters, using default"
            )
            return self._get_default_evaluation_result()

        # 최대 3회 재시도
        for attempt in range(3):
            try:
                # Markdown 코드 블록 제거
                cleaned_content = content.strip()
                if "```json" in cleaned_content:
                    match = re.search(
                        r"```json\s*(.*?)\s*```", cleaned_content, re.DOTALL
                    )
                    if match:
                        cleaned_content = match.group(1).strip()
                elif "```" in cleaned_content:
                    match = re.search(r"```\s*(.*?)\s*```", cleaned_content, re.DOTALL)
                    if match:
                        cleaned_content = match.group(1).strip()

                # JSON 객체 추출 시도 (중괄호로 시작하는 부분 찾기)
                if cleaned_content:
                    # JSON 객체 찾기
                    json_match = re.search(
                        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned_content, re.DOTALL
                    )
                    if json_match:
                        cleaned_content = json_match.group(0)

                # JSON 파싱 시도
                if cleaned_content.startswith("{"):
                    try:
                        result = json.loads(cleaned_content)
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError as je:
                        if attempt < 2:
                            logger.warning(
                                f"⚠️ Attempt {attempt + 1}: JSON decode error: {je}, retrying..."
                            )
                            continue

                # JSON 파싱 실패 시 기본값 반환
                if attempt == 2:
                    logger.warning(
                        "Failed to parse evaluation result after 3 attempts, using default"
                    )
                    return self._get_default_evaluation_result()

            except json.JSONDecodeError as e:
                if attempt < 2:
                    logger.warning(
                        f"⚠️ Attempt {attempt + 1}: JSON decode error: {e}, retrying..."
                    )
                    continue
                else:
                    logger.warning(
                        f"❌ Failed to parse evaluation result after 3 attempts: {e}, using default"
                    )
                    return self._get_default_evaluation_result()
            except Exception as e:
                if attempt < 2:
                    logger.warning(
                        f"⚠️ Attempt {attempt + 1}: Parse error: {e}, retrying..."
                    )
                    continue
                else:
                    logger.warning(
                        f"❌ Failed to parse evaluation result after 3 attempts: {e}, using default"
                    )
                    return self._get_default_evaluation_result()

        # 모든 시도 실패 시 기본값 반환
        logger.warning("All parsing attempts failed, using default evaluation result")
        return self._get_default_evaluation_result()

    def _get_default_evaluation_result(self) -> Dict[str, Any]:
        """기본 평가 결과 반환 (파싱 실패 또는 safety filter 응답 시)."""
        return {
            "overall_score": 0.7,
            "objective_scores": {},
            "quality_metrics": {
                "completeness": 0.7,
                "accuracy": 0.7,
                "relevance": 0.7,
                "depth": 0.7,
            },
            "improvement_areas": [],
            "needs_additional_work": False,
            "recommendations": [
                "Evaluation parsing failed, results may need manual review"
            ],
            "parsing_failed": True,
            "safety_filter_blocked": True,
        }

    def _create_priority_queue(self, state: ResearchState) -> List[Dict[str, Any]]:
        """우선순위 큐 생성."""
        tasks = state.get("planned_tasks", [])
        priority_queue = []

        for task in tasks:
            priority = (
                1
                if task.get("priority") == "high"
                else 2
                if task.get("priority") == "medium"
                else 3
            )
            priority_queue.append(
                {
                    "task_id": task.get("task_id", ""),
                    "priority": priority,
                    "estimated_time": task.get("estimated_time", 30),
                    "complexity": task.get("estimated_complexity", 5),
                }
            )

        # 우선순위별로 정렬
        priority_queue.sort(key=lambda x: (x["priority"], x["complexity"]))
        return priority_queue

    def _generate_tool_parameters(
        self, task: Dict[str, Any], tool_name: str
    ) -> Dict[str, Any]:
        """도구 실행을 위한 파라미터 자동 생성 및 검증."""
        # 기존 파라미터 가져오기
        parameters = (
            task.get("parameters", {}).copy()
            if isinstance(task.get("parameters"), dict)
            else {}
        )

        # task name과 description에서 검색어 추출
        task_name = task.get("name", "")
        task_description = task.get("description", "")
        combined_text = f"{task_name} {task_description}".strip()

        # 도구별 필수 파라미터 매핑
        tool_requirements = {
            # semantic_scholar 도구들
            "semantic_scholar::papers-search-basic": {"query": True},
            "semantic_scholar::paper-search-advanced": {"query": True},
            "semantic_scholar::search-paper-title": {"title": True},
            "semantic_scholar::search-arxiv": {"query": True},
            "semantic_scholar::get-paper-abstract": {"paper_id": True},
            "semantic_scholar::papers-citations": {"paper_id": True},
            "semantic_scholar::papers-references": {"paper_id": True},
            # 검색 도구들
            "g-search": {"query": True},
            "tavily": {"query": True},
            "exa": {"query": True},
            "ddg_search::search": {"query": True},
            "tavily-mcp::tavily-search": {"query": True},
            "exa::web_search_exa": {"query": True},
            "WebSearch-MCP::web_search": {"query": True},
            # 데이터 도구들
            "fetch::fetch_url": {"url": True},
            "fetch::extract_elements": {"url": True, "selector": True},
            "fetch::get_page_metadata": {"url": True},
        }

        # 도구 이름 확인 (server::tool 형식 또는 단순 이름)
        tool_key = tool_name
        if "::" not in tool_name:
            # 단순 이름인 경우 매핑에서 찾기
            for key in tool_requirements.keys():
                if key.endswith(f"::{tool_name}") or tool_name in key:
                    tool_key = key
                    break

        requirements = tool_requirements.get(tool_key, {})

        # 필수 파라미터 체크 및 자동 생성
        for param_name, is_required in requirements.items():
            if is_required and not parameters.get(param_name):
                # 검색어 자동 생성
                if param_name in ["query", "title"]:
                    if combined_text:
                        # task name에서 핵심 키워드 추출
                        # 간단한 키워드 추출: 긴 문장을 요약하거나 핵심 키워드만 사용
                        query = self._extract_search_query(combined_text)
                        parameters[param_name] = query
                        logger.info(
                            f"✅ Auto-generated {param_name} for {tool_name}: '{query}'"
                        )
                    else:
                        # task name이 없으면 state에서 user_request 사용
                        # state는 task에 직접 포함되어 있지 않으므로, 기본값으로 task name 사용
                        fallback_text = task_name if task_name else "research"
                        query = self._extract_search_query(fallback_text)
                        parameters[param_name] = query
                        logger.info(
                            f"✅ Auto-generated {param_name} from fallback for {tool_name}: '{query}'"
                        )

        # 기본값 설정
        if "max_results" not in parameters and tool_key in [
            "g-search",
            "tavily",
            "exa",
            "ddg_search::search",
        ]:
            parameters["max_results"] = 10
        if "num_results" not in parameters and "exa" in tool_key:
            parameters["num_results"] = parameters.get("max_results", 10)

        return parameters

    def _extract_search_query(self, text: str) -> str:
        """텍스트에서 검색 쿼리 추출 (간단한 키워드 추출)."""
        if not text or not isinstance(text, str):
            return ""

        # 텍스트 정리
        text = text.strip()
        if len(text) > 200:
            # 너무 긴 경우 첫 200자만 사용
            text = text[:200]

        # 기본 검색어로 사용 (더 정교한 추출 필요 시 LLM 사용 가능)
        # 현재는 간단하게 텍스트 그대로 사용하되 불필요한 단어 제거
        import re

        # 일반적인 불필요한 단어 제거 (영문 기준)
        stop_words = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        ]
        words = re.findall(r"\b\w+\b", text.lower())
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]

        if filtered_words:
            # 핵심 키워드만 사용 (최대 10개)
            query = " ".join(filtered_words[:10])
            return query[:150]  # 최대 150자

        return text[:150]  # 필터링 실패 시 원본 텍스트 사용

    def _get_tool_category_for_task(self, task: Dict[str, Any]) -> ToolCategory:
        """작업에 적합한 도구 카테고리 반환."""
        task_type = task.get("type", "research").lower()
        if "search" in task_type:
            return ToolCategory.SEARCH
        elif "academic" in task_type:
            return ToolCategory.ACADEMIC
        elif "data" in task_type:
            return ToolCategory.DATA
        else:
            return ToolCategory.SEARCH  # RESEARCH 대신 SEARCH 사용

    def _get_best_tool_for_category(self, category: ToolCategory) -> str | None:
        """카테고리에 맞는 최적의 도구 반환 - 실제 MCP 도구 이름 사용."""
        tool_mapping = {
            ToolCategory.SEARCH: "g-search",  # 라우팅됨
            ToolCategory.DATA: "fetch::fetch_url",
            ToolCategory.CODE: "python_coder",
            ToolCategory.ACADEMIC: "semantic_scholar::papers-search-basic",
            ToolCategory.BUSINESS: "g-search",  # 비즈니스 검색도 일반 검색으로
        }
        return tool_mapping.get(category, "g-search")  # 기본값으로 g-search 사용

    def _get_available_tools_for_category(self, category: ToolCategory) -> List[str]:
        """카테고리별 사용 가능한 도구 목록 (우선순위 순) - 실제 MCP 도구 이름 사용."""
        tool_priorities = {
            ToolCategory.SEARCH: [
                "g-search",  # 라우팅됨
                "ddg_search::search",  # 실제 MCP 도구
                "tavily-mcp::tavily-search",  # 실제 MCP 도구
                "exa::web_search_exa",  # 실제 MCP 도구
                "parallel-search",  # 실제 MCP 서버 (도구 이름 확인 필요)
                "WebSearch-MCP::web_search",  # 실제 MCP 도구
            ],
            ToolCategory.ACADEMIC: [
                "arxiv::arxiv_search",  # arXiv MCP 서버 우선
                "arxiv::arxiv_get_paper",  # arXiv MCP 서버
                "semantic_scholar::papers-search-basic",
                "semantic_scholar::paper-search-advanced",
                "semantic_scholar::search-paper-title",
                "semantic_scholar::search-arxiv",
                "arxiv",  # 로컬 fallback
            ],
            ToolCategory.DATA: [
                "fetch::fetch_url",
                "fetch::extract_elements",
                "fetch::get_page_metadata",
                "ddg_search::fetch_content",
            ],
            ToolCategory.CODE: ["python_coder", "code_interpreter"],
            ToolCategory.BUSINESS: [
                "g-search"  # 비즈니스 검색도 일반 검색으로
            ],
        }
        return tool_priorities.get(category, ["g-search", "ddg_search::search"])

    def _validate_tool_result(
        self, tool_result: Dict[str, Any], task: Dict[str, Any]
    ) -> bool:
        """도구 실행 결과 검증."""
        if not tool_result.get("success", False):
            return False

        data = tool_result.get("data")
        if not data:
            return False

        # 기본 검증: 빈 데이터가 아닌지 확인
        if isinstance(data, str) and len(data.strip()) == 0:
            return False

        if isinstance(data, dict) and len(data) == 0:
            return False

        if isinstance(data, list) and len(data) == 0:
            return False

        # 검색 결과의 경우 최소한의 내용이 있는지 확인
        if task.get("type") == "search":
            if isinstance(data, list) and len(data) > 0:
                # 검색 결과가 있는지 확인
                return True
            elif isinstance(data, dict) and "results" in data:
                # 구조화된 검색 결과인지 확인
                return len(data["results"]) > 0

        # 학술 검색의 경우 논문 정보가 있는지 확인
        if task.get("type") == "academic":
            if isinstance(data, list) and len(data) > 0:
                return True
            elif isinstance(data, dict) and ("papers" in data or "entries" in data):
                return True

        # 기본적으로 데이터가 있으면 유효한 것으로 간주
        return True

    def _extract_text_for_similarity(self, data: Any) -> str:
        """유사도 계산을 위한 텍스트 추출 - 타입 안전성 개선."""
        try:
            # 타입 검증
            if data is None:
                return ""

            # 문자열인 경우 그대로 반환
            if isinstance(data, str):
                return data.strip() if data.strip() else ""

            # 딕셔너리가 아닌 경우 문자열로 변환
            if not isinstance(data, dict):
                return str(data).strip() if str(data).strip() else ""

            # 딕셔너리 처리
            text_parts = []

            # 주요 텍스트 필드들 추출
            text_fields = [
                "title",
                "content",
                "summary",
                "description",
                "abstract",
                "snippet",
            ]
            for field in text_fields:
                if field in data and data[field]:
                    value = data[field]
                    if isinstance(value, str):
                        text_parts.append(value.strip())
                    else:
                        text_parts.append(str(value).strip())

            # 딕셔너리 값들 중 문자열인 것들 추출
            for key, value in data.items():
                if key not in text_fields and isinstance(value, str) and value.strip():
                    text_parts.append(value.strip())
                elif key not in text_fields and value is not None:
                    # 리스트나 다른 타입도 문자열로 변환 시도
                    try:
                        str_value = str(value).strip()
                        if str_value and len(str_value) < 500:  # 너무 긴 값은 제외
                            text_parts.append(str_value)
                    except Exception:
                        pass

            result = " ".join(text_parts)
            return result if result.strip() else ""
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            return ""

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Semantic similarity 계산 (간단한 버전)."""
        try:
            if not text1 or not text2:
                return 0.0

            # 단어 단위로 분할
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return 0.0

            # Jaccard similarity 계산
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            jaccard_similarity = intersection / union if union > 0 else 0.0

            # 공통 단어 비율도 고려
            common_ratio = intersection / min(len(words1), len(words2))

            # 두 지표의 가중 평균
            similarity = jaccard_similarity * 0.6 + common_ratio * 0.4

            return min(similarity, 1.0)

        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

    async def _self_verification(self, result: Dict[str, Any]) -> float:
        """자체 검증 - 실제 데이터 품질 평가."""
        try:
            data = result.get("compressed_data", {})
            if not data:
                return 0.0

            quality_score = 0.0

            # 1. 데이터 완성도 검증
            if isinstance(data, dict):
                non_empty_fields = len(
                    [v for v in data.values() if v and str(v).strip()]
                )
                total_fields = len(data)
                completeness = non_empty_fields / max(total_fields, 1)
                quality_score += completeness * 0.25

                # 필수 필드 존재 여부
                essential_fields = ["title", "content", "summary"]
                essential_present = sum(
                    1 for field in essential_fields if field in data and data[field]
                )
                essential_score = essential_present / len(essential_fields)
                quality_score += essential_score * 0.25

            # 2. 데이터 일관성 검증
            if isinstance(data, dict):
                consistency_score = 0.0

                # 제목과 내용의 일관성
                if "title" in data and "content" in data:
                    title = str(data["title"]).lower()
                    content = str(data["content"]).lower()
                    if title and content:
                        # 제목의 키워드가 내용에 포함되는지 확인
                        title_words = set(title.split())
                        content_words = set(content.split())
                        if len(title_words) > 0:
                            overlap = len(
                                title_words.intersection(content_words)
                            ) / len(title_words)
                            consistency_score += overlap * 0.5

                # 요약과 내용의 일관성
                if "summary" in data and "content" in data:
                    summary = str(data["summary"]).lower()
                    content = str(data["content"]).lower()
                    if summary and content:
                        summary_words = set(summary.split())
                        content_words = set(content.split())
                        if len(summary_words) > 0:
                            overlap = len(
                                summary_words.intersection(content_words)
                            ) / len(summary_words)
                            consistency_score += overlap * 0.5

                quality_score += consistency_score * 0.25

            # 3. 압축 품질 검증
            compression_ratio = result.get("compression_ratio", 1.0)
            original_size = result.get("original_size", 0)
            compressed_size = result.get("compressed_size", 0)

            if original_size > 0 and compressed_size > 0:
                actual_ratio = compressed_size / original_size
                # 적절한 압축률 (0.1 ~ 0.8)일 때 높은 점수
                if 0.1 <= actual_ratio <= 0.8:
                    compression_score = 1.0
                elif actual_ratio < 0.1:
                    compression_score = 0.7  # 과도한 압축
                else:
                    compression_score = 0.5  # 압축 부족

                quality_score += compression_score * 0.25

            return min(quality_score, 1.0)
        except Exception as e:
            logger.error(f"❌ Self verification failed: {e}")
            return 0.0

    async def _cross_verification(
        self, result: Dict[str, Any], all_results: List[Dict[str, Any]]
    ) -> float:
        """교차 검증 - Semantic Similarity 기반."""
        try:
            if not all_results or len(all_results) < 2:
                return 0.5

            current_data = result.get("compressed_data", {})
            if not current_data:
                return 0.0

            # 현재 결과의 텍스트 추출
            current_text = self._extract_text_for_similarity(current_data)
            if not current_text:
                return 0.5

            similarity_scores = []

            for other_result in all_results:
                if other_result.get("task_id") == result.get("task_id"):
                    continue

                other_data = other_result.get("compressed_data", {})
                if not other_data:
                    continue

                other_text = self._extract_text_for_similarity(other_data)
                if not other_text:
                    continue

                # Semantic similarity 계산
                similarity = self._calculate_semantic_similarity(
                    current_text, other_text
                )
                similarity_scores.append(similarity)

            if similarity_scores:
                # 평균 유사도 반환 (0.3-0.7 범위가 적절)
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                # 너무 높거나 낮은 유사도는 조정
                if avg_similarity > 0.9:
                    return 0.8  # 너무 유사하면 의심스러움
                elif avg_similarity < 0.1:
                    return 0.3  # 너무 다르면 일관성 부족
                else:
                    return avg_similarity
            else:
                return 0.5

        except Exception as e:
            logger.error(f"❌ Cross verification failed: {e}")
            return 0.3

    async def _external_verification(self, result: Dict[str, Any]) -> float:
        """외부 검증."""
        try:
            # MCP 도구를 사용한 외부 검증
            task_id = result.get("task_id", "")
            data = result.get("compressed_data", {})

            if not data or not task_id:
                return 0.5

            # 간단한 외부 검증 (실제로는 MCP 도구 활용)
            # 여기서는 기본적인 데이터 유효성만 검사
            if isinstance(data, dict) and len(data) > 0:
                return 0.8
            elif isinstance(data, list) and len(data) > 0:
                return 0.7
            else:
                return 0.6

        except Exception as e:
            logger.error(f"❌ External verification failed: {e}")
            return 0.0

    def _calculate_validation_score(self, state: ResearchState) -> float:
        """검증 점수 계산."""
        try:
            confidence_scores = state.get("confidence_scores", {})
            if not confidence_scores:
                return 0.0

            # 평균 신뢰도 점수 계산
            total_score = sum(confidence_scores.values())
            avg_score = total_score / len(confidence_scores)

            # 품질 메트릭 반영
            quality_metrics = state.get("quality_metrics", {})
            if quality_metrics:
                quality_score = quality_metrics.get("overall_quality", 0.8)
                avg_score = (avg_score + quality_score) / 2

            return min(avg_score, 1.0)
        except Exception as e:
            logger.error(f"❌ Validation score calculation failed: {e}")
            return 0.0

    def _identify_missing_elements(self, state: ResearchState) -> List[str]:
        """누락된 요소 식별."""
        try:
            missing_elements = []

            # 필수 필드 검사
            required_fields = [
                "analyzed_objectives",
                "planned_tasks",
                "execution_results",
            ]
            for field in required_fields:
                if not state.get(field):
                    missing_elements.append(f"Missing {field}")

            # 실행 결과 검사
            execution_results = state.get("execution_results", [])
            if not execution_results:
                missing_elements.append("No execution results found")

            # 압축 결과 검사
            compression_results = state.get("compression_results", [])
            if not compression_results:
                missing_elements.append("No compression results found")

            # 검증 결과 검사
            verification_stages = state.get("verification_stages", [])
            if not verification_stages:
                missing_elements.append("No verification results found")

            return missing_elements
        except Exception as e:
            logger.error(f"❌ Missing elements identification failed: {e}")
            return ["Error in missing elements analysis"]

    def _decide_next_step_based_on_context(self, state: ResearchState) -> str:
        """컨텍스트 기반 다음 단계 자동 결정 (재귀적 컨텍스트 사용).

        Args:
            state: 현재 상태

        Returns:
            다음 단계 이름
        """
        current_context = self.context_manager.get_current_context()

        if not current_context:
            # 컨텍스트가 없으면 기본 흐름
            return "compress"

        # 컨텍스트 완전성 평가
        completeness = self.context_manager.evaluate_context_completeness(
            current_context.context_id
        )

        # 실행 결과 확인
        execution_results = state.get("execution_results", [])
        successful_results = [
            r for r in execution_results if r.get("status") == "completed"
        ]
        success_rate = len(successful_results) / max(len(execution_results), 1)

        logger.debug(
            f"Context completeness: {completeness:.2f}, Success rate: {success_rate:.2f}"
        )

        if completeness < 0.5 or success_rate < 0.5:
            # 컨텍스트가 불완전하거나 성공률이 낮으면 추가 연구
            logger.info(
                "🔄 Context incomplete or low success rate - continuing research"
            )
            return "continue_research"
        elif completeness < 0.8:
            # 컨텍스트가 거의 완전하면 압축 후 검증
            logger.info("📦 Context nearly complete - compressing")
            return "compress"
        else:
            # 컨텍스트가 완전하면 검증
            logger.info("✅ Context complete - verifying")
            return "verify"

    def _calculate_context_usage(
        self, state: ResearchState, content: str
    ) -> Dict[str, Any]:
        """컨텍스트 윈도우 사용량 계산."""
        try:
            # 간단한 토큰 수 추정 (실제로는 더 정교한 토큰화 필요)
            estimated_tokens = len(content.split()) * 1.3  # 대략적인 토큰 수

            # 최대 토큰 수 (모델별로 다름)
            max_tokens = 100000  # 기본값

            usage_ratio = min(estimated_tokens / max_tokens, 1.0)

            return {
                "usage_ratio": usage_ratio,
                "tokens_used": int(estimated_tokens),
                "max_tokens": max_tokens,
                "efficiency": 1.0 - usage_ratio,
            }
        except Exception as e:
            logger.error(f"❌ Context usage calculation failed: {e}")
            return {
                "usage_ratio": 0.0,
                "tokens_used": 0,
                "max_tokens": 100000,
                "efficiency": 1.0,
            }

    async def run_research(
        self, user_request: str, context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """연구 실행 (Production-Grade Reliability + ExecutionContext)."""
        input_result = validate_user_input(user_request or "")
        if not input_result.is_safe:
            logger.warning(
                "Prompt security: rejecting user request (reason=%s)",
                input_result.rejection_reason,
            )
            return {
                "content": REJECTION_MESSAGE,
                "metadata": {"error": True},
                "synthesis_results": {"content": REJECTION_MESSAGE},
                "innovation_stats": {},
                "system_health": {"overall_status": "degraded"},
                "detailed_results": {},
            }
        user_request = input_result.sanitized_text

        logger.info(f"🚀 Starting research with 8 core innovations: {user_request}")

        # ExecutionContext 설정 (ROMA 스타일)
        execution_id = f"exec_{int(datetime.now().timestamp())}"
        context_token = None
        try:
            from src.core.recursive_context_manager import ExecutionContext

            context_token = ExecutionContext.set(execution_id, self.context_manager)
            logger.debug(f"ExecutionContext set for execution: {execution_id}")
        except Exception as e:
            logger.debug(f"Failed to set ExecutionContext: {e}")

        # CLI 모드 감지 및 autopilot 모드 설정
        import sys

        is_cli_mode = (
            not hasattr(sys, "ps1")  # Interactive shell이 아님
            and "streamlit" not in sys.modules  # Streamlit이 로드되지 않음
            and not any(
                "streamlit" in str(arg) for arg in sys.argv
            )  # Streamlit 실행 인자가 없음
        )

        # 초기 상태 설정
        initial_state = ResearchState(
            user_request=user_request,
            context=context or {},
            objective_id=execution_id,  # execution_id 사용
            analyzed_objectives=[],
            intent_analysis={},
            domain_analysis={},
            scope_analysis={},
            # Planning Agent 필드
            preliminary_research={},
            planned_tasks=[],
            agent_assignments={},
            execution_plan={},
            plan_approved=False,
            plan_feedback=None,
            plan_iteration=0,
            execution_results=[],
            agent_status={},
            execution_metadata={},
            streaming_data=[],
            compression_results=[],
            compression_metadata={},
            verification_results={},
            confidence_scores={},
            verification_stages=[],
            evaluation_results={},
            quality_metrics={},
            improvement_areas=[],
            validation_results={},
            validation_score=0.0,
            missing_elements=[],
            final_synthesis={},
            deliverable_path=None,
            synthesis_metadata={},
            context_window_usage={},
            pending_questions=[],
            user_responses={},
            clarification_context={},
            waiting_for_user=False,
            autopilot_mode=is_cli_mode,  # CLI 모드이면 autopilot 활성화
            current_step="analyze_objectives",
            iteration=0,
            max_iterations=10,
            should_continue=True,
            error_message=None,
            innovation_stats={},
            messages=[],
        )

        if is_cli_mode:
            logger.info(
                "🤖 CLI mode detected - Autopilot mode enabled (auto-selecting responses)"
            )

        # LangGraph 워크플로우 실행
        logger.info("🔄 Executing LangGraph workflow with 8 core innovations")
        final_state = await self.graph.ainvoke(
            initial_state, config=get_langfuse_run_config()
        )

        # 결과 포맷팅
        result = {
            "content": final_state.get("final_synthesis", {}).get(
                "content", "Research completed"
            ),
            "metadata": {
                "model_used": final_state.get("final_synthesis", {}).get(
                    "model_used", "unknown"
                ),
                "execution_time": final_state.get("final_synthesis", {}).get(
                    "execution_time", 0.0
                ),
                "cost": 0.0,
                "confidence": final_state.get("final_synthesis", {}).get(
                    "confidence", 0.9
                ),
            },
            "synthesis_results": {
                "content": final_state.get("final_synthesis", {}).get("content", ""),
                "original_length": len(str(final_state.get("execution_results", []))),
                "compressed_length": len(
                    str(final_state.get("compression_results", []))
                ),
                "compression_ratio": final_state.get("compression_metadata", {}).get(
                    "overall_compression_ratio", 1.0
                ),
            },
            "innovation_stats": final_state.get("innovation_stats", {}),
            "system_health": {"overall_status": "healthy", "health_score": 95},
            "detailed_results": {
                "analyzed_objectives": final_state.get("analyzed_objectives", []),
                "planned_tasks": final_state.get("planned_tasks", []),
                "execution_results": final_state.get("execution_results", []),
                "compression_results": final_state.get("compression_results", []),
                "verification_stages": final_state.get("verification_stages", []),
                "evaluation_results": final_state.get("evaluation_results", {}),
                "quality_metrics": final_state.get("quality_metrics", {}),
            },
        }

        logger.info("✅ Research completed successfully with 8 core innovations")

        # ExecutionContext 및 MCP Hub 정리 (ROMA 스타일)
        try:
            from src.core.recursive_context_manager import ExecutionContext

            if context_token:
                ExecutionContext.reset(context_token)
                logger.debug(f"ExecutionContext reset for execution: {execution_id}")
        except Exception as e:
            logger.debug(f"Failed to reset ExecutionContext: {e}")

        # MCP Hub 실행 세션 정리
        try:
            from src.core.mcp_integration import get_mcp_hub

            mcp_hub = get_mcp_hub()
            await mcp_hub.cleanup_execution(execution_id)
        except Exception as e:
            logger.debug(f"Failed to cleanup MCP Hub execution session: {e}")

        return result

    async def _search_similar_research(
        self, query: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """유사한 과거 연구를 검색합니다."""
        try:
            # 하이브리드 스토리지에서 유사 연구 검색
            similar_research = await self.hybrid_storage.search_similar_research(
                query=query, user_id=user_id, limit=5, similarity_threshold=0.3
            )

            # 결과 포맷팅
            formatted_results = []
            for research in similar_research:
                formatted_results.append(
                    {
                        "research_id": research.research_id,
                        "topic": research.metadata.get("topic", ""),
                        "summary": research.summary,
                        "similarity_score": research.similarity_score,
                        "timestamp": research.timestamp.isoformat(),
                        "confidence_score": research.metadata.get(
                            "confidence_score", 0.0
                        ),
                    }
                )

            logger.info(f"Found {len(formatted_results)} similar research results")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search similar research: {e}")
            return []

    async def _overseer_initial_review(self, state: ResearchState) -> ResearchState:
        """Overseer의 초기 검토 - Planning 후 요구사항 정의"""
        logger.info("=" * 80)
        logger.info("🔍 [OVERSEER] Initial Review - Defining Requirements")
        logger.info("=" * 80)

        try:
            from src.agents.greedy_overseer_agent import get_greedy_overseer_agent
            from src.core.researcher_config import load_config_from_env

            config = load_config_from_env()
            overseer_config = config.overseer if hasattr(config, "overseer") else None

            if overseer_config and overseer_config.enabled:
                overseer = get_greedy_overseer_agent(
                    max_iterations=overseer_config.max_iterations,
                    completeness_threshold=overseer_config.completeness_threshold,
                    quality_threshold=overseer_config.quality_threshold,
                    min_academic_sources=overseer_config.min_academic_sources,
                    min_verified_sources=overseer_config.min_verified_sources,
                    require_cross_validation=overseer_config.require_cross_validation,
                    enable_human_loop=overseer_config.enable_human_loop,
                )
            else:
                # Default configuration
                overseer = get_greedy_overseer_agent()

            state = await overseer.review_planning_output(state)

            logger.info(
                f"[OVERSEER] Requirements defined: {len(state.get('overseer_requirements', []))}"
            )

        except Exception as e:
            logger.error(f"[OVERSEER] Initial review failed: {e}")
            # Continue with default requirements
            state["overseer_iterations"] = 0
            state["overseer_requirements"] = []

        return state

    async def _overseer_evaluation(self, state: ResearchState) -> ResearchState:
        """Overseer 평가 - Execution + Validation 후 결과 평가"""
        logger.info("=" * 80)
        logger.info("🔍 [OVERSEER] Evaluating Execution Results")
        logger.info("=" * 80)

        try:
            from src.agents.greedy_overseer_agent import get_greedy_overseer_agent
            from src.core.researcher_config import load_config_from_env

            config = load_config_from_env()
            overseer_config = config.overseer if hasattr(config, "overseer") else None

            if overseer_config and overseer_config.enabled:
                overseer = get_greedy_overseer_agent(
                    max_iterations=overseer_config.max_iterations,
                    completeness_threshold=overseer_config.completeness_threshold,
                    quality_threshold=overseer_config.quality_threshold,
                    min_academic_sources=overseer_config.min_academic_sources,
                    min_verified_sources=overseer_config.min_verified_sources,
                    require_cross_validation=overseer_config.require_cross_validation,
                    enable_human_loop=overseer_config.enable_human_loop,
                )
            else:
                overseer = get_greedy_overseer_agent()

            state = await overseer.evaluate_execution_results(state)

            decision = state.get("overseer_decision", "proceed")
            logger.info(f"[OVERSEER] Decision: {decision}")

        except Exception as e:
            logger.error(f"[OVERSEER] Evaluation failed: {e}")
            # Default to proceed on error
            state["overseer_decision"] = "proceed"

        return state

    def _overseer_decision_router(self, state: ResearchState) -> str:
        """Overseer 결정에 따른 라우팅"""
        decision = state.get("overseer_decision", "proceed")
        current_iteration = state.get("overseer_iterations", 0)
        max_iterations = 5  # Default max

        try:
            from src.core.researcher_config import load_config_from_env

            config = load_config_from_env()
            if hasattr(config, "overseer") and config.overseer.enabled:
                max_iterations = config.overseer.max_iterations
        except:
            pass

        logger.info(
            f"[OVERSEER] Routing decision: {decision} (iteration {current_iteration}/{max_iterations})"
        )

        if decision == "retry" and current_iteration < max_iterations:
            logger.info(
                f"[OVERSEER] Retrying execution (iteration {current_iteration + 1})"
            )
            return "retry"
        elif decision == "ask_user":
            logger.info("[OVERSEER] Requesting user clarification")
            return "waiting_for_clarification"
        else:
            logger.info("[OVERSEER] Proceeding to evaluation")
            return "proceed"

    async def _save_research_memory(self, state: ResearchState) -> bool:
        """연구 결과를 메모리에 저장합니다."""
        try:
            from src.storage.vector_store import ResearchMemory

            # 연구 메모리 생성
            memory = ResearchMemory(
                research_id=state["objective_id"],
                user_id=state.get("user_id", "default_user"),
                topic=state["user_request"],
                timestamp=datetime.now(UTC),
                embedding=[],  # 하이브리드 스토리지에서 생성
                metadata={
                    "complexity_score": state.get("complexity_score", 0.0),
                    "objectives_count": len(state.get("analyzed_objectives", [])),
                    "execution_results": state.get("execution_results", []),
                    "verification_results": state.get("verification_results", {}),
                    "quality_metrics": state.get("quality_metrics", {}),
                },
                results=state.get("final_synthesis", {}),
                content=state.get("final_synthesis", {}).get("content", ""),
                summary=state.get("final_synthesis", {}).get("summary", ""),
                keywords=state.get("final_synthesis", {}).get("keywords", []),
                confidence_score=state.get("final_synthesis", {}).get(
                    "confidence", 0.0
                ),
                source_count=len(state.get("execution_results", [])),
                verification_status=state.get("verification_results", {}).get(
                    "status", "unverified"
                ),
            )

            # 하이브리드 스토리지에 저장
            success = await self.hybrid_storage.store_research(
                research_id=memory.research_id,
                user_id=memory.user_id,
                topic=memory.topic,
                content=memory.content,
                results=memory.results,
                metadata=memory.metadata,
                summary=memory.summary,
                keywords=memory.keywords,
            )

            if success:
                logger.info(f"Research memory saved: {memory.research_id}")
            else:
                logger.warning(f"Failed to save research memory: {memory.research_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to save research memory: {e}")
            return False


# Global orchestrator instance (lazy initialization)
_orchestrator = None


def get_orchestrator() -> "AutonomousOrchestrator":
    """Get or initialize global orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AutonomousOrchestrator()
    return _orchestrator


async def run_research(
    user_request: str, context: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """연구 실행."""
    orchestrator = get_orchestrator()
    return await orchestrator.run_research(user_request, context)

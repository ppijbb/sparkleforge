from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage


def override_reducer(a: Any, b: Any) -> Any:
    """새로운 값으로 덮어쓰는 리듀서"""
    return b


def add_messages(left: list, right: list) -> list:
    """메시지 리스트 병합 (LangGraph의 표준 동작 모사)"""
    return left + right


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

    # Runtime sub-agent delegation (Anvil Phase Σ-2, issue #495/#509).
    # Mirrors the overseer_iterations/max_iterations guard pattern above:
    # delegate_to_agent() increments delegation_depth for the call and
    # refuses to delegate once it reaches max_delegation_depth.
    delegation_depth: int
    max_delegation_depth: int

    # Innovation Stats
    innovation_stats: Dict[str, Any]

    # Multi-Modal Forge: Cross-Modal Feature Alignment Engine (issue #1013).
    # Maps natural language service requirements onto visual workflow graphs
    # and database ERD schemas.
    cross_modal_alignment: Dict[str, Any]
    aligned_modalities: List[str]

    # Messages for LangGraph
    messages: Annotated[List[BaseMessage], "Messages in the conversation"]
